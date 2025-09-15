import os
import random
import time
from io import BytesIO

import argparse

# import base64
import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# from pydantic import BaseModel
from PIL import Image
import requests
from mmgp import offload, profile_type
import uuid

from hy3dgen.shapegen.utils import logger


# class ImagePayload(BaseModel):
#     uuid: str
#     image: str


class App:
    MAX_SEED = int(1e7)

    def __init__(self, db_endpoint: str):
        self.__router = APIRouter()
        self.__app = FastAPI()
        self.__app.mount("/assets", StaticFiles(directory="assets"), name="assets")
        self.__app.mount(
            "/gradio_cache", StaticFiles(directory="gradio_cache"), name="gradio_cache"
        )
        self.__db_endpoint = db_endpoint

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route(
            "/generate",
            self.generate_api,
            methods=["POST"],
            response_class=JSONResponse,
        )

    def __gen_save_folder(self):
        # a folder to save the generated files
        folder_name = str(uuid.uuid4())
        save_folder = os.path.join("gradio_cache", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

    # def __decode_base64_image(self, base64_string: str) -> Image.Image:
    #     # "data:image/png;base64," のようなプレフィックスを削除
    #     if "," in base64_string:
    #         _, encoded = base64_string.split(",", 1)
    #     else:
    #         encoded = base64_string
    #     try:
    #         image_data = base64.b64decode(encoded)
    #         return Image.open(BytesIO(image_data))
    #     except Exception as e:
    #         raise ValueError(f"Invalid base64 image: {e}")

    def __export_mesh(self, mesh, save_folder, textured=False, type="glb"):
        if textured:
            path = os.path.join(save_folder, f"textured_mesh.{type}")
        else:
            path = os.path.join(save_folder, f"white_mesh.{type}")
        if type not in ["glb", "obj"]:
            mesh.export(path)
        else:
            mesh.export(path, include_normals=textured)
        return path

    def __randomize_seed_fn(self, seed: int, randomize_seed: bool) -> int:
        if randomize_seed:
            seed = random.randint(0, App.MAX_SEED)
        return seed

    def __gen_shape(
        self,
        caption=None,
        image=None,
        mv_image_front=None,
        mv_image_back=None,
        mv_image_left=None,
        mv_image_right=None,
        steps=50,
        guidance_scale=7.5,
        seed=1234,
        octree_resolution=256,
        check_box_rembg=False,
        num_chunks=200000,
        randomize_seed: bool = False,
    ):
        if not MV_MODE and image is None and caption is None:
            raise gr.Error("Please provide either a caption or an image.")
        if MV_MODE:
            if (
                mv_image_front is None
                and mv_image_back is None
                and mv_image_left is None
                and mv_image_right is None
            ):
                raise gr.Error("Please provide at least one view image.")
            image = {}
            if mv_image_front:
                image["front"] = mv_image_front
            if mv_image_back:
                image["back"] = mv_image_back
            if mv_image_left:
                image["left"] = mv_image_left
            if mv_image_right:
                image["right"] = mv_image_right

        seed = int(self.__randomize_seed_fn(seed, randomize_seed))

        octree_resolution = int(octree_resolution)
        if caption:
            print("prompt is", caption)
        save_folder = self.__gen_save_folder()
        stats = {
            "model": {
                "shapegen": f"{args.model_path}/{args.subfolder}",
                "texgen": f"{args.texgen_model_path}",
            },
            "params": {
                "caption": caption,
                "steps": steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "octree_resolution": octree_resolution,
                "check_box_rembg": check_box_rembg,
                "num_chunks": num_chunks,
            },
        }
        time_meta = {}

        if image is None:
            start_time = time.time()
            try:
                if t2i_worker is None:
                    raise gr.Error(
                        "Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`."
                    )
                image = t2i_worker(caption)
            except Exception:
                raise gr.Error(
                    "Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`."
                )
            time_meta["text2image"] = time.time() - start_time

        # remove disk io to make responding faster, uncomment at your will.
        # image.save(os.path.join(save_folder, 'input.png'))
        if MV_MODE:
            start_time = time.time()
            for k, v in image.items():
                if check_box_rembg or v.mode == "RGB":
                    img = rmbg_worker(v.convert("RGB"))
                    image[k] = img
            time_meta["remove background"] = time.time() - start_time
        else:
            if not isinstance(image, Image.Image):
                return None, None, None, None, None

            if check_box_rembg or image.mode == "RGB":
                start_time = time.time()
                image = rmbg_worker(image.convert("RGB"))
                time_meta["remove background"] = time.time() - start_time

        # remove disk io to make responding faster, uncomment at your will.
        # image.save(os.path.join(save_folder, 'rembg.png'))

        # image to white model
        start_time = time.time()

        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        outputs = i23d_worker(
            image=image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="mesh",
        )
        time_meta["shape generation"] = time.time() - start_time
        logger.info(
            "---Shape generation takes %s seconds ---" % (time.time() - start_time)
        )

        tmp_start = time.time()
        mesh = export_to_trimesh(outputs)[0]
        time_meta["export to trimesh"] = time.time() - tmp_start

        stats["number_of_faces"] = mesh.faces.shape[0]
        stats["number_of_vertices"] = mesh.vertices.shape[0]

        stats["time"] = time_meta
        main_image = (
            image
            if not MV_MODE
            else image["front"]
            if isinstance(image, dict) and ("front" in image)
            else None
        )
        return mesh, main_image, save_folder, stats, seed

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    async def generate_api(self, user_id: str, image_file: UploadFile) -> JSONResponse:
        start_time_0 = time.time()
        image = Image.open(BytesIO(await image_file.read())).convert("RGB")
        mesh, image, save_folder, stats, _ = self.__gen_shape(
            image=image,
            randomize_seed=True,
        )

        if mesh is None:
            return JSONResponse({"error": "Failed to generate mesh."}, status_code=500)

        if image is None:
            return JSONResponse({"error": "Failed to process image."}, status_code=500)

        if save_folder is None:
            return JSONResponse(
                {"error": "Failed to create save folder."}, status_code=500
            )

        if stats is None:
            return JSONResponse({"error": "Failed to generate stats."}, status_code=500)

        tmp_time = time.time()
        mesh = face_reduce_worker(mesh)
        stats["time"]["face reduction"] = time.time() - tmp_time

        tmp_time = time.time()

        if texgen_worker is None:
            return JSONResponse(
                {"error": "Texture generator is not available."}, status_code=500
            )

        textured_mesh = texgen_worker(mesh, image)
        stats["time"]["texture generation"] = time.time() - tmp_time
        stats["time"]["total"] = time.time() - start_time_0

        textured_mesh.metadata["extras"] = stats
        path_textured = self.__export_mesh(textured_mesh, save_folder, textured=True)

        if args.low_vram_mode:
            torch.cuda.empty_cache()

        try:
            payload = {"user_id": user_id}
            file = {"file": open(path_textured, "rb")}
            response = requests.post(
                f"{self.__db_endpoint}/save/model", data=payload, files=file
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail="Failed to save model to database."
                )
        except Exception as e:
            print(e)
            return JSONResponse({"error": str(e)}, status_code=500)

        return JSONResponse(
            {"message": "Model generated and saved successfully"}, status_code=200
        )


def replace_property_getter(instance, property_name, new_getter):
    # Get the original class and property
    original_class = type(instance)
    original_property = getattr(original_class, property_name)

    # Create a custom subclass for this instance
    custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})

    # Create a new property with the new getter but same setter
    new_property = property(new_getter, original_property.fset)
    setattr(custom_class, property_name, new_property)

    # Change the instance's class
    instance.__class__ = custom_class

    return instance


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2mini")
parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mini")
parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
parser.add_argument("-p", "--port", type=int, default=8080)
parser.add_argument("--host", type=str, default="0.0.0.0")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--mc_algo", type=str, default="dmc")
parser.add_argument("--cache-path", type=str, default="gradio_cache")
parser.add_argument("--enable_t23d", action="store_true")
parser.add_argument("--profile", type=str, default="3")
parser.add_argument("--verbose", type=str, default="1")

parser.add_argument("--disable_tex", action="store_true")
parser.add_argument("--enable_flashvdm", action="store_true")
parser.add_argument("--low-vram-mode", action="store_true")
parser.add_argument("--compile", action="store_true")
parser.add_argument("--mini", action="store_true")
parser.add_argument("--turbo", action="store_true")
parser.add_argument("--mv", action="store_true")
parser.add_argument("--h2", action="store_true")

parser.add_argument(
    "-db",
    "--database-endpoint",
    type=str,
    default="http://localhost:8000",
    help="The endpoint of the database server.",
)

args = parser.parse_args()

app = App(args.database_endpoint).get_app()

if __name__ == "__main__":
    if args.mini:
        args.model_path = "tencent/Hunyuan3D-2mini"
        args.subfolder = "hunyuan3d-dit-v2-mini"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.mv:
        args.model_path = "tencent/Hunyuan3D-2mv"
        args.subfolder = "hunyuan3d-dit-v2-mv"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.h2:
        args.model_path = "tencent/Hunyuan3D-2"
        args.subfolder = "hunyuan3d-dit-v2-0"
        args.texgen_model_path = "tencent/Hunyuan3D-2"

    if args.turbo:
        args.subfolder = args.subfolder + "-turbo"
        args.enable_flashvdm = True

    SAVE_DIR = args.cache_path
    os.makedirs(SAVE_DIR, exist_ok=True)

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    MV_MODE = "mv" in args.model_path
    TURBO_MODE = "turbo" in args.subfolder

    HTML_HEIGHT = 690 if MV_MODE else 650
    HTML_WIDTH = 500

    torch.set_default_device("cpu")
    # try:
    #     from hy3dgen.texgen import Hunyuan3DPaintPipeline

    SUPPORTED_FORMATS = ["glb", "obj", "ply", "stl"]

    HAS_TEXTUREGEN = False
    texgen_worker = None
    if not args.disable_tex:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline

            texgen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                args.texgen_model_path
            )
            # texgen_worker.enable_model_cpu_offload()
            # Not help much, ignore for now.
            # if args.compile:
            #     texgen_worker.models['delight_model'].pipeline.unet.compile()
            #     texgen_worker.models['delight_model'].pipeline.vae.compile()
            #     texgen_worker.models['multiview_model'].pipeline.unet.compile()
            #     texgen_worker.models['multiview_model'].pipeline.vae.compile()
            HAS_TEXTUREGEN = True
        except Exception as e:
            print(e)
            print("Failed to load texture generator.")
            print("Please try to install requirements by following README.md")
            HAS_TEXTUREGEN = False

    HAS_T2I = False
    t2i_worker = None
    if args.enable_t23d:
        from hy3dgen.text2image import HunyuanDiTPipeline

        t2i_worker = HunyuanDiTPipeline(
            "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
        )
        HAS_T2I = True

    from hy3dgen.shapegen import (
        FaceReducer,
        FloaterRemover,
        DegenerateFaceRemover,
        Hunyuan3DDiTFlowMatchingPipeline,
    )
    from hy3dgen.shapegen.pipelines import export_to_trimesh
    from hy3dgen.rembg import BackgroundRemover

    rmbg_worker = BackgroundRemover()
    i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        args.model_path,
        subfolder=args.subfolder,
        use_safetensors=True,
        device=args.device,
    )
    if args.enable_flashvdm:
        mc_algo = "mc" if args.device in ["cpu", "mps"] else args.mc_algo
        i23d_worker.enable_flashvdm(mc_algo=mc_algo)
    if args.compile:
        i23d_worker.compile()

    floater_remove_worker = FloaterRemover()
    degenerate_face_remove_worker = DegenerateFaceRemover()
    face_reduce_worker = FaceReducer()

    profile = int(args.profile)
    kwargs = {}
    replace_property_getter(i23d_worker, "_execution_device", lambda _: "cuda")
    pipe = offload.extract_models("i23d_worker", i23d_worker)
    if HAS_TEXTUREGEN and texgen_worker is not None:
        pipe.update(offload.extract_models("texgen_worker", texgen_worker))
        texgen_worker.models["multiview_model"].pipeline.vae.use_slicing = True
    if HAS_T2I:
        pipe.update(offload.extract_models("t2i_worker", t2i_worker))

    if profile < 5:
        kwargs["pinnedMemory"] = "i23d_worker/model"
    if profile != 1 and profile != 3:
        kwargs["budgets"] = {"*": 2200}
    offload.default_verboseLevel = verboseLevel = int(args.verbose)
    offload.profile(
        pipe, profile_no=profile_type(profile), verboseLevel=int(args.verbose), **kwargs
    )

    if args.low_vram_mode:
        torch.cuda.empty_cache()

    uvicorn.run("yummy_api_server:app", host=args.host, port=args.port)
