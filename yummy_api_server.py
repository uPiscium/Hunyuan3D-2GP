import os
import random
import time
from io import BytesIO

import argparse

import gradio as gr
import torch
import uvicorn
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import requests
from mmgp import offload, profile_type
import uuid

from hy3dgen.shapegen.utils import logger
from hy3dgen.shapegen import (
    FaceReducer,
    Hunyuan3DDiTFlowMatchingPipeline,
)
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.rembg import BackgroundRemover


class App:
    MAX_SEED = int(1e7)
    SUPPORTED_FORMATS = ["glb", "obj", "ply", "stl"]

    def __init__(self, args: argparse.Namespace):
        torch.set_default_device("cpu")
        self.__args = args

        if self.__args.mini:
            self.__args.model_path = "tencent/Hunyuan3D-2mini"
            self.__args.subfolder = "hunyuan3d-dit-v2-mini"
            self.__args.texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__args.mv:
            self.__args.model_path = "tencent/Hunyuan3D-2mv"
            self.__args.subfolder = "hunyuan3d-dit-v2-mv"
            self.__args.texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__args.h2:
            self.__args.model_path = "tencent/Hunyuan3D-2"
            self.__args.subfolder = "hunyuan3d-dit-v2-0"
            self.__args.texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__args.turbo:
            self.__args.subfolder = self.__args.subfolder + "-turbo"
            self.__args.enable_flashvdm = True

        self.__save_dir = self.__args.cache_path
        os.makedirs(self.__save_dir, exist_ok=True)

        self.__mv_mode = "mv" in self.__args.model_path
        self.__router = APIRouter()
        self.__app = FastAPI()
        self.__app.mount("/assets", StaticFiles(directory="assets"), name="assets")
        self.__app.mount(
            "/gradio_cache", StaticFiles(directory="gradio_cache"), name="gradio_cache"
        )
        self.__db_endpoint = self.__args.database_endpoint

        self.__setup_routes()

        self.__has_texturegen = False
        self.__texturegen_worker = None
        self.__load_texture_generator()

        self.__has_t2i = False
        self.__t2i_worker = None
        self.__load_text2image()

        self.__rmbg_worker = BackgroundRemover()
        self.__i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.__args.model_path,
            subfolder=self.__args.subfolder,
            use_safetensors=True,
            device=self.__args.device,
        )

        if self.__args.enable_flashvdm:
            mc_algo = (
                "mc" if self.__args.device in ["cpu", "mps"] else self.__args.mc_algo
            )
            self.__i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        if self.__args.compile:
            self.__i23d_worker.compile()

        self.__replace_property_getter(
            self.__i23d_worker, "_execution_device", lambda _: "cuda"
        )
        self.__face_reduce_worker = FaceReducer()

        pipe = offload.extract_models("i23d_worker", self.__i23d_worker)
        if self.__has_texturegen and self.__texturegen_worker is not None:
            pipe.update(
                offload.extract_models("texgen_worker", self.__texturegen_worker)
            )
            self.__texturegen_worker.models[
                "multiview_model"
            ].pipeline.vae.use_slicing = True
        if self.__has_t2i and self.__t2i_worker is not None:
            pipe.update(offload.extract_models("t2i_worker", self.__t2i_worker))

        profile = int(self.__args.profile)
        kwargs = {}
        if profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if profile != 1 and profile != 3:
            kwargs["budgets"] = {"*": 2200}
        offload.default_verboseLevel = int(args.verbose)
        offload.profile(
            pipe,
            profile_no=profile_type(profile),
            verboseLevel=int(args.verbose),
            **kwargs,
        )

        if args.low_vram_mode:
            torch.cuda.empty_cache()

    def __setup_routes(self):
        self.__router.add_api_route(
            "/generate",
            self.generate_api,
            methods=["POST"],
            response_class=JSONResponse,
        )

    def __load_texture_generator(self):
        if not self.__args.disable_tex:
            try:
                from hy3dgen.texgen import Hunyuan3DPaintPipeline

                self.__texturegen_worker = Hunyuan3DPaintPipeline.from_pretrained(
                    self.__args.texgen_model_path
                )
                self.__has_texturegen = True
            except Exception as e:
                print(e)
                print("Failed to load texture generator.")
                print("Please try to install requirements by following README.md")
                self.__has_texturegen = False

    def __load_text2image(self):
        if self.__args.enable_t23d:
            from hy3dgen.text2image import HunyuanDiTPipeline

            self.__t2i_worker = HunyuanDiTPipeline(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
            )
            self.__has_t2i = True

    def __replace_property_getter(self, instance, property_name, new_getter):
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

    def __gen_save_folder(self):
        # a folder to save the generated files
        folder_name = str(uuid.uuid4())
        save_folder = os.path.join("gradio_cache", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

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
        if not self.__mv_mode and image is None and caption is None:
            raise gr.Error("Please provide either a caption or an image.")
        if self.__mv_mode:
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
                if self.__t2i_worker is None:
                    raise gr.Error(
                        "Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`."
                    )
                image = self.__t2i_worker(caption)
            except Exception:
                raise gr.Error(
                    "Text to 3D is disable. Please enable it by `python gradio_app.py --enable_t23d`."
                )
            time_meta["text2image"] = time.time() - start_time

        if self.__mv_mode:
            start_time = time.time()
            for k, v in image.items():
                if check_box_rembg or v.mode == "RGB":
                    img = self.__rmbg_worker(v.convert("RGB"))
                    image[k] = img
            time_meta["remove background"] = time.time() - start_time
        else:
            if not isinstance(image, Image.Image):
                return None, None, None, None, None

            if check_box_rembg or image.mode == "RGB":
                start_time = time.time()
                image = self.__rmbg_worker(image.convert("RGB"))
                time_meta["remove background"] = time.time() - start_time

        # image to white model
        start_time = time.time()

        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        outputs = self.__i23d_worker(
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
            if not self.__mv_mode
            else image["front"]
            if isinstance(image, dict) and ("front" in image)
            else None
        )
        return mesh, main_image, save_folder, stats, seed

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    async def generate_api(
        self, user_id: str = Form(...), file: UploadFile = File(...)
    ) -> JSONResponse:
        print(file.file)

        start_time_0 = time.time()
        byte = await file.read()
        image = Image.open(BytesIO(byte), formats=["png"]).convert("RGB")
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
        mesh = self.__face_reduce_worker(mesh)
        stats["time"]["face reduction"] = time.time() - tmp_time

        tmp_time = time.time()

        if self.__texturegen_worker is None:
            return JSONResponse(
                {"error": "Texture generator is not available."}, status_code=500
            )

        textured_mesh = self.__texturegen_worker(mesh, image)
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


parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="tencent/Hunyuan3D-2mini")
parser.add_argument("--subfolder", type=str, default="hunyuan3d-dit-v2-mini")
parser.add_argument("--texgen_model_path", type=str, default="tencent/Hunyuan3D-2")
parser.add_argument("-p", "--port", type=int, default=8005)
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
    default="http://192.168.11.129:8001",
    help="The endpoint of the database server.",
)

args = parser.parse_args()
app = App(args).get_app()

if __name__ == "__main__":
    uvicorn.run("yummy_api_server:app", host=args.host, port=args.port)
