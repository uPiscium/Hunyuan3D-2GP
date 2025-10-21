import os
import random
import time
import uuid
import torch

from PIL import Image
from mmgp import offload, profile_type
from trimesh import Trimesh

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.shapegen import (
    FaceReducer,
    Hunyuan3DDiTFlowMatchingPipeline,
)
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.rembg import BackgroundRemover


class Hunyuan3DController:
    def __init__(self, config: dict):
        torch.set_grad_enabled(False)
        self.__config = config.get("hunyuan3d", {})

        self.__seed = int(self.__config.get("seed", 0))
        self.__random_seed = self.__config.get("random_seed", False)
        self.__max_seed = int(self.__config.get("max_seed", 1e7))
        self.__model_path = self.__config.get("model", "tencent/Hunyuan3D-2mini")
        self.__subfolder = self.__config.get("subfolder", "hunyuan3d-dit-v2-mini")
        self.__texgen_model_path = self.__config.get("texgen", "tencent/Hunyuan3D-2")
        self.__device = self.__config.get("device", "cuda")
        self.__mc_algo = self.__config.get("mc_algo", "dmc")
        self.__save_dir = self.__config.get("cache_path", "gradio_cache")
        self.__profile = self.__config.get("profile", "3")
        self.__verbose = self.__config.get("verbose", "1")
        self.__enable_t23d = self.__config.get("enable_t23d", False)
        self.__enable_flashvdm = self.__config.get("enable_flashvdm", False)
        self.__disable_tex = self.__config.get("disable_tex", False)
        self.__low_vram_mode = self.__config.get("low_vram_mode", False)
        self.__compile = self.__config.get("compile", False)

        if self.__config.get("mini", False):
            self.__model_path = "tencent/Hunyuan3D-2mini"
            self.__subfolder = "hunyuan3d-dit-v2-mini"
            self.__texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__config.get("mv", False):
            self.__model_path = "tencent/Hunyuan3D-2mv"
            self.__subfolder = "hunyuan3d-dit-v2-mv"
            self.__texgen_model_path = "tencent/Hunyuan3D-2mv"

        if self.__config.get("h2", False):
            self.__model_path = "tencent/Hunyuan3D-2"
            self.__subfolder = "hunyuan3d-dit-v2-0"
            self.__texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__config.get("turbo", False):
            self.__subfolder = self.__subfolder + "-turbo"
            self.__enable_flashvdm = True

        os.makedirs(self.__save_dir, exist_ok=True)

        self.__mv_mode = "mv" in self.__model_path
        self.__texturegen_worker = self.__load_texture_generator()
        self.__t2i_worker = self.__load_text2image()
        self.__rmbg_worker = BackgroundRemover()
        self.__i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.__model_path,
            subfolder=self.__subfolder,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device=self.__device,
        )
        self.__face_reducer = FaceReducer()

        self.__setup_models()

    def __load_texture_generator(self) -> Hunyuan3DPaintPipeline | None:
        generator = None
        if not self.__disable_tex:
            try:
                generator = Hunyuan3DPaintPipeline.from_pretrained(
                    self.__texgen_model_path
                )
            except Exception as e:
                print(e)
                print("[ERROR] Failed to load texture generator.")
                print(
                    "[ERROR] Please try to install requirements by following README.md"
                )
        return generator

    def __load_text2image(self) -> HunyuanDiTPipeline | None:
        t2i_worker = None
        if self.__enable_t23d:
            t2i_worker = HunyuanDiTPipeline(
                "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
            )
        return t2i_worker

    def __create_kwargs(self) -> dict:
        kwargs = {}
        if self.__profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if self.__profile != 1 and self.__profile != 3:
            kwargs["budgets"] = {"*": 2200}
        return kwargs

    def __setup_models(self) -> None:
        if self.__enable_flashvdm:
            mc_algo = "mc" if self.__device in ["mps", "cpu"] else self.__mc_algo
            self.__i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        if self.__compile:
            self.__i23d_worker.compile()

        property_name = "_execution_device"
        # Get the original class and property
        original_class = type(self.__i23d_worker)
        original_property = getattr(original_class, property_name)

        # Create a custom subclass for this instance
        custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})

        # Create a new property with the new getter but same setter
        new_property = property(lambda _: "cuda", original_property.fset)
        setattr(custom_class, property_name, new_property)

        # Change the instance's class
        self.__i23d_worker.__class__ = custom_class

        pipe = offload.extract_models("i23d_worker", self.__i23d_worker)
        if self.__texturegen_worker is not None:
            pipe.update(
                offload.extract_models("texturegen_worker", self.__texturegen_worker)
            )
            self.__texturegen_worker.models[
                "multiview_model"
            ].pipeline.vae.use_slicing = True
        if self.__t2i_worker is not None:
            pipe.update(offload.extract_models("t2i_worker", self.__t2i_worker))

        offload.default_verboseLevel = self.__verbose
        kwargs = self.__create_kwargs()
        offload.profile(
            pipe,
            profile_no=profile_type(self.__profile),
            verboseLevel=self.__verbose,
            **kwargs,
        )

        if self.__low_vram_mode:
            torch.cuda.empty_cache()

    def __gen_save_folder(self) -> str:
        # a folder to save the generated files
        folder_name = str(uuid.uuid4())
        save_folder = os.path.join("gradio_cache", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

    def __export(
        self,
        mesh: Trimesh,
        save_folder: str,
        textured: bool = False,
        type: str = "glb",
    ):
        if textured:
            path = os.path.join(save_folder, f"textured_mesh.{type}")
        else:
            path = os.path.join(save_folder, f"white_mesh.{type}")
        if type not in ["glb", "obj"]:
            mesh.export(path)
        else:
            mesh.export(path, include_normals=textured)
        return path

    def __get_seed(self) -> int:
        seed = self.__seed
        if self.__random_seed:
            seed = random.randint(0, self.__max_seed)
        return seed

    async def generate(
        self,
        caption: str | None = None,
        image: dict[str, Image.Image | None] | Image.Image | None = None,
        mv_image_front: Image.Image | None = None,
        mv_image_back: Image.Image | None = None,
        mv_image_left: Image.Image | None = None,
        mv_image_right: Image.Image | None = None,
        steps: int = 50,
        guidance_scale: float = 7.5,
        octree_resolution: int = 256,
        check_box_rembg: bool = False,
        num_chunks: int = 200000,
    ) -> tuple[str, dict[str, Image.Image | None] | Image.Image | None]:
        if not self.__mv_mode and image is None and caption is None:
            print("[ERROR] Please provide either a caption or an image.")
            return "", None
        if self.__mv_mode:
            if (
                mv_image_front is None
                and mv_image_back is None
                and mv_image_left is None
                and mv_image_right is None
            ):
                print("[ERROR] Please provide at least one view image.")
                return "", None
            image = {}
            if mv_image_front:
                image["front"] = mv_image_front
            if mv_image_back:
                image["back"] = mv_image_back
            if mv_image_left:
                image["left"] = mv_image_left
            if mv_image_right:
                image["right"] = mv_image_right

        seed = int(self.__get_seed())

        if caption:
            print("[INFO] prompt is", caption)
        save_folder = self.__gen_save_folder()

        if image is None:
            try:
                if self.__t2i_worker is None:
                    print(
                        "[ERROR] Text to 3D is disabled. Please enable it by `python gradio_app.py --enable_t23d`."
                    )
                    return "", None
                image = self.__t2i_worker(caption)
            except Exception:
                print(
                    "[ERROR] Text to 3D is disabled. Please enable it by `python gradio_app.py --enable_t23d`."
                )
                return "", None

        converted_image = None
        if self.__mv_mode and isinstance(image, dict):
            start_time = time.time()
            for k, v in image.items():
                if v is None:
                    continue

                if check_box_rembg or v.mode == "RGB":
                    img = self.__rmbg_worker(v.convert("RGB"))
                    image[k] = img
            converted_image = image
        else:
            if not isinstance(image, Image.Image):
                return "", None

            if check_box_rembg or image.mode == "RGB":
                start_time = time.time()
                converted_image = self.__rmbg_worker(image.convert("RGB"))
            else:
                converted_image = image

        # image to white model
        start_time = time.time()

        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        outputs = self.__i23d_worker(
            image=converted_image,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=num_chunks,
            output_type="custom",  # ignore trimesh process
        )

        mesh = export_to_trimesh(outputs)[0]
        print("[INFO] Export to tri-mesh")
        mesh = self.__face_reducer(mesh, 20000)
        print("[INFO] Face reducer execution completed.")

        main_image = (
            converted_image
            if not self.__mv_mode
            else converted_image["front"]
            if isinstance(converted_image, dict) and ("front" in converted_image)
            else None
        )
        mesh = (
            self.__texturegen_worker(mesh, main_image)
            if self.__texturegen_worker
            else mesh
        )
        print("[INFO] Texture generation completed")

        path = self.__export(mesh, save_folder, textured=True)
        print("[INFO] Export model")
        print(f"[INFO] Generation takes {time.time() - start_time:.6f} seconds.")

        return path, main_image
