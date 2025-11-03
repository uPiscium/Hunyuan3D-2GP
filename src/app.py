from PIL import Image
from io import BytesIO
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from controller import Hunyuan3DController
import json


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(self, config: dict):
        self.__config = config
        self.__hunyuan3D_controller = Hunyuan3DController(config)
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route(
            "/process",
            self.generate_by_image,
            methods=["POST"],
            response_class=JSONResponse,
        )
        self.__router.add_api_route(
            "/ping", self.ping, methods=["GET"], response_class=JSONResponse
        )
        self.__router.add_api_route(
            "/config", self.config, methods=["PATCH"], response_class=JSONResponse
        )

    async def __generate_with_separate_models(
        self, file: UploadFile = File(...)
    ) -> FileResponse:
        byte = await file.read()
        image = Image.open(BytesIO(byte)).convert("RGB")
        steps: int = int(self.__config.get("steps", 50))
        gscale: float = self.__config.get("guidance_scale", 7.5)
        ores: int = int(self.__config.get("octree_resolution", 256))
        rembg: bool = self.__config.get("remove_bg", False)
        chunks: int = int(self.__config.get("chunks", 200000))
        path, _ = await self.__hunyuan3D_controller.generate(
            image=image,
            steps=steps,
            guidance_scale=gscale,
            octree_resolution=ores,
            check_box_rembg=rembg,
            num_chunks=chunks,
        )
        return FileResponse(
            path, media_type="application/octet-stream", filename="output.glb"
        )

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /process
    async def generate_by_image(self, file: UploadFile = File(...)) -> FileResponse:
        return await self.__generate_with_separate_models(file)

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)

    # /config
    async def config(self):
        try:
            CONFIG_PATH = "settings/meta.json"
            with open(CONFIG_PATH, "r") as f:
                self.__config = json.load(f)
            return JSONResponse({"config": self.__config}, status_code=200)
        except Exception as e:
            return JSONResponse({"error": str(e)}, status_code=400)
