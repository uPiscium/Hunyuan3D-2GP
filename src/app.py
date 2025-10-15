from PIL import Image
from io import BytesIO
from fastapi import FastAPI, APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from controller import Hunyuan3DController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(self, config: dict):
        self.__hunyuan3D_controller = Hunyuan3DController(config)
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __setup_routes(self):
        self.__router.add_api_route(
            "/generate",
            self.generate_by_image,
            methods=["POST"],
            response_class=JSONResponse,
        )
        self.__router.add_api_route(
            "/ping", self.ping, methods=["GET"], response_class=JSONResponse
        )

    async def __generate_with_separate_models(
        self, file: UploadFile = File(...)
    ) -> FileResponse:
        byte = await file.read()
        image = Image.open(BytesIO(byte)).convert("RGB")
        path, _ = await self.__hunyuan3D_controller.generate(image=image)
        return FileResponse(
            path, media_type="application/octet-stream", filename="output.glb"
        )

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate_by_image(self, file: UploadFile = File(...)) -> FileResponse:
        return await self.__generate_with_separate_models(file)

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)
