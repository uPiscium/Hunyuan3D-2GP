import asyncio
import requests

from PIL import Image
from io import BytesIO
from fastapi import FastAPI, APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from controller import Hunyuan3DController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(self, config: dict, debug_mode: bool = False):
        self.__debug = debug_mode
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__hunyuan3D_controller = Hunyuan3DController(config)
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __setup_routes(self):
        if self.__config.get("use_t23d", True):
            self.__router.add_api_route(
                "/generate",
                self.generate_by_prompt,
                methods=["POST"],
                response_class=JSONResponse,
            )
        else:
            self.__router.add_api_route(
                "/generate",
                self.generate_by_image,
                methods=["POST"],
                response_class=JSONResponse,
            )
        self.__router.add_api_route(
            "/ping", self.ping, methods=["GET"], response_class=JSONResponse
        )

    async def __save_model(self, user_id: str, path: str) -> None:
        if not path:
            raise HTTPException(status_code=500, detail="Model path is empty.")

        try:
            payload = {"user_id": user_id}
            files = {"file": open(path, "rb")}

            response = requests.post(
                f"{self.__control_endpoint}/save/model", data=payload, files=files
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail="Failed to save model to database."
                )
        except Exception as e:
            print(f"[ERROR] Exception while saving model: {e}")
            raise HTTPException(status_code=500, detail="Error saving model.")

    async def __save_image(self, user_id: str, image: Image.Image) -> None:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        data = {"user_id": user_id}
        file = {"file": ("image.png", buffered, "image/png")}

        try:
            response = requests.post(
                f"{self.__control_endpoint}/save/image",
                files=file,
                data=data,
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail="Failed to save image to database."
                )
        except Exception as e:
            print(f"[ERROR] Exception while saving image: {e}")
            raise HTTPException(status_code=500, detail="Error saving image.")

    async def __generate_with_t23d(self, request: UserRequest) -> None:
        path, image = await self.__hunyuan3D_controller.generate(caption=request.prompt)

        if image is not None and isinstance(image, dict):
            image = list(image.values())[0]

        if image is None:
            raise HTTPException(status_code=500, detail="Failed to generate image.")

        asyncio.create_task(self.__save_image(user_id=request.user_id, image=image))
        asyncio.create_task(self.__save_model(user_id=request.user_id, path=path))

    async def __generate_with_separate_models(
        self, user_id: str = Form(...), file: UploadFile = File(...)
    ) -> None:
        byte = await file.read()
        image = Image.open(BytesIO(byte)).convert("RGB")
        path, _ = await self.__hunyuan3D_controller.generate(image=image)

        asyncio.create_task(self.__save_image(user_id=user_id, image=image))
        asyncio.create_task(self.__save_model(user_id=user_id, path=path))

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate_by_prompt(self, request: UserRequest) -> JSONResponse:
        await self.__generate_with_t23d(request)
        return JSONResponse(
            {"message": "Model and image generation completed."},
            status_code=200,
        )

    # /generate
    async def generate_by_image(
        self, user_id: str = Form(...), file: UploadFile = File(...)
    ) -> JSONResponse:
        await self.__generate_with_separate_models(user_id, file)
        return JSONResponse(
            {"message": "Model and image generation completed."},
            status_code=200,
        )

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)
