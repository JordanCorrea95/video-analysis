from fastapi import FastAPI

from app.api.routes.inference import router as inference_router
from app.core.config import settings

app = FastAPI(title=settings.app_name, version=settings.app_version, debug=settings.debug)
app.include_router(inference_router)
