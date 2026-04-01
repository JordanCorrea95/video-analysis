from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Video AI Backend"
    app_version: str = "0.1.0"
    debug: bool = False

    upload_dir: Path = Field(default=Path("data/uploads"))
    output_dir: Path = Field(default=Path("data/outputs"))

    # Modelos YOLO26m (ubicados en directorio models/)
    model_detect: str = "models/yolo26m.pt"
    model_segment: str = "models/yolo26m-seg.pt"
    model_pose: str = "models/yolo26m-pose.pt"

    default_confidence: float = 0.25
    max_upload_size_mb: int = 500

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.output_dir.mkdir(parents=True, exist_ok=True)
