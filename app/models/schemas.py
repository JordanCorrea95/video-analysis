from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class InferenceMode(str, Enum):
    detect = "detect"
    segment = "segment"
    pose = "pose"


class InferenceResponse(BaseModel):
    mode: InferenceMode
    input_filename: str
    output_video_path: str
    total_frames: int
    fps: float
    summary: dict[str, Any] = Field(default_factory=dict)
