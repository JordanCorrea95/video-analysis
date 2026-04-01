from pathlib import Path
from uuid import uuid4

from fastapi import UploadFile


def build_unique_path(base_dir: Path, original_name: str) -> Path:
    suffix = Path(original_name).suffix
    stem = Path(original_name).stem
    return base_dir / f"{stem}_{uuid4().hex}{suffix}"


def save_upload_file(upload: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as f:
        while chunk := upload.file.read(1024 * 1024):
            f.write(chunk)
