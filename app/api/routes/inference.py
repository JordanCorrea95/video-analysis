from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from app.core.config import settings
from app.models.schemas import InferenceMode, InferenceResponse
from app.services.storage import build_unique_path, save_upload_file
from app.services.yolo_service import yolo_service

router = APIRouter(prefix="/api/v1", tags=["inference"])


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/infer", response_model=InferenceResponse)
def run_inference(
    mode: InferenceMode = Query(..., description="detect | segment | pose"),
    confidence: float = Query(settings.default_confidence, ge=0.0, le=1.0),
    video: UploadFile = File(...),
) -> InferenceResponse:
    if not video.filename:
        raise HTTPException(status_code=400, detail="Archivo invalido")

    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".avi", ".mkv"}:
        raise HTTPException(status_code=400, detail="Formato no soportado")

    input_path = build_unique_path(settings.upload_dir, video.filename)
    output_path = input_path.with_suffix(".processed.mp4")
    output_path = settings.output_dir / output_path.name

    try:
        save_upload_file(video, input_path)
        processed = yolo_service.process_video(
            mode=mode,
            input_video=input_path,
            output_video=output_path,
            confidence=confidence,
        )
    except FileNotFoundError as exc:
        # Limpiar archivo de entrada si existe
        if input_path.exists():
            input_path.unlink()
        raise HTTPException(status_code=404, detail=f"Archivo no encontrado: {exc}") from exc
    except ValueError as exc:
        # Limpiar archivos si hay error de validación
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=400, detail=f"Error de validación: {exc}") from exc
    except Exception as exc:
        # Limpiar archivos en caso de error general
        if input_path.exists():
            input_path.unlink()
        if output_path.exists():
            output_path.unlink()
        raise HTTPException(status_code=500, detail=f"Error al procesar video: {str(exc)}") from exc
    finally:
        video.file.close()

    return InferenceResponse(
        mode=mode,
        input_filename=video.filename,
        output_video_path=str(output_path),
        total_frames=processed["total_frames"],
        fps=processed["fps"],
        summary=processed["summary"],
    )
