from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from app.core.config import settings
from app.models.schemas import InferenceMode


class YOLOService:
    def __init__(self) -> None:
        self._models: dict[InferenceMode, YOLO] = {}

    def _resolve_weights(self, mode: InferenceMode) -> str:
        mapping = {
            InferenceMode.detect: settings.model_detect,
            InferenceMode.segment: settings.model_segment,
            InferenceMode.pose: settings.model_pose,
        }
        return mapping[mode]

    def get_model(self, mode: InferenceMode) -> YOLO:
        if mode not in self._models:
            weights = self._resolve_weights(mode)
            self._models[mode] = YOLO(weights)
        return self._models[mode]

    def process_video(
        self,
        mode: InferenceMode,
        input_video: Path,
        output_video: Path,
        confidence: float,
    ) -> dict[str, Any]:
        model = self.get_model(mode)

        capture = cv2.VideoCapture(str(input_video))
        if not capture.isOpened():
            raise RuntimeError(f"No se pudo abrir el video: {input_video}")

        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if width <= 0 or height <= 0:
            capture.release()
            raise RuntimeError("Resolucion de video invalida")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"No se pudo crear el video de salida: {output_video}")

        total_frames = 0
        class_counter: Counter[str] = Counter()
        per_frame_counts: list[int] = []

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                results = model.predict(frame, conf=confidence, verbose=False)
                result = results[0]
                plotted = result.plot(font_size=20, line_width=2)
                writer.write(plotted)

                frame_count = 0

                # Manejo según el modo de inferencia
                if mode == InferenceMode.detect:
                    if result.boxes is not None and len(result.boxes) > 0:
                        cls_ids = result.boxes.cls.tolist()
                        frame_count = len(cls_ids)
                        names = result.names
                        for cls_id in cls_ids:
                            class_counter[names[int(cls_id)]] += 1

                elif mode == InferenceMode.segment:
                    if result.masks is not None and len(result.masks) > 0:
                        frame_count = len(result.masks)
                        if result.boxes is not None:
                            cls_ids = result.boxes.cls.tolist()
                            names = result.names
                            for cls_id in cls_ids:
                                class_counter[names[int(cls_id)]] += 1

                elif mode == InferenceMode.pose:
                    if result.keypoints is not None and len(result.keypoints) > 0:
                        frame_count = len(result.keypoints)
                        if result.boxes is not None:
                            cls_ids = result.boxes.cls.tolist()
                            names = result.names
                            for cls_id in cls_ids:
                                class_counter[names[int(cls_id)]] += 1

                per_frame_counts.append(frame_count)
                total_frames += 1
        finally:
            capture.release()
            writer.release()

        avg_detections = (
            sum(per_frame_counts) / len(per_frame_counts) if per_frame_counts else 0.0
        )

        summary = {
            "total_detections": int(sum(per_frame_counts)),
            "avg_detections_per_frame": round(avg_detections, 3),
            "class_counts": dict(class_counter),
            "mode": mode.value,
        }

        return {
            "total_frames": total_frames,
            "fps": float(fps),
            "summary": summary,
        }


yolo_service = YOLOService()
