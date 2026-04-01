# Video Summary Backend

Backend en Python con FastAPI para procesar video con YOLO en tres modos:
- `detect` (identificacion de objetos)
- `segment` (segmentacion)
- `pose` (pose-estimation)

## 1) Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 2) Modelos YOLO26m

Los modelos YOLO26m ya están incluidos en el directorio `models/`:
- `models/yolo26m.pt` (42 MB) - Detección de objetos
- `models/yolo26m-seg.pt` (52 MB) - Segmentación
- `models/yolo26m-pose.pt` (47 MB) - Estimación de pose

Si necesitas descargarlos manualmente:
```bash
mkdir -p models
cd models
curl -L -o yolo26m.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m.pt"
curl -L -o yolo26m-seg.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-seg.pt"
curl -L -o yolo26m-pose.pt "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26m-pose.pt"
```

## 3) Ejecutar API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger UI:
- `http://localhost:8000/docs`

## 4) Endpoint principal

`POST /api/v1/infer`

Parámetros:
- `mode`: `detect`, `segment`, `pose`
- `confidence`: float entre `0` y `1` (default `0.25`)
- `video`: archivo (`.mp4`, `.mov`, `.avi`, `.mkv`)

Ejemplo con curl:

```bash
curl -X POST "http://localhost:8000/api/v1/infer?mode=detect&confidence=0.25" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "video=@/ruta/a/tu/video.mp4"
```

Respuesta (ejemplo):

```json
{
  "mode": "detect",
  "input_filename": "video.mp4",
  "output_video_path": "data/outputs/video_xxx.processed.mp4",
  "total_frames": 340,
  "fps": 29.97,
  "summary": {
    "total_detections": 800,
    "avg_detections_per_frame": 2.35,
    "class_counts": {
      "person": 500,
      "car": 120
    }
  }
}
```

## 5) Notas

- **Ultralytics versión**: Requiere `ultralytics>=8.4.33` para soporte completo de YOLO26
- Los videos subidos quedan en `data/uploads` y las salidas anotadas en `data/outputs`
- En caso de error, los archivos temporales se limpian automáticamente
- Los modelos se cargan lazy (solo cuando se necesitan) y se mantienen en memoria
