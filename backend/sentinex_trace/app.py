from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .clip_embedder import ClipEmbedder
from .scanner import (
    ContinuousScanner,
    DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    DEFAULT_HUMAN_CONFIDENCE_THRESHOLD,
    HumanDetector,
    ScannerConfig,
)
from .store import VectorStore
from .stream import VideoStream


class StreamRequest(BaseModel):
    url: str


class EnrollRequest(BaseModel):
    name: str
    tag: str = ""
    sampleIds: List[str]


class ScanRequest(BaseModel):
    cooldownSeconds: float = 10
    embeddingMatchThreshold: Optional[float] = None
    humanConfidenceThreshold: Optional[float] = None
    intervalSeconds: float = 1.4
    opencvHumanThreshold: Optional[float] = None
    requireHuman: bool = True
    reuseThreshold: Optional[float] = None
    threshold: Optional[float] = None
    sampleCount: int = 3
    sampleDelaySeconds: float = 0.18
    autoEnrollUnknown: bool = True


class RenameIdentityRequest(BaseModel):
    name: str
    tag: Optional[str] = None


app = FastAPI(title="Sentinex Vision", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = VectorStore()
embedder = ClipEmbedder()
detector = HumanDetector()

class CameraInstance:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self.stream = VideoStream()
        self.scanner = ContinuousScanner(
            stream=self.stream,
            store=store,
            embedder=embedder,
            detector=detector
        )

cameras: Dict[str, CameraInstance] = {}

def get_camera(camera_id: str, create: bool = False) -> CameraInstance:
    if camera_id not in cameras:
        if create:
            cameras[camera_id] = CameraInstance(camera_id)
        else:
            raise HTTPException(status_code=404, detail=f"Camara {camera_id} no encontrada")
    return cameras[camera_id]


@app.get("/api/vision/status")
def global_status() -> dict:
    return {
        "clip": embedder.status(),
        "detector": detector.status(),
        "store": store.status(),
        "cameras": list(cameras.keys())
    }


@app.get("/api/vision/{camera_id}/status")
def camera_status(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=True)
    return {
        "scanner": cam.scanner.status(),
        "stream": cam.stream.status(),
    }


@app.post("/api/vision/{camera_id}/stream")
def configure_stream(camera_id: str, request: StreamRequest) -> dict:
    cam = get_camera(camera_id, create=True)
    return {"stream": cam.stream.configure(request.url, camera_id=camera_id)}


@app.post("/api/vision/{camera_id}/stream/stop")
def stop_stream(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=False)
    cam.scanner.stop()
    return {"scanner": cam.scanner.status(), "stream": cam.stream.stop(clear_url=False)}


@app.get("/api/vision/{camera_id}/stream.mjpg")
def stream_mjpeg(camera_id: str) -> StreamingResponse:
    cam = get_camera(camera_id, create=True)
    return StreamingResponse(
        cam.stream.mjpeg_frames(),
        media_type="multipart/x-mixed-replace; boundary=sentinexframe",
    )


@app.get("/api/vision/{camera_id}/snapshot")
def snapshot(camera_id: str) -> Response:
    cam = get_camera(camera_id, create=False)
    jpeg = cam.stream.latest_jpeg()
    if not jpeg:
        raise HTTPException(status_code=503, detail="Aun no hay frame del stream")
    return Response(content=jpeg, media_type="image/jpeg", headers={"cache-control": "no-store"})


@app.post("/api/vision/{camera_id}/scan/start")
def start_scan(camera_id: str, request: ScanRequest) -> dict:
    cam = get_camera(camera_id, create=True)
    raw_human_threshold = (
        request.opencvHumanThreshold
        if request.opencvHumanThreshold is not None
        else request.humanConfidenceThreshold
    )
    raw_embedding_threshold = (
        request.embeddingMatchThreshold
        if request.embeddingMatchThreshold is not None
        else request.threshold
    )
    human_threshold = raw_human_threshold if raw_human_threshold is not None else DEFAULT_HUMAN_CONFIDENCE_THRESHOLD
    embedding_threshold = (
        raw_embedding_threshold if raw_embedding_threshold is not None else DEFAULT_EMBEDDING_MATCH_THRESHOLD
    )
    threshold = min(0.99, max(0.1, embedding_threshold))
    reuse_threshold = request.reuseThreshold if request.reuseThreshold is not None else threshold
    config = ScannerConfig(
        cooldown_seconds=max(1, request.cooldownSeconds),
        human_confidence_threshold=min(0.99, max(0.05, human_threshold)),
        interval_seconds=max(0.25, request.intervalSeconds),
        require_human=request.requireHuman,
        reuse_threshold=min(threshold, max(0.1, reuse_threshold)),
        threshold=threshold,
        sample_count=max(1, min(5, request.sampleCount)),
        sample_delay_seconds=max(0, min(1, request.sampleDelaySeconds)),
        auto_enroll_unknown=request.autoEnrollUnknown,
    )
    return {"scanner": cam.scanner.start(config)}


@app.post("/api/vision/{camera_id}/scan/stop")
def stop_scan(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=False)
    return {"scanner": cam.scanner.stop()}


@app.get("/api/vision/enroll/captures")
def list_pending_captures() -> dict:
    captures = []
    for path in sorted(store.pending_dir.glob("*.jpg")):
        capture_id = path.stem
        captures.append({"id": capture_id, "url": f"/api/vision/media/{store.media_key(path)}"})
    return {"captures": captures}


@app.post("/api/vision/{camera_id}/enroll/capture")
def capture_for_enrollment(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=False)
    jpeg = cam.stream.latest_jpeg()
    if not jpeg:
        raise HTTPException(status_code=503, detail="No hay frame para capturar")
    capture = store.save_pending_capture(jpeg)
    return {"capture": capture, "captures": list_pending_captures()["captures"]}


@app.delete("/api/vision/enroll/captures")
def clear_pending_captures() -> dict:
    store.clear_pending()
    return {"captures": []}


@app.post("/api/vision/enroll")
def enroll_identity(request: EnrollRequest) -> dict:
    name = request.name.strip()
    tag = request.tag.strip()
    sample_ids = request.sampleIds[:5]

    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Nombre requerido")
    if len(sample_ids) < 3:
        raise HTTPException(status_code=400, detail="Se necesitan al menos 3 imagenes")
    if len(sample_ids) > 5:
        raise HTTPException(status_code=400, detail="Maximo 5 imagenes")

    sample_paths = [store.pending_capture_path(sample_id) for sample_id in sample_ids]
    missing = [path.name for path in sample_paths if not path.exists()]
    if missing:
        raise HTTPException(status_code=404, detail=f"Capturas no encontradas: {', '.join(missing)}")

    try:
        frames = [read_bgr(path) for path in sample_paths]
        vectors = [embedder.embed_bgr(frame) for frame in frames]
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    average = np.mean(np.stack(vectors, axis=0), axis=0)
    identity = store.add_identity(name=name, tag=tag, embedding=average, sample_paths=sample_paths)
    store.clear_pending()
    return {"identity": identity, "identities": store.list_identities()}


@app.get("/api/vision/identities")
def identities() -> dict:
    return {"identities": store.list_identities()}


@app.delete("/api/vision/identities/{identity_id}")
def delete_identity(identity_id: str) -> dict:
    deleted = store.delete_identity(identity_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Identidad no encontrada")
    return {"identities": store.list_identities()}


@app.patch("/api/vision/identities/{identity_id}")
def rename_identity(identity_id: str, request: RenameIdentityRequest) -> dict:
    name = request.name.strip()
    tag = request.tag.strip() if request.tag is not None else None
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Nombre requerido")

    identity = store.rename_identity(identity_id, name=name, tag=tag)
    if not identity:
        raise HTTPException(status_code=404, detail="Identidad no encontrada")
    return {"identity": identity, "identities": store.list_identities()}


@app.get("/api/vision/detections")
def detections(limit: int = 50) -> dict:
    return {"detections": store.list_detections(limit)}


@app.delete("/api/vision/detections/{detection_id}")
def delete_detection(detection_id: str, limit: int = 80) -> dict:
    deleted = store.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    return {"detections": store.list_detections(limit)}


@app.get("/api/vision/media/{media_key:path}")
def media(media_key: str) -> FileResponse:
    try:
        path = store.resolve_media_path(media_key)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Media no encontrada") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return FileResponse(path, media_type="image/jpeg")


def read_bgr(path: Path) -> np.ndarray:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV no esta instalado") from exc

    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"No se pudo leer imagen {path.name}")
    return frame
