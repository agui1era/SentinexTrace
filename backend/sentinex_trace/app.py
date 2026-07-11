from __future__ import annotations

import json
import os
import re
import threading
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Dict

import numpy as np
from dotenv import dotenv_values, load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel

from .clip_embedder import ClipEmbedder
from .face_embedder import FaceEmbedder
from . import reporter
from .scanner import (
    ContinuousScanner,
    DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    DEFAULT_HUMAN_CONFIDENCE_THRESHOLD,
    HumanDetector,
    ScannerConfig,
)
from .store import VectorStore
from .stream import VideoStream

load_dotenv()


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
    savedSampleCount: Optional[int] = None
    threshold: Optional[float] = None
    sampleCount: int = 3
    sampleDelaySeconds: float = 0.18
    autoEnrollUnknown: bool = True
    unknownCooldownSeconds: Optional[float] = None


class RenameIdentityRequest(BaseModel):
    name: str
    tag: Optional[str] = None


class ReportRequest(BaseModel):
    fromDate: str
    toDate: str
    prompt: str = ""
    send: bool = True


class ChatHistoryItem(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    fromDate: str
    toDate: str
    message: str
    history: List[ChatHistoryItem] = []


class ThresholdsRequest(BaseModel):
    humanConfidenceThreshold: Optional[float] = None
    embeddingMatchThreshold: Optional[float] = None
    reuseThreshold: Optional[float] = None


CAMERA_ENV_PREFIXES = ("VITE_RTSP_URL_", "VITE_WEBCAM_URL_", "VITE_CAMERA_URL_")
PUBLIC_CONFIG_KEYS = (
    "VITE_SOURCE_MODE",
    "VITE_OPENCV_HUMAN_THRESHOLD",
    "VITE_HUMAN_CONFIDENCE_THRESHOLD",
    "VITE_REQUIRE_HUMAN",
    "VITE_SCAN_REQUIRE_HUMAN",
    "VITE_SCAN_COOLDOWN_SECONDS",
    "VITE_AUTO_SAMPLE_COUNT",
    "VITE_AUTO_ENROLL_UNKNOWN",
    "VITE_SCAN_AUTO_ENROLL_UNKNOWN",
    "VITE_COOLDOWN_SECONDS",
    "VITE_EMBEDDING_MATCH_THRESHOLD",
    "VITE_PATTERN_MATCH_THRESHOLD",
    "VITE_UNKNOWN_PATTERN_COOLDOWN_SECONDS",
    "VITE_CONFIDENCE_THRESHOLD",
    "YOLO_MODEL",
    "SENTINEX_EMBEDDER",
)
backend_logs: deque[Dict[str, Any]] = deque(maxlen=240)
backend_logs_lock = threading.RLock()
_last_logged_detection_count: Optional[int] = None
_last_logged_camera_status: Dict[str, str] = {}

app = FastAPI(title="Sentinex Vision", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def _build_embedder():
    mode = os.getenv("SENTINEX_EMBEDDER", os.getenv("VITE_EMBEDDER_MODE", "opencv")).strip().lower()
    if mode in {"face", "insightface", "arcface"}:
        return FaceEmbedder()
    return ClipEmbedder()


store = VectorStore()
embedder = _build_embedder()
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


@app.on_event("startup")
def start_configured_cameras() -> None:
    configured = configured_camera_urls()
    backend_log("info", f"Backend Vision iniciado. Camaras configuradas: {len(configured)}")
    for camera_id, url in configured:
        try:
            embedding_threshold = effective_float(
                (
                    "VITE_EMBEDDING_MATCH_THRESHOLD",
                    "VITE_PATTERN_MATCH_THRESHOLD",
                    "VITE_CONFIDENCE_THRESHOLD",
                ),
                DEFAULT_EMBEDDING_MATCH_THRESHOLD,
            )
            configure_stream(camera_id, StreamRequest(url=url))
            start_scan(
                camera_id,
                ScanRequest(
                    cooldownSeconds=env_float(("VITE_SCAN_COOLDOWN_SECONDS", "VITE_COOLDOWN_SECONDS"), 60.0),
                    embeddingMatchThreshold=embedding_threshold,
                    humanConfidenceThreshold=effective_float(
                        ("VITE_OPENCV_HUMAN_THRESHOLD", "VITE_HUMAN_CONFIDENCE_THRESHOLD"),
                        DEFAULT_HUMAN_CONFIDENCE_THRESHOLD,
                    ),
                    intervalSeconds=env_float(("VITE_SCAN_INTERVAL_SECONDS",), 1.0),
                    requireHuman=env_bool(("VITE_REQUIRE_HUMAN", "VITE_SCAN_REQUIRE_HUMAN"), False),
                    autoEnrollUnknown=env_bool(("VITE_AUTO_ENROLL_UNKNOWN", "VITE_SCAN_AUTO_ENROLL_UNKNOWN"), True),
                    reuseThreshold=embedding_threshold,
                    savedSampleCount=int(env_float(("VITE_AUTO_SAMPLE_COUNT",), 1)),
                    unknownCooldownSeconds=env_float(("VITE_UNKNOWN_PATTERN_COOLDOWN_SECONDS",), 120.0),
                ),
            )
            backend_log("ok", f"{camera_id}: stream y scanner automaticos iniciados")
        except Exception as exc:
            backend_log("error", f"{camera_id}: no se pudo iniciar escaneo automatico", error=str(exc))
            print(f"No se pudo iniciar escaneo automatico para {camera_id}: {exc}")


@app.on_event("shutdown")
def stop_configured_cameras() -> None:
    for camera_id, cam in list(cameras.items()):
        try:
            cam.scanner.stop()
            cam.stream.stop(clear_url=False)
            backend_log("info", f"{camera_id}: stream y scanner apagados")
        except Exception as exc:
            backend_log("error", f"{camera_id}: error apagando camara", error=str(exc))


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
    status = cam.stream.configure(request.url, camera_id=camera_id)
    backend_log("ok", f"{camera_id}: stream configurado")
    return {"stream": status}


@app.post("/api/vision/{camera_id}/stream/stop")
def stop_stream(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=False)
    cam.scanner.stop()
    stream_status = cam.stream.stop(clear_url=False)
    scanner_status = cam.scanner.status()
    backend_log("warn", f"{camera_id}: stream y scanner detenidos")
    return {"scanner": scanner_status, "stream": stream_status}


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
    reuse_threshold = threshold
    saved_sample_count = request.savedSampleCount if request.savedSampleCount is not None else 1
    unknown_cooldown_seconds = request.unknownCooldownSeconds if request.unknownCooldownSeconds is not None else 120.0
    config = ScannerConfig(
        cooldown_seconds=max(1, request.cooldownSeconds),
        human_confidence_threshold=min(0.99, max(0.05, human_threshold)),
        interval_seconds=max(0.25, request.intervalSeconds),
        require_human=request.requireHuman,
        reuse_threshold=min(threshold, max(0.1, reuse_threshold)),
        threshold=threshold,
        sample_count=max(1, min(5, request.sampleCount)),
        saved_sample_count=max(1, min(5, saved_sample_count)),
        sample_delay_seconds=max(0, min(1, request.sampleDelaySeconds)),
        auto_enroll_unknown=request.autoEnrollUnknown,
        unknown_cooldown_seconds=max(1, unknown_cooldown_seconds),
    )
    status = cam.scanner.start(config)
    backend_log(
        "ok",
        f"{camera_id}: scanner iniciado",
        threshold=config.threshold,
        humanConfidenceThreshold=config.human_confidence_threshold,
        intervalSeconds=config.interval_seconds,
        reuseThreshold=config.reuse_threshold,
        savedSampleCount=config.saved_sample_count,
        unknownCooldownSeconds=config.unknown_cooldown_seconds,
        requireHuman=config.require_human,
    )
    return {"scanner": status}


@app.post("/api/vision/{camera_id}/scan/stop")
def stop_scan(camera_id: str) -> dict:
    cam = get_camera(camera_id, create=False)
    status = cam.scanner.stop()
    backend_log("warn", f"{camera_id}: scanner detenido")
    return {"scanner": status}


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
    backend_log("warn", f"Patron eliminado: {identity_id}", identityId=identity_id)
    return {"identities": store.list_identities()}


@app.patch("/api/vision/identities/{identity_id}")
def rename_identity(identity_id: str, request: RenameIdentityRequest) -> dict:
    name = request.name.strip()
    tag = request.tag.strip() if request.tag is not None else None
    if len(name) < 2:
        raise HTTPException(status_code=400, detail="Nombre requerido")

    merge_threshold = effective_float(
        ("VITE_EMBEDDING_MATCH_THRESHOLD", "VITE_PATTERN_MATCH_THRESHOLD", "VITE_CONFIDENCE_THRESHOLD"),
        DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    )
    identity = store.rename_identity(identity_id, name=name, tag=tag, merge_threshold=merge_threshold)
    if not identity:
        raise HTTPException(status_code=404, detail="Identidad no encontrada")
    merged_count = len(identity.get("mergedIdentities", []))
    reassigned_count = int(identity.get("reassignedDetections", 0) or 0)
    if merged_count:
        backend_log(
            "ok",
            f"Identidad consolidada: {name} absorbe {merged_count} patrones y {reassigned_count} detecciones",
            identityId=identity_id,
            tag=tag or "",
            mergedIdentities=identity.get("mergedIdentities", []),
            mergeThreshold=merge_threshold,
        )
    else:
        backend_log("ok", f"Identidad renombrada: {name}", identityId=identity_id, tag=tag or "", mergeThreshold=merge_threshold)
    return {"identity": identity, "identities": store.list_identities()}


@app.get("/api/vision/detections")
def detections(limit: int = 50) -> dict:
    global _last_logged_detection_count
    items = store.list_detections(limit)
    total = len(items)
    if _last_logged_detection_count != total:
        backend_log("info", f"Detecciones disponibles: {total}")
        _last_logged_detection_count = total
    return {"detections": items}


@app.delete("/api/vision/detections/{detection_id}")
def delete_detection(detection_id: str, limit: int = 80) -> dict:
    deleted = store.delete_detection(detection_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Registro no encontrado")
    backend_log("warn", "Deteccion borrada", detectionId=detection_id)
    return {"detections": store.list_detections(limit)}


@app.get("/api/vision/report/config")
def report_config() -> dict:
    return reporter.report_config()


def thresholds_payload() -> Dict[str, Any]:
    embedding_threshold = effective_float(
        ("VITE_EMBEDDING_MATCH_THRESHOLD", "VITE_PATTERN_MATCH_THRESHOLD", "VITE_CONFIDENCE_THRESHOLD"),
        DEFAULT_EMBEDDING_MATCH_THRESHOLD,
    )
    return {
        "humanConfidenceThreshold": effective_float(
            ("VITE_OPENCV_HUMAN_THRESHOLD", "VITE_HUMAN_CONFIDENCE_THRESHOLD"),
            DEFAULT_HUMAN_CONFIDENCE_THRESHOLD,
        ),
        "embeddingMatchThreshold": embedding_threshold,
        "reuseThreshold": embedding_threshold,
        "embedder": embedder.status().get("mode", ""),
    }


@app.get("/api/vision/thresholds")
def get_thresholds() -> dict:
    return thresholds_payload()


@app.post("/api/vision/thresholds")
def update_thresholds(request: ThresholdsRequest) -> dict:
    overrides = load_runtime_overrides()
    scanner_changes: Dict[str, float] = {}

    def apply(value: float, env_key: str, scanner_field: str) -> None:
        clamped = max(0.0, min(1.0, float(value)))
        overrides[env_key] = clamped
        scanner_changes[scanner_field] = clamped

    if request.humanConfidenceThreshold is not None:
        apply(request.humanConfidenceThreshold, "VITE_OPENCV_HUMAN_THRESHOLD", "human_confidence_threshold")
    vector_threshold = (
        request.embeddingMatchThreshold
        if request.embeddingMatchThreshold is not None
        else request.reuseThreshold
    )
    if vector_threshold is not None:
        clamped = max(0.0, min(1.0, float(vector_threshold)))
        overrides["VITE_EMBEDDING_MATCH_THRESHOLD"] = clamped
        overrides["VITE_PATTERN_REUSE_THRESHOLD"] = clamped
        scanner_changes["threshold"] = clamped
        scanner_changes["reuse_threshold"] = clamped

    if not scanner_changes:
        raise HTTPException(status_code=400, detail="No hay umbrales para actualizar")

    save_runtime_overrides(overrides)
    for cam in cameras.values():
        try:
            cam.scanner.update_config(**scanner_changes)
        except Exception as exc:
            backend_log("error", f"No se pudo aplicar umbrales en {cam.camera_id}", error=str(exc))
    backend_log("ok", f"Umbrales actualizados en caliente: {scanner_changes}")
    return thresholds_payload()


@app.post("/api/vision/report")
def send_report(request: ReportRequest) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY no configurada")

    detections = store.detections_between(request.fromDate, request.toDate)
    identities = store.list_identities()
    summary = reporter.build_summary(detections, identities, request.fromDate, request.toDate)

    try:
        report_text = reporter.generate_report(summary, request.prompt, model=model, api_key=api_key)
    except Exception as exc:
        backend_log("error", "Fallo al generar informe con OpenAI", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Error OpenAI: {exc}")

    sent = False
    telegram_error = ""
    if request.send and token and chat_id:
        try:
            reporter.send_telegram(report_text, token=token, chat_id=chat_id)
            sent = True
        except Exception as exc:
            telegram_error = str(exc)
            backend_log("error", "Fallo al enviar informe a Telegram", error=str(exc))
    elif request.send and not (token and chat_id):
        telegram_error = "Telegram no configurado (falta TELEGRAM_BOT_TOKEN o TELEGRAM_CHAT_ID)"

    backend_log(
        "ok" if (sent or not request.send) else "warn",
        f"Informe generado ({len(detections)} detecciones)"
        + (" y enviado a Telegram" if sent else ""),
    )
    return {
        "report": report_text,
        "sent": sent,
        "telegramError": telegram_error,
        "detections": len(detections),
        "model": model,
    }


@app.post("/api/vision/chat")
def vision_chat(request: ChatRequest) -> dict:
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip() or "gpt-4o-mini"
    message = request.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Mensaje requerido")
    if not api_key:
        raise HTTPException(status_code=400, detail="OPENAI_API_KEY no configurada")

    detections = store.detections_between(request.fromDate, request.toDate)
    identities = store.list_identities()
    summary = reporter.build_summary(detections, identities, request.fromDate, request.toDate)
    history = [{"role": item.role, "content": item.content} for item in request.history]

    try:
        answer = reporter.generate_chat_response(summary, message, history, model=model, api_key=api_key)
    except Exception as exc:
        backend_log("error", "Fallo al responder chat Vision", error=str(exc))
        raise HTTPException(status_code=502, detail=f"Error OpenAI: {exc}")

    backend_log("ok", f"Chat IA respondido ({len(detections)} detecciones)")
    return {
        "answer": answer,
        "detections": len(detections),
        "fromDate": request.fromDate,
        "toDate": request.toDate,
        "model": model,
        "summary": summary,
    }


@app.get("/api/vision/logs")
def vision_logs(limit: int = 80) -> dict:
    log_camera_health()
    with backend_logs_lock:
        bounded_limit = max(1, min(limit, 200))
        return {"logs": list(backend_logs)[:bounded_limit]}


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


def backend_log(level: str, message: str, **extra: Any) -> None:
    print(f"[vision][{level}] {message}", flush=True)
    entry = {
        "id": f"{datetime.now(timezone.utc).timestamp()}-{len(backend_logs)}",
        "level": level,
        "message": message,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    with backend_logs_lock:
        backend_logs.appendleft(entry)


def log_camera_health() -> None:
    for camera_id, cam in list(cameras.items()):
        stream_status = cam.stream.status()
        scanner_status = cam.scanner.status()
        stream_error = stream_status.get("error") or ""
        scanner_error = scanner_status.get("error") or ""
        scanner_config = scanner_status.get("config") or {}

        if stream_error:
            level = "error"
            message = f"{camera_id}: stream con error: {stream_error}"
        elif not stream_status.get("running") or not stream_status.get("lastFrameAt"):
            level = "warn"
            message = f"{camera_id}: esperando frames del stream"
        elif not scanner_status.get("running"):
            level = "warn"
            message = f"{camera_id}: scanner detenido"
        elif scanner_error:
            level = "warn" if "Sin humano" in scanner_error else "error"
            message = f"{camera_id}: scanner {scanner_error}"
        else:
            require_human = bool(scanner_config.get("requireHuman"))
            human_count = int(scanner_status.get("humanCount") or 0)
            if require_human:
                message = f"{camera_id}: buscando personas; humanos={human_count}"
            else:
                message = f"{camera_id}: capturando patrones"
            level = "ok"

        signature = f"{level}:{message}"
        if _last_logged_camera_status.get(camera_id) == signature:
            continue
        _last_logged_camera_status[camera_id] = signature
        backend_log(level, message)


def configured_camera_urls() -> List[tuple[str, str]]:
    camera_url_map: Dict[str, str] = {}
    for key, value in os.environ.items():
        if not value.strip():
            continue
        camera_id = camera_id_from_env_key(key)
        if camera_id:
            camera_url_map[camera_id] = value.strip()

    if not camera_url_map and os.getenv("VITE_RTSP_URL", "").strip():
        camera_url_map["CAM_MAIN"] = os.getenv("VITE_RTSP_URL", "").strip()

    return sorted(camera_url_map.items(), key=lambda item: item[0])


def camera_id_from_env_key(key: str) -> str:
    for prefix in CAMERA_ENV_PREFIXES:
        if key.startswith(prefix):
            return normalize_camera_id(key[len(prefix):])

    match = re.match(r"^VITE_(CAM[A-Z0-9_-]+|WEBCAM[A-Z0-9_-]*)_(?:RTSP_)?URL$", key, re.IGNORECASE)
    if match:
        return normalize_camera_id(match.group(1))
    return ""


def normalize_camera_id(raw_id: str) -> str:
    cleaned = re.sub(r"[^a-z0-9_-]", "", raw_id.strip(), flags=re.IGNORECASE).upper()
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return f"CAM{cleaned}"
    if cleaned.startswith("CAM") or cleaned.startswith("WEBCAM"):
        return cleaned
    return f"CAM{cleaned}"


def env_float(keys: tuple[str, ...], fallback: float) -> float:
    for key in keys:
        raw = os.getenv(key, "").strip()
        if not raw:
            continue
        try:
            return float(raw)
        except ValueError:
            continue
    return fallback


def env_bool(keys: tuple[str, ...], fallback: bool) -> bool:
    truthy = {"1", "true", "yes", "y", "on", "si", "sí"}
    falsy = {"0", "false", "no", "n", "off"}
    for key in keys:
        raw = os.getenv(key, "").strip().lower()
        if not raw:
            continue
        if raw in truthy:
            return True
        if raw in falsy:
            return False
    return fallback


def runtime_config_path() -> Path:
    return store.data_dir / "runtime_config.json"


def load_runtime_overrides() -> Dict[str, Any]:
    path = runtime_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_runtime_overrides(data: Dict[str, Any]) -> None:
    path = runtime_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def effective_float(keys: tuple[str, ...], fallback: float) -> float:
    overrides = load_runtime_overrides()
    for key in keys:
        if key in overrides:
            try:
                return float(overrides[key])
            except (TypeError, ValueError):
                continue
    return env_float(keys, fallback)

@app.get("/api/vision/config")
def get_config() -> dict:
    return {"config": public_config_values()}


def public_config_values() -> Dict[str, str]:
    env_file_values = dotenv_values(Path(".env")) if Path(".env").exists() else {}
    config: Dict[str, str] = {}
    for key in PUBLIC_CONFIG_KEYS:
        raw_value = env_file_values.get(key)
        value = (raw_value if isinstance(raw_value, str) else os.getenv(key, "")).strip()
        if value:
            config[key] = value

    for camera_id, _url in configured_camera_urls():
        config[f"VITE_RTSP_URL_{camera_id}"] = "configured"

    return config
