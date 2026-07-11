from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import dotenv_values, load_dotenv


CAMERA_ENV_PREFIXES = ("VITE_RTSP_URL_", "VITE_WEBCAM_URL_", "VITE_CAMERA_URL_")
COCO_PERSON_CLASS_ID = 0
PERSON_CLASS_NAME = "person"
WINDOW_NAME = "Sentinex Basic Human YOLO"
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 360
PROCESS_WIDTH = 1280

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sentinex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sentinex-cache")


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str


@dataclass(frozen=True)
class Detection:
    bbox_xyxy: Tuple[int, int, int, int]
    confidence: float


class CameraReader(threading.Thread):
    def __init__(self, camera: CameraConfig) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.status = "connecting"
        self.frame: np.ndarray | None = None
        self._lock = threading.RLock()
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> Tuple[np.ndarray | None, str]:
        with self._lock:
            return (None if self.frame is None else self.frame.copy(), self.status)

    def run(self) -> None:
        cap = self._open_capture()
        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                self._set_status("reconnecting")
                time.sleep(1.0)
                cap = self._open_capture()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self._set_status("no frame")
                cap.release()
                cap = None
                continue

            self._set_frame(resize_frame(frame, PROCESS_WIDTH), "stream ok")
            time.sleep(0.005)

        if cap is not None:
            cap.release()

    def _open_capture(self) -> cv2.VideoCapture | None:
        source = int(self.camera.url) if self.camera.url.isdigit() else self.camera.url
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            try:
                cap.release()
            except Exception:
                pass
            return None
        self._set_status("stream ok")
        return cap

    def _set_frame(self, frame: np.ndarray, status: str) -> None:
        with self._lock:
            self.frame = frame
            self.status = status

    def _set_status(self, status: str) -> None:
        with self._lock:
            self.status = status


class HumanYolo:
    def __init__(self, model_name: str) -> None:
        from ultralytics import YOLO

        self.model_name = model_name
        self.model = YOLO(model_name)

    def detect(self, frame: np.ndarray, min_confidence: float) -> List[Detection]:
        results = self.model(
            frame,
            classes=[COCO_PERSON_CLASS_ID],
            conf=max(0.01, min(0.99, min_confidence)),
            verbose=False,
        )

        height, width = frame.shape[:2]
        detections: List[Detection] = []
        for result in results:
            for raw_box, raw_confidence in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = [int(value) for value in raw_box.tolist()]
                detections.append(
                    Detection(
                        bbox_xyxy=(
                            max(0, min(width - 1, x1)),
                            max(0, min(height - 1, y1)),
                            max(1, min(width, x2)),
                            max(1, min(height, y2)),
                        ),
                        confidence=float(raw_confidence),
                    )
                )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections


class FpsMeter:
    def __init__(self, window_seconds: float = 2.0) -> None:
        self.window_seconds = window_seconds
        self.timestamps: List[float] = []

    def tick(self, now: float) -> float:
        self.timestamps.append(now)
        cutoff = now - self.window_seconds
        self.timestamps = [timestamp for timestamp in self.timestamps if timestamp >= cutoff]
        if len(self.timestamps) < 2:
            return 0.0
        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def draw_detections(
    frame: np.ndarray,
    camera_id: str,
    status: str,
    model_name: str,
    threshold: float,
    process_resolution: Tuple[int, int],
    yolo_fps: float,
    inference_ms: float,
    detections: List[Detection],
    best_candidate_confidence: float | None,
) -> np.ndarray:
    out = frame.copy()
    process_width, process_height = process_resolution
    cv2.putText(out, camera_id, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, f"{status} | threshold {threshold:.2f}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 220, 255), 2)
    cv2.putText(
        out,
        f"{model_name} | proc {process_width}x{process_height} | fps {yolo_fps:.1f} | infer {inference_ms:.0f}ms",
        (16, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (190, 220, 255),
        2,
    )
    best_text = f"best person {best_candidate_confidence:.2f}" if best_candidate_confidence is not None else "best person none"
    cv2.putText(out, f"humans {len(detections)} | {best_text}", (16, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (190, 220, 255), 2)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        label = f"{PERSON_CLASS_NAME} {detection.confidence:.2f}"
        color = (0, 255, 70)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(out, (x1, label_y), (x1 + label_size[0] + 8, label_y + label_size[1] + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, label_y + label_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return resize_to_tile(out)


def log_detections(
    log_path: Path,
    *,
    camera_id: str,
    frame: np.ndarray,
    model_name: str,
    crop_dir: Path,
    threshold: float,
    process_resolution: Tuple[int, int],
    yolo_fps: float,
    inference_ms: float,
    detections: List[Detection],
) -> None:
    if not detections:
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    crop_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc)
    timestamp_slug = timestamp.strftime("%Y%m%dT%H%M%S%fZ")
    logged_detections = []
    for index, item in enumerate(detections, start=1):
        x1, y1, x2, y2 = item.bbox_xyxy
        crop = frame[y1:y2, x1:x2]
        crop_path = ""
        if crop.size:
            crop_file = crop_dir / f"{timestamp_slug}_{camera_id}_{index}_conf{item.confidence:.2f}.jpg"
            cv2.imwrite(str(crop_file), crop)
            crop_path = str(crop_file)
        logged_detections.append(
            {
                "confidence": round(item.confidence, 4),
                "bbox_xyxy": list(item.bbox_xyxy),
                "cropPath": crop_path,
            }
        )

    event = {
        "timestamp": timestamp.isoformat(),
        "cameraId": camera_id,
        "model": model_name,
        "classId": COCO_PERSON_CLASS_ID,
        "className": PERSON_CLASS_NAME,
        "threshold": threshold,
        "processResolution": {
            "width": process_resolution[0],
            "height": process_resolution[1],
        },
        "yoloFps": round(yolo_fps, 2),
        "inferenceMs": round(inference_ms, 2),
        "count": len(detections),
        "detections": logged_detections,
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(json.dumps(event, ensure_ascii=False), flush=True)


def build_mosaic(frames: List[np.ndarray], title: str) -> np.ndarray:
    if not frames:
        frames = [placeholder_frame("Sentinex", "no cameras")]

    columns = 2 if len(frames) > 1 else 1
    rows = int(np.ceil(len(frames) / columns))
    tile_height, tile_width = DISPLAY_HEIGHT, DISPLAY_WIDTH
    header_height = 54
    mosaic = np.zeros((rows * tile_height + header_height, columns * tile_width, 3), dtype=np.uint8)
    mosaic[:] = (17, 21, 27)

    cv2.putText(mosaic, title, (16, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (238, 242, 248), 2)
    for index, frame in enumerate(frames):
        row = index // columns
        column = index % columns
        y = header_height + row * tile_height
        x = column * tile_width
        mosaic[y : y + tile_height, x : x + tile_width] = resize_to_tile(frame)
        cv2.rectangle(mosaic, (x, y), (x + tile_width - 1, y + tile_height - 1), (46, 55, 69), 1)
    return mosaic


def placeholder_frame(camera_id: str, text: str) -> np.ndarray:
    frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    frame[:] = (13, 17, 24)
    cv2.putText(frame, camera_id, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    cv2.putText(frame, text[:72], (16, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 185, 205), 2)
    return frame


def resize_frame(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    target_height = max(1, int(height * target_width / width))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def resize_to_tile(frame: np.ndarray) -> np.ndarray:
    return cv2.resize(frame, (DISPLAY_WIDTH, DISPLAY_HEIGHT), interpolation=cv2.INTER_AREA)


def load_cameras() -> List[CameraConfig]:
    load_dotenv()
    env_values = dotenv_values(Path(".env")) if Path(".env").exists() else {}
    cameras: Dict[str, str] = {}
    for key, value in env_values.items():
        if not isinstance(value, str) or not value.strip():
            continue
        camera_id = camera_id_from_key(key)
        if camera_id:
            cameras[camera_id] = value.strip()

    return [CameraConfig(camera_id, url) for camera_id, url in sorted(cameras.items())]


def camera_id_from_key(key: str) -> str:
    for prefix in CAMERA_ENV_PREFIXES:
        if key.startswith(prefix):
            raw = key[len(prefix) :].upper()
            cleaned = "".join(char for char in raw if char.isalnum() or char in ("_", "-"))
            if not cleaned:
                return ""
            return cleaned if cleaned.startswith(("CAM", "WEBCAM")) else f"CAM{cleaned}"
    return ""


def env_float(key: str, fallback: float) -> float:
    try:
        return float(os.getenv(key, fallback))
    except (TypeError, ValueError):
        return fallback


def env_int(key: str, fallback: int) -> int:
    try:
        return int(os.getenv(key, fallback))
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    cameras = load_cameras()
    if not cameras:
        print("No cameras configured in .env")
        return 1

    threshold = env_float("VITE_OPENCV_HUMAN_THRESHOLD", 0.50)
    model_name = os.getenv("YOLO_MODEL", "yolov8m.pt").strip() or "yolov8m.pt"
    log_path = Path(os.getenv("HUMAN_DETECTION_LOG", "human_yolo_basic.log"))
    crop_dir = Path(os.getenv("HUMAN_DETECTION_CROP_DIR", "data/human_yolo_basic/crops"))
    log_interval_seconds = env_float("HUMAN_DETECTION_LOG_INTERVAL_SECONDS", 1.0)
    max_crops_per_event = max(1, min(5, env_int("HUMAN_DETECTION_MAX_CROPS_PER_EVENT", 1)))
    status_interval_seconds = env_float("HUMAN_DETECTION_STATUS_INTERVAL_SECONDS", 10.0)
    inference_confidence = threshold

    print(
        f"Starting basic YOLO human detector: model={model_name} class={COCO_PERSON_CLASS_ID}:{PERSON_CLASS_NAME} "
        f"threshold={threshold:.2f} cameras={','.join(camera.camera_id for camera in cameras)} "
        f"log={log_path} crops={crop_dir}",
        flush=True,
    )

    yolo = HumanYolo(model_name)
    readers = [CameraReader(camera) for camera in cameras]
    last_log_at: Dict[str, float] = {}
    last_status_at: Dict[str, float] = {}
    fps_meters = {camera.camera_id: FpsMeter() for camera in cameras}
    for reader in readers:
        reader.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1320, 460 if len(readers) <= 2 else 820)

    try:
        while True:
            rendered_frames: List[np.ndarray] = []
            for reader in readers:
                frame, status = reader.snapshot()
                if frame is None:
                    rendered_frames.append(placeholder_frame(reader.camera.camera_id, status))
                    continue

                process_resolution = (int(frame.shape[1]), int(frame.shape[0]))
                started = time.perf_counter()
                candidates = yolo.detect(frame, inference_confidence)
                detections = [candidate for candidate in candidates if candidate.confidence >= threshold]
                best_candidate_confidence = candidates[0].confidence if candidates else None
                finished = time.perf_counter()
                inference_ms = (finished - started) * 1000
                yolo_fps = fps_meters[reader.camera.camera_id].tick(finished)
                now = time.time()
                if now - last_status_at.get(reader.camera.camera_id, 0.0) >= status_interval_seconds:
                    best_status = f"{best_candidate_confidence:.3f}" if best_candidate_confidence is not None else "none"
                    print(
                        f"status camera={reader.camera.camera_id} humans={len(detections)} "
                        f"bestPerson={best_status} threshold={threshold:.2f} fps={yolo_fps:.1f} inferMs={inference_ms:.0f}",
                        flush=True,
                    )
                    last_status_at[reader.camera.camera_id] = now
                if detections and now - last_log_at.get(reader.camera.camera_id, 0.0) >= log_interval_seconds:
                    logged_detections = detections[:max_crops_per_event]
                    log_detections(
                        log_path,
                        camera_id=reader.camera.camera_id,
                        frame=frame,
                        model_name=model_name,
                        crop_dir=crop_dir,
                        threshold=threshold,
                        process_resolution=process_resolution,
                        yolo_fps=yolo_fps,
                        inference_ms=inference_ms,
                        detections=logged_detections,
                    )
                    last_log_at[reader.camera.camera_id] = now

                rendered_frames.append(
                    draw_detections(
                        frame,
                        reader.camera.camera_id,
                        status,
                        model_name,
                        threshold,
                        process_resolution,
                        yolo_fps,
                        inference_ms,
                        detections,
                        best_candidate_confidence,
                    )
                )

            title = f"Basic YOLO Human Detection | {model_name} | class 0 person | threshold {threshold:.2f} | q/esc cerrar"
            cv2.imshow(WINDOW_NAME, build_mosaic(rendered_frames, title))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for reader in readers:
            reader.stop()
        for reader in readers:
            reader.join(timeout=2.0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
