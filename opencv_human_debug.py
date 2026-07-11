from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import dotenv_values, load_dotenv


CAMERA_ENV_PREFIXES = ("VITE_RTSP_URL_", "VITE_WEBCAM_URL_", "VITE_CAMERA_URL_")
WINDOW_NAME = "Sentinex Human Debug"
DISPLAY_WIDTH = 640
DETECT_EVERY_FRAMES = 3
PROBE_HUMAN_CONFIDENCE = 0.05
MAX_HUMAN_AREA_RATIO = 0.45
MAX_HUMAN_WIDTH_RATIO = 0.85
MIN_HUMAN_AREA_RATIO = 0.0025
MIN_HUMAN_HEIGHT_RATIO = 0.05

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sentinex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sentinex-cache")


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str


@dataclass(frozen=True)
class HumanBox:
    xyxy: Tuple[int, int, int, int]
    confidence: float


@dataclass(frozen=True)
class DetectionResult:
    boxes: List[HumanBox]
    raw_count: int
    rejected_count: int
    best_raw_confidence: float
    threshold: float
    probe_threshold: float


class SharedYolo:
    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO("yolov8n.pt")
        self._lock = threading.RLock()

    def detect(self, frame: np.ndarray, threshold: float) -> DetectionResult:
        probe_threshold = min(PROBE_HUMAN_CONFIDENCE, max(0.03, threshold))
        with self._lock:
            results = self._model(
                frame,
                classes=[0],
                conf=probe_threshold,
                verbose=False,
            )

        height, width = frame.shape[:2]
        boxes: List[HumanBox] = []
        raw_count = 0
        rejected_count = 0
        best_raw_confidence = 0.0
        for result in results:
            for raw_box, raw_confidence in zip(result.boxes.xyxy, result.boxes.conf):
                raw_count += 1
                confidence = float(raw_confidence)
                best_raw_confidence = max(best_raw_confidence, confidence)
                x1, y1, x2, y2 = [int(value) for value in raw_box.tolist()]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))
                if confidence < threshold or not is_reasonable_human_box((x1, y1, x2, y2), width, height):
                    rejected_count += 1
                    continue
                boxes.append(HumanBox((x1, y1, x2, y2), confidence))

        boxes.sort(key=lambda item: item.confidence, reverse=True)
        return DetectionResult(
            boxes=boxes,
            raw_count=raw_count,
            rejected_count=rejected_count,
            best_raw_confidence=best_raw_confidence,
            threshold=threshold,
            probe_threshold=probe_threshold,
        )


class CameraWorker(threading.Thread):
    def __init__(self, camera: CameraConfig, yolo: SharedYolo, threshold_getter) -> None:
        super().__init__(daemon=True)
        self.camera = camera
        self.yolo = yolo
        self.threshold_getter = threshold_getter
        self.frame = placeholder_frame(camera.camera_id, "conectando stream")
        self.status = "conectando stream"
        self.result = empty_detection_result()
        self._last_log_at = 0.0
        self._stop_event = threading.Event()
        self._lock = threading.RLock()

    def stop(self) -> None:
        self._stop_event.set()

    def snapshot(self) -> Tuple[np.ndarray, str]:
        with self._lock:
            return self.frame.copy(), self.status

    def run(self) -> None:
        frame_index = 0
        result = empty_detection_result()
        cap = self._open_capture()
        while not self._stop_event.is_set():
            if cap is None or not cap.isOpened():
                self._set_frame(placeholder_frame(self.camera.camera_id, "reconectando stream"), "reconectando stream")
                time.sleep(1.2)
                cap = self._open_capture()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self._set_frame(placeholder_frame(self.camera.camera_id, "sin frames; reconectando"), "sin frames; reconectando")
                cap.release()
                cap = None
                continue

            frame = resize_for_display(frame, DISPLAY_WIDTH)
            frame_index += 1
            threshold = self.threshold_getter()
            if frame_index % DETECT_EVERY_FRAMES == 0:
                try:
                    result = self.yolo.detect(frame, threshold)
                    self._log_detection(result)
                except Exception as exc:
                    result = empty_detection_result(threshold)
                    self._set_frame(placeholder_frame(self.camera.camera_id, f"YOLO error: {exc}"), f"YOLO error: {exc}")
                    time.sleep(0.2)
                    continue

            status = status_text(result)
            self._set_frame(draw_overlay(frame, self.camera.camera_id, threshold, result), status)
            time.sleep(0.01)

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
        return cap

    def _set_frame(self, frame: np.ndarray, status: str) -> None:
        with self._lock:
            self.frame = frame
            self.status = status

    def _log_detection(self, result: DetectionResult) -> None:
        now = time.time()
        if now - self._last_log_at < 2.0:
            return
        self._last_log_at = now
        accepted = len(result.boxes)
        best_accepted = result.boxes[0].confidence if result.boxes else 0.0
        print(
            (
                f"[{time.strftime('%H:%M:%S')}] {self.camera.camera_id} "
                f"accepted={accepted} best={best_accepted:.2f} "
                f"raw={result.raw_count} rejected={result.rejected_count} "
                f"best_raw={result.best_raw_confidence:.2f} threshold={result.threshold:.2f} probe={result.probe_threshold:.2f}"
            ),
            flush=True,
        )


def draw_overlay(frame: np.ndarray, camera_id: str, threshold: float, result: DetectionResult) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, camera_id, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, f"threshold {threshold:.2f}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 220, 255), 2)
    cv2.putText(
        out,
        f"accepted {len(result.boxes)} raw {result.raw_count} rejected {result.rejected_count} probe {result.probe_threshold:.2f}",
        (16, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (190, 220, 255),
        2,
    )

    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy
        color = (0, 255, 70)
        label = f"human {box.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(out, (x1, label_y), (x1 + label_size[0] + 8, label_y + label_size[1] + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, label_y + label_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return out


def placeholder_frame(camera_id: str, text: str) -> np.ndarray:
    frame = np.zeros((360, DISPLAY_WIDTH, 3), dtype=np.uint8)
    frame[:] = (13, 17, 24)
    cv2.putText(frame, camera_id, (16, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    cv2.putText(frame, text[:70], (16, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 185, 205), 2)
    return frame


def status_text(result: DetectionResult) -> str:
    if not result.boxes:
        return (
            f"sin humano >= {result.threshold:.2f} "
            f"(raw {result.raw_count}, rechazados {result.rejected_count}, best raw {result.best_raw_confidence:.2f})"
        )
    scores = ", ".join(f"{box.confidence:.2f}" for box in result.boxes[:4])
    return f"{len(result.boxes)} humano(s): {scores} | raw {result.raw_count}, rechazados {result.rejected_count}"


def empty_detection_result(threshold: float = 0.0) -> DetectionResult:
    return DetectionResult(
        boxes=[],
        raw_count=0,
        rejected_count=0,
        best_raw_confidence=0.0,
        threshold=threshold,
        probe_threshold=PROBE_HUMAN_CONFIDENCE,
    )


def resize_for_display(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = target_width / width
    target_height = max(1, int(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_mosaic(frames: List[np.ndarray], statuses: List[str], threshold: float) -> np.ndarray:
    if not frames:
        return placeholder_frame("Sentinex", "sin camaras")

    normalized = [resize_to_tile(frame, DISPLAY_WIDTH, 360) for frame in frames]
    columns = 2 if len(normalized) > 1 else 1
    rows = int(np.ceil(len(normalized) / columns))
    tile_height, tile_width = normalized[0].shape[:2]
    mosaic = np.zeros((rows * tile_height + 52, columns * tile_width, 3), dtype=np.uint8)
    mosaic[:] = (17, 21, 27)

    cv2.putText(
        mosaic,
        f"Sentinex Human Debug | threshold {threshold:.2f} | q/esc cerrar",
        (16, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (238, 242, 248),
        2,
    )

    for index, frame in enumerate(normalized):
        row = index // columns
        column = index % columns
        y = 52 + row * tile_height
        x = column * tile_width
        mosaic[y : y + tile_height, x : x + tile_width] = frame
        cv2.rectangle(mosaic, (x, y), (x + tile_width - 1, y + tile_height - 1), (46, 55, 69), 1)
        cv2.putText(mosaic, statuses[index][:80], (x + 14, y + tile_height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 232), 2)

    return mosaic


def resize_to_tile(frame: np.ndarray, tile_width: int, tile_height: int) -> np.ndarray:
    resized = cv2.resize(frame, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
    return resized


def is_reasonable_human_box(box: Tuple[int, int, int, int], frame_width: int, frame_height: int) -> bool:
    x1, y1, x2, y2 = box
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    frame_area = max(1, frame_width * frame_height)
    area_ratio = (width * height) / frame_area
    width_ratio = width / max(1, frame_width)
    height_ratio = height / max(1, frame_height)
    if area_ratio < MIN_HUMAN_AREA_RATIO:
        return False
    if height_ratio < MIN_HUMAN_HEIGHT_RATIO:
        return False
    if area_ratio > MAX_HUMAN_AREA_RATIO:
        return False
    if width_ratio > MAX_HUMAN_WIDTH_RATIO:
        return False
    return True


def load_cameras() -> List[CameraConfig]:
    load_dotenv()
    env_values = dotenv_values(Path(".env")) if Path(".env").exists() else {}
    camera_map: Dict[str, str] = {}
    for key, value in env_values.items():
        if not isinstance(value, str) or not value.strip():
            continue
        camera_id = camera_id_from_key(key)
        if camera_id:
            camera_map[camera_id] = value.strip()

    if not camera_map and os.getenv("VITE_RTSP_URL", "").strip():
        camera_map["CAM_MAIN"] = os.getenv("VITE_RTSP_URL", "").strip()

    return [
        CameraConfig(camera_id, url)
        for camera_id, url in sorted(camera_map.items(), key=lambda item: item[0])
    ]


def camera_id_from_key(key: str) -> str:
    for prefix in CAMERA_ENV_PREFIXES:
        if key.startswith(prefix):
            return normalize_camera_id(key[len(prefix) :])
    return ""


def normalize_camera_id(raw_id: str) -> str:
    cleaned = "".join(char for char in raw_id.upper() if char.isalnum() or char in ("_", "-"))
    if not cleaned:
        return ""
    if cleaned.isdigit():
        return f"CAM{cleaned}"
    if cleaned.startswith("CAM") or cleaned.startswith("WEBCAM"):
        return cleaned
    return f"CAM{cleaned}"


def env_float(key: str, fallback: float) -> float:
    try:
        return float(os.getenv(key, fallback))
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    cameras = load_cameras()
    if not cameras:
        print("No hay camaras configuradas en .env")
        return 1

    threshold_value = int(round(env_float("VITE_OPENCV_HUMAN_THRESHOLD", 0.18) * 100))
    threshold_lock = threading.RLock()

    def get_threshold() -> float:
        with threshold_lock:
            return threshold_value_holder[0] / 100.0

    def on_threshold_change(value: int) -> None:
        with threshold_lock:
            threshold_value_holder[0] = max(3, min(90, value))

    threshold_value_holder = [max(3, min(90, threshold_value))]
    yolo = SharedYolo()
    workers = [CameraWorker(camera, yolo, get_threshold) for camera in cameras]
    for worker in workers:
        worker.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1320, 820)
    cv2.createTrackbar("threshold x100", WINDOW_NAME, threshold_value_holder[0], 90, on_threshold_change)
    cv2.setTrackbarMin("threshold x100", WINDOW_NAME, 3)

    try:
        while True:
            snapshots = [worker.snapshot() for worker in workers]
            frames = [frame for frame, _ in snapshots]
            statuses = [status for _, status in snapshots]
            cv2.imshow(WINDOW_NAME, build_mosaic(frames, statuses, get_threshold()))
            key = cv2.waitKey(30) & 0xFF
            if key in (27, ord("q")):
                break
    finally:
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.join(timeout=2.0)
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
