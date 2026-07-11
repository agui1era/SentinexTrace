from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
from dotenv import dotenv_values, load_dotenv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sentinex")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/sentinex-cache")

from PyQt6.QtCore import QCoreApplication, QLibraryInfo, Qt, QThread, pyqtSignal as Signal, pyqtSlot as Slot  # noqa: E402
from PyQt6.QtGui import QImage, QPixmap  # noqa: E402
from PyQt6.QtWidgets import (  # noqa: E402
    QApplication,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


CAMERA_ENV_PREFIXES = ("VITE_RTSP_URL_", "VITE_WEBCAM_URL_", "VITE_CAMERA_URL_")
DISPLAY_WIDTH = 640
DETECT_EVERY_FRAMES = 3
MAX_HUMAN_AREA_RATIO = 0.45
MAX_HUMAN_WIDTH_RATIO = 0.85
MIN_HUMAN_AREA_RATIO = 0.0025
MIN_HUMAN_HEIGHT_RATIO = 0.05


@dataclass(frozen=True)
class CameraConfig:
    camera_id: str
    url: str


@dataclass(frozen=True)
class HumanBox:
    xyxy: Tuple[int, int, int, int]
    confidence: float


class SharedYolo:
    def __init__(self) -> None:
        from ultralytics import YOLO

        self._model = YOLO("yolov8n.pt")
        self._lock = threading.RLock()

    def detect(self, frame: np.ndarray, threshold: float) -> List[HumanBox]:
        with self._lock:
            results = self._model(
                frame,
                classes=[0],
                conf=max(0.03, min(0.99, threshold)),
                verbose=False,
            )

        height, width = frame.shape[:2]
        boxes: List[HumanBox] = []
        for result in results:
            for raw_box, raw_confidence in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = [int(value) for value in raw_box.tolist()]
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                x2 = max(x1 + 1, min(width, x2))
                y2 = max(y1 + 1, min(height, y2))
                if not is_reasonable_human_box((x1, y1, x2, y2), width, height):
                    continue
                boxes.append(HumanBox((x1, y1, x2, y2), float(raw_confidence)))

        boxes.sort(key=lambda item: item.confidence, reverse=True)
        return boxes


class CameraWorker(QThread):
    frame_ready = Signal(str, QImage, str)
    status_ready = Signal(str, str)

    def __init__(self, camera: CameraConfig, yolo: SharedYolo, threshold: float) -> None:
        super().__init__()
        self.camera = camera
        self.yolo = yolo
        self.threshold = threshold
        self._last_boxes: List[HumanBox] = []

    @Slot(float)
    def set_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def run(self) -> None:
        frame_index = 0
        cap = self._open_capture()
        while not self.isInterruptionRequested():
            if cap is None or not cap.isOpened():
                self.status_ready.emit(self.camera.camera_id, "reconectando stream")
                self.msleep(1200)
                cap = self._open_capture()
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                self.status_ready.emit(self.camera.camera_id, "sin frames; reconectando")
                cap.release()
                cap = None
                continue

            frame = resize_for_display(frame, DISPLAY_WIDTH)
            frame_index += 1
            if frame_index % DETECT_EVERY_FRAMES == 0:
                try:
                    self._last_boxes = self.yolo.detect(frame, self.threshold)
                except Exception as exc:
                    self._last_boxes = []
                    self.status_ready.emit(self.camera.camera_id, f"YOLO error: {exc}")

            annotated = draw_overlay(frame, self.camera.camera_id, self.threshold, self._last_boxes)
            status = status_text(self._last_boxes, self.threshold)
            self.frame_ready.emit(self.camera.camera_id, image_from_bgr(annotated), status)
            self.msleep(15)

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
        self.status_ready.emit(self.camera.camera_id, "stream conectado")
        return cap


class CameraTile(QFrame):
    def __init__(self, camera: CameraConfig) -> None:
        super().__init__()
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("cameraTile")

        self.video = QLabel()
        self.video.setMinimumSize(420, 240)
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setStyleSheet("background: #090b0f; color: #8d99aa;")

        self.title = QLabel(camera.camera_id)
        self.title.setObjectName("tileTitle")
        self.status = QLabel("esperando stream")
        self.status.setObjectName("tileStatus")

        layout = QVBoxLayout()
        layout.addWidget(self.title)
        layout.addWidget(self.video, 1)
        layout.addWidget(self.status)
        self.setLayout(layout)

    def set_frame(self, image: QImage, status: str) -> None:
        self.video.setPixmap(QPixmap.fromImage(image))
        self.status.setText(status)

    def set_status(self, status: str) -> None:
        self.status.setText(status)


class HumanDebugWindow(QMainWindow):
    threshold_changed = Signal(float)

    def __init__(self, cameras: List[CameraConfig]) -> None:
        super().__init__()
        self.setWindowTitle("Sentinex Human Debug - Qt")
        self.resize(1440, 900)

        self.threshold = env_float("VITE_OPENCV_HUMAN_THRESHOLD", 0.18)
        self.yolo = SharedYolo()
        self.tiles: Dict[str, CameraTile] = {}
        self.workers: List[CameraWorker] = []

        root = QWidget()
        main_layout = QVBoxLayout()
        main_layout.addLayout(self._build_toolbar())

        grid = QGridLayout()
        grid.setSpacing(10)
        for index, camera in enumerate(cameras):
            tile = CameraTile(camera)
            self.tiles[camera.camera_id] = tile
            grid.addWidget(tile, index // 2, index % 2)
        main_layout.addLayout(grid, 1)

        root.setLayout(main_layout)
        self.setCentralWidget(root)
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #11151b; color: #eef2f8; font-size: 14px; }
            #cameraTile { border: 1px solid #2e3745; border-radius: 6px; background: #171d25; }
            #tileTitle { font-size: 18px; font-weight: 700; }
            #tileStatus { color: #aeb8c8; }
            QPushButton { background: #253142; border: 1px solid #3c4b61; padding: 8px 12px; border-radius: 4px; }
            QPushButton:hover { background: #314057; }
            QSlider::groove:horizontal { height: 6px; background: #2e3745; border-radius: 3px; }
            QSlider::handle:horizontal { width: 16px; margin: -6px 0; background: #4da3ff; border-radius: 8px; }
            """
        )

        self._start_workers(cameras)

    def _build_toolbar(self) -> QHBoxLayout:
        layout = QHBoxLayout()
        self.threshold_label = QLabel()
        self._update_threshold_label()

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(3, 90)
        slider.setValue(int(round(self.threshold * 100)))
        slider.valueChanged.connect(self._set_threshold_from_slider)

        quit_button = QPushButton("Cerrar")
        quit_button.clicked.connect(self.close)

        layout.addWidget(QLabel("Umbral humano"))
        layout.addWidget(slider, 1)
        layout.addWidget(self.threshold_label)
        layout.addWidget(quit_button)
        return layout

    def _start_workers(self, cameras: List[CameraConfig]) -> None:
        for camera in cameras:
            worker = CameraWorker(camera, self.yolo, self.threshold)
            self.threshold_changed.connect(worker.set_threshold)
            worker.frame_ready.connect(self._set_camera_frame)
            worker.status_ready.connect(self._set_camera_status)
            worker.start()
            self.workers.append(worker)

    @Slot(int)
    def _set_threshold_from_slider(self, value: int) -> None:
        self.threshold = value / 100.0
        self._update_threshold_label()
        self.threshold_changed.emit(self.threshold)

    @Slot(str, QImage, str)
    def _set_camera_frame(self, camera_id: str, image: QImage, status: str) -> None:
        tile = self.tiles.get(camera_id)
        if tile:
            tile.set_frame(image, status)

    @Slot(str, str)
    def _set_camera_status(self, camera_id: str, status: str) -> None:
        tile = self.tiles.get(camera_id)
        if tile:
            tile.set_status(status)

    def _update_threshold_label(self) -> None:
        self.threshold_label.setText(f"{self.threshold:.2f}")

    def closeEvent(self, event) -> None:  # type: ignore[override]
        for worker in self.workers:
            worker.requestInterruption()
        for worker in self.workers:
            worker.wait(2000)
        event.accept()


def draw_overlay(
    frame: np.ndarray,
    camera_id: str,
    threshold: float,
    boxes: List[HumanBox],
) -> np.ndarray:
    out = frame.copy()
    cv2.putText(out, camera_id, (16, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(out, f"threshold {threshold:.2f}", (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (190, 220, 255), 2)

    for box in boxes:
        x1, y1, x2, y2 = box.xyxy
        color = (0, 255, 70)
        label = f"human {box.confidence:.2f}"
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        label_y = max(0, y1 - label_size[1] - 8)
        cv2.rectangle(out, (x1, label_y), (x1 + label_size[0] + 8, label_y + label_size[1] + 8), color, -1)
        cv2.putText(out, label, (x1 + 4, label_y + label_size[1] + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    return out


def status_text(boxes: List[HumanBox], threshold: float) -> str:
    if not boxes:
        return f"sin humano >= {threshold:.2f}"
    scores = ", ".join(f"{box.confidence:.2f}" for box in boxes[:4])
    return f"{len(boxes)} humano(s): {scores}"


def image_from_bgr(frame: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb.shape
    image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
    return image.copy()


def resize_for_display(frame: np.ndarray, target_width: int) -> np.ndarray:
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        return frame
    scale = target_width / width
    target_height = max(1, int(height * scale))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)


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
            return normalize_camera_id(key[len(prefix):])
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

    QCoreApplication.setLibraryPaths([QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)])
    app = QApplication(sys.argv)
    window = HumanDebugWindow(cameras)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
