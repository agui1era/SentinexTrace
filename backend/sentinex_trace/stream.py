from __future__ import annotations

import threading
import time
from typing import Any, Dict, Generator, Optional
from urllib.parse import parse_qs, urlsplit

import numpy as np

try:
    import cv2
except Exception:  # pragma: no cover - reported through status at runtime.
    cv2 = None  # type: ignore[assignment]


class VideoStream:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture: Optional[Any] = None
        self._url = ""
        self._camera_id = ""
        self._latest_frame: Optional[np.ndarray] = None
        self._latest_jpeg: Optional[bytes] = None
        self._last_frame_at: Optional[float] = None
        self._error = ""
        self._clients = 0

    def configure(self, url: str, camera_id: str = "") -> Dict[str, Any]:
        next_url = url.strip()
        next_camera_id = camera_id.strip().upper()
        if not next_url:
            self.stop(clear_url=True)
            with self._lock:
                self._error = ""
                self._camera_id = ""
            return self.status()

        self.stop(clear_url=False)
        with self._lock:
            self._url = next_url
            self._camera_id = next_camera_id or infer_camera_id(next_url)
            self._error = ""
            self._latest_frame = None
            self._latest_jpeg = None
            self._last_frame_at = None

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sentinex-rtsp", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self, *, clear_url: bool = False) -> Dict[str, Any]:
        self._stop_event.set()
        capture = None
        with self._lock:
            capture = self._capture
            self._capture = None

        if capture is not None:
            try:
                capture.release()
            except Exception:
                pass

        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)

        with self._lock:
            self._thread = None
            self._latest_frame = None
            self._latest_jpeg = None
            self._last_frame_at = None
            if clear_url:
                self._url = ""
                self._camera_id = ""
            if not clear_url and self._url:
                self._error = ""

        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "clients": self._clients,
                "cameraId": self._camera_id,
                "configured": bool(self._url),
                "error": self._error,
                "lastFrameAt": self._last_frame_at,
                "running": bool(self._thread and self._thread.is_alive()),
                "url": mask_stream_url(self._url),
            }

    def camera_id(self) -> str:
        with self._lock:
            return self._camera_id or infer_camera_id(self._url)

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._latest_frame is None else self._latest_frame.copy()

    def latest_jpeg(self) -> Optional[bytes]:
        with self._lock:
            return self._latest_jpeg

    def mjpeg_frames(self) -> Generator[bytes, None, None]:
        with self._lock:
            self._clients += 1

        try:
            while not self._stop_event.is_set():
                frame = self.latest_jpeg()
                if frame:
                    yield (
                        b"--sentinexframe\r\n"
                        b"content-type: image/jpeg\r\n"
                        + f"content-length: {len(frame)}\r\n\r\n".encode("ascii")
                        + frame
                        + b"\r\n"
                    )
                time.sleep(0.12)
        finally:
            with self._lock:
                self._clients = max(0, self._clients - 1)

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        if cv2 is None:
            raise RuntimeError("OpenCV no esta instalado")

        ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            raise RuntimeError("No se pudo codificar JPEG")
        return bytes(buffer)

    def _run(self) -> None:
        if cv2 is None:
            with self._lock:
                self._error = "OpenCV no esta instalado. Ejecuta pip install -r requirements-vision.txt"
            return

        while not self._stop_event.is_set():
            url = self._url
            source: Any = int(url) if url.isdigit() else url
            capture = cv2.VideoCapture(source)

            with self._lock:
                self._capture = capture

            if not capture.isOpened():
                with self._lock:
                    self._error = "No se pudo abrir el stream RTSP"
                time.sleep(2)
                continue

            try:
                capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except Exception:
                pass

            with self._lock:
                self._error = ""

            while not self._stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    with self._lock:
                        self._error = "Stream sin frames; reintentando conexion"
                    time.sleep(0.35)
                    break

                try:
                    jpeg = self.encode_jpeg(frame)
                except Exception as exc:
                    with self._lock:
                        self._error = str(exc)
                    continue

                with self._lock:
                    self._latest_frame = frame
                    self._latest_jpeg = jpeg
                    self._last_frame_at = time.time()
                    self._error = ""

            capture.release()
            with self._lock:
                if self._capture is capture:
                    self._capture = None

            if not self._stop_event.is_set():
                time.sleep(1)


def mask_stream_url(url: str) -> str:
    if not url:
        return ""

    try:
        from urllib.parse import urlsplit, urlunsplit

        parsed = urlsplit(url)
        if parsed.username or parsed.password:
            host = parsed.hostname or ""
            if parsed.port:
                host = f"{host}:{parsed.port}"
            return urlunsplit((parsed.scheme, f"***:***@{host}", parsed.path, parsed.query, parsed.fragment))
    except Exception:
        return url.replace("//", "//***:***@", 1) if "@" in url else url

    return url


def infer_camera_id(url: str) -> str:
    if not url:
        return ""

    if url.isdigit():
        return f"CAM{url}"

    try:
        parsed = urlsplit(url)
        params = parse_qs(parsed.query)
        channel = params.get("channel", [""])[0]
        if channel:
            return f"CAM{channel}"

        path_digits = "".join(char for char in parsed.path if char.isdigit())
        if path_digits:
            return f"CAM{path_digits[-1]}"
    except Exception:
        pass

    return "CAM"
