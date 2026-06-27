from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .clip_embedder import ClipEmbedder
from .store import VectorStore
from .stream import VideoStream

try:
    import cv2
except Exception:  # pragma: no cover - reported through status at runtime.
    cv2 = None  # type: ignore[assignment]


BBox = Tuple[int, int, int, int]
DEFAULT_HUMAN_CONFIDENCE_THRESHOLD = 0.45
DEFAULT_EMBEDDING_MATCH_THRESHOLD = 0.6


@dataclass
class ScannerConfig:
    threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD
    reuse_threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD
    human_confidence_threshold: float = DEFAULT_HUMAN_CONFIDENCE_THRESHOLD
    interval_seconds: float = 1.4
    cooldown_seconds: float = 10
    require_human: bool = True
    sample_count: int = 3
    sample_delay_seconds: float = 0.18
    auto_enroll_unknown: bool = True


class HumanDetector:
    def __init__(self) -> None:
        self._model: Optional[Any] = None
        self._error = ""

    def status(self) -> Dict[str, Any]:
        return {
            "available": True,
            "error": self._error,
            "mode": "yolov8n",
        }

    def detect(self, frame: np.ndarray, confidence: float = DEFAULT_HUMAN_CONFIDENCE_THRESHOLD) -> List[BBox]:
        try:
            from ultralytics import YOLO
        except ImportError:
            self._error = "ultralytics no esta instalado"
            return []

        if self._model is None:
            import logging
            logging.getLogger("ultralytics").setLevel(logging.WARNING)
            try:
                self._model = YOLO("yolov8n.pt")
            except Exception as exc:
                self._error = f"Error cargando YOLO: {exc}"
                return []

        try:
            bounded_confidence = min(0.99, max(0.05, confidence))
            results = self._model(frame, classes=[0], conf=bounded_confidence, verbose=False)
        except Exception as exc:
            self._error = f"Error ejecutando YOLO: {exc}"
            return []

        height, width = frame.shape[:2]
        boxes: List[BBox] = []
        for result in results:
            for box in result.boxes.xyxy:
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                boxes.append(clamp_bbox((x1, y1, x2 - x1, y2 - y1), width, height))

        self._error = ""
        # Sort by area
        boxes.sort(key=lambda item: item[2] * item[3], reverse=True)
        return boxes[:4]


class ContinuousScanner:
    def __init__(
        self,
        *,
        stream: VideoStream,
        store: VectorStore,
        embedder: ClipEmbedder,
        detector: HumanDetector,
    ) -> None:
        self.stream = stream
        self.store = store
        self.embedder = embedder
        self.detector = detector
        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._config = ScannerConfig()
        self._last_error = ""
        self._last_scan_at: Optional[float] = None
        self._last_match: Optional[Dict[str, Any]] = None
        self._human_count = 0
        self._cooldowns: Dict[str, float] = {}

    def start(self, config: ScannerConfig) -> Dict[str, Any]:
        self.stop()
        with self._lock:
            self._config = config
            self._last_error = ""
            self._last_match = None
            self._human_count = 0

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sentinex-scanner", daemon=True)
        self._thread.start()
        return self.status()

    def stop(self) -> Dict[str, Any]:
        self._stop_event.set()
        thread = self._thread
        if thread and thread.is_alive():
            thread.join(timeout=1.5)
        with self._lock:
            self._thread = None
        return self.status()

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "config": {
                    "cooldownSeconds": self._config.cooldown_seconds,
                    "embeddingMatchThreshold": self._config.threshold,
                    "humanConfidenceThreshold": self._config.human_confidence_threshold,
                    "intervalSeconds": self._config.interval_seconds,
                    "opencvHumanThreshold": self._config.human_confidence_threshold,
                    "requireHuman": self._config.require_human,
                    "reuseThreshold": self._config.reuse_threshold,
                    "sampleCount": self._config.sample_count,
                    "threshold": self._config.threshold,
                },
                "error": self._last_error,
                "humanCount": self._human_count,
                "lastMatch": self._last_match,
                "lastScanAt": self._last_scan_at,
                "running": bool(self._thread and self._thread.is_alive()),
            }

    def _run(self) -> None:
        while not self._stop_event.is_set():
            started = time.time()
            try:
                self._scan_once()
            except Exception as exc:
                with self._lock:
                    self._last_error = str(exc)

            interval = max(0.25, self._config.interval_seconds)
            elapsed = time.time() - started
            self._stop_event.wait(max(0.05, interval - elapsed))

    def _scan_once(self) -> None:
        frame = self.stream.latest_frame()
        jpeg = self.stream.latest_jpeg()
        if frame is None or jpeg is None:
            with self._lock:
                self._last_error = "Esperando frames RTSP"
            return

        config = self._config
        boxes = self.detector.detect(frame, config.human_confidence_threshold) if config.require_human else []
        if config.require_human and not boxes:
            with self._lock:
                self._human_count = 0
                self._last_error = "Sin humano detectado"
                self._last_scan_at = time.time()
            return

        targets: List[Tuple[Optional[BBox], np.ndarray]] = []
        if boxes:
            bbox = boxes[0]
            crop = crop_bbox(frame, bbox)
            if crop.size:
                targets.append((bbox, crop))
        else:
            targets.append((None, frame))

        best: Optional[Dict[str, Any]] = None
        best_embedding: Optional[np.ndarray] = None
        best_samples: List[Tuple[Optional[BBox], np.ndarray]] = []
        for bbox, target in targets:
            samples = self._collect_samples(bbox, target)
            vectors = [self.embedder.embed_bgr(sample_target) for _sample_bbox, sample_target in samples]
            embedding = np.mean(np.stack(vectors, axis=0), axis=0)
            matches = self.store.search(embedding, limit=1)
            candidate = matches[0] if matches else {"identity": None, "score": 0.0}
            candidate["bbox"] = list(bbox) if bbox else None
            if best is None or candidate["score"] > best["score"]:
                best = candidate
                best_embedding = embedding
                best_samples = samples

        now = time.time()
        with self._lock:
            self._human_count = len(boxes) if boxes else 1
            self._last_scan_at = now
            self._last_match = best
            self._last_error = ""

        if best is None or best_embedding is None or not best_samples:
            return

        matched_identity = best.get("identity")
        score = float(best["score"])
        reuse_threshold = min(config.threshold, config.reuse_threshold)
        is_known = bool(matched_identity) and score >= config.threshold
        is_reusable_pattern = bool(matched_identity) and score >= reuse_threshold
        camera_id = self.stream.camera_id()
        if is_known or is_reusable_pattern:
            identity = matched_identity
            last_seen = self._cooldowns.get(identity["id"], 0)
            if now - last_seen < config.cooldown_seconds:
                return
            match_status = "known" if is_known else "known_low_confidence"
        elif config.auto_enroll_unknown:
            sample_paths = self._save_sample_frames(best_samples)
            pattern_id = self.store.next_unknown_name(camera_id)
            identity = self.store.add_identity(
                name=pattern_id,
                tag="",
                embedding=best_embedding,
                sample_paths=sample_paths,
                generic=True,
                camera_id=camera_id,
                pattern_id=pattern_id,
            )
            match_status = "unknown_auto_enrolled"
        else:
            return

        frame_path = self.store.save_detection_frame(jpeg)
        sample_paths = self._save_sample_frames(best_samples) if is_known else sample_paths
        detection = self.store.add_detection(
            identity=identity,
            score=score,
            frame_path=frame_path,
            bbox=best.get("bbox"),
            source=camera_id or "rtsp",
            match_status=match_status,
            sample_frames=sample_paths,
        )
        self._cooldowns[identity["id"]] = now
        with self._lock:
            self._last_match = {"identity": identity, "score": score, "detection": detection}

    def _collect_samples(self, bbox: Optional[BBox], fallback_target: np.ndarray) -> List[Tuple[Optional[BBox], np.ndarray]]:
        samples: List[Tuple[Optional[BBox], np.ndarray]] = []
        count = max(1, min(5, self._config.sample_count))
        delay = max(0.0, min(1.0, self._config.sample_delay_seconds))

        for index in range(count):
            if index > 0:
                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    break

            frame = self.stream.latest_frame()
            if frame is None:
                continue

            sample_bbox = bbox
            target = fallback_target
            if self._config.require_human:
                boxes = self.detector.detect(frame, self._config.human_confidence_threshold)
                if boxes:
                    sample_bbox = boxes[0]
                    target = crop_bbox(frame, sample_bbox)
                elif bbox:
                    target = crop_bbox(frame, bbox)
            else:
                target = frame

            if target.size:
                samples.append((sample_bbox, target))

        return samples or [(bbox, fallback_target)]

    def _save_sample_frames(self, samples: List[Tuple[Optional[BBox], np.ndarray]]) -> List[Any]:
        paths = []
        for _bbox, sample in samples[:5]:
            paths.append(self.store.save_auto_sample(encode_jpeg(sample)))
        return paths


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def crop_bbox(frame: np.ndarray, bbox: BBox) -> np.ndarray:
    x, y, w, h = bbox
    pad_x = int(w * 0.08)
    pad_y = int(h * 0.08)
    height, width = frame.shape[:2]
    x0 = max(0, x - pad_x)
    y0 = max(0, y - pad_y)
    x1 = min(width, x + w + pad_x)
    y1 = min(height, y + h + pad_y)
    return frame[y0:y1, x0:x1]


def encode_jpeg(frame: np.ndarray) -> bytes:
    if cv2 is None:
        raise RuntimeError("OpenCV no esta instalado")
    ok, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("No se pudo codificar muestra JPEG")
    return bytes(buffer)
