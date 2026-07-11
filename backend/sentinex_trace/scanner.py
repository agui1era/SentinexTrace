from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, replace
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


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# Gate de tamano del bounding box humano. El minimo controla la "proximidad":
# solo se capturan personas que ocupan al menos esta fraccion del alto/area del
# frame (configurable por .env). El maximo descarta cajas demasiado grandes.
MAX_HUMAN_AREA_RATIO = _env_float("VITE_MAX_HUMAN_AREA_RATIO", 0.45)
MAX_HUMAN_WIDTH_RATIO = _env_float("VITE_MAX_HUMAN_WIDTH_RATIO", 0.85)
MIN_HUMAN_AREA_RATIO = _env_float("VITE_MIN_HUMAN_AREA_RATIO", 0.0025)
MIN_HUMAN_HEIGHT_RATIO = _env_float("VITE_MIN_HUMAN_HEIGHT_RATIO", 0.05)


@dataclass(frozen=True)
class HumanDetection:
    bbox: BBox
    confidence: float


@dataclass
class ScannerConfig:
    threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD
    reuse_threshold: float = DEFAULT_EMBEDDING_MATCH_THRESHOLD
    human_confidence_threshold: float = DEFAULT_HUMAN_CONFIDENCE_THRESHOLD
    interval_seconds: float = 1.4
    cooldown_seconds: float = 10
    unknown_cooldown_seconds: float = 120
    require_human: bool = True
    sample_count: int = 3
    saved_sample_count: int = 1
    sample_delay_seconds: float = 0.18
    auto_enroll_unknown: bool = True


class HumanDetector:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._model: Optional[Any] = None
        self._model_name = os.getenv("YOLO_MODEL", "yolov8n.pt").strip() or "yolov8n.pt"
        self._error = ""

    def status(self) -> Dict[str, Any]:
        return {
            "available": True,
            "error": self._error,
            "maxAreaRatio": MAX_HUMAN_AREA_RATIO,
            "maxWidthRatio": MAX_HUMAN_WIDTH_RATIO,
            "minAreaRatio": MIN_HUMAN_AREA_RATIO,
            "minHeightRatio": MIN_HUMAN_HEIGHT_RATIO,
            "mode": self._model_name,
        }

    def detect(self, frame: np.ndarray, confidence: float = DEFAULT_HUMAN_CONFIDENCE_THRESHOLD) -> List[HumanDetection]:
        try:
            from ultralytics import YOLO
        except ImportError:
            self._error = "ultralytics no esta instalado"
            return []

        with self._lock:
            if self._model is None:
                import logging
                logging.getLogger("ultralytics").setLevel(logging.WARNING)
                try:
                    self._model = YOLO(self._model_name)
                except Exception as exc:
                    self._error = f"Error cargando YOLO: {exc}"
                    return []

            try:
                bounded_confidence = min(0.99, max(0.03, confidence))
                results = self._model(frame, classes=[0], conf=bounded_confidence, verbose=False)
            except Exception as exc:
                self._error = f"Error ejecutando YOLO: {exc}"
                return []

        height, width = frame.shape[:2]
        detections: List[HumanDetection] = []
        for result in results:
            for box, confidence_score in zip(result.boxes.xyxy, result.boxes.conf):
                x1, y1, x2, y2 = [int(v) for v in box.tolist()]
                bbox = clamp_bbox((x1, y1, x2 - x1, y2 - y1), width, height)
                if not reasonable_human_bbox(bbox, width, height):
                    continue
                detections.append(
                    HumanDetection(
                        bbox=bbox,
                        confidence=float(confidence_score),
                    )
                )

        self._error = ""
        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[:4]


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
        self._human_scores: List[float] = []
        self._cooldowns: Dict[str, float] = {}
        self._unknown_cooldowns: Dict[str, float] = {}

    def start(self, config: ScannerConfig) -> Dict[str, Any]:
        self.stop()
        with self._lock:
            self._config = config
            self._last_error = ""
            self._last_match = None
            self._human_count = 0
            self._human_scores = []

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="sentinex-scanner", daemon=True)
        self._thread.start()
        return self.status()

    def update_config(self, **changes: Any) -> Dict[str, Any]:
        with self._lock:
            new_config = replace(self._config, **changes)
        return self.start(new_config)

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
                    "savedSampleCount": self._config.saved_sample_count,
                    "sampleCount": self._config.sample_count,
                    "threshold": self._config.threshold,
                    "unknownCooldownSeconds": self._config.unknown_cooldown_seconds,
                },
                "error": self._last_error,
                "bestHumanConfidence": max(self._human_scores) if self._human_scores else 0,
                "humanCount": self._human_count,
                "humanScores": list(self._human_scores),
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
        detections: List[HumanDetection] = []
        detection_frame = frame
        detection_jpeg = jpeg
        if config.require_human:
            detections, detection_frame, detection_jpeg = self._collect_human_detections(frame, jpeg, config)
            if not detections:
                best_score = max(self._human_scores) if self._human_scores else 0
                with self._lock:
                    self._human_count = 0
                    self._last_error = (
                        f"Sin humano detectado "
                        f"(mejor confianza {best_score:.2f}, umbral {config.human_confidence_threshold:.2f})"
                    )
                    self._last_match = None
                    self._last_scan_at = time.time()
                return

        targets: List[Tuple[Optional[BBox], np.ndarray]] = []
        if detections:
            for detection in detections:
                crop = crop_bbox(detection_frame, detection.bbox)
                if crop.size:
                    targets.append((detection.bbox, crop))
        else:
            targets.append((None, frame))

        if not targets:
            with self._lock:
                self._last_error = "Humano detectado, pero el recorte quedo vacio"
                self._last_scan_at = time.time()
            return

        processed_targets = []
        for bbox, target in targets:
            samples = self._collect_samples(bbox, target)
            vectors = []
            for _sample_bbox, sample_target in samples:
                vector = self.embedder.embed_bgr(sample_target)
                if vector is not None:
                    vectors.append(vector)
            if not vectors:
                # Embedder de caras: sin cara detectable no hay identidad, no se
                # enrola ni matchea (evita patrones duplicados por pose).
                continue
            embedding = np.mean(np.stack(vectors, axis=0), axis=0)
            matches = self.store.search(embedding, limit=1)
            candidate = matches[0] if matches else {"identity": None, "score": 0.0}
            candidate["bbox"] = list(bbox) if bbox else None
            processed_targets.append({
                "candidate": candidate,
                "embedding": embedding,
                "samples": samples,
            })

        now = time.time()
        with self._lock:
            self._human_count = len(detections)
            self._last_scan_at = now
            if not processed_targets:
                self._last_error = "Humano detectado sin cara visible"
            else:
                self._last_error = ""

        if not processed_targets:
            return

        camera_id = self.stream.camera_id()
        best_match_for_ui = None
        frame_path = None

        for item in processed_targets:
            candidate = item["candidate"]
            best_embedding = item["embedding"]
            best_samples = item["samples"]

            matched_identity = candidate.get("identity")
            score = float(candidate["score"])
            reuse_threshold = min(config.threshold, config.reuse_threshold)
            is_known = bool(matched_identity) and score >= config.threshold
            is_reusable_pattern = bool(matched_identity) and score >= reuse_threshold

            detection_sample_paths: List[Any] = []

            if is_known or is_reusable_pattern:
                identity = matched_identity
                last_seen = self._cooldowns.get(identity["id"], 0)
                if now - last_seen < config.cooldown_seconds:
                    continue
                match_status = "known" if is_known else "known_low_confidence"
                detection_sample_paths = self._save_sample_frames(best_samples)
            elif config.auto_enroll_unknown:
                last_unknown_at = self._unknown_cooldowns.get(camera_id, 0)
                if now - last_unknown_at < config.unknown_cooldown_seconds:
                    continue
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
                detection_sample_paths = sample_paths
                self._unknown_cooldowns[camera_id] = now
            else:
                continue

            if frame_path is None:
                frame_path = self.store.save_detection_frame(detection_jpeg)

            detection = self.store.add_detection(
                identity=identity,
                score=score,
                frame_path=frame_path,
                bbox=candidate.get("bbox"),
                source=camera_id or "rtsp",
                match_status=match_status,
                sample_frames=detection_sample_paths,
            )
            self._cooldowns[identity["id"]] = now

            match_for_ui = {"identity": identity, "score": score, "detection": detection}
            if best_match_for_ui is None or score > best_match_for_ui["score"]:
                best_match_for_ui = match_for_ui

        if best_match_for_ui is not None:
            with self._lock:
                self._last_match = best_match_for_ui

    def _collect_human_detections(
        self,
        initial_frame: np.ndarray,
        initial_jpeg: bytes,
        config: ScannerConfig,
    ) -> Tuple[List[HumanDetection], np.ndarray, bytes]:
        attempts = max(1, min(5, config.sample_count))
        delay = max(0.0, min(1.0, config.sample_delay_seconds))
        probe_confidence = min(config.human_confidence_threshold, 0.05)
        best_detections: List[HumanDetection] = []
        best_frame = initial_frame
        best_jpeg = initial_jpeg
        seen_scores: List[float] = []

        for index in range(attempts):
            if index == 0:
                frame = initial_frame
                jpeg = initial_jpeg
            else:
                self._stop_event.wait(delay)
                if self._stop_event.is_set():
                    break
                frame = self.stream.latest_frame()
                jpeg = self.stream.latest_jpeg()
                if frame is None or jpeg is None:
                    continue

            candidates = self.detector.detect(frame, probe_confidence)
            seen_scores.extend(candidate.confidence for candidate in candidates)
            detections = [
                candidate
                for candidate in candidates
                if candidate.confidence >= config.human_confidence_threshold
            ]
            if detections and (
                not best_detections or detections[0].confidence > best_detections[0].confidence
            ):
                best_detections = detections
                best_frame = frame
                best_jpeg = jpeg

        with self._lock:
            self._human_scores = sorted(
                (round(score, 3) for score in seen_scores),
                reverse=True,
            )[:8]

        return best_detections[:4], best_frame, best_jpeg

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
                detections = self.detector.detect(frame, min(self._config.human_confidence_threshold, 0.05))
                detections = [
                    detection
                    for detection in detections
                    if detection.confidence >= self._config.human_confidence_threshold
                ]
                if detections:
                    sample_bbox = detections[0].bbox
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
        count = max(1, min(5, self._config.saved_sample_count))
        for _bbox, sample in samples[:count]:
            paths.append(self.store.save_auto_sample(encode_jpeg(sample)))
        return paths


def clamp_bbox(bbox: BBox, width: int, height: int) -> BBox:
    x, y, w, h = bbox
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def reasonable_human_bbox(bbox: BBox, frame_width: int, frame_height: int) -> bool:
    _x, _y, width, height = bbox
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
