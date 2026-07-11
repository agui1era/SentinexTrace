from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import numpy as np


class FaceEmbedder:
    """Embedding de identidad basado en la cara (InsightFace / ArcFace).

    A diferencia del embedder CLIP de cuerpo entero, si el recorte no tiene una
    cara detectable devuelve None: el scanner interpreta eso como "sin identidad"
    y no crea ni matchea patron. Asi se evita la proliferacion de patrones por
    cambios de pose (misma persona de frente vs de espaldas)."""

    def __init__(self) -> None:
        self.mode = "face"
        self.model_name = os.getenv("SENTINEX_FACE_MODEL", "buffalo_l").strip() or "buffalo_l"
        self.det_threshold = _env_float("SENTINEX_FACE_DET_THRESHOLD", 0.5)
        self.det_size = int(_env_float("SENTINEX_FACE_DET_SIZE", 640))
        self._lock = threading.Lock()
        self._app: Optional[Any] = None
        self._device = "cpu"
        self._load_error = ""

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.dependencies_available(),
            "device": self._device,
            "error": self._load_error,
            "loaded": self._app is not None,
            "mode": self.mode,
            "model": self.model_name,
        }

    def dependencies_available(self) -> bool:
        try:
            import insightface  # noqa: F401
        except Exception as exc:  # pragma: no cover - reported through status.
            self._load_error = f"insightface faltante: {exc}"
            return False
        self._load_error = ""
        return True

    def embed_bgr(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return None

        self._ensure_loaded()
        if self._app is None:
            return None

        try:
            faces = self._app.get(frame_bgr)
        except Exception as exc:
            self._load_error = f"Error detectando cara: {exc}"
            return None

        faces = [face for face in faces if float(getattr(face, "det_score", 0.0)) >= self.det_threshold]
        if not faces:
            return None

        faces.sort(key=lambda face: float(getattr(face, "det_score", 0.0)), reverse=True)
        embedding = getattr(faces[0], "normed_embedding", None)
        if embedding is None:
            embedding = getattr(faces[0], "embedding", None)
        if embedding is None:
            return None
        return np.asarray(embedding, dtype=np.float32)

    def _ensure_loaded(self) -> None:
        if self._app is not None:
            return

        with self._lock:
            if self._app is not None:
                return
            try:
                from insightface.app import FaceAnalysis
            except Exception as exc:
                self._load_error = f"insightface faltante: {exc}"
                raise RuntimeError(self._load_error) from exc

            try:
                app = FaceAnalysis(name=self.model_name, providers=["CPUExecutionProvider"])
                app.prepare(ctx_id=-1, det_size=(self.det_size, self.det_size))
                self._app = app
                self._load_error = ""
            except Exception as exc:
                self._app = None
                self._load_error = f"No se pudo cargar InsightFace: {exc}"
                raise RuntimeError(self._load_error) from exc


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default
