from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import numpy as np


class ClipEmbedder:
    def __init__(self) -> None:
        self.mode = os.getenv("SENTINEX_EMBEDDER", os.getenv("VITE_EMBEDDER_MODE", "opencv")).strip().lower()
        self.model_name = os.getenv("SENTINEX_CLIP_MODEL", "openai/clip-vit-base-patch32")
        self._lock = threading.Lock()
        self._device = "cpu"
        self._model: Optional[Any] = None
        self._processor: Optional[Any] = None
        self._torch: Optional[Any] = None
        self._load_error = ""

    def status(self) -> Dict[str, Any]:
        return {
            "available": self.dependencies_available(),
            "device": self._device,
            "error": self._load_error,
            "loaded": self._model is not None,
            "mode": self.mode or "opencv",
            "model": self.model_name if self._uses_clip() else "opencv-local-512",
        }

    def dependencies_available(self) -> bool:
        if not self._uses_clip():
            try:
                import cv2  # noqa: F401
            except Exception as exc:
                self._load_error = f"OpenCV faltante para embedder local: {exc}"
                return False
            self._load_error = ""
            return True

        try:
            import PIL  # noqa: F401
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except Exception as exc:
            self._load_error = f"Dependencias CLIP faltantes: {exc}"
            return False
        self._load_error = ""
        return True

    def embed_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        if not self._uses_clip():
            return self._embed_opencv(frame_bgr)

        self._ensure_loaded()

        import cv2
        from PIL import Image

        if self._model is None or self._processor is None or self._torch is None:
            raise RuntimeError(self._load_error or "Modelo CLIP no disponible")

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        inputs = self._processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self._device) for key, value in inputs.items()}

        with self._torch.no_grad():
            outputs = self._model.get_image_features(**inputs)
            # transformers <5 devuelve un tensor; >=5 devuelve BaseModelOutputWithPooling
            # cuyo pooler_output ya es el image-embed proyectado (512 dims).
            if isinstance(outputs, self._torch.Tensor):
                features = outputs
            else:
                features = getattr(outputs, "image_embeds", None)
                if features is None:
                    features = outputs.pooler_output
            features = features / features.norm(dim=-1, keepdim=True)

        return features.detach().cpu().numpy()[0].astype(np.float32)

    def _uses_clip(self) -> bool:
        return self.mode in {"clip", "transformers", "huggingface"}

    def _embed_opencv(self, frame_bgr: np.ndarray) -> np.ndarray:
        import cv2

        if frame_bgr.size == 0:
            raise RuntimeError("Frame vacio para embedding local")

        resized = cv2.resize(frame_bgr, (16, 16), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY).astype(np.float32).reshape(-1) / 255.0

        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 4], [0, 180, 0, 256, 0, 256])
        hist = hist.astype(np.float32).reshape(-1)
        hist_sum = float(hist.sum())
        if hist_sum > 0:
            hist = hist / hist_sum

        vector = np.concatenate([gray, hist]).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0:
            vector = vector / norm
        return vector

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        with self._lock:
            if self._model is not None:
                return

            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor
            except Exception as exc:
                self._load_error = f"Dependencias CLIP faltantes: {exc}"
                raise RuntimeError(self._load_error) from exc

            try:
                if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    self._device = "mps"
                elif torch.cuda.is_available():
                    self._device = "cuda"
                else:
                    self._device = "cpu"

                self._processor = CLIPProcessor.from_pretrained(self.model_name)
                self._model = CLIPModel.from_pretrained(self.model_name)
                self._model.eval()
                self._model.to(self._device)
                self._torch = torch
                self._load_error = ""
            except Exception as exc:
                self._model = None
                self._processor = None
                self._torch = None
                self._load_error = f"No se pudo cargar CLIP: {exc}"
                raise RuntimeError(self._load_error) from exc
