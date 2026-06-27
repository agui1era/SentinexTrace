from __future__ import annotations

import os
import threading
from typing import Any, Dict, Optional

import numpy as np


class ClipEmbedder:
    def __init__(self) -> None:
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
            "model": self.model_name,
        }

    def dependencies_available(self) -> bool:
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
            features = self._model.get_image_features(**inputs)
            features = features / features.norm(dim=-1, keepdim=True)

        return features.detach().cpu().numpy()[0].astype(np.float32)

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
