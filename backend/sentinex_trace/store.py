from __future__ import annotations

import json
import os
import shutil
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - dotenv is optional at runtime.
    load_dotenv = None  # type: ignore[assignment]

if load_dotenv is not None:
    load_dotenv()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorStore:
    def __init__(self, data_dir: Optional[str] = None) -> None:
        self.data_dir = Path(data_dir or os.getenv("SENTINEX_DATA_DIR", "data/vision")).resolve()
        self.db_path = self.data_dir / "vector_db.json"
        self.chroma_dir = Path(os.getenv("SENTINEX_CHROMA_DIR", str(self.data_dir / "chroma_db"))).resolve()
        self.chroma_collection_name = os.getenv("SENTINEX_CHROMA_COLLECTION", "sentinex_face_identities")
        self.mongo_uri = os.getenv("SENTINEX_MONGO_URI") or os.getenv("MONGO_URI", "")
        self.mongo_db_name = os.getenv("SENTINEX_MONGO_DB") or os.getenv("MONGO_DB_NAME", "sentinex_face")
        self.mongo_detection_collection = os.getenv(
            "SENTINEX_MONGO_DETECTION_COLLECTION",
            os.getenv("MONGO_COLL_SENTINEX", "sentinex_face_rtsp_detections"),
        )
        self.enroll_dir = self.data_dir / "enrollments"
        self.pending_dir = self.data_dir / "pending"
        self.detection_dir = self.data_dir / "detections"
        self._lock = threading.RLock()
        self._db: Dict[str, Any] = {"identities": [], "detections": []}
        self._chroma: Optional[Any] = None
        self._mongo_client: Optional[Any] = None
        self._mongo_detections: Optional[Any] = None
        self._chroma_error = ""
        self._mongo_error = ""

        for path in (self.data_dir, self.enroll_dir, self.pending_dir, self.detection_dir):
            path.mkdir(parents=True, exist_ok=True)

        self._load()
        self._connect_chroma()
        self._connect_mongo()

    def _load(self) -> None:
        if not self.db_path.exists():
            self._save()
            return

        try:
            with self.db_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except json.JSONDecodeError:
            backup = self.db_path.with_suffix(".corrupt.json")
            shutil.copy2(self.db_path, backup)
            payload = {}

        self._db = {
            "identities": list(payload.get("identities", [])),
            "detections": list(payload.get("detections", [])),
        }

    def _save(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self.db_path.with_suffix(".tmp")
        with temp_path.open("w", encoding="utf-8") as handle:
            json.dump(self._db, handle, ensure_ascii=False, indent=2)
        temp_path.replace(self.db_path)

    def status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "chroma": {
                    "available": self._chroma is not None,
                    "collection": self.chroma_collection_name,
                    "error": self._chroma_error,
                    "path": str(self.chroma_dir),
                },
                "dataDir": str(self.data_dir),
                "detections": len(self._db["detections"]),
                "identities": len(self._db["identities"]),
                "mongo": {
                    "available": self._mongo_detections is not None,
                    "collection": self.mongo_detection_collection,
                    "database": self.mongo_db_name,
                    "error": self._mongo_error,
                },
            }

    def list_identities(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [self._public_identity(identity) for identity in self._db["identities"]]

    def add_identity(
        self,
        *,
        name: str,
        tag: str,
        embedding: np.ndarray,
        sample_paths: List[Path],
        generic: bool = False,
        camera_id: str = "",
        pattern_id: str = "",
    ) -> Dict[str, Any]:
        identity_id = self._next_identity_id(pattern_id or name) if generic else uuid4().hex
        target_dir = self.enroll_dir / identity_id
        target_dir.mkdir(parents=True, exist_ok=True)
        stored_samples: List[str] = []

        for index, source in enumerate(sample_paths, start=1):
            target = target_dir / f"sample-{index}.jpg"
            shutil.copy2(source, target)
            stored_samples.append(self.media_key(target))

        vector = normalize_vector(embedding).astype(float).tolist()
        now = utc_now()
        identity = {
            "id": identity_id,
            "name": name,
            "tag": tag,
            "cameraId": camera_id,
            "embedding": vector,
            "generic": generic,
            "patternId": pattern_id or name,
            "samples": len(stored_samples),
            "thumbnails": stored_samples,
            "createdAt": now,
            "updatedAt": now,
        }

        with self._lock:
            self._db["identities"].insert(0, identity)
            self._save()
            self._upsert_chroma_identity(identity)

        return self._public_identity(identity)

    def _next_identity_id(self, candidate: str) -> str:
        base = normalize_identifier(candidate) or uuid4().hex
        identity_id = base
        suffix = 2
        with self._lock:
            existing = {item.get("id", "") for item in self._db["identities"]}
            while identity_id in existing:
                identity_id = f"{base}-{suffix}"
                suffix += 1
        return identity_id

    def rename_identity(self, identity_id: str, *, name: str, tag: Optional[str] = None) -> Optional[Dict[str, Any]]:
        now = utc_now()
        updated: Optional[Dict[str, Any]] = None
        with self._lock:
            for identity in self._db["identities"]:
                if identity["id"] != identity_id:
                    continue
                identity["name"] = name
                if tag is not None:
                    identity["tag"] = tag
                identity["generic"] = False
                identity["updatedAt"] = now
                updated = identity
                break
            if updated is None:
                return None
            for detection in self._db["detections"]:
                if detection.get("identityId") != identity_id:
                    continue
                detection["name"] = name
                if tag is not None:
                    detection["tag"] = tag
                detection["generic"] = False
            self._save()
            self._upsert_chroma_identity(updated)

        return self._public_identity(updated)

    def delete_identity(self, identity_id: str) -> bool:
        with self._lock:
            before = len(self._db["identities"])
            self._db["identities"] = [item for item in self._db["identities"] if item["id"] != identity_id]
            deleted = len(self._db["identities"]) != before
            if deleted:
                self._save()
                self._delete_chroma_identity(identity_id)

        if deleted:
            shutil.rmtree(self.enroll_dir / identity_id, ignore_errors=True)

        return deleted

    def search(self, embedding: np.ndarray, limit: int = 5) -> List[Dict[str, Any]]:
        probe = normalize_vector(embedding)
        chroma_matches = self._search_chroma(probe, limit)
        if chroma_matches:
            return chroma_matches

        matches: List[Dict[str, Any]] = []

        with self._lock:
            identities = list(self._db["identities"])

        for identity in identities:
            stored = np.asarray(identity.get("embedding", []), dtype=np.float32)
            if stored.size == 0 or stored.shape != probe.shape:
                continue
            score = float(np.dot(probe, normalize_vector(stored)))
            matches.append({"identity": self._public_identity(identity), "score": score})

        matches.sort(key=lambda item: item["score"], reverse=True)
        return matches[:limit]

    def add_detection(
        self,
        *,
        identity: Dict[str, Any],
        score: float,
        frame_path: Path,
        bbox: Optional[List[int]],
        source: str,
        match_status: str = "known",
        sample_frames: Optional[List[Path]] = None,
    ) -> Dict[str, Any]:
        detection = {
            "id": uuid4().hex,
            "identityId": identity["id"],
            "name": identity["name"],
            "tag": identity.get("tag", ""),
            "cameraId": identity.get("cameraId", source if source.startswith("CAM") else ""),
            "generic": bool(identity.get("generic", False)),
            "matchStatus": match_status,
            "patternId": identity.get("patternId", identity.get("name", "")),
            "score": score,
            "timestamp": utc_now(),
            "frame": self.media_key(frame_path),
            "sampleFrames": [self.media_key(path) for path in sample_frames or []],
            "bbox": bbox,
            "source": source,
        }

        with self._lock:
            self._db["detections"].insert(0, detection)
            self._db["detections"] = self._db["detections"][:5000]
            self._save()
            self._insert_mongo_detection(detection)

        return detection

    def list_detections(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._db["detections"][: max(1, min(limit, 500))])

    def delete_detection(self, detection_id: str) -> bool:
        with self._lock:
            before = len(self._db["detections"])
            self._db["detections"] = [
                item for item in self._db["detections"] if item.get("id") != detection_id
            ]
            deleted = len(self._db["detections"]) != before
            if deleted:
                self._save()
                self._delete_mongo_detection(detection_id)
        return deleted

    def save_pending_capture(self, jpeg: bytes) -> Dict[str, Any]:
        capture_id = uuid4().hex
        path = self.pending_dir / f"{capture_id}.jpg"
        path.write_bytes(jpeg)
        return {
            "id": capture_id,
            "path": str(path),
            "url": f"/api/vision/media/{self.media_key(path)}",
        }

    def pending_capture_path(self, capture_id: str) -> Path:
        safe_id = "".join(char for char in capture_id if char.isalnum() or char in ("-", "_"))
        return self.pending_dir / f"{safe_id}.jpg"

    def clear_pending(self) -> None:
        shutil.rmtree(self.pending_dir, ignore_errors=True)
        self.pending_dir.mkdir(parents=True, exist_ok=True)

    def save_detection_frame(self, jpeg: bytes) -> Path:
        date_dir = self.detection_dir / datetime.now().strftime("%Y-%m-%d")
        date_dir.mkdir(parents=True, exist_ok=True)
        path = date_dir / f"{uuid4().hex}.jpg"
        path.write_bytes(jpeg)
        return path

    def save_auto_sample(self, jpeg: bytes) -> Path:
        sample_dir = self.pending_dir / "auto"
        sample_dir.mkdir(parents=True, exist_ok=True)
        path = sample_dir / f"{uuid4().hex}.jpg"
        path.write_bytes(jpeg)
        return path

    def next_unknown_name(self, camera_id: str = "") -> str:
        prefix = f"Patron {normalize_camera_id(camera_id)}_"
        with self._lock:
            count = 1
            while True:
                candidate = f"{prefix}{count}"
                exists = any(item.get("name", "") == candidate for item in self._db["identities"])
                if not exists:
                    return candidate
                count += 1

    def resolve_media_path(self, media_key: str) -> Path:
        candidate = (self.data_dir / media_key).resolve()
        try:
            candidate.relative_to(self.data_dir)
        except ValueError as exc:
            raise ValueError("Ruta de media invalida") from exc
        if not candidate.exists():
            raise FileNotFoundError(media_key)
        return candidate

    def media_key(self, path: Path) -> str:
        return str(path.resolve().relative_to(self.data_dir)).replace("\\", "/")

    def _public_identity(self, identity: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": identity["id"],
            "name": identity["name"],
            "tag": identity.get("tag", ""),
            "cameraId": identity.get("cameraId", ""),
            "generic": bool(identity.get("generic", False)),
            "patternId": identity.get("patternId", identity.get("name", "")),
            "samples": identity.get("samples", 0),
            "thumbnails": [f"/api/vision/media/{item}" for item in identity.get("thumbnails", [])],
            "dimensions": len(identity.get("embedding", [])),
            "createdAt": identity.get("createdAt", ""),
            "updatedAt": identity.get("updatedAt", ""),
        }

    def _connect_chroma(self) -> None:
        os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
        try:
            import chromadb
        except Exception as exc:
            self._chroma_error = f"Chroma no instalado: {exc}"
            return

        try:
            self.chroma_dir.mkdir(parents=True, exist_ok=True)
            from chromadb.config import Settings

            client = chromadb.PersistentClient(
                path=str(self.chroma_dir),
                settings=Settings(anonymized_telemetry=False),
            )
            self._chroma = client.get_or_create_collection(
                name=self.chroma_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._chroma_error = ""
            with self._lock:
                for identity in self._db["identities"]:
                    self._upsert_chroma_identity(identity)
        except Exception as exc:
            self._chroma = None
            self._chroma_error = f"No se pudo abrir Chroma: {exc}"

    def _connect_mongo(self) -> None:
        if not self.mongo_uri:
            self._mongo_error = "SENTINEX_MONGO_URI no configurado"
            return

        try:
            from pymongo import MongoClient
        except Exception as exc:
            self._mongo_error = f"pymongo no instalado: {exc}"
            return

        try:
            self._mongo_client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=1800)
            self._mongo_client.admin.command("ping")
            self._mongo_detections = self._mongo_client[self.mongo_db_name][self.mongo_detection_collection]
            self._mongo_error = ""
        except Exception as exc:
            self._mongo_client = None
            self._mongo_detections = None
            self._mongo_error = f"No se pudo conectar Mongo: {exc}"

    def _upsert_chroma_identity(self, identity: Dict[str, Any]) -> None:
        if self._chroma is None:
            return
        embedding = identity.get("embedding", [])
        if not embedding:
            return

        try:
            metadata = {
                "name": identity.get("name", ""),
                "tag": identity.get("tag", ""),
                "cameraId": identity.get("cameraId", ""),
                "generic": bool(identity.get("generic", False)),
                "patternId": identity.get("patternId", identity.get("name", "")),
                "samples": int(identity.get("samples", 0)),
                "createdAt": identity.get("createdAt", ""),
                "updatedAt": identity.get("updatedAt", ""),
            }
            self._chroma.upsert(
                ids=[identity["id"]],
                embeddings=[embedding],
                metadatas=[metadata],
                documents=[identity.get("name", identity["id"])],
            )
            self._chroma_error = ""
        except Exception as exc:
            self._chroma_error = f"No se pudo escribir en Chroma: {exc}"

    def _delete_chroma_identity(self, identity_id: str) -> None:
        if self._chroma is None:
            return
        try:
            self._chroma.delete(ids=[identity_id])
            self._chroma_error = ""
        except Exception as exc:
            self._chroma_error = f"No se pudo borrar en Chroma: {exc}"

    def _search_chroma(self, embedding: np.ndarray, limit: int) -> List[Dict[str, Any]]:
        if self._chroma is None:
            return []

        try:
            result = self._chroma.query(
                query_embeddings=[embedding.astype(float).tolist()],
                n_results=max(1, limit),
            )
            ids = result.get("ids", [[]])[0]
            distances = result.get("distances", [[]])[0]
        except Exception as exc:
            self._chroma_error = f"No se pudo buscar en Chroma: {exc}"
            return []

        by_id: Dict[str, Dict[str, Any]]
        with self._lock:
            by_id = {identity["id"]: identity for identity in self._db["identities"]}

        matches: List[Dict[str, Any]] = []
        for identity_id, distance in zip(ids, distances):
            identity = by_id.get(identity_id)
            if identity is None:
                continue
            score = distance_to_score(float(distance))
            matches.append({"identity": self._public_identity(identity), "score": score})

        matches.sort(key=lambda item: item["score"], reverse=True)
        self._chroma_error = ""
        return matches[:limit]

    def _insert_mongo_detection(self, detection: Dict[str, Any]) -> None:
        if self._mongo_detections is None:
            return
        try:
            self._mongo_detections.insert_one({**detection, "system": "sentinex-face-rtsp"})
            self._mongo_error = ""
        except Exception as exc:
            self._mongo_error = f"No se pudo registrar deteccion en Mongo: {exc}"

    def _delete_mongo_detection(self, detection_id: str) -> None:
        if self._mongo_detections is None:
            return
        try:
            self._mongo_detections.delete_many({"id": detection_id, "system": "sentinex-face-rtsp"})
            self._mongo_error = ""
        except Exception as exc:
            self._mongo_error = f"No se pudo borrar deteccion en Mongo: {exc}"


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0:
        return array
    return array / norm


def distance_to_score(distance: float) -> float:
    if distance <= 0:
        return 1.0
    if distance <= 2:
        return max(-1.0, min(1.0, 1 - distance))
    return 1 / (1 + distance)


def normalize_camera_id(camera_id: str) -> str:
    cleaned = "".join(char for char in camera_id.upper() if char.isalnum())
    if not cleaned:
        return "CAM"
    if cleaned.startswith("CAM") or cleaned.startswith("WEBCAM"):
        return cleaned
    return f"CAM{cleaned}"


def normalize_identifier(value: str) -> str:
    cleaned = []
    last_dash = False
    for char in value.strip().lower():
        if char.isalnum():
            cleaned.append(char)
            last_dash = False
        elif not last_dash:
            cleaned.append("-")
            last_dash = True
    return "".join(cleaned).strip("-")
