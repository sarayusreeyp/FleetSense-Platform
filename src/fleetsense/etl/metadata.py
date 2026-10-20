"""
fleetsense.etl.metadata
=======================

Gap 10 fix: IngestionMetadata now persists state in Azure Blob Storage
when the storage backend is ``azure``.  Local-file fallback is kept for
development.  Containers are stateless — on restart the metadata is
re-read from blob storage so we never re-download already-processed files.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from filelock import FileLock

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import METADATA_DIR
from fleetsense.core.logger import get_logger

logger = get_logger(__name__)

DEFAULT_METADATA_PATH = METADATA_DIR / "ingestion_state.json"
_BLOB_METADATA_CATEGORY = "metadata"
_BLOB_METADATA_FILENAME = "ingestion_state.json"


class IngestionMetadata:
    """Tracks successfully downloaded and ingested datasets.

    Storage strategy:
      - ``azure`` backend: reads/writes ``metadata/ingestion_state.json``
        inside the configured Azure Blob container.
      - ``local`` backend (default): reads/writes from the local
        ``METADATA_DIR`` path (unchanged behaviour).

    This ensures the ETL state survives container restarts in production.
    """

    def __init__(
        self,
        metadata_path: Path | str | None = None,
    ):
        self._local_path = (
            Path(metadata_path)
            if metadata_path is not None
            else DEFAULT_METADATA_PATH
        )
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self._local_path) + ".lock", timeout=30)
        self._backend = (
            os.getenv("FLEETSENSE_STORAGE_BACKEND")
            or os.getenv("STORAGE_BACKEND")
            or getattr(getattr(settings, "storage", None), "backend", "local")
        ).lower()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_azure_storage(self):
        """Return AzureBlobStorage instance (lazy import to avoid circular deps)."""
        from fleetsense.storage.factory import get_storage
        return get_storage(backend="azure")

    def _load_from_azure(self) -> dict[str, Any]:
        """Download ingestion state JSON from Azure Blob Storage."""
        try:
            storage = self._get_azure_storage()
            raw_bytes = storage.load(
                f"{_BLOB_METADATA_CATEGORY}/{_BLOB_METADATA_FILENAME}"
            )
            return json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            logger.info(
                "No existing ingestion state in Azure Blob (%s). Starting fresh.", exc
            )
            return {}

    def _save_to_azure(self, data: dict[str, Any]) -> None:
        """Upload ingestion state JSON to Azure Blob Storage."""
        temporary_path = self._local_path.with_suffix(".tmp")
        try:
            storage = self._get_azure_storage()
            with temporary_path.open("w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)

            if hasattr(storage, "save_file"):
                storage.save_file(
                    source_path=temporary_path,
                    category=_BLOB_METADATA_CATEGORY,
                    filename=_BLOB_METADATA_FILENAME,
                )
            elif hasattr(storage, "container_client"):
                import io
                buf = json.dumps(data, indent=2).encode("utf-8")
                blob_path = f"{_BLOB_METADATA_CATEGORY}/{_BLOB_METADATA_FILENAME}"
                storage.container_client.get_blob_client(blob_path).upload_blob(
                    io.BytesIO(buf), overwrite=True
                )
            logger.debug("Persisted ingestion metadata to Azure Blob: %s", _BLOB_METADATA_FILENAME)
        except Exception as exc:
            logger.warning(
                "Failed to save ingestion metadata to Azure Blob: %s. "
                "Falling back to local file.",
                exc,
            )
            self._save_local(data)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def _load_local(self) -> dict[str, Any]:
        """Load ingestion metadata from local filesystem."""
        if not self._local_path.exists():
            return {}
        if self._local_path.stat().st_size == 0:
            return {}
        try:
            with self._local_path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid ingestion metadata file: {self._local_path}"
            ) from exc

    def _save_local(self, data: dict[str, Any]) -> None:
        """Persist metadata atomically using a temporary file."""
        temporary_path = self._local_path.with_suffix(".tmp")
        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)
        temporary_path.replace(self._local_path)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def load(self) -> dict[str, Any]:
        """Load ingestion metadata from the appropriate backend."""
        if self._backend == "azure":
            return self._load_from_azure()
        return self._load_local()

    def _save(self, data: dict[str, Any]) -> None:
        """Persist metadata to the appropriate backend."""
        if self._backend == "azure":
            self._save_to_azure(data)
        else:
            self._save_local(data)

    def is_downloaded(self, filename: str) -> bool:
        """Return True if a file was successfully downloaded."""
        data = self.load()
        for dataset in data.values():
            if filename in dataset.get("downloaded_files", []):
                return True
        return False

    def mark_downloaded(self, filename: str, dataset: str) -> None:
        """Record a successfully downloaded file atomically under lock."""
        with self._lock:
            data = self.load()
            dataset_state = data.setdefault(
                dataset,
                {
                    "latest_downloaded": None,
                    "downloaded_files": [],
                    "last_run": None,
                },
            )

            if filename not in dataset_state["downloaded_files"]:
                dataset_state["downloaded_files"].append(filename)

            dataset_state["latest_downloaded"] = filename
            dataset_state["last_run"] = datetime.now(timezone.utc).isoformat()
            files_meta = dataset_state.setdefault("file_details", {})
            files_meta[filename] = {
                "status": "DOWNLOADED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            self._save(data)

    def mark_processed(self, filename: str, dataset: str, report: dict[str, Any] | None = None) -> None:
        """Record that a downloaded dataset has been cleaned and processed."""
        with self._lock:
            data = self.load()
            dataset_state = data.setdefault(
                dataset,
                {
                    "latest_downloaded": None,
                    "downloaded_files": [],
                    "last_run": None,
                },
            )
            files_meta = dataset_state.setdefault("file_details", {})
            files_meta[filename] = {
                "status": "PROCESSED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "report": report or {},
            }
            self._save(data)

    def mark_failed(self, filename: str, dataset: str, error: str) -> None:
        """Record a failure during processing."""
        with self._lock:
            data = self.load()
            dataset_state = data.setdefault(
                dataset,
                {
                    "latest_downloaded": None,
                    "downloaded_files": [],
                    "last_run": None,
                },
            )
            files_meta = dataset_state.setdefault("file_details", {})
            files_meta[filename] = {
                "status": "FAILED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": error,
            }
            self._save(data)

    def get_file_status(self, filename: str, dataset: str = "yellow") -> str | None:
        """Return the current status of a specific file ('DOWNLOADED', 'PROCESSED', 'FAILED', or None)."""
        data = self.load()
        details = data.get(dataset, {}).get("file_details", {})
        if filename in details:
            return details[filename].get("status")
        if filename in data.get(dataset, {}).get("downloaded_files", []):
            return "DOWNLOADED"
        return None

    def get_state(self, dataset: str = "yellow") -> dict[str, Any]:
        """Return ingestion state for a dataset."""
        data = self.load()
        return data.get(
            dataset,
            {
                "latest_downloaded": None,
                "downloaded_files": [],
                "last_run": None,
            },
        )
