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
DEFAULT_PROCESSED_METADATA_PATH = METADATA_DIR / "processed_state.json"
DEFAULT_FEATURES_METADATA_PATH = METADATA_DIR / "features_state.json"

_BLOB_METADATA_CATEGORY = "metadata"
_BLOB_METADATA_FILENAME = "ingestion_state.json"
_BLOB_PROCESSED_METADATA_FILENAME = "processed_state.json"
_BLOB_FEATURES_METADATA_FILENAME = "features_state.json"


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
        blob_filename: str | None = None,
    ):
        self._blob_filename = blob_filename or _BLOB_METADATA_FILENAME
        metadata_dir = Path(os.getenv("FLEETSENSE_METADATA_DIR", METADATA_DIR))
        self._local_path = (
            Path(metadata_path)
            if metadata_path is not None
            else metadata_dir / self._blob_filename
        )
        self._local_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = FileLock(str(self._local_path) + ".lock", timeout=30)
        # If an explicit local path is given (e.g. in tests/tmp_path), stay on local backend
        if metadata_path is not None:
            self._backend = "local"
        else:
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
                f"{_BLOB_METADATA_CATEGORY}/{self._blob_filename}"
            )
            return json.loads(raw_bytes.decode("utf-8"))
        except Exception as exc:
            logger.info(
                "No existing state in Azure Blob (%s/%s: %s). Starting fresh.",
                _BLOB_METADATA_CATEGORY,
                self._blob_filename,
                exc,
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
                    filename=self._blob_filename,
                )
            elif hasattr(storage, "container_client"):
                import io
                buf = json.dumps(data, indent=2).encode("utf-8")
                blob_path = f"{_BLOB_METADATA_CATEGORY}/{self._blob_filename}"
                storage.container_client.get_blob_client(blob_path).upload_blob(
                    io.BytesIO(buf), overwrite=True
                )
            logger.debug("Persisted metadata to Azure Blob: %s", self._blob_filename)
        except Exception as exc:
            logger.warning(
                "Failed to save metadata to Azure Blob (%s): %s. "
                "Falling back to local file.",
                self._blob_filename,
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

    def get_latest_downloaded(self, dataset: str = "yellow") -> str | None:
        """Return the latest downloaded filename recorded for a dataset."""
        data = self.load()
        return data.get(dataset, {}).get("latest_downloaded")

    def get_downloaded_files(self, dataset: str = "yellow") -> list[str]:
        """Return list of all downloaded filenames for a dataset."""
        data = self.load()
        return list(data.get(dataset, {}).get("downloaded_files", []))

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

            if filename in dataset_state.get("downloaded_files", []):
                # Already recorded as downloaded; avoid rewriting or adding duplicates
                return

            dataset_state.setdefault("downloaded_files", []).append(filename)
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
            if files_meta.get(filename, {}).get("status") == "PROCESSED":
                # Already marked as processed; don't re-append duplicate metadata
                return

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


class ProcessedMetadata(IngestionMetadata):
    """Tracks state and validation reports for cleaned/processed datasets."""

    def __init__(self, metadata_path: Path | str | None = None):
        super().__init__(
            metadata_path=metadata_path,
            blob_filename=_BLOB_PROCESSED_METADATA_FILENAME,
        )

    def is_processed(self, filename: str, dataset: str = "yellow") -> bool:
        """Check whether a dataset has been successfully processed."""
        data = self.load()
        file_details = data.get(dataset, {}).get("file_details", {})
        if filename in file_details:
            return file_details[filename].get("status") == "PROCESSED"
        return filename in data.get(dataset, {}).get("processed_files", [])

    def mark_processed_file(
        self,
        filename: str,
        dataset: str = "yellow",
        report: dict[str, Any] | None = None,
        processed_path: str | None = None,
    ) -> None:
        """Record a successfully processed file atomically under lock."""
        with self._lock:
            data = self.load()
            dataset_state = data.setdefault(
                dataset,
                {
                    "latest_processed": None,
                    "processed_files": [],
                    "last_run": None,
                    "file_details": {},
                },
            )
            files_meta = dataset_state.setdefault("file_details", {})
            if files_meta.get(filename, {}).get("status") == "PROCESSED":
                return

            processed_list = dataset_state.setdefault("processed_files", [])
            if filename not in processed_list:
                processed_list.append(filename)

            dataset_state["latest_processed"] = filename
            dataset_state["last_run"] = datetime.now(timezone.utc).isoformat()
            files_meta[filename] = {
                "status": "PROCESSED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": processed_path,
                "report": report or {},
            }
            self._save(data)


class FeaturesMetadata(IngestionMetadata):
    """Tracks state and lineage for generated ML feature matrices."""

    def __init__(self, metadata_path: Path | str | None = None):
        super().__init__(
            metadata_path=metadata_path,
            blob_filename=_BLOB_FEATURES_METADATA_FILENAME,
        )

    def is_feature_generated(self, filename: str, dataset: str = "yellow") -> bool:
        """Check whether features have been generated for a dataset file."""
        data = self.load()
        file_details = data.get(dataset, {}).get("file_details", {})
        if filename in file_details:
            return file_details[filename].get("status") == "FEATURES_GENERATED"
        return filename in data.get(dataset, {}).get("feature_files", [])

    def mark_feature_generated(
        self,
        filename: str,
        dataset: str = "yellow",
        row_count: int | None = None,
        feature_columns: list[str] | None = None,
        features_path: str | None = None,
    ) -> None:
        """Record feature generation completion atomically under lock."""
        with self._lock:
            data = self.load()
            dataset_state = data.setdefault(
                dataset,
                {
                    "latest_features": None,
                    "feature_files": [],
                    "last_run": None,
                    "file_details": {},
                },
            )
            files_meta = dataset_state.setdefault("file_details", {})
            if files_meta.get(filename, {}).get("status") == "FEATURES_GENERATED":
                return

            feat_list = dataset_state.setdefault("feature_files", [])
            if filename not in feat_list:
                feat_list.append(filename)

            dataset_state["latest_features"] = filename
            dataset_state["last_run"] = datetime.now(timezone.utc).isoformat()
            files_meta[filename] = {
                "status": "FEATURES_GENERATED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "path": features_path,
                "row_count": row_count,
                "feature_count": len(feature_columns) if feature_columns else None,
            }
            self._save(data)
