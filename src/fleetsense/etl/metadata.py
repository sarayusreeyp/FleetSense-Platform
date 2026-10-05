from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fleetsense.core.constants.paths import METADATA_DIR

DEFAULT_METADATA_PATH = METADATA_DIR / "ingestion_state.json"


class IngestionMetadata:
    """Tracks successfully downloaded and ingested datasets."""

    def __init__(
        self,
        metadata_path: Path | str | None = None,
    ):
        self.path = (
            Path(metadata_path)
            if metadata_path is not None
            else DEFAULT_METADATA_PATH
        )
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def load(self) -> dict[str, Any]:
        """Load ingestion metadata."""
        if not self.path.exists():
            return {}

        if self.path.stat().st_size == 0:
            return {}

        try:
            with self.path.open("r", encoding="utf-8") as file:
                return json.load(file)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid ingestion metadata file: {self.path}"
            ) from exc

    def _save(self, data: dict[str, Any]) -> None:
        """Persist metadata atomically using a temporary file."""
        temporary_path = self.path.with_suffix(".tmp")

        with temporary_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2)

        temporary_path.replace(self.path)

    def is_downloaded(self, filename: str) -> bool:
        """Return True if a file was successfully downloaded."""
        data = self.load()
        for dataset in data.values():
            if filename in dataset.get("downloaded_files", []):
                return True
        return False

    def mark_downloaded(self, filename: str, dataset: str) -> None:
        """Record a successfully downloaded file."""
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

        self._save(data)

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