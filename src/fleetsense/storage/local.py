from __future__ import annotations

import os
from pathlib import Path
import shutil
from typing import TYPE_CHECKING

import pandas as pd

from fleetsense.core.exceptions.storage import (
    StorageFileNotFoundError,
    StorageWriteError,
)
from fleetsense.storage.base import Storage


class LocalStorage(Storage):
    """
    Local filesystem storage backend with atomic operations and streaming support.
    """

    def __init__(self, root_dir: str | Path = "data") -> None:
        self.root = Path(root_dir).resolve()

        self.raw_dir = self.root / "raw"
        self.processed_dir = self.root / "processed"
        self.features_dir = self.root / "features"
        self.models_dir = self.root / "models"

        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.features_dir,
            self.models_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def _get_category_dir(self, category: str) -> Path:
        """Resolve and ensure category subdirectory exists."""
        safe_category = Path(category).name
        target_dir = self.root / safe_category
        target_dir.mkdir(parents=True, exist_ok=True)
        return target_dir

    def save_file(
        self,
        source_path: Path | str,
        category: str = "raw",
        filename: str | None = None,
    ) -> Path:
        """
        Copy a file atomically without reading its entire content into memory.
        """
        source = Path(source_path)
        if not source.exists():
            raise StorageFileNotFoundError(
                message=f"Source file to save does not exist: {source}"
            )

        target_name = Path(filename).name if filename else source.name
        target_dir = self._get_category_dir(category)
        final_path = target_dir / target_name
        tmp_path = target_dir / f"{target_name}.tmp"

        try:
            shutil.copy2(source, tmp_path)
            tmp_path.replace(final_path)
            return final_path
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise StorageWriteError(
                message=f"Failed to copy file to local storage: {final_path}",
                details={"source": str(source), "error": str(exc)},
            ) from exc

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        category: str,
        filename: str,
    ) -> Path:
        """
        Save a pandas DataFrame directly to Parquet format atomically.
        """
        target_name = Path(filename).name
        target_dir = self._get_category_dir(category)
        final_path = target_dir / target_name
        tmp_path = target_dir / f"{target_name}.tmp"

        try:
            dataframe.to_parquet(
                tmp_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            tmp_path.replace(final_path)
            return final_path
        except Exception as exc:
            if tmp_path.exists():
                tmp_path.unlink()
            raise StorageWriteError(
                message=f"Failed to save dataframe to {final_path}",
                details={"error": str(exc)},
            ) from exc

    def load_dataframe(
        self,
        category: str,
        filename: str,
    ) -> pd.DataFrame:
        """
        Load a stored Parquet file into a pandas DataFrame.
        """
        target_path = self._get_category_dir(category) / Path(filename).name
        if not target_path.exists():
            raise StorageFileNotFoundError(
                message=f"Parquet file not found in storage: {target_path}"
            )
        return pd.read_parquet(target_path)

    def download_to_file(
        self,
        category: str,
        filename: str,
        destination_path: Path | str,
    ) -> Path:
        """
        Copy a file from local storage to another destination path.
        """
        source_path = self._get_category_dir(category) / Path(filename).name
        if not source_path.exists():
            raise StorageFileNotFoundError(
                message=f"File not found in storage: {source_path}"
            )

        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination)
        return destination

    def list_files(self, category: str) -> list[str]:
        """
        List all filenames in a category directory.
        """
        category_dir = self._get_category_dir(category)
        return [
            entry.name
            for entry in category_dir.iterdir()
            if entry.is_file() and not entry.name.endswith(".tmp")
        ]

    def delete(self, category: str, filename: str) -> bool:
        """
        Delete a file from storage.
        """
        target_path = self._get_category_dir(category) / Path(filename).name
        if target_path.exists():
            target_path.unlink()
            return True
        return False

    # --- Backward compatibility methods ---

    def _save_bytes(self, directory: Path, filename: str, data: bytes) -> Path:
        safe_name = Path(filename).name
        path = directory / safe_name
        tmp_path = directory / f"{safe_name}.tmp"

        with open(tmp_path, "wb") as f:
            f.write(data)

        tmp_path.replace(path)
        return path

    def save_raw(self, filename: str, data: bytes) -> Path:
        return self._save_bytes(self.raw_dir, filename, data)

    def save_processed(self, filename: str, data: bytes) -> Path:
        return self._save_bytes(self.processed_dir, filename, data)

    def save_features(self, filename: str, data: bytes) -> Path:
        return self._save_bytes(self.features_dir, filename, data)

    def exists(self, relative_path: str) -> bool:
        target = self.root / relative_path
        return target.exists()

    def load(self, relative_path: str) -> bytes:
        target = self.root / relative_path
        if not target.exists():
            raise StorageFileNotFoundError(
                message=f"File not found: {target}"
            )
        with open(target, "rb") as f:
            return f.read()