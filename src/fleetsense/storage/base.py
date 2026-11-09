from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    import pandas as pd

CANONICAL_STORAGE_CATEGORIES: tuple[str, ...] = (
    "raw",
    "processed",
    "features",
    "models",
    "metadata",
    "tmp",
)


class Storage(ABC):
    """
    Abstract base class for FleetSense storage backends.

    Supports both file-based streaming operations (memory-safe for large Parquet files)
    and in-memory byte buffers (for lightweight artifacts).
    """

    @abstractmethod
    def save_file(
        self,
        source_path: Path | str,
        category: str = "raw",
        filename: str | None = None,
    ) -> Path | str:
        """
        Save a file from disk into storage without loading the entire content into RAM.

        Parameters
        ----------
        source_path : Path | str
            Local path to the source file.
        category : str
            Storage category: 'raw', 'processed', 'features', 'models', 'metadata', 'tmp'.
        filename : str | None
            Optional target filename. Defaults to source_path.name.

        Returns
        -------
        Path | str
            Path or URI to the stored object.
        """
        pass

    @abstractmethod
    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        category: str,
        filename: str,
    ) -> Path | str:
        """
        Save a pandas DataFrame directly to Parquet format in storage.
        """
        pass

    @abstractmethod
    def load_dataframe(
        self,
        category: str,
        filename: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Load a stored Parquet file directly into a pandas DataFrame.
        When columns is specified, only the projected columns are read from disk/storage.
        """
        pass

    @abstractmethod
    def download_to_file(
        self,
        category: str,
        filename: str,
        destination_path: Path | str,
    ) -> Path:
        """
        Download/copy an object from storage to a local disk destination.
        """
        pass

    @abstractmethod
    def list_files(self, category: str) -> list[str]:
        """
        List all filenames available in a specific category.
        """
        pass

    @abstractmethod
    def delete(self, category: str, filename: str) -> bool:
        """
        Delete a file from storage.
        """
        pass

    @abstractmethod
    def clean_temp_files(self, category: str) -> int:
        """
        Clean up orphaned temporary (.tmp) files in a given storage category.

        Returns
        -------
        int
            Number of temporary files removed.
        """
        pass

    # --- Backward compatibility methods ---

    @abstractmethod
    def save_raw(self, filename: str, data: bytes) -> Path | str:
        """Save raw dataset bytes."""
        pass

    @abstractmethod
    def save_processed(self, filename: str, data: bytes) -> Path | str:
        """Save processed dataset bytes."""
        pass

    @abstractmethod
    def save_features(self, filename: str, data: bytes) -> Path | str:
        """Save feature dataset bytes."""
        pass

    @abstractmethod
    def exists(self, relative_path: str, filename: str | None = None) -> bool:
        """Check whether a file exists.

        Parameters
        ----------
        relative_path : str
            Relative path or storage category (e.g. 'models/latest.joblib' or 'models').
        filename : str | None
            Optional filename when relative_path is a category name.
        """
        pass

    @abstractmethod
    def load(self, relative_path: str) -> bytes:
        """Load a file as bytes."""
        pass
