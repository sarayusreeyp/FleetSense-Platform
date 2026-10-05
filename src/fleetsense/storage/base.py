from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


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
            Storage category: 'raw', 'processed', 'features', 'models', etc.
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
    ) -> pd.DataFrame:
        """
        Load a stored Parquet file directly into a pandas DataFrame.
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
    def exists(self, relative_path: str) -> bool:
        """Check whether a file exists."""
        pass

    @abstractmethod
    def load(self, relative_path: str) -> bytes:
        """Load a file as bytes."""
        pass