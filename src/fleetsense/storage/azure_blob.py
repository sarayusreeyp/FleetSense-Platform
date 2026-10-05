from __future__ import annotations

import io
import os
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

try:
    from azure.core.exceptions import (
        HttpResponseError,
        ResourceExistsError,
        ResourceNotFoundError,
    )
    from azure.storage.blob import BlobServiceClient
    AZURE_AVAILABLE = True
except ImportError:
    AZURE_AVAILABLE = False

from fleetsense.core.exceptions.storage import (
    StorageConnectionError,
    StorageFileNotFoundError,
    StorageWriteError,
)
from fleetsense.storage.base import Storage


class AzureBlobStorage(Storage):
    """
    Azure Blob Storage backend for production FleetSense workloads.
    Supports streaming large datasets directly to and from Azure Blob containers.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        container_name: str = "fleetsense-data",
    ) -> None:
        if not AZURE_AVAILABLE:
            raise StorageConnectionError(
                message="azure-storage-blob package is not installed."
            )

        self.connection_string = (
            connection_string
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            or ""
        )
        self.container_name = (
            container_name
            or os.getenv("AZURE_CONTAINER_NAME")
            or "fleetsense-data"
        )

        if not self.connection_string:
            raise StorageConnectionError(
                message=(
                    "Azure Storage connection string is missing. "
                    "Set AZURE_STORAGE_CONNECTION_STRING in your environment or config."
                )
            )

        try:
            self.service_client = BlobServiceClient.from_connection_string(
                self.connection_string
            )
            self.container_client = self.service_client.get_container_client(
                self.container_name
            )
            # Create container if it does not already exist
            try:
                self.container_client.create_container()
            except ResourceExistsError:
                pass
        except Exception as exc:
            raise StorageConnectionError(
                message=f"Failed to connect to Azure Blob container '{self.container_name}'",
                details={"error": str(exc)},
            ) from exc

    def _blob_name(self, category: str, filename: str) -> str:
        safe_filename = Path(filename).name
        safe_category = Path(category).name
        return f"{safe_category}/{safe_filename}"

    def save_file(
        self,
        source_path: Path | str,
        category: str = "raw",
        filename: str | None = None,
    ) -> str:
        """
        Stream a file from local disk to Azure Blob Storage without full memory buffering.
        """
        source = Path(source_path)
        if not source.exists():
            raise StorageFileNotFoundError(
                message=f"Source file does not exist: {source}"
            )

        target_name = Path(filename).name if filename else source.name
        blob_path = self._blob_name(category, target_name)
        blob_client = self.container_client.get_blob_client(blob_path)

        try:
            with source.open("rb") as stream:
                blob_client.upload_blob(stream, overwrite=True)
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to upload {source} to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def save_dataframe(
        self,
        dataframe: pd.DataFrame,
        category: str,
        filename: str,
    ) -> str:
        """
        Write a DataFrame to Parquet and upload it to Azure Blob Storage.
        """
        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)

        buffer = io.BytesIO()
        try:
            dataframe.to_parquet(
                buffer,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            buffer.seek(0)
            blob_client.upload_blob(buffer, overwrite=True)
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to save DataFrame to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def load_dataframe(
        self,
        category: str,
        filename: str,
    ) -> pd.DataFrame:
        """
        Download Parquet data from Azure Blob Storage into a DataFrame.
        """
        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)

        try:
            download_stream = blob_client.download_blob()
            buffer = io.BytesIO(download_stream.readall())
            return pd.read_parquet(buffer)
        except ResourceNotFoundError as exc:
            raise StorageFileNotFoundError(
                message=f"Azure Blob not found: {blob_path}"
            ) from exc
        except Exception as exc:
            raise StorageConnectionError(
                message=f"Failed to download DataFrame from Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def download_to_file(
        self,
        category: str,
        filename: str,
        destination_path: Path | str,
    ) -> Path:
        """
        Stream an Azure Blob directly to a local file.
        """
        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)

        destination = Path(destination_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp_destination = destination.with_suffix(destination.suffix + ".tmp")

        try:
            with tmp_destination.open("wb") as file:
                download_stream = blob_client.download_blob()
                download_stream.readinto(file)
            tmp_destination.replace(destination)
            return destination
        except ResourceNotFoundError as exc:
            if tmp_destination.exists():
                tmp_destination.unlink()
            raise StorageFileNotFoundError(
                message=f"Azure Blob not found: {blob_path}"
            ) from exc
        except Exception as exc:
            if tmp_destination.exists():
                tmp_destination.unlink()
            raise StorageConnectionError(
                message=f"Failed to download blob {blob_path} to {destination}",
                details={"error": str(exc)},
            ) from exc

    def list_files(self, category: str) -> list[str]:
        """
        List all filenames within a specific category prefix.
        """
        prefix = f"{category}/"
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            filenames = []
            for b in blobs:
                name = b.name[len(prefix) :]
                if name and "/" not in name:
                    filenames.append(name)
            return filenames
        except Exception as exc:
            raise StorageConnectionError(
                message=f"Failed to list blobs with prefix {prefix}",
                details={"error": str(exc)},
            ) from exc

    def delete(self, category: str, filename: str) -> bool:
        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)
        try:
            blob_client.delete_blob()
            return True
        except ResourceNotFoundError:
            return False
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to delete blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    # --- Backward compatibility methods ---

    def save_raw(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("raw", filename)
        self.container_client.get_blob_client(blob_path).upload_blob(
            data, overwrite=True
        )
        return f"azure://{self.container_name}/{blob_path}"

    def save_processed(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("processed", filename)
        self.container_client.get_blob_client(blob_path).upload_blob(
            data, overwrite=True
        )
        return f"azure://{self.container_name}/{blob_path}"

    def save_features(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("features", filename)
        self.container_client.get_blob_client(blob_path).upload_blob(
            data, overwrite=True
        )
        return f"azure://{self.container_name}/{blob_path}"

    def exists(self, relative_path: str) -> bool:
        blob_client = self.container_client.get_blob_client(relative_path)
        return blob_client.exists()

    def load(self, relative_path: str) -> bytes:
        blob_client = self.container_client.get_blob_client(relative_path)
        try:
            return blob_client.download_blob().readall()
        except ResourceNotFoundError as exc:
            raise StorageFileNotFoundError(
                message=f"Blob not found: {relative_path}"
            ) from exc
