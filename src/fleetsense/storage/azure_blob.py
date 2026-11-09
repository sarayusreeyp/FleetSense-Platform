from __future__ import annotations

import io
import os
import re
from pathlib import Path

import pandas as pd

_AZURE_IMPORT_ERROR: str | None = None
try:
    from azure.core.exceptions import (
        ResourceExistsError,
        ResourceNotFoundError,
    )
    from azure.storage.blob import BlobServiceClient
    try:
        from azure.identity import DefaultAzureCredential
    except ImportError:
        DefaultAzureCredential = None  # type: ignore[assignment, misc]
    AZURE_AVAILABLE = True
except Exception as exc:
    AZURE_AVAILABLE = False
    _AZURE_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

from fleetsense.core.exceptions.storage import (
    StorageConnectionError,
    StorageError,
    StorageFileNotFoundError,
    StorageWriteError,
)
from fleetsense.storage.base import CANONICAL_STORAGE_CATEGORIES, Storage

_SAFE_CATEGORY_PATTERN = re.compile(r"^[a-zA-Z0-9_\-]+(/[a-zA-Z0-9_\-]+)*$")


class AzureBlobStorage(Storage):
    """
    Azure Blob Storage backend for production FleetSense workloads.
    Supports streaming large datasets directly to and from Azure Blob containers.
    Maintains 1:1 structural parity with LocalStorage categories.
    """

    def __init__(
        self,
        connection_string: str | None = None,
        account_url: str | None = None,
        container_name: str = "fleetsense-data",
    ) -> None:
        if not AZURE_AVAILABLE:
            raise StorageConnectionError(
                message=f"azure-storage-blob package is not installed or failed to import ({_AZURE_IMPORT_ERROR})."
            )

        self.connection_string = (
            connection_string
            or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            or ""
        )
        self.account_url = (
            account_url
            or os.getenv("AZURE_STORAGE_ACCOUNT_URL")
            or ""
        )
        self.container_name = (
            container_name
            or os.getenv("AZURE_STORAGE_CONTAINER_NAME")
            or os.getenv("AZURE_CONTAINER_NAME")
            or "fleetsense-data"
        )

        if not self.connection_string and not self.account_url:
            raise StorageConnectionError(
                message=(
                    "Azure Storage connection string is missing. "
                    "Set AZURE_STORAGE_CONNECTION_STRING or AZURE_STORAGE_ACCOUNT_URL in your environment or config."
                )
            )

        try:
            if self.connection_string:
                self.service_client = BlobServiceClient.from_connection_string(
                    self.connection_string
                )
            else:
                if DefaultAzureCredential is None:
                    raise StorageConnectionError(
                        message="azure-identity package is required when authenticating with account_url without a connection string."
                    )
                self.service_client = BlobServiceClient(
                    account_url=self.account_url,
                    credential=DefaultAzureCredential(),
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
        """Validate category and format blob path with 1:1 parity to LocalStorage."""
        if not category or not isinstance(category, str):
            raise StorageError(message="Storage category must be a non-empty string.")

        clean_category = category.strip()
        if not _SAFE_CATEGORY_PATTERN.match(clean_category):
            raise StorageError(
                message=f"Invalid storage category '{category}'. Must be alphanumeric.",
                details={"category": category},
            )

        safe_filename = Path(filename).name
        return f"{clean_category}/{safe_filename}"

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
        Write a DataFrame to Parquet and upload it to Azure Blob Storage using disk streaming.
        """
        import tempfile

        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)

        try:
            dataframe.to_parquet(
                tmp_path,
                index=False,
                engine="pyarrow",
                compression="snappy",
            )
            with tmp_path.open("rb") as stream:
                blob_client.upload_blob(stream, overwrite=True)
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to save DataFrame to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def load_dataframe(
        self,
        category: str,
        filename: str,
        columns: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Download Parquet data from Azure Blob Storage directly into a DataFrame.
        Uses in-memory buffer or tempfile with guaranteed cleanup.
        """
        blob_path = self._blob_name(category, filename)
        blob_client = self.container_client.get_blob_client(blob_path)

        try:
            download_stream = blob_client.download_blob()
            parquet_buffer = io.BytesIO(download_stream.readall())
            return pd.read_parquet(parquet_buffer, columns=columns)
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
                if name and "/" not in name and not name.startswith(".") and not name.endswith(".tmp"):
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

    def clean_temp_files(self, category: str) -> int:
        """
        Clean up orphaned temporary (.tmp) blobs in Azure Blob Storage under category prefix.
        """
        prefix = f"{category}/"
        removed_count = 0
        try:
            blobs = self.container_client.list_blobs(name_starts_with=prefix)
            for b in blobs:
                name = b.name[len(prefix) :]
                if name.endswith(".tmp"):
                    try:
                        self.container_client.get_blob_client(b.name).delete_blob()
                        removed_count += 1
                    except Exception:
                        pass
            return removed_count
        except Exception as exc:
            raise StorageConnectionError(
                message=f"Failed to clean temporary blobs with prefix {prefix}",
                details={"error": str(exc)},
            ) from exc

    def save_raw(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("raw", filename)
        try:
            self.container_client.get_blob_client(blob_path).upload_blob(
                data, overwrite=True
            )
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to upload raw data to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def save_processed(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("processed", filename)
        try:
            self.container_client.get_blob_client(blob_path).upload_blob(
                data, overwrite=True
            )
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to upload processed data to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def save_features(self, filename: str, data: bytes) -> str:
        blob_path = self._blob_name("features", filename)
        try:
            self.container_client.get_blob_client(blob_path).upload_blob(
                data, overwrite=True
            )
            return f"azure://{self.container_name}/{blob_path}"
        except Exception as exc:
            raise StorageWriteError(
                message=f"Failed to upload features data to Azure Blob {blob_path}",
                details={"error": str(exc)},
            ) from exc

    def exists(self, relative_path: str, filename: str | None = None) -> bool:
        if filename is not None:
            blob_path = self._blob_name(relative_path, filename)
        else:
            blob_path = relative_path.replace("\\", "/").strip("/")
        blob_client = self.container_client.get_blob_client(blob_path)
        return blob_client.exists()

    def load(self, relative_path: str) -> bytes:
        blob_client = self.container_client.get_blob_client(relative_path)
        try:
            return blob_client.download_blob().readall()
        except ResourceNotFoundError as exc:
            raise StorageFileNotFoundError(
                message=f"Blob not found: {relative_path}"
            ) from exc
