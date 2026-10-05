from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fleetsense.core.exceptions.storage import (
    StorageConnectionError,
    StorageFileNotFoundError,
)
from fleetsense.storage.azure_blob import AzureBlobStorage
from fleetsense.storage.factory import get_storage
from fleetsense.storage.local import LocalStorage


@pytest.fixture
def temp_local_storage(tmp_path: Path) -> LocalStorage:
    return LocalStorage(root_dir=tmp_path / "test_data")


def test_local_storage_dirs_created(temp_local_storage: LocalStorage):
    assert temp_local_storage.raw_dir.exists()
    assert temp_local_storage.processed_dir.exists()
    assert temp_local_storage.features_dir.exists()
    assert temp_local_storage.models_dir.exists()


def test_local_storage_save_file(
    temp_local_storage: LocalStorage, tmp_path: Path
):
    src_file = tmp_path / "source.parquet"
    src_file.write_bytes(b"sample-parquet-content")

    stored_path = temp_local_storage.save_file(
        source_path=src_file,
        category="raw",
        filename="trip_2026.parquet",
    )

    assert stored_path.exists()
    assert stored_path.name == "trip_2026.parquet"
    assert stored_path.read_bytes() == b"sample-parquet-content"
    assert "trip_2026.parquet" in temp_local_storage.list_files("raw")


def test_local_storage_save_file_non_existent(temp_local_storage: LocalStorage):
    with pytest.raises(StorageFileNotFoundError):
        temp_local_storage.save_file(
            source_path="non_existent_file.parquet",
            category="raw",
        )


def test_local_storage_dataframe_roundtrip(temp_local_storage: LocalStorage):
    df = pd.DataFrame(
        {
            "trip_id": [1, 2, 3],
            "distance": [2.5, 4.0, 1.2],
        }
    )

    saved_path = temp_local_storage.save_dataframe(
        dataframe=df,
        category="processed",
        filename="cleaned_trips.parquet",
    )

    assert saved_path.exists()

    loaded_df = temp_local_storage.load_dataframe(
        category="processed",
        filename="cleaned_trips.parquet",
    )

    pd.testing.assert_frame_equal(df, loaded_df)


def test_local_storage_load_missing_dataframe(temp_local_storage: LocalStorage):
    with pytest.raises(StorageFileNotFoundError):
        temp_local_storage.load_dataframe(
            category="processed",
            filename="missing.parquet",
        )


def test_local_storage_download_to_file(
    temp_local_storage: LocalStorage, tmp_path: Path
):
    df = pd.DataFrame({"x": [10, 20]})
    temp_local_storage.save_dataframe(df, "features", "f.parquet")

    dest = tmp_path / "downloaded" / "f_copy.parquet"
    out = temp_local_storage.download_to_file("features", "f.parquet", dest)

    assert out.exists()
    assert dest.exists()


def test_local_storage_delete(temp_local_storage: LocalStorage):
    df = pd.DataFrame({"a": [1]})
    temp_local_storage.save_dataframe(df, "raw", "to_delete.parquet")
    assert "to_delete.parquet" in temp_local_storage.list_files("raw")

    deleted = temp_local_storage.delete("raw", "to_delete.parquet")
    assert deleted is True
    assert "to_delete.parquet" not in temp_local_storage.list_files("raw")

    # Deleting non-existent returns False
    assert temp_local_storage.delete("raw", "already_gone.parquet") is False


def test_local_storage_legacy_methods(temp_local_storage: LocalStorage):
    raw_p = temp_local_storage.save_raw("raw.bin", b"raw-bytes")
    assert raw_p.exists()

    proc_p = temp_local_storage.save_processed("proc.bin", b"proc-bytes")
    assert proc_p.exists()

    feat_p = temp_local_storage.save_features("feat.bin", b"feat-bytes")
    assert feat_p.exists()

    assert temp_local_storage.exists("raw/raw.bin")
    assert temp_local_storage.load("raw/raw.bin") == b"raw-bytes"


def test_get_storage_factory_local(tmp_path: Path):
    storage = get_storage(backend="local", root_dir=str(tmp_path))
    assert isinstance(storage, LocalStorage)


def test_get_storage_factory_unsupported():
    with pytest.raises(ValueError, match="Unsupported storage backend"):
        get_storage(backend="google_cloud_storage")


def test_azure_storage_missing_credentials(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)
    with pytest.raises(StorageConnectionError, match="Azure Storage connection string is missing"):
        AzureBlobStorage(connection_string="")


@patch("fleetsense.storage.azure_blob.BlobServiceClient")
def test_azure_storage_save_and_list(mock_blob_service_client, tmp_path: Path):
    mock_service = MagicMock()
    mock_container = MagicMock()
    mock_blob = MagicMock()

    mock_blob_service_client.from_connection_string.return_value = mock_service
    mock_service.get_container_client.return_value = mock_container
    mock_container.get_blob_client.return_value = mock_blob

    # Mock list blobs
    fake_blob = MagicMock()
    fake_blob.name = "raw/yellow_2026.parquet"
    mock_container.list_blobs.return_value = [fake_blob]

    azure_storage = AzureBlobStorage(
        connection_string="DefaultEndpointsProtocol=https;AccountName=test;AccountKey=fake==;",
        container_name="test-container",
    )

    src_file = tmp_path / "upload.parquet"
    src_file.write_bytes(b"cloud-data")

    result_uri = azure_storage.save_file(
        source_path=src_file,
        category="raw",
        filename="yellow_2026.parquet",
    )

    assert result_uri == "azure://test-container/raw/yellow_2026.parquet"
    mock_blob.upload_blob.assert_called_once()

    files = azure_storage.list_files("raw")
    assert files == ["yellow_2026.parquet"]
