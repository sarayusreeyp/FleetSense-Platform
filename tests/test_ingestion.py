from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from fleetsense.etl.ingestion import (
    clean_and_process_dataset,
    run_ingestion,
)


def test_run_ingestion_downloads_and_stores_file(
    tmp_path: Path,
):
    fake_file = tmp_path / "yellow_tripdata_2026-01.parquet"
    fake_file.write_bytes(b"fake-parquet-data")

    with patch(
        "fleetsense.etl.ingestion.TLCDownloader"
    ) as mock_downloader, patch(
        "fleetsense.etl.ingestion.IngestionMetadata"
    ) as mock_metadata, patch(
        "fleetsense.etl.ingestion.get_storage"
    ) as mock_get_storage:

        metadata = mock_metadata.return_value
        downloader = mock_downloader.return_value
        storage = mock_get_storage.return_value

        metadata.is_downloaded.return_value = False
        downloader.download.return_value = fake_file

        expected_path = Path("data/raw/yellow_tripdata_2026-01.parquet")
        storage.save_file.return_value = expected_path

        result = run_ingestion(
            dataset="yellow",
            year=2026,
            months=[1],
        )

    assert result == [expected_path]
    downloader.download.assert_called_once()
    storage.save_file.assert_called_once()
    metadata.mark_downloaded.assert_called_once_with(
        filename="yellow_tripdata_2026-01.parquet",
        dataset="yellow",
    )


def test_run_ingestion_skips_downloaded_file():
    with patch(
        "fleetsense.etl.ingestion.TLCDownloader"
    ) as mock_downloader, patch(
        "fleetsense.etl.ingestion.IngestionMetadata"
    ) as mock_metadata, patch(
        "fleetsense.etl.ingestion.get_storage"
    ) as mock_get_storage:

        metadata = mock_metadata.return_value
        downloader = mock_downloader.return_value
        storage = mock_get_storage.return_value

        metadata.is_downloaded.return_value = True

        result = run_ingestion(
            dataset="yellow",
            year=2026,
            months=[1],
        )

    assert result == []
    downloader.download.assert_not_called()
    metadata.mark_downloaded.assert_not_called()


def test_clean_and_process_dataset():
    sample_df = pd.DataFrame(
        {
            "VendorID": [1, 1],
            "tpep_pickup_datetime": [
                "2026-01-01 10:00:00",
                "2026-01-01 10:00:00",
            ],
            "tpep_dropoff_datetime": [
                "2026-01-01 10:20:00",
                "2026-01-01 10:00:00",  # Dirty row: 0 duration
            ],
            "passenger_count": [1, 1],
            "trip_distance": [2.5, 0.0],
            "PULocationID": [100, 100],
            "DOLocationID": [101, 101],
            "payment_type": [1, 1],
            "fare_amount": [12.0, 12.0],
            "total_amount": [15.0, 15.0],
        }
    )

    mock_storage = MagicMock()
    mock_storage.load_dataframe.return_value = sample_df
    mock_storage.save_dataframe.return_value = Path(
        "data/processed/yellow_tripdata_2026-01.parquet"
    )

    stored_path, report = clean_and_process_dataset(
        filename="yellow_tripdata_2026-01.parquet",
        storage=mock_storage,
        max_allowed_loss_pct=70.0,
    )

    assert stored_path == Path("data/processed/yellow_tripdata_2026-01.parquet")
    assert report["initial_rows"] == 2
    assert report["cleaned_rows"] == 1
    assert report["dropped_rows"] == 1
    mock_storage.load_dataframe.assert_called_once_with(
        category="raw", filename="yellow_tripdata_2026-01.parquet"
    )
    mock_storage.save_dataframe.assert_called_once()