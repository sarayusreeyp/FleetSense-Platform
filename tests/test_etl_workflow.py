from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd

from fleetsense.workflows.etl import etl_flow, ingestion_task


def test_ingestion_task_delegates():
    with patch(
        "fleetsense.workflows.etl.run_ingestion",
        return_value=["sample.parquet"],
    ) as mock_ingest:
        # Use .fn to test the task logic directly without Prefect engine overhead
        result = ingestion_task.fn(dataset="yellow", year=2026, months=[1])
        assert result == ["sample.parquet"]
        mock_ingest.assert_called_once_with(
            dataset="yellow", year=2026, months=[1]
        )


def test_etl_flow_empty_ingestion():
    with patch(
        "fleetsense.workflows.etl.run_ingestion",
        return_value=[],
    ):
        result = etl_flow(dataset="yellow", year=2026, months=[1])

    assert result == {
        "ingested": [],
        "processed": [],
        "features": [],
    }


def test_etl_flow_orchestration():
    raw_file = "yellow_tripdata_2026-01.parquet"
    proc_file = Path("data/processed") / raw_file
    feat_file = Path("data/features") / raw_file

    mock_df = pd.DataFrame(
        {
            "VendorID": [1],
            "tpep_pickup_datetime": ["2026-01-01 10:00:00"],
            "tpep_dropoff_datetime": ["2026-01-01 10:20:00"],
            "passenger_count": [1],
            "trip_distance": [2.5],
            "PULocationID": [100],
            "DOLocationID": [101],
            "payment_type": [1],
            "fare_amount": [12.0],
            "total_amount": [15.0],
        }
    )

    with patch(
        "fleetsense.workflows.etl.run_ingestion",
        return_value=[f"data/raw/{raw_file}"],
    ), patch(
        "fleetsense.workflows.etl.clean_and_process_dataset",
        return_value=(proc_file, {"cleaned_rows": 1, "loss_percentage": 0.0}),
    ), patch(
        "fleetsense.workflows.etl.get_storage"
    ) as mock_get_storage:

        mock_storage = MagicMock()
        mock_get_storage.return_value = mock_storage
        mock_storage.load_dataframe.return_value = mock_df
        mock_storage.save_dataframe.return_value = feat_file

        result = etl_flow(dataset="yellow", year=2026, months=[1])

    assert len(result["ingested"]) == 1
    assert len(result["processed"]) == 1
    assert len(result["features"]) == 1
    assert result["processed"][0]["filename"] == raw_file
    assert result["features"][0] == str(feat_file)