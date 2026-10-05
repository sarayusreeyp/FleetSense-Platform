from pathlib import Path
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from fleetsense.workflows.train import (
    evaluate_model_task,
    load_features_task,
    split_data_task,
    train_flow,
    train_model_task,
)


@pytest.fixture
def sample_feature_df():
    # 20 rows with complete feature matrix
    return pd.DataFrame(
        {
            "VendorID": [1 if i % 2 == 0 else 2 for i in range(20)],
            "passenger_count": [1 if i % 2 == 0 else 2 for i in range(20)],
            "trip_distance": [float(i + 1) for i in range(20)],
            "PULocationID": [100 + i for i in range(20)],
            "DOLocationID": [150 + i for i in range(20)],
            "pickup_hour": [8 + (i % 12) for i in range(20)],
            "pickup_day_of_week": [i % 7 for i in range(20)],
            "pickup_month": [10 for _ in range(20)],
            "pickup_day": [1 + (i % 28) for i in range(20)],
            "is_weekend": [1 if (i % 7) >= 5 else 0 for i in range(20)],
            "trip_duration_minutes": [10.0 + i * 2.5 for i in range(20)],
        }
    )


def test_load_features_task(sample_feature_df):
    mock_storage = MagicMock()
    mock_storage.list_files.return_value = ["feat_2026.parquet"]
    mock_storage.load_dataframe.return_value = sample_feature_df

    with patch("fleetsense.workflows.train.get_storage", return_value=mock_storage):
        X, y, fname = load_features_task.fn()

    assert len(X) == 20
    assert len(y) == 20
    assert "trip_duration_minutes" not in X.columns
    assert fname == "feat_2026.parquet"


def test_train_flow_e2e(sample_feature_df):
    mock_storage = MagicMock()
    mock_storage.list_files.return_value = ["features_sample.parquet"]
    mock_storage.load_dataframe.return_value = sample_feature_df

    with patch("fleetsense.workflows.train.get_storage", return_value=mock_storage):
        result = train_flow(
            feature_filename="features_sample.parquet",
            algorithm="HistGradientBoostingRegressor",
        )

    assert "model_artifact" in result
    assert "metrics" in result
    assert "validation" in result["metrics"]
    assert "test" in result["metrics"]
    assert Path(result["model_artifact"]).exists()
