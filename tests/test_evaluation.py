import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.ml.evaluation import evaluate_model
from fleetsense.ml.training import train_model


def sample_data():
    features = pd.DataFrame(
        {
            "trip_distance": [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
            ],
            "passenger_count": [
                1,
                1,
                2,
                2,
                3,
                1,
                2,
                1,
            ],
            "pickup_hour": [
                8,
                9,
                10,
                11,
                17,
                18,
                19,
                20,
            ],
        }
    )

    target = pd.Series(
        [
            8.0,
            12.0,
            16.0,
            21.0,
            28.0,
            32.0,
            37.0,
            42.0,
        ],
        name="trip_duration_minutes",
    )

    return features, target


def test_evaluate_model_returns_required_metrics():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=20,
    )

    metrics = evaluate_model(
        model,
        features,
        target,
    )

    assert set(metrics.keys()) == {
        "mae",
        "rmse",
        "r2",
    }


def test_metrics_are_numeric():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=20,
    )

    metrics = evaluate_model(
        model,
        features,
        target,
    )

    assert isinstance(metrics["mae"], float)
    assert isinstance(metrics["rmse"], float)
    assert isinstance(metrics["r2"], float)


def test_error_metrics_are_non_negative():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=20,
    )

    metrics = evaluate_model(
        model,
        features,
        target,
    )

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0


def test_evaluation_uses_predictions():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=20,
    )

    metrics = evaluate_model(
        model,
        features,
        target,
    )

    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["r2"] <= 1.0


def test_invalid_model_fails():
    features, target = sample_data()

    with pytest.raises(DataValidationError):
        evaluate_model(
            model=object(),
            features=features,
            target=target,
        )


def test_mismatched_lengths_fail():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=10,
    )

    target = target.iloc[:-1]

    with pytest.raises(DataValidationError):
        evaluate_model(
            model,
            features,
            target,
        )


def test_empty_features_fail():
    _, target = sample_data()

    model = RandomForestRegressor(
        n_estimators=10,
        random_state=42,
    )

    with pytest.raises(DataValidationError):
        evaluate_model(
            model,
            pd.DataFrame(),
            target,
        )


def test_null_values_fail():
    features, target = sample_data()

    model = train_model(
        features,
        target,
        n_estimators=10,
    )

    features.loc[
        0,
        "trip_distance",
    ] = None

    with pytest.raises(DataValidationError):
        evaluate_model(
            model,
            features,
            target,
        )