import pandas as pd
import pytest

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.ml.splitting import split_dataset


def sample_data():
    features = pd.DataFrame(
        {
            "trip_distance": range(1, 21),
            "pickup_hour": range(20),
        }
    )

    target = pd.Series(
        range(100, 120),
        name="trip_duration_minutes",
    )

    return features, target


def test_dataset_is_split_chronologically():
    features, target = sample_data()

    result = split_dataset(
        features,
        target,
    )

    X_train, y_train = result["train"]
    X_validation, y_validation = result[
        "validation"
    ]
    X_test, y_test = result["test"]

    assert len(X_train) == 14
    assert len(X_validation) == 3
    assert len(X_test) == 3

    assert y_train.iloc[0] == 100
    assert y_train.iloc[-1] == 113

    assert y_validation.iloc[0] == 114
    assert y_validation.iloc[-1] == 116

    assert y_test.iloc[0] == 117
    assert y_test.iloc[-1] == 119


def test_split_does_not_shuffle_data():
    features, target = sample_data()

    result = split_dataset(
        features,
        target,
    )

    X_train, _ = result["train"]
    X_validation, _ = result["validation"]
    X_test, _ = result["test"]

    assert X_train.iloc[0]["trip_distance"] == 1
    assert X_validation.iloc[0]["trip_distance"] == 15
    assert X_test.iloc[0]["trip_distance"] == 18


def test_all_rows_are_preserved():
    features, target = sample_data()

    result = split_dataset(
        features,
        target,
    )

    total_rows = sum(
        len(X)
        for X, _ in result.values()
    )

    assert total_rows == 20


def test_features_and_target_must_have_same_length():
    features, target = sample_data()

    target = target.iloc[:-1]

    with pytest.raises(DataValidationError):
        split_dataset(
            features,
            target,
        )


def test_invalid_ratios_fail():
    features, target = sample_data()

    with pytest.raises(DataValidationError):
        split_dataset(
            features,
            target,
            train_ratio=0.8,
            validation_ratio=0.3,
            test_ratio=-0.1,
        )


def test_ratios_must_sum_to_one():
    features, target = sample_data()

    with pytest.raises(DataValidationError):
        split_dataset(
            features,
            target,
            train_ratio=0.7,
            validation_ratio=0.2,
            test_ratio=0.2,
        )


def test_empty_features_fail():
    features, target = sample_data()

    features = pd.DataFrame()

    with pytest.raises(DataValidationError):
        split_dataset(
            features,
            target,
        )