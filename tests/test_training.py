from pathlib import Path
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from fleetsense.core.exceptions.data import DataValidationError
from fleetsense.ml.training import load_model, save_model, train_model


def sample_training_data():
    features = pd.DataFrame(
        {
            "trip_distance": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "passenger_count": [1, 1, 2, 2, 3, 1],
            "pickup_hour": [8, 9, 10, 17, 18, 19],
        }
    )

    target = pd.Series(
        [10.0, 15.0, 20.0, 25.0, 30.0, 35.0],
        name="trip_duration_minutes",
    )

    return features, target


def test_train_model_returns_fitted_random_forest():
    features, target = sample_training_data()

    model = train_model(
        features,
        target,
        n_estimators=10,
    )

    assert isinstance(model, RandomForestRegressor)
    assert hasattr(model, "estimators_")
    assert len(model.estimators_) == 10


def test_trained_model_can_predict():
    features, target = sample_training_data()

    model = train_model(
        features,
        target,
        n_estimators=10,
    )

    predictions = model.predict(features)
    assert len(predictions) == len(target)
    assert all(prediction >= 0 for prediction in predictions)


def test_empty_features_fail():
    _, target = sample_training_data()
    features = pd.DataFrame()
    with pytest.raises(DataValidationError):
        train_model(features, target)


def test_empty_target_fails():
    features, _ = sample_training_data()
    target = pd.Series(dtype="float64")
    with pytest.raises(DataValidationError):
        train_model(features, target)


def test_mismatched_lengths_fail():
    features, target = sample_training_data()
    target = target.iloc[:-1]
    with pytest.raises(DataValidationError):
        train_model(features, target)


def test_missing_feature_values_fail():
    features, target = sample_training_data()
    features.loc[0, "trip_distance"] = None
    with pytest.raises(DataValidationError):
        train_model(features, target)


def test_non_numeric_features_fail():
    features, target = sample_training_data()
    features["vehicle_type"] = [
        "sedan", "sedan", "van", "van", "sedan", "van"
    ]
    with pytest.raises(DataValidationError):
        train_model(features, target)


def test_invalid_estimator_count_fails():
    features, target = sample_training_data()
    with pytest.raises(DataValidationError):
        train_model(features, target, n_estimators=0)


def test_train_model_hist_gradient_boosting():
    features, target = sample_training_data()
    model = train_model(
        features=features,
        target=target,
        algorithm="HistGradientBoostingRegressor",
        n_estimators=20,
    )
    assert isinstance(model, HistGradientBoostingRegressor)
    preds = model.predict(features)
    assert len(preds) == len(target)


def test_save_and_load_model(tmp_path: Path):
    features, target = sample_training_data()
    model = train_model(features, target, n_estimators=10)

    model_path = tmp_path / "model.joblib"
    saved = save_model(model, model_path)
    assert saved.exists()

    loaded = load_model(saved)
    preds_orig = model.predict(features)
    preds_loaded = loaded.predict(features)
    assert (preds_orig == preds_loaded).all()