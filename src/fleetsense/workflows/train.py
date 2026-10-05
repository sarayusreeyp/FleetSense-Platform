from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import ARTIFACTS_DIR
from fleetsense.core.logger import get_logger
from fleetsense.ml.evaluation import evaluate_model
from fleetsense.ml.features import TARGET_COLUMN
from fleetsense.ml.splitting import split_dataset
from fleetsense.ml.tracking import MLflowTracker
from fleetsense.ml.training import save_model, train_model
from fleetsense.storage.factory import get_storage

logger = get_logger(__name__)


@task(name="load-features-dataset")
def load_features_task(
    filename: str | None = None,
) -> tuple[Any, Any, str]:
    """Load latest or specified feature matrix from storage."""
    storage = get_storage()
    available = storage.list_files("features")

    if not available:
        raise FileNotFoundError(
            "No feature datasets found in storage. Run ETL workflow first."
        )

    target_file = filename or sorted(available)[-1]
    logger.info("Loading feature dataset: %s", target_file)

    df = storage.load_dataframe(category="features", filename=target_file)
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' missing from features.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]

    return X, y, target_file


@task(name="split-dataset")
def split_data_task(
    X: Any,
    y: Any,
) -> dict[str, tuple[Any, Any]]:
    """Chronologically split features and target."""
    return split_dataset(
        features=X,
        target=y,
        train_ratio=settings.model.train_ratio,
        validation_ratio=settings.model.validation_ratio,
        test_ratio=settings.model.test_ratio,
    )


@task(name="train-regressor")
def train_model_task(
    X_train: Any,
    y_train: Any,
    algorithm: str | None = None,
) -> Any:
    """Train the regression model."""
    algo = algorithm or settings.model.algorithm
    logger.info("Training model using algorithm: %s", algo)

    return train_model(
        features=X_train,
        target=y_train,
        algorithm=algo,
        n_estimators=settings.model.n_estimators,
        learning_rate=settings.model.learning_rate,
        max_depth=settings.model.max_depth,
        random_state=settings.model.random_state,
        categorical_features=settings.model.categorical_features,
    )


@task(name="evaluate-model")
def evaluate_model_task(
    model: Any,
    splits: dict[str, tuple[Any, Any]],
) -> dict[str, dict[str, float]]:
    """Evaluate model on validation and test partitions."""
    X_val, y_val = splits["validation"]
    X_test, y_test = splits["test"]

    val_metrics = evaluate_model(model=model, features=X_val, target=y_val)
    test_metrics = evaluate_model(model=model, features=X_test, target=y_test)

    logger.info("Validation metrics: %s", val_metrics)
    logger.info("Test metrics: %s", test_metrics)

    return {
        "validation": val_metrics,
        "test": test_metrics,
    }


@task(name="persist-and-track-model")
def persist_and_track_task(
    model: Any,
    dataset_name: str,
    metrics: dict[str, dict[str, float]],
    algorithm: str,
) -> dict[str, Any]:
    """Persist model artifact and log experiment metadata."""
    tracker = MLflowTracker()
    tracker.start_run(run_name=f"train_{algorithm}_{dataset_name}")

    params = {
        "algorithm": algorithm,
        "dataset": dataset_name,
        "n_estimators": settings.model.n_estimators,
        "learning_rate": settings.model.learning_rate,
        "max_depth": settings.model.max_depth,
    }
    tracker.log_params(params)

    flat_metrics = {
        f"val_{k}": v for k, v in metrics["validation"].items()
    }
    flat_metrics.update(
        {f"test_{k}": v for k, v in metrics["test"].items()}
    )
    tracker.log_metrics(flat_metrics)

    # Save model locally and in storage
    model_name = f"{settings.model.name}_{algorithm}.joblib"
    local_model_path = ARTIFACTS_DIR / model_name
    save_model(model, local_model_path)
    tracker.log_artifact(local_model_path)

    storage = get_storage()
    if hasattr(storage, "save_file"):
        storage.save_file(
            source_path=local_model_path,
            category="models",
            filename=model_name,
        )

    run_info = tracker.end_run()
    return {
        "model_artifact": str(local_model_path),
        "run_info": run_info,
        "metrics": metrics,
    }


@flow(
    name="fleetsense-model-training",
    description="Orchestrated ML training pipeline: Load Features -> Split -> Train -> Evaluate -> MLflow Log",
)
def train_flow(
    feature_filename: str | None = None,
    algorithm: str | None = None,
) -> dict[str, Any]:
    """
    FleetSense ML training workflow.
    """
    logger.info("Starting FleetSense model training flow...")
    chosen_algo = algorithm or settings.model.algorithm

    X, y, target_file = load_features_task(filename=feature_filename)
    splits = split_data_task(X=X, y=y)

    X_train, y_train = splits["train"]
    model = train_model_task(
        X_train=X_train,
        y_train=y_train,
        algorithm=chosen_algo,
    )

    metrics = evaluate_model_task(model=model, splits=splits)
    result = persist_and_track_task(
        model=model,
        dataset_name=target_file,
        metrics=metrics,
        algorithm=chosen_algo,
    )

    logger.info("Training workflow successfully completed.")
    return result


if __name__ == "__main__":
    train_flow()
