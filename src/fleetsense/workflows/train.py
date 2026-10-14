from __future__ import annotations

import os
from typing import Any

import pandas as pd
from prefect import flow, task

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import ARTIFACTS_DIR, MODELS_DIR
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
    use_all_months: bool = True,
) -> tuple[Any, Any, str]:
    """Load feature matrices from storage.

    Gap 1 fix: when ``use_all_months=True`` (the default), ALL available feature
    files are concatenated into a single training dataset instead of only the most
    recent one.  Pass ``filename`` to pin to a specific file.
    """
    storage = get_storage()
    available = sorted(storage.list_files("features"))

    if not available:
        raise FileNotFoundError(
            "No feature datasets found in storage. Run ETL workflow first."
        )

    if filename:
        # Explicit single-file mode
        target_files = [filename]
        logger.info("Loading single feature dataset: %s", filename)
    elif use_all_months:
        target_files = available
        logger.info(
            "Loading ALL %d feature file(s) for multi-month training: %s",
            len(target_files),
            target_files,
        )
    else:
        # Legacy behaviour: latest file only
        target_files = [available[-1]]
        logger.info("Loading latest feature dataset: %s", target_files[0])

    frames: list[pd.DataFrame] = []
    for fname in target_files:
        df = storage.load_dataframe(category="features", filename=fname)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' missing from {fname}.")
        frames.append(df)
        logger.info("Loaded %d rows from %s", len(df), fname)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d rows from %d file(s).", len(combined), len(frames)
    )

    if filename:
        dataset_label = filename
    elif len(target_files) == 1:
        dataset_label = target_files[0]
    else:
        dataset_label = f"{len(target_files)}_files_{target_files[0]}_to_{target_files[-1]}"

    X = combined.drop(columns=[TARGET_COLUMN])
    y = combined[TARGET_COLUMN]
    return X, y, dataset_label


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
    logger.info("Training model using algorithm: %s on %d rows", algo, len(X_train))

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


@task(name="quality-gate")
def quality_gate_task(
    metrics: dict[str, dict[str, float]],
    max_val_mae: float | None = None,
    min_val_r2: float | None = None,
) -> None:
    """Gap 14: Reject a bad retrain if metrics fall outside acceptable bounds.

    Thresholds are read from ``model.yaml`` (quality_gate section) with
    sensible fall-backs.  Raises ``ValueError`` if the model fails the gate,
    which causes the Prefect run to fail and prevents the bad model from
    replacing the production artifact.
    """
    gate_max_mae = max_val_mae or settings.model.quality_gate_max_mae
    gate_min_r2 = min_val_r2 or settings.model.quality_gate_min_r2

    val = metrics["validation"]
    failed: list[str] = []

    if gate_max_mae is not None and val["mae"] > gate_max_mae:
        failed.append(
            f"Validation MAE {val['mae']:.4f} exceeds maximum allowed {gate_max_mae:.4f}"
        )
    if gate_min_r2 is not None and val["r2"] < gate_min_r2:
        failed.append(
            f"Validation R² {val['r2']:.4f} is below minimum required {gate_min_r2:.4f}"
        )

    if failed:
        msg = "MODEL QUALITY GATE FAILED — retrain rejected:\n" + "\n".join(
            f"  ✗ {f}" for f in failed
        )
        logger.error(msg)
        raise ValueError(msg)

    logger.info(
        "Model quality gate PASSED — MAE=%.4f (max=%.4f), R²=%.4f (min=%.4f)",
        val["mae"],
        gate_max_mae or float("inf"),
        val["r2"],
        gate_min_r2 or float("-inf"),
    )


@task(name="persist-and-track-model")
def persist_and_track_task(
    model: Any,
    dataset_name: str,
    metrics: dict[str, dict[str, float]],
    algorithm: str,
    sample_X: Any = None,
    sample_y: Any = None,
) -> dict[str, Any]:
    """Persist model artifact and log experiment metadata.

    Saves model into artifacts/models/ (tracked by DVC) and logs signature/alias to MLflow.
    """
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

    # Gap 18: Log feature importance if the model exposes it
    _log_feature_importance(tracker, model)

    # Save model into artifacts/models/ for DVC and container deployment
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"{settings.model.name}_{algorithm}.joblib"
    local_model_path = MODELS_DIR / model_name
    save_model(model, local_model_path)
    tracker.log_artifact(local_model_path)

    # Also save latest_model.joblib pointer and backward-compatible root artifact
    latest_model_path = MODELS_DIR / "latest_model.joblib"
    save_model(model, latest_model_path)
    legacy_path = ARTIFACTS_DIR / model_name
    save_model(model, legacy_path)

    # Infer model signature if sample data is provided
    signature = None
    input_example = None
    if sample_X is not None and sample_y is not None:
        try:
            import mlflow.models
            signature = mlflow.models.infer_signature(sample_X, sample_y)
            input_example = sample_X.head(2).to_dict(orient="records") if hasattr(sample_X, "to_dict") else None
        except Exception as sig_err:
            logger.debug("Signature inference skipped: %s", sig_err)

    # Export metrics.json for DVC tracking
    metrics_path = ARTIFACTS_DIR / "metrics.json"
    try:
        import json
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(flat_metrics, f, indent=2)
        tracker.log_artifact(metrics_path)
    except Exception as m_err:
        logger.warning("Failed saving metrics.json: %s", m_err)

    # Export dynamic reference distribution for drift monitoring
    _export_reference_distribution(sample_X, tracker)

    # Persist model artifacts to storage backend (e.g. Azure Blob or Local)
    try:
        storage = get_storage()
        storage.save_file(local_model_path, category="models", filename=model_name)
        storage.save_file(latest_model_path, category="models", filename="latest_model.joblib")
        logger.info("Persisted model artifacts to storage backend under category='models'")
    except Exception as store_err:
        logger.warning("Storage backend model persist failed (non-critical): %s", store_err)

    # Register the model in MLflow Model Registry with Champion vs Challenger comparison
    val_mae = metrics.get("validation", {}).get("mae")
    tracker.log_and_register_model(
        model=model,
        artifact_path="model",
        registered_model_name=settings.model.name,
        signature=signature,
        input_example=input_example,
        alias="champion",
        challenger_metric=val_mae,
        metric_name="val_mae",
        higher_is_better=False,
    )

    run_info = tracker.end_run()
    return {
        "model_artifact": str(local_model_path),
        "latest_artifact": str(latest_model_path),
        "run_info": run_info,
        "metrics": metrics,
    }


def _log_feature_importance(tracker: MLflowTracker, model: Any) -> None:
    """Gap 18: Extract and log feature importance scores."""
    try:

        # HistGradientBoostingRegressor exposes feature_importances_
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else [f"feature_{i}" for i in range(len(importances))]
            )
            # Log each importance as a metric (prefixed with "fi_")
            fi_metrics = {
                f"fi_{name}": float(imp)
                for name, imp in zip(feature_names, importances)
            }
            tracker.log_metrics(fi_metrics)

            # Also write a human-readable summary as an artifact
            fi_lines = sorted(
                zip(feature_names, importances), key=lambda x: x[1], reverse=True
            )
            summary = "Feature Importances (descending):\n" + "\n".join(
                f"  {name:30s}: {imp:.6f}"
                for name, imp in fi_lines
            )
            logger.info(summary)

            fi_path = ARTIFACTS_DIR / "feature_importance.txt"
            fi_path.parent.mkdir(parents=True, exist_ok=True)
            fi_path.write_text(summary, encoding="utf-8")
            tracker.log_artifact(fi_path)
        else:
            logger.info(
                "Model %s does not expose feature_importances_ — skipping Gap 18 logging.",
                type(model).__name__,
            )
    except Exception as exc:
        logger.warning("Feature importance logging failed (non-critical): %s", exc)


def _export_reference_distribution(features: Any, tracker: MLflowTracker) -> None:
    """Extract trip_distance distribution from training features and persist as drift reference."""
    try:
        import json
        if features is not None and hasattr(features, "columns") and "trip_distance" in features.columns:
            distances = features["trip_distance"].dropna()
            if not distances.empty:
                from fleetsense.monitoring.drift import REFERENCE_DISTANCE_BUCKETS
                n_buckets = len(REFERENCE_DISTANCE_BUCKETS) + 1
                counts = [0] * n_buckets
                for d in distances:
                    idx = n_buckets - 1
                    for i, edge in enumerate(REFERENCE_DISTANCE_BUCKETS):
                        if d < edge:
                            idx = i
                            break
                    counts[idx] += 1
                total = sum(counts)
                fractions = [c / total for c in counts] if total > 0 else []
                ref_payload = {
                    "buckets": list(REFERENCE_DISTANCE_BUCKETS),
                    "fractions": fractions,
                    "sample_count": len(distances),
                }
                ref_path = ARTIFACTS_DIR / "reference_distribution.json"
                ref_path.parent.mkdir(parents=True, exist_ok=True)
                with ref_path.open("w", encoding="utf-8") as f:
                    json.dump(ref_payload, f, indent=2)
                tracker.log_artifact(ref_path)
                logger.info("Exported dynamic drift reference distribution to %s", ref_path)
    except Exception as exc:
        logger.warning("Failed exporting drift reference distribution: %s", exc)


@flow(
    name="fleetsense-model-training",
    description=(
        "Orchestrated ML training pipeline: Load Features (all months) "
        "-> Split -> Train -> Evaluate -> Quality Gate -> MLflow Log"
    ),
)
def train_flow(
    feature_filename: str | None = None,
    algorithm: str | None = None,
    use_all_months: bool = True,
    skip_quality_gate: bool = False,
) -> dict[str, Any]:
    """
    FleetSense ML training workflow.

    Parameters
    ----------
    feature_filename:
        Pin to a specific feature file (bypasses multi-month loading).
    algorithm:
        Override the model algorithm from config.
    use_all_months:
        When True (default), concatenates ALL available feature files.
    skip_quality_gate:
        When True, bypasses the quality gate (use only for debugging).
    """
    logger.info("Starting FleetSense model training flow...")
    chosen_algo = algorithm or settings.model.algorithm

    X, y, target_file = load_features_task(
        filename=feature_filename,
        use_all_months=use_all_months,
    )
    splits = split_data_task(X=X, y=y)

    X_train, y_train = splits["train"]
    model = train_model_task(
        X_train=X_train,
        y_train=y_train,
        algorithm=chosen_algo,
    )

    metrics = evaluate_model_task(model=model, splits=splits)

    # Gap 14: Quality gate — reject bad retrains in production/staging environments
    is_testing = os.getenv("FLEETSENSE_APP_ENV") in ("testing", "test")
    if not skip_quality_gate and not is_testing:
        quality_gate_task(metrics=metrics)

    result = persist_and_track_task(
        model=model,
        dataset_name=target_file,
        metrics=metrics,
        algorithm=chosen_algo,
        sample_X=X_train.head(10),
        sample_y=y_train.head(10),
    )

    logger.info("Training workflow successfully completed.")
    return result


if __name__ == "__main__":
    train_flow()
