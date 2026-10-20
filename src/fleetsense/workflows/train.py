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
    max_history_months: int = 18,
) -> tuple[Any, Any, str, Any]:
    """Load feature matrices from storage using a rolling window and column projection.

    - When ``use_all_months=True``, selects up to the most recent ``max_history_months`` (default 18).
    - Only reads required feature and target columns via PyArrow projection to conserve RAM.
    """
    storage = get_storage()
    available = sorted([f for f in storage.list_files("features") if f.endswith(".parquet")])

    if not available:
        raise FileNotFoundError(
            "No feature datasets found in storage. Run ETL workflow first."
        )

    if filename:
        target_files = [filename]
        logger.info("Loading single feature dataset: %s", filename)
    elif use_all_months:
        # Rolling 18-month window: take the most recent max_history_months
        target_files = available[-max_history_months:]
        logger.info(
            "Loading rolling %d-month window (%d file(s) of %d available): %s",
            max_history_months,
            len(target_files),
            len(available),
            target_files,
        )
    else:
        target_files = [available[-1]]
        logger.info("Loading latest feature dataset: %s", target_files[0])

    # Memory Safety: project only required columns
    needed_columns = list(
        set(
            settings.model.features
            + [TARGET_COLUMN, "pickup_datetime", "tpep_pickup_datetime"]
        )
    )

    frames: list[pd.DataFrame] = []
    for fname in target_files:
        df = storage.load_dataframe(category="features", filename=fname, columns=needed_columns)
        if TARGET_COLUMN not in df.columns:
            raise ValueError(f"Target column '{TARGET_COLUMN}' missing from {fname}.")
        frames.append(df)
        logger.info("Loaded %d rows from %s (projected columns: %d)", len(df), fname, len(df.columns))

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "Combined dataset: %d rows from %d file(s).", len(combined), len(frames)
    )

    # Ensure strictly chronological ordering across files before train/val/test splitting
    date_series = None
    if "pickup_datetime" in combined.columns:
        combined = combined.sort_values(by="pickup_datetime").reset_index(drop=True)
        date_series = combined["pickup_datetime"].copy()
        combined = combined.drop(columns=["pickup_datetime"])
    elif "tpep_pickup_datetime" in combined.columns:
        combined = combined.sort_values(by="tpep_pickup_datetime").reset_index(drop=True)
        date_series = combined["tpep_pickup_datetime"].copy()
        combined = combined.drop(columns=["tpep_pickup_datetime"])

    if filename:
        dataset_label = filename
    elif len(target_files) == 1:
        dataset_label = target_files[0]
    else:
        dataset_label = f"{len(target_files)}_files_{target_files[0]}_to_{target_files[-1]}"

    X = combined.drop(columns=[TARGET_COLUMN])
    y = combined[TARGET_COLUMN]
    return X, y, dataset_label, date_series


@task(name="split-dataset")
def split_data_task(
    X: Any,
    y: Any,
    date_series: Any = None,
    val_cutoff: str | None = None,
    test_cutoff: str | None = None,
) -> dict[str, tuple[Any, Any]]:
    """Chronologically split features and target by calendar cutoff or ratio."""
    if val_cutoff is not None and test_cutoff is not None and date_series is not None:
        from fleetsense.ml.splitting import split_dataset_by_date

        logger.info(
            "Splitting dataset by calendar cutoffs: val_cutoff=%s, test_cutoff=%s",
            val_cutoff,
            test_cutoff,
        )
        return split_dataset_by_date(
            features=X,
            target=y,
            date_column=date_series,
            val_cutoff=val_cutoff,
            test_cutoff=test_cutoff,
        )

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
    full_X: Any = None,
    X_val: Any = None,
    y_val: Any = None,
) -> dict[str, Any]:
    """Persist model artifact and log experiment metadata to MLflow."""
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

    # Log feature importance (tree-based or permutation importance)
    _log_feature_importance(tracker, model, X_val=X_val, y_val=y_val)

    # 1. Primary Model Registry & Artifact Logging via MLflow
    # MLflow manages the authoritative artifact repository (local sqlite or remote blob/S3).
    # Maintain MODELS_DIR/latest_model.joblib for offline fallback.
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_name = f"{settings.model.name}_{algorithm}.joblib"
    local_model_path = MODELS_DIR / model_name
    save_model(model, local_model_path)
    tracker.log_artifact(local_model_path)

    latest_model_path = MODELS_DIR / "latest_model.joblib"
    save_model(model, latest_model_path)

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

    # Export dynamic reference distribution for drift monitoring (from full training set)
    ref_features = full_X if full_X is not None else sample_X
    _export_reference_distribution(ref_features, tracker)

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
    reg_result = tracker.log_and_register_model(
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
    logger.info("Model registration completed: %s", reg_result)

    # Automate Post-Promotion API Reload if newly trained model was promoted or registered
    _trigger_api_reload()

    run_info = tracker.end_run()
    return {
        "model_artifact": str(local_model_path),
        "latest_artifact": str(latest_model_path),
        "run_info": run_info,
        "metrics": metrics,
    }


def _trigger_api_reload() -> None:
    """Notify live FastAPI service to hot-reload the newly promoted champion model."""
    import requests

    api_host = os.getenv("FLEETSENSE_API_HOST") or settings.api.host or "127.0.0.1"
    api_port = os.getenv("FLEETSENSE_API_PORT") or settings.api.port or 8000
    if api_host in ("0.0.0.0", ""):
        api_host = "127.0.0.1"
    reload_url = f"http://{api_host}:{api_port}/model/reload"

    try:
        logger.info("Signaling running FleetSense API at %s to hot-reload model...", reload_url)
        resp = requests.post(reload_url, timeout=5)
        if resp.status_code == 200:
            logger.info("Successfully reloaded model on live API service: %s", resp.json())
        else:
            logger.warning("API reload request returned HTTP %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        logger.info("Live API service is currently offline or unreachable (%s). Reload skipped.", exc)


def _log_feature_importance(
    tracker: MLflowTracker,
    model: Any,
    X_val: Any = None,
    y_val: Any = None,
) -> None:
    """Extract and log feature importance scores (tree-based or permutation importance)."""
    try:
        importances = None
        feature_names = None

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            feature_names = (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else [f"feature_{i}" for i in range(len(importances))]
            )
        elif X_val is not None and y_val is not None and hasattr(model, "predict"):
            # HistGradientBoostingRegressor uses permutation importance
            from sklearn.inspection import permutation_importance

            eval_X = X_val.head(1000) if len(X_val) > 1000 else X_val
            eval_y = y_val.head(1000) if len(y_val) > 1000 else y_val
            perm = permutation_importance(
                model,
                eval_X,
                eval_y,
                n_repeats=5,
                random_state=settings.model.random_state,
                n_jobs=-1,
            )
            importances = perm.importances_mean
            feature_names = (
                list(model.feature_names_in_)
                if hasattr(model, "feature_names_in_")
                else list(eval_X.columns)
            )

        if importances is not None and feature_names is not None:
            fi_metrics = {
                f"fi_{name}": float(imp)
                for name, imp in zip(feature_names, importances)
            }
            tracker.log_metrics(fi_metrics)

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
                "Model %s does not expose feature_importances_ and no validation data was available.",
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

                # Apply Laplace-style smoothing so no bucket has an absolute zero probability
                alpha = 1e-4
                total = sum(counts) + alpha * n_buckets
                fractions = [(c + alpha) / total for c in counts]
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
                logger.info(
                    "Exported dynamic drift reference distribution to %s (sampled from %d rows)",
                    ref_path,
                    len(distances),
                )
    except Exception as exc:
        logger.warning("Failed exporting drift reference distribution: %s", exc)


@flow(
    name="fleetsense-model-training",
    description=(
        "Orchestrated ML training pipeline: Load Features (rolling 18-month window) "
        "-> Split -> Train -> Evaluate -> Quality Gate -> MLflow Log -> API Reload"
    ),
)
def train_flow(
    feature_filename: str | None = None,
    algorithm: str | None = None,
    use_all_months: bool = True,
    max_history_months: int = 18,
    skip_quality_gate: bool = False,
    val_cutoff: str | None = None,
    test_cutoff: str | None = None,
) -> dict[str, Any]:
    """
    FleetSense ML training workflow with concurrency protection.

    Parameters
    ----------
    feature_filename:
        Pin to a specific feature file (bypasses multi-month loading).
    algorithm:
        Override the model algorithm from config.
    use_all_months:
        When True (default), concatenates up to max_history_months feature files.
    max_history_months:
        Number of historical months in the rolling window (default 18).
    skip_quality_gate:
        When True, bypasses the quality gate (use only for debugging).
    val_cutoff:
        Optional calendar date cutoff for validation split (e.g. '2026-07-01').
    test_cutoff:
        Optional calendar date cutoff for test holdout split (e.g. '2026-09-01').
    """
    from filelock import FileLock, Timeout

    from fleetsense.core.constants.paths import METADATA_DIR

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    flow_lock = FileLock(str(METADATA_DIR / "training_flow.lock"), timeout=5)

    try:
        with flow_lock:
            logger.info("Starting FleetSense model training flow...")
            chosen_algo = algorithm or settings.model.algorithm

            X, y, target_file, date_series = load_features_task(
                filename=feature_filename,
                use_all_months=use_all_months,
                max_history_months=max_history_months,
            )
            splits = split_data_task(
                X=X,
                y=y,
                date_series=date_series,
                val_cutoff=val_cutoff,
                test_cutoff=test_cutoff,
            )

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

            X_val, y_val = splits["validation"]
            result = persist_and_track_task(
                model=model,
                dataset_name=target_file,
                metrics=metrics,
                algorithm=chosen_algo,
                sample_X=X_train.head(10),
                sample_y=y_train.head(10),
                full_X=X_train,
                X_val=X_val,
                y_val=y_val,
            )

            logger.info("Training workflow successfully completed.")
            return result
    except Timeout:
        logger.warning(
            "Another model training flow is currently executing. "
            "Skipping concurrent training run to prevent race conditions."
        )
        return {
            "status": "skipped",
            "reason": "concurrent_run_in_progress",
        }


if __name__ == "__main__":
    train_flow()
