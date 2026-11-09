from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd
from prefect import flow, task

from fleetsense.core.logger import get_logger
from fleetsense.etl.ingestion import (
    clean_and_process_dataset,
    get_pending_unprocessed_files,
    run_ingestion,
)
from fleetsense.etl.metadata import FeaturesMetadata
from fleetsense.ml.features import TARGET_COLUMN, create_features
from fleetsense.storage.factory import get_storage

logger = get_logger(__name__)


@task(
    name="ingest-tlc-data",
    retries=2,
    retry_delay_seconds=30,
)
def ingestion_task(
    dataset: str | None = None,
    year: int | None = None,
    months: list[int] | None = None,
) -> list[Path | str]:
    """Execute the TLC download & raw ingestion task."""
    logger.info("Executing ingestion task...")
    return run_ingestion(
        dataset=dataset,
        year=year,
        months=months,
    )


@task(
    name="clean-and-process-data",
    retries=1,
    retry_delay_seconds=10,
)
def cleaning_task(
    raw_files: list[Path | str],
) -> list[dict[str, Any]]:
    """Clean and validate raw TLC datasets into processed storage."""
    storage = get_storage()
    results = []

    for raw_file in raw_files:
        filename = Path(raw_file).name
        logger.info("Processing raw file: %s", filename)
        stored_path, report = clean_and_process_dataset(
            filename=filename,
            storage=storage,
        )
        results.append(
            {
                "filename": filename,
                "processed_path": str(stored_path),
                "quality_report": report,
            }
        )

    return results


@task(
    name="generate-features",
    retries=1,
    retry_delay_seconds=10,
)
def feature_engineering_task(
    processed_items: list[dict[str, Any]],
) -> list[str]:
    """Generate ML feature matrices and targets from processed datasets."""
    storage = get_storage()
    feature_files = []

    for item in processed_items:
        filename = item["filename"]
        logger.info("Generating features for: %s", filename)
        processed_df = storage.load_dataframe(
            category="processed", filename=filename
        )

        X, y = create_features(processed_df)
        features_df = X.copy()
        features_df[TARGET_COLUMN] = y

        # Preserve pickup timestamp for robust cross-file chronological sorting during training
        for ts_col in ("tpep_pickup_datetime", "lpep_pickup_datetime", "pickup_datetime"):
            if ts_col in processed_df.columns:
                features_df["pickup_datetime"] = pd.to_datetime(processed_df[ts_col], errors="coerce")
                break

        saved_path = storage.save_dataframe(
            dataframe=features_df,
            category="features",
            filename=filename,
        )
        feature_files.append(str(saved_path))
        logger.info("Saved features to: %s", saved_path)

        # Update features state management
        try:
            feat_meta = FeaturesMetadata()
            ds_name = filename.split("_")[0] if "_" in filename else "yellow"
            feat_meta.mark_feature_generated(
                filename=filename,
                dataset=ds_name,
                row_count=len(features_df),
                feature_columns=list(X.columns),
                features_path=str(saved_path),
            )
        except Exception as f_meta_err:
            logger.warning("Could not update features_state: %s", f_meta_err)

    return feature_files


@flow(
    name="fleetsense-etl",
    description="End-to-end FleetSense ETL pipeline: Ingestion -> Cleaning -> Feature Engineering",
)
def etl_flow(
    dataset: str = "yellow",
    year: int | None = None,
    months: list[int] | None = None,
    run_features: bool = True,
    reprocess_existing: bool = False,
    trigger_training: bool = False,
) -> dict[str, Any]:
    """
    FleetSense end-to-end ETL flow.

    Parameters
    ----------
    dataset:
        Taxi dataset type ('yellow', 'green', etc.)
    year:
        Specific year to ingest.
    months:
        Specific list of months to ingest.
    run_features:
        Whether to generate feature matrices after cleaning.
    reprocess_existing:
        When True, processes any existing raw files in storage if no new downloads occurred.
    trigger_training:
        When True, automatically triggers train_flow upon successful feature engineering.
    """
    from filelock import FileLock, Timeout

    from fleetsense.core.constants.paths import METADATA_DIR

    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    lock_timeout = int(os.getenv("FLEETSENSE_FLOW_LOCK_TIMEOUT", "60"))
    etl_lock = FileLock(str(METADATA_DIR / "etl_flow.lock"), timeout=lock_timeout)

    try:
        with etl_lock:
            logger.info("Starting FleetSense ETL workflow...")
            ingested_files = ingestion_task(
                dataset=dataset,
                year=year,
                months=months,
            )

            files_to_clean: list[Path | str] = list(ingested_files)

            # Metadata-driven self-healing: automatically check if any raw files are pending processing
            if not files_to_clean:
                pending_unprocessed = get_pending_unprocessed_files(dataset=dataset)
                if pending_unprocessed:
                    logger.info(
                        "Found %d pending unprocessed file(s) in storage: %s",
                        len(pending_unprocessed),
                        pending_unprocessed,
                    )
                    files_to_clean = pending_unprocessed

            # If still no pending files and reprocess_existing is forced, allow reprocessing raw files from storage
            if not files_to_clean:
                storage = get_storage()
                raw_files = sorted(storage.list_files("raw"))
                matching_raw = [
                    f for f in raw_files
                    if f.startswith(f"{dataset}_tripdata_") and f.endswith(".parquet")
                ]
                if year is not None:
                    matching_raw = [f for f in matching_raw if f"{year:04d}" in f]
                if months is not None:
                    month_strs = [f"{year or 2026:04d}-{m:02d}" for m in months]
                    matching_raw = [f for f in matching_raw if any(ms in f for ms in month_strs)]

                if reprocess_existing and matching_raw:
                    logger.info(
                        "reprocess_existing=True: Processing %d existing raw file(s) from storage: %s",
                        len(matching_raw),
                        matching_raw,
                    )
                    files_to_clean = matching_raw

            if not files_to_clean:
                logger.info("No files to ingest or process.")
                return {
                    "ingested": [str(f) for f in ingested_files],
                    "processed": [],
                    "features": [],
                    "training_triggered": False,
                }

            processed_results = cleaning_task(raw_files=files_to_clean)

            feature_results = []
            if run_features:
                feature_results = feature_engineering_task(
                    processed_items=processed_results
                )

            # Event-driven training trigger
            training_triggered = False
            if trigger_training:
                logger.info("trigger_training=True: Automatically triggering train_flow...")
                from fleetsense.workflows.train import train_flow
                train_flow()
                training_triggered = True

            logger.info("ETL workflow completed successfully.")
            return {
                "ingested": [str(f) for f in ingested_files],
                "processed": processed_results,
                "features": feature_results,
                "training_triggered": training_triggered,
            }
    except Timeout:
        logger.warning(
            "Another ETL workflow is currently executing. "
            "Skipping concurrent ETL run to prevent race conditions."
        )
        return {
            "status": "skipped",
            "reason": "concurrent_run_in_progress",
        }


if __name__ == "__main__":
    etl_flow()
