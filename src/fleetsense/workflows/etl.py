from __future__ import annotations

from pathlib import Path
from typing import Any

from prefect import flow, task

from fleetsense.core.logger import get_logger
from fleetsense.etl.ingestion import clean_and_process_dataset, run_ingestion
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

        saved_path = storage.save_dataframe(
            dataframe=features_df,
            category="features",
            filename=filename,
        )
        feature_files.append(str(saved_path))
        logger.info("Saved features to: %s", saved_path)

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
    logger.info("Starting FleetSense ETL workflow...")
    ingested_files = ingestion_task(
        dataset=dataset,
        year=year,
        months=months,
    )

    files_to_clean: list[Path | str] = list(ingested_files)

    # If no newly ingested files, allow reprocessing existing raw files from storage
    if not files_to_clean:
        storage = get_storage()
        raw_files = sorted(storage.list_files("raw"))
        matching_raw = [
            f for f in raw_files
            if f.startswith(f"{dataset}_tripdata_") and f.endswith(".parquet")
        ]
        feature_files = storage.list_files("features")
        should_reprocess = reprocess_existing or (matching_raw and not feature_files)

        if should_reprocess and matching_raw:
            logger.info(
                "Processing %d existing raw file(s) from storage: %s",
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
    if trigger_training and feature_results:
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


if __name__ == "__main__":
    etl_flow()
