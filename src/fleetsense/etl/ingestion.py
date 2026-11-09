from __future__ import annotations

import re
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

from fleetsense.core.config import settings
from fleetsense.core.logger import get_logger
from fleetsense.etl.downloader import TLCDownloader
from fleetsense.etl.metadata import (
    FeaturesMetadata,
    IngestionMetadata,
    ProcessedMetadata,
)
from fleetsense.etl.validation import clean_and_filter_tlc_dataframe
from fleetsense.storage.base import Storage
from fleetsense.storage.factory import get_storage

logger = get_logger(__name__)

_FILENAME_DATE_PATTERN = re.compile(r"_(\d{4})-(\d{2})\.parquet$")


def _parse_year_month_from_filename(filename: str) -> tuple[int, int] | None:
    """Extract (year, month) from a standard TLC filename."""
    match = _FILENAME_DATE_PATTERN.search(filename)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def get_pending_unprocessed_files(
    dataset: str = "yellow",
    storage: Storage | None = None,
) -> list[str]:
    """
    Find raw files in storage that have been downloaded but not yet cleaned/processed.
    """
    active_storage = storage or get_storage()
    ingestion_meta = IngestionMetadata()
    processed_meta = ProcessedMetadata()

    raw_files = active_storage.list_files("raw")
    pending = []

    for filename in sorted(raw_files):
        if not filename.startswith(f"{dataset}_tripdata_") or not filename.endswith(".parquet"):
            continue
        if not processed_meta.is_processed(filename, dataset=dataset):
            pending.append(filename)

    return pending


def _get_latest_available_month(
    downloader: TLCDownloader,
    dataset: str,
    start_year: int,
    start_month: int,
) -> tuple[int, int] | None:
    """Find the latest published TLC month scanning backwards from the current date."""
    today = date.today()
    year = today.year
    month = today.month

    while year > start_year or (year == start_year and month >= start_month):
        if downloader.is_available(dataset=dataset, year=year, month=month):
            return year, month

        if month == 1:
            year -= 1
            month = 12
        else:
            month -= 1

    return None


def _get_pending_months(
    dataset: str,
    start_year: int,
    start_month: int,
    latest_year: int,
    latest_month: int,
    metadata: IngestionMetadata,
) -> list[tuple[int, int]]:
    """Determine which available months still need ingestion, resuming from latest downloaded."""
    # Check if there is already a latest downloaded file
    latest_downloaded = metadata.get_latest_downloaded(dataset=dataset)
    if latest_downloaded:
        parsed = _parse_year_month_from_filename(latest_downloaded)
        if parsed:
            prev_year, prev_month = parsed
            # Start from the month immediately following the latest downloaded
            if prev_month == 12:
                eff_start_year = prev_year + 1
                eff_start_month = 1
            else:
                eff_start_year = prev_year
                eff_start_month = prev_month + 1

            # Only advance if it's forward of the configured start_year/month
            if (eff_start_year > start_year) or (
                eff_start_year == start_year and eff_start_month > start_month
            ):
                start_year = eff_start_year
                start_month = eff_start_month

    pending: list[tuple[int, int]] = []
    year = start_year
    month = start_month

    while year < latest_year or (year == latest_year and month <= latest_month):
        filename = f"{dataset}_tripdata_{year:04d}-{month:02d}.parquet"

        if not metadata.is_downloaded(filename):
            pending.append((year, month))

        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    return pending


def run_ingestion(
    dataset: str | None = None,
    year: int | None = None,
    months: list[int] | None = None,
    storage: Storage | None = None,
) -> list[Path | str]:
    """
    Download and store new TLC monthly files into raw storage.
    Uses memory-safe streaming and scoped temporary directory isolation.
    """
    dataset = dataset or settings.ingestion.dataset
    metadata = IngestionMetadata()
    active_storage = storage or get_storage()

    downloader = TLCDownloader()
    active_storage = storage or get_storage()

    downloaded_files: list[Path | str] = []

    if months is not None:
        if year is None:
            raise ValueError("year is required when months are provided.")
        months_to_process = [(year, month) for month in months]
    else:
        start_year = (
            year if year is not None else settings.ingestion.start_year
        )
        start_month = settings.ingestion.start_month

        latest_month = _get_latest_available_month(
            downloader=downloader,
            dataset=dataset,
            start_year=start_year,
            start_month=start_month,
        )

        if latest_month is None:
            return []

        latest_year, latest_month_number = latest_month
        months_to_process = _get_pending_months(
            dataset=dataset,
            start_year=start_year,
            start_month=start_month,
            latest_year=latest_year,
            latest_month=latest_month_number,
            metadata=metadata,
        )

    with tempfile.TemporaryDirectory(prefix="fleetsense_download_") as scratch_dir:
        scratch_path = Path(scratch_dir)

        for current_year, current_month in months_to_process:
            filename = (
                f"{dataset}_tripdata_{current_year:04d}-{current_month:02d}.parquet"
            )

            if metadata.is_downloaded(filename):
                continue

            target_scratch = scratch_path / filename

            logger.info("Downloading %s ...", filename)
            downloaded_path = downloader.download(
                dataset=dataset,
                year=current_year,
                month=current_month,
                output_path=target_scratch,
            )

            stored_path = active_storage.save_file(
                source_path=downloaded_path,
                category="raw",
                filename=filename,
            )

            metadata.mark_downloaded(filename=filename, dataset=dataset)
            downloaded_files.append(stored_path)
            logger.info("Successfully ingested %s to %s", filename, stored_path)

    return downloaded_files


def clean_and_process_dataset(
    filename: str,
    storage: Storage | None = None,
    max_allowed_loss_pct: float = 35.0,
) -> tuple[Path | str, dict[str, Any]]:
    """
    Load a raw parquet file from storage, clean anomalies, and persist to processed storage.

    Parameters
    ----------
    filename : str
        Name of the parquet file in raw storage.
    storage : Storage | None
        Storage backend to use.
    max_allowed_loss_pct : float
        Maximum allowable data loss percentage.

    Returns
    -------
    tuple[Path | str, dict[str, Any]]
        Stored processed path/URI and data quality audit report.
    """
    active_storage = storage or get_storage()
    metadata = IngestionMetadata()

    logger.info("Loading raw dataset %s for cleaning...", filename)
    try:
        raw_df = active_storage.load_dataframe(category="raw", filename=filename)

        cleaned_df, report = clean_and_filter_tlc_dataframe(
            raw_df, max_allowed_loss_pct=max_allowed_loss_pct
        )

        stored_path = active_storage.save_dataframe(
            dataframe=cleaned_df,
            category="processed",
            filename=filename,
        )

        # Record successful processing in metadata states (both ingestion_state and processed_state)
        dataset_name = filename.split("_")[0] if "_" in filename else "yellow"
        metadata.mark_processed(filename=filename, dataset=dataset_name, report=report)
        try:
            processed_meta = ProcessedMetadata()
            processed_meta.mark_processed_file(
                filename=filename,
                dataset=dataset_name,
                report=report,
                processed_path=str(stored_path),
            )
        except Exception as proc_meta_err:
            logger.warning("Could not update processed_state: %s", proc_meta_err)

        logger.info(
            "Successfully processed %s. Rows kept: %d / %d (%.2f%% loss). Stored at: %s",
            filename,
            report["cleaned_rows"],
            report["initial_rows"],
            report["loss_percentage"],
            stored_path,
        )
        return stored_path, report
    except Exception as exc:
        dataset_name = filename.split("_")[0] if "_" in filename else "yellow"
        metadata.mark_failed(filename=filename, dataset=dataset_name, error=str(exc))
        raise
