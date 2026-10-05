from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from fleetsense.core.config import settings
from fleetsense.core.constants.paths import TMP_DATA_DIR
from fleetsense.core.logger import get_logger
from fleetsense.etl.downloader import TLCDownloader
from fleetsense.etl.metadata import IngestionMetadata
from fleetsense.etl.validation import clean_and_filter_tlc_dataframe
from fleetsense.storage.base import Storage
from fleetsense.storage.factory import get_storage

logger = get_logger(__name__)


def _get_latest_available_month(
    downloader: TLCDownloader,
    dataset: str,
    start_year: int,
    start_month: int,
) -> tuple[int, int] | None:
    """Find the latest published TLC month."""
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
    """Determine which available months still need ingestion."""
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
    Uses memory-safe streaming/copying instead of loading full parquet bytes into RAM.
    """
    dataset = dataset or settings.ingestion.dataset
    metadata = IngestionMetadata()
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

    TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for current_year, current_month in months_to_process:
        filename = (
            f"{dataset}_tripdata_{current_year:04d}-{current_month:02d}.parquet"
        )

        if metadata.is_downloaded(filename):
            continue

        temporary_path = TMP_DATA_DIR / filename
        downloaded_path: Path | None = None

        try:
            logger.info("Downloading %s ...", filename)
            downloaded_path = downloader.download(
                dataset=dataset,
                year=current_year,
                month=current_month,
                output_path=temporary_path,
            )

            # Memory-safe storage: save file path directly if supported
            if hasattr(active_storage, "save_file"):
                stored_path = active_storage.save_file(
                    source_path=downloaded_path,
                    category="raw",
                    filename=filename,
                )
            else:
                # Fallback for legacy mocks
                stored_path = active_storage.save_raw(
                    filename=filename,
                    data=downloaded_path.read_bytes(),
                )

            metadata.mark_downloaded(filename=filename, dataset=dataset)
            downloaded_files.append(stored_path)
            logger.info("Successfully ingested %s to %s", filename, stored_path)

        finally:
            if downloaded_path is not None and downloaded_path.exists():
                downloaded_path.unlink()

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

    logger.info("Loading raw dataset %s for cleaning...", filename)
    raw_df = active_storage.load_dataframe(category="raw", filename=filename)

    cleaned_df, report = clean_and_filter_tlc_dataframe(
        raw_df, max_allowed_loss_pct=max_allowed_loss_pct
    )

    stored_path = active_storage.save_dataframe(
        dataframe=cleaned_df,
        category="processed",
        filename=filename,
    )

    logger.info(
        "Successfully processed %s. Rows kept: %d / %d (%.2f%% loss). Stored at: %s",
        filename,
        report["cleaned_rows"],
        report["initial_rows"],
        report["loss_percentage"],
        stored_path,
    )

    return stored_path, report