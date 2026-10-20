import os
import sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

# 1. Load production environment
project_root = Path.cwd().resolve()
if not (project_root / "src").exists() and (project_root.parent / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

env_path = project_root / ".env.production"
load_dotenv(dotenv_path=env_path, override=True)

# %%
from fleetsense.storage.factory import get_storage
from fleetsense.etl.downloader import TLCDownloader
from fleetsense.etl.metadata import IngestionMetadata
from fleetsense.core.constants.paths import TMP_DATA_DIR

storage = get_storage()
downloader = TLCDownloader()
metadata = IngestionMetadata()

print(f"Connected storage: {type(storage).__name__} (Container: {storage.container_name})")

# %%
# Calculate the target 18-month range
def get_last_n_months(num_months: int = 18):
    today = date.today()
    months = []
    y, m = today.year, today.month
    for _ in range(num_months):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        months.append((y, m))
    months.reverse()
    return months

target_months = get_last_n_months(18)
print(f"Target 18 months: {target_months[0]} to {target_months[-1]}")

# %%
dataset = "yellow"
TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)

for year, month in target_months:
    filename = f"{dataset}_tripdata_{year:04d}-{month:02d}.parquet"
    print(f"\nChecking {filename}...")

    # Check if file already exists in raw Azure storage or metadata
    if metadata.is_downloaded(filename):
        print(f"-> Already marked as downloaded in metadata. Skipping.")
        continue

    # Verify if NYC TLC has published this month
    if not downloader.is_available(dataset=dataset, year=year, month=month):
        print(f"-> {filename} is not yet available from TLC. Skipping.")
        continue

    tmp_target = TMP_DATA_DIR / filename
    try:
        print(f"-> Downloading {filename}...")
        downloaded_file = downloader.download(
            dataset=dataset, year=year, month=month, output_path=tmp_target
        )

        print(f"-> Uploading {filename} to Azure Blob (raw/)...")
        stored_uri = storage.save_file(
            source_path=downloaded_file,
            category="raw",
            filename=filename,
        )
        metadata.mark_downloaded(filename=filename, dataset=dataset)
        print(f"Uploaded: {stored_uri}")

    finally:
        if tmp_target.exists():
            tmp_target.unlink()

print("\nStage 1 Complete: All available months uploaded to Azure Blob raw/.")

# # %% [markdown]
# # # Stage 1: Ingest TLC Data (Year: 2025, Month: 03) to Azure Blob Storage

# # %%
# import os
# import sys
# from pathlib import Path
# from dotenv import load_dotenv

# # 1. Resolve project root and add 'src' to path
# project_root = Path.cwd().resolve()
# if not (project_root / "src").exists() and (project_root.parent / "src").exists():
#     project_root = project_root.parent
# sys.path.insert(0, str(project_root / "src"))

# # 2. Load production environment variables
# env_path = project_root / ".env.production"
# if not env_path.exists():
#     raise FileNotFoundError(f"Missing production env file at: {env_path}")

# load_dotenv(dotenv_path=env_path, override=True)

# # %%
# from fleetsense.storage.factory import get_storage
# from fleetsense.etl.downloader import TLCDownloader
# from fleetsense.etl.metadata import IngestionMetadata
# from fleetsense.core.constants.paths import TMP_DATA_DIR

# # 3. Initialize production storage, downloader, and metadata tracker
# storage = get_storage()
# downloader = TLCDownloader()
# metadata = IngestionMetadata()

# print(f"Connected storage backend : {type(storage).__name__}")
# print(f"Target container          : {getattr(storage, 'container_name', 'Local')}")

# # %%
# # Target: Year 2025, Month 3
# YEAR = 2025
# MONTH = 3
# DATASET = "yellow"

# filename = f"{DATASET}_tripdata_{YEAR:04d}-{MONTH:02d}.parquet"
# url = downloader.build_url(dataset=DATASET, year=YEAR, month=MONTH)

# print(f"\nProcessing file : {filename}")
# print(f"Source URL      : {url}")

# # Step A: Availability Check
# print("Checking availability from NYC TLC CloudFront...")
# if not downloader.is_available(dataset=DATASET, year=YEAR, month=MONTH):
#     raise FileNotFoundError(f"Dataset not available on TLC servers: {url}")
# print("-> File is available for download.")

# # Step B: Download to temporary folder
# TMP_DATA_DIR.mkdir(parents=True, exist_ok=True)
# tmp_file = TMP_DATA_DIR / filename

# try:
#     print(f"Downloading to local temp storage ({tmp_file})...")
#     downloaded_path = downloader.download(
#         dataset=DATASET,
#         year=YEAR,
#         month=MONTH,
#         output_path=tmp_file,
#     )
#     file_size_mb = downloaded_path.stat().st_size / (1024 * 1024)
#     print(f"-> Download complete. Size: {file_size_mb:.2f} MB")

#     # Step C: Upload directly to Azure Blob raw/
#     print(f"Uploading to Azure Blob under 'raw/{filename}'...")
#     stored_uri = storage.save_file(
#         source_path=downloaded_path,
#         category="raw",
#         filename=filename,
#     )
#     print(f"-> Successfully uploaded to: {stored_uri}")

#     # Step D: Update metadata tracking
#     metadata.mark_downloaded(filename=filename, dataset=DATASET)
#     print("-> Metadata recorded as downloaded.")

# finally:
#     # Cleanup local temporary file to conserve disk space
#     if tmp_file.exists():
#         tmp_file.unlink()
#         print("-> Cleaned up local temp file.")

# print("\n--- Stage 1 Complete for 2025-03 ---")