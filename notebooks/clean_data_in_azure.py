import os
import sys
from pathlib import Path
from dotenv import load_dotenv

project_root = Path.cwd().resolve()
if not (project_root / "src").exists() and (project_root.parent / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

load_dotenv(dotenv_path=project_root / ".env.production", override=True)

from fleetsense.storage.factory import get_storage
from fleetsense.etl.ingestion import clean_and_process_dataset
from fleetsense.etl.metadata import IngestionMetadata

storage = get_storage()
metadata = IngestionMetadata()

raw_files = storage.list_files("raw")
processed_files = set(storage.list_files("processed"))

print(f"Total raw files in Azure Blob      : {len(raw_files)}")
print(f"Already processed files in Azure   : {len(processed_files)}")

# Filter only files that have not yet been processed
pending_files = [
    filename for filename in raw_files
    if filename not in processed_files and metadata.get_file_status(filename) != "PROCESSED"
]

print(f"Files pending processing           : {len(pending_files)}")
if pending_files:
    for f in pending_files:
        print(f"  -> {f}")
else:
    print("  -> All raw files are already processed. Nothing to do!")

# %%
# Process only the pending raw files
reports = {}
for filename in pending_files:
    print(f"\nCleaning {filename}...")
    try:
        dest_uri, report = clean_and_process_dataset(
            filename=filename,
            storage=storage,
            max_allowed_loss_pct=35.0,  # Failsafe if data corruption occurs
        )
        reports[filename] = report
        print(f"-> Successfully processed: {dest_uri}")
        print(f"   Clean rows: {report['cleaned_rows']:,} / {report['initial_rows']:,} | Loss: {report['loss_percentage']:.2f}%")
    except Exception as exc:
        print(f"-> Error processing {filename}: {exc}")

print("\nStage 2 Complete.")