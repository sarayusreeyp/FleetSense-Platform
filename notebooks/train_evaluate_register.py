# %% [markdown]
# # Stage 3: Train, Evaluate, and Promote Champion to Azure Blob Storage

# %%
import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import joblib

# 1. Setup paths and load production environment
project_root = Path.cwd().resolve()
if not (project_root / "src").exists() and (project_root.parent / "src").exists():
    project_root = project_root.parent
sys.path.insert(0, str(project_root / "src"))

# Loads your updated .env.production directly
load_dotenv(dotenv_path=project_root / ".env.production", override=True)

from fleetsense.storage.factory import get_storage
from fleetsense.ml.features import create_features
from fleetsense.ml.splitting import split_dataset
from fleetsense.ml.training import train_model
from fleetsense.ml.evaluation import evaluate_model
from fleetsense.ml.tracking import MLflowTracker

storage = get_storage()
processed_files = sorted(storage.list_files("processed"))
print(f"Available processed files in Azure: {len(processed_files)}")
for f in processed_files:
    print(f" - {f}")

# %%
# 2. Load data safely (sampling to avoid RAM/OOM issues)
SAMPLE_PER_MONTH = 100_000

dfs = []
for f in processed_files:
    print(f"Loading {f}...")
    df_month = storage.load_dataframe("processed", f)
    if len(df_month) > SAMPLE_PER_MONTH:
        df_month = df_month.sample(n=SAMPLE_PER_MONTH, random_state=42)
    dfs.append(df_month)

combined_df = pd.concat(dfs, ignore_index=True)

# Sort chronologically by pickup time before splitting
pickup_col = "tpep_pickup_datetime" if "tpep_pickup_datetime" in combined_df.columns else "pickup_datetime"
combined_df[pickup_col] = pd.to_datetime(combined_df[pickup_col])
combined_df = combined_df.sort_values(by=pickup_col).reset_index(drop=True)

print(f"\nTotal dataset shape for modeling: {combined_df.shape}")

# %%
# 3. Feature Engineering
X, y = create_features(combined_df)
print(f"Features: {list(X.columns)}")
print(f"\nTarget summary:\n{y.describe()}")

# 4. Temporal Split
splits = split_dataset(
    features=X,
    target=y,
    train_ratio=0.70,
    validation_ratio=0.15,
    test_ratio=0.15,
)

X_train, y_train = splits["train"]
X_val, y_val = splits["validation"]
X_test, y_test = splits["test"]

print(f"\nTrain set: {X_train.shape}, Val set: {X_val.shape}, Test set: {X_test.shape}")

# %%
# 5. Train, Evaluate, and Deploy Champion
tracker = MLflowTracker()

with tracker.start_run(run_name="prod_champion_hist_gradient_boosting"):
    params = {
        "algorithm": "HistGradientBoostingRegressor",
        "n_estimators": 150,
        "learning_rate": 0.08,
        "max_depth": 10,
        "dataset_total_rows": len(combined_df),
        "months_covered": len(processed_files),
    }
    tracker.log_params(params)

    print("Training model...")
    model = train_model(
        features=X_train,
        target=y_train,
        algorithm=params["algorithm"],
        n_estimators=params["n_estimators"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
    )

    # Evaluate
    val_metrics = evaluate_model(model, X_val, y_val)
    test_metrics = evaluate_model(model, X_test, y_test)

    all_metrics = {
        "val_mae": val_metrics["mae"],
        "val_rmse": val_metrics["rmse"],
        "val_r2": val_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }
    tracker.log_metrics(all_metrics)

    print("\n--- Model Evaluation Results ---")
    for k, v in all_metrics.items():
        print(f"{k}: {v:.4f}")

    # Decision Gate
    ACCURACY_THRESHOLD_MAE = 5.0  # Acceptable threshold in minutes
    print(f"\nChecking test MAE: {test_metrics['mae']:.2f} <= {ACCURACY_THRESHOLD_MAE}")

    if test_metrics["mae"] <= ACCURACY_THRESHOLD_MAE:
        print("Acceptance criteria met! Uploading Champion model to Azure Blob Storage...")
        
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            joblib.dump(model, tmp_path)

        try:
            remote_uri = storage.save_file(
                source_path=tmp_path,
                category="models/champion",
                filename="trip_duration_model.joblib",
            )
            print(f"-> Successfully deployed Champion to: {remote_uri}")
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
    else:
        print("Model did not meet accuracy threshold. Skipped Champion deployment.")

tracker.end_run()
print("\nProcess finished successfully.")