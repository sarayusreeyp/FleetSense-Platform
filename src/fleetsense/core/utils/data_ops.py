from __future__ import annotations

import pandas as pd


def normalize_datetime_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Normalize Green taxi and FHV datetime column names to standard TLC columns."""
    df = dataframe
    col_mapping = {
        "lpep_pickup_datetime": "tpep_pickup_datetime",
        "lpep_dropoff_datetime": "tpep_dropoff_datetime",
        "pickup_datetime": "tpep_pickup_datetime",
        "dropoff_datetime": "tpep_dropoff_datetime",
    }
    for old_col, new_col in col_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            if df is dataframe:
                df = dataframe.copy()
            df[new_col] = df[old_col]
    return df
