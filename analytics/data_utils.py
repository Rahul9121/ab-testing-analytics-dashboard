from __future__ import annotations

from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

TRUE_VALUES = {"1", "true", "t", "yes", "y", "converted", "success"}
FALSE_VALUES = {"0", "false", "f", "no", "n", "not_converted", "fail"}


def _extract_series(df: pd.DataFrame, column_name: str, role: str) -> pd.Series:
    if column_name not in df.columns:
        raise ValueError(f"Selected {role} column '{column_name}' is not in the dataset.")

    selected = df.loc[:, column_name]
    if isinstance(selected, pd.DataFrame):
        selected = selected.iloc[:, 0]

    if not isinstance(selected, pd.Series):
        raise ValueError(f"Could not read {role} column '{column_name}' as a valid series.")
    return selected.copy()


def _normalize_conversion_column(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(int)

    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        non_null = numeric.dropna()
        if non_null.empty:
            raise ValueError("Conversion column has no valid numeric values.")
        if not non_null.isin([0, 1]).all():
            raise ValueError("Conversion column must be binary (0/1, true/false, yes/no).")
        return numeric.fillna(0).astype(int)

    normalized = series.astype(str).str.strip().str.lower()
    mapped = normalized.map(lambda x: 1 if x in TRUE_VALUES else (0 if x in FALSE_VALUES else np.nan))
    if mapped.isna().any():
        invalid_values = sorted(normalized[mapped.isna()].unique().tolist())[:5]
        raise ValueError(
            "Could not parse some conversion values as binary: "
            f"{invalid_values}. Use 0/1, true/false, or yes/no."
        )
    return mapped.astype(int)


def prepare_experiment_dataframe(
    df: pd.DataFrame,
    variant_col: str,
    conversion_col: str,
    date_col: Optional[str] = None,
    segment_col: Optional[str] = None,
) -> pd.DataFrame:
    if variant_col == conversion_col:
        raise ValueError("Variant/group column and conversion column must be different.")

    prepared = pd.DataFrame(
        {
            "variant_raw": _extract_series(df, variant_col, "variant/group"),
            "conversion_raw": _extract_series(df, conversion_col, "conversion"),
        }
    )

    if date_col:
        prepared["date_raw"] = _extract_series(df, date_col, "time")
    if segment_col:
        prepared["segment_raw"] = _extract_series(df, segment_col, "segment")

    prepared = prepared.dropna(subset=["variant_raw", "conversion_raw"])
    prepared["variant"] = prepared["variant_raw"].astype(str).str.strip()
    prepared = prepared[prepared["variant"] != ""]
    prepared["converted"] = _normalize_conversion_column(prepared["conversion_raw"])

    if date_col:
        parsed_time = pd.to_datetime(prepared["date_raw"], errors="coerce")
        prepared["event_time"] = parsed_time
        prepared = prepared.dropna(subset=["event_time"])

    if segment_col:
        prepared["segment"] = prepared["segment_raw"].astype(str).str.strip().replace("", "Unknown")

    if prepared.empty:
        raise ValueError("No valid rows available after cleaning and parsing the selected columns.")

    output_cols = ["variant", "converted"]
    if date_col:
        output_cols.append("event_time")
    if segment_col:
        output_cols.append("segment")
    return prepared[output_cols].reset_index(drop=True)


def generate_synthetic_experiment(
    control_size: int,
    treatment_size: int,
    control_rate: float,
    treatment_lift: float,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    treatment_rate = min(max(control_rate * (1 + treatment_lift), 0.0001), 0.9999)
    control_converted = rng.binomial(1, control_rate, size=control_size)
    treatment_converted = rng.binomial(1, treatment_rate, size=treatment_size)

    total_rows = control_size + treatment_size
    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - timedelta(days=29)
    date_choices = pd.date_range(start=start_date, end=end_date, freq="D")

    event_times = rng.choice(date_choices, size=total_rows, replace=True)
    countries = rng.choice(["US", "IN", "UK", "DE"], size=total_rows, replace=True, p=[0.45, 0.30, 0.18, 0.07])
    devices = rng.choice(["mobile", "desktop", "tablet"], size=total_rows, replace=True, p=[0.62, 0.32, 0.06])

    control_df = pd.DataFrame(
        {
            "user_id": np.arange(1, control_size + 1),
            "group": "control",
            "converted": control_converted,
        }
    )
    treatment_df = pd.DataFrame(
        {
            "user_id": np.arange(control_size + 1, total_rows + 1),
            "group": "treatment",
            "converted": treatment_converted,
        }
    )

    synthetic = pd.concat([control_df, treatment_df], ignore_index=True)
    synthetic["event_time"] = pd.to_datetime(event_times)
    synthetic["country"] = countries
    synthetic["device"] = devices
    return synthetic.sample(frac=1, random_state=seed).reset_index(drop=True)

