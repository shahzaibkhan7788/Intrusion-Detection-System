"""Utility to summarize and clean CIC-IDS2017 CSV files.

Run: python cic_prepare.py

What it does per CSV in ./data:
- Prints a compact summary (row/col counts, label distribution, dropped items).
- Cleans and writes a preprocessed copy to ./data_preprocessed with:
    * stripped column names
    * duplicates removed
    * rows with NaN/inf/blank removed
    * non-numeric feature columns removed (label kept)
    * explicit drop of Flow ID, Src IP, Dst IP, Timestamp
    * constant or all-zero feature columns removed
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
OUTPUT_DIR = Path(__file__).parent / "data_preprocessed"
REPORTS_DIR = Path(__file__).parent / "reports"

# Columns to drop regardless of content (identifiers / time).
ALWAYS_DROP = {"Flow ID", "Src IP", "Dst IP", "Timestamp"}


def find_label_column(columns: List[str]) -> str:
    """Return the label column name (case-insensitive, stripped)."""
    cleaned = {col.strip(): col for col in columns}
    for candidate in ("Label", "label", "LABEL"):
        if candidate in cleaned:
            return cleaned[candidate]
    raise ValueError("No Label column found")


def summarize_and_clean(csv_path: Path) -> Dict:
    raw_df = pd.read_csv(
        csv_path,
        low_memory=False,
        na_values=["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan", ""],
        keep_default_na=True,
    )

    # Normalize column names (strip spaces/newlines).
    raw_df.rename(columns=lambda c: c.strip(), inplace=True)

    label_col = find_label_column(list(raw_df.columns))

    summary: Dict = {
        "file": csv_path.name,
        "rows_raw": int(len(raw_df)),
        "cols_raw": int(len(raw_df.columns)),
        "label_counts_raw": raw_df[label_col].value_counts(dropna=False).to_dict(),
    }

    df = raw_df.copy()

    # Remove identifier/time columns.
    drop_cols = [c for c in df.columns if c.strip() in ALWAYS_DROP]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    # Replace infinity markers with NaN then drop duplicate rows.
    df.replace([np.inf, -np.inf, "Infinity", "-Infinity", "inf", "-inf"], np.nan, inplace=True)
    before_dupes = len(df)
    df.drop_duplicates(inplace=True)
    summary["duplicates_removed"] = int(before_dupes - len(df))

    # Split out label to avoid unintended conversions.
    labels = df[label_col]
    features = df.drop(columns=[label_col])

    # Convert features to numeric, coerce errors to NaN to flag non-numeric entries.
    features = features.apply(pd.to_numeric, errors="coerce")

    # Identify and drop columns that are entirely non-numeric after coercion.
    non_numeric_cols = [c for c in features.columns if features[c].isna().all()]
    features.drop(columns=non_numeric_cols, inplace=True)
    summary["dropped_non_numeric_cols"] = non_numeric_cols

    # Reattach label for downstream operations.
    df = pd.concat([features, labels], axis=1)

    # Drop rows with any missing values (includes coerced NaNs and blank strings).
    before_missing = len(df)
    df.dropna(inplace=True)
    summary["rows_dropped_missing"] = int(before_missing - len(df))

    # Remove constant or all-zero feature columns (label is preserved).
    feature_cols = [c for c in df.columns if c != label_col]
    constant_cols = [c for c in feature_cols if df[c].nunique(dropna=False) <= 1]
    zero_cols = [c for c in feature_cols if (df[c] == 0).all()]
    drop_const_zero = sorted(set(constant_cols + zero_cols))
    df.drop(columns=drop_const_zero, inplace=True)
    summary["dropped_constant_zero_cols"] = drop_const_zero

    summary["rows_clean"] = int(len(df))
    summary["cols_clean"] = int(len(df.columns))
    summary["label_counts_clean"] = df[label_col].value_counts(dropna=False).to_dict()

    # Output cleaned file.
    OUTPUT_DIR.mkdir(exist_ok=True)
    cleaned_path = OUTPUT_DIR / csv_path.name
    df.to_csv(cleaned_path, index=False)
    summary["cleaned_path"] = str(cleaned_path.relative_to(Path(__file__).parent))

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    summaries: List[Dict] = []

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            info = summarize_and_clean(csv_path)
            summaries.append(info)
            print(
                f"Processed {csv_path.name}: {info['rows_raw']} -> {info['rows_clean']} rows, "
                f"labels {info['label_counts_clean']}"
            )
        except Exception as exc:  # noqa: BLE001
            msg = f"Failed on {csv_path.name}: {exc}"
            summaries.append({"file": csv_path.name, "error": str(exc)})
            print(msg)

    report_path = REPORTS_DIR / "cic_preprocessing_summary.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"Summary written to {report_path}")


if __name__ == "__main__":
    main()
