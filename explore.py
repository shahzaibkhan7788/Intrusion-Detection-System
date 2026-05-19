"""Comprehensive data exploration for CIC-IDS2017 CSV files.

Run: python explore.py

This script analyzes all CSV files in ./data and provides detailed statistics:
- File size and basic info
- Label distribution (attacks vs normal, with percentages)
- Types of attacks and their counts
- Duplicated rows count
- Rows with missing values
- Constant or all-zero columns
- Data distribution insights
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"
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


def analyze_file(csv_path: Path) -> Dict:
    """Analyze a single CSV file and return comprehensive statistics."""
    # Get file size
    file_size_mb = csv_path.stat().st_size / (1024 * 1024)

    # Read raw data
    raw_df = pd.read_csv(
        csv_path,
        low_memory=False,
        na_values=["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan", ""],
        keep_default_na=True,
    )

    # Normalize column names
    raw_df.rename(columns=lambda c: c.strip(), inplace=True)

    label_col = find_label_column(list(raw_df.columns))

    # Basic info
    total_rows = len(raw_df)
    total_cols = len(raw_df.columns)

    # Label distribution
    label_counts = raw_df[label_col].value_counts(dropna=False)
    if total_rows > 0:
        label_percentages_series = (label_counts / total_rows * 100).apply(round, args=(2,))
    else:
        label_percentages_series = pd.Series(0.0, index=label_counts.index)
    label_percentages = dict(label_percentages_series)

    # Identify normal vs attacks
    normal_labels = ['BENIGN', 'benign', 'Benign']
    normal_count = sum(label_counts.get(label, 0) for label in normal_labels)
    attack_count = total_rows - normal_count

    # Attack types and their distributions
    attack_labels = {k: v for k, v in label_counts.items() if k not in normal_labels}
    if total_rows > 0:
        attack_percentages = {k: round(v / total_rows * 100, 2) for k, v in attack_labels.items()}
    else:
        attack_percentages = {k: 0.0 for k in attack_labels}

    # Duplicates
    duplicates_count = raw_df.duplicated().sum()

    # Missing values
    missing_rows = raw_df.isnull().any(axis=1).sum()
    if total_rows > 0:
        missing_percentage = round(missing_rows / total_rows * 100, 2)
    else:
        missing_percentage = 0.0

    # Constant columns (excluding label)
    feature_cols = [c for c in raw_df.columns if c != label_col and c not in ALWAYS_DROP]
    constant_cols = []
    all_zero_cols = []

    for col in feature_cols:
        unique_vals = raw_df[col].nunique(dropna=False)
        if unique_vals <= 1:
            constant_cols.append(col)
        elif (raw_df[col] == 0).all():
            all_zero_cols.append(col)

    # Data types distribution
    dtypes_count = raw_df.dtypes.value_counts().to_dict()

    # Memory usage
    memory_usage_mb = raw_df.memory_usage(deep=True).sum() / (1024 * 1024)

    analysis = {
        "total_rows": total_rows,
        "total_columns": total_cols,
        "data_types": dtypes_count,
        "label_distribution": {
            "normal_count": int(normal_count),
            "normal_percentage": round(normal_count / total_rows * 100, 2) if total_rows > 0 else 0,
            "attack_count": int(attack_count),
            "attack_percentage": round(attack_count / total_rows * 100, 2) if total_rows > 0 else 0,
            "attack_types": dict(attack_labels),
            "attack_percentages": attack_percentages,
            "all_labels": dict(label_counts),
            "all_percentages": dict(label_percentages)
        },
        "duplicates": {
            "count": int(duplicates_count),
            "percentage": round(duplicates_count / total_rows * 100, 2) if total_rows > 0 else 0
        },
        "missing_data": {
            "rows_with_missing": int(missing_rows),
            "percentage": missing_percentage
        },
        "constant_columns": {
            "constant_cols": constant_cols,
            "all_zero_cols": all_zero_cols,
            "total_to_drop": len(constant_cols) + len(all_zero_cols)
        },
        "columns_to_drop_always": list(ALWAYS_DROP.intersection(set(raw_df.columns)))
    }

    return analysis


def main() -> None:
    analyses: List[Dict] = []

    print("Starting comprehensive data exploration...\n")

    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            analysis = analyze_file(csv_path)
            analyses.append(analysis)

            print(f"📊 Analysis for {csv_path.name}:")
          
            print(f"   Rows: {analysis['total_rows']:,}, Columns: {analysis['total_columns']}")
           
            print(f"   Normal: {analysis['label_distribution']['normal_count']:,} ({analysis['label_distribution']['normal_percentage']}%)")
            print(f"   Attacks: {analysis['label_distribution']['attack_count']:,} ({analysis['label_distribution']['attack_percentage']}%)")
            print(f"   Attack Types: {len(analysis['label_distribution']['attack_types'])}")
            for attack, count in analysis['label_distribution']['attack_types'].items():
                pct = analysis['label_distribution']['attack_percentages'][attack]
                print(f"     - {attack}: {count:,} ({pct}%)")
            print(f"   Duplicates: {analysis['duplicates']['count']:,} ({analysis['duplicates']['percentage']}%)")
            print(f"   Missing Rows: {analysis['missing_data']['rows_with_missing']:,} ({analysis['missing_data']['percentage']}%)")
            print(f"   Constant Columns: {len(analysis['constant_columns']['constant_cols'])}")
            print(f"   All-Zero Columns: {len(analysis['constant_columns']['all_zero_cols'])}")
            print(f"   Always Drop: {analysis['columns_to_drop_always']}")
            print()

        except Exception as exc:
            error_analysis = {
                "file_name": csv_path.name,
                "error": str(exc)
            }
            analyses.append(error_analysis)
            print(f"❌ Failed to analyze {csv_path.name}: {exc}\n")

    # Save detailed repor


    # Print overall summary
    print("\n📈 OVERALL SUMMARY:")
    total_files = len(analyses)
    successful_files = sum(1 for a in analyses if "error" not in a)
    total_size = sum(a.get("file_size_mb", 0) for a in analyses if "error" not in a)
    total_rows = sum(a.get("total_rows", 0) for a in analyses if "error" not in a)
    total_normal = sum(a.get("label_distribution", {}).get("normal_count", 0) for a in analyses if "error" not in a)
    total_attacks = sum(a.get("label_distribution", {}).get("attack_count", 0) for a in analyses if "error" not in a)

    print(f"   Files analyzed: {successful_files}/{total_files}")
    print(f"   Total size: {total_size:.2f} MB")
    print(f"   Total rows: {total_rows:,}")
    print(f"   Total normal: {total_normal:,} ({total_normal/total_rows*100:.2f}%)" if total_rows > 0 else "   Total normal: 0")
    print(f"   Total attacks: {total_attacks:,} ({total_attacks/total_rows*100:.2f}%)" if total_rows > 0 else "   Total attacks: 0")


if __name__ == "__main__":
    main()