"""Comprehensive data cleaning and preparation for CIC-IDS2017 CSV files.

Run:
    python cic_prepare.py

What it does:
- Reads all CSV files from ./data
- Cleans each file:
    * Normalizes column names
    * Removes duplicates
    * Removes rows with NaN / inf values
    * Removes Flow ID and Timestamp
    * Tries to convert every feature column to numeric
    * Drops only truly non-numeric columns
    * Removes constant / all-zero feature columns
- Keeps all attack classes and all label values
- Standardizes all output files to the common feature columns across files
- Saves cleaned files to ./filter-data
- Generates a detailed JSON report in ./Save_history
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "filter-data"
REPORTS_DIR = BASE_DIR / "Save_history"

ALWAYS_DROP = {"Flow ID", "Timestamp"}


def normalize_columns(columns: List[str]) -> List[str]:
    return [
        str(col).replace("\ufeff", "").strip()
        for col in columns
    ]


def find_label_column(columns: List[str]) -> str:
    """Return the label column name (case-insensitive, stripped)."""
    cleaned = {col.strip(): col for col in columns}
    for candidate in ("Label", "label", "LABEL"):
        if candidate in cleaned:
            return cleaned[candidate]
    raise ValueError("No Label column found")


def clean_file_advanced(csv_path: Path, label_col_name: str) -> Tuple[pd.DataFrame, Dict]:
    """Clean one CSV while preserving the final label column."""
    raw_df = pd.read_csv(
        csv_path,
        low_memory=False,
        na_values=["Infinity", "-Infinity", "inf", "-inf", "NaN", "nan", ""],
        keep_default_na=True,
    )

    raw_df.columns = normalize_columns(list(raw_df.columns))

    if label_col_name not in raw_df.columns:
        raise ValueError(f"Label column '{label_col_name}' not found in {csv_path.name}")

    df = raw_df.copy()

    drop_cols = [col for col in df.columns if col in ALWAYS_DROP]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")

    df.replace([float("inf"), float("-inf"), "Infinity", "-Infinity", "inf", "-inf"], pd.NA, inplace=True)

    before_duplicates = len(df)
    df.drop_duplicates(inplace=True)
    duplicates_removed = before_duplicates - len(df)

    labels = df[label_col_name].copy()
    labels = labels.astype(str).str.strip().str.replace("\ufffd", "-", regex=False)

    features = df.drop(columns=[label_col_name]).copy()

    converted_features = pd.DataFrame(index=features.index)
    dropped_non_numeric = []

    for col in features.columns:
        converted = pd.to_numeric(features[col], errors="coerce")
        if converted.notna().any():
            converted_features[col] = converted
        else:
            dropped_non_numeric.append(col)

    df = pd.concat([converted_features, labels.rename(label_col_name)], axis=1)

    before_missing = len(df)
    df.dropna(inplace=True)
    rows_dropped_missing = before_missing - len(df)

    feature_cols = [col for col in df.columns if col != label_col_name]
    constant_cols = []
    zero_cols = []

    for col in feature_cols:
        unique_vals = df[col].nunique(dropna=False)
        if unique_vals <= 1:
            constant_cols.append(col)
        elif (df[col] == 0).all():
            zero_cols.append(col)

    drop_const_zero = sorted(set(constant_cols + zero_cols))
    df.drop(columns=drop_const_zero, inplace=True)

    metadata = {
        "rows_initial": int(len(raw_df)),
        "rows_final": int(len(df)),
        "rows_dropped_missing": int(rows_dropped_missing),
        "duplicates_removed": int(duplicates_removed),
        "always_dropped_columns": drop_cols,
        "non_numeric_dropped": dropped_non_numeric,
        "constant_or_zero_dropped": drop_const_zero,
        "label_column": label_col_name,
    }

    return df.reset_index(drop=True), metadata


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    print("Starting comprehensive data cleaning...\n")

    file_stats: List[Dict] = []
    cleaned_dfs: Dict[str, pd.DataFrame] = {}
    file_metadata: Dict[str, Dict] = {}
    label_col = None

    print("Step 1: Cleaning individual files...")
    for csv_path in sorted(DATA_DIR.glob("*.csv")):
        try:
            temp_df = pd.read_csv(csv_path, low_memory=False, nrows=1)
            temp_df.columns = normalize_columns(list(temp_df.columns))

            if label_col is None:
                label_col = find_label_column(list(temp_df.columns))

            current_label_col = find_label_column(list(temp_df.columns))
            df_clean, metadata = clean_file_advanced(csv_path, current_label_col)

            cleaned_dfs[csv_path.name] = df_clean
            file_metadata[csv_path.name] = metadata

            label_dist = df_clean[label_col].value_counts().to_dict()
            total_rows = len(df_clean)

            file_info = {
                "file_name": csv_path.name,
                "rows": int(total_rows),
                "columns": int(len(df_clean.columns)),
                "label_dist": {str(k): int(v) for k, v in label_dist.items()},
                "label_dist_pct": {
                    str(k): round(v / total_rows * 100, 4) for k, v in label_dist.items()
                } if total_rows > 0 else {},
                "metadata": metadata,
            }
            file_stats.append(file_info)

            print(f"✓ {csv_path.name}: {total_rows:,} rows, {len(df_clean.columns)} columns")

        except Exception as exc:
            print(f"✗ Error processing {csv_path.name}: {exc}")

    if not cleaned_dfs:
        raise RuntimeError("No files were cleaned successfully.")

    print("\nStep 2: Computing global label distribution...")
    normal_labels = {"BENIGN", "benign", "Benign"}
    global_label_counts: Dict[str, int] = {}
    total_global_rows = 0

    for df in cleaned_dfs.values():
        for label, count in df[label_col].value_counts().items():
            global_label_counts[label] = global_label_counts.get(label, 0) + int(count)
            total_global_rows += int(count)

    global_label_pct = {
        str(label): round(count / total_global_rows * 100, 4)
        for label, count in global_label_counts.items()
    } if total_global_rows > 0 else {}

    print(f"✓ Found {len(global_label_counts)} total label classes")

    print("\nStep 3: Standardizing columns across all files...")
    common_feature_columns = None
    per_file_feature_columns = {}

    for file_name, df in cleaned_dfs.items():
        feature_set = set(df.columns) - {label_col}
        per_file_feature_columns[file_name] = sorted(feature_set)
        if common_feature_columns is None:
            common_feature_columns = feature_set
        else:
            common_feature_columns = common_feature_columns.intersection(feature_set)

    if common_feature_columns is None:
        raise RuntimeError("Failed to determine common feature columns.")

    common_feature_columns = sorted(common_feature_columns)
    standard_columns = common_feature_columns + [label_col]

    filtered_dfs: Dict[str, pd.DataFrame] = {}
    filtering_stats: Dict[str, Dict] = {}

    for file_name, df in cleaned_dfs.items():
        before_cols = len(df.columns)
        df_standard = df[standard_columns].copy()
        after_cols = len(df_standard.columns)

        filtered_dfs[file_name] = df_standard

        filtering_stats[file_name] = {
            "initial_rows": int(len(df)),
            "final_rows": int(len(df_standard)),
            "initial_columns": int(before_cols),
            "final_columns": int(after_cols),
            "columns_removed_for_standardization": int(before_cols - after_cols),
        }

        print(f"✓ {file_name}: {len(df_standard):,} rows, {after_cols} columns")

    print("\nStep 4: Saving preprocessed files...")
    saved_files = []

    for file_name, df in filtered_dfs.items():
        output_path = OUTPUT_DIR / file_name
        df.to_csv(output_path, index=False)

        saved_files.append(
            {
                "input": file_name,
                "output": str(output_path.relative_to(BASE_DIR)),
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
            }
        )
        print(f"✓ Saved: {output_path.name}")

    print("\nStep 5: Generating report...")
    final_label_counts: Dict[str, int] = {}
    for df in filtered_dfs.values():
        for label, count in df[label_col].value_counts().items():
            final_label_counts[label] = final_label_counts.get(label, 0) + int(count)

    final_total = sum(final_label_counts.values())
    final_label_pct = {
        str(label): round(count / final_total * 100, 4)
        for label, count in final_label_counts.items()
    } if final_total > 0 else {}

    normal_count_after = sum(
        count for label, count in final_label_counts.items() if label in normal_labels
    )
    attack_count_after = final_total - normal_count_after

    report = {
        "summary": {
            "total_files": int(len(cleaned_dfs)),
            "files_processed": int(len(filtered_dfs)),
        },
        "global_statistics": {
            "total_rows_before_filtering": int(total_global_rows),
            "total_rows_after_filtering": int(final_total),
            "label_distribution_before": {str(k): int(v) for k, v in global_label_counts.items()},
            "label_distribution_pct_before": global_label_pct,
            "label_distribution_after": {str(k): int(v) for k, v in final_label_counts.items()},
            "label_distribution_pct_after": final_label_pct,
            "attacks_removed": {
                "count": 0,
                "attack_types": [],
                "threshold": "no attack classes removed",
            },
            "normal_samples": {
                "count": int(normal_count_after),
                "percentage": round(normal_count_after / final_total * 100, 4) if final_total > 0 else 0,
            },
            "attack_samples": {
                "count": int(attack_count_after),
                "percentage": round(attack_count_after / final_total * 100, 4) if final_total > 0 else 0,
            },
        },
        "column_standardization": {
            "common_feature_column_count": int(len(common_feature_columns)),
            "common_feature_columns": common_feature_columns,
            "standard_columns": standard_columns,
            "always_dropped_columns": sorted(ALWAYS_DROP),
            "feature_column_rule": "convert features to numeric where possible; drop only columns that stay fully non-numeric; then keep common feature columns across files",
        },
        "file_details": file_stats,
        "filtering_results": filtering_stats,
        "saved_files": saved_files,
    }

    report_path = REPORTS_DIR / "data_preparation_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print(f"✓ Report saved: {report_path}\n")

    print("=" * 80)
    print("DATA PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nFinal statistics:")
    print(f"  Files processed: {len(filtered_dfs)}")
    print(f"  Total rows: {final_total:,}")
    print(f"  Common feature columns: {len(common_feature_columns)}")
    print(f"  Total columns including label: {len(standard_columns)}")
    print(f"  Normal samples: {normal_count_after:,} ({report['global_statistics']['normal_samples']['percentage']}%)")
    print(f"  Attack samples: {attack_count_after:,} ({report['global_statistics']['attack_samples']['percentage']}%)")

    print("\nAttack distribution (all attacks kept):")
    for label in sorted(final_label_counts.keys()):
        if label not in normal_labels:
            count = final_label_counts[label]
            pct = final_label_pct[str(label)]
            print(f"  - {label}: {count:,} ({pct}%)")

    print(f"\nOutput location: {OUTPUT_DIR}")
    print(f"Report location: {report_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
