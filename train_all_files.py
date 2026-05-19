"""
Iterative TabNet attack-pruning experiment runner for CIC-IDS2017.

Run:
    python train_all_files.py

What it does:
- Builds the combined dataset from ./filter-data if needed
- Starts with all attack classes
- Trains TabNet with self-supervised pretraining + supervised fine-tuning
- Evaluates on the test split after each run
- Scores which attack classes are harming macro F1 the most
- Tries different attack-drop combinations across 10 runs
- Saves each run into its own reports/weights folder so nothing is overwritten
- Waits 60 minutes between runs to let the GPU cool
- Writes a final experiment summary with the best attack combination
"""


from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.callbacks import Callback
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from custom_metrics import (
    F1Macro,
    PrecisionMacro,
    RecallMacro,
)


BASE = Path(__file__).parent
DATA_DIR = BASE / "filter-data"
EXPERIMENT_ROOT = BASE / "attack_pruning_search"
RUNS_DIR = EXPERIMENT_ROOT / "runs"
SUMMARY_JSON = EXPERIMENT_ROOT / "experiment_summary.json"
SUMMARY_CSV = EXPERIMENT_ROOT / "experiment_summary.csv"

SEED = 7
LABEL_COL = "Label"
DATASET_NAME = "All"
TOTAL_RUNS = 10
COOLDOWN_SECONDS = 60 * 60
MIN_ATTACK_CLASSES_TO_KEEP = 7
TARGET_MACRO_F1 = 0.80
NORMAL_LABELS = {"Benign", "BENIGN", "benign"}

np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class EpochCSVLogger(Callback):
    """Append per-epoch logs to CSV in real time."""

    def __init__(self, path: Path):
        self.path = path
        if self.path.exists():
            self.path.unlink()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.header_written = False

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        row = {"epoch": epoch}
        row.update(logs)
        with self.path.open("a", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=row.keys())
            if not self.header_written:
                writer.writeheader()
                self.header_written = True
            writer.writerow(row)


def convert_to_serializable(obj: Any) -> Any:
    if hasattr(obj, "history") and isinstance(getattr(obj, "history"), dict):
        return convert_to_serializable(obj.history)
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def normalize_history(history: Any) -> Dict[str, Any]:
    """Convert TabNet History objects or dict-like histories to a plain dict."""
    if history is None:
        return {}
    if isinstance(history, dict):
        return history
    if hasattr(history, "history") and isinstance(getattr(history, "history"), dict):
        return dict(history.history)
    try:
        return dict(history)
    except Exception:
        return {}


def relative_to_base(path: Path) -> str:
    try:
        return str(path.relative_to(BASE))
    except ValueError:
        return str(path)


def zipped_model_path(model_path: Path) -> Path:
    return model_path if model_path.suffix == ".zip" else Path(f"{model_path}.zip")


def attack_labels_from_series(labels: pd.Series) -> List[str]:
    return sorted(label for label in labels.astype(str).unique() if label not in NORMAL_LABELS)


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for value in values:
        if value not in seen:
            ordered.append(value)
            seen.add(value)
    return ordered


def stratified_time_split(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return boolean masks for train/val/test while preserving order within each class.
    Rare classes are kept; if a class is too small to appear in every split, all rows
    are still retained in the earliest possible splits.
    """
    n_rows = len(df)
    train_mask = np.zeros(n_rows, dtype=bool)
    val_mask = np.zeros(n_rows, dtype=bool)
    test_mask = np.zeros(n_rows, dtype=bool)

    for label in df[label_col].unique():
        idx = df.index[df[label_col] == label]
        count = len(idx)
        if count == 0:
            continue

        n_train = max(1, int(round(count * train_ratio)))
        n_val = max(1 if count >= 3 else 0, int(round(count * val_ratio)))
        n_train = min(n_train, count)
        n_val = min(n_val, max(0, count - n_train))
        n_test = count - n_train - n_val

        if n_test == 0 and count >= 3:
            if n_train > 1:
                n_train -= 1
                n_test += 1
            elif n_val > 1:
                n_val -= 1
                n_test += 1

        train_idx = idx[:n_train]
        val_idx = idx[n_train : n_train + n_val]
        test_idx = idx[n_train + n_val :]

        train_mask[train_idx] = True
        val_mask[val_idx] = True
        test_mask[test_idx] = True

    return train_mask, val_mask, test_mask


def drop_non_numeric_features(
    df: pd.DataFrame,
    label_col: str = LABEL_COL,
) -> Tuple[pd.DataFrame, List[str]]:
    feature_cols = [column for column in df.columns if column != label_col]
    dropped_columns = [
        column
        for column in feature_cols
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if dropped_columns:
        df = df.drop(columns=dropped_columns).copy()
    return df, dropped_columns


def prepare_xy(df: pd.DataFrame, label_col: str = LABEL_COL) -> Tuple[np.ndarray, np.ndarray]:
    y = df[label_col].to_numpy()
    X = df.drop(columns=[label_col]).to_numpy(dtype=np.float32)
    return X, y


def save_history(name: str, hist: Any, phase: str, report_dir: Path) -> Tuple[Path | None, Path]:
    """Save history as raw JSON and normalized CSV."""
    report_dir.mkdir(parents=True, exist_ok=True)
    history_dict = normalize_history(hist)

    raw_path = report_dir / f"{name}_{phase}_history_raw.json"
    with raw_path.open("w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(history_dict), handle, indent=2)

    if not history_dict:
        return None, raw_path

    csv_path: Path | None = None
    if isinstance(history_dict, dict):
        max_len = max(
            len(value) if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else 1
            for value in history_dict.values()
        )
        normalized = {}
        for key, value in history_dict.items():
            if not hasattr(value, "__len__") or isinstance(value, (str, bytes)):
                normalized[key] = [value] * max_len
            else:
                values = list(value)
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                normalized[key] = values
        history_df = pd.DataFrame(normalized)
    else:
        history_df = pd.DataFrame(history_dict if isinstance(history_dict, list) else [history_dict])

    if not history_df.empty:
        csv_path = report_dir / f"{name}_{phase}_history.csv"
        history_df.to_csv(csv_path, index=False)

    return csv_path, raw_path


def history_to_frame(history: Any) -> pd.DataFrame:
    history_dict = normalize_history(history)
    if not history_dict:
        return pd.DataFrame()

    max_len = max(
        len(value) if hasattr(value, "__len__") and not isinstance(value, (str, bytes)) else 1
        for value in history_dict.values()
    )
    normalized = {}
    for key, value in history_dict.items():
        if not hasattr(value, "__len__") or isinstance(value, (str, bytes)):
            normalized[key] = [value] * max_len
        else:
            values = list(value)
            if len(values) < max_len:
                values = values + [None] * (max_len - len(values))
            normalized[key] = values
    history_df = pd.DataFrame(normalized)
    if not history_df.empty and "epoch" not in history_df.columns:
        history_df.insert(0, "epoch", range(len(history_df)))
    return history_df


def summarize_history(history_df: pd.DataFrame, phase: str) -> Dict[str, Any]:
    if history_df.empty:
        return {"phase": phase, "available": False}

    summary: Dict[str, Any] = {
        "phase": phase,
        "available": True,
        "epochs_recorded": int(len(history_df)),
    }

    if "loss" in history_df.columns:
        summary["last_loss"] = float(history_df["loss"].iloc[-1])
        summary["best_loss"] = float(history_df["loss"].min())
        summary["best_loss_epoch"] = int(history_df["loss"].idxmin())

    priority_cols = [
        "valid_f1_macro",
        "valid_recall_macro",
        "valid_precision_macro",
        "train_f1_macro",
        "val_0_unsup_loss_numpy",
    ]
    for column in priority_cols:
        if column in history_df.columns:
            if "loss" in column.lower():
                best_idx = int(history_df[column].idxmin())
                summary["selection_metric"] = column
                summary["selection_mode"] = "min"
                summary["selection_value"] = float(history_df.loc[best_idx, column])
                summary["selection_epoch"] = int(best_idx)
            else:
                best_idx = int(history_df[column].idxmax())
                summary["selection_metric"] = column
                summary["selection_mode"] = "max"
                summary["selection_value"] = float(history_df.loc[best_idx, column])
                summary["selection_epoch"] = int(best_idx)
            break

    return summary


def save_feature_importances(
    name: str,
    clf: TabNetClassifier,
    feature_names: List[str],
    report_dir: Path,
) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    importance = np.asarray(clf.feature_importances_, dtype=float)
    total_importance = float(importance.sum()) if len(importance) else 0.0

    df = pd.DataFrame(
        {
            "feature_idx": range(len(feature_names)),
            "feature_name": feature_names,
            "importance": importance,
        }
    ).sort_values("importance", ascending=False).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    df["importance_pct"] = (df["importance"] / total_importance * 100) if total_importance > 0 else 0.0
    df["cumulative_importance_pct"] = df["importance_pct"].cumsum()

    out_path = report_dir / f"{name}_feature_importance.csv"
    df.to_csv(out_path, index=False)
    return out_path


def save_confusion_matrix_csv(name: str, cm: np.ndarray, classes: List[str], report_dir: Path) -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    cm_df = pd.DataFrame(cm, index=classes, columns=classes)
    cm_df.index.name = "true_label"
    out_path = report_dir / f"{name}_confusion_matrix.csv"
    cm_df.to_csv(out_path)
    return out_path


def save_top_confusions_csv(
    name: str,
    cm: np.ndarray,
    classes: List[str],
    report_dir: Path,
) -> Tuple[Path, pd.DataFrame]:
    rows = []
    for true_idx, true_label in enumerate(classes):
        for pred_idx, pred_label in enumerate(classes):
            count = int(cm[true_idx, pred_idx])
            if true_idx != pred_idx and count > 0:
                rows.append(
                    {
                        "true_label": true_label,
                        "predicted_label": pred_label,
                        "count": count,
                    }
                )

    confusions_df = (
        pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
        if rows
        else pd.DataFrame(columns=["true_label", "predicted_label", "count"])
    )
    out_path = report_dir / f"{name}_top_confusions.csv"
    confusions_df.to_csv(out_path, index=False)
    return out_path, confusions_df


def save_class_metrics_csv(
    name: str,
    report: Dict[str, Any],
    cm: np.ndarray,
    classes: List[str],
    report_dir: Path,
) -> Tuple[Path, pd.DataFrame]:
    rows = []
    for idx, class_name in enumerate(classes):
        class_report = report.get(class_name, {})
        support = int(class_report.get("support", 0))
        true_positive = int(cm[idx, idx])
        false_negative = int(cm[idx, :].sum() - true_positive)
        false_positive = int(cm[:, idx].sum() - true_positive)

        row_without_diag = cm[idx, :].copy()
        row_without_diag[idx] = 0
        top_confusion_idx = int(row_without_diag.argmax()) if row_without_diag.sum() > 0 else -1

        rows.append(
            {
                "class_name": class_name,
                "precision": float(class_report.get("precision", 0.0)),
                "recall": float(class_report.get("recall", 0.0)),
                "f1_score": float(class_report.get("f1-score", 0.0)),
                "support": support,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "predicted_as_class": int(cm[:, idx].sum()),
                "class_accuracy": float(true_positive / support) if support > 0 else 0.0,
                "top_confused_with": classes[top_confusion_idx] if top_confusion_idx >= 0 else None,
                "top_confusion_count": int(row_without_diag[top_confusion_idx]) if top_confusion_idx >= 0 else 0,
            }
        )

    class_df = pd.DataFrame(rows).sort_values(["f1_score", "recall", "support"], ascending=[True, True, False])
    out_path = report_dir / f"{name}_class_metrics.csv"
    class_df.to_csv(out_path, index=False)
    return out_path, class_df


def save_metrics_summary_csv(name: str, metrics: Dict[str, Any], report_dir: Path) -> Path:
    summary_rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            summary_rows.append({"metric": key, "value": value})

    out_path = report_dir / f"{name}_metrics_summary.csv"
    pd.DataFrame(summary_rows).to_csv(out_path, index=False)
    return out_path


def build_split_coverage(classes: List[str], y_train: np.ndarray, y_val: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, int]]:
    coverage = {}
    for class_name in classes:
        coverage[class_name] = {
            "train": int((y_train == class_name).sum()),
            "validation": int((y_val == class_name).sum()),
            "test": int((y_test == class_name).sum()),
        }
    return coverage


def analyze_attack_harm(
    class_metrics_df: pd.DataFrame,
    confusion_df: pd.DataFrame,
    benign_label: str | None,
) -> pd.DataFrame:
    if class_metrics_df.empty:
        return pd.DataFrame()

    harm_df = class_metrics_df.copy()
    harm_df = harm_df[~harm_df["class_name"].isin(NORMAL_LABELS)].copy()
    if harm_df.empty:
        return harm_df

    benign_to_attack = {}
    if benign_label and not confusion_df.empty and benign_label in confusion_df.index:
        benign_row = confusion_df.loc[benign_label]
        benign_to_attack = {
            class_name: int(benign_row.get(class_name, 0))
            for class_name in confusion_df.columns
            if class_name != benign_label
        }

    harm_df["benign_false_positive"] = harm_df["class_name"].map(lambda value: benign_to_attack.get(value, 0))
    harm_df["precision_penalty"] = 1.0 - harm_df["precision"].clip(lower=0.0, upper=1.0)
    harm_df["f1_penalty"] = 1.0 - harm_df["f1_score"].clip(lower=0.0, upper=1.0)
    harm_df["false_positive_ratio"] = harm_df["false_positive"] / harm_df["predicted_as_class"].clip(lower=1)

    max_false_positive = max(float(harm_df["false_positive"].max()), 1.0)
    max_benign_fp = max(float(harm_df["benign_false_positive"].max()), 1.0)

    harm_df["normalized_false_positive"] = harm_df["false_positive"] / max_false_positive
    harm_df["normalized_benign_false_positive"] = harm_df["benign_false_positive"] / max_benign_fp

    def support_factor(support: int) -> float:
        if support < 20:
            return 0.35
        if support < 50:
            return 0.55
        if support < 200:
            return 0.75
        return 1.0

    harm_df["support_factor"] = harm_df["support"].apply(support_factor)
    harm_df["harm_score"] = harm_df["support_factor"] * (
        0.35 * harm_df["precision_penalty"]
        + 0.25 * harm_df["f1_penalty"]
        + 0.20 * harm_df["false_positive_ratio"]
        + 0.10 * harm_df["normalized_false_positive"]
        + 0.10 * harm_df["normalized_benign_false_positive"]
    )

    return harm_df.sort_values(
        ["harm_score", "benign_false_positive", "false_positive", "f1_score"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def compare_run_scores(run_result: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(run_result["test_f1_macro"]),
        float(run_result["test_precision_macro"]),
        float(run_result["test_recall_macro"]),
    )


def is_feasible_removed_set(removed_attacks: List[str], all_attack_classes: List[str]) -> bool:
    remaining_attack_count = len(all_attack_classes) - len(set(removed_attacks))
    return remaining_attack_count >= MIN_ATTACK_CLASSES_TO_KEEP


def choose_next_removed_attacks(
    run_index: int,
    current_removed: List[str],
    best_removed: List[str],
    harm_df: pd.DataFrame,
    all_attack_classes: List[str],
    seen_removed_sets: set[Tuple[str, ...]],
) -> Tuple[List[str] | None, str]:
    if harm_df.empty:
        return None, "no_harm_data"

    harm_ranked = harm_df["class_name"].tolist()
    benign_ranked = harm_df.sort_values(
        ["benign_false_positive", "false_positive", "precision"],
        ascending=[False, False, True],
    )["class_name"].tolist()
    precision_ranked = harm_df.sort_values(
        ["precision", "false_positive", "support"],
        ascending=[True, False, False],
    )["class_name"].tolist()
    f1_ranked = harm_df.sort_values(
        ["f1_score", "false_positive", "support"],
        ascending=[True, False, False],
    )["class_name"].tolist()

    def candidate(base_removed: List[str], extras: List[str]) -> List[str]:
        return unique_keep_order(list(base_removed) + list(extras))

    strategies = [
        ("current_plus_top_harm", candidate(current_removed, harm_ranked[:1])),
        ("current_plus_top2_harm", candidate(current_removed, harm_ranked[:2])),
        ("current_plus_top_benign_confuser", candidate(current_removed, benign_ranked[:1])),
        ("current_plus_low_precision", candidate(current_removed, precision_ranked[:1])),
        ("best_plus_top_harm", candidate(best_removed, harm_ranked[:1])),
        ("best_plus_top2_harm", candidate(best_removed, harm_ranked[:2])),
        ("best_plus_top_benign_confuser", candidate(best_removed, benign_ranked[:1])),
        ("best_plus_low_precision", candidate(best_removed, precision_ranked[:1])),
        ("current_plus_mixed_pair", candidate(current_removed, unique_keep_order([harm_ranked[0], benign_ranked[0]])[:2])),
        ("best_plus_mixed_pair", candidate(best_removed, unique_keep_order([precision_ranked[0], harm_ranked[0]])[:2])),
        ("current_plus_f1_and_benign", candidate(current_removed, unique_keep_order([f1_ranked[0], benign_ranked[0]])[:2])),
        ("best_plus_three_harm", candidate(best_removed, harm_ranked[:3])),
    ]

    rotation = (run_index - 1) % len(strategies)
    ordered_strategies = strategies[rotation:] + strategies[:rotation]

    for strategy_name, removed in ordered_strategies:
        removed_key = tuple(sorted(set(removed)))
        if removed_key in seen_removed_sets:
            continue
        if not is_feasible_removed_set(list(removed_key), all_attack_classes):
            continue
        return list(removed_key), strategy_name

    # Fallback: try any unseen single or pair removals from the top harmful classes.
    top_pool = unique_keep_order(harm_ranked[:5] + benign_ranked[:5] + precision_ranked[:5])
    for attack in top_pool:
        removed = tuple(sorted(set(candidate(best_removed, [attack]))))
        if removed not in seen_removed_sets and is_feasible_removed_set(list(removed), all_attack_classes):
            return list(removed), "fallback_single"

    for idx, first_attack in enumerate(top_pool):
        for second_attack in top_pool[idx + 1 :]:
            removed = tuple(sorted(set(candidate(best_removed, [first_attack, second_attack]))))
            if removed not in seen_removed_sets and is_feasible_removed_set(list(removed), all_attack_classes):
                return list(removed), "fallback_pair"

    return None, "no_candidate_left"


def save_run_configuration(run_dir: Path, config: Dict[str, Any]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "run_configuration.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(config), handle, indent=2)


def load_run_configuration(run_dir: Path) -> Dict[str, Any] | None:
    config_path = run_dir / "run_configuration.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_run_result_from_metrics(metrics_path: Path) -> Dict[str, Any]:
    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    return {
        "run_index": int(metrics["experiment_run_index"]),
        "strategy_used": metrics["strategy_used"],
        "removed_attacks": metrics.get("removed_attacks", []),
        "kept_attacks": metrics.get("kept_attacks", []),
        "attack_count_kept": len(metrics.get("kept_attacks", [])),
        "test_f1_macro": float(metrics["test_f1_macro"]),
        "test_precision_macro": float(metrics["test_precision_macro"]),
        "test_recall_macro": float(metrics["test_recall_macro"]),
        "best_cost": metrics.get("best_cost"),
        "report_dir": relative_to_base(metrics_path.parent),
        "weights_dir": relative_to_base(metrics_path.parent.parent / "weights"),
        "metrics_json": relative_to_base(metrics_path),
        "harmful_attacks_ranked": metrics.get("harmful_attacks_ranked", []),
    }


def load_existing_pretrain_history(report_dir: Path, name: str) -> Tuple[Path | None, Path, pd.DataFrame]:
    csv_path = report_dir / f"{name}_pretrain_history.csv"
    raw_path = report_dir / f"{name}_pretrain_history_raw.json"

    history_df = pd.DataFrame()
    if csv_path.exists():
        history_df = pd.read_csv(csv_path)
    elif raw_path.exists():
        with raw_path.open("r", encoding="utf-8") as handle:
            history_df = history_to_frame(json.load(handle))

    return (csv_path if csv_path.exists() else None), raw_path, history_df


def discover_resume_state(
    all_attack_classes: List[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any] | None, set[Tuple[str, ...]], int, List[str], str, Path | None]:
    run_summaries: List[Dict[str, Any]] = []
    best_run: Dict[str, Any] | None = None
    seen_removed_sets: set[Tuple[str, ...]] = {tuple()}
    resume_run_index = 1
    planned_removed: List[str] = []
    planned_strategy = "start_with_all_attacks"
    resume_pretrain_zip: Path | None = None

    first_incomplete_index: int | None = None

    for run_index in range(1, TOTAL_RUNS + 1):
        run_dir = RUNS_DIR / f"run_{run_index:02d}"
        metrics_path = run_dir / "reports" / f"{DATASET_NAME}_metrics.json"

        if metrics_path.exists():
            run_result = load_run_result_from_metrics(metrics_path)
            run_summaries.append(run_result)
            seen_removed_sets.add(tuple(sorted(set(run_result["removed_attacks"]))))
            if best_run is None or compare_run_scores(run_result) > compare_run_scores(best_run):
                best_run = run_result
            continue

        first_incomplete_index = run_index
        break

    if first_incomplete_index is None:
        return run_summaries, best_run, seen_removed_sets, TOTAL_RUNS + 1, [], planned_strategy, None

    resume_run_index = first_incomplete_index
    run_dir = RUNS_DIR / f"run_{resume_run_index:02d}"
    run_config = load_run_configuration(run_dir)

    if run_config is not None:
        planned_removed = sorted(set(run_config.get("removed_attacks", [])))
        planned_strategy = run_config.get("strategy_used", "resume_saved_configuration")
    elif run_summaries:
        previous_run = run_summaries[-1]
        next_removed, next_strategy = choose_next_removed_attacks(
            run_index=resume_run_index,
            current_removed=previous_run["removed_attacks"],
            best_removed=best_run["removed_attacks"] if best_run else previous_run["removed_attacks"],
            harm_df=pd.DataFrame(previous_run["harmful_attacks_ranked"]),
            all_attack_classes=all_attack_classes,
            seen_removed_sets=seen_removed_sets,
        )
        if next_removed is None:
            planned_removed = previous_run["removed_attacks"]
            planned_strategy = "resume_no_new_candidate"
        else:
            planned_removed = next_removed
            planned_strategy = next_strategy
    else:
        planned_removed = []
        planned_strategy = "start_with_all_attacks"

    pretrain_zip = run_dir / "weights" / f"{DATASET_NAME}_pretrain.zip"
    if pretrain_zip.exists():
        resume_pretrain_zip = pretrain_zip

    return (
        run_summaries,
        best_run,
        seen_removed_sets,
        resume_run_index,
        planned_removed,
        planned_strategy,
        resume_pretrain_zip,
    )


def save_experiment_summary(run_summaries: List[Dict[str, Any]], best_run: Dict[str, Any] | None) -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "total_runs_completed": len(run_summaries),
        "target_macro_f1": TARGET_MACRO_F1,
        "best_run": best_run,
        "all_runs": run_summaries,
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(summary_payload), handle, indent=2)

    csv_rows = []
    for run in run_summaries:
        csv_rows.append(
            {
                "run_index": run["run_index"],
                "strategy_used": run["strategy_used"],
                "test_f1_macro": run["test_f1_macro"],
                "test_precision_macro": run["test_precision_macro"],
                "test_recall_macro": run["test_recall_macro"],
                "attack_count_kept": len(run["kept_attacks"]),
                "attacks_removed": "|".join(run["removed_attacks"]),
                "report_dir": run["report_dir"],
                "weights_dir": run["weights_dir"],
            }
        )
    pd.DataFrame(csv_rows).to_csv(SUMMARY_CSV, index=False)


def train_one(
    name: str,
    path: Path,
    report_dir: Path,
    weights_dir: Path,
    removed_attacks: List[str],
    strategy_used: str,
    run_index: int,
    resume_pretrain_zip: Path | None = None,
) -> Dict[str, Any]:
    print(f"\n=== Processing {name} | run {run_index:02d} ===")
    report_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(path, low_memory=False)
    df.columns = [column.strip() for column in df.columns]
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.replace("\ufffd", "-", regex=False)
    df.dropna(subset=[LABEL_COL], inplace=True)

    removed_attacks = sorted(set(removed_attacks))
    if removed_attacks:
        initial_rows = len(df)
        df = df[~df[LABEL_COL].isin(removed_attacks)].copy()
        rows_removed = initial_rows - len(df)
        print(f"   Removed {rows_removed:,} rows from dropped attack classes: {removed_attacks}")
    else:
        print("   Starting with all attack classes.")

    df, dropped_non_numeric_columns = drop_non_numeric_features(df, label_col=LABEL_COL)
    if dropped_non_numeric_columns:
        print(f"   Dropped non-numeric feature columns: {dropped_non_numeric_columns}")
    else:
        print("   No non-numeric feature columns found.")

    before_dropna = len(df)
    df = df.dropna().reset_index(drop=True)
    dropped_na = before_dropna - len(df)
    if dropped_na > 0:
        print(f"   Dropped {dropped_na:,} rows with NaN values before splitting.")

    if df.empty:
        raise RuntimeError(f"No data left for {name} after cleaning.")

    feature_names = [column for column in df.columns if column != LABEL_COL]
    attack_classes_kept = attack_labels_from_series(df[LABEL_COL])
    print(f"   Final numeric feature count: {len(feature_names)}")
    print(f"   Attack classes kept: {len(attack_classes_kept)}")

    train_mask, val_mask, test_mask = stratified_time_split(df, label_col=LABEL_COL, train_ratio=0.8, val_ratio=0.1)
    train_df = df[train_mask].reset_index(drop=True)
    val_df = df[val_mask].reset_index(drop=True)
    test_df = df[test_mask].reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise RuntimeError(f"Split failed for {name}: one of the sets is empty.")

    X_train, y_train = prepare_xy(train_df)
    X_val, y_val = prepare_xy(val_df)
    X_test, y_test = prepare_xy(test_df)

    classes, y_train_enc = np.unique(y_train, return_inverse=True)
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    y_val_enc = np.array([class_to_idx[class_name] for class_name in y_val], dtype=int)
    y_test_enc = np.array([class_to_idx[class_name] for class_name in y_test], dtype=int)

    split_coverage = build_split_coverage(list(classes), y_train, y_val, y_test)
    missing_split_classes = {
        class_name: counts
        for class_name, counts in split_coverage.items()
        if min(counts.values()) == 0
    }
    for class_name in classes:
        counts = split_coverage[class_name]
        print(
            f"   Label '{class_name}': "
            f"train={counts['train']}, val={counts['validation']}, test={counts['test']}"
        )

    pretrainer = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        verbose=10,
        device_name=DEVICE,
    )

    pretrain_live_path = report_dir / f"{name}_pretrain_history_live.csv"
    if resume_pretrain_zip and resume_pretrain_zip.exists():
        print(f"   Reusing saved pretraining weights: {resume_pretrain_zip}")
        pretrainer.load_model(str(resume_pretrain_zip))
        pretrain_time = 0.0
        pre_hist_file, pre_hist_raw, pretrain_history_df = load_existing_pretrain_history(report_dir, name)
    else:
        pre_csv_logger = EpochCSVLogger(pretrain_live_path)
        pretrain_start = time.time()
        pretrainer.fit(
            X_train=X_train,
            eval_set=[X_val],
            max_epochs=200,
            patience=15,
            batch_size=2048,
            virtual_batch_size=256,
            num_workers=0,
            drop_last=False,
            callbacks=[pre_csv_logger],
        )
        pretrain_time = time.time() - pretrain_start
        pre_hist_file, pre_hist_raw = save_history(name, pretrainer.history, "pretrain", report_dir)
        pretrain_history_df = history_to_frame(pretrainer.history)
    pretrain_weights_path = weights_dir / f"{name}_pretrain"
    if not (resume_pretrain_zip and resume_pretrain_zip.exists()):
        pretrainer.save_model(str(pretrain_weights_path))

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 20, "gamma": 0.9},
        mask_type="sparsemax",
        verbose=10,
        device_name=DEVICE,
    )

    finetune_live_path = report_dir / f"{name}_finetune_history_live.csv"
    ft_csv_logger = EpochCSVLogger(finetune_live_path)
    finetune_start = time.time()
    clf.fit(
        X_train=X_train,
        y_train=y_train_enc,
        eval_set=[(X_train, y_train_enc), (X_val, y_val_enc)],
        eval_name=["train", "valid"],
        eval_metric=[
            PrecisionMacro,
            RecallMacro,
            F1Macro,
        ],
        max_epochs=200,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        from_unsupervised=pretrainer,
        weights=1,
        callbacks=[ft_csv_logger],
    )
    finetune_time = time.time() - finetune_start
    ft_hist_file, ft_hist_raw = save_history(name, clf.history, "finetune", report_dir)
    finetune_history_df = history_to_frame(clf.history)
    finetune_weights_path = weights_dir / f"{name}_finetune"
    clf.save_model(str(finetune_weights_path))

    test_prob = clf.predict_proba(X_test)
    y_pred = np.argmax(test_prob, axis=1)

    cm = confusion_matrix(y_test_enc, y_pred, labels=range(len(classes)))
    class_report = classification_report(
        y_test_enc,
        y_pred,
        labels=range(len(classes)),
        target_names=list(classes),
        output_dict=True,
        zero_division=0,
    )
    filtered_class_report = {
        key: value
        for key, value in class_report.items()
        if key in set(classes).union({"macro avg"})
    }

    confusion_df = pd.DataFrame(cm, index=list(classes), columns=list(classes))
    benign_label = next((label for label in classes if label in NORMAL_LABELS), None)

    feature_importance_path = save_feature_importances(name, clf, feature_names, report_dir)
    confusion_csv_path = save_confusion_matrix_csv(name, cm, list(classes), report_dir)
    top_confusions_path, top_confusions_df = save_top_confusions_csv(name, cm, list(classes), report_dir)
    class_metrics_path, class_metrics_df = save_class_metrics_csv(name, filtered_class_report, cm, list(classes), report_dir)
    harm_df = analyze_attack_harm(class_metrics_df, confusion_df, benign_label)

    metrics: Dict[str, Any] = {
        "dataset_name": name,
        "experiment_run_index": run_index,
        "strategy_used": strategy_used,
        "device": DEVICE,
        "total_rows": int(len(df)),
        "feature_count": int(len(feature_names)),
        "feature_names": feature_names,
        "dropped_non_numeric_columns": dropped_non_numeric_columns,
        "removed_attacks": removed_attacks,
        "kept_attacks": attack_classes_kept,
        "classes": list(classes),
        "class_counts_train": {class_name: int((y_train == class_name).sum()) for class_name in classes},
        "class_counts_val": {class_name: int((y_val == class_name).sum()) for class_name in classes},
        "class_counts_test": {class_name: int((y_test == class_name).sum()) for class_name in classes},
        "split_coverage": split_coverage,
        "classes_missing_in_some_split": missing_split_classes,
        "pretrain_time_sec": float(pretrain_time),
        "finetune_time_sec": float(finetune_time),
        "best_cost": getattr(clf, "best_cost", None),
        "best_step": getattr(clf, "best_step", None),
        "test_f1_macro": float(f1_score(y_test_enc, y_pred, average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test_enc, y_pred, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test_enc, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": convert_to_serializable(filtered_class_report),
        "strongest_classes_by_f1": class_metrics_df.sort_values("f1_score", ascending=False).head(5).to_dict("records"),
        "struggling_classes_by_f1": class_metrics_df.sort_values("f1_score", ascending=True).head(5).to_dict("records"),
        "top_confusions": top_confusions_df.head(20).to_dict("records"),
        "harmful_attacks_ranked": harm_df.head(10).to_dict("records"),
        "history_summary": {
            "pretrain": summarize_history(pretrain_history_df, "pretrain"),
            "finetune": summarize_history(finetune_history_df, "finetune"),
        },
        "history_files": {
            "pretrain": relative_to_base(pre_hist_file) if pre_hist_file else None,
            "finetune": relative_to_base(ft_hist_file) if ft_hist_file else None,
            "pretrain_raw": relative_to_base(pre_hist_raw),
            "finetune_raw": relative_to_base(ft_hist_raw),
            "pretrain_live": relative_to_base(pretrain_live_path),
            "finetune_live": relative_to_base(finetune_live_path),
        },
        "report_files": {
            "feature_importance_csv": relative_to_base(feature_importance_path),
            "confusion_matrix_csv": relative_to_base(confusion_csv_path),
            "top_confusions_csv": relative_to_base(top_confusions_path),
            "class_metrics_csv": relative_to_base(class_metrics_path),
        },
        "weights_files": {
            "pretrain_model": relative_to_base(zipped_model_path(pretrain_weights_path)),
            "finetune_model": relative_to_base(zipped_model_path(finetune_weights_path)),
        },
        "resume_used_existing_pretrain": bool(resume_pretrain_zip and resume_pretrain_zip.exists()),
    }

    metrics_summary_path = save_metrics_summary_csv(name, metrics, report_dir)
    metrics["report_files"]["metrics_summary_csv"] = relative_to_base(metrics_summary_path)

    metrics_path = report_dir / f"{name}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(metrics), handle, indent=2)

    run_result = {
        "run_index": run_index,
        "strategy_used": strategy_used,
        "removed_attacks": removed_attacks,
        "kept_attacks": attack_classes_kept,
        "attack_count_kept": len(attack_classes_kept),
        "test_f1_macro": metrics["test_f1_macro"],
        "test_precision_macro": metrics["test_precision_macro"],
        "test_recall_macro": metrics["test_recall_macro"],
        "best_cost": metrics["best_cost"],
        "report_dir": relative_to_base(report_dir),
        "weights_dir": relative_to_base(weights_dir),
        "metrics_json": relative_to_base(metrics_path),
        "harmful_attacks_ranked": harm_df.to_dict("records"),
    }

    save_run_configuration(
        report_dir.parent,
        {
            "run_index": run_index,
            "strategy_used": strategy_used,
            "removed_attacks": removed_attacks,
            "kept_attacks": attack_classes_kept,
            "target_macro_f1": TARGET_MACRO_F1,
        },
    )

    print(f"Saved metrics JSON: {metrics_path}")
    print(f"Saved run reports to: {report_dir}")
    print(f"Saved run weights to: {weights_dir}")
    print(
        f"Run {run_index:02d} finished | "
        f"macro_f1={metrics['test_f1_macro']:.4f} | "
        f"macro_precision={metrics['test_precision_macro']:.4f} | "
        f"macro_recall={metrics['test_recall_macro']:.4f}"
    )
    return run_result


def build_combined_dataset() -> Path:
    combined_path = DATA_DIR / "All-WorkingHours-Combined.csv"
    if combined_path.exists():
        print(f"Found existing combined dataset: {combined_path}")
        return combined_path

    print("Combined file not found. Concatenating all CSVs...")
    dataframes = {}
    common_cols = None

    for path in sorted(DATA_DIR.glob("*.csv")):
        if path.name == "All-WorkingHours-Combined.csv":
            continue

        print(f"  Reading {path.name}...")
        df = pd.read_csv(path, low_memory=False)
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
        )

        dataframes[path.name] = df
        cols = set(df.columns)
        common_cols = cols if common_cols is None else common_cols.intersection(cols)

    if not dataframes:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    common_cols = sorted(common_cols)
    print(f"\nCommon columns across all files: {len(common_cols)}")

    frames = []
    for name, df in dataframes.items():
        df = df[common_cols].copy()
        frames.append(df)
        print(f"  Added {name}: {len(df):,} rows")

    df_all = pd.concat(frames, ignore_index=True)
    null_rows = int(df_all.isna().any(axis=1).sum())
    print(f"\nCombined rows with nulls: {null_rows:,}")

    df_all.to_csv(combined_path, index=False)
    print(f"Created combined dataset: {combined_path}")
    return combined_path


def sleep_for_cooldown(run_index: int) -> None:
    next_start_time = datetime.now() + timedelta(seconds=COOLDOWN_SECONDS)
    print(
        f"\nRun {run_index:02d} complete. Cooling GPU for {COOLDOWN_SECONDS // 60} minutes.\n"
        f"Next run will start automatically at {next_start_time.strftime('%Y-%m-%d %H:%M:%S')}."
    )
    time.sleep(COOLDOWN_SECONDS)


def load_all_attack_classes(combined_path: Path) -> List[str]:
    df = pd.read_csv(combined_path, usecols=[LABEL_COL], low_memory=False)
    df[LABEL_COL] = df[LABEL_COL].astype(str).str.replace("\ufffd", "-", regex=False)
    return attack_labels_from_series(df[LABEL_COL])


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    combined_path = build_combined_dataset()
    all_attack_classes = load_all_attack_classes(combined_path)

    if len(all_attack_classes) < MIN_ATTACK_CLASSES_TO_KEEP:
        raise RuntimeError(
            f"Only {len(all_attack_classes)} attack classes are available, "
            f"but the script is configured to keep at least {MIN_ATTACK_CLASSES_TO_KEEP}."
        )

    print(f"\nTotal available attack classes: {len(all_attack_classes)}")
    print(f"Minimum attack classes to keep per run: {MIN_ATTACK_CLASSES_TO_KEEP}")
    print(f"Target macro F1: {TARGET_MACRO_F1:.2f}")

    (
        run_summaries,
        best_run,
        seen_removed_sets,
        start_run_index,
        current_removed,
        last_strategy,
        resume_pretrain_zip,
    ) = discover_resume_state(all_attack_classes)

    if start_run_index > TOTAL_RUNS:
        print("All configured runs are already complete. Nothing to resume.")
        save_experiment_summary(run_summaries, best_run)
        return

    if run_summaries:
        print(f"\nFound {len(run_summaries)} completed run(s). Resuming from run {start_run_index:02d}.")
    else:
        print("\nNo completed runs found. Starting fresh from run 01.")

    if current_removed:
        print(f"Run {start_run_index:02d} will remove attacks: {current_removed} | strategy={last_strategy}")
    else:
        print(f"Run {start_run_index:02d} will start with all attack classes | strategy={last_strategy}")

    for run_index in range(start_run_index, TOTAL_RUNS + 1):
        run_dir = RUNS_DIR / f"run_{run_index:02d}"
        report_dir = run_dir / "reports"
        weights_dir = run_dir / "weights"

        save_run_configuration(
            run_dir,
            {
                "run_index": run_index,
                "strategy_used": last_strategy,
                "removed_attacks": current_removed,
                "target_macro_f1": TARGET_MACRO_F1,
                "resume_pretrain_zip": relative_to_base(resume_pretrain_zip) if resume_pretrain_zip else None,
            },
        )

        run_result = train_one(
            name=DATASET_NAME,
            path=combined_path,
            report_dir=report_dir,
            weights_dir=weights_dir,
            removed_attacks=current_removed,
            strategy_used=last_strategy,
            run_index=run_index,
            resume_pretrain_zip=resume_pretrain_zip if run_index == start_run_index else None,
        )
        run_summaries.append(run_result)

        if best_run is None or compare_run_scores(run_result) > compare_run_scores(best_run):
            best_run = run_result
            print(
                f"New best run: {run_index:02d} | "
                f"macro_f1={run_result['test_f1_macro']:.4f} | "
                f"removed={run_result['removed_attacks']}"
            )

        save_experiment_summary(run_summaries, best_run)

        if run_index >= TOTAL_RUNS:
            break

        next_removed, strategy_name = choose_next_removed_attacks(
            run_index=run_index + 1,
            current_removed=run_result["removed_attacks"],
            best_removed=best_run["removed_attacks"] if best_run else run_result["removed_attacks"],
            harm_df=pd.DataFrame(run_result["harmful_attacks_ranked"]),
            all_attack_classes=all_attack_classes,
            seen_removed_sets=seen_removed_sets,
        )

        if next_removed is None:
            print("No new feasible attack-drop combination remains. Stopping early.")
            break

        current_removed = next_removed
        seen_removed_sets.add(tuple(sorted(current_removed)))
        last_strategy = strategy_name
        resume_pretrain_zip = None
        print(f"Next run will remove attacks: {current_removed} | strategy={strategy_name}")

        sleep_for_cooldown(run_index)

    save_experiment_summary(run_summaries, best_run)

    print("\n" + "=" * 100)
    print("ITERATIVE ATTACK PRUNING EXPERIMENT COMPLETE")
    print("=" * 100)
    print(f"Runs completed: {len(run_summaries)} / {TOTAL_RUNS}")
    if best_run:
        print(
            f"Best run: {best_run['run_index']:02d} | "
            f"macro_f1={best_run['test_f1_macro']:.4f} | "
            f"macro_precision={best_run['test_precision_macro']:.4f} | "
            f"macro_recall={best_run['test_recall_macro']:.4f}"
        )
        print(f"Best removed attacks: {best_run['removed_attacks']}")
        print(f"Best kept attacks ({len(best_run['kept_attacks'])}): {best_run['kept_attacks']}")
        if best_run["test_f1_macro"] >= TARGET_MACRO_F1:
            print(f"Target reached: macro F1 crossed {TARGET_MACRO_F1:.2f}")
        else:
            print(f"Target not reached yet: best macro F1 stayed below {TARGET_MACRO_F1:.2f}")
        print(f"Best reports folder: {best_run['report_dir']}")
        print(f"Best weights folder: {best_run['weights_dir']}")
    print(f"Full experiment summary JSON: {relative_to_base(SUMMARY_JSON)}")
    print(f"Full experiment summary CSV: {relative_to_base(SUMMARY_CSV)}")
    print("=" * 100)


if __name__ == "__main__":
    main()
