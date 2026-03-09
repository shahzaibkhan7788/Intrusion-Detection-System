"""Pretrain on Monday benign flows, then fine-tune binary classifier on Tuesday.

Usage:
  python run_monday_tuesday.py

Assumptions:
- Cleaned CSVs exist in ./data_preprocessed/ as produced by cic_prepare.py.
- Columns are all numeric, with a single label column named 'Label'.
- Rows remain in time order; we split by index to approximate time-based splits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from custom_metrics import (
    AveragePrecision,
    F1Macro,
    F1Micro,
    PrecisionMacro,
    RecallForClass,
    RecallMacro,
    confusion_matrix_counts,
)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

SEED = 7
np.random.seed(SEED)
torch.manual_seed(SEED)


BASE_DIR = Path(__file__).parent
MONDAY_PATH = BASE_DIR / "data_preprocessed/Monday-WorkingHours.pcap_ISCX.csv"
TUESDAY_PATH = BASE_DIR / "data_preprocessed/Tuesday-WorkingHours.pcap_ISCX.csv"
REPORT_PATH = BASE_DIR / "reports/monday_tuesday_results.json"


def time_split(df: pd.DataFrame, ratios=(0.7, 0.15, 0.15)) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assert abs(sum(ratios) - 1.0) < 1e-6
    n = len(df)
    n_train = int(n * ratios[0])
    n_valid = int(n * ratios[1])
    train = df.iloc[:n_train]
    valid = df.iloc[n_train : n_train + n_valid]
    test = df.iloc[n_train + n_valid :]
    return train, valid, test


def load_features_labels(df: pd.DataFrame):
    label_col = "Label"
    y = df[label_col]
    X = df.drop(columns=[label_col])
    return X.values, y.values


def to_binary(y_raw):
    return np.where(y_raw == "BENIGN", 0, 1)


def pretrain_monday():
    df = pd.read_csv(MONDAY_PATH)
    train_df, valid_df, _ = time_split(df)
    X_train, _ = load_features_labels(train_df)
    X_valid, _ = load_features_labels(valid_df)

    pretrainer = TabNetPretrainer(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        verbose=10,
        device_name=device,
    )

    pretrainer.fit(
        X_train=X_train,
        eval_set=[X_valid],
        max_epochs=50,
        patience=5,
        batch_size=2048,
        virtual_batch_size=256,
        num_workers=0,
        drop_last=False,
    )

    ckpt_path = BASE_DIR / "models" / "monday_pretrain"
    ckpt_path.parent.mkdir(exist_ok=True)
    pretrainer.save_model(str(ckpt_path))
    return pretrainer, ckpt_path


def finetune_tuesday(pretrainer: TabNetPretrainer):
    df = pd.read_csv(TUESDAY_PATH)
    train_df, valid_df, test_df = time_split(df)

    X_train, y_train_raw = load_features_labels(train_df)
    X_valid, y_valid_raw = load_features_labels(valid_df)
    X_test, y_test_raw = load_features_labels(test_df)

    y_train = to_binary(y_train_raw)
    y_valid = to_binary(y_valid_raw)
    y_test = to_binary(y_test_raw)

    clf = TabNetClassifier(
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-3),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params={"step_size": 10, "gamma": 0.9},
        mask_type="sparsemax",
        verbose=10,
        device_name=device,
    )

    clf.fit(
        X_train=X_train,
        y_train=y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_name=["train", "valid"],
        eval_metric=[
            "auc",
            F1Macro,
            F1Micro,
            PrecisionMacro,
            RecallMacro,
            RecallForClass,
            AveragePrecision,
        ],
        max_epochs=100,
        patience=15,
        batch_size=1024,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        from_unsupervised=pretrainer,
        weights=1,
    )

    # Evaluate
    preds_valid = clf.predict_proba(X_valid)[:, 1]
    preds_test = clf.predict_proba(X_test)[:, 1]

    pred_labels_valid = (preds_valid >= 0.5).astype(int)
    pred_labels_test = (preds_test >= 0.5).astype(int)

    results = {
        "valid_auc": float(roc_auc_score(y_valid, preds_valid)),
        "valid_pr_auc": float(average_precision_score(y_valid, preds_valid)),
        "test_auc": float(roc_auc_score(y_test, preds_test)),
        "test_pr_auc": float(average_precision_score(y_test, preds_test)),
        "valid_precision_macro": float(precision_score(y_valid, pred_labels_valid, average="macro", zero_division=0)),
        "valid_recall_macro": float(recall_score(y_valid, pred_labels_valid, average="macro", zero_division=0)),
        "valid_f1_macro": float(f1_score(y_valid, pred_labels_valid, average="macro", zero_division=0)),
        "valid_f1_micro": float(f1_score(y_valid, pred_labels_valid, average="micro", zero_division=0)),
        "valid_recall_attack": float(recall_score(y_valid, pred_labels_valid, labels=[1], average="macro", zero_division=0)),
        "test_precision_macro": float(precision_score(y_test, pred_labels_test, average="macro", zero_division=0)),
        "test_recall_macro": float(recall_score(y_test, pred_labels_test, average="macro", zero_division=0)),
        "test_f1_macro": float(f1_score(y_test, pred_labels_test, average="macro", zero_division=0)),
        "test_f1_micro": float(f1_score(y_test, pred_labels_test, average="micro", zero_division=0)),
        "test_recall_attack": float(recall_score(y_test, pred_labels_test, labels=[1], average="macro", zero_division=0)),
        "class_balance_train": {
            "benign": int((y_train == 0).sum()),
            "attack": int((y_train == 1).sum()),
        },
        "class_balance_valid": {
            "benign": int((y_valid == 0).sum()),
            "attack": int((y_valid == 1).sum()),
        },
        "class_balance_test": {
            "benign": int((y_test == 0).sum()),
            "attack": int((y_test == 1).sum()),
        },
        "confusion_valid": confusion_matrix_counts(y_valid, pred_labels_valid),
        "confusion_test": confusion_matrix_counts(y_test, pred_labels_test),
    }

    model_path = BASE_DIR / "models" / "tuesday_binary"
    model_path.parent.mkdir(exist_ok=True)
    clf.save_model(str(model_path))

    return results, model_path


def main():
    pretrainer, ckpt_path = pretrain_monday()
    results, model_path = finetune_tuesday(pretrainer)

    REPORT_PATH.parent.mkdir(exist_ok=True)
    with REPORT_PATH.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "pretrain_checkpoint": str(ckpt_path.relative_to(BASE_DIR)),
                "finetune_model": str(model_path.relative_to(BASE_DIR)),
                "metrics": results,
            },
            f,
            indent=2,
        )

    print("Saved:", REPORT_PATH)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
