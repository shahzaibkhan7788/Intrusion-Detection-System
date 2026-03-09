"""Custom evaluation metrics for PyTorch TabNet.

These metrics extend the built-in options with common classification scores:
- Precision (macro)
- Recall (macro)
- Per-class recall (configurable class index)
- F1 (macro and micro)
- Average Precision / PR-AUC (binary)

Usage:
    from pytorch_tabnet.metrics import Metric
from custom_metrics import (
    F1Macro,
    F1Micro,
    PrecisionMacro,
    RecallMacro,
    RecallForClass,
    AveragePrecision,
)

    clf = TabNetClassifier(...)
    clf.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric=[F1Macro, PrecisionMacro, RecallMacro, AveragePrecision],
        ...
    )

Notes:
- For multi-class, scores are computed with macro averaging on hard predictions (argmax).
- For binary, y_score is expected to have shape (n_samples, 2); positive class is column 1.
"""

from __future__ import annotations

import numpy as np
from pytorch_tabnet.metrics import Metric
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score


class PrecisionMacro(Metric):
    def __init__(self):
        self._name = "precision_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return precision_score(y_true, preds, average="macro", zero_division=0)


class RecallMacro(Metric):
    def __init__(self):
        self._name = "recall_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return recall_score(y_true, preds, average="macro", zero_division=0)


class RecallForClass(Metric):
    """Recall for a specific class index (multiclass or binary).

    Args:
        class_id: integer class index as used in labels.
    """

    def __init__(self, class_id: int = 1):
        self.class_id = class_id
        self._name = f"recall_class_{class_id}"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return recall_score(y_true, preds, labels=[self.class_id], average="macro", zero_division=0)


class F1Macro(Metric):
    def __init__(self):
        self._name = "f1_macro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds, average="macro", zero_division=0)


class F1Micro(Metric):
    def __init__(self):
        self._name = "f1_micro"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds, average="micro", zero_division=0)


class AveragePrecision(Metric):
    """Average Precision / PR-AUC for binary classification.

    Assumes y_score is array of shape (n_samples, 2) with positive class in column 1.
    """

    def __init__(self):
        self._name = "average_precision"
        self._maximize = True

    def __call__(self, y_true, y_score):
        if y_score.shape[1] < 2:
            # Fallback: use the single-column score directly
            pos_scores = y_score[:, 0]
        else:
            pos_scores = y_score[:, 1]
        return average_precision_score(y_true, pos_scores)


# Optional: confusion matrix helper (not a Metric because it is not scalar)
def confusion_matrix_counts(y_true, y_pred):
    """Return TP, FP, FN, TN counts for binary arrays.

    Not used by TabNet's eval loop directly but can be called after predictions.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}
