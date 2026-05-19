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
from sklearn.preprocessing import label_binarize
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
    """Average Precision / PR-AUC for binary or multi-class classification.

    - Binary: uses the positive-class column when available, otherwise the single column.
    - Multi-class: macro-averaged AP over one-vs-rest binarized targets.
    """

    def __init__(self):
        self._name = "average_precision"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)

        # Ensure 2D score array
        if y_score.ndim == 1:
            y_score = y_score.reshape(-1, 1)

        n_classes = y_score.shape[1]

        # Binary shortcuts
        if n_classes == 1:
            return average_precision_score(y_true, y_score[:, 0])
        if n_classes == 2 and np.unique(y_true).size <= 2:
            return average_precision_score(y_true, y_score[:, 1])

        # Multi-class: one-vs-rest macro AP
        classes = np.arange(n_classes)
        y_true_bin = label_binarize(y_true, classes=classes)
        return average_precision_score(y_true_bin, y_score, average="macro")


class PrecisionWeighted(Metric):
    def __init__(self):
        self._name = "precision_weighted"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return precision_score(y_true, preds, average="weighted", zero_division=0)


class RecallWeighted(Metric):
    def __init__(self):
        self._name = "recall_weighted"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return recall_score(y_true, preds, average="weighted", zero_division=0)


class F1Weighted(Metric):
    def __init__(self):
        self._name = "f1_weighted"
        self._maximize = True

    def __call__(self, y_true, y_score):
        preds = np.argmax(y_score, axis=1)
        return f1_score(y_true, preds, average="weighted", zero_division=0)
    
    

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
