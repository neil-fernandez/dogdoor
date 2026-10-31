"""Evaluation metrics and report generation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def compute_report(y_true, y_pred, class_names) -> dict:
    labels = list(range(len(class_names)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    return {
        "accuracy": float(acc),
        "macro_f1": float(np.mean(f1)),
        "class_names": list(class_names),
        "confusion_matrix": cm.tolist(),
        "per_class": [
            {
                "name": name,
                "precision": float(p[i]),
                "recall": float(r[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        ],
    }
