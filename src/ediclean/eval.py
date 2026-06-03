from __future__ import annotations

from typing import Dict, Sequence, Optional, Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def _safe_probs(y_proba: Optional[Sequence[Sequence[float]]]) -> Optional[np.ndarray]:
    if y_proba is None:
        return None
    arr = np.asarray(y_proba, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        return None
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    row_sums = arr.sum(axis=1, keepdims=True)
    # If rows do not sum to 1, normalize them safely.
    arr = np.divide(arr, np.maximum(row_sums, 1e-12))
    return np.clip(arr, 1e-12, 1.0)


def _expected_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> float:
    ece = 0.0
    for lo, hi in zip(np.linspace(0.0, 1.0, n_bins, endpoint=False), np.linspace(0.1, 1.0, n_bins)):
        if hi >= 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_pred[mask] == y_true[mask])
        bin_conf = np.mean(confidences[mask])
        ece += (np.sum(mask) / len(y_true)) * abs(bin_acc - bin_conf)
    return float(ece)


def _maximum_calibration_error(y_true: np.ndarray, y_pred: np.ndarray, confidences: np.ndarray, n_bins: int = 10) -> float:
    gaps = []
    for lo, hi in zip(np.linspace(0.0, 1.0, n_bins, endpoint=False), np.linspace(0.1, 1.0, n_bins)):
        if hi >= 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if not np.any(mask):
            continue
        bin_acc = np.mean(y_pred[mask] == y_true[mask])
        bin_conf = np.mean(confidences[mask])
        gaps.append(abs(bin_acc - bin_conf))
    return float(max(gaps)) if gaps else 0.0


def compute_confidence_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[Sequence[float]]],
    labels: Sequence[int],
    n_bins: int = 10,
) -> Dict[str, Any]:
    """Compute confidence/calibration metrics from class probabilities.

    Expected y_proba shape: [num_examples, num_classes], ordered according to labels.
    Returned metrics:
    - avg_confidence: mean max probability
    - avg_confidence_correct / wrong: confidence split by correctness
    - avg_margin: mean top1-top2 probability gap
    - avg_entropy: mean predictive entropy
    - nll: negative log likelihood of true labels
    - brier_score: multiclass Brier score
    - ece / mce: calibration errors
    """
    probs = _safe_probs(y_proba)
    if probs is None:
        return {}

    y_true = np.asarray(list(y_true), dtype=int)
    y_pred = np.asarray(list(y_pred), dtype=int)
    labels = list(labels)
    label_to_col = {int(label): i for i, label in enumerate(labels)}

    valid_true_mask = np.array([int(y) in label_to_col for y in y_true])
    if not np.any(valid_true_mask):
        return {}

    probs = probs[valid_true_mask]
    y_true_valid = y_true[valid_true_mask]
    y_pred_valid = y_pred[valid_true_mask]

    confidences = np.max(probs, axis=1)
    sorted_probs = np.sort(probs, axis=1)
    if probs.shape[1] >= 2:
        margins = sorted_probs[:, -1] - sorted_probs[:, -2]
    else:
        margins = np.ones_like(confidences)

    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)

    true_cols = np.array([label_to_col[int(y)] for y in y_true_valid])
    true_probs = probs[np.arange(len(probs)), true_cols]
    nll = -np.mean(np.log(np.maximum(true_probs, 1e-12)))

    onehot = np.zeros_like(probs)
    onehot[np.arange(len(probs)), true_cols] = 1.0
    brier = np.mean(np.sum((probs - onehot) ** 2, axis=1))

    correct_mask = y_pred_valid == y_true_valid
    avg_correct = float(np.mean(confidences[correct_mask])) if np.any(correct_mask) else None
    avg_wrong = float(np.mean(confidences[~correct_mask])) if np.any(~correct_mask) else None

    return {
        "avg_confidence": float(np.mean(confidences)),
        "avg_confidence_correct": avg_correct,
        "avg_confidence_wrong": avg_wrong,
        "avg_margin": float(np.mean(margins)),
        "avg_entropy": float(np.mean(entropy)),
        "nll": float(nll),
        "brier_score": float(brier),
        "ece": _expected_calibration_error(y_true_valid, y_pred_valid, confidences, n_bins=n_bins),
        "mce": _maximum_calibration_error(y_true_valid, y_pred_valid, confidences, n_bins=n_bins),
    }


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    label_names: Dict[int, str],
    y_proba: Optional[Sequence[Sequence[float]]] = None,
    n_bins: int = 10,
) -> Dict:
    labels = sorted(int(k) for k in label_names.keys())
    y_true = [int(x) for x in y_true]
    y_pred = [int(x) for x in y_pred]
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "num_examples": len(y_true),
    }
    metrics.update(compute_confidence_metrics(y_true, y_pred, y_proba, labels=labels, n_bins=n_bins))
    return metrics


def add_comparison_metrics(clean_metrics: Dict, other_metrics: Dict) -> Dict:
    out = dict(other_metrics)
    if "accuracy" in clean_metrics and "accuracy" in other_metrics:
        out["accuracy_drop_from_clean"] = float(clean_metrics["accuracy"] - other_metrics["accuracy"])
        out["relative_accuracy"] = float(other_metrics["accuracy"] / max(clean_metrics["accuracy"], 1e-12))
    for key in ["avg_confidence", "nll", "brier_score", "ece", "avg_margin", "avg_entropy"]:
        if key in clean_metrics and key in other_metrics and clean_metrics[key] is not None and other_metrics[key] is not None:
            out[f"{key}_change_from_clean"] = float(other_metrics[key] - clean_metrics[key])
    return out


def add_recovery_metrics(clean_metrics: Dict, noisy_metrics: Dict, cleaned_metrics: Dict) -> Dict:
    out = dict(cleaned_metrics)
    clean_acc = clean_metrics.get("accuracy")
    noisy_acc = noisy_metrics.get("accuracy")
    cleaned_acc = cleaned_metrics.get("accuracy")
    if clean_acc is not None and noisy_acc is not None and cleaned_acc is not None:
        denom = clean_acc - noisy_acc
        out["accuracy_recovery_rate"] = float((cleaned_acc - noisy_acc) / denom) if abs(denom) > 1e-12 else None
        out["accuracy_drop_from_clean"] = float(clean_acc - cleaned_acc)

    # For loss/calibration metrics lower is better, so "recovery" means reducing noisy damage.
    for key in ["nll", "brier_score", "ece"]:
        clean_v, noisy_v, cleaned_v = clean_metrics.get(key), noisy_metrics.get(key), cleaned_metrics.get(key)
        if clean_v is not None and noisy_v is not None and cleaned_v is not None:
            denom = noisy_v - clean_v
            out[f"{key}_recovery_rate"] = float((noisy_v - cleaned_v) / denom) if abs(denom) > 1e-12 else None
            out[f"{key}_change_from_clean"] = float(cleaned_v - clean_v)

    # For confidence/margin higher is usually more confident; recovery restores clean level.
    for key in ["avg_confidence", "avg_margin"]:
        clean_v, noisy_v, cleaned_v = clean_metrics.get(key), noisy_metrics.get(key), cleaned_metrics.get(key)
        if clean_v is not None and noisy_v is not None and cleaned_v is not None:
            denom = clean_v - noisy_v
            out[f"{key}_recovery_rate"] = float((cleaned_v - noisy_v) / denom) if abs(denom) > 1e-12 else None
            out[f"{key}_change_from_clean"] = float(cleaned_v - clean_v)
    return out
