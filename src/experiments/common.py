"""Shared helpers for the staged benchmark experiments."""
import json
import os
import numpy as np
import yaml
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, accuracy_score,
)

RESULTS_PATH = "experiments_results/benchmark_results.json"
STAGE_CKPT_DIR = "models/stage_checkpoints"


def load_config(path: str = "config/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split(processed_dir: str, name: str):
    """Returns (X float32 numpy, y int numpy, ctype str numpy) for train/val/test."""
    X = np.load(os.path.join(processed_dir, f"X_{name}.npy"))
    y = np.load(os.path.join(processed_dir, f"y_{name}.npy")).astype(int)
    ctype = np.load(os.path.join(processed_dir, f"type_{name}.npy"), allow_pickle=True)
    return X, y, ctype


def classification_metrics(y_true, prob, threshold: float = 0.5) -> dict:
    """ROC-AUC plus precision/recall/F1/accuracy at the given threshold.

    With threshold=None only ranking-based metrics are computed (used for
    unsupervised anomaly scores, where a separate tau decides flagging).
    """
    y_true = np.asarray(y_true).astype(int)
    out = {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "n_samples": int(len(y_true)),
        "n_positives": int(y_true.sum()),
    }
    if threshold is None:
        out.update({"f1": None, "precision": None, "recall": None, "accuracy": None,
                    "threshold": None})
        return out
    pred = (np.asarray(prob) >= threshold).astype(int)
    out.update({
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "threshold": float(threshold),
    })
    return out


def best_f1_threshold(y_true, prob) -> float:
    """Threshold maximising F1 on the given (probability, label) pairs."""
    best_t, best_f1 = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.05):
        f1 = f1_score(y_true, (prob >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return round(best_t, 2)


def save_stage_results(stage: str, payload: dict):
    """Append/replace one stage's results in the shared benchmark JSON."""
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    results = {}
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = {}
    results[stage] = payload
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[results] {stage} written to {RESULTS_PATH}")


def print_metrics(title: str, m: dict):
    print(f"--- {title} ---")
    print(f"  ROC-AUC: {m['roc_auc']}   F1: {m['f1']}   precision: {m['precision']}   "
          f"recall: {m['recall']}   accuracy: {m['accuracy']}   (threshold={m['threshold']}, "
          f"n={m['n_samples']}, positives={m['n_positives']})")


def count_by_type(ctype, y, name):
    """Print the consumer-type x label cross-tabulation for one split."""
    ind = ctype == "industrial"
    print(f"[counts] {name:<5} total={len(y):>6}  industrial={ind.sum():>4} "
          f"(normal {(y[ind] == 0).sum():>3} / theft {(y[ind] == 1).sum():>3})  "
          f"residential={(~ind).sum():>6} "
          f"(normal {(y[~ind] == 0).sum():>6} / theft {(y[~ind] == 1).sum():>5})")
    return {"total": int(len(y)), "industrial_total": int(ind.sum()),
            "industrial_normal": int((y[ind] == 0).sum()),
            "industrial_theft": int((y[ind] == 1).sum()),
            "residential_total": int((~ind).sum()),
            "residential_normal": int((y[~ind] == 0).sum()),
            "residential_theft": int((y[~ind] == 1).sum())}
