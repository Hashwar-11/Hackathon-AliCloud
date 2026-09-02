"""
Stage 1 — tabular-feature baseline (XGBoost).

Engineers per-customer statistical/time-series features from the preprocessed
daily series and trains an XGBoost binary classifier. Evaluated on the
untouched (imbalanced) validation split; the untouched test split is reported
as a leak-free final check.

Run:
    python -m src.experiments.stage1_xgboost --config config/config.yaml
"""
import argparse

import numpy as np
from scipy import stats

from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics,
)


def extract_features(X: np.ndarray) -> np.ndarray:
    """Per-customer tabular features from the (scaled) daily series.

    X rows are min-max scaled per account, so magnitude features are
    shape-based; level information lives in the raw domain and was used only
    for the consumer-type proxy at preprocessing time.
    """
    n, T = X.shape
    idx = np.arange(T)

    mean = X.mean(axis=1)
    std = X.std(axis=1)
    median = np.median(X, axis=1)
    p5 = np.percentile(X, 5, axis=1)
    p25 = np.percentile(X, 25, axis=1)
    p75 = np.percentile(X, 75, axis=1)
    p95 = np.percentile(X, 95, axis=1)

    diff1 = np.diff(X, axis=1)
    max_drop = (-diff1).max(axis=1)
    max_rise = diff1.max(axis=1)
    mean_abs_diff = np.abs(diff1).mean(axis=1)

    # least-squares slope over day index, and Spearman rank trend (robust)
    slope = ((X * idx).sum(axis=1) - mean * idx.sum()) / (
        (idx ** 2).sum() - T * idx.mean() ** 2
    )
    rank_corr = np.array([
        stats.spearmanr(idx, row).statistic if row.std() > 0 else 0.0
        for row in X
    ])

    # windowed volatility: std of 30-day means (trim the trailing partial window;
    # 1,034 days = 34 full windows + 14 leftover days)
    nwin = T // 30
    win_means = X[:, : nwin * 30].reshape(n, nwin, 30).mean(axis=2)
    win_std = win_means.std(axis=1)

    # structural features around the theft signature (drop toward zero)
    windowed_min = win_means.min(axis=1)
    crash_ratio = (X < 0.05).mean(axis=1)          # share of near-zero days
    zero_days = (X == 0.0).mean(axis=1)
    below_half = (X < 0.5 * mean[:, None]).mean(axis=1)

    skew = stats.skew(X, axis=1)
    kurt = stats.kurtosis(X, axis=1)

    feats = np.column_stack([
        mean, std, median, p5, p25, p75, p95, p95 - p5,
        zero_days, crash_ratio, below_half,
        max_drop, max_rise, mean_abs_diff,
        slope, rank_corr, win_std, windowed_min, skew, kurt,
    ])
    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


FEATURE_NAMES = [
    "mean", "std", "median", "p5", "p25", "p75", "p95", "p95_minus_p5",
    "zero_days_frac", "crash_ratio", "below_half_mean_frac",
    "max_single_day_drop", "max_single_day_rise", "mean_abs_day_to_day_change",
    "linear_trend_slope", "spearman_rank_trend", "window30_std",
    "window30_min_mean", "skewness", "kurtosis",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]

    X_tr, y_tr, _ = load_split(processed, "train")
    X_va, y_va, _ = load_split(processed, "val")
    X_te, y_te, _ = load_split(processed, "test")

    print(f"[stage1] train n={len(y_tr)} balance={np.bincount(y_tr)} (SMOTE-balanced)")
    print(f"[stage1] val   n={len(y_va)} balance={np.bincount(y_va)} (real-world imbalance)")
    print(f"[stage1] test  n={len(y_te)} balance={np.bincount(y_te)} (real-world imbalance)")

    print("[stage1] Extracting tabular features ...")
    F_tr, F_va, F_te = extract_features(X_tr), extract_features(X_va), extract_features(X_te)

    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            scale_pos_weight=float((y_tr == 0).sum()) / max((y_tr == 1).sum(), 1),
            eval_metric="auc",
            n_jobs=-1,
        )
        backend = "xgboost"
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        print("[stage1] xgboost not installed — falling back to sklearn "
              "HistGradientBoostingClassifier. Install xgboost for the real baseline.")
        clf = HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.1, max_depth=6,
            class_weight="balanced", random_state=42,
        )
        backend = "sklearn_hist_gb"

    print(f"[stage1] Training {backend} ...")
    clf.fit(F_tr, y_tr)

    prob_va = clf.predict_proba(F_va)[:, 1]
    thr = best_f1_threshold(y_va, prob_va)
    val_m = classification_metrics(y_va, prob_va, thr)
    val_m_05 = classification_metrics(y_va, prob_va, 0.5)

    prob_te = clf.predict_proba(F_te)[:, 1]
    test_m = classification_metrics(y_te, prob_te, thr)

    print_metrics("Stage 1 XGBoost — VALIDATION (imbalanced)", val_m)
    print_metrics("Stage 1 XGBoost — validation @0.5", val_m_05)
    print_metrics("Stage 1 XGBoost — TEST (held out, same threshold)", test_m)

    if backend == "xgboost":
        gain = getattr(clf, "feature_importances_", None)
        if gain is not None:
            order = np.argsort(gain)[::-1][:10]
            print("[stage1] Top-10 features by importance:")
            for i in order:
                print(f"    {FEATURE_NAMES[i]:<32} {gain[i]:.4f}")

    save_stage_results("stage1_xgboost", {
        "backend": backend,
        "n_features": F_tr.shape[1],
        "val_metrics": val_m,
        "val_metrics_thr05": val_m_05,
        "test_metrics": test_m,
    })


if __name__ == "__main__":
    main()
