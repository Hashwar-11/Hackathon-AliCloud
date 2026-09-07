"""
Preprocessing pipeline for the SGCC electricity theft dataset.

Run:
    python -m src.preprocessing.preprocess --config config/config.yaml

Produces (in config.paths.processed_dir):
    X_train.npy, y_train.npy, type_train.npy
    X_val.npy,   y_val.npy,   type_val.npy
    X_test.npy,  y_test.npy,  type_test.npy

Changes vs. the original version (all documented in PROJECT_OVERVIEW.md):
- Day columns are SORTED CHRONOLOGICALLY before use. The raw CSV stores them in
  lexical order (2014/1/1, 2014/1/10, ..., 2014/1/2, ...), so consuming them in
  file order hands the CNN/BiLSTM a month-internally-shuffled series (§9).
- The per-row transform is fully vectorized (§8.6). The old iterrows() version
  looped over 42,372 rows in pure Python and took hours.
- Imputation keeps the same rule as before (mean of the two immediate ORIGINAL
  neighbours, else 0.0) but is computed vectorized without forward propagation
  of already-imputed values — cleaner and far faster.
- SMOTE now keeps type_train aligned with X_train (§8.2). The encoded consumer
  type rides along as an auxiliary SMOTE feature, so every row — real or
  synthetic — carries a consistent type afterwards.
- Rows that are 100% NaN (5 in the complete dataset) are excluded outright.
- X is saved as float32 (halves the on-disk size; models consume float32).
"""
import argparse
import os
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_sgcc(csv_path: str) -> pd.DataFrame:
    """Loads the raw SGCC CSV: CONS_NO, 1,034 daily-kWh columns, FLAG.
    Also accepts .xlsx/.xls (used for the Pakistani target-domain file)."""
    if str(csv_path).lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(csv_path)
    else:
        df = pd.read_csv(csv_path)
    if "FLAG" not in df.columns:
        raise ValueError(
            "Expected a FLAG column (0=normal, 1=theft). "
            "Check that the extracted CSV matches the SGCC schema."
        )
    return df


def sorted_day_columns(df: pd.DataFrame) -> list:
    """Day columns sorted chronologically (the file stores them in lexical order)."""
    day_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    return sorted(day_cols, key=lambda c: pd.to_datetime(c, format="%Y/%m/%d", errors="coerce"))


def impute_missing(daily_values: np.ndarray) -> np.ndarray:
    """Adjacent temporal linear interpolation: avg(x[i-1], x[i+1]); 0 if both neighbors missing."""
    values = daily_values.copy().astype(float)
    n = len(values)
    for i in range(n):
        if np.isnan(values[i]):
            left = values[i - 1] if i - 1 >= 0 else np.nan
            right = values[i + 1] if i + 1 < n else np.nan
            if not np.isnan(left) and not np.isnan(right):
                values[i] = (left + right) / 2.0
            else:
                values[i] = 0.0
    return values


def impute_missing_matrix(X: np.ndarray) -> np.ndarray:
    """Vectorized equivalent of impute_missing for the full (n_rows, seq_len) matrix.

    Rule per missing cell: mean of the two immediate ORIGINAL neighbours when both
    exist, else 0.0. Leading/trailing NaN runs therefore still become zeros — a
    known weakness (§9: it can mimic the residential theft signature); kept for
    comparability across all benchmark stages and flagged in the report.
    """
    out = X.astype(float).copy()
    nan = np.isnan(out)
    left = np.empty_like(out)
    left[:, 1:] = out[:, :-1]
    left[:, 0] = np.nan
    right = np.empty_like(out)
    right[:, :-1] = out[:, 1:]
    right[:, -1] = np.nan
    both = nan & ~np.isnan(left) & ~np.isnan(right)
    out[both] = (left[both] + right[both]) / 2.0
    out[np.isnan(out)] = 0.0
    return out


def cap_outliers(daily_values: np.ndarray, sigma_mult: float) -> np.ndarray:
    """Cap any reading exceeding mu + sigma_mult*sigma."""
    mu, sigma = np.nanmean(daily_values), np.nanstd(daily_values)
    cap = mu + sigma_mult * sigma
    return np.minimum(daily_values, cap)


def cap_outliers_matrix(X: np.ndarray, sigma_mult: float) -> np.ndarray:
    """Vectorized per-row cap at mu + sigma_mult*sigma."""
    mu = np.nanmean(X, axis=1, keepdims=True)
    sigma = np.nanstd(X, axis=1, keepdims=True)
    return np.minimum(X, mu + sigma_mult * sigma)


def min_max_scale(daily_values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    dmin, dmax = daily_values.min(), daily_values.max()
    if dmax - dmin == 0:
        return np.full_like(daily_values, lo)
    return lo + (daily_values - dmin) * (hi - lo) / (dmax - dmin)


def min_max_scale_matrix(X: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Vectorized per-row min-max scaling. Per-row on purpose: it removes absolute
    magnitude so the detectors see shape; the consumer-type proxy is computed on
    RAW values before this step."""
    dmin = X.min(axis=1, keepdims=True)
    dmax = X.max(axis=1, keepdims=True)
    rng = dmax - dmin
    out = np.where(rng > 0, lo + (X - dmin) * (hi - lo) / np.maximum(rng, 1e-12), lo)
    return out


def assign_consumer_type_proxy(daily_values: np.ndarray, threshold_kwh: float) -> str:
    """
    HEURISTIC ONLY. SGCC has no real consumer-type field.
    Anything averaging above `threshold_kwh`/day is routed to the Industrial Agent.
    This WILL misclassify small industrial users and unusually large households.
    State this limitation in any report using this pipeline.
    """
    return "industrial" if np.nanmean(daily_values) >= threshold_kwh else "residential"


def assign_consumer_type_vectorized(raw_matrix: np.ndarray, threshold_kwh: float) -> np.ndarray:
    """Same magnitude heuristic as assign_consumer_type_proxy, applied to all rows.
    Computed on RAW (unscaled) values — after min-max scaling every row's mean is
    ~0.5 and the threshold is meaningless."""
    means = np.nanmean(raw_matrix, axis=1)
    return np.where(means >= threshold_kwh, "industrial", "residential")


def preprocess(df: pd.DataFrame, cfg: dict):
    day_cols = sorted_day_columns(df)
    X_raw = df[day_cols].to_numpy(dtype=float)

    # Drop rows with no usable readings at all (5 in the complete dataset).
    keep = ~np.isnan(X_raw).all(axis=1)
    if (~keep).any():
        print(f"[CHECK] Excluded {(~keep).sum()} all-NaN rows")
        X_raw = X_raw[keep]

    # Consumer-type proxy on RAW kWh, before any scaling (§4.2 / §8.5).
    ctype = assign_consumer_type_vectorized(
        X_raw, cfg["consumer_type_proxy"]["daily_kwh_industrial_threshold"]
    )

    X = impute_missing_matrix(X_raw)
    X = cap_outliers_matrix(X, cfg["preprocessing"]["outlier_sigma_cap"])
    X = min_max_scale_matrix(X, *cfg["preprocessing"]["scale_range"])

    y = df["FLAG"].to_numpy(dtype=int)[keep]
    return X.astype(np.float32), y, ctype


def split_and_balance(X, y, ctype, cfg):
    val_size = cfg["preprocessing"]["val_split"]
    test_size = cfg["preprocessing"]["test_split"]
    seed = cfg["preprocessing"]["random_seed"]

    X_train, X_temp, y_train, y_temp, ct_train, ct_temp = train_test_split(
        X, y, ctype, test_size=(val_size + test_size), stratify=y, random_state=seed
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test, ct_val, ct_test = train_test_split(
        X_temp, y_temp, ct_temp, test_size=relative_test, stratify=y_temp, random_state=seed
    )

    if cfg["preprocessing"]["smote_on_train_only"]:
        # §8.2 fix: carry the encoded consumer type through SMOTE as an auxiliary
        # feature, so synthetic rows inherit a plausible type and type_train stays
        # aligned with X_train. SMOTE interpolates between same-class neighbours,
        # so the interpolated type stays inside [0, 1]; threshold at 0.5.
        ct_enc = (ct_train == "industrial").astype(np.float32).reshape(-1, 1)
        sm = SMOTE(random_state=seed)
        X_aug, y_train = sm.fit_resample(np.hstack([X_train, ct_enc]), y_train)
        X_train = X_aug[:, :-1].astype(np.float32)
        ct_train = np.where(X_aug[:, -1] >= 0.5, "industrial", "residential")

    # Sanity check you must run before trusting anything downstream:
    print(f"[CHECK] Train class balance:  {np.bincount(y_train)}")
    print(f"[CHECK] Val class balance:    {np.bincount(y_val)}  (must NOT be balanced)")
    print(f"[CHECK] Test class balance:   {np.bincount(y_test)}  (must NOT be balanced)")
    print(f"[CHECK] Train consumer types: {np.unique(ct_train, return_counts=True)}")

    return (X_train, y_train, ct_train), (X_val, y_val, ct_val), (X_test, y_test, ct_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_sgcc(cfg["paths"]["raw_sgcc_csv"])
    print(f"[CHECK] Loaded {len(df)} rows from {cfg['paths']['raw_sgcc_csv']}")
    if len(df) != 42372:
        print("[CHECK] WARNING: expected 42,372 rows for the COMPLETE SGCC dataset. "
              "33,841 means you are on the truncated copy — stop and fix the config path.")
    print(f"[CHECK] Theft rate: {df['FLAG'].mean():.4f}  "
          f"(complete dataset ~0.0853; truncated copy ~0.1068)")

    X, y, ctype = preprocess(df, cfg)
    train, val, test = split_and_balance(X, y, ctype, cfg)

    out_dir = cfg["paths"]["processed_dir"]
    os.makedirs(out_dir, exist_ok=True)
    names = ["train", "val", "test"]
    for name, (Xs, ys, cts) in zip(names, [train, val, test]):
        np.save(os.path.join(out_dir, f"X_{name}.npy"), Xs)
        np.save(os.path.join(out_dir, f"y_{name}.npy"), ys)
        np.save(os.path.join(out_dir, f"type_{name}.npy"), cts)

    print(f"Saved processed splits to {out_dir}")


if __name__ == "__main__":
    main()
