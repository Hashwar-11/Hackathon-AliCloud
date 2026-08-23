"""
Preprocessing pipeline for the SGCC electricity theft dataset.

Run:
    python -m src.preprocessing.preprocess --config config/config.yaml

Produces (in config.paths.processed_dir):
    X_train.npy, y_train.npy, type_train.npy
    X_val.npy,   y_val.npy,   type_val.npy
    X_test.npy,  y_test.npy,  type_test.npy
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
    """Loads the raw SGCC CSV: CONS_NO, 1035 daily-kWh columns, FLAG."""
    df = pd.read_csv(csv_path)
    if "FLAG" not in df.columns:
        raise ValueError(
            "Expected a FLAG column (0=normal, 1=theft). "
            "Check that the extracted CSV matches the SGCC schema."
        )
    return df


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


def cap_outliers(daily_values: np.ndarray, sigma_mult: float) -> np.ndarray:
    """Cap any reading exceeding mu + sigma_mult*sigma."""
    mu, sigma = np.nanmean(daily_values), np.nanstd(daily_values)
    cap = mu + sigma_mult * sigma
    return np.minimum(daily_values, cap)


def min_max_scale(daily_values: np.ndarray, lo: float, hi: float) -> np.ndarray:
    dmin, dmax = daily_values.min(), daily_values.max()
    if dmax - dmin == 0:
        return np.full_like(daily_values, lo)
    return lo + (daily_values - dmin) * (hi - lo) / (dmax - dmin)


def assign_consumer_type_proxy(daily_values: np.ndarray, threshold_kwh: float) -> str:
    """
    HEURISTIC ONLY. SGCC has no real consumer-type field.
    Anything averaging above `threshold_kwh`/day is routed to the Industrial Agent.
    This WILL misclassify small industrial users and unusually large households.
    State this limitation in any report using this pipeline.
    """
    return "industrial" if np.nanmean(daily_values) >= threshold_kwh else "residential"


def preprocess(df: pd.DataFrame, cfg: dict):
    day_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    X, y, ctype = [], [], []

    for _, row in df.iterrows():
        series = row[day_cols].values.astype(float)
        series = impute_missing(series)
        series = cap_outliers(series, cfg["preprocessing"]["outlier_sigma_cap"])
        series = min_max_scale(series, *cfg["preprocessing"]["scale_range"])

        X.append(series)
        y.append(int(row["FLAG"]))
        ctype.append(
            assign_consumer_type_proxy(
                row[day_cols].values.astype(float),
                cfg["consumer_type_proxy"]["daily_kwh_industrial_threshold"],
            )
        )

    X = np.array(X)
    y = np.array(y)
    ctype = np.array(ctype)
    return X, y, ctype


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
        sm = SMOTE(random_state=seed)
        X_train, y_train = sm.resample(X_train, y_train) if hasattr(sm, "resample") else sm.fit_resample(X_train, y_train)
        # ct_train is intentionally NOT resampled to match length here — regenerate
        # the proxy type for synthetic rows downstream if you need per-row typing
        # post-SMOTE. Flagging this so it isn't silently wrong.

    # Sanity check you must run before trusting anything downstream:
    print(f"[CHECK] Train class balance:  {np.bincount(y_train)}")
    print(f"[CHECK] Val class balance:    {np.bincount(y_val)}  (must NOT be balanced)")
    print(f"[CHECK] Test class balance:   {np.bincount(y_test)}  (must NOT be balanced)")

    return (X_train, y_train, ct_train), (X_val, y_val, ct_val), (X_test, y_test, ct_test)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    df = load_sgcc(cfg["paths"]["raw_sgcc_csv"])
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
