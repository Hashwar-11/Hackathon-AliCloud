"""
Single-series preprocessing utility for real-time inference.
Applies the identical transform pipeline as batch preprocessing:
1. Interpolate / impute missing readings.
2. Cap outliers at mu + 2*sigma.
3. Per-account min-max scale to [0, 1].
4. Length alignment (pad / truncate) to target_seq_len (default 1034).
5. Infer proxy client type from raw mean consumption.
"""
from typing import List, Tuple, Union
import numpy as np


def impute_single(values: np.ndarray) -> np.ndarray:
    """Missing-value imputation IDENTICAL to the training/batch pipeline.

    Mirrors ``preprocess.impute_missing_matrix`` exactly: a missing cell becomes
    the mean of its two ORIGINAL neighbours ONLY when both exist; otherwise it
    becomes 0.0. Neighbours are always read from the un-imputed input (no forward
    propagation of already-filled values, and no single-neighbour fallback).

    The previous version fell back to a single available neighbour, which produced
    different numbers than training for runs of missing days -> train/serve skew.
    Keeping this byte-for-byte consistent with training is a demo-critical fix.
    """
    original = values.copy().astype(float)
    arr = original.copy()
    n = len(arr)
    for i in range(n):
        if np.isnan(original[i]):
            left = original[i - 1] if i - 1 >= 0 else np.nan
            right = original[i + 1] if i + 1 < n else np.nan
            if not np.isnan(left) and not np.isnan(right):
                arr[i] = (left + right) / 2.0
            else:
                arr[i] = 0.0
    arr[np.isnan(arr)] = 0.0
    return arr


def cap_outliers_single(values: np.ndarray, sigma_mult: float = 2.0) -> np.ndarray:
    """Cap any reading exceeding mu + sigma_mult*sigma."""
    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values))
    cap = mu + sigma_mult * sigma
    return np.minimum(values, cap)


def min_max_scale_single(values: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    """Min-max scale a 1D array to [lo, hi]."""
    dmin, dmax = float(np.min(values)), float(np.max(values))
    rng = dmax - dmin
    if rng <= 1e-12:
        return np.full_like(values, lo)
    return lo + (values - dmin) * (hi - lo) / rng


def align_sequence_length(values: np.ndarray, target_seq_len: int = 1034) -> np.ndarray:
    """
    Pad (left-pad with zero or edge) or truncate series to match target_seq_len.
    For shorter series, left-pads with zeros so recent history occupies the final timesteps.
    For longer series, keeps the most recent target_seq_len days.
    """
    cur_len = len(values)
    if cur_len == target_seq_len:
        return values
    elif cur_len < target_seq_len:
        pad_size = target_seq_len - cur_len
        return np.pad(values, (pad_size, 0), mode="constant", constant_values=0.0)
    else:
        return values[-target_seq_len:]


def transform_daily_series(
    daily_kwh: Union[List[float], np.ndarray],
    target_seq_len: int = 1034,
    sigma_mult: float = 2.0,
    industrial_threshold: float = 500.0,
) -> Tuple[np.ndarray, float, str]:
    """
    Processes an arbitrary list of daily kWh values into a normalized float32 sequence ready for models.

    Returns:
        scaled_series: np.ndarray of shape (target_seq_len,), float32 in [0, 1]
        raw_mean: float, mean daily kWh before scaling
        inferred_client_type: str, 'industrial' if raw_mean >= industrial_threshold else 'residential'
    """
    arr = np.array(daily_kwh, dtype=float)
    if len(arr) == 0:
        return np.zeros(target_seq_len, dtype=np.float32), 0.0, "residential"

    # Compute raw statistics
    raw_mean = float(np.nanmean(arr)) if not np.all(np.isnan(arr)) else 0.0
    inferred_client_type = "industrial" if raw_mean >= industrial_threshold else "residential"

    # 1. Impute missing
    cleaned = impute_single(arr)

    # 2. Cap outliers
    capped = cap_outliers_single(cleaned, sigma_mult=sigma_mult)

    # 3. Min-Max Scale
    scaled = min_max_scale_single(capped, lo=0.0, hi=1.0)

    # 4. Length Alignment
    aligned = align_sequence_length(scaled, target_seq_len=target_seq_len)

    return aligned.astype(np.float32), raw_mean, inferred_client_type
