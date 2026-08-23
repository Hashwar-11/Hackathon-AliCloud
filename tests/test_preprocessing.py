import numpy as np
from src.preprocessing.preprocess import impute_missing, cap_outliers, min_max_scale, assign_consumer_type_proxy


def test_impute_missing_averages_neighbors():
    x = np.array([1.0, np.nan, 3.0])
    result = impute_missing(x)
    assert result[1] == 2.0


def test_impute_missing_both_neighbors_nan_becomes_zero():
    x = np.array([np.nan, np.nan, np.nan])
    result = impute_missing(x)
    assert np.all(result == 0.0)


def test_cap_outliers_reduces_extreme_value():
    # Realistic-length sequence (SGCC rows are 1035 days). At this scale one
    # outlier day does not swamp mu/sigma the way it does in a tiny array —
    # a 5-element array with one huge outlier inflates its own cap and won't
    # get capped at all. Test at the scale the function is actually used at.
    rng = np.random.default_rng(0)
    x = rng.normal(loc=5.0, scale=1.0, size=1035)
    x[500] = 1000.0  # inject one theft-scale spike
    result = cap_outliers(x, sigma_mult=2)
    assert result[500] < 1000.0


def test_min_max_scale_range():
    x = np.array([0.0, 5.0, 10.0])
    result = min_max_scale(x, 0, 1)
    assert result.min() == 0.0
    assert result.max() == 1.0


def test_consumer_type_proxy_thresholds_correctly():
    low_usage = np.full(10, 20.0)
    high_usage = np.full(10, 800.0)
    assert assign_consumer_type_proxy(low_usage, threshold_kwh=500) == "residential"
    assert assign_consumer_type_proxy(high_usage, threshold_kwh=500) == "industrial"
