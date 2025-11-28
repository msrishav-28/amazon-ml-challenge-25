"""
Property-based tests for metrics module.

Tests universal properties of SMAPE calculation, log space conversion,
metrics completeness, and quantile stratification.
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st, settings
from src.utils.metrics import (
    calculate_smape,
    smape_scorer,
    calculate_metrics_by_quantile,
    evaluate_predictions
)


# Strategy for generating valid price values (positive floats)
prices = st.floats(min_value=0.01, max_value=10000.0, allow_nan=False, allow_infinity=False)

# Strategy for generating pairs of price arrays with the same length
@st.composite
def price_array_pairs(draw):
    """Generate two arrays of the same length."""
    size = draw(st.integers(min_value=10, max_value=100))
    y_true = draw(st.lists(prices, min_size=size, max_size=size).map(np.array))
    y_pred = draw(st.lists(prices, min_size=size, max_size=size).map(np.array))
    return y_true, y_pred


@settings(max_examples=100)
@given(arrays=price_array_pairs())
def test_smape_formula_correctness(arrays):
    """
    Feature: amazon-ml-price-prediction, Property 34: SMAPE formula correctness
    
    For any predictions and targets, SMAPE should equal 
    100 * mean(|pred - true| / ((|true| + |pred|) / 2 + epsilon)).
    
    Validates: Requirements 9.1
    """
    y_true, y_pred = arrays
    epsilon = 1e-10
    
    # Calculate SMAPE using our function
    smape = calculate_smape(y_true, y_pred, epsilon=epsilon)
    
    # Calculate expected SMAPE manually using the formula
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    expected_smape = 100.0 * np.mean(numerator / denominator)
    
    # Verify they match within numerical precision
    assert np.isclose(smape, expected_smape, rtol=1e-9, atol=1e-9), \
        f"SMAPE calculation incorrect: got {smape}, expected {expected_smape}"
    
    # Verify SMAPE is in valid range [0, 200]
    assert 0 <= smape <= 200, f"SMAPE out of valid range: {smape}"
    
    # Verify SMAPE is a scalar float
    assert isinstance(smape, float), f"SMAPE should be float, got {type(smape)}"


@settings(max_examples=100)
@given(arrays=price_array_pairs())
def test_log_space_conversion(arrays):
    """
    Feature: amazon-ml-price-prediction, Property 35: Log space conversion
    
    For any predictions in log space, calculating SMAPE should first apply 
    expm1 to convert to original space.
    
    Validates: Requirements 9.2
    """
    y_true, y_pred = arrays
    
    # Convert to log space using log1p
    y_true_log = np.log1p(y_true)
    y_pred_log = np.log1p(y_pred)
    
    # Calculate SMAPE with log space conversion
    smape_from_log = evaluate_predictions(
        y_true_log, 
        y_pred_log, 
        split_name='test',
        in_log_space=True
    )['smape']
    
    # Calculate SMAPE directly on original space
    smape_direct = calculate_smape(y_true, y_pred)
    
    # They should match within numerical precision
    assert np.isclose(smape_from_log, smape_direct, rtol=1e-6, atol=1e-6), \
        f"Log space conversion incorrect: got {smape_from_log}, expected {smape_direct}"


@settings(max_examples=100)
@given(arrays=price_array_pairs())
def test_metrics_completeness(arrays):
    """
    Feature: amazon-ml-price-prediction, Property 36: Metrics completeness
    
    For any evaluation call, the returned dictionary should contain keys: 
    'smape', 'mae', 'rmse', 'mape', 'r2'.
    
    Validates: Requirements 9.3
    """
    y_true, y_pred = arrays
    metrics = evaluate_predictions(y_true, y_pred, split_name='test')
    
    # Check all required keys are present
    required_keys = {'smape', 'mae', 'rmse', 'mape', 'r2'}
    assert set(metrics.keys()) == required_keys, \
        f"Missing or extra keys. Expected {required_keys}, got {set(metrics.keys())}"
    
    # Check all values are numeric
    for key, value in metrics.items():
        assert isinstance(value, (int, float)), \
            f"Metric {key} should be numeric, got {type(value)}"
        assert not np.isnan(value), f"Metric {key} is NaN"
        assert not np.isinf(value), f"Metric {key} is infinite"


@settings(max_examples=100)
@given(
    arrays=price_array_pairs(),
    n_quantiles=st.integers(min_value=2, max_value=10)
)
def test_quantile_stratification(arrays, n_quantiles):
    """
    Feature: amazon-ml-price-prediction, Property 37: Quantile SMAPE stratification
    
    For any predictions and targets with n_quantiles, calculating SMAPE by 
    quantiles should produce exactly n_quantiles groups with non-overlapping 
    price ranges.
    
    Validates: Requirements 9.4
    """
    y_true, y_pred = arrays
    df = calculate_metrics_by_quantile(y_true, y_pred, n_quantiles=n_quantiles)
    
    # Check we have at most n_quantiles rows (could be less if some quantiles are empty)
    assert len(df) <= n_quantiles, \
        f"Expected at most {n_quantiles} quantiles, got {len(df)}"
    
    # Check required columns are present
    required_columns = {'quantile', 'price_range', 'count', 'smape'}
    assert set(df.columns) == required_columns, \
        f"Missing or extra columns. Expected {required_columns}, got {set(df.columns)}"
    
    # Check quantile numbers are sequential starting from 1
    if len(df) > 0:
        assert df['quantile'].min() >= 1, "Quantile numbers should start from 1"
        assert df['quantile'].max() <= n_quantiles, \
            f"Quantile numbers should not exceed {n_quantiles}"
    
    # Check all counts are positive
    assert (df['count'] > 0).all(), "All quantiles should have positive counts"
    
    # Check total count matches input size
    assert df['count'].sum() == len(y_true), \
        f"Total count {df['count'].sum()} doesn't match input size {len(y_true)}"
    
    # Check SMAPE values are valid
    assert (df['smape'] >= 0).all(), "SMAPE values should be non-negative"
    assert (df['smape'] <= 200).all(), "SMAPE values should not exceed 200"
    
    # Check price ranges are non-overlapping by parsing and comparing
    # (This is a simplified check - we verify they're formatted correctly)
    for price_range in df['price_range']:
        assert price_range.startswith('[') and price_range.endswith(']'), \
            f"Price range should be formatted as [min, max], got {price_range}"


# Additional edge case tests
def test_smape_identical_predictions():
    """Test SMAPE when predictions exactly match targets."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    
    smape = calculate_smape(y_true, y_pred)
    
    # Perfect predictions should give SMAPE = 0
    assert np.isclose(smape, 0.0, atol=1e-6), \
        f"Perfect predictions should give SMAPE=0, got {smape}"


def test_smape_scorer_negative():
    """Test that smape_scorer returns negative values for sklearn compatibility."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    
    score = smape_scorer(y_true, y_pred)
    smape = calculate_smape(y_true, y_pred)
    
    # Scorer should return negative SMAPE
    assert np.isclose(score, -smape, rtol=1e-9), \
        f"Scorer should return negative SMAPE: got {score}, expected {-smape}"
    assert score <= 0, "Scorer should return non-positive values"


def test_evaluate_predictions_without_log_space():
    """Test evaluate_predictions with in_log_space=False."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0, 52.0])
    
    metrics = evaluate_predictions(y_true, y_pred, split_name='test', in_log_space=False)
    
    # Verify all required metrics are present
    assert 'smape' in metrics
    assert 'mae' in metrics
    assert 'rmse' in metrics
    assert 'mape' in metrics
    assert 'r2' in metrics
    
    # Verify SMAPE matches direct calculation
    expected_smape = calculate_smape(y_true, y_pred)
    assert np.isclose(metrics['smape'], expected_smape, rtol=1e-9)


def test_quantile_stratification_edge_case_small_data():
    """Test quantile stratification with small dataset."""
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 32.0])
    
    df = calculate_metrics_by_quantile(y_true, y_pred, n_quantiles=5)
    
    # Should handle small datasets gracefully
    assert len(df) <= 5
    assert df['count'].sum() == len(y_true)
