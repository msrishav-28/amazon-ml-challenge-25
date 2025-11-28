"""
Metrics and evaluation utilities for the Amazon ML Challenge.

This module provides SMAPE calculation and comprehensive evaluation metrics
for price prediction models.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-10) -> float:
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).
    
    SMAPE formula: 100 * mean(|pred - true| / ((|true| + |pred|) / 2 + epsilon))
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero
        
    Returns:
        SMAPE value as a percentage (0-200 range, typically 0-100)
        
    Validates: Requirements 9.1
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    
    smape = 100.0 * np.mean(numerator / denominator)
    
    return float(smape)


def smape_scorer(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    SMAPE scorer compatible with scikit-learn's scoring interface.
    
    Returns negative SMAPE for maximization (sklearn convention).
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Negative SMAPE value (for use with sklearn's maximization)
    """
    return -calculate_smape(y_true, y_pred)


def calculate_metrics_by_quantile(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    n_quantiles: int = 5
) -> pd.DataFrame:
    """
    Calculate SMAPE stratified by price quantiles for error analysis.
    
    This helps identify if the model performs differently across price ranges.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        n_quantiles: Number of quantiles to create (default: 5 for quintiles)
        
    Returns:
        DataFrame with columns: quantile, price_range, count, smape
        
    Validates: Requirements 9.4
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Create quantile bins based on true values
    quantile_edges = np.percentile(y_true, np.linspace(0, 100, n_quantiles + 1))
    quantile_labels = np.digitize(y_true, quantile_edges[1:-1])
    
    results = []
    for q in range(n_quantiles):
        mask = quantile_labels == q
        if np.sum(mask) == 0:
            continue
            
        q_true = y_true[mask]
        q_pred = y_pred[mask]
        
        q_smape = calculate_smape(q_true, q_pred)
        
        price_min = np.min(q_true)
        price_max = np.max(q_true)
        price_range = f"[{price_min:.2f}, {price_max:.2f}]"
        
        results.append({
            'quantile': q + 1,
            'price_range': price_range,
            'count': np.sum(mask),
            'smape': q_smape
        })
    
    return pd.DataFrame(results)


def evaluate_predictions(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    split_name: str = 'validation',
    in_log_space: bool = False
) -> Dict[str, float]:
    """
    Comprehensive evaluation with multiple metrics.
    
    Computes SMAPE, MAE, RMSE, MAPE, and R² for model evaluation.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        split_name: Name of the split being evaluated (for logging)
        in_log_space: If True, convert from log space before calculating metrics
        
    Returns:
        Dictionary containing: smape, mae, rmse, mape, r2
        
    Validates: Requirements 9.2, 9.3, 9.5
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # Convert from log space if needed
    if in_log_space:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)
    
    # Calculate all metrics
    smape = calculate_smape(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # MAPE calculation with epsilon to avoid division by zero
    mape = 100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10)))
    
    # R² score
    r2 = r2_score(y_true, y_pred)
    
    metrics = {
        'smape': float(smape),
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2)
    }
    
    # Print formatted summary
    print(f"\n{'='*60}")
    print(f"{split_name.upper()} METRICS")
    print(f"{'='*60}")
    print(f"SMAPE:  {smape:8.4f}%")
    print(f"MAE:    {mae:8.4f}")
    print(f"RMSE:   {rmse:8.4f}")
    print(f"MAPE:   {mape:8.4f}%")
    print(f"R²:     {r2:8.4f}")
    print(f"{'='*60}\n")
    
    return metrics
