"""
QuaRCAA Directional Accuracy Calculator
Explicit Baseline Definition:
- Iteration 1: Baseline_1 = Vanilla pre-agent pipeline 3-seed mean.
- Iteration i > 1: Baseline_i = Empirical 3-seed mean of iteration i-1.
"""
from typing import Dict, Any

def compute_directional_accuracy(predicted_direction: str, actual_3seed_mean: float, baseline_val: float, threshold: float = 0.001) -> bool:
    """
    Checks if predicted metric shift direction matches actual 3-seed execution mean shift.
    """
    actual_delta = actual_3seed_mean - baseline_val
    pred_dir = predicted_direction.upper()
    
    if pred_dir == "UP":
        return actual_delta > threshold
    elif pred_dir == "DOWN":
        return actual_delta < -threshold
    else: # STABLE
        return abs(actual_delta) <= 0.01
