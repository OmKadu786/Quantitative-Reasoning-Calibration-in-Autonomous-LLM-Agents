"""
QuaRCAA Directional Accuracy Calculator
Explicit Baseline Definition:
- Iteration 1: Baseline_1 = Vanilla pre-agent pipeline 3-seed mean.
- Iteration i > 1: Baseline_i = Empirical 3-seed mean of iteration i-1.

Calculates both LLM Agent Directional Accuracy and Trivial "Always Predict UP" Baseline Control per metric.
"""
from typing import Dict, Any, Tuple

def compute_directional_accuracy(
    predicted_direction: str, 
    actual_3seed_mean: float, 
    baseline_val: float, 
    direction_threshold: float = 0.001,
    stable_threshold: float = 0.01
) -> bool:
    """
    Checks if predicted metric shift direction matches actual 3-seed execution mean shift.
    """
    actual_delta = actual_3seed_mean - baseline_val
    pred_dir = predicted_direction.upper()
    
    if pred_dir == "UP":
        return actual_delta > direction_threshold
    elif pred_dir == "DOWN":
        return actual_delta < -direction_threshold
    else: # STABLE
        return abs(actual_delta) <= stable_threshold

def compute_trivial_always_up_accuracy(
    actual_3seed_mean: float, 
    baseline_val: float, 
    direction_threshold: float = 0.001
) -> bool:
    """
    Control Baseline: Checks if a trivial 'Always Predict UP' rule is correct for this metric iteration.
    """
    actual_delta = actual_3seed_mean - baseline_val
    return actual_delta > direction_threshold
