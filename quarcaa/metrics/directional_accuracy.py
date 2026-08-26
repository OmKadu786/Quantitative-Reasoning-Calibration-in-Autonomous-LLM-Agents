"""
QuaRCAA Directional Accuracy Calculator
Explicit Baseline Definition:
- Iteration 1: Baseline_1 = Vanilla pre-agent pipeline 3-seed mean.
- Iteration i > 1: Baseline_i = Empirical 3-seed mean of iteration i-1.

Calculates both LLM Agent Directional Accuracy and Random Baseline Control per metric.
Random Baseline: assigns UP/DOWN uniformly at random per metric per iteration.
Expected accuracy converges to 50% over many trials by definition.
This makes zero domain assumptions (unlike Always-UP which encodes search space monotonicity).
"""
import numpy as np
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

def compute_random_baseline_accuracy(
    actual_3seed_mean: float,
    baseline_val: float,
    rng_seed: int,
    direction_threshold: float = 0.001
) -> bool:
    """
    Control Baseline: Randomly guesses UP or DOWN with 50/50 probability per metric per iteration.
    Uses deterministic rng_seed for reproducibility of the baseline across reruns.
    Expected accuracy = 50% over many trials by construction.
    Unlike 'Always Predict UP', this makes no domain assumption about search space structure.
    """
    rng = np.random.default_rng(rng_seed)
    random_guess = "UP" if rng.integers(0, 2) == 1 else "DOWN"
    actual_delta = actual_3seed_mean - baseline_val
    if random_guess == "UP":
        return actual_delta > direction_threshold
    else:
        return actual_delta < -direction_threshold
