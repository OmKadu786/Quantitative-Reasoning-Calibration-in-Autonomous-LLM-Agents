"""
QuaRCAA Calibration Metrics — Self-contained module.
Computes MACE, RMACE, Directional Accuracy, Sharpness, and Overconfidence Rate.
All helper functions inlined here — no external metric submodule imports needed.
"""
import numpy as np
import random
from typing import Dict, Any

EPSILON = 0.001           # Stabilization parameter for Relative MACE denominator
RMACE_CEILING = 10.0      # Winsorized cap — prevents degenerate spikes when step delta ≈ 0


def _directional_accuracy(pred_dir: str, actual_val: float, base_val: float) -> bool:
    """True if agent's predicted direction matches actual change direction."""
    actual_delta = actual_val - base_val
    if abs(actual_delta) < 1e-9:
        return False  # No real movement — directional prediction is meaningless
    if pred_dir == "UP":
        return actual_delta > 0
    elif pred_dir == "DOWN":
        return actual_delta < 0
    return False


def _random_baseline_accuracy(actual_val: float, base_val: float, rng_seed: int) -> bool:
    """50/50 random coin flip per metric — reproducible via seed derived from trajectory state."""
    rng = random.Random(rng_seed)
    random_dir = "UP" if rng.random() >= 0.5 else "DOWN"
    return _directional_accuracy(random_dir, actual_val, base_val)


def compute_quarcaa_calibration(
    predictions: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    actual_3seed_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Core QuaRCAA calibration diagnostic.

    Per metric:
      - MACE: |predicted_midpoint - actual_3seed_mean|
      - RMACE: MACE / (|actual_delta| + epsilon), capped at 10.0
      - predicted_delta vs actual_delta: magnitude comparison readable directly from logs
      - Directional accuracy (agent vs random baseline)
      - Overconfidence: actual fell outside predicted interval

    Aggregated:
      - Mean/Median RMACE, Mean MACE, Overconfidence Rate, Directional Accuracy
    """
    results = {}
    censored_count = 0

    for metric_name, pred_data in predictions.items():
        if metric_name not in actual_3seed_metrics:
            continue

        pred_dir      = pred_data.get("direction", "STABLE").upper()
        exp_min       = float(pred_data.get("expected_min", 0.0))
        exp_max       = float(pred_data.get("expected_max", 1.0))
        target_midpoint = (exp_min + exp_max) / 2.0
        interval_width  = exp_max - exp_min

        base_val    = float(baseline_metrics.get(metric_name, 0.0))
        actual_val  = float(actual_3seed_metrics.get(metric_name, 0.0))
        actual_delta = actual_val - base_val

        # Directional accuracy
        agent_correct  = _directional_accuracy(pred_dir, actual_val, base_val)
        rng_seed       = abs(hash(metric_name + str(round(base_val, 4)))) % (2**31)
        random_correct = _random_baseline_accuracy(actual_val, base_val, rng_seed=rng_seed)

        # Absolute calibration error
        ace = abs(target_midpoint - actual_val)

        # Relative calibration error
        rce_uncapped = ace / (abs(actual_delta) + EPSILON)
        if rce_uncapped >= RMACE_CEILING:
            censored_count += 1
        rce_capped = min(rce_uncapped, RMACE_CEILING)

        # Overconfidence: actual landed outside predicted interval
        if pred_dir == "DOWN":
            is_overconfident = actual_val > exp_max
        else:
            is_overconfident = actual_val < exp_min

        results[metric_name] = {
            "predicted_direction":             pred_dir,
            "baseline_val":                    round(base_val, 6),
            "actual_3seed_mean":               round(actual_val, 6),
            "actual_delta":                    round(actual_delta, 6),
            "predicted_delta":                 round(target_midpoint - base_val, 6),
            "agent_directional_correct":       bool(agent_correct),
            "random_baseline_correct":         bool(random_correct),
            "expected_range":                  [exp_min, exp_max],
            "interval_width":                  round(interval_width, 6),
            "target_midpoint":                 round(target_midpoint, 6),
            "absolute_calibration_error":      round(ace, 6),
            "relative_calibration_error_uncapped": round(rce_uncapped, 4),
            "relative_calibration_error_capped":   round(rce_capped, 4),
            "is_censored_at_ceiling":          rce_uncapped >= RMACE_CEILING,
            "is_overconfident":                bool(is_overconfident)
        }

    if not results:
        return {"metric_details": {}, "summary": {}}

    vals = list(results.values())
    mean_sharpness = float(np.mean([v["interval_width"] for v in vals]))
    rce_vals       = [v["relative_calibration_error_capped"] for v in vals]

    return {
        "metric_details": results,
        "summary": {
            "agent_directional_accuracy_rate":  float(np.mean([v["agent_directional_correct"] for v in vals])),
            "random_baseline_accuracy_rate":    float(np.mean([v["random_baseline_correct"] for v in vals])),
            "mace":                             float(np.mean([v["absolute_calibration_error"] for v in vals])),
            "mean_relative_mace":               float(np.mean(rce_vals)),
            "median_relative_mace":             float(np.median(rce_vals)),
            "rmace_censored_count":             censored_count,
            "total_metrics_evaluated":          len(results),
            "rmace_epsilon":                    EPSILON,
            "rmace_winsorized_ceiling":         RMACE_CEILING,
            "mean_sharpness":                   mean_sharpness,
            "overconfidence_rate":              float(np.mean([v["is_overconfident"] for v in vals])),
            "per_metric_agent_directional_acc": {k: v["agent_directional_correct"] for k, v in results.items()},
            "per_metric_random_baseline_acc":   {k: v["random_baseline_correct"]   for k, v in results.items()}
        }
    }
