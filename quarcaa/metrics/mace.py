"""
QuaRCAA Calibration Metrics — Self-contained module.
Computes MACE, RMACE, Directional Accuracy, Sharpness, Overconfidence Rate,
and the Detectability Split (signal iterations vs. noise-floor iterations).

Key methodological addition:
  Detectability flag per metric per iteration:
    is_detectable_signal = |μᵢ − μᵢ₋₁| > k · √(σᵢ² + σᵢ₋₁²)
  where k=1.0 (one combined standard deviation — conservative choice).

  This separates iterations where a real pipeline change was distinguishable
  from seed-to-seed noise (signal) from iterations where the change was
  smaller than measurement noise (noise-floor). Directional accuracy on
  noise-floor iterations is meaningless for any predictor, human or LLM.
  Pooling them degrades apparent calibration without reflecting reasoning quality.
"""
import numpy as np
import random
from typing import Dict, Any, Optional

EPSILON = 0.001           # Stabilization parameter for Relative MACE denominator
RMACE_CEILING = 10.0      # Winsorized cap — prevents degenerate spikes when step delta ≈ 0
DETECTABILITY_K = 1.0     # Signal threshold: |Δμ| > k·√(σᵢ² + σᵢ₋₁²)


def _directional_accuracy(pred_dir: str, actual_val: float, base_val: float) -> bool:
    """True if agent's predicted direction matches actual change direction."""
    actual_delta = actual_val - base_val
    if abs(actual_delta) < 1e-9:
        return False
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
    actual_3seed_metrics: Dict[str, float],
    baseline_stds: Optional[Dict[str, float]] = None,
    actual_stds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Core QuaRCAA calibration diagnostic.

    Parameters
    ----------
    predictions          : agent JSON predictions {metric: {direction, expected_min, expected_max}}
    baseline_metrics     : previous iteration's 3-seed means (μᵢ₋₁)
    actual_3seed_metrics : current iteration's 3-seed means (μᵢ)
    baseline_stds        : previous iteration's 3-seed stds (σᵢ₋₁) — enables detectability
    actual_stds          : current iteration's 3-seed stds (σᵢ)   — enables detectability

    Per-metric outputs
    ------------------
    - MACE:              |predicted_midpoint − actual_3seed_mean|
    - RMACE:             MACE / (|actual_delta| + epsilon), capped at 10.0
    - predicted_delta vs actual_delta: direct human-readable magnitude comparison in logs
    - Directional accuracy (agent vs random baseline coin flip)
    - Overconfidence:    actual landed outside predicted interval
    - is_detectable_signal: |Δμ| > k·√(σᵢ² + σᵢ₋₁²) — key novel flag

    Aggregate summary
    -----------------
    - Mean/Median RMACE, Mean MACE, Overconfidence Rate
    - Directional accuracy: pooled AND signal-only (excluding noise-floor iterations)
    - Fraction of iterations that are detectable signal vs. noise-floor
    """
    results = {}
    censored_count = 0

    for metric_name, pred_data in predictions.items():
        if metric_name not in actual_3seed_metrics:
            continue

        pred_dir        = pred_data.get("direction", "STABLE").upper()
        exp_min         = float(pred_data.get("expected_min", 0.0))
        exp_max         = float(pred_data.get("expected_max", 1.0))
        target_midpoint = (exp_min + exp_max) / 2.0
        interval_width  = exp_max - exp_min

        base_val    = float(baseline_metrics.get(metric_name, 0.0))
        actual_val  = float(actual_3seed_metrics.get(metric_name, 0.0))
        actual_delta = actual_val - base_val

        # Detectability: is the step distinguishable from seed-to-seed noise?
        is_detectable = None
        noise_floor   = None
        if baseline_stds is not None and actual_stds is not None:
            sigma_prev = float(baseline_stds.get(metric_name, 0.0))
            sigma_curr = float(actual_stds.get(metric_name, 0.0))
            noise_floor = DETECTABILITY_K * np.sqrt(sigma_prev ** 2 + sigma_curr ** 2)
            is_detectable = bool(abs(actual_delta) > noise_floor)

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
            "predicted_direction":                 pred_dir,
            "baseline_val":                        round(base_val, 6),
            "actual_3seed_mean":                   round(actual_val, 6),
            "actual_delta":                        round(actual_delta, 6),
            "predicted_delta":                     round(target_midpoint - base_val, 6),
            "agent_directional_correct":           bool(agent_correct),
            "random_baseline_correct":             bool(random_correct),
            "expected_range":                      [exp_min, exp_max],
            "interval_width":                      round(interval_width, 6),
            "target_midpoint":                     round(target_midpoint, 6),
            "absolute_calibration_error":          round(ace, 6),
            "relative_calibration_error_uncapped": round(rce_uncapped, 4),
            "relative_calibration_error_capped":   round(rce_capped, 4),
            "is_censored_at_ceiling":              rce_uncapped >= RMACE_CEILING,
            "is_overconfident":                    bool(is_overconfident),
            # Detectability fields (None if stds not provided)
            "is_detectable_signal":                is_detectable,
            "noise_floor_threshold":               round(noise_floor, 6) if noise_floor is not None else None,
        }

    if not results:
        return {"metric_details": {}, "summary": {}}

    vals = list(results.values())
    rce_vals = [v["relative_calibration_error_capped"] for v in vals]

    # Aggregate directional accuracy — pooled
    pooled_agent_acc  = float(np.mean([v["agent_directional_correct"] for v in vals]))
    pooled_random_acc = float(np.mean([v["random_baseline_correct"]   for v in vals]))

    # Signal-only directional accuracy (only iterations where |Δμ| > noise floor)
    signal_vals = [v for v in vals if v["is_detectable_signal"] is True]
    noise_vals  = [v for v in vals if v["is_detectable_signal"] is False]

    signal_agent_acc  = float(np.mean([v["agent_directional_correct"] for v in signal_vals])) if signal_vals else None
    signal_random_acc = float(np.mean([v["random_baseline_correct"]   for v in signal_vals])) if signal_vals else None
    noise_agent_acc   = float(np.mean([v["agent_directional_correct"] for v in noise_vals]))  if noise_vals  else None

    pct_signal = float(len(signal_vals)) / len(vals) if vals else 0.0

    return {
        "metric_details": results,
        "summary": {
            # Core calibration
            "mace":                              float(np.mean([v["absolute_calibration_error"] for v in vals])),
            "mean_relative_mace":                float(np.mean(rce_vals)),
            "median_relative_mace":              float(np.median(rce_vals)),
            "rmace_censored_count":              censored_count,
            "total_metrics_evaluated":           len(results),
            "rmace_epsilon":                     EPSILON,
            "rmace_winsorized_ceiling":          RMACE_CEILING,
            "mean_sharpness":                    float(np.mean([v["interval_width"] for v in vals])),
            "overconfidence_rate":               float(np.mean([v["is_overconfident"] for v in vals])),
            # Directional accuracy — pooled (all iterations)
            "agent_directional_accuracy_rate":   pooled_agent_acc,
            "random_baseline_accuracy_rate":     pooled_random_acc,
            # Directional accuracy — split by detectability
            "signal_agent_directional_acc":      signal_agent_acc,
            "signal_random_baseline_acc":        signal_random_acc,
            "noise_floor_agent_directional_acc": noise_agent_acc,
            "pct_detectable_signal_metrics":     pct_signal,
            "n_signal_metrics":                  len(signal_vals),
            "n_noise_floor_metrics":             len(noise_vals),
            # Per-metric breakdowns
            "per_metric_agent_directional_acc":  {k: v["agent_directional_correct"] for k, v in results.items()},
            "per_metric_random_baseline_acc":    {k: v["random_baseline_correct"]   for k, v in results.items()},
            "per_metric_is_detectable_signal":   {k: v["is_detectable_signal"]      for k, v in results.items()},
        }
    }
