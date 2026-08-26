import numpy as np
from typing import Dict, Any
from quarcaa.metrics.sharpness import compute_sharpness
from quarcaa.metrics.directional_accuracy import compute_directional_accuracy, compute_random_baseline_accuracy

EPSILON = 0.001  # Fixed stabilization parameter for Relative MACE
WINSORED_RMACE_CEILING = 10.0  # Cap ceiling to prevent degenerate spikes when step delta approaches zero

def compute_quarcaa_calibration(
    predictions: Dict[str, Any],
    baseline_metrics: Dict[str, float],
    actual_3seed_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Computes QuaRCAA Diagnostic Metrics:
    - MACE (Raw Mean Absolute Calibration Error)
    - Relative MACE (RMACE) normalized by actual step delta with epsilon=0.001 and Winsorized capping at 10.0
    - Mean & Median RMACE
    - RMACE Censored Count (number of observations hitting the 10.0 ceiling)
    - Per-Metric LLM Directional Accuracy vs. Per-Metric Trivial 'Always Predict UP' Control Accuracy
    - Prediction Interval Sharpness (Expected_Max - Expected_Min)
    - Overconfidence Rate (%)
    """
    results = {}
    censored_count = 0
    
    for metric_name, pred_data in predictions.items():
        if metric_name not in actual_3seed_metrics:
            continue
            
        pred_dir = pred_data.get("direction", "STABLE").upper()
        exp_min = float(pred_data.get("expected_min", 0.0))
        exp_max = float(pred_data.get("expected_max", 1.0))
        target_midpoint = (exp_min + exp_max) / 2.0
        interval_width = exp_max - exp_min
        
        base_val = baseline_metrics.get(metric_name, 0.0)
        actual_val = actual_3seed_metrics.get(metric_name, 0.0)
        actual_delta = actual_val - base_val
        abs_actual_delta = abs(actual_delta)
        
        # 1. Directional Accuracy (Agent vs. Random Baseline Control)
        agent_dir_correct = compute_directional_accuracy(pred_dir, actual_val, base_val)
        # Random baseline seed derived from metric name hash + base_val for full reproducibility
        rng_seed = abs(hash(metric_name + str(round(base_val, 4)))) % (2**31)
        random_baseline_correct = compute_random_baseline_accuracy(actual_val, base_val, rng_seed=rng_seed)
            
        # 2. Absolute Calibration Error (ACE)
        ace = abs(target_midpoint - actual_val)
        
        # 3. Relative Calibration Error (RCE) with epsilon=0.001 & Winsorized capping at 10.0
        rce_uncapped = ace / (abs_actual_delta + EPSILON)
        if rce_uncapped >= WINSORED_RMACE_CEILING:
            censored_count += 1
        rce_capped = min(rce_uncapped, WINSORED_RMACE_CEILING)
        
        # 4. Overconfidence Check
        if pred_dir == "DOWN":
            is_overconfident = actual_val > exp_max
        else:
            is_overconfident = actual_val < exp_min

        results[metric_name] = {
            "predicted_direction": pred_dir,
            "baseline_val": base_val,
            "actual_3seed_mean": actual_val,
            "actual_delta": round(actual_delta, 6),
            "predicted_delta": round(target_midpoint - base_val, 6),  # Agent predicted this much change
            "agent_directional_correct": bool(agent_dir_correct),
            "random_baseline_correct": bool(random_baseline_correct),
            "expected_range": [exp_min, exp_max],
            "interval_width": interval_width,
            "target_midpoint": target_midpoint,
            "absolute_calibration_error": ace,
            "relative_calibration_error_uncapped": rce_uncapped,
            "relative_calibration_error_capped": rce_capped,
            "is_censored_at_ceiling": rce_uncapped >= WINSORED_RMACE_CEILING,
            "is_overconfident": bool(is_overconfident)
        }
        
    sharpness_info = compute_sharpness(predictions)
    
    agent_dir_acc = float(np.mean([v["agent_directional_correct"] for v in results.values()])) if results else 0.0
    random_baseline_acc = float(np.mean([v["random_baseline_correct"] for v in results.values()])) if results else 0.0
    
    mace = float(np.mean([v["absolute_calibration_error"] for v in results.values()])) if results else 0.0
    
    rce_capped_vals = [v["relative_calibration_error_capped"] for v in results.values()]
    mean_relative_mace = float(np.mean(rce_capped_vals)) if rce_capped_vals else 0.0
    median_relative_mace = float(np.median(rce_capped_vals)) if rce_capped_vals else 0.0
    
    overconf_rate = float(np.mean([v["is_overconfident"] for v in results.values()])) if results else 0.0
    
    # Per-metric directional accuracy breakdowns
    per_metric_agent_acc = {k: v["agent_directional_correct"] for k, v in results.items()}
    
    return {
        "metric_details": results,
        "summary": {
            "agent_directional_accuracy_rate": agent_dir_acc,
            "random_baseline_accuracy_rate": random_baseline_acc,
            "mace": mace,
            "mean_relative_mace": mean_relative_mace,
            "median_relative_mace": median_relative_mace,
            "rmace_censored_count": censored_count,
            "total_metrics_evaluated": len(results),
            "rmace_epsilon": EPSILON,
            "rmace_winsorized_ceiling": WINSORED_RMACE_CEILING,
            "mean_sharpness": sharpness_info["mean_sharpness"],
            "overconfidence_rate": overconf_rate,
            "per_metric_agent_directional_acc": {k: v["agent_directional_correct"] for k, v in results.items()},
            "per_metric_random_baseline_acc": {k: v["random_baseline_correct"] for k, v in results.items()}
        }
    }
