import numpy as np
from typing import Dict, Any

def compute_quarcaa_calibration(predictions: Dict[str, Any], baseline_metrics: Dict[str, float], actual_3seed_metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes QuaRCAA Diagnostic Metrics:
    - MACE (Mean Absolute Calibration Error)
    - Directional Accuracy Rate (%)
    - Overconfidence Rate (%)
    """
    results = {}
    
    for metric_name, pred_data in predictions.items():
        if metric_name not in actual_3seed_metrics:
            continue
            
        pred_dir = pred_data.get("direction", "STABLE").upper()
        exp_min = float(pred_data.get("expected_min", 0.0))
        exp_max = float(pred_data.get("expected_max", 1.0))
        target_midpoint = (exp_min + exp_max) / 2.0
        
        base_val = baseline_metrics.get(metric_name, 0.0)
        actual_val = actual_3seed_metrics.get(metric_name, 0.0)
        actual_delta = actual_val - base_val
        
        # 1. Directional Accuracy
        if pred_dir == "UP":
            dir_correct = actual_delta > 0.001
        elif pred_dir == "DOWN":
            dir_correct = actual_delta < -0.001
        else: # STABLE
            dir_correct = abs(actual_delta) <= 0.01
            
        # 2. Absolute Calibration Error (ACE)
        ace = abs(target_midpoint - actual_val)
        
        # 3. Overconfidence Check (Actual fell short of expected minimum)
        is_overconfident = actual_val < exp_min if pred_dir in ["UP", "STABLE"] else actual_val > exp_max
        
        results[metric_name] = {
            "predicted_direction": pred_dir,
            "actual_delta": actual_delta,
            "directional_correct": bool(dir_correct),
            "expected_range": [exp_min, exp_max],
            "target_midpoint": target_midpoint,
            "actual_3seed_mean": actual_val,
            "absolute_calibration_error": ace,
            "is_overconfident": bool(is_overconfident)
        }
        
    dir_acc = np.mean([v["directional_correct"] for v in results.values()]) if results else 0.0
    mace = np.mean([v["absolute_calibration_error"] for v in results.values()]) if results else 0.0
    overconf_rate = np.mean([v["is_overconfident"] for v in results.values()]) if results else 0.0
    
    return {
        "metric_details": results,
        "summary": {
            "directional_accuracy_rate": float(dir_acc),
            "mace": float(mace),
            "overconfidence_rate": float(overconf_rate)
        }
    }
