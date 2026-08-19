"""
QuaRCAA Sharpness / Interval Width Metric Calculator
Prevents models from hedging calibration by outputting artificially wide confidence bounds.
"""
import numpy as np
from typing import Dict, Any

def compute_sharpness(predictions: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes interval width (expected_max - expected_min) for each metric prediction.
    A well-reasoned prediction is sharp (narrow range) AND calibrated (low MACE).
    """
    sharpness_scores = {}
    widths = []
    
    for metric_name, pred_data in predictions.items():
        exp_min = float(pred_data.get("expected_min", 0.0))
        exp_max = float(pred_data.get("expected_max", 1.0))
        width = exp_max - exp_min
        
        sharpness_scores[metric_name] = width
        widths.append(width)
        
    avg_sharpness = float(np.mean(widths)) if widths else 0.0
    return {
        "metric_sharpness": sharpness_scores,
        "mean_sharpness": avg_sharpness
    }
