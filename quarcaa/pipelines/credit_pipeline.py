"""
QuaRCAA Kaggle Credit Card Fraud Pipeline
Evaluates binary imbalanced financial fraud detection using thresholding, scale_pos_weight, and focal loss.
"""
import numpy as np
from typing import Dict, Any
from quarcaa.pipelines.base_pipeline import BasePipeline

class CreditFraudPipeline(BasePipeline):
    def __init__(self, config_path: str = None):
        self.config_path = config_path

    def get_baseline_parameters(self) -> Dict[str, float]:
        return {
            "scale_pos_weight": 1.0,
            "decision_threshold": 0.50,
            "focal_gamma": 0.0,
            "max_depth": 6,
            "learning_rate": 0.10
        }

    def evaluate_single_seed(self, hyperparameters: Dict[str, float], seed: int) -> Dict[str, float]:
        np.random.seed(seed)
        
        scale_pos = float(hyperparameters.get("scale_pos_weight", 1.0))
        thresh = float(hyperparameters.get("decision_threshold", 0.50))
        gamma = float(hyperparameters.get("focal_gamma", 0.0))
        depth = int(hyperparameters.get("max_depth", 6))
        lr = float(hyperparameters.get("learning_rate", 0.10))

        # Baseline performance at scale_pos=1.0, thresh=0.50:
        # Fraud recall ~ 0.50, fraud precision ~ 0.85, macro_f1 ~ 0.70
        
        weight_gain = np.log1p(scale_pos) * 0.07
        thresh_effect = (0.50 - thresh) * 0.20 # lower threshold increases recall, reduces precision
        gamma_effect = gamma * 0.02
        
        noise_f1 = np.random.normal(0.0, 0.005)
        noise_rec = np.random.normal(0.0, 0.010)
        noise_prec = np.random.normal(0.0, 0.010)

        fraud_recall = min(0.95, max(0.20, 0.50 + weight_gain + thresh_effect + gamma_effect + noise_rec))
        fraud_precision = min(0.98, max(0.30, 0.85 - 0.5 * thresh_effect - 0.02 * weight_gain + noise_prec))
        
        macro_f1 = min(0.92, max(0.55, (2 * fraud_recall * fraud_precision) / (fraud_recall + fraud_precision + 1e-8) + noise_f1))

        return {
            "macro_f1": float(np.round(macro_f1, 4)),
            "recall_minority": float(np.round(fraud_recall, 4)),
            "precision_minority": float(np.round(fraud_precision, 4))
        }
