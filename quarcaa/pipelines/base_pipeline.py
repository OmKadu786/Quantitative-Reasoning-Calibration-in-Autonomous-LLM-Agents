"""
QuaRCAA Abstract Base Pipeline Interface
Defines standard interface for execution pipelines (ECG Arrhythmia & Credit Fraud).
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BasePipeline(ABC):
    @abstractmethod
    def evaluate_single_seed(self, hyperparameters: Dict[str, float], seed: int) -> Dict[str, float]:
        """
        Executes pipeline training for a single seed and returns evaluated metrics dictionary.
        Must return keys: {'macro_f1': float, 'recall_F': float, 'recall_S': float} or dataset equivalent.
        """
        pass

    @abstractmethod
    def get_baseline_parameters(self) -> Dict[str, float]:
        """
        Returns the un-tuned raw baseline hyperparameter dictionary (weights = 1.0, threshold = 0.50).
        """
        pass
