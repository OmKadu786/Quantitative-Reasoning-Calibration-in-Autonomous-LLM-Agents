"""
QuaRCAA Raw Trial Audit Logger
Persists full raw API request prompts, raw response text, HTTP status codes, timestamps,
seed metrics, and parsed JSON for 100% auditable per-trial verification.
"""
import os
import json
import time
from typing import Dict, Any

class TrialLogger:
    def __init__(self, log_dir: str = "logs/raw_trials"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_trial(
        self,
        trial_id: str,
        model_name: str,
        dataset_name: str,
        iteration: int,
        raw_prompt: str,
        raw_response_text: str,
        parsed_json: Dict[str, Any],
        executed_params: Dict[str, float],
        seed_metrics: Dict[str, Any],
        was_gated_by_c4: bool = False,
        gate_reason: str = "NONE"
    ) -> str:
        """
        Saves an immutable raw JSON trial record to disk.
        """
        record = {
            "trial_id": trial_id,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "iteration": iteration,
            "was_gated_by_c4": was_gated_by_c4,
            "gate_reason": gate_reason,
            "raw_prompt": raw_prompt,
            "raw_response_text": raw_response_text,
            "parsed_json": parsed_json,
            "executed_hyperparameters": executed_params,
            "seed_metrics_summary": seed_metrics
        }
        
        filename = f"{dataset_name}_{model_name}_iter{iteration:02d}_{trial_id}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            
        return filepath
