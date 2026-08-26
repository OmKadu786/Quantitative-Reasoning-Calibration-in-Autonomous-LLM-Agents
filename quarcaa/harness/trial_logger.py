"""
QuaRCAA Raw Trial Audit Logger
Persists full raw API request prompts, raw response text, HTTP status codes, timestamps,
seed metrics, and parsed JSON for 100% auditable verification.
Saves 1 JSON file per trajectory run containing all iterations in a structured list.
"""
import os
import json
import time
from typing import Dict, Any, List

class TrialLogger:
    def __init__(self, log_dir: str = "logs/raw_trials"):
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)

    def log_run_trajectory(
        self,
        run_idx: int,
        model_name: str,
        dataset_name: str,
        trajectory_records: List[Dict[str, Any]]
    ) -> str:
        """
        Saves 1 immutable raw JSON file for an entire trajectory run containing all iterations.
        """
        record = {
            "run_index": run_idx,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "model_name": model_name,
            "dataset_name": dataset_name,
            "total_iterations": len(trajectory_records),
            "iterations": trajectory_records
        }
        
        timestamp_str = int(time.time())
        filename = f"{dataset_name}_{model_name}_run{run_idx:02d}_t{timestamp_str}.json"
        filepath = os.path.join(self.log_dir, filename)
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2)
            
        return filepath
