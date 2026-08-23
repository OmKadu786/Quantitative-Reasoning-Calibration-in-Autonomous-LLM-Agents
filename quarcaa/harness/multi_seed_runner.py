"""
QuaRCAA Multi-Seed Execution Runner
Executes pipeline training across 3 seeds ([42, 123, 999]) and computes mean μ and std σ.
"""
import numpy as np
from typing import Dict, Any, List
from quarcaa.pipelines.base_pipeline import BasePipeline

class MultiSeedRunner:
    def __init__(self, pipeline: BasePipeline, seeds: List[int] = None):
        self.pipeline = pipeline
        self.seeds = seeds or [42, 123, 999]

    def run_multi_seed_evaluation(self, hyperparameters: Dict[str, float]) -> Dict[str, Any]:
        """
        Runs the pipeline across all 3 seeds and returns 3-seed means and standard errors.
        """
        seed_results = []
        for seed in self.seeds:
            metrics = self.pipeline.evaluate_single_seed(hyperparameters, seed)
            seed_results.append(metrics)

        # Aggregate across seeds
        aggregated_means = {}
        aggregated_stds = {}
        
        all_keys = seed_results[0].keys()
        for k in all_keys:
            vals = [res[k] for res in seed_results]
            aggregated_means[k] = float(np.round(np.mean(vals), 4))
            aggregated_stds[k] = float(np.round(np.std(vals), 4))

        return {
            "3seed_means": aggregated_means,
            "3seed_stds": aggregated_stds,
            "per_seed_runs": seed_results,
            "seeds_evaluated": self.seeds
        }
