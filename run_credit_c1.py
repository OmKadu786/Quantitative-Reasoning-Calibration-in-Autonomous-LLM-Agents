"""
Credit Fraud C1 — Standard Unconstrained Run
Agent receives empirical 3-seed feedback history only. No calibration warning. No parameter guard.
Credit Fraud features a non-monotonic search space: boosting class weights initially improves
Fraud Recall but past a threshold causes Precision to collapse and Macro F1 to plunge.

Usage:
    python3 run_credit_c1.py --model deepseek --runs 3 --iterations 15
    python3 run_credit_c1.py --model gpt --runs 3 --iterations 15
    python3 run_credit_c1.py --model claude --runs 3 --iterations 15
"""
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.harness.benchmark_runner import run_benchmark


def instructions_c1(dataset_name: str, iteration: int, history: list) -> str:
    return (
        f"Optimize the {dataset_name.upper()} pipeline. Primary goal: maximize macro_f1. "
        f"Secondary goal: maximize recall_fraud (catching all fraudulent transactions). "
        f"Warning: aggressively increasing scale_pos_weight or loss weights will boost fraud_recall "
        f"but may cause precision_fraud to collapse (false alarm explosion), which will severely "
        f"damage macro_f1. There is a non-linear precision-recall cliff in this search space."
    )


if __name__ == "__main__":
    run_benchmark(
        pipeline=CreditFraudPipeline(),
        dataset_name="credit",
        condition="c1",
        instructions_fn=instructions_c1,
        apply_c3_guard=False
    )
