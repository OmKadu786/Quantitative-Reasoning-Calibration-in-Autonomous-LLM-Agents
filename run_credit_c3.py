"""
Credit Fraud C3 — Architectural Parameter Guard Run
Same as Credit C1 but with programmatic 50% parameter clamping when rolling MACE > 0.15.
This is the most critical condition: Credit Fraud's precision-recall cliff means unconstrained
agents may catastrophically over-push weights. C3 tests whether the guard prevents collapse.

Usage:
    python3 run_credit_c3.py --model deepseek --runs 3 --iterations 15
    python3 run_credit_c3.py --model gpt --runs 3 --iterations 15
    python3 run_credit_c3.py --model claude --runs 3 --iterations 15
"""
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.harness.benchmark_runner import run_benchmark


def instructions_c3(dataset_name: str, iteration: int, history: list) -> str:
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
        condition="c3",
        instructions_fn=instructions_c3,
        apply_c3_guard=True
    )
