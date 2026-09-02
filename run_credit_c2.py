"""
Credit Fraud C2 — Epistemic Calibration Prompt Run
Same as Credit C1 but with explicit calibration audit warning in the prompt.

Usage:
    python3 run_credit_c2.py --model deepseek --runs 3 --iterations 15
    python3 run_credit_c2.py --model gpt --runs 3 --iterations 15
    python3 run_credit_c2.py --model claude --runs 3 --iterations 15
"""
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.harness.benchmark_runner import run_benchmark


def instructions_c2(dataset_name: str, iteration: int, history: list) -> str:
    return (
        f"Optimize the {dataset_name.upper()} pipeline. Primary goal: maximize macro_f1. "
        f"Secondary goal: maximize recall_fraud (catching all fraudulent transactions). "
        f"Warning: aggressively increasing scale_pos_weight or loss weights will boost fraud_recall "
        f"but may cause precision_fraud to collapse (false alarm explosion), which will severely "
        f"damage macro_f1. There is a non-linear precision-recall cliff in this search space. "
        f"\n\n[CALIBRATION AUDIT NOTICE]: You are being benchmarked for quantitative prediction calibration. "
        f"Your predicted intervals (expected_min, expected_max) will be scored for accuracy against the true "
        f"3-seed execution outcomes. Overconfident intervals that miss the true outcome will increase your "
        f"calibration error score. Set realistic, well-grounded bounds based on the trajectory history."
    )


if __name__ == "__main__":
    run_benchmark(
        pipeline=CreditFraudPipeline(),
        dataset_name="credit",
        condition="c2",
        instructions_fn=instructions_c2,
        apply_c3_guard=False
    )
