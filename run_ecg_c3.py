"""
ECG C3 — Architectural Parameter Guard Run
Same as C1 but applies a programmatic parameter clamping guard:
If the rolling 3-iteration mean MACE exceeds 0.15, parameter updates are clamped by 50%
toward the baseline before pipeline execution.
Tests whether code-level intervention improves calibration outcomes vs prompt-only (C2).

Usage:
    python3 run_ecg_c3.py --model deepseek --runs 3 --iterations 15
    python3 run_ecg_c3.py --model gpt --runs 3 --iterations 15
    python3 run_ecg_c3.py --model claude --runs 3 --iterations 15
"""
from quarcaa.pipelines.ecg_pipeline import ECGArmyPipeline
from quarcaa.harness.benchmark_runner import run_benchmark


def instructions_c3(dataset_name: str, iteration: int, history: list) -> str:
    return (
        f"Optimize the {dataset_name.upper()} pipeline. Primary goal: maximize macro_f1. "
        f"Secondary goal: improve recalls for all 3 minority classes (F=Fusion, S=Supraventricular, V=Ventricular). "
        f"Note that pushing minority class weights will increase minority recalls but will decrease their precision "
        f"and reduce recall_N (Normal class), creating a precision-recall trade-off that affects macro_f1."
    )


if __name__ == "__main__":
    run_benchmark(
        pipeline=ECGArmyPipeline(),
        dataset_name="ecg",
        condition="c3",
        instructions_fn=instructions_c3,
        apply_c3_guard=True
    )
