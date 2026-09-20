"""Credit Fraud C3 - calibration feedback condition."""
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.harness.benchmark_runner import run_benchmark
from run_credit_c2 import instructions_c2


def instructions_c3(dataset_name: str, iteration: int, history: list) -> str:
    return instructions_c2(dataset_name, iteration, history) + (
        "\n\n[FEEDBACK CONDITION]: Use the previous iteration prediction feedback included "
        "in the history to correct recurring forecasting errors. Do not assume feedback "
        "about the current iteration; it is only from completed earlier iterations."
    )


if __name__ == "__main__":
    run_benchmark(
        pipeline=CreditFraudPipeline(),
        dataset_name="credit",
        condition="c3",
        instructions_fn=instructions_c3,
    )
