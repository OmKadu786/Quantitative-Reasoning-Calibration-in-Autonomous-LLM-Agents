"""
QuaRCAA Benchmark Execution Entrypoint
Executes Primary Arm 1 diagnostic benchmark runs across specified model and dataset.
Supports multi-run trajectory replicates (e.g. 3 runs x 5 iterations) to compute inter-run variance.

Usage:
    python3 run_benchmark.py --model deepseek --dataset ecg --runs 3 --iterations 5
"""
import os
import argparse
import json
import time
from typing import Dict, Any
from dotenv import load_dotenv

# Automatically load environment variables from .env
load_dotenv()

from quarcaa.pipelines.ecg_pipeline import ECGArmyPipeline
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.agents.deepseek_agent import DeepSeekAgent
from quarcaa.agents.gpt_agent import GPTAgent
from quarcaa.agents.claude_agent import ClaudeAgent
from quarcaa.harness.multi_seed_runner import MultiSeedRunner
from quarcaa.harness.trial_logger import TrialLogger
from quarcaa.schema.parser import extract_json_prediction
from quarcaa.metrics.mace import compute_quarcaa_calibration

def get_pipeline(dataset_name: str):
    if dataset_name.lower() in ["ecg", "mitbih"]:
        return ECGArmyPipeline()
    elif dataset_name.lower() in ["credit", "fraud"]:
        return CreditFraudPipeline()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Must be 'ecg' or 'credit'.")

def get_agent(model_name: str):
    m_lower = model_name.lower()
    if "deepseek" in m_lower:
        return DeepSeekAgent()
    elif "gpt" in m_lower or "openai" in m_lower:
        return GPTAgent()
    elif "claude" in m_lower or "anthropic" in m_lower:
        return ClaudeAgent()
    else:
        raise ValueError(f"Unknown model: {model_name}. Must be 'deepseek', 'gpt', or 'claude'.")

def run_single_trajectory(agent, pipeline, runner, logger, run_idx: int, num_iterations: int, dataset_name: str):
    print(f"\n" + "=" * 60)
    print(f"🔄 STARTING TRAJECTORY RUN {run_idx:02d} ({num_iterations} Iterations)")
    print("=" * 60)

    # Reset to fresh un-tuned baseline defaults for each run
    current_params = pipeline.get_baseline_parameters()
    print(f"[Run {run_idx:02d} - Iteration 00] Running 3-Seed Raw Un-tuned Baseline ([42, 123, 999])...")
    baseline_run = runner.run_multi_seed_evaluation(current_params)
    baseline_metrics = baseline_run["3seed_means"]
    print(f"   Baseline 3-Seed Means: {baseline_metrics}")

    history_records = []

    for i in range(1, num_iterations + 1):
        trial_id = f"r{run_idx:02d}_t{int(time.time())}_{i:02d}"
        print(f"\n--- [Run {run_idx:02d}] Iteration {i:02d}/{num_iterations:02d} ---")
        
        history_str = f"Iteration 0 (Baseline): Parameters = {current_params}, 3-Seed Means = {baseline_metrics}\n"
        for idx, record in enumerate(history_records, 1):
            history_str += f"Iteration {idx}: Proposed = {record['proposed_params']}, 3-Seed Means = {record['actual_means']}\n"

        instructions_str = f"Optimize the {dataset_name.upper()} pipeline. Maximize macro_f1 while improving minority recalls."

        print(f"  [1/4] Querying {agent.model_name} API...")
        try:
            raw_response = agent.generate_recommendation(
                instructions=instructions_str,
                history_str=history_str,
                defaults=current_params
            )
        except Exception as e:
            print(f"   ❌ API Query Failed: {e}")
            break

        print("  [2/4] Parsing JSON predictions...")
        try:
            parsed_data = extract_json_prediction(raw_response)
            proposed_params = parsed_data.get("proposed_parameters", current_params)
            predictions = parsed_data.get("predictions", {})
        except Exception as e:
            print(f"   ❌ Schema Parsing Failed: {e}")
            break

        print("  [3/4] Executing 3-Seed Pipeline Run ([42, 123, 999])...")
        eval_run = runner.run_multi_seed_evaluation(proposed_params)
        actual_means = eval_run["3seed_means"]
        print(f"        3-Seed Means: {actual_means}")

        prev_baseline = history_records[-1]["actual_means"] if history_records else baseline_metrics
        calib_result = compute_quarcaa_calibration(
            predictions=predictions,
            baseline_metrics=prev_baseline,
            actual_3seed_metrics=actual_means
        )
        mace_val = calib_result["summary"]["mace"]
        print(f"  [4/4] Calibration Diagnostic -> MACE: {mace_val:.4f} | Directional Acc: {calib_result['summary']['directional_accuracy_rate']*100:.1f}%")

        log_path = logger.log_trial(
            trial_id=trial_id,
            model_name=agent.model_name,
            dataset_name=dataset_name,
            iteration=i,
            raw_prompt=instructions_str + "\n" + history_str,
            raw_response_text=raw_response,
            parsed_json=parsed_data,
            executed_params=proposed_params,
            seed_metrics=eval_run,
            was_gated_by_c4=False,
            gate_reason="NONE"
        )
        print(f"        Immutable Log Persisted: {log_path}")

        history_records.append({
            "run_index": run_idx,
            "iteration": i,
            "proposed_params": proposed_params,
            "actual_means": actual_means,
            "calibration": calib_result["summary"]
        })
        current_params = proposed_params

    return history_records

def main():
    parser = argparse.ArgumentParser(description="Run QuaRCAA Calibration Benchmark")
    parser.add_argument("--model", type=str, default="deepseek", help="Model to evaluate (deepseek, gpt, claude)")
    parser.add_argument("--dataset", type=str, default="ecg", help="Dataset pipeline to evaluate (ecg, credit)")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent trajectory runs (default: 3)")
    parser.add_argument("--iterations", type=int, default=5, help="Number of iterations per run (default: 5)")
    args = parser.parse_args()

    pipeline = get_pipeline(args.dataset)
    agent = get_agent(args.model)
    runner = MultiSeedRunner(pipeline=pipeline, seeds=[42, 123, 999])
    logger = TrialLogger(log_dir="logs/raw_trials")

    print("=" * 80)
    print(f"🚀 STARTING QUARCAA BENCHMARK RUN")
    print(f"   Model Adapter: {agent.model_name}")
    print(f"   Dataset Pipeline: {args.dataset.upper()}")
    print(f"   Execution Design: {args.runs} Independent Runs x {args.iterations} Iterations ({args.runs * args.iterations} total trials)")
    print("=" * 80)

    all_run_records = []
    for r in range(1, args.runs + 1):
        trajectory_records = run_single_trajectory(
            agent=agent,
            pipeline=pipeline,
            runner=runner,
            logger=logger,
            run_idx=r,
            num_iterations=args.iterations,
            dataset_name=args.dataset
        )
        all_run_records.append({
            "run_index": r,
            "trajectory": trajectory_records
        })

    os.makedirs("logs", exist_ok=True)
    summary_file = f"logs/summary_{args.dataset}_{args.model}.json"
    with open(summary_file, "w") as f:
        json.dump(all_run_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ QUARCAA BENCHMARK RUN COMPLETED ({args.runs} Runs x {args.iterations} Iterations)")
    print(f"   Summary results saved to: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()
