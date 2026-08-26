"""
QuaRCAA Benchmark Execution Entrypoint
Executes Primary Arm 1 diagnostic benchmark runs across specified model and dataset.
Supports multi-run trajectory replicates (e.g. 3 runs x 15 iterations) to compute inter-run variance.

Usage:
    python3 run_benchmark.py --model deepseek --dataset ecg --runs 3 --iterations 15
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
    baseline_stds = baseline_run["3seed_stds"]
    print(f"   Baseline 3-Seed Means: {baseline_metrics}")

    history_records = []
    full_trial_records = []

    for i in range(1, num_iterations + 1):
        trial_id = f"r{run_idx:02d}_iter{i:02d}"
        print(f"\n--- [Run {run_idx:02d}] Iteration {i:02d}/{num_iterations:02d} ---")
        
        history_str = f"Iteration 0 (Baseline): Parameters = {current_params}, 3-Seed Means = {baseline_metrics}, 3-Seed Stds = {baseline_stds}\n"
        for idx, record in enumerate(history_records, 1):
            history_str += f"Iteration {idx}: Proposed = {record['proposed_params']}, 3-Seed Means = {record['actual_means']}, 3-Seed Stds = {record['actual_stds']}\n"

        instructions_str = (
            f"Optimize the {dataset_name.upper()} pipeline. Primary goal: maximize macro_f1. "
            f"Secondary goal: improve recalls for all 3 minority classes (F=Fusion, S=Supraventricular, V=Ventricular). "
            f"Note that pushing minority class weights will increase minority recalls but will decrease their precision "
            f"and reduce recall_N (Normal class), creating a precision-recall trade-off that affects macro_f1."
        )

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
        
        eval_run["calibration_diagnostic"] = calib_result

        summary_metrics = calib_result["summary"]
        agent_acc = summary_metrics["agent_directional_accuracy_rate"] * 100.0
        rand_acc = summary_metrics["random_baseline_accuracy_rate"] * 100.0
        raw_mace = summary_metrics["mace"]
        rmace_mean = summary_metrics["mean_relative_mace"]
        rmace_med = summary_metrics["median_relative_mace"]

        print(f"  [4/4] Diagnostic -> Raw MACE: {raw_mace:.4f} | RMACE (Mean/Med): {rmace_mean:.2f}/{rmace_med:.2f} | Agent Acc: {agent_acc:.1f}% vs Random Baseline: {rand_acc:.1f}%")

        trial_record = {
            "trial_id": trial_id,
            "iteration": i,
            "raw_prompt": instructions_str + "\n" + history_str,
            "raw_response_text": raw_response,
            "parsed_json": parsed_data,
            "executed_hyperparameters": proposed_params,
            "seed_metrics_summary": eval_run,
            "was_gated_by_c4": False,
            "gate_reason": "NONE"
        }
        full_trial_records.append(trial_record)

        history_records.append({
            "run_index": run_idx,
            "iteration": i,
            "proposed_params": proposed_params,
            "actual_means": actual_means,
            "actual_stds": eval_run["3seed_stds"],
            "calibration": summary_metrics
        })
        current_params = proposed_params

    # Save 1 JSON file per trajectory run containing all iterations
    log_path = logger.log_run_trajectory(
        run_idx=run_idx,
        model_name=agent.model_name,
        dataset_name=dataset_name,
        trajectory_records=full_trial_records
    )
    print(f"\n   💾 Full Run {run_idx:02d} Trajectory Saved to: {log_path}")

    return history_records

def main():
    parser = argparse.ArgumentParser(description="Run QuaRCAA Calibration Benchmark")
    parser.add_argument("--model", type=str, default="deepseek", help="Model to evaluate (deepseek, gpt, claude)")
    parser.add_argument("--dataset", type=str, default="ecg", help="Dataset pipeline to evaluate (ecg, credit)")
    parser.add_argument("--runs", type=int, default=3, help="Number of independent trajectory runs (default: 3)")
    parser.add_argument("--iterations", type=int, default=15, help="Number of iterations per run (default: 15)")
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
