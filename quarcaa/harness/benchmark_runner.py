"""
QuaRCAA Shared Trajectory Runner — Core Harness
Contains all shared logic used by all 6 condition-specific benchmark scripts.
Do not run this file directly. Use the condition-specific scripts:

  ECG:
    python3 run_ecg_c1.py --model deepseek --runs 3 --iterations 15
    python3 run_ecg_c2.py --model deepseek --runs 3 --iterations 15
    python3 run_ecg_c3.py --model deepseek --runs 3 --iterations 15

  Credit Fraud:
    python3 run_credit_c1.py --model deepseek --runs 3 --iterations 15
    python3 run_credit_c2.py --model deepseek --runs 3 --iterations 15
    python3 run_credit_c3.py --model deepseek --runs 3 --iterations 15
"""
import os
import json
import time
import argparse
from dotenv import load_dotenv

load_dotenv()

from quarcaa.agents.deepseek_agent import DeepSeekAgent
from quarcaa.agents.gpt_agent import GPTAgent
from quarcaa.agents.claude_agent import ClaudeAgent
from quarcaa.harness.multi_seed_runner import MultiSeedRunner
from quarcaa.harness.trial_logger import TrialLogger
from quarcaa.schema.parser import extract_json_prediction
from quarcaa.metrics.mace import compute_quarcaa_calibration


def get_agent(model_name: str):
    m = model_name.lower()
    if "deepseek" in m:
        return DeepSeekAgent()
    elif "gpt" in m or "openai" in m:
        return GPTAgent()
    elif "claude" in m or "anthropic" in m:
        return ClaudeAgent()
    else:
        raise ValueError(f"Unknown model: {model_name}. Must be 'deepseek', 'gpt', or 'claude'.")


def run_single_trajectory(
    agent, pipeline, runner, logger,
    run_idx: int, num_iterations: int,
    dataset_name: str, condition: str,
    instructions_fn,           # callable(dataset_name, iteration, history) -> str
    apply_c3_guard: bool = False
):
    """
    Core trajectory execution loop shared across all conditions.
    instructions_fn: function that returns the prompt instruction string.
                     Allows C1/C2/C3 to inject different instruction text.
    apply_c3_guard: if True, clamps parameter updates by 50% when rolling MACE > 0.15.
    """
    print(f"\n" + "=" * 60)
    print(f"🔄 [{condition.upper()}] TRAJECTORY RUN {run_idx:02d} ({num_iterations} Iterations)")
    print("=" * 60)

    current_params = pipeline.get_baseline_parameters()
    print(f"[Run {run_idx:02d} - Iteration 00] Running 3-Seed Baseline...")
    baseline_run = runner.run_multi_seed_evaluation(current_params)
    baseline_metrics = baseline_run["3seed_means"]
    baseline_stds = baseline_run["3seed_stds"]
    print(f"   Baseline 3-Seed Means: {baseline_metrics}")

    history_records = []
    full_trial_records = []
    rolling_mace_window = []

    for i in range(1, num_iterations + 1):
        trial_id = f"r{run_idx:02d}_iter{i:02d}"
        print(f"\n--- [{condition.upper()}] [Run {run_idx:02d}] Iteration {i:02d}/{num_iterations:02d} ---")

        history_str = f"Iteration 0 (Baseline): Parameters = {current_params}, 3-Seed Means = {baseline_metrics}, 3-Seed Stds = {baseline_stds}\n"
        for idx, record in enumerate(history_records, 1):
            history_str += f"Iteration {idx}: Proposed = {record['proposed_params']}, 3-Seed Means = {record['actual_means']}, 3-Seed Stds = {record['actual_stds']}\n"

        instructions_str = instructions_fn(dataset_name, i, history_records)

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

        # C3 Guard: clamp parameter updates by 50% if rolling MACE > 0.15
        was_gated = False
        gate_reason = "NONE"
        if apply_c3_guard and rolling_mace_window:
            rolling_mace = sum(rolling_mace_window[-3:]) / len(rolling_mace_window[-3:])
            if rolling_mace > 0.15:
                clamped = {}
                for k, v in proposed_params.items():
                    baseline_v = pipeline.get_baseline_parameters().get(k, v)
                    clamped[k] = baseline_v + 0.5 * (v - baseline_v)
                proposed_params = clamped
                was_gated = True
                gate_reason = f"C3_GUARD: rolling_mace={rolling_mace:.4f} > 0.15 threshold"
                print(f"  ⚠️  C3 Guard TRIGGERED — parameter update clamped by 50%")

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
        raw_mace = summary_metrics["mace"]
        rmace_mean = summary_metrics["mean_relative_mace"]
        rmace_med = summary_metrics["median_relative_mace"]
        rolling_mace_window.append(raw_mace)

        print(f"  [4/4] Diagnostic -> MACE: {raw_mace:.4f} | RMACE (Mean/Med): {rmace_mean:.2f}/{rmace_med:.2f} | Agent Acc: {agent_acc:.1f}%")

        full_trial_records.append({
            "trial_id": trial_id,
            "condition": condition,
            "iteration": i,
            "raw_prompt": instructions_str + "\n" + history_str,
            "raw_response_text": raw_response,
            "parsed_json": parsed_data,
            "executed_hyperparameters": proposed_params,
            "was_gated_by_c3": was_gated,
            "gate_reason": gate_reason,
            "seed_metrics_summary": eval_run,
        })

        history_records.append({
            "run_index": run_idx,
            "iteration": i,
            "proposed_params": proposed_params,
            "actual_means": actual_means,
            "actual_stds": eval_run["3seed_stds"],
            "calibration": summary_metrics
        })
        current_params = proposed_params

    log_path = logger.log_run_trajectory(
        run_idx=run_idx,
        model_name=agent.model_name,
        dataset_name=f"{dataset_name}_{condition}",
        trajectory_records=full_trial_records
    )
    print(f"\n   💾 Run {run_idx:02d} [{condition.upper()}] Saved to: {log_path}")
    return history_records


def run_benchmark(pipeline, dataset_name: str, condition: str, instructions_fn, apply_c3_guard: bool = False):
    parser = argparse.ArgumentParser(description=f"QuaRCAA {dataset_name.upper()} {condition.upper()} Benchmark")
    parser.add_argument("--model", type=str, default="deepseek")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=15)
    args = parser.parse_args()

    agent = get_agent(args.model)
    runner = MultiSeedRunner(pipeline=pipeline, seeds=[42, 123, 999])
    logger = TrialLogger(log_dir="logs")

    print("=" * 80)
    print(f"🚀 QUARCAA BENCHMARK — {dataset_name.upper()} / {condition.upper()}")
    print(f"   Model: {agent.model_name} | Runs: {args.runs} | Iterations: {args.iterations}")
    print("=" * 80)

    all_run_records = []
    for r in range(1, args.runs + 1):
        trajectory_records = run_single_trajectory(
            agent=agent, pipeline=pipeline, runner=runner, logger=logger,
            run_idx=r, num_iterations=args.iterations,
            dataset_name=dataset_name, condition=condition,
            instructions_fn=instructions_fn,
            apply_c3_guard=apply_c3_guard
        )
        all_run_records.append({"run_index": r, "trajectory": trajectory_records})

    os.makedirs("logs", exist_ok=True)
    summary_file = f"logs/summary_{dataset_name}_{condition}_{args.model}.json"
    with open(summary_file, "w") as f:
        json.dump(all_run_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ COMPLETED — {dataset_name.upper()} / {condition.upper()} ({args.runs} x {args.iterations} trials)")
    print(f"   Summary: {summary_file}")
    print("=" * 80)
