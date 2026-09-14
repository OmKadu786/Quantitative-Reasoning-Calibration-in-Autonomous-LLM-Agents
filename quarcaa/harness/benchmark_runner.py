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
import glob
import time
import hashlib
import argparse
from dotenv import load_dotenv

load_dotenv()

from quarcaa.agents.deepseek_agent import DeepSeekAgent
from quarcaa.agents.gpt_agent import GPTAgent
from quarcaa.agents.claude_agent import ClaudeAgent
from quarcaa.agents.gemini_agent import GeminiAgent
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
    elif "gemini" in m:
        return GeminiAgent()
    else:
        raise ValueError(f"Unknown model: {model_name}. Must be 'deepseek', 'gpt', 'claude', or 'gemini'.")


def _response_hash(raw_response: str) -> str:
    return hashlib.sha256(raw_response.encode("utf-8")).hexdigest()


def _validate_parameter_keys(proposed_params: dict, expected_keys: set, dataset_name: str):
    if not isinstance(proposed_params, dict):
        raise ValueError(f"Model returned non-dictionary proposed_parameters for {dataset_name}: {type(proposed_params)}")
    provided_keys = set(proposed_params.keys())
    missing = expected_keys - provided_keys
    extra = provided_keys - expected_keys
    if missing or extra:
        raise ValueError(
            f"Model returned incompatible parameter keys for {dataset_name}: "
            f"missing={sorted(missing)}, extra={sorted(extra)}. "
            f"The prompt schema likely mismatched the pipeline schema."
        )


def _validate_prediction_keys(predictions: dict, expected_keys: set, dataset_name: str):
    if not isinstance(predictions, dict):
        raise ValueError(f"Model returned non-dictionary predictions for {dataset_name}: {type(predictions)}")
    provided_keys = set(predictions.keys())
    missing = expected_keys - provided_keys
    if missing:
        raise ValueError(
            f"Model returned incomplete predictions for {dataset_name}: missing={sorted(missing)}. "
            "The response cannot be calibrated against every pipeline metric."
        )


def _expected_metric_keys(dataset_name: str) -> set:
    if dataset_name == "credit":
        return {"macro_f1", "recall_fraud", "precision_fraud", "recall_normal", "precision_normal"}
    return {"macro_f1", "recall_F", "precision_F", "recall_S", "precision_S", "recall_V", "precision_V", "recall_N", "precision_N"}


def _get_validated_prediction(agent, instructions_str: str, history_str: str, defaults: dict, dataset_name: str):
    expected_parameter_keys = set(defaults.keys())
    expected_metric_keys = _expected_metric_keys(dataset_name)
    last_error = None
    for attempt in range(1, 3):
        try:
            raw_response = agent.generate_recommendation(
                instructions=instructions_str,
                history_str=history_str,
                defaults=defaults,
            )
            if raw_response is None or not str(raw_response).strip():
                raise ValueError("empty API response")
            parsed_data = extract_json_prediction(raw_response)
            proposed_params = parsed_data.get("proposed_parameters", defaults)
            predictions = parsed_data.get("predictions", {})
            _validate_parameter_keys(proposed_params, expected_parameter_keys, dataset_name)
            _validate_prediction_keys(predictions, expected_metric_keys, dataset_name)
            return raw_response, parsed_data, proposed_params, predictions
        except Exception as error:
            last_error = error
            if attempt < 2:
                print(f"   ⚠️ Schema response rejected; requesting one replacement ({error})")
    raise ValueError(f"Model response validation failed after 2 attempts: {last_error}") from last_error


def _validate_fit_parameter_match(logged_params: dict, fitted_params: dict, tol: float = 1e-6):
    if set(logged_params.keys()) != set(fitted_params.keys()):
        raise ValueError(
            f"Parameter mismatch before fit: logged={sorted(logged_params.keys())}, fitted={sorted(fitted_params.keys())}"
        )
    for key in logged_params:
        a = float(logged_params[key])
        b = float(fitted_params[key])
        if abs(a - b) > tol:
            raise ValueError(
                f"Parameter mismatch for '{key}': logged={a}, fitted={b}, diff={abs(a-b)} > {tol}"
            )


def _validate_iteration_not_stale(history_records: list, response_hash: str):
    if not history_records:
        return
    prev_record = history_records[-1]
    if prev_record.get("response_hash") == response_hash:
        raise ValueError(
            "Duplicate stale iteration detected: the model response is identical "
            "to the previous iteration."
        )


def _audit_to_summary_record(audit_record: dict) -> dict:
    trajectory = []
    for item in audit_record.get("iterations", []):
        seed_summary = item.get("seed_metrics_summary", {})
        trajectory.append({
            "run_index": audit_record.get("run_index"),
            "iteration": item.get("iteration"),
            "proposed_params": item.get("executed_hyperparameters", {}),
            "actual_means": seed_summary.get("3seed_means", {}),
            "actual_stds": seed_summary.get("3seed_stds", {}),
            "calibration": seed_summary.get("calibration_diagnostic", {}).get("summary", {}),
            "response_hash": item.get("response_hash"),
        })
    return {"run_index": audit_record.get("run_index"), "trajectory": trajectory}


def _load_existing_trajectory(log_dir: str, dataset_name: str, condition: str, model_name: str) -> list:
    audit_pattern = f"{log_dir}/{dataset_name}_{condition}_{model_name}_run*.json"
    records = []
    for path in sorted(glob.glob(audit_pattern)):
        with open(path, "r", encoding="utf-8") as f:
            records.append(_audit_to_summary_record(json.load(f)))
    return records


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
            raw_response, parsed_data, proposed_params, predictions = _get_validated_prediction(
                agent, instructions_str, history_str, current_params, dataset_name
            )
        except Exception as e:
            raise RuntimeError(f"API/schema query failed for {dataset_name} run {run_idx} iteration {i}: {e}") from e

        response_hash = _response_hash(str(raw_response))

        print("  [2/4] Parsing JSON predictions...")

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

        # Baseline integrity check: iteration 1 must be based on the real 3-seed baseline, not a stale or single-run value.
        if not history_records and baseline_metrics is None:
            raise ValueError(f"Baseline evaluation for {dataset_name} run {run_idx} is missing or invalid.")

        prev_baseline = history_records[-1]["actual_means"] if history_records else baseline_metrics
        prev_stds     = history_records[-1]["actual_stds"]  if history_records else baseline_stds
        curr_stds     = eval_run["3seed_stds"]

        calib_result = compute_quarcaa_calibration(
            predictions=predictions,
            baseline_metrics=prev_baseline,
            actual_3seed_metrics=actual_means,
            baseline_stds=prev_stds,
            actual_stds=curr_stds,
        )
        eval_run["calibration_diagnostic"] = calib_result

        summary_metrics = calib_result["summary"]
        if not summary_metrics:
            raise ValueError(f"Calibration summary is empty for {dataset_name} run {run_idx} iteration {i}.")
        if len(summary_metrics.get("per_metric_is_detectable_signal", {})) != summary_metrics.get("total_metrics_evaluated", 0):
            raise ValueError(
                f"Metric completeness mismatch for {dataset_name} run {run_idx} iteration {i}: "
                f"total_metrics_evaluated={summary_metrics.get('total_metrics_evaluated')} but "
                f"detected keys={len(summary_metrics.get('per_metric_is_detectable_signal', {}))}."
            )

        _validate_iteration_not_stale(history_records, response_hash)
        agent_acc        = summary_metrics["agent_directional_accuracy_rate"] * 100.0
        signal_acc       = summary_metrics.get("signal_agent_directional_acc")
        signal_acc_str   = f"{signal_acc*100:.0f}%" if signal_acc is not None else "N/A"
        pct_signal       = summary_metrics.get("pct_detectable_signal_metrics", 0.0) * 100.0
        raw_mace         = summary_metrics["mace"]
        rmace_mean       = summary_metrics["mean_relative_mace"]
        rmace_med        = summary_metrics["median_relative_mace"]
        rolling_mace_window.append(raw_mace)

        print(f"  [4/4] MACE: {raw_mace:.4f} | RMACE: {rmace_mean:.2f}/{rmace_med:.2f} | "
              f"Acc(all): {agent_acc:.0f}% | Acc(signal): {signal_acc_str} | Signal%: {pct_signal:.0f}%")

        full_trial_records.append({
            "trial_id": trial_id,
            "condition": condition,
            "iteration": i,
            "status": "ok",
            "response_hash": response_hash,
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
            "calibration": summary_metrics,
            "response_hash": response_hash,
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
    parser.add_argument("--start-run", type=int, default=1)
    args = parser.parse_args()

    agent = get_agent(args.model)
    runner = MultiSeedRunner(pipeline=pipeline, seeds=[42, 123, 999])
    # logs/{model_short_name}/{dataset_name}/  e.g. logs/claude/ecg/
    model_short = args.model.lower()
    log_dir = f"logs/{model_short}/{dataset_name}"
    logger = TrialLogger(log_dir=log_dir)

    print("=" * 80)
    print(f"🚀 QUARCAA BENCHMARK — {dataset_name.upper()} / {condition.upper()}")
    print(f"   Model: {agent.model_name} | Runs: {args.runs} | Iterations: {args.iterations}")
    print("=" * 80)

    summary_file = f"{log_dir}/summary_{dataset_name}_{condition}_{args.model}.json"
    all_run_records = _load_existing_trajectory(log_dir, dataset_name, condition, agent.model_name) if args.start_run > 1 else []

    for r in range(args.start_run, args.start_run + args.runs):
        trajectory_records = run_single_trajectory(
            agent=agent, pipeline=pipeline, runner=runner, logger=logger,
            run_idx=r, num_iterations=args.iterations,
            dataset_name=dataset_name, condition=condition,
            instructions_fn=instructions_fn,
            apply_c3_guard=apply_c3_guard
        )
        all_run_records.append({"run_index": r, "trajectory": trajectory_records})

    os.makedirs(log_dir, exist_ok=True)
    with open(summary_file, "w") as f:
        json.dump(all_run_records, f, indent=2)

    print("\n" + "=" * 80)
    print(f"✅ COMPLETED — {dataset_name.upper()} / {condition.upper()} ({args.runs} x {args.iterations} trials)")
    print(f"   Summary: {summary_file}")
    print("=" * 80)
