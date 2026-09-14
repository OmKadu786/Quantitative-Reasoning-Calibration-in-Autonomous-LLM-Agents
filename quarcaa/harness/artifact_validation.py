"""Validation rules for summaries eligible for research aggregation."""
from typing import Any


def validate_summary_artifact(data: Any) -> tuple[bool, str]:
    """Return whether a summary contains only accepted, provenance-backed trials."""
    if not isinstance(data, list) or not data:
        return False, "summary must be a non-empty list"

    trajectories = []
    for run in data:
        if not isinstance(run, dict):
            return False, "summary contains a non-dictionary run"
        run_trajectory = run.get("trajectory")
        if not isinstance(run_trajectory, list) or not run_trajectory:
            return False, "run has no trajectory records"
        trajectories.extend(run_trajectory)

    response_hashes = []
    for trial in trajectories:
        if trial.get("status", "ok") != "ok":
            return False, "summary contains a non-ok trial"
        response_hash = trial.get("response_hash")
        if not isinstance(response_hash, str) or not response_hash:
            return False, "trial is missing response_hash provenance"
        calibration = trial.get("calibration")
        if not isinstance(calibration, dict) or not calibration:
            return False, "trial is missing calibration metrics"
        total_metrics = calibration.get("total_metrics_evaluated")
        detected_metrics = calibration.get("per_metric_is_detectable_signal")
        if not isinstance(total_metrics, int) or not isinstance(detected_metrics, dict):
            return False, "trial has incomplete calibration metrics"
        if len(detected_metrics) != total_metrics:
            return False, "trial calibration metric count is inconsistent"
        response_hashes.append(response_hash)

    if len(response_hashes) != len(set(response_hashes)):
        return False, "summary contains duplicate response hashes"
    return True, "accepted"