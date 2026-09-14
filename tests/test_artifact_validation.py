from quarcaa.harness.artifact_validation import validate_summary_artifact


def _trial(response_hash="hash"):
    return {
        "status": "ok",
        "response_hash": response_hash,
        "calibration": {
            "total_metrics_evaluated": 1,
            "per_metric_is_detectable_signal": {"macro_f1": False},
        },
    }


def test_accepts_complete_hashed_summary():
    accepted, reason = validate_summary_artifact([{"run_index": 1, "trajectory": [_trial()]}])
    assert accepted is True
    assert reason == "accepted"


def test_rejects_legacy_summary_without_hash():
    accepted, reason = validate_summary_artifact([{"run_index": 1, "trajectory": [_trial(response_hash=None)]}])
    assert accepted is False
    assert "response_hash" in reason