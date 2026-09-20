from quarcaa.harness.cost_tracker import CostTracker


def test_cost_tracker_records_usage_and_accumulates_cost(tmp_path):
    tracker = CostTracker(model_name="gpt-4o")
    record = tracker.record(
        {"input_tokens": 1_000_000, "output_tokens": 100_000},
        "prompt",
        "response",
    )
    assert record["usage_source"] == "api"
    assert tracker.total_cost_usd == 3.5
    assert tracker.calls == 1
    tracker.ensure_budget()

    cost_file = tmp_path / "cost.json"
    tracker.write(str(cost_file))
    assert '"estimated_cost_usd": 3.5' in cost_file.read_text()
