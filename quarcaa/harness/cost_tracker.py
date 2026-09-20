"""Provider token usage and estimated API-cost tracking."""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any


PRICES_PER_MILLION = {
    "deepseek": (0.15, 1.98),
    "gpt": (2.50, 10.00),
    "claude": (3.00, 15.00),
    "gemini": (2.00, 12.00),
}


@dataclass
class CostTracker:
    model_name: str
    max_cost_usd: float | None = None
    total_cost_usd: float = 0.0
    calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    records: list[dict[str, Any]] = field(default_factory=list)

    @property
    def provider(self) -> str:
        model = self.model_name.lower()
        if "deepseek" in model:
            return "deepseek"
        if "gpt" in model or "openai" in model:
            return "gpt"
        if "claude" in model or "anthropic" in model:
            return "claude"
        return "gemini"

    def ensure_budget(self):
        if self.max_cost_usd is not None and self.total_cost_usd >= self.max_cost_usd:
            raise RuntimeError(
                f"API budget reached before request: spent ${self.total_cost_usd:.4f} "
                f"of ${self.max_cost_usd:.4f}."
            )

    def record(self, usage: dict[str, Any] | None, raw_prompt: str, raw_response: str) -> dict[str, Any]:
        usage = usage or {}
        input_tokens = int(usage.get("input_tokens") or max(1, len(raw_prompt) // 4))
        output_tokens = int(usage.get("output_tokens") or max(1, len(raw_response) // 4))
        input_rate, output_rate = PRICES_PER_MILLION[self.provider]
        cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
        self.calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        record = {
            "model": self.model_name,
            "provider": self.provider,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": round(cost, 6),
            "cumulative_cost_usd": round(self.total_cost_usd, 6),
            "usage_source": "api" if usage else "text_estimate",
        }
        self.records.append(record)
        return record

    def summary(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "provider": self.provider,
            "api_calls": self.calls,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "estimated_cost_usd": round(self.total_cost_usd, 6),
        }

    def write(self, path: str):
        Path(path).write_text(json.dumps(self.summary(), indent=2))
