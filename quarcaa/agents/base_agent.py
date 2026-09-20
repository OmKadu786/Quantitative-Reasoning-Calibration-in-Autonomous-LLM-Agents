"""
QuaRCAA Abstract Agent Adapter Interface
Defines standard API query method for all LLM agent adapters.
"""
import os
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Dict, Any

from dotenv import load_dotenv


class BaseAgent(ABC):
    def __init__(self, model_name: str, temperature: float = 0.2):
        repo_root = Path(__file__).resolve().parents[2]
        load_dotenv(dotenv_path=repo_root / ".env", override=False)
        self.model_name = model_name
        self.temperature = temperature
        self.last_usage: dict[str, int] = {}

    @staticmethod
    def normalize_usage(data: Dict[str, Any]) -> dict[str, int]:
        usage = data.get("usage", {}) or data.get("usageMetadata", {})
        output_tokens = usage.get("completion_tokens", usage.get("output_tokens", usage.get("candidatesTokenCount", 0)))
        output_tokens += usage.get("thoughtsTokenCount", 0) or 0
        return {
            "input_tokens": int(usage.get("prompt_tokens", usage.get("input_tokens", usage.get("promptTokenCount", 0))) or 0),
            "output_tokens": int(output_tokens or 0),
        }

    @abstractmethod
    def generate_recommendation(self, instructions: str, history_str: str, defaults: Dict[str, float]) -> str:
        """
        Sends byte-identical system prompt to model provider API and returns raw text response.
        """
        pass
