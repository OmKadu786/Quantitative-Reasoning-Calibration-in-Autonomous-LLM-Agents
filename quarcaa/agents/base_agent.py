"""
QuaRCAA Abstract Agent Adapter Interface
Defines standard API query method for all LLM agent adapters.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseAgent(ABC):
    def __init__(self, model_name: str, temperature: float = 0.2):
        self.model_name = model_name
        self.temperature = temperature

    @abstractmethod
    def generate_recommendation(self, instructions: str, history_str: str, defaults: Dict[str, float]) -> str:
        """
        Sends byte-identical system prompt to model provider API and returns raw text response.
        """
        pass
