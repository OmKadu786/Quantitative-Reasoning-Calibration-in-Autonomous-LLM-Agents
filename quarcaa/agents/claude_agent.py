"""
QuaRCAA Claude 3.5 Sonnet Agent Adapter
Connects to Anthropic API endpoint using ANTHROPIC_API_KEY environment variable.
"""
import os
import requests
from typing import Dict, Any
from quarcaa.agents.base_agent import BaseAgent
from quarcaa.prompts.template import get_system_prompt
from quarcaa.harness.retry_handler import retry_with_exponential_backoff

class ClaudeAgent(BaseAgent):
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", temperature: float = 0.2):
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")

    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0)
    def generate_recommendation(self, instructions: str, history_str: str, defaults: Dict[str, float]) -> str:
        prompt = get_system_prompt(instructions=instructions, history=history_str, defaults=defaults)
        
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "max_tokens": 2048,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }
        
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
