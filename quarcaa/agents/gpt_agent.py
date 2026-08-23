"""
QuaRCAA GPT-4o Agent Adapter
Connects to OpenAI API endpoint using OPENAI_API_KEY environment variable.
"""
import os
import requests
from typing import Dict, Any
from quarcaa.agents.base_agent import BaseAgent
from quarcaa.prompts.template import get_system_prompt
from quarcaa.harness.retry_handler import retry_with_exponential_backoff

class GPTAgent(BaseAgent):
    def __init__(self, model_name: str = "gpt-4o", temperature: float = 0.2):
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = os.getenv("OPENAI_API_KEY")

    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0)
    def generate_recommendation(self, instructions: str, history_str: str, defaults: Dict[str, float]) -> str:
        prompt = get_system_prompt(instructions=instructions, history=history_str, defaults=defaults)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": self.temperature
        }
        
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=90)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
