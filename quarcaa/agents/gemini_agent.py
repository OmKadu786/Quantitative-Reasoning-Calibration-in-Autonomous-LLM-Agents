"""
QuaRCAA Gemini Agent Adapter
Connects to Google Gemini API endpoint using GEMINI_API_KEY environment variable.
"""
import os
import requests
from typing import Dict, Any
from quarcaa.agents.base_agent import BaseAgent
from quarcaa.prompts.template import get_system_prompt
from quarcaa.harness.retry_handler import retry_with_exponential_backoff

class GeminiAgent(BaseAgent):
    def __init__(self, model_name: str = "gemini-3.1-pro-preview", temperature: float = 0.2):
        super().__init__(model_name=model_name, temperature=temperature)
        self.api_key = os.getenv("GEMINI_API_KEY")

    @retry_with_exponential_backoff(max_retries=5, initial_delay=2.0)
    def generate_recommendation(self, instructions: str, history_str: str, defaults: Dict[str, float]) -> str:
        prompt = get_system_prompt(instructions=instructions, history=history_str, defaults=defaults)
        
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }],
            "generationConfig": {
                "temperature": self.temperature
            }
        }
        
        response = requests.post(url, headers=headers, json=payload, timeout=90)
        
        # We catch explicit errors to prevent silent fails
        response.raise_for_status()
        data = response.json()
        
        # Parse standard Gemini response structure
        try:
            return data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to parse Gemini response: {data}") from e
