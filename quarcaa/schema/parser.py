import json
import re
from typing import Dict, Any

def extract_json_prediction(llm_response: str) -> Dict[str, Any]:
    """
    Extracts the structured JSON prediction block from the LLM's natural language / CoT output.
    Looks for ```json ... ``` code blocks or raw JSON dictionary strings containing 'predictions'.
    """
    # 1. Try markdown code block
    json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_response, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 2. Try raw JSON substring searching for 'predictions'
    json_match = re.search(r"(\{[\s\S]*\"predictions\"[\s\S]*\})", llm_response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError("QuaRCAA Schema Error: Could not parse valid JSON prediction block from LLM response.")
