import json
import re
from typing import Dict, Any


def _extract_balanced_json_object(text: str, start_index: int) -> str | None:
    """Return the matching JSON object string beginning at start_index, handling nested braces and strings."""
    if start_index < 0 or start_index >= len(text) or text[start_index] != '{':
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start_index, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return text[start_index:i + 1]

    return None


def extract_json_prediction(llm_response: str) -> Dict[str, Any]:
    """
    Extracts the structured JSON prediction block from the LLM's natural language / CoT output.
    Handles nested braces in markdown code blocks and raw JSON fragments.
    """
    if not isinstance(llm_response, str):
        raise ValueError("QuaRCAA Schema Error: LLM response must be a string.")

    # 1. Try fenced JSON blocks first.
    code_block_match = re.search(r"```(?:json)?\s*(\{)", llm_response, re.IGNORECASE | re.DOTALL)
    if code_block_match:
        block_start = code_block_match.start(1)
        json_candidate = _extract_balanced_json_object(llm_response, block_start)
        if json_candidate is not None:
            try:
                return json.loads(json_candidate)
            except json.JSONDecodeError:
                pass

    # 2. Try raw JSON substring containing the prediction schema.
    for match in re.finditer(r"\{", llm_response):
        candidate = _extract_balanced_json_object(llm_response, match.start())
        if candidate is None:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and ("predictions" in parsed or "proposed_parameters" in parsed):
            return parsed

    raise ValueError("QuaRCAA Schema Error: Could not parse valid JSON prediction block from LLM response.")
