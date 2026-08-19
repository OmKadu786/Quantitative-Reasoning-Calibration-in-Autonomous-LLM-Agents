"""
QuaRCAA Shared Prompt Template Module
Guarantees byte-identical instructions across DeepSeek R1, GPT-4o, and Claude 3.5 Sonnet.
"""

SYSTEM_PROMPT_TEMPLATE = """You are an Autonomous AI ML Experimenter optimizing a machine learning classification pipeline.
Your task is to analyze previous iteration metrics, identify performance bottlenecks, and recommend optimal hyperparameter configurations for the next run.

CURRENT PIPELINE CONTEXT & INSTRUCTIONS:
{instructions}

CURRENT MATHEMATICAL MECHANICS & ITERATION HISTORY:
{history}

CRITICAL INSTRUCTION: Along with your natural language Chain-of-Thought reasoning, you MUST end your response with a structured JSON block matching this exact schema:

```json
{{
  "proposed_parameters": {{
    "shield_threshold": {shield_threshold_default},
    "f_weight": {f_weight_default},
    "s_weight": {s_weight_default},
    "f_prob_multiplier": {f_mult_default},
    "s_prob_multiplier": {s_mult_default}
  }},
  "predictions": {{
    "macro_f1": {{
      "direction": "UP",
      "expected_min": 0.70,
      "expected_max": 0.75
    }},
    "recall_F": {{
      "direction": "UP",
      "expected_min": 0.40,
      "expected_max": 0.50
    }},
    "recall_S": {{
      "direction": "UP",
      "expected_min": 0.85,
      "expected_max": 0.90
    }}
  }}
}}
```
"""

def get_system_prompt(instructions: str, history: str, defaults: dict = None) -> str:
    """Returns byte-identical system prompt string for all models."""
    defaults = defaults or {
        "shield_threshold_default": 0.60,
        "f_weight_default": 12.0,
        "s_weight_default": 18.0,
        "f_mult_default": 1.50,
        "s_mult_default": 2.10
    }
    return SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions,
        history=history,
        **defaults
    )
