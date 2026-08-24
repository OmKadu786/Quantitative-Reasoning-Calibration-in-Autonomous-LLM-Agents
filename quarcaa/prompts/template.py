"""
QuaRCAA Shared Prompt Template Module
Guarantees byte-identical instructions across DeepSeek R1, GPT-4o, and Claude 3.5 Sonnet.
Initial defaults start strictly at un-tuned raw baseline values (weights = 1.0, threshold = 0.50).
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
    "v_weight": {v_weight_default},
    "s_weight": {s_weight_default},
    "f_weight": {f_weight_default},
    "v_prob_multiplier": {v_mult_default},
    "s_prob_multiplier": {s_mult_default},
    "f_prob_multiplier": {f_mult_default}
  }},
  "predictions": {{
    "macro_f1": {{
      "direction": "UP",
      "expected_min": 0.55,
      "expected_max": 0.65
    }},
    "recall_F": {{
      "direction": "UP",
      "expected_min": 0.20,
      "expected_max": 0.35
    }},
    "recall_S": {{
      "direction": "UP",
      "expected_min": 0.60,
      "expected_max": 0.75
    }}
  }},
  "meta_predictions": {{
    "predicted_execution_time_sec": 12.5,
    "predicted_compute_cost_usd": 0.002
  }}
}}
```
"""

def get_system_prompt(instructions: str, history: str, defaults: dict = None) -> str:
    """Returns byte-identical system prompt string for all models using raw un-tuned baseline defaults."""
    d = defaults or {}
    fmt_defaults = {
        "shield_threshold_default": d.get("shield_threshold", 0.50),
        "v_weight_default": d.get("v_weight", 1.0),
        "s_weight_default": d.get("s_weight", 1.0),
        "f_weight_default": d.get("f_weight", 1.0),
        "v_mult_default": d.get("v_prob_multiplier", 1.00),
        "s_mult_default": d.get("s_prob_multiplier", 1.00),
        "f_mult_default": d.get("f_prob_multiplier", 1.00)
    }
    return SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions,
        history=history,
        **fmt_defaults
    )
