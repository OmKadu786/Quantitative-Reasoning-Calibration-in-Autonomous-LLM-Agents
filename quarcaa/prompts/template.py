"""
QuaRCAA Shared Prompt Template Module
Guarantees the model sees the correct parameter schema for the current dataset.
This avoids silent fallback to baseline defaults when Credit Fraud and ECG use different hyperparameter names.
"""

ECG_JSON_SCHEMA = """{{
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
      "expected_min": 0.52,
      "expected_max": 0.62
    }},
    "recall_F": {{
      "direction": "UP",
      "expected_min": 0.20,
      "expected_max": 0.35
    }},
    "precision_F": {{
      "direction": "DOWN",
      "expected_min": 0.35,
      "expected_max": 0.48
    }},
    "recall_S": {{
      "direction": "UP",
      "expected_min": 0.55,
      "expected_max": 0.70
    }},
    "precision_S": {{
      "direction": "DOWN",
      "expected_min": 0.50,
      "expected_max": 0.62
    }},
    "recall_V": {{
      "direction": "UP",
      "expected_min": 0.65,
      "expected_max": 0.78
    }},
    "precision_V": {{
      "direction": "DOWN",
      "expected_min": 0.60,
      "expected_max": 0.73
    }},
    "recall_N": {{
      "direction": "DOWN",
      "expected_min": 0.85,
      "expected_max": 0.94
    }},
    "precision_N": {{
      "direction": "DOWN",
      "expected_min": 0.88,
      "expected_max": 0.95
    }}
  }},
  "meta_predictions": {{
    "predicted_execution_time_sec": 12.5,
    "predicted_compute_cost_usd": 0.002
  }}
}}"""

CREDIT_JSON_SCHEMA = """{{
  "proposed_parameters": {{
    "scale_pos_weight": {scale_pos_weight_default},
    "decision_threshold": {decision_threshold_default},
    "min_child_weight": {min_child_weight_default},
    "max_depth": {max_depth_default},
    "learning_rate": {learning_rate_default}
  }},
  "predictions": {{
    "macro_f1": {{
      "direction": "UP",
      "expected_min": 0.80,
      "expected_max": 0.95
    }},
    "recall_fraud": {{
      "direction": "UP",
      "expected_min": 0.70,
      "expected_max": 0.95
    }},
    "precision_fraud": {{
      "direction": "DOWN",
      "expected_min": 0.90,
      "expected_max": 0.99
    }},
    "recall_normal": {{
      "direction": "DOWN",
      "expected_min": 0.95,
      "expected_max": 1.00
    }},
    "precision_normal": {{
      "direction": "DOWN",
      "expected_min": 0.99,
      "expected_max": 1.00
    }}
  }},
  "meta_predictions": {{
    "predicted_execution_time_sec": 12.5,
    "predicted_compute_cost_usd": 0.002
  }}
}}"""

SYSTEM_PROMPT_TEMPLATE = """You are an Autonomous AI ML Experimenter optimizing a machine learning classification pipeline.
Your task is to analyze previous iteration metrics, identify performance bottlenecks, and recommend optimal hyperparameter configurations for the next run.

CURRENT PIPELINE CONTEXT & INSTRUCTIONS:
{instructions}

CURRENT MATHEMATICAL MECHANICS & ITERATION HISTORY:
{history}

CRITICAL INSTRUCTION: Along with your natural language Chain-of-Thought reasoning, you MUST end your response with a structured JSON block matching this exact schema:

```json
{json_schema}
```
"""


def get_system_prompt(instructions: str, history: str, defaults: dict = None) -> str:
    """Build the right JSON schema for the active pipeline instead of hard-coding ECG fields."""
    d = defaults or {}

    if "scale_pos_weight" in d or "decision_threshold" in d:
        fmt_defaults = {
            "scale_pos_weight_default": d.get("scale_pos_weight", 1.0),
            "decision_threshold_default": d.get("decision_threshold", 0.50),
            "min_child_weight_default": d.get("min_child_weight", 1.0),
            "max_depth_default": d.get("max_depth", 6),
            "learning_rate_default": d.get("learning_rate", 0.10),
        }
        json_schema = CREDIT_JSON_SCHEMA.format(**fmt_defaults)
    else:
        fmt_defaults = {
            "shield_threshold_default": d.get("shield_threshold", 0.50),
            "v_weight_default": d.get("v_weight", 1.0),
            "s_weight_default": d.get("s_weight", 1.0),
            "f_weight_default": d.get("f_weight", 1.0),
            "v_mult_default": d.get("v_prob_multiplier", 1.00),
            "s_mult_default": d.get("s_prob_multiplier", 1.00),
            "f_mult_default": d.get("f_prob_multiplier", 1.00),
        }
        json_schema = ECG_JSON_SCHEMA.format(**fmt_defaults)

    return SYSTEM_PROMPT_TEMPLATE.format(
        instructions=instructions,
        history=history,
        json_schema=json_schema,
    )
