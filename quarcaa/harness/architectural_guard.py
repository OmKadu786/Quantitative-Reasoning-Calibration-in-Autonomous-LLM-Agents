"""
QuaRCAA Architectural Calibration Guard (MIRROR C4-inspired)
External policy that gates/clamps overconfident or overly vague agent hyperparameter 
updates before execution, based on prediction interval width and rolling MACE error.
"""
from typing import Dict, Any, Tuple

class ArchitecturalCalibrationGuard:
    """
    Implements MIRROR C4 Architectural Constraint:
    If an LLM agent outputs an overly vague prediction range (high Sharpness) 
    or has demonstrated high rolling MACE calibration error, this guard 
    damps/clamps the proposed parameter shift before sending it to training execution.
    """
    def __init__(self, max_allowed_interval_width: float = 0.25, max_allowed_mace: float = 0.15):
        self.max_width = max_allowed_interval_width
        self.max_mace = max_allowed_mace

    def evaluate_and_gate_proposal(
        self, 
        proposed_params: Dict[str, float], 
        baseline_params: Dict[str, float],
        predictions: Dict[str, Any], 
        rolling_mace: float
    ) -> Tuple[Dict[str, float], bool, str]:
        """
        Audits the proposed update. Returns (final_params, was_gated, reason_string).
        """
        # Check 1: Vague Interval Width (Hedging)
        is_vague = False
        for m_name, pred in predictions.items():
            exp_min = float(pred.get("expected_min", 0.0))
            exp_max = float(pred.get("expected_max", 1.0))
            if (exp_max - exp_min) > self.max_width:
                is_vague = True
                break

        # Check 2: High Rolling Miscalibration Error
        is_high_mace = rolling_mace > self.max_mace

        if is_vague or is_high_mace:
            reason = "VAGUE_CONFIDENCE_INTERVAL" if is_vague else "HIGH_ROLLING_MACE"
            # Damp proposed parameter changes by 50% towards baseline (conservative clamping)
            clamped_params = {}
            for k, val in proposed_params.items():
                base_v = baseline_params.get(k, val)
                clamped_params[k] = base_v + 0.5 * (val - base_v)
            return clamped_params, True, f"GATED_BY_C4_GUARD ({reason})"

        return proposed_params, False, "APPROVED"
