"""
QuaRCAA Architectural Calibration Guard (MIRROR C4-inspired)
External policy that gates/clamps overconfident or overly vague agent hyperparameter 
updates before execution in Stage 2 intervention studies.

Exact Mathematical Specification:
1. Sharpness (Vagueness) Threshold: tau_width = 0.25
   Triggers if any predicted metric interval width (expected_max - expected_min) > 0.25
2. Miscalibration Threshold: tau_mace = 0.15
   Triggers if rolling MACE > 0.15
3. Clamping Formula:
   theta_clamped,k = theta_baseline,k + 0.5 * (theta_proposed,k - theta_baseline,k)
"""
from typing import Dict, Any, Tuple

class ArchitecturalCalibrationGuard:
    def __init__(self, tau_width: float = 0.25, tau_mace: float = 0.15):
        self.tau_width = tau_width
        self.tau_mace = tau_mace

    def evaluate_and_gate_proposal(
        self, 
        proposed_params: Dict[str, float], 
        baseline_params: Dict[str, float],
        predictions: Dict[str, Any], 
        rolling_mace: float
    ) -> Tuple[Dict[str, float], bool, str]:
        """
        Audits the proposed update. Returns (clamped_params, was_gated, reason_string).
        """
        # Check 1: Vague Interval Width (Hedging Check)
        is_vague = False
        for m_name, pred in predictions.items():
            if not isinstance(pred, dict):
                continue
            exp_min = float(pred.get("expected_min", 0.0))
            exp_max = float(pred.get("expected_max", 1.0))
            if (exp_max - exp_min) > self.tau_width:
                is_vague = True
                break

        # Check 2: High Rolling Miscalibration Error Check
        is_high_mace = rolling_mace > self.tau_mace

        if is_vague or is_high_mace:
            reason = "VAGUE_CONFIDENCE_INTERVAL" if is_vague else "HIGH_ROLLING_MACE"
            # Apply exact mathematical 50% shift clamping towards baseline
            clamped_params = {}
            for k, val in proposed_params.items():
                val_f = float(val)
                base_f = float(baseline_params.get(k, val_f))
                clamped_params[k] = base_f + 0.5 * (val_f - base_f)
                
            return clamped_params, True, f"GATED_BY_C4_GUARD ({reason})"

        return proposed_params, False, "APPROVED"
