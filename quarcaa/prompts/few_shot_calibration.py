"""
QuaRCAA Stage 2 Mitigation Module: In-Context Self-Calibration Prompting
Formats historical [Prediction vs. 3-Seed Actual] calibration gaps and feeds them 
back into the agent's prompt to evaluate if LLMs can self-correct forecasting error.
"""
from typing import List, Dict, Any

def format_calibration_history_feedback(history_records: List[Dict[str, Any]]) -> str:
    """
    Formats past calibration performance into an in-context reflection block.
    """
    if not history_records:
        return "NO PAST CALIBRATION HISTORY AVAILABLE (ITERATION 1)."
        
    feedback_lines = ["PAST QUANTITATIVE FORECASTING CALIBRATION PERFORMANCE:"]
    
    for rec in history_records:
        iter_num = rec.get("iteration", 0)
        metrics = rec.get("metrics_details", {})
        
        feedback_lines.append(f"\n--- Iteration {iter_num} Self-Correction Feedback ---")
        for m_name, m_data in metrics.items():
            exp_range = m_data.get("expected_range", [0, 0])
            actual_mean = m_data.get("actual_3seed_mean", 0.0)
            mace = m_data.get("absolute_calibration_error", 0.0)
            is_overconf = m_data.get("is_overconfident", False)
            
            status = "⚠️ OVERCONFIDENT (FELL SHORT)" if is_overconf else "✅ CALIBRATED"
            feedback_lines.append(
                f"  * {m_name}: You predicted [{exp_range[0]:.4f}, {exp_range[1]:.4f}]. "
                f"Empirical 3-Seed Actual = {actual_mean:.4f} (Error = {mace:.4f}). [{status}]"
            )
            
    feedback_lines.append("\nINSTRUCTION FOR NEXT PREDICTION: Adjust your expected_min and expected_max ranges to be realistically calibrated based on your past forecasting gaps.")
    return "\n".join(feedback_lines)
