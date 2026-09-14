from quarcaa.prompts.template import get_system_prompt
from quarcaa.pipelines.credit_pipeline import CreditFraudPipeline
from quarcaa.pipelines.ecg_pipeline import ECGArmyPipeline


def test_credit_prompt_uses_credit_parameter_names():
    prompt = get_system_prompt(
        instructions="optimize",
        history="",
        defaults=CreditFraudPipeline().get_baseline_parameters(),
    )
    assert '"scale_pos_weight"' in prompt
    assert '"decision_threshold"' in prompt
    assert '"shield_threshold"' not in prompt


def test_ecg_prompt_uses_ecg_parameter_names():
    prompt = get_system_prompt(
        instructions="optimize",
        history="",
        defaults=ECGArmyPipeline().get_baseline_parameters(),
    )
    assert '"shield_threshold"' in prompt
    assert '"v_weight"' in prompt
    assert '"scale_pos_weight"' not in prompt
