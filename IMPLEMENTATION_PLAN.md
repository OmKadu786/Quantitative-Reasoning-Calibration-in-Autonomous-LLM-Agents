# ⚙️ QuaRCAA: Quantitative Reasoning Calibration in Autonomous Agents
## System Implementation & Benchmark Architecture Plan

**Framework Codename:** `QuaRCAA` (**Qua**ntitative **R**easoning **C**alibration in **A**utonomous **A**gents)  
**Target Repository:** `Quantitative Reasoning Calibration in Autonomous LLM Agents`  
**Core Purpose:** Rigorous empirical evaluation harness measuring continuous quantitative calibration error (MACE), prediction interval sharpness, and overconfidence rates in LLM optimization agents.

---

## 🏗️ 1. QuaRCAA System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                      Byte-Identical Prompt Template                     │
 │                     (quarcaa/prompts/template.py)                       │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Injects Context & System Prompt
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            QuaRCAA Agent Loop                           │
 │   DeepSeek R1 (Reasoner)  │   GPT-4o (OpenAI)   │  Claude 3.5 (Anthropic)│
 │   (Temperature = 0.2)     │ (Temperature = 0.2) │ (Temperature = 0.2)   │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Enforces Structured JSON Schema
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   QuaRCAA Prediction Schema Extractor                   │
 │  - Hyperparameters: {shield_threshold, class_weights, multipliers}    │
 │  - Predictions: {macro_f1, recall_minority1, recall_minority2}         │
 │  - Fields: {direction: UP|DOWN|STABLE, expected_min, expected_max}     │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Config Parameters
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │            QuaRCAA 3-Seed Multi-Seed Harness + Retry Handler            │
 │  - Exponential Backoff & Retry Logic (quarcaa/harness/retry_handler.py)│
 │  - Seed 1 (42)    │    Seed 2 (123)    │    Seed 3 (999)                  │
 │  - Evaluates on ECG Arrhythmia & Kaggle Credit Fraud Pipelines          │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ 3-Seed Means vs. Predictions
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   QuaRCAA Calibration Audit Engine                      │
 │  - MACE (Mean Absolute Calibration Error)                               │
 │  - Directional Accuracy Rate (%) [sign(μ_i - Baseline_i)]               │
 │  - Interval Sharpness (expected_max - expected_min)                     │
 │  - Overconfidence Rate (%) [μ_i < expected_min]                         │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 2. QuaRCAA Codebase Directory Structure

```
Quantitative Reasoning Calibration in Autonomous LLM Agents/
├── README.md                          # Research overview & literature comparison
├── IMPLEMENTATION_PLAN.md             # System implementation architecture document
├── proposal.txt                       # Executive research proposal summary
├── requirements.txt                   # Dependency specifications
├── configs/
│   ├── default_config.yaml            # Pinned temperature=0.2 & seed configs
│   ├── ecg_config.yaml                # ECG dataset pipeline configuration
│   └── credit_config.yaml             # Credit Card Fraud pipeline configuration
└── quarcaa/
    ├── __init__.py
    ├── prompts/                       # Single Source of Truth Prompting
    │   ├── __init__.py
    │   └── template.py                # Byte-identical prompt template across models
    ├── agents/                        # Multi-Model Agent Adapters
    │   ├── __init__.py
    │   ├── base_agent.py              # Abstract Agent Base Class
    │   ├── deepseek_agent.py          # DeepSeek R1 API Adapter (Extended Reasoning)
    │   ├── gpt_agent.py               # GPT-4o API Adapter (Prompted Reasoning)
    │   └── claude_agent.py            # Claude 3.5 Sonnet Adapter (Prompted Reasoning)
    ├── schema/                        # Prediction Schema Enforcement
    │   ├── __init__.py
    │   └── parser.py                  # Robust JSON prediction parser & validator
    ├── pipelines/                     # Execution Benchmark Pipelines
    │   ├── __init__.py
    │   ├── base_pipeline.py           # Pipeline Interface
    │   ├── ecg_pipeline.py            # MIT-BIH Arrhythmia Inter-Patient Pipeline
    │   └── credit_pipeline.py         # Kaggle Credit Card Fraud Imbalanced Pipeline
    ├── harness/                       # Execution & Retry Engine
    │   ├── __init__.py
    │   ├── retry_handler.py           # Exponential backoff decorator (429/500/timeouts)
    │   └── multi_seed_runner.py       # 3-Seed ([42, 123, 999]) Execution Engine
    └── metrics/                       # Calibration Diagnostic Engine
        ├── __init__.py
        ├── mace.py                    # Mean Absolute Calibration Error (MACE)
        ├── directional_accuracy.py    # Directional Accuracy Rate Calculator (Defined Baseline)
        ├── sharpness.py               # Prediction Interval Width / Sharpness Calculator
        └── audit_logger.py            # MLflow & JSON trace exporter
```

---

## 📐 3. Mathematical Metric Definitions

### 3.1 Mean Absolute Calibration Error (MACE)
$$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\text{ExpectedMin}_i + \text{ExpectedMax}_i}{2} - \mu_i \right|$$

### 3.2 Directional Accuracy Rate (%)
$$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{sign}(\mu_i - \text{Baseline}_i) = \text{PredictedDirection}_i \right)$$
* **Baseline Definition ($\text{Baseline}_i$):**
  * For Iteration 1 ($i=1$): $\text{Baseline}_1$ is defined as the pre-agent vanilla pipeline 3-seed mean.
  * For Iterations $i > 1$: $\text{Baseline}_i$ is defined as the empirical 3-seed execution mean of iteration $i-1$.

### 3.3 Prediction Interval Sharpness (Interval Width)
$$\text{Sharpness} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{ExpectedMax}_i - \text{ExpectedMin}_i \right)$$
* Prevents agents from hedging calibration metrics by outputting artificially wide confidence ranges.

### 3.4 Overconfidence Rate (%)
$$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \mu_i < \text{ExpectedMin}_i \right)$$

---

## 🛠️ 4. Core Module Implementation Specifications

### 4.1 Byte-Identical Prompt Template (`quarcaa/prompts/template.py`)
Guarantees identical wording across all provider calls:
```python
SYSTEM_PROMPT_TEMPLATE = """
You are an Autonomous AI ML Experimenter optimizing an imbalanced classification pipeline.
Analyze the current iteration results and recommend calibrated parameters.

Along with your Chain-of-Thought reasoning, you MUST output a JSON block matching this schema:
{
  "proposed_parameters": {"shield_threshold": float, "f_weight": float, "s_weight": float},
  "predictions": {
    "macro_f1": {"direction": "UP"|"DOWN"|"STABLE", "expected_min": float, "expected_max": float},
    "recall_F": {"direction": "UP"|"DOWN"|"STABLE", "expected_min": float, "expected_max": float},
    "recall_S": {"direction": "UP"|"DOWN"|"STABLE", "expected_min": float, "expected_max": float}
  }
}
"""
```

### 4.2 Retry & Rate-Limit Handler (`quarcaa/harness/retry_handler.py`)
Decorates API calls with exponential backoff to handle 429 rate limits and 500 timeouts cleanly without corrupting multi-seed execution state.

---

## 🚦 5. Staged Execution Plan (Sample Size & Compute Arithmetic)

### Trial & Execution Arithmetic Breakdown:
* **90 Primary Experimental Units:** 3 models × 2 datasets × 15 iterations. (Primary unit of prediction analysis)
* **270 Model Training Executions:** 90 experimental units × 3 seeds (`[42, 123, 999]`). (Pipeline training runs)
* **270 Calibration Observations:** 90 experimental units × 3 evaluated metrics per iteration. (Calibration dataset size)

| Phase | Target Model | Model Architecture & Reasoning Axis | Experimental Units | Pipeline Runs (3-Seed) | Calibration Obs (3-Metric) | Milestone |
|---|---|---|---|---|---|---|
| **Phase 1** | **DeepSeek R1** | Open-Weights / Trained Extended Reasoning | **30 units** | 90 runs | 90 obs | Baseline MACE & Schema Validation |
| **Phase 2** | **GPT-4o** | Proprietary / Prompted Step-by-Step | **30 units** | 90 runs | 90 obs | Open vs. Proprietary Comparison |
| **Phase 3** | **Claude 3.5** | Proprietary / Prompted Step-by-Step | **30 units** | 90 runs | 90 obs | Coding Agent Calibration Benchmark |
| **Total** | **3 Models** | **Dual-Axis Comparison** | **90 units** | **270 runs** | **270 obs** | **Final Empirical Diagnostic Audit** |

---

## 📋 6. Next Steps for Execution

1. **Implement `quarcaa/prompts/template.py`**: Byte-identical prompt manager.
2. **Implement `quarcaa/metrics/sharpness.py` & `directional_accuracy.py`**: Add interval width and explicit baseline tracking.
3. **Implement `quarcaa/harness/retry_handler.py`**: Exponential backoff wrapper.
4. **Trigger Phase 1 DeepSeek R1 Execution**: Run 15 iterations on ECG and Credit Fraud pipelines.
