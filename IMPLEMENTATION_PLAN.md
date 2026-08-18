# ⚙️ QuaRCAA: Quantitative Reasoning Calibration in Autonomous Agents
## System Implementation & Benchmark Architecture Plan

**Framework Codename:** `QuaRCAA` (**Qua**ntitative **R**easoning **C**alibration in **A**utonomous **A**gents)  
**Target Repository:** `Quantitative Reasoning Calibration in Autonomous LLM Agents`  
**Core Purpose:** Rigorous empirical evaluation harness measuring continuous quantitative calibration error (MACE) and overconfidence rates in LLM optimization agents.

---

## 🏗️ 1. QuaRCAA System Architecture

```
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                            QuaRCAA Agent Loop                           │
 │   DeepSeek R1 (Reasoner)  │   GPT-4o (OpenAI)   │  Claude 3.5 (Anthropic)│
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Enforces Structured Schema
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   QuaRCAA Prediction Schema Extractor                   │
 │  - Hyperparameters: {shield_threshold, class_weights, multipliers}    │
 │  - Predictions: {macro_f1, recall_minority, precision_minority}        │
 │  - Fields: {direction: UP|DOWN|STABLE, expected_min, expected_max}     │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Config Parameters
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │               QuaRCAA 3-Seed Multi-Seed Execution Engine                │
 │  - Seed 1 (42)    │    Seed 2 (123)    │    Seed 3 (999)                  │
 │  - Evaluates on ECG Arrhythmia & Kaggle Credit Fraud Pipelines          │
 │  - Computes True Execution Means (μ) & Standard Errors (σ)              │
 └────────────────────────────────────┬────────────────────────────────────┘
                                      │ Execution Means vs. Predictions
                                      ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │                   QuaRCAA Calibration Audit Engine                      │
 │  - MACE (Mean Absolute Calibration Error)                               │
 │  - Directional Accuracy Rate (%)                                        │
 │  - Overconfidence Rate (%) [Actual_Mean < Expected_Min]                 │
 │  - MLflow Logging & JSON Trace Artifacts                                │
 └─────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 2. QuaRCAA Codebase Directory Structure

```
Quantitative Reasoning Calibration in Autonomous LLM Agents/
├── README.md                          # Research overview & literature comparison
├── IMPLEMENTATION_PLAN.md             # This system implementation architecture document
├── proposal.txt                       # Executive research proposal summary
├── requirements.txt                   # Dependency specifications
├── configs/
│   ├── ecg_config.yaml                # ECG dataset pipeline configuration
│   └── credit_config.yaml             # Credit Card Fraud pipeline configuration
└── quarcaa/
    ├── __init__.py
    ├── agents/                        # Multi-Model Agent Harnesses
    │   ├── __init__.py
    │   ├── base_agent.py              # Abstract Agent Base Class
    │   ├── deepseek_agent.py          # DeepSeek R1 API Integration (Reasoner)
    │   ├── gpt_agent.py               # GPT-4o API Integration
    │   └── claude_agent.py            # Claude 3.5 Sonnet Integration
    ├── schema/                        # Prediction Schema Enforcement
    │   ├── __init__.py
    │   └── parser.py                  # Robust JSON prediction parser & validator
    ├── pipelines/                     # Execution Benchmark Pipelines
    │   ├── __init__.py
    │   ├── base_pipeline.py           # Pipeline Interface
    │   ├── ecg_pipeline.py            # MIT-BIH Arrhythmia Inter-Patient Pipeline
    │   └── credit_pipeline.py         # Kaggle Credit Card Fraud Imbalanced Pipeline
    ├── harness/                       # Execution Engine
    │   ├── __init__.py
    │   └── multi_seed_runner.py       # 3-Seed ([42, 123, 999]) Execution Engine
    └── metrics/                       # Calibration Diagnostic Engine
        ├── __init__.py
        ├── mace.py                    # Mean Absolute Calibration Error (MACE)
        ├── directional_accuracy.py    # Directional Accuracy Rate Calculator
        └── audit_logger.py            # MLflow & JSON artifact exporter
```

---

## 📐 3. Mathematical Metric Definitions

### 3.1 Mean Absolute Calibration Error (MACE)
Quantifies the absolute distance between the midpoint of the LLM's predicted metric range $[\text{Min}_i, \text{Max}_i]$ and the empirical 3-seed execution mean $\mu_i$:

$$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\text{ExpectedMin}_i + \text{ExpectedMax}_i}{2} - \mu_i \right|$$

### 3.2 Directional Accuracy Rate (%)
Measures the percentage of trials where the LLM correctly predicts whether a metric will increase, decrease, or remain stable relative to the previous iteration baseline:

$$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{sign}(\mu_i - \text{Baseline}_i) = \text{PredictedDirection}_i \right)$$

### 3.3 Overconfidence Rate (%)
Measures the proportion of optimization trials where the actual 3-seed execution mean falls strictly below the LLM's lower-bound prediction:

$$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \mu_i < \text{ExpectedMin}_i \right)$$

---

## 🛠️ 4. Core Module Specifications

### 4.1 Schema Parser (`quarcaa/schema/parser.py`)
Enforces and extracts structured predictions from Chain-of-Thought agent output:

```json
{
  "proposed_parameters": {
    "shield_threshold": 0.60,
    "f_weight": 12.0,
    "s_weight": 18.0,
    "f_prob_multiplier": 1.50,
    "s_prob_multiplier": 2.10
  },
  "predictions": {
    "macro_f1": {
      "direction": "UP",
      "expected_min": 0.70,
      "expected_max": 0.75
    },
    "recall_F": {
      "direction": "UP",
      "expected_min": 0.40,
      "expected_max": 0.50
    },
    "recall_S": {
      "direction": "UP",
      "expected_min": 0.85,
      "expected_max": 0.90
    }
  }
}
```

### 4.2 Multi-Seed Harness (`quarcaa/harness/multi_seed_runner.py`)
Executes proposed configurations across 3 seeds (`[42, 123, 999]`), logging:
* Mean metric score $\mu$
* Standard deviation $\sigma$
* Per-seed execution history to ensure 100% auditability

---

## 🚦 5. Staged Execution Plan (180 Total Multi-Seed Trials)

| Phase | Target Model | Datasets | Iterations | Seeds / Trial | Total Runs | Milestone |
|---|---|---|---|---|---|---|
| **Phase 1** | **DeepSeek R1** | ECG + Credit Fraud | 15 / dataset | 3 seeds | **30 trials** | Baseline MACE & Schema Validation |
| **Phase 2** | **GPT-4o** | ECG + Credit Fraud | 15 / dataset | 3 seeds | **30 trials** | Open-Source vs. Proprietary Comparison |
| **Phase 3** | **Claude 3.5 Sonnet** | ECG + Credit Fraud | 15 / dataset | 3 seeds | **30 trials** | Coding Agent Calibration Benchmark |
| **Total** | **3 Models** | **2 Datasets** | **15 Iterations** | **3 Seeds** | **180 Trials** | **Final Empirical Diagnostic Study** |

---

## 📋 6. Next Steps for Execution

1. **Initialize `quarcaa` Package Structure:** Create the module files inside `quarcaa/`.
2. **Migrate Evaluation Core:** Integrate `calibration_evaluator.py` into `quarcaa/metrics/mace.py` and `quarcaa/schema/parser.py`.
3. **Run Phase 1 Execution:** Trigger DeepSeek R1 on MIT-BIH ECG pipeline for 15 iterations.
