# 📚 QuaRCAA Master Documentation & Technical Memory

**Project Title:** Do LLMs Mean What They Say? Evaluating Quantitative Reasoning Calibration in Autonomous LLM Agents  
**Framework Codename:** `QuaRCAA` (**Qua**ntitative **R**easoning **C**alibration in **A**utonomous **A**gents)  
**Location:** `/Users/omkadu/code/UTP/Quantitative Reasoning Calibration in Autonomous LLM Agents/`  
**Repository:** [GitHub - OmKadu786/Quantitative-Reasoning-Calibration-in-Autonomous-LLM-Agents](https://github.com/OmKadu786/Quantitative-Reasoning-Calibration-in-Autonomous-LLM-Agents)  

---

## 📌 Executive Overview & Core Problem Statement

Autonomous Large Language Model (LLM) agents are increasingly deployed to design and optimize Machine Learning (ML) pipelines. When suggesting code changes or hyperparameter updates, these agents write natural language **Chain-of-Thought (CoT)** text promising specific quantitative performance gains:
> *"I am setting the loss weight to 18.0. I expect the Macro F1 score to go UP and land between **0.72 and 0.78**."*

**The Research Goal:** QuaRCAA performs the **first empirical diagnostic evaluation** measuring the gap between an LLM agent's self-reported quantitative predictions and actual **3-seed downstream execution means** across imbalanced classification domains.

---

## 🎯 Key Literature Positioning & Citation Audit

| Research Area | Key Paper | What They Evaluated | **What OUR Project Evaluates** |
|---|---|---|---|
| **Atomic Factual Calibration** | Kadavath et al. *(Anthropic, 2022, arXiv:2207.05221)* | Multiple-choice probability P(True) / P(IK) across TriviaQA, MMLU & arithmetic | **Continuous metric range calibration (ΔF1, ΔRecall)** |
| **Agentic Capability Overconfidence** | Barkan et al. *(NeurIPS 2024 Workshop on Metacognition)* | Self-predicted task completion success in multi-step agentic workflows | **Iterative parameter prediction vs. 3-seed execution mean** |
| **Knowing-Doing Gap & Mitigation** | Wang et al. *(MIRROR, 2026, arXiv:2604.19809)* | Evaluated C2 (epistemic self-knowledge) vs. C4 (architectural constraint) | **Extends C4 Architectural Gating to continuous quantitative AutoML search** |
| **Autonomous Agent HPO** | AgentHPO *(Zheng et al., 2024)* & OPRO *(Yang et al., 2024)* | Evaluates final target ML model test set performance only | **Audits calibration & overconfidence of agent's internal reasoning text** |

---

## 🔬 Experimental Architecture: Disjoint Two-Arm Structure

To prevent measurement confounding (where external parameter clamping distorts the prediction-outcome pairing), QuaRCAA strictly splits execution into **two disjoint arms**:

```
                       ┌──────────────────────────────────────────────┐
                       │     LLM Optimization Agent Prompt            │
                       │  (DeepSeek R1 / GPT-4o / Claude 3.5)        │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┴─────────────────────────┐
                    ▼                                                   ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │    ARM 1: PRIMARY DIAGNOSTIC BENCHMARK │            │ ARM 2: SECONDARY C4 INTERVENTION STUDY│
 │    (Pristine 90/270/270 Benchmark)   │            │   (MIRROR-Style Outcome Guarding)    │
 ├──────────────────────────────────────┤            ├──────────────────────────────────────┤
 │ - No architectural guard or clamping │            │ - C4 Architectural Guard Active      │
 │ - Runs EXACT proposed parameters theta│            │ - Clamps updates by 50% if vague      │
 │ - Measures pristine MACE, Sharpness &│            │   or high rolling MACE               │
 │   Overconfidence Rate                │            │ - Evaluates outcome-level pipeline   │
 │ - Baseline: Raw un-tuned defaults    │            │   accuracy & bad update rejection    │
 └──────────────────┬───────────────────┘            └──────────────────┬───────────────────┘
                    │                                                   │
                    ▼                                                   ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ 3-Seed Execution ([42, 123, 999])    │            │ 3-Seed Execution ([42, 123, 999])    │
 └──────────────────┬───────────────────┘            └──────────────────┬───────────────────┘
                    │                                                   │
                    ▼                                                   ▼
 ┌──────────────────────────────────────┐            ┌──────────────────────────────────────┐
 │ Calibration Diagnostic Audit Engine  │            │ Intervention Outcome Analysis        │
 └──────────────────────────────────────┘            └──────────────────────────────────────┘
```

---

## 🤖 Models & Benchmark Datasets

### 1. Evaluated Models & Dual-Axis Matrix (3 Models)
1. **DeepSeek R1** (`deepseek-reasoner`): Open-weights, trained extended reasoning model. *(Note: `deepseek-reasoner` manages internal temperature natively during thinking phase).*
2. **GPT-4o** (`gpt-4o`): Proprietary, instruction-tuned prompted reasoning model (`temperature: 0.2`).
3. **Claude 3.5 Sonnet** (`claude-3-5-sonnet-20241022`): Proprietary, instruction-tuned prompted reasoning model (`temperature: 0.2`).

### 2. Benchmark Datasets (2 Datasets)
1. **MIT-BIH Arrhythmia (ECG):** Cardiology signal classification under canonical de Chazal inter-patient split (5-class imbalanced).
2. **Kaggle Credit Card Fraud (`creditcard.csv`):** Financial fraud detection on 284,807 transactions (binary 0.17% heavily imbalanced).

### 3. Sample Size & Compute Arithmetic
* **90 Primary Experimental Units:** 3 models × 2 datasets × 15 iterations.
* **270 Model Training Executions:** 90 units × 3 seeds (`[42, 123, 999]`).
* **270 Calibration Observations:** 90 units × 3 evaluated metrics per iteration.

---

## 🛠️ Hyperparameter Baseline Physics

Iteration 1 **must start at raw, un-tuned baseline defaults** (equal weights = `1.0`, threshold = `0.50`) to prevent "Metric Saturation" (plateauing) and guarantee maximum room for metric trajectories.

### ECG Arrhythmia Baseline Defaults (`configs/ecg_config.yaml`)
* `shield_threshold`: `0.50` *(Range: `[0.30, 0.90]`)*
* `v_weight`: `1.0`, `s_weight`: `1.0`, `f_weight`: `1.0` *(Range: `[1.0, 40.0]`)*
* `v_prob_multiplier`: `1.00`, `s_prob_multiplier`: `1.00`, `f_prob_multiplier`: `1.00` *(Range: `[0.50, 4.00]`)*

### Credit Card Fraud Baseline Defaults (`configs/credit_config.yaml`)
* `scale_pos_weight`: `1.0` *(Range: `[1.0, 50.0]`)*
* `decision_threshold`: `0.50` *(Range: `[0.10, 0.90]`)*
* `focal_gamma`: `0.0` *(Range: `[0.0, 5.0]`)*

---

## 📊 Mathematical Metric Definitions

1. **Mean Absolute Calibration Error (MACE):**
   $$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{\text{ExpectedMin}_i + \text{ExpectedMax}_i}{2} - \mu_i \right|$$

2. **Relative MACE (RMACE) with Stabilized Epsilon & Winsorized Capping:**
   $$\text{RMACE}_k = \min\left( \frac{\text{ACE}_k}{|\mu_{k, i} - \mu_{k, i-1}| + \epsilon}, \; 10.0 \right) \quad \text{where } \epsilon = 0.001$$
   *(Calculated per metric $k$, reporting both Mean RMACE and Median RMACE to prevent search saturation artifacts from distorting calibration trends).*

3. **LLM Directional Accuracy vs. Trivial "Always Predict UP" Control Baseline:**
   $$\text{Agent Directional Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{sign}(\mu_i - \text{Baseline}_i) = \text{PredictedDirection}_i \right)$$
   $$\text{Trivial UP Acc} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \mu_i - \text{Baseline}_i > 0.001 \right)$$
   *(Calculated per metric: `macro_f1`, `recall_F`, `recall_S` to detect whether metric monotonicity is driving directional accuracy).*

4. **Prediction Interval Sharpness (Interval Width):**
   $$\text{Sharpness} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{ExpectedMax}_i - \text{ExpectedMin}_i \right)$$

5. **Overconfidence Rate (%):**
   $$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( (\text{Direction}_i \neq \text{"DOWN"} \land \mu_i < \text{ExpectedMin}_i) \lor (\text{Direction}_i == \text{"DOWN"} \land \mu_i > \text{ExpectedMax}_i) \right)$$


5. **Exact C4 Clamping Formula (Arm 2 Intervention):**
   $$\theta_{\text{clamped}, k} = \theta_{\text{baseline}, k} + 0.5 \cdot (\theta_{\text{proposed}, k} - \theta_{\text{baseline}, k}) \quad \forall k$$
   *(Triggered if interval width $> 0.25$ or rolling MACE $> 0.15$).*

---

## 📁 Repository Directory Structure

```
Quantitative Reasoning Calibration in Autonomous LLM Agents/
├── DOCUMENTATION.md                   # Master technical documentation & memory (THIS FILE)
├── README.md                          # Quick start repo entrance
├── SIMPLIFIED_EVALUATION_GUIDE.md     # Plain English step-by-step guide
├── FRAMEWORK_TECHNICAL_SPECIFICATION.txt # Detailed low-level technical specification
├── run_benchmark.py                   # Main benchmark execution entrypoint script
├── requirements.txt                   # Dependency specifications
├── configs/
│   ├── default_config.yaml            # Pinned temperature=0.2 & seed configs
│   ├── ecg_config.yaml                # ECG dataset pipeline configuration
│   └── credit_config.yaml             # Credit Card Fraud pipeline configuration
├── dataset/
│   ├── creditcard.csv                 # Kaggle Credit Card Fraud dataset (150.8 MB)
│   └── mit-bih-arrhythmia-database-1.0.0/ # ECG arrhythmia database
├── quarcaa/
│   ├── __init__.py                    # System module exports
│   ├── prompts/
│   │   ├── template.py                # Single Source of Truth prompt template
│   │   └── few_shot_calibration.py    # Stage 2 C2 ablation history module
│   ├── schema/
│   │   └── parser.py                  # JSON prediction parser & validator
│   ├── pipelines/
│   │   ├── base_pipeline.py           # Abstract pipeline interface
│   │   ├── ecg_pipeline.py            # ECG Arrhythmia inter-patient pipeline
│   │   └── credit_pipeline.py         # Credit Card Fraud pipeline
│   ├── agents/
│   │   ├── base_agent.py              # Abstract agent interface
│   │   ├── deepseek_agent.py          # DeepSeek R1 API adapter
│   │   ├── gpt_agent.py               # GPT-4o API adapter
│   │   └── claude_agent.py            # Claude 3.5 Sonnet API adapter
│   ├── harness/
│   │   ├── multi_seed_runner.py       # 3-Seed ([42, 123, 999]) execution engine
│   │   ├── trial_logger.py            # Raw JSON audit log engine (logs/raw_trials/)
│   │   ├── architectural_guard.py    # MIRROR C4-inspired Architectural Guard
│   │   └── retry_handler.py           # Exponential backoff decorator (429/500 retries)
│   └── metrics/
│       ├── mace.py                    # Core MACE & overconfidence audit engine
│       ├── directional_accuracy.py    # Directional accuracy calculator
│       └── sharpness.py               # Prediction interval width calculator
└── logs/
    ├── raw_trials/                    # Immutable raw trial JSON audit records
    └── summary_results.json           # Aggregated benchmark run summaries
```

---

## 🚀 How to Run Benchmarks & Log Results

### 1. Execute Benchmark Run
To run the primary diagnostic benchmark for DeepSeek R1 on ECG:
```bash
python3 run_benchmark.py --model deepseek --dataset ecg --iterations 15
```

To run on Credit Card Fraud:
```bash
python3 run_benchmark.py --model deepseek --dataset credit --iterations 15
```

To run GPT-4o or Claude 3.5:
```bash
python3 run_benchmark.py --model gpt --dataset ecg --iterations 15
python3 run_benchmark.py --model claude --dataset ecg --iterations 15
```

### 2. Result Logging Mechanism
* **Raw Per-Trial Audit Logs:** Saved automatically in `logs/raw_trials/` as immutable JSON files containing the full raw prompt, raw LLM text response, UTC timestamp, executed parameters, and 3-seed metrics.
* **Run Summaries:** Aggregated benchmark results are exported to `logs/summary_[dataset]_[model].json`.
