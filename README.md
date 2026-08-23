# 🔬 Empirical Evaluation of Quantitative Reasoning Calibration in Autonomous LLM Agents

**Title:** Do LLMs Mean What They Say? Evaluating Quantitative Reasoning Calibration in Autonomous LLM Agents  
**Location:** `/Users/omkadu/code/UTP/Quantitative Reasoning Calibration in Autonomous LLM Agents/`  

---

## 📌 Executive Overview

Autonomous Large Language Model (LLM) agents are rapidly being adopted to design and optimize Machine Learning (ML) pipelines. When suggesting code changes or hyperparameter updates, these agents produce natural language **Chain-of-Thought (CoT)** reasoning detailing *why* a hyperparameter is modified and *what numeric performance gains* ($\Delta \text{Recall}$, $\Delta \text{F1}$) are expected.

This project performs the **first empirical diagnostic evaluation** measuring the gap between an LLM agent's written quantitative predictions and actual **3-seed downstream execution means**.

---

## 🎯 Experimental Architecture: Disjoint Two-Arm Structure

To prevent measurement confounding (where external parameter clamping distorts the prediction-outcome pairing), QuaRCAA strictly splits evaluation into **two disjoint arms**:

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

## 🔬 Evaluated Models & Datasets

### 1. Evaluated Models & Dual-Axis Framing (3 Models)
1. **DeepSeek R1** (Open-weights, trained extended reasoning model)
2. **GPT-4o** (Proprietary, instruction-tuned prompted reasoning model)
3. **Claude 3.5 Sonnet** (Proprietary, instruction-tuned prompted reasoning model)

*API Sampling Note:* `temperature: 0.2` and `top_p: 0.95` are passed across provider adapters. `deepseek-reasoner` manages sampling temperature natively during its internal thinking phase.

### 2. Imbalanced Benchmark Domains (2 Datasets)
1. **MIT-BIH Arrhythmia (ECG):** Cardiology signal classification under canonical de Chazal inter-patient split (5-class imbalanced).
2. **Kaggle Credit Card Fraud:** Financial fraud detection (binary heavily imbalanced dataset).

### 3. Sample Size & Execution Arithmetic (Primary Arm 1)
* **90 Primary Experimental Units:** 3 models × 2 datasets × 15 iterations.
* **270 Model Training Executions:** 90 units × 3 seeds (`[42, 123, 999]`).
* **270 Calibration Observations:** 90 units × 3 evaluated metrics per iteration.

---

## 📊 Mathematical Metric Definitions

1. **Directional Accuracy Rate (%):**
   $$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{sign}(\mu_i - \text{Baseline}_i) = \text{PredictedDirection}_i \right)$$
   *(Where $\text{Baseline}_1$ is the pre-agent vanilla pipeline 3-seed mean, and $\text{Baseline}_i$ for $i > 1$ is the empirical 3-seed mean of iteration $i-1$.)*

2. **Mean Absolute Calibration Error (MACE):**
   $$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} \left| \text{TargetMidpoint}_i - \mu_i \right|$$

3. **Prediction Interval Sharpness (Interval Width):**
   $$\text{Sharpness} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{ExpectedMax}_i - \text{ExpectedMin}_i \right)$$

4. **Overconfidence Rate (%):**
   $$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( (\text{Direction}_i \neq \text{"DOWN"} \land \mu_i < \text{ExpectedMin}_i) \lor (\text{Direction}_i == \text{"DOWN"} \land \mu_i > \text{ExpectedMax}_i) \right)$$

5. **Raw Audit Logging Engine (`quarcaa/harness/trial_logger.py`):**
   Persists immutable raw trial logs (`logs/raw_trials/`) containing full prompt text, raw LLM text responses, timestamps, status codes, seed metrics, and parsed JSON for 100% auditability.
