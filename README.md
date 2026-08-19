# 🔬 Empirical Evaluation of Quantitative Reasoning Calibration in Autonomous LLM Agents

**Title:** Do LLMs Mean What They Say? Evaluating Quantitative Reasoning Calibration in Autonomous LLM Agents  
**Location:** `/Users/omkadu/code/UTP/Quantitative Reasoning Calibration in Autonomous LLM Agents/`  

---

## 📌 Executive Overview

Autonomous Large Language Model (LLM) agents are rapidly being adopted to design and optimize Machine Learning (ML) pipelines. When suggesting code changes or hyperparameter updates, these agents produce natural language **Chain-of-Thought (CoT)** reasoning detailing *why* a hyperparameter is modified and *what numeric performance gains* ($\Delta \text{Recall}$, $\Delta \text{F1}$) are expected.

This project performs the **first empirical diagnostic evaluation** measuring the gap between an LLM agent's written quantitative predictions and actual **3-seed downstream execution means**.

---

## 🎯 Key Literature Positioning & Distinction

| Research Area | Key Paper | What They Evaluated | **What OUR Project Evaluates** |
|---|---|---|---|
| **Atomic Factual Calibration** | Kadavath et al. *(Anthropic, 2022, arXiv:2207.05221)* | Multiple-choice probability P(True) / P(IK) across TriviaQA, MMLU & arithmetic | **Continuous metric range calibration (ΔF1, ΔRecall)** |
| **Agentic Capability Overconfidence** | Barkan et al. *(NeurIPS 2024 Workshop on Metacognition)* | Self-predicted task completion success in multi-step agentic workflows | **Iterative parameter prediction vs. 3-seed execution mean** |
| **Knowing-Doing Gap & Mitigation** | Wang et al. *(MIRROR, 2026, arXiv:2604.19809)* | Evaluated C2 (epistemic self-knowledge) vs. C4 (architectural constraint) | **Extends C4 Architectural Gating to continuous quantitative AutoML search** |
| **Autonomous Agent HPO** | AgentHPO *(Zheng et al., 2024)* & OPRO *(Yang et al., 2024)* | Evaluates final target ML model test set performance only | **Audits calibration & overconfidence of agent's internal reasoning text** |

---

## 🔬 Experimental Setup & 3-Seed Protocol

```
                               ┌──────────────────────────────────────────────┐
                               │     LLM Optimization Agent Prompt            │
                               │  (DeepSeek R1 / GPT-4o / Claude 3.5)        │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │  Structured Elicitation (Kadavath-style)     │
                               │  - Proposed Hyperparameters                  │
                               │  - Predicted Direction (UP / DOWN / STABLE)  │
                               │  - Expected Range [min, max]                 │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │ Architectural Calibration Guard (C4-inspired)│
                               │  - Audits Interval Width & Rolling MACE      │
                               │  - Gates Overconfident Agent Updates         │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │   3-Seed Execution Harness ([42, 123, 999])  │
                               │  - Inter-Patient ECG & Credit Fraud Pipelines │
                               └──────────────────────┬───────────────────────┘
                                                      │
                                                      ▼
                               ┌──────────────────────────────────────────────┐
                               │        Calibration Audit Engine              │
                               │  - Directional Accuracy Rate (%)             │
                               │  - Mean Absolute Calibration Error (MACE)    │
                               │  - Prediction Interval Sharpness             │
                               │  - Overconfidence Rate (%)                   │
                               └──────────────────────────────────────────────┘
```

### 1. Evaluated Models & Dual-Axis Framing (3 Models)
1. **DeepSeek R1** (Open-weights, trained extended reasoning model)
2. **GPT-4o** (Proprietary, instruction-tuned prompted reasoning model)
3. **Claude 3.5 Sonnet** (Proprietary, instruction-tuned prompted reasoning model)

*Note on Model Axis:* DeepSeek R1 employs trained extended reasoning (internal CoT mechanism), whereas GPT-4o and Claude 3.5 Sonnet rely on prompted step-by-step reasoning. Model comparisons explore both the open/proprietary and reasoning-mechanism axes.

### 2. Imbalanced Benchmark Domains (2 Datasets)
1. **MIT-BIH Arrhythmia (ECG):** Cardiology signal classification under canonical de Chazal inter-patient split (5-class imbalanced).
2. **Kaggle Credit Card Fraud:** Financial fraud detection (binary heavily imbalanced dataset).

### 3. Sample Size & Execution Breakdown
To avoid confusion between experimental units, compute budget, and evaluation sample size:
* **90 Experimental Units:** 3 models × 2 datasets × 15 iterations. (Primary prediction sample unit)
* **270 Model Training Executions:** 90 units × 3 seeds (`[42, 123, 999]`). (Pipeline training compute runs)
* **270 Calibration Observations:** 90 units × 3 evaluated metrics per iteration (`macro_f1`, `recall_F`, `recall_S`). (Statistical calibration sample)

---

## 📊 Evaluation Metrics

1. **Directional Accuracy Rate (%):**
   $$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{sign}(\text{ActualMean}_i - \text{Baseline}_i) = \text{PredictedDirection}_i \right)$$
   *(Where $\text{Baseline}_1$ is the pre-agent vanilla pipeline 3-seed mean, and $\text{Baseline}_i$ for $i > 1$ is the empirical 3-seed mean of iteration $i-1$.)*

2. **Mean Absolute Calibration Error (MACE):**
   $$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} \left| \text{TargetMidpoint}_i - \text{ActualMean}_i \right|$$

3. **Prediction Interval Sharpness (Interval Width):**
   $$\text{Sharpness} = \frac{1}{N} \sum_{i=1}^{N} \left( \text{ExpectedMax}_i - \text{ExpectedMin}_i \right)$$

4. **Overconfidence Rate (%):**
   $$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}\left( \text{ActualMean}_i < \text{ExpectedMin}_i \right)$$

5. **C4 Architectural Calibration Guard Effectiveness:**
   Evaluates error reduction rate when an external architectural policy clamps or rejects overconfident agent parameter changes prior to execution.

---

## 💡 Practical Field Impact

1. **Architectural Safety Filters:** Demonstrates how external architectural gating (C4) prevents uncalibrated autonomous agents from deploying harmful model updates.
2. **Compute Savings Potential:** Informs early-stopping pre-filter mechanisms to prune ungrounded agent proposals before running expensive GPU training jobs.
3. **Model Selection Insights:** Offers empirical benchmark data guiding AI system engineers on model family selection for grounded autonomous optimization reasoning.
