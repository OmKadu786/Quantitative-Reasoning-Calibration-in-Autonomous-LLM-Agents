# 🔬 Empirical Evaluation of Quantitative Reasoning Calibration in Autonomous LLM Agents

**Title:** Do LLMs Mean What They Say? Evaluating Quantitative Reasoning Calibration in Autonomous LLM Agents  
**Location:** `/Users/omkadu/code/UTP/confidence/`  
**Timeline:** 5-Week Empirical Benchmark Plan  

---

## 📌 Executive Overview

Autonomous Large Language Model (LLM) agents are rapidly being adopted to design and optimize Machine Learning (ML) pipelines. When suggesting code changes or hyperparameter updates, these agents produce natural language **Chain-of-Thought (CoT)** reasoning detailing *why* a hyperparameter is modified and *what numeric performance gains* ($\Delta \text{Recall}$, $\Delta \text{F1}$) are expected.

This project performs the **first empirical diagnostic evaluation** measuring the gap between an LLM agent's written quantitative predictions and actual **3-seed downstream execution means**.

---

## 🎯 Key Literature Positioning & Distinction

| Research Area | Key Paper | What They Evaluated | **What OUR Project Evaluates** |
|---|---|---|---|
| **Atomic Factual Calibration** | Kadavath et al. *(Anthropic, 2022)* | Single-turn true/false trivia confidence ($P(\text{True})$) | **Continuous metric range calibration ($\Delta \text{F1}$, $\Delta \text{Recall}$)** |
| **In-Advance Overconfidence** | Barkan et al. *(NeurIPS 2025 WS, arXiv:2512.24661)* | In-advance confidence vs. static code pass/fail | **Iterative parameter prediction vs. 3-seed execution mean** |
| **Knowing-Doing Gap** | Wang *(MIRROR, arXiv:2604.19809)* | Discrete action choices (opt-out vs. tool call) | **Quantitative numeric reasoning in dynamic ML search loops** |
| **Autonomous Agent HPO** | AgentHPO *(CPAL 2025)* & OPRO *(ICLR 2024)* | Final test set accuracy of ML models | **Evaluates calibration of the agent's internal reasoning text** |

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
                               │        Deterministic JSON Schema             │
                               │  - Proposed Hyperparameters                  │
                               │  - Predicted Direction (UP / DOWN / STABLE)  │
                               │  - Expected Range [min, max]                 │
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
                               │  - Overconfidence Rate (%)                   │
                               └──────────────────────────────────────────────┘
```

### 1. Evaluated Models (3 Models)
1. **DeepSeek R1** (Open-weights reasoning model)
2. **GPT-4o** (Proprietary general model)
3. **Claude 3.5 Sonnet** (Proprietary coding/reasoning model)

### 2. Imbalanced Benchmark Domains (2 Datasets)
1. **MIT-BIH Arrhythmia (ECG):** Cardiology signal classification under canonical de Chazal inter-patient split (5-class imbalanced).
2. **Kaggle Credit Card Fraud:** Financial fraud detection (binary heavily imbalanced dataset).

### 3. Multi-Seed & Noise Control Protocol
* **JSON Schema Enforcement:** Every model must output a structured JSON block (`direction`, `expected_min`, `expected_max`) alongside CoT text.
* **3-Seed Execution Averaging:** Every proposed configuration is evaluated across **3 random seeds** (`[42, 123, 999]`) to separate true agent miscalibration from stochastic training variance.
* **Total Sample Size:** 2 datasets × 3 models × 15 iterations × 3 seeds = **180 total multi-seed trials**.

---

## 📊 Evaluation Metrics

1. **Directional Accuracy Rate (%):**
   $$\text{Directional Accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( \text{sign}(\text{Actual\_Mean}_i - \text{Baseline}_i) == \text{Predicted\_Direction}_i \right)$$

2. **Mean Absolute Calibration Error (MACE):**
   $$\text{MACE} = \frac{1}{N} \sum_{i=1}^{N} | \text{Target\_Midpoint}_i - \text{Actual\_Mean}_i |$$

3. **Overconfidence Rate (%):**
   $$\text{Overconfidence Rate} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{I}\left( \text{Actual\_Mean}_i < \text{Expected\_Min}_i \right)$$

---

## 🚀 5-Week Project Execution Roadmap

* **Week 1 (Infrastructure & Schema Setup):** Implement structured JSON prompt schema, build `calibration_evaluator.py`, and test 3-seed multi-seed pipeline execution.
* **Week 2 (DeepSeek R1 Evaluation Run):** Run 15-iteration optimization loops on ECG and Credit Fraud datasets using DeepSeek R1; log 3-seed execution means and compute initial MACE baseline.
* **Week 3 (GPT-4o & Claude 3.5 Sonnet Runs):** Execute parallel evaluation loops for GPT-4o and Claude 3.5 Sonnet across both datasets.
* **Week 4 (Diagnostic & Statistical Analysis):** Analyze Directional Accuracy vs. MACE across model families; compute overconfidence rates and plot calibration curves.
* **Week 5 (Paper Writing & Submission Prep):** Draft formal manuscript targeting Tier-1 AI Workshop / Tier-2 AI Journal.

---

## 💡 Practical Real-World Impact

1. **Solving the Automation Dilemma:** Quantifies empirically when AI developers can safely trust an agent's self-reported reasoning vs. when manual verification is required.
2. **GPU Cloud Compute Savings:** Enables early-stopping pre-filters that reject overconfident agent proposals before running long, expensive training jobs.
3. **AI Safety Rule for Biomedical ML:** Establishes a verification rule preventing autonomous clinical AI agents from deploying ungrounded model updates.
