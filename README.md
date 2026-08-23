# 🔬 Empirical Evaluation of Quantitative Reasoning Calibration in Autonomous LLM Agents

**Title:** Do LLMs Mean What They Say? Evaluating Quantitative Reasoning Calibration in Autonomous LLM Agents  
**Framework Codename:** `QuaRCAA`  
**Location:** `/Users/omkadu/code/UTP/Quantitative Reasoning Calibration in Autonomous LLM Agents/`  
**Master Documentation:** 📖 **[DOCUMENTATION.md](DOCUMENTATION.md)**

---

## 📌 Quick Overview

Autonomous Large Language Model (LLM) agents are rapidly being adopted to design and optimize Machine Learning (ML) pipelines. When suggesting hyperparameter updates, these agents produce natural language Chain-of-Thought (CoT) reasoning detailing *why* a hyperparameter is modified and *what numeric performance gains* ($\Delta \text{Recall}$, $\Delta \text{F1}$) are expected.

QuaRCAA performs the **first empirical diagnostic evaluation** measuring the gap between an LLM agent's written quantitative predictions and actual **3-seed downstream execution means**.

---

## 🚀 Quick Start & Running Benchmarks

### 1. Run Baseline Optimization Loop
Execute the 15-iteration benchmark loop for DeepSeek R1, GPT-4o, or Claude 3.5:

```bash
# DeepSeek R1 on MIT-BIH ECG Arrhythmia Pipeline
python3 run_benchmark.py --model deepseek --dataset ecg --iterations 15

# DeepSeek R1 on Kaggle Credit Card Fraud Pipeline
python3 run_benchmark.py --model deepseek --dataset credit --iterations 15

# GPT-4o or Claude 3.5 Sonnet
python3 run_benchmark.py --model gpt --dataset ecg --iterations 15
python3 run_benchmark.py --model claude --dataset ecg --iterations 15
```

---

## 📊 Result Logging Architecture

* **Immutable Raw Trial Logs:** `logs/raw_trials/[dataset]_[model]_iter[N]_[trial_id].json`  
  Persists full raw prompt strings, raw LLM text responses, timestamps, status codes, seed metrics, and parsed JSON.
* **Run Summaries:** `logs/summary_[dataset]_[model].json`  
  Exports aggregated trajectory metrics across iterations.

---

## 📚 Complete Technical Specification & Documentation

For the full literature breakdown (Kadavath, Barkan, MIRROR), disjoint two-arm architecture, mathematical metric formulas, seed protocols, and code walkthroughs:  
👉 **[Read Master DOCUMENTATION.md](DOCUMENTATION.md)**
