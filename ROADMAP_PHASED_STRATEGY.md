# 🗺️ QuaRCAA: Phased Research Roadmap & Tier-1 Expansion Plan

**Framework Codename:** `QuaRCAA` (**Qua**ntitative **R**easoning **C**alibration in **A**utonomous **A**gents)  
**Strategy Objective:** Protect current UTP Attachment submission deadlines with Stage 1 core diagnostic benchmark while establishing a clear, high-leverage Stage 2 expansion for Tier-1 AI Venue submission (NeurIPS/ICLR Workshop or AI Journal).

---

## 🎯 1. Two-Stage Phased Roadmap Strategy

```
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                      STAGE 1: Core Diagnostic Benchmark                     │
 │              (UTP Attachment Paper & Immediate Implementation)             │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ - 3 Models: DeepSeek R1 (Extended CoT), GPT-4o, Claude 3.5 Sonnet           │
 │ - 2 Domains: MIT-BIH ECG Arrhythmia & Kaggle Credit Card Fraud              │
 │ - 90 Experimental Units (270 3-seed pipeline runs / 270 calibration obs)   │
 │ - Deliverable: Pure Empirical Diagnostic of Continuous Miscalibration & MACE│
 └──────────────────────────────────────┬──────────────────────────────────────┘
                                        │ Immediate Target Secured
                                        ▼
 ┌─────────────────────────────────────────────────────────────────────────────┐
 │                     STAGE 2: Mitigation & Tier-1 Upgrade                    │
 │               (Post-UTP / Top AI Workshop & Journal Submission)             │
 ├─────────────────────────────────────────────────────────────────────────────┤
 │ 1. Intervention Experiment: In-Context Self-Calibration Prompting           │
 │    (Feeds past [Prediction vs. 3-Seed Actual] history back into CoT prompt) │
 │ 2. Balanced 2x2 Model Matrix: Adds o3-mini / Gemini 2.5 Flash Thinking      │
 │    (2 Trained Extended Reasoning vs. 2 Prompted Step-by-Step Models)       │
 │ 3. Longitudinal Calibration Drift (3-6 Month re-evaluation of API drift)   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 2. Stage 2 High-Leverage Upgrades Detailed

### 2.1 Mitigation Experiment: In-Context Self-Calibration Prompting
* **The Core Research Question:** *Can LLM agents learn to self-correct their quantitative forecasting within an optimization session when shown their own past calibration errors?*
* **Implementation:** At iteration $k$, the prompt injects a structured calibration feedback trace of iterations $1 \dots k-1$:
  $$\text{Feedback}_i = \left\{ \text{Predicted Range: } [\text{Min}_i, \text{Max}_i], \text{ Actual 3-Seed Mean: } \mu_i, \text{ Gap: } \Delta_i \right\}$$
* **Impact:** Transforms the paper from a purely descriptive benchmark into a **validated algorithmic intervention study**, significantly increasing review score strength.

### 2.2 Balanced 2×2 Model Architecture Matrix
To resolve the $n=1 \text{ vs. } n=2$ reasoning model limitation, Stage 2 expands the model matrix to a balanced $2 \times 2$ grid:

| Reasoning Mechanism | Open-Weights | Proprietary |
|---|---|---|
| **Trained Extended Thinking (Native CoT)** | **DeepSeek R1** | **OpenAI o3-mini / Gemini 2.5** |
| **Prompted Step-by-Step (Standard)** | *(Exploratory)* | **GPT-4o / Claude 3.5 Sonnet** |

### 2.3 Longitudinal Calibration Drift (3-6 Month Audit)
* **The Core Research Question:** *Does an agent's reasoning calibration degrade or drift as providers silently update backend API weights?*
* **Implementation:** Re-run the identical 90-experimental-unit benchmark harness 3 to 6 months post-initial baseline without modifying prompt or code logic.

---

## 🚦 3. Immediate Action Plan

1. **Lock Stage 1 Execution:** Complete the 90 experimental unit execution across DeepSeek R1, GPT-4o, and Claude 3.5 Sonnet on ECG & Credit Card Fraud.
2. **Modular Architecture Readiness:** Include stub interfaces in `quarcaa/prompts/few_shot_calibration.py` to seamlessly enable Stage 2 self-calibration prompting without refactoring codebase.
