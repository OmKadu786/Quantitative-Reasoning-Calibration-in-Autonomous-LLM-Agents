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
 │ 1. Architectural Calibration Guard (MIRROR C4-inspired Intervention):       │
 │    (External policy gating/clamping agent updates when predicted range     │
 │     width is excessively wide or rolling MACE > threshold)                  │
 │ 2. Balanced 2x2 Model Matrix: Adds o3-mini / Gemini 2.5 Flash Thinking      │
 │    (2 Trained Extended Reasoning vs. 2 Prompted Step-by-Step Models)       │
 │ 3. Longitudinal Calibration Drift (3-6 Month re-evaluation of API drift)   │
 └─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔬 2. Stage 2 High-Leverage Upgrades Detailed

### 2.1 Mitigation Intervention: MIRROR C4-Inspired Architectural Calibration Guard
* **Why Not Epistemic Self-Knowledge (C2)?** MIRROR (Wang et al., 2026) evaluated feeding calibration history back to agents (C2) across 16 models (~250K instances) and proved it has **no statistically significant effect** ($p=0.90$). Conversely, external architectural constraints (C4) yielded a **76% error reduction**.
* **The Core Research Question:** *Does an external Architectural Calibration Guard (C4) that clamps/gates hyperparameter updates when the agent's predicted interval width is overly vague (high Sharpness) or rolling MACE is high significantly reduce pipeline failure rates?*
* **Implementation (`quarcaa/harness/architectural_guard.py`):**
  $$\text{Action} = \begin{cases} \text{Execute } \theta_{\text{proposed}} & \text{if Sharpness} \le \tau_{\text{width}} \text{ and Rolling MACE} \le \tau_{\text{mace}} \\ \text{Damp/Reject Update} & \text{otherwise} \end{cases}$$
* **Impact:** Evaluates the one intervention paradigm proven to work in literature (C4 architectural constraint) within dynamic AutoML optimization loops.

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
2. **Architectural Guard Module:** Implement `quarcaa/harness/architectural_guard.py` to support the C4 architectural gating mechanism.
