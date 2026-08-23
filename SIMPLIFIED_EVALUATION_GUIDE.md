# 📖 Plain English Guide: How We Evaluate Autonomous LLM Agents

## 🎯 The Main Objective
When an AI agent (like DeepSeek R1, GPT-4o, or Claude 3.5) optimizes a Machine Learning model, it writes out reasoning text promising specific performance gains:
> *"I am changing the loss weight to 18.0. I expect the Macro F1 score to go UP and land between **0.72 and 0.78**."*

**Our Research Goal:** Measure how accurate and honest these AI numerical promises actually are when we run the code in real life.

---

## 🤖 The 3 AI Models & 2 Datasets We Are Testing

### **3 AI Models:**
1. **DeepSeek R1** (Open-weights model with internal extended thinking)
2. **GPT-4o** (OpenAI proprietary model with prompted step-by-step reasoning)
3. **Claude 3.5 Sonnet** (Anthropic proprietary model with prompted step-by-step reasoning)

### **2 Benchmark ML Datasets:**
1. **MIT-BIH Heart ECG Beats:** 5-class cardiology signal classification.
2. **Kaggle Credit Card Fraud:** Financial fraud detection.

---

## 🔄 The 4-Step Evaluation Test Loop

```
  ┌─────────────────────────────────────────────────────────┐
  │ 1. ASK THE AI AGENT FOR PARAMETERS & PREDICTION        │
  │    AI gives new parameters + JSON range prediction      │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 2. C4 SAFETY GUARD CHECK                                │
  │    If prediction range is too vague/overconfident,      │
  │    the Guard clamps the change to keep it safe.        │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 3. RUN REAL CODE 3 TIMES                                │
  │    Train ML pipeline with seeds [42, 123, 999]           │
  │    Calculate true average score                         │
  └────────────────────────────┬────────────────────────────┘
                               │
                               ▼
  ┌─────────────────────────────────────────────────────────┐
  │ 4. AUDIT CALIBRATION & OVERCONFIDENCE                   │
  │    - Directional Accuracy: Did score go UP as promised? │
  │    - MACE Error: How far off was prediction midpoint?  │
  │    - Overconfidence: Did real score fall short?        │
  └─────────────────────────────────────────────────────────┘
```

---

## 📝 Concrete Example Walkthrough

### **Step 1: AI Prediction**
DeepSeek R1 recommends setting `f_weight = 18.0` and predicts:
* **Direction:** `UP`
* **Expected Range:** `[0.72, 0.78]` (Midpoint = **`0.75`**)

### **Step 2: Real Code Execution (3 Seeds)**
We train the ECG model 3 times:
* Seed 42 = `0.68`
* Seed 123 = `0.67`
* Seed 999 = `0.69`
* **True 3-Seed Average:** **`0.68`**

### **Step 3: Calibration Metrics Audit**
1. **Directional Accuracy:** Did it go UP from baseline (`0.65` $\to$ `0.68`)?  
   👉 **PASS ✅** (Direction was correct).
2. **MACE (Calibration Error):**  
   👉 $|0.75 \text{ (predicted midpoint)} - 0.68 \text{ (actual average)}| = \mathbf{0.07}$ *(Off by 7 percentage points)*.
3. **Overconfidence Check:**  
   👉 AI promised at least `0.72`, but real code only hit `0.68`.  
   👉 **OVERCONFIDENT ⚠️**

### **Step 4: C4 Safety Guard Action**
In future steps, if the AI's predicted interval is too vague or its miscalibration error is high, our **C4 Architectural Guard** automatically clamps the parameter shift by 50% before running execution, keeping the optimization safe and stable.

---

## 🔢 Total Experiment Sample Size

* **90 Primary Tests:** 3 Models × 2 Datasets × 15 Iterations.
* **270 Code Training Runs:** 90 tests × 3 random seeds.
* **270 Calibration Observations:** 90 tests × 3 metric scores per iteration.
