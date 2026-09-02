"""
QuaRCAA Kaggle Credit Card Fraud Detection Pipeline — REAL TRAINING
Trains a real XGBoost classifier on the actual creditcard.csv dataset (284,807 transactions).
Each call to evaluate_single_seed() performs a genuine .fit()/.predict_proba() training run.
No synthetic formulas. No hand-crafted response surfaces.

Hyperparameters tuned:
  - scale_pos_weight: XGBoost's native class imbalance handler (ratio of negative:positive)
  - decision_threshold: Post-hoc probability cutoff for fraud flagging
  - min_child_weight: Minimum sum of instance weights in a child node (controls minority overfitting)
  - max_depth: Tree depth
  - learning_rate: Gradient boosting step size

Metrics computed:
  - macro_f1: Macro-averaged F1 across both classes
  - recall_fraud: Fraud class recall (sensitivity) — catches missed frauds
  - precision_fraud: Fraud class precision — controls false alarms
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import xgboost as xgb
from typing import Dict, Any
from quarcaa.pipelines.base_pipeline import BasePipeline


class CreditFraudPipeline(BasePipeline):
    def __init__(self, csv_path: str = "dataset/creditcard.csv"):
        self.csv_path = csv_path
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(
                f"Credit card dataset not found at '{self.csv_path}'. "
                f"Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
            )
        df = pd.read_csv(self.csv_path)
        self.X = df.drop("Class", axis=1).values.astype(np.float32)
        self.y = df["Class"].values.astype(int)
        fraud_count = int(self.y.sum())
        normal_count = int(len(self.y) - fraud_count)
        print(f"[CreditFraudPipeline] Loaded {len(self.y):,} transactions "
              f"({fraud_count} fraud / {normal_count} normal, ratio 1:{normal_count//fraud_count})")

    def get_baseline_parameters(self) -> Dict[str, float]:
        return {
            "scale_pos_weight": 1.0,    # No class weighting (un-tuned baseline)
            "decision_threshold": 0.50,  # Standard 0.5 cutoff
            "min_child_weight": 1.0,     # XGBoost default
            "max_depth": 6,              # XGBoost default
            "learning_rate": 0.10        # XGBoost default
        }

    def evaluate_single_seed(self, hyperparameters: Dict[str, float], seed: int) -> Dict[str, float]:
        """
        Trains a real XGBoost classifier on the real creditcard.csv dataset.
        Uses a deterministic stratified 70/30 train-test split per seed.
        Returns genuine empirical precision, recall, and F1 scores.
        """
        scale_pos   = float(hyperparameters.get("scale_pos_weight", 1.0))
        thresh      = float(hyperparameters.get("decision_threshold", 0.50))
        min_cw      = float(hyperparameters.get("min_child_weight", 1.0))
        max_depth   = int(round(hyperparameters.get("max_depth", 6)))
        lr          = float(hyperparameters.get("learning_rate", 0.10))

        # Deterministic stratified split (preserves class ratio across seeds)
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=0.30,
            random_state=seed,
            stratify=self.y
        )

        # Real XGBoost training with the proposed hyperparameters
        model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos,
            min_child_weight=min_cw,
            max_depth=max_depth,
            learning_rate=lr,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            eval_metric="logloss",
            verbosity=0,
            use_label_encoder=False
        )
        model.fit(X_train, y_train)

        # Apply post-hoc decision threshold to predicted probabilities
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs >= thresh).astype(int)

        # Compute metrics — genuine empirical outcomes from real training
        recall_fraud    = float(recall_score(y_test, preds, pos_label=1, zero_division=0.0))
        precision_fraud = float(precision_score(y_test, preds, pos_label=1, zero_division=0.0))
        recall_normal   = float(recall_score(y_test, preds, pos_label=0, zero_division=0.0))
        precision_normal = float(precision_score(y_test, preds, pos_label=0, zero_division=0.0))
        macro_f1        = float(f1_score(y_test, preds, average="macro", zero_division=0.0))

        return {
            "macro_f1":          float(np.round(macro_f1, 4)),
            "recall_fraud":      float(np.round(recall_fraud, 4)),
            "precision_fraud":   float(np.round(precision_fraud, 4)),
            "recall_normal":     float(np.round(recall_normal, 4)),
            "precision_normal":  float(np.round(precision_normal, 4))
        }
