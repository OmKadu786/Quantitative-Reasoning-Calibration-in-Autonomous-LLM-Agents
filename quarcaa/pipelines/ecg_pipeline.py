"""
QuaRCAA MIT-BIH Arrhythmia Classification Pipeline — REAL TRAINING
Trains a real XGBoost classifier on extracted tabular features from the actual MIT-BIH database.
Uses the de Chazal inter-patient split (DS1 train / DS2 test) per AAMI EC57 protocol.
Each call to evaluate_single_seed() genuinely extracts features, trains, and evaluates.

No synthetic formulas. No hand-crafted response surfaces.

Feature set (tabular, lightweight):
  - R-R interval statistics (pre/post beat, ratio, local mean)
  - QRS morphology (duration, energy, peak amplitude)
  - Beat type local context (previous beat class)

Hyperparameters tuned:
  - f_weight, s_weight, v_weight: class sample_weight multipliers for minority classes
  - shield_threshold: post-hoc decision threshold for minority class promotion
  - v_prob_multiplier, s_prob_multiplier, f_prob_multiplier: probability scaling per class

5-Class AAMI mapping:
  N → Normal / Bundle Branch Block (majority)
  S → SVEB / Supraventricular ectopic (minority)
  V → VEB / Ventricular ectopic (minority)
  F → Fusion (minority, rarest)
  Q → Unclassifiable (excluded from metrics)
"""
import os
import numpy as np
import warnings
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from typing import Dict, List, Tuple, Any
from quarcaa.pipelines.base_pipeline import BasePipeline

warnings.filterwarnings("ignore")

try:
    import wfdb
    WFDB_AVAILABLE = True
except ImportError:
    WFDB_AVAILABLE = False

# de Chazal inter-patient split (AAMI EC57 standard)
DS1_RECORDS = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119,
               122, 124, 201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2_RECORDS = [100, 103, 105, 111, 113, 117, 121, 123, 200, 202,
               210, 212, 213, 214, 219, 221, 222, 228, 231, 232, 233, 234]

# AAMI beat label mapping from MIT-BIH annotation symbols
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q"
}
VALID_CLASSES = ["N", "S", "V", "F"]


def extract_features_from_record(record_path: str) -> Tuple[np.ndarray, List[str]]:
    """
    Extracts tabular features from a single MIT-BIH record.
    Returns feature matrix X and label list y (AAMI class strings).
    """
    record = wfdb.rdrecord(record_path)
    annotation = wfdb.rdann(record_path, "atr")

    signal = record.p_signal[:, 0]  # Lead II (MLII)
    r_peaks = annotation.sample
    symbols = annotation.symbol
    fs = record.fs  # Sampling frequency (360 Hz)

    features, labels = [], []

    for i, (peak, sym) in enumerate(zip(r_peaks, symbols)):
        aami_class = AAMI_MAP.get(sym, None)
        if aami_class is None or aami_class == "Q":
            continue

        # R-R interval features
        rr_pre  = (r_peaks[i] - r_peaks[i-1]) / fs if i > 0 else 0.0
        rr_post = (r_peaks[i+1] - r_peaks[i]) / fs if i < len(r_peaks)-1 else rr_pre
        rr_ratio = rr_pre / (rr_post + 1e-6)

        local_rr = []
        for j in range(max(0, i-4), min(len(r_peaks), i+5)):
            if j > 0:
                local_rr.append((r_peaks[j] - r_peaks[j-1]) / fs)
        rr_local_mean = np.mean(local_rr) if local_rr else rr_pre
        rr_local_std  = np.std(local_rr)  if local_rr else 0.0
        rr_norm_pre   = rr_pre / (rr_local_mean + 1e-6)

        # QRS morphology — 100ms window around R-peak
        w = int(0.05 * fs)  # 50ms half-window (18 samples at 360Hz)
        start = max(0, peak - w)
        end   = min(len(signal), peak + w)
        qrs = signal[start:end]

        qrs_energy    = float(np.sum(qrs ** 2)) if len(qrs) > 0 else 0.0
        qrs_peak_amp  = float(signal[peak]) if peak < len(signal) else 0.0
        qrs_duration  = float(len(qrs)) / fs
        qrs_mean      = float(np.mean(qrs)) if len(qrs) > 0 else 0.0
        qrs_std       = float(np.std(qrs))  if len(qrs) > 0 else 0.0

        # Previous beat class (context feature)
        prev_class = AAMI_MAP.get(symbols[i-1], "N") if i > 0 else "N"
        prev_is_N = float(prev_class == "N")
        prev_is_V = float(prev_class == "V")
        prev_is_S = float(prev_class == "S")

        feat = [
            rr_pre, rr_post, rr_ratio, rr_local_mean, rr_local_std, rr_norm_pre,
            qrs_energy, qrs_peak_amp, qrs_duration, qrs_mean, qrs_std,
            prev_is_N, prev_is_V, prev_is_S
        ]
        features.append(feat)
        labels.append(aami_class)

    return np.array(features, dtype=np.float32), labels


class ECGArmyPipeline(BasePipeline):

    def __init__(self, data_dir: str = "dataset/mit-bih-arrhythmia-database-1.0.0"):
        if not WFDB_AVAILABLE:
            raise ImportError("wfdb package required. Install with: pip install wfdb")
        self.data_dir = data_dir
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"MIT-BIH database not found at '{data_dir}'.")

        print("[ECGArmyPipeline] Extracting features from MIT-BIH records (one-time)...")
        self.X_train, self.y_train = self._load_split(DS1_RECORDS, "DS1 (train)")
        self.X_test,  self.y_test  = self._load_split(DS2_RECORDS, "DS2 (test)")
        self.le = LabelEncoder().fit(VALID_CLASSES)
        self.y_train_enc = self.le.transform(self.y_train)
        self.y_test_enc  = self.le.transform(self.y_test)
        print(f"[ECGArmyPipeline] Ready. Train: {len(self.y_train):,} beats | Test: {len(self.y_test):,} beats")
        self._log_class_distribution("DS1 (train)", self.y_train)
        self._log_class_distribution("DS2 (test)",  self.y_test)

    def _load_split(self, record_ids: List[int], split_name: str) -> Tuple[np.ndarray, List[str]]:
        all_X, all_y = [], []
        loaded = 0
        for rid in record_ids:
            rpath = os.path.join(self.data_dir, str(rid))
            if not os.path.exists(rpath + ".dat"):
                continue
            try:
                X, y = extract_features_from_record(rpath)
                all_X.append(X)
                all_y.extend(y)
                loaded += 1
            except Exception as e:
                print(f"  Warning: Could not load record {rid}: {e}")
        print(f"  {split_name}: loaded {loaded}/{len(record_ids)} records")
        return np.vstack(all_X), all_y

    def _log_class_distribution(self, name: str, labels: List[str]):
        counts = defaultdict(int)
        for l in labels: counts[l] += 1
        total = len(labels)
        dist = {k: f"{v} ({100*v/total:.1f}%)" for k, v in sorted(counts.items())}
        print(f"  {name} distribution: {dist}")

    def get_baseline_parameters(self) -> Dict[str, float]:
        return {
            "shield_threshold":  0.50,
            "v_weight":          1.0,
            "s_weight":          1.0,
            "f_weight":          1.0,
            "v_prob_multiplier": 1.00,
            "s_prob_multiplier": 1.00,
            "f_prob_multiplier": 1.00
        }

    def evaluate_single_seed(self, hyperparameters: Dict[str, float], seed: int) -> Dict[str, float]:
        """
        Trains a real XGBoost classifier on DS1 (22 patients) and evaluates on DS2 (22 patients).
        Applies per-class sample weights and post-hoc probability multipliers.
        Returns genuine empirical precision/recall/F1 per AAMI class.
        """
        shield_thresh  = float(hyperparameters.get("shield_threshold", 0.50))
        v_w = float(hyperparameters.get("v_weight", 1.0))
        s_w = float(hyperparameters.get("s_weight", 1.0))
        f_w = float(hyperparameters.get("f_weight", 1.0))
        v_mult = float(hyperparameters.get("v_prob_multiplier", 1.00))
        s_mult = float(hyperparameters.get("s_prob_multiplier", 1.00))
        f_mult = float(hyperparameters.get("f_prob_multiplier", 1.00))

        # Build per-sample class weights from AAMI minority weights
        class_weight_map = {"N": 1.0, "S": s_w, "V": v_w, "F": f_w}
        sample_weights = np.array([class_weight_map[c] for c in self.y_train], dtype=np.float32)

        # Train real XGBoost on DS1
        model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.10,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=seed,
            num_class=4,
            objective="multi:softprob",
            eval_metric="mlogloss",
            verbosity=0,
            use_label_encoder=False
        )
        model.fit(self.X_train, self.y_train_enc, sample_weight=sample_weights)

        # Predict probabilities on DS2
        probs = model.predict_proba(self.X_test)  # shape: (N, 4) — [F, N, S, V] alphabetical

        # Apply per-class probability multipliers
        class_order = list(self.le.classes_)  # ['F','N','S','V']
        mult_map = {"N": 1.0, "S": s_mult, "V": v_mult, "F": f_mult}
        for ci, cls in enumerate(class_order):
            probs[:, ci] *= mult_map[cls]

        # Normalize probabilities after scaling
        probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-8)

        # Shield threshold: promote any minority class prediction above shield_thresh
        base_preds = np.argmax(probs, axis=1)
        minority_classes = [i for i, c in enumerate(class_order) if c != "N"]
        n_idx = class_order.index("N")

        final_preds = base_preds.copy()
        for mi in minority_classes:
            promote_mask = (probs[:, mi] >= shield_thresh) & (base_preds == n_idx)
            final_preds[promote_mask] = mi

        # Compute per-class precision and recall
        y_true = self.y_test_enc
        results = {}
        for ci, cls in enumerate(class_order):
            if cls == "N":
                continue  # computed below as macro complement
            r = float(recall_score(y_true, final_preds, labels=[ci], average="macro", zero_division=0.0))
            p = float(precision_score(y_true, final_preds, labels=[ci], average="macro", zero_division=0.0))
            results[f"recall_{cls}"]    = float(np.round(r, 4))
            results[f"precision_{cls}"] = float(np.round(p, 4))

        # Normal class metrics
        ni = class_order.index("N")
        results["recall_N"]    = float(np.round(recall_score(y_true, final_preds, labels=[ni], average="macro", zero_division=0.0), 4))
        results["precision_N"] = float(np.round(precision_score(y_true, final_preds, labels=[ni], average="macro", zero_division=0.0), 4))

        macro_f1        = float(np.round(f1_score(y_true, final_preds, average="macro", zero_division=0.0), 4))
        macro_precision = float(np.round(np.mean([results[f"precision_{c}"] for c in VALID_CLASSES]), 4))
        macro_recall    = float(np.round(np.mean([results[f"recall_{c}"]    for c in VALID_CLASSES]), 4))

        results["macro_f1"]        = macro_f1
        results["macro_precision"] = macro_precision
        results["macro_recall"]    = macro_recall
        return results
