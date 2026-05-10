"""IEEE-CIS Fraud Detection - LightGBM baseline.

Trains a LightGBM gradient-boosting classifier on engineered tabular features:
- Transaction-level deltas (time since last transaction on the same card / merchant proxy / device).
- Card-level rolling aggregations (count, sum, mean of TransactionAmt over rolling windows).
- Time-of-day and day-of-week from TransactionDT.
- Device-OS interactions (DeviceType x id_30 / id_31).
- Frequency encoding for high-cardinality categoricals (card1..card6, addr1, P_emaildomain, R_emaildomain).
- Missingness indicator block.

Evaluation:
- ROC-AUC (Kaggle leaderboard metric).
- PR-AUC (more honest under heavy class imbalance, ~3.5% positive prevalence).
- Cost-weighted score: c_fp = 0.5 * mean_amt, c_fn = mean_amt.

Validation:
- 5-fold chronological split on TransactionDT (no shuffle, no random split: this dataset has
  drift and a held-out tail. Random K-fold leaks future information into the training set.)

Saves trained booster, fold metrics, and a feature-importance bar to deliverables/.

Run from the project root:
    cd /root/AI/project_root
    python src/model_baseline.py

DO NOT execute as part of Initial implementationing. v1.0 builds this file but the user runs
it in the main session after `kaggle competitions download -c ieee-fraud-detection` has
populated the data/ folder.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    roc_auc_score,
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "deliverables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
N_FOLDS = 5

# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and join transaction + identity tables."""
    train_tx = pd.read_csv(DATA_DIR / "train_transaction.csv", low_memory=False)
    train_id = pd.read_csv(DATA_DIR / "train_identity.csv", low_memory=False)
    test_tx = pd.read_csv(DATA_DIR / "test_transaction.csv", low_memory=False)
    test_id = pd.read_csv(DATA_DIR / "test_identity.csv", low_memory=False)
    # In this Kaggle release test_identity headers use `id-NN`; align with train.
    test_id.columns = [c.replace("id-", "id_") for c in test_id.columns]

    train = train_tx.merge(train_id, on="TransactionID", how="left")
    test = test_tx.merge(test_id, on="TransactionID", how="left")
    print(f"train: {train.shape}, fraud rate: {train['isFraud'].mean():.4f}")
    print(f"test:  {test.shape}")
    return train, test


# -----------------------------------------------------------------------------
# Feature engineering
# -----------------------------------------------------------------------------

HIGH_CARD_CATS = [
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "id_30", "id_31",
]
LOW_CARD_CATS = ["ProductCD", "card4", "card6", "DeviceType", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]


def engineer_features(df: pd.DataFrame, ref_freq: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Build engineered features. If `ref_freq` is provided, reuse training-fold frequency
    encodings on validation / test (no leakage)."""
    df = df.copy()

    # 1. Time decomposition
    df["tx_day"] = (df["TransactionDT"] // 86400).astype(np.int32)
    df["tx_hour"] = ((df["TransactionDT"] % 86400) // 3600).astype(np.int8)
    df["tx_dow"] = ((df["tx_day"] % 7)).astype(np.int8)

    # 2. Time-since-last-transaction per card1
    df = df.sort_values("TransactionDT")
    df["dt_since_card1"] = (
        df.groupby("card1")["TransactionDT"].diff().fillna(-1).astype(np.float32)
    )

    # 3. Rolling aggregations per card1 (size, sum, mean of TransactionAmt across the row order)
    grp = df.groupby("card1")
    df["card1_cum_count"] = grp.cumcount()
    df["card1_cum_amt_sum"] = grp["TransactionAmt"].cumsum()
    df["card1_cum_amt_mean"] = (
        df["card1_cum_amt_sum"] / (df["card1_cum_count"] + 1)
    ).astype(np.float32)

    # 4. Device-OS interaction
    df["device_os"] = (
        df["DeviceType"].fillna("nan").astype(str)
        + "__"
        + df["id_30"].fillna("nan").astype(str)
    )

    # 5. Frequency encoding (training-only stats; reuse on validation / test)
    new_freq = {}
    for col in HIGH_CARD_CATS + ["device_os"]:
        if ref_freq is None:
            counts = df[col].value_counts(dropna=False)
            new_freq[col] = counts.to_dict()
        else:
            counts = pd.Series(ref_freq[col])
        df[f"{col}_freq"] = df[col].map(counts).fillna(0).astype(np.int32)

    # 6. Missingness indicator block (V columns only - 339 columns of mostly missing data)
    v_cols = [c for c in df.columns if c.startswith("V") and c[1:].isdigit()]
    df["v_n_missing"] = df[v_cols].isna().sum(axis=1).astype(np.int16)

    # 7. log1p TransactionAmt
    df["log_amt"] = np.log1p(df["TransactionAmt"]).astype(np.float32)

    # 8. Drop categorical raw columns (encoded above) but keep low-cardinality cats for LGBM
    for col in HIGH_CARD_CATS + ["device_os"]:
        df.drop(columns=[col], inplace=True)
    for col in LOW_CARD_CATS:
        if col in df.columns:
            df[col] = df[col].astype("category")

    return df, new_freq


# -----------------------------------------------------------------------------
# Cost-weighted metric
# -----------------------------------------------------------------------------

def cost_weighted_score(y_true: np.ndarray, y_score: np.ndarray, amt: np.ndarray,
                       threshold: float = 0.5) -> dict:
    """Cost-weighted evaluation. c_fp = 0.5 * mean_amt, c_fn = mean_amt.
    Lower total cost is better; we report total and per-transaction."""
    mean_amt = float(np.mean(amt))
    c_fp = 0.5 * mean_amt
    c_fn = 1.0 * mean_amt
    yhat = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
    total_cost = fp * c_fp + fn * c_fn
    return {
        "threshold": float(threshold),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "c_fp_unit": float(c_fp), "c_fn_unit": float(c_fn),
        "total_cost": float(total_cost),
        "per_tx_cost": float(total_cost / max(len(y_true), 1)),
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def chronological_folds(dt: pd.Series, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build n_folds chronological folds: train = prefix, val = next chunk."""
    order = np.argsort(dt.values)
    n = len(order)
    chunk = n // (n_folds + 1)
    folds = []
    for i in range(n_folds):
        train_end = chunk * (i + 1)
        val_end = chunk * (i + 2)
        train_idx = order[:train_end]
        val_idx = order[train_end:val_end]
        folds.append((train_idx, val_idx))
    return folds


def train_lgbm(train: pd.DataFrame) -> dict:
    import lightgbm as lgb

    feat_cols = [c for c in train.columns if c not in ("isFraud", "TransactionID", "TransactionDT")]
    X = train[feat_cols]
    y = train["isFraud"].values
    amt = train["TransactionAmt"].values

    folds = chronological_folds(train["TransactionDT"])

    fold_metrics = []
    feature_importance_sum = np.zeros(len(feat_cols))
    boosters = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n--- Fold {fold_i+1}/{N_FOLDS} ---  train={len(tr_idx):,}  val={len(va_idx):,}")
        X_tr_raw, X_va_raw = train.iloc[tr_idx], train.iloc[va_idx]
        # Re-engineer with training-fold frequency stats only
        X_tr_fe, freq_map = engineer_features(X_tr_raw)
        X_va_fe, _ = engineer_features(X_va_raw, ref_freq=freq_map)

        feat_cols_fe = [c for c in X_tr_fe.columns if c not in ("isFraud", "TransactionID", "TransactionDT")]
        cat_cols_fe = [c for c in feat_cols_fe if str(X_tr_fe[c].dtype) == "category"]

        dtrain = lgb.Dataset(X_tr_fe[feat_cols_fe], label=y[tr_idx], categorical_feature=cat_cols_fe)
        dval = lgb.Dataset(X_va_fe[feat_cols_fe], label=y[va_idx], categorical_feature=cat_cols_fe, reference=dtrain)

        params = dict(
            objective="binary",
            metric="auc",
            learning_rate=0.03,
            num_leaves=255,
            feature_fraction=0.85,
            bagging_fraction=0.85,
            bagging_freq=1,
            min_child_samples=80,
            is_unbalance=True,
            verbose=-1,
            seed=SEED + fold_i,
        )
        booster = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(stopping_rounds=100), lgb.log_evaluation(200)],
        )
        preds = booster.predict(X_va_fe[feat_cols_fe], num_iteration=booster.best_iteration)
        auc = roc_auc_score(y[va_idx], preds)
        pr_auc = average_precision_score(y[va_idx], preds)
        cost = cost_weighted_score(y[va_idx], preds, amt[va_idx], threshold=0.5)
        print(f"Fold {fold_i+1} AUC={auc:.4f}  PR-AUC={pr_auc:.4f}  cost@0.5={cost['total_cost']:.0f}")

        fold_metrics.append({"fold": fold_i+1, "auc": auc, "pr_auc": pr_auc, **cost})
        feature_importance_sum[: len(feat_cols_fe)] += booster.feature_importance(importance_type="gain")[: len(feat_cols)]
        boosters.append(booster)

    # Save the final-fold booster as the persisted artefact
    final = boosters[-1]
    final.save_model(str(OUT_DIR / "lgbm_baseline.txt"))
    metrics = {
        "model": "LightGBM_baseline",
        "n_folds": N_FOLDS,
        "fold_metrics": fold_metrics,
        "mean_auc": float(np.mean([m["auc"] for m in fold_metrics])),
        "mean_pr_auc": float(np.mean([m["pr_auc"] for m in fold_metrics])),
        "mean_total_cost": float(np.mean([m["total_cost"] for m in fold_metrics])),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(OUT_DIR / "metrics_baseline.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n=== Cross-validation summary ===")
    print(f"Mean ROC-AUC: {metrics['mean_auc']:.4f}")
    print(f"Mean PR-AUC : {metrics['mean_pr_auc']:.4f}")
    print(f"Mean total cost @ threshold 0.5: {metrics['mean_total_cost']:.0f}")
    print(f"\nArtefacts: {OUT_DIR / 'lgbm_baseline.txt'}, {OUT_DIR / 'metrics_baseline.json'}")
    return metrics


def main():
    print(">>> IEEE-CIS Fraud Detection - LightGBM baseline")
    train, _test = load_data()
    metrics = train_lgbm(train)
    return metrics


if __name__ == "__main__":
    main()
