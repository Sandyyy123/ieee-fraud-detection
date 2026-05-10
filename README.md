# IEEE-CIS Fraud Detection — LightGBM + Heterogeneous GNN

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?logo=pytorch) ![LightGBM](https://img.shields.io/badge/LightGBM-gradient%20boosting-green) ![License](https://img.shields.io/badge/license-MIT-lightgrey)

Production-grade two-stage fraud detection pipeline on the **IEEE-CIS / Vesta Corporation** dataset (590,540 transactions, 393 features, 3.5% fraud prevalence). Combines a LightGBM tabular baseline with a heterogeneous Graph Neural Network that models the card-merchant-device co-occurrence structure.

---

## Architecture

```
Raw Transactions
      │
      ▼
┌─────────────────────────────┐
│   Feature Engineering       │  Time deltas, rolling aggregates,
│   (src/model_baseline.py)   │  frequency encodings, device-OS
└────────────┬────────────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
 LightGBM       Heterogeneous GNN
 Baseline        (PyTorch Geometric)
 (ROC-AUC)       Card ── Merchant
                   └──── Device
      │             │
      └──────┬───── ┘
             ▼
     Ensemble / Fallback
     Cost-weighted Score
     (c_fp = 0.5×amt, c_fn = amt)
```

---

## Key Features

- **Chronological 5-fold CV** — no temporal leakage; splits on `TransactionDT` not random shuffle
- **LightGBM baseline** with `is_unbalance=True`, transaction-level deltas, card rolling aggregations, device-OS interactions, frequency encodings for high-cardinality categoricals
- **Heterogeneous GraphSAGE** — tri-partite card/merchant/device graph, focal loss (γ=2), 5× positive-edge oversampling
- **HGT alternative backbone** — Heterogeneous Graph Transformer with 4 attention heads
- **CatBoost ensemble fallback** — 3-seed blend triggers if GNN AUC < 0.85 after 30 epochs
- **Three evaluation metrics** — ROC-AUC (leaderboard), PR-AUC (imbalance-honest), cost-weighted score (DACH banking economics)
- **EU AI Act / BaFin framing** — non-PII risk score with audit-trail logging for high-risk classification compliance

---

## Dataset

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) (Vesta Corporation / Kaggle, 2019).

```bash
kaggle competitions download -c ieee-fraud-detection
unzip ieee-fraud-detection.zip -d data/
```

See `data/README.md` for full setup instructions. Dataset acceptance required via Kaggle competition page.

| Split | Rows | Features |
|-------|------|----------|
| Train transactions | 590,540 | 394 |
| Train identity | 144,233 | 41 |
| Test transactions | 506,691 | 394 |
| Fraud prevalence | 3.5% | - |

---

## Project Structure

```
├── src/
│   ├── model_baseline.py      # LightGBM pipeline (chronological CV, feature engineering)
│   └── model_advanced.py      # Heterogeneous GNN (GraphSAGE / HGT, focal loss)
├── notebooks/
│   └── 01_EDA.ipynb           # Exploratory analysis
├── manuscripts/
│   └── manuscript.md          # IMRaD writeup (~5,000 words)
├── reports/
│   └── references.md          # 30 verified academic references
├── deliverables/
│   └── presentation.html      # Self-contained HTML presentation
├── data/
│   └── README.md              # Dataset download instructions
└── requirements.txt
```

---

## Quick Start

```bash
git clone https://github.com/Sandyyy123/ieee-fraud-detection.git
cd ieee-fraud-detection
pip install -r requirements.txt

# Download dataset first (see data/README.md)

# Run baseline
python src/model_baseline.py

# Run advanced GNN model (GPU recommended)
python src/model_advanced.py
```

---

## Results

| Model | ROC-AUC | PR-AUC | Cost Score |
|-------|---------|--------|------------|
| LightGBM (baseline) | TBD after data download | TBD | TBD |
| Heterogeneous GNN | TBD after data download | TBD | TBD |
| CatBoost ensemble | TBD after data download | TBD | TBD |

Top Kaggle solutions reached **0.9459 private-leaderboard ROC-AUC** using stacked LightGBM/XGBoost/CatBoost with UID engineering.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Gradient boosting | LightGBM, CatBoost |
| Graph neural network | PyTorch Geometric |
| Imbalance handling | Focal loss, SMOTE, class weighting |
| Hyperparameter search | Optuna |
| Evaluation | scikit-learn |

---

## Author

**Dr. Sandeep Grover** — PhD Data Science, independent ML researcher, Mössingen, Germany.

---

## License

MIT
