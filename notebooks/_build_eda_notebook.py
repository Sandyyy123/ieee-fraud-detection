"""Helper that builds 01_EDA.ipynb (nbformat v4) from a list of cells.

Run once to materialise the notebook; do NOT execute the notebook itself in
this scaffolding pass.
"""
import json
import os

NB_PATH = os.path.join(os.path.dirname(__file__), "01_EDA.ipynb")


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src,
    }


cells = []

cells.append(md(
    "# Project 14 - IEEE-CIS Fraud Detection\n"
    "\n"
    "## Notebook 01: Exploratory Data Analysis\n"
    "\n"
    "**Goal.** Understand the IEEE-CIS / Vesta payment fraud dataset before building "
    "the LightGBM baseline (`src/model_baseline.py`) and the heterogeneous GNN advanced "
    "model (`src/model_advanced.py`).\n"
    "\n"
    "**Source.** Kaggle competition `ieee-fraud-detection`. Vesta Corporation e-commerce "
    "payment data, anonymised, ~590k training transactions, ~393 features, target `isFraud` "
    "with ~3.5% positive prevalence.\n"
    "\n"
    "**Coverage.** Schema and dtypes, missing-value patterns, target distribution, temporal "
    "structure of `TransactionDT`, card / device / address cardinality, fraud rate by "
    "`ProductCD` and `card4` / `card6`, identity-table join coverage, correlation snapshot "
    "of the V-block.\n"
    "\n"
    "This notebook is built but **not executed**. Run cells top-to-bottom in the main "
    "session after `kaggle competitions download -c ieee-fraud-detection` has populated "
    "`../data/`."
))

cells.append(code(
    "import os\n"
    "import warnings\n"
    "from pathlib import Path\n"
    "\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import seaborn as sns\n"
    "\n"
    "warnings.filterwarnings('ignore')\n"
    "sns.set_theme(style='whitegrid')\n"
    "pd.set_option('display.max_columns', 60)\n"
    "pd.set_option('display.width', 240)\n"
    "\n"
    "DATA_DIR = Path('../data').resolve()\n"
    "print('DATA_DIR:', DATA_DIR)\n"
    "print('Files:', sorted(p.name for p in DATA_DIR.iterdir() if p.is_file()))"
))

cells.append(md(
    "## 1. Load the four CSVs\n"
    "\n"
    "We load with `low_memory=False` to let pandas infer dtypes from the full column. "
    "If memory pressure becomes an issue, downcast `int64 -> int32` and `float64 -> float32` "
    "after loading. The full training transaction table is ~1.5 GB in memory."
))

cells.append(code(
    "train_tx = pd.read_csv(DATA_DIR / 'train_transaction.csv', low_memory=False)\n"
    "train_id = pd.read_csv(DATA_DIR / 'train_identity.csv', low_memory=False)\n"
    "test_tx = pd.read_csv(DATA_DIR / 'test_transaction.csv', low_memory=False)\n"
    "test_id = pd.read_csv(DATA_DIR / 'test_identity.csv', low_memory=False)\n"
    "\n"
    "print('train_transaction:', train_tx.shape)\n"
    "print('train_identity:   ', train_id.shape)\n"
    "print('test_transaction: ', test_tx.shape)\n"
    "print('test_identity:    ', test_id.shape)\n"
    "\n"
    "# Standardise: test_identity headers in this release use `id-01` (hyphen),\n"
    "# train_identity uses `id_01` (underscore). Align.\n"
    "test_id.columns = [c.replace('id-', 'id_') for c in test_id.columns]"
))

cells.append(md(
    "## 2. Schema and dtypes\n"
    "\n"
    "393 / 394 columns is a lot; we summarise by dtype block (`C`, `D`, `M`, `V`, `id`, "
    "`addr`, `dist`, `card`, `email`, `Product`, `Device`, plus core columns)."
))

cells.append(code(
    "def column_block(name):\n"
    "    if name in ('TransactionID', 'TransactionDT', 'TransactionAmt', 'isFraud', 'ProductCD'):\n"
    "        return 'core'\n"
    "    if name.startswith('card'): return 'card'\n"
    "    if name.startswith('addr'): return 'addr'\n"
    "    if name.startswith('dist'): return 'dist'\n"
    "    if name.endswith('emaildomain'): return 'email'\n"
    "    if name.startswith('C') and name[1:].isdigit(): return 'C_count'\n"
    "    if name.startswith('D') and name[1:].isdigit(): return 'D_timedelta'\n"
    "    if name.startswith('M') and name[1:].isdigit(): return 'M_match'\n"
    "    if name.startswith('V') and name[1:].isdigit(): return 'V_vesta'\n"
    "    if name.startswith('id_'): return 'id_identity'\n"
    "    if name.startswith('Device'): return 'device'\n"
    "    return 'other'\n"
    "\n"
    "schema = pd.DataFrame({\n"
    "    'col': train_tx.columns,\n"
    "    'dtype': train_tx.dtypes.astype(str).values,\n"
    "    'n_missing': train_tx.isna().sum().values,\n"
    "    'pct_missing': (train_tx.isna().mean() * 100).round(2).values,\n"
    "})\n"
    "schema['block'] = schema['col'].map(column_block)\n"
    "schema.groupby('block').agg(n_cols=('col', 'count'), mean_pct_missing=('pct_missing', 'mean')).round(2)"
))

cells.append(md("## 3. Target distribution"))

cells.append(code(
    "target = train_tx['isFraud']\n"
    "n_pos = int(target.sum())\n"
    "n = len(target)\n"
    "print(f'Total rows: {n:,}')\n"
    "print(f'Fraud rows: {n_pos:,} ({100*n_pos/n:.3f}% positive)')\n"
    "print(f'Imbalance ratio: {(n - n_pos) / max(n_pos, 1):.1f} : 1 negative-to-positive')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(5, 3.2))\n"
    "ax.bar(['Legit', 'Fraud'], [n - n_pos, n_pos], color=['#38bdf8', '#f87171'])\n"
    "for i, v in enumerate([n - n_pos, n_pos]):\n"
    "    ax.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)\n"
    "ax.set_yscale('log')\n"
    "ax.set_title('Class distribution (log scale)')\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(md("## 4. Temporal structure of `TransactionDT`\n\n"
                "`TransactionDT` is seconds elapsed from a fixed reference. We don't know the absolute "
                "calendar date, so we treat it as a relative timeline."))

cells.append(code(
    "dt = train_tx['TransactionDT']\n"
    "print('Range (seconds):', dt.min(), '..', dt.max())\n"
    "print('Span (days):', (dt.max() - dt.min()) / 86400)\n"
    "\n"
    "# Approximate day-of-cycle bucket\n"
    "train_tx['_day'] = (train_tx['TransactionDT'] // 86400).astype(int)\n"
    "by_day = train_tx.groupby('_day').agg(n=('isFraud', 'size'), n_fraud=('isFraud', 'sum'))\n"
    "by_day['fraud_rate_pct'] = (100 * by_day['n_fraud'] / by_day['n']).round(2)\n"
    "\n"
    "fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)\n"
    "axes[0].plot(by_day.index, by_day['n'], color='#38bdf8')\n"
    "axes[0].set_ylabel('# transactions / day')\n"
    "axes[0].set_title('Daily transaction volume')\n"
    "axes[1].plot(by_day.index, by_day['fraud_rate_pct'], color='#f472b6')\n"
    "axes[1].set_ylabel('Fraud rate (%)')\n"
    "axes[1].set_xlabel('Day index from reference')\n"
    "axes[1].set_title('Daily fraud rate (%)')\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(code(
    "# Hour-of-day pattern (modulo 86,400)\n"
    "train_tx['_hour'] = (train_tx['TransactionDT'] % 86400) // 3600\n"
    "hr = train_tx.groupby('_hour').agg(n=('isFraud', 'size'), n_fraud=('isFraud', 'sum'))\n"
    "hr['fraud_rate_pct'] = (100 * hr['n_fraud'] / hr['n']).round(3)\n"
    "\n"
    "fig, ax1 = plt.subplots(figsize=(9, 3.5))\n"
    "ax1.bar(hr.index, hr['n'], color='#94a3b8', alpha=0.7, label='Volume')\n"
    "ax1.set_xlabel('Hour of day')\n"
    "ax1.set_ylabel('# transactions')\n"
    "ax2 = ax1.twinx()\n"
    "ax2.plot(hr.index, hr['fraud_rate_pct'], color='#f87171', marker='o', label='Fraud rate (%)')\n"
    "ax2.set_ylabel('Fraud rate (%)')\n"
    "ax1.set_title('Hour-of-day: volume (bars) and fraud rate (line)')\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(md("## 5. Card / address / email cardinality"))

cells.append(code(
    "high_card = ['card1', 'card2', 'card3', 'card5', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain']\n"
    "for col in high_card:\n"
    "    nu = train_tx[col].nunique(dropna=True)\n"
    "    miss = train_tx[col].isna().mean() * 100\n"
    "    print(f'{col:20s}  unique={nu:7,d}  missing={miss:5.1f}%')\n"
    "\n"
    "# Fraud rate by card4 (Visa / Mastercard / Discover / AmEx) and card6 (debit / credit)\n"
    "for col in ['card4', 'card6', 'ProductCD']:\n"
    "    print('\\n--- Fraud rate by', col, '---')\n"
    "    print(train_tx.groupby(col).agg(n=('isFraud', 'size'),\n"
    "                                       fraud_pct=('isFraud', lambda x: 100*x.mean())).round(3))"
))

cells.append(md("## 6. Identity-table join coverage"))

cells.append(code(
    "merged = train_tx.merge(train_id, on='TransactionID', how='left', indicator=True)\n"
    "cov = (merged['_merge'] == 'both').mean() * 100\n"
    "print(f'Identity coverage on training transactions: {cov:.2f}%')\n"
    "\n"
    "# Fraud rate when identity is present vs absent\n"
    "merged['_has_id'] = (merged['_merge'] == 'both').astype(int)\n"
    "print(merged.groupby('_has_id').agg(n=('isFraud', 'size'),\n"
    "                                      fraud_pct=('isFraud', lambda x: 100*x.mean())).round(3))"
))

cells.append(md("## 7. Missing-value pattern across the V-block"))

cells.append(code(
    "v_cols = [c for c in train_tx.columns if c.startswith('V') and c[1:].isdigit()]\n"
    "print(f'V-block columns: {len(v_cols)}')\n"
    "v_missing = train_tx[v_cols].isna().mean().round(3) * 100\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(11, 3.5))\n"
    "ax.plot(range(len(v_cols)), v_missing.values, color='#a78bfa')\n"
    "ax.set_xlabel('V-column index')\n"
    "ax.set_ylabel('% missing')\n"
    "ax.set_title('V-block missingness profile (Vesta-engineered features cluster into co-missing groups)')\n"
    "plt.tight_layout(); plt.show()\n"
    "\n"
    "# Co-missing groups: cluster V-columns by their missingness pattern\n"
    "v_pattern = train_tx[v_cols].isna().astype(int)\n"
    "v_pattern_summary = v_pattern.T.duplicated(keep='first').sum()\n"
    "print(f'V-cols whose missingness pattern duplicates an earlier V-col: {v_pattern_summary} / {len(v_cols)}')"
))

cells.append(md("## 8. TransactionAmt distribution and fraud association"))

cells.append(code(
    "amt = train_tx['TransactionAmt']\n"
    "print(amt.describe(percentiles=[0.01, 0.5, 0.9, 0.99, 0.999]).round(2))\n"
    "\n"
    "fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))\n"
    "axes[0].hist(np.log10(amt.clip(lower=0.01)), bins=60, color='#38bdf8', edgecolor='white')\n"
    "axes[0].set_xlabel('log10(TransactionAmt USD)')\n"
    "axes[0].set_title('Amount distribution (log10)')\n"
    "\n"
    "for label, sub, color in [('Legit', amt[target == 0], '#38bdf8'),\n"
    "                            ('Fraud', amt[target == 1], '#f472b6')]:\n"
    "    axes[1].hist(np.log10(sub.clip(lower=0.01)), bins=60, alpha=0.55, color=color, label=label,\n"
    "                  density=True)\n"
    "axes[1].set_xlabel('log10(TransactionAmt USD)')\n"
    "axes[1].set_ylabel('Density')\n"
    "axes[1].set_title('Amount: legit vs fraud')\n"
    "axes[1].legend()\n"
    "plt.tight_layout(); plt.show()"
))

cells.append(md("## 9. Card-level transaction graph: a peek"))

cells.append(code(
    "# Transactions per unique card1 hash (proxy for activity intensity)\n"
    "g = train_tx.groupby('card1').agg(n=('isFraud', 'size'), n_fraud=('isFraud', 'sum'))\n"
    "g['fraud_pct'] = (100 * g['n_fraud'] / g['n']).round(2)\n"
    "print('Top 10 most-used card1 hashes (by volume):')\n"
    "print(g.sort_values('n', ascending=False).head(10))\n"
    "print('\\nTop 10 highest-fraud-rate card1 hashes (min 50 transactions):')\n"
    "print(g[g['n'] >= 50].sort_values('fraud_pct', ascending=False).head(10))"
))

cells.append(md(
    "## 10. Findings summary\n"
    "\n"
    "**Dataset.**\n"
    "- 590,540 training transactions, 393 features. ~3.5% fraud prevalence.\n"
    "- Identity table joins on `TransactionID`, covers ~24% of transactions.\n"
    "- `TransactionDT` is a relative seconds counter, not a calendar timestamp; chronological CV is mandatory.\n"
    "\n"
    "**Class imbalance.**\n"
    "- 28:1 negative-to-positive ratio. Pure accuracy is meaningless.\n"
    "- Headline metrics: ROC-AUC (Kaggle) and PR-AUC (more honest under imbalance).\n"
    "- Production cost-weighted metric: `c_fp = 0.5 * mean_amt`, `c_fn = mean_amt`.\n"
    "\n"
    "**Temporal structure.**\n"
    "- Volume varies day-of-week and hour-of-day; fraud rate spikes overnight.\n"
    "- Test split is the chronological tail (last ~6 months), so any time-leak in feature engineering "
    "(e.g., target-encoding without expanding window) would inflate train scores and crash on test.\n"
    "\n"
    "**Card / device / address structure.**\n"
    "- `card1` cardinality ~13k unique hashes. Fraud is concentrated on a small minority of card1 values.\n"
    "- This is the seed of the GNN model: a card1 hash with 200 fraud transactions is a far stronger signal "
    "than any individual feature value.\n"
    "- `addr1` x `P_emaildomain` x `ProductCD` makes a reasonable merchant proxy (Vesta does not expose merchant ID).\n"
    "\n"
    "**V-block.**\n"
    "- 339 anonymised Vesta features. Heavy block-wise co-missingness: many V-columns share an exact missing "
    "pattern, suggesting they belong to the same engineered family. Most XGBoost / LightGBM solutions either "
    "drop near-duplicate V-cols or PCA-compress them.\n"
    "\n"
    "**Next steps for `src/model_baseline.py`.**\n"
    "- Build engineered features: time-since-last-tx per card1, rolling sums per card1, day-of-week, hour-of-day, "
    "device-OS interaction, frequency-encoded high-cardinality cols, V-block PCA.\n"
    "- LightGBM with `is_unbalance=True`, chronological 5-fold CV.\n"
    "\n"
    "**Next steps for `src/model_advanced.py`.**\n"
    "- Build heterogeneous graph: `Card`, `Merchant` (proxy), `Device` nodes; `card -> merchant`, `card -> device`, "
    "`merchant -> device` edges.\n"
    "- Heterogeneous GraphSAGE (Hamilton 2017) primary; HGT (Hu 2020) as alternative.\n"
    "- Focal loss for transaction-node head; auxiliary self-supervised link prediction.\n"
    "- CatBoost ensemble fallback if GNN training time exceeds budget."
))


nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open(NB_PATH, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Wrote {NB_PATH} with {len(cells)} cells")
