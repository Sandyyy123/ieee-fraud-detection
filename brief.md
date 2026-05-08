# Project 14 - IEEE-CIS Fraud Detection

**Track:** Data Scientist / MLE
**Difficulty:** 8/10
**Status:** Phase 1 scaffolded (code-only, not executed)

## Goal

Detect fraudulent payment-card transactions on the IEEE-CIS / Vesta Corporation dataset, a real-world e-commerce fraud benchmark with ~590,540 training transactions, ~393 features, and a strongly imbalanced binary target (`isFraud`, ~3.5% positive class). The deliverable is a two-stage modelling stack: a strong gradient-boosting baseline on engineered tabular features, and an advanced graph-neural-network model that exploits the card-merchant-device co-occurrence structure.

## Business framing (DACH banking / AML angle)

Production fraud and anti-money-laundering (AML) systems in DACH financial services share three constraints:

1. **Cost-asymmetric errors.** A false negative (missed fraud) directly costs the chargeback amount; a false positive (genuine transaction blocked) costs customer churn plus operational review. Pure ROC-AUC is insufficient; the deliverable evaluates PR-AUC and a cost-weighted score.
2. **Strict latency budget.** Authorisation decisions resolve in <250 ms. Tree ensembles fit; full GNN inference does not, so the GNN model is positioned as a near-real-time scoring layer (seconds to minutes, e.g., post-authorisation review queue).
3. **Heavy class imbalance.** ~3.5% positive class on IEEE-CIS, often <0.5% in DACH retail-banking AML transaction monitoring. Resampling (SMOTE, undersampling), class weighting, and focal loss are all evaluated.

## Dataset

| Item | Value |
|------|-------|
| Source | Kaggle competition `ieee-fraud-detection` (Vesta Corporation, IEEE-CIS, 2019) |
| Download | `kaggle competitions download -c ieee-fraud-detection` (~700 MB, requires accepted competition rules) |
| Files | `train_transaction.csv` (590,540 x 394), `train_identity.csv` (144,233 x 41), `test_transaction.csv`, `test_identity.csv` |
| Target | `isFraud` (binary, ~3.5% positive in train) |
| Time grain | `TransactionDT` is seconds elapsed since a fixed reference point (not a calendar date), so temporal CV must split chronologically on `TransactionDT` rather than calendar weeks |
| Identity join | `TransactionID` joins identity and transaction on a left key; identity covers ~24% of transactions |

## Deliverables (Liora Phase 1 layout)

- [x] `brief.md` (this file)
- [x] `data/README.md` (Kaggle CLI command + post-download steps; data not auto-downloaded - see blockers below)
- [x] `notebooks/01_EDA.ipynb` (raw, not executed)
- [x] `reports/references.md` (verified academic references)
- [x] `src/model_baseline.py` (LightGBM on engineered features)
- [x] `src/model_advanced.py` (PyG heterogeneous graph: card-merchant-device tri-partite, GraphSAGE / HGT, CatBoost ensemble fallback)
- [x] `manuscripts/manuscript.md` (IMRaD, 4000-5000 words)
- [x] `deliverables/presentation.html` (self-contained, inline CSS)
- [x] `checkpoint.json`

## Methodology

### Baseline (LightGBM)

Engineered feature blocks:
- Transaction-level deltas (time since last transaction on same card, merchant, device).
- Card-level rolling aggregations (count, sum, mean of `TransactionAmt` over rolling windows).
- Time-of-day and day-of-week from `TransactionDT` modulo 86,400 / 604,800.
- Device-OS interactions (`DeviceType` x `id_30` / `id_31`).
- High-cardinality encodings: frequency encoding for `card1`...`card6`, `addr1`, `P_emaildomain`, `R_emaildomain`.
- Missingness indicators on `M1`...`M9` and the Vesta `V` columns.

Loss: standard binary log-loss with `is_unbalance=True`. Evaluation: ROC-AUC (Kaggle metric), PR-AUC (handles class imbalance fairly), and a cost-weighted score using fixed false-positive and false-negative costs (`c_fp = 0.5 * mean_amt`, `c_fn = mean_amt`).

### Advanced (Heterogeneous GNN)

Graph construction:
- Three node types: `Card` (`card1`-`card6` hash), `Merchant` (proxy via `ProductCD` x `addr1` x `P_emaildomain`), `Device` (`DeviceInfo` + `id_30`).
- Edge types: `card -> merchant` (transacted_at), `card -> device` (used_from), `merchant -> device` (seen_via). Edge attributes: `TransactionAmt`, `TransactionDT`.
- Temporal edges: chronological ordering preserved; no future leakage.

Architecture:
- Primary: heterogeneous GraphSAGE with 2 message-passing layers, hidden dim 128, dropout 0.3.
- Alternative: Heterogeneous Graph Transformer (Hu et al. 2020) with 4 attention heads.
- Loss: focal loss (gamma=2) on the transaction-node head, plus auxiliary link-prediction loss for self-supervision.
- Imbalance handling: subgraph sampling that oversamples positive transaction nodes 5x.

Fallback: CatBoost ensemble (3 models, different seeds + categorical encoding strategies) blended via simple averaging. Used if the GNN does not converge or train-time exceeds budget.

## Open questions for Phase 2

1. Should the `card1` hash be treated as a node identity or as a feature? Top Kaggle solutions hashed it both ways and ensembled. Default Phase 2: treat as node identity in the GNN, frequency-encode in the baseline.
2. Temporal CV strategy: 5-fold chronological vs 1 holdout (last 20% by `TransactionDT`)? Default: chronological 5-fold, with the public/private leaderboard split (test set, last ~50% of timeline) serving as the final holdout.
3. PII / regulatory framing for the DACH narrative: position outputs as a non-PII risk score, with audit-trail logging required by BaFin Schedule 4 (operational risk) and the EU AI Act high-risk classification for credit / fraud scoring.

## Blockers

- Dataset is behind Kaggle competition acceptance and weighs ~700 MB (just under the 2 GB rule, but competition T&C requires explicit acceptance of competition rules). Per `liora_agent_rules.md`, dataset download is documented in `data/README.md` rather than auto-fetched. The user has Kaggle credentials at `~/.kaggle/kaggle.json` and accepts the rules out-of-band.

## References

Top 5 references for the methodology stack are listed in `reports/references.md`. Full list is 30 verified entries.
