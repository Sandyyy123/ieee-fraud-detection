# Improvements - Project 14: IEEE-CIS Fraud Detection

Role B (Improver) review of `/root/AI/liora_projects/14_ieee_fraud/`. Recommendations only; no files modified.

---

## Top recommendation (single highest-leverage change)

**Add a `card1`-aware UID feature block to `model_baseline.py` and use it as a graph-node identity in `model_advanced.py`.** Public top-10 IEEE-CIS solutions all converged on the same dominant trick: deriving stable client UIDs by combining `card1` with `addr1`, `D1n = D1 - tx_day`, and a frequency-stable subset of the `V`-block, which collapses millions of transactions onto roughly 600k-900k unique customers. Without this, both the LightGBM baseline and the GNN are stuck at the ~0.94-0.95 single-model frontier; with it, single LightGBM crosses 0.96 private-LB and the GNN gains a far cleaner card-node identity than `card1` alone (which collides heavily because Vesta hashed it). Concrete next step: add a `derive_uid(df)` helper that builds `uid = card1 _ addr1 _ (D1 - tx_day).round()`, join it back as a feature, and replace `card1` in the GNN node-id map with this UID. See Konstantin Yakovlev's 1st-place writeup (Kaggle, 2019) for the construction. **Priority: HIGH.**

---

## Weaknesses and proposed improvements

### 1. Reproducibility: no `requirements.txt`, no random-seed lock for torch / numpy

The project root has no `requirements.txt`, no `environment.yml`, and no `Dockerfile`. `model_baseline.py` sets `SEED = 42` but only feeds it to LightGBM; numpy and python-random are not seeded. `model_advanced.py` defines `SEED = 42` but never calls `torch.manual_seed`, `np.random.seed`, or `random.seed`, so neither the GNN nor the CatBoost ensemble is deterministic.

**Fix.** Add `requirements.txt` pinning `lightgbm==4.3.0`, `catboost==1.2.5`, `torch==2.2.*`, `torch-geometric==2.5.*`, `pandas==2.2.*`, `scikit-learn==1.4.*`, plus a `set_global_seed(seed)` helper at the top of both scripts that calls `random.seed`, `np.random.seed`, `torch.manual_seed`, `torch.cuda.manual_seed_all`, and `torch.use_deterministic_algorithms(True)`. **Priority: HIGH.**

### 2. Calibration is missing - cost-weighted score reported only at threshold 0.5

`cost_weighted_score` in `model_baseline.py` is hard-coded to threshold 0.5 in the cross-validation loop, even though the manuscript Discussion (Section 4.7) explicitly promises threshold optimisation and probability calibration. With `is_unbalance=True` LightGBM produces wildly miscalibrated probabilities (over-confident on the positive class), so the 0.5-threshold cost figure is uninformative.

**Fix.** Add a `calibrate_and_threshold(y_val, p_val, amt_val)` step that (a) fits an `IsotonicRegression` on out-of-fold predictions and (b) sweeps thresholds in [0.01, 0.99] on a 200-point grid to report the cost-minimising threshold per fold and the calibrated PR-AUC. Add a reliability-diagram PNG to `deliverables/`. Reference: Niculescu-Mizil and Caruana 2005 for calibration choice; Dal Pozzolo 2015 (already cited) for calibration after class rebalancing. **Priority: HIGH.**

### 3. GNN graph leaks the supervised target through node features

`build_graph` in `model_advanced.py` computes per-node aggregates including `fraud_rate=("isFraud", "mean")` over the entire DataFrame, then assigns those features to all nodes used by both the train and the val edges. Even though the supervised loss is masked, the node-feature pipeline has already absorbed val-fold labels into `card.x[:, 3]`. The reported AUC will be optimistic; this is the GNN equivalent of fitting a target encoder on the full data before splitting.

**Fix.** Build node features only from training-mask edges. Concretely: split the DataFrame chronologically first, then call `agg(...)` with a filter `df[df.train_mask]` for the fraud-rate column (count and mean-amount columns can stay full-data since they don't carry the label). Add a unit test that re-runs `build_graph` on a label-shuffled training partition and asserts AUC drops to ~0.5. **Priority: HIGH.**

### 4. CV is chronological, but folds are nested (each fold trains on the previous fold's val)

`chronological_folds` produces train = `[0:k*chunk]`, val = `[k*chunk:(k+1)*chunk]`. By fold 5 the model is trained on 5/6 of the timeline, by fold 1 on 1/6. The five folds are not exchangeable, and the cross-validation mean conflates "small training set, near-future val" with "large training set, distant val". This is "expanding window" CV, and the manuscript should say so; right now Section 2.2 just says "5-fold chronological cross-validation".

**Fix.** Either (a) switch to fixed-size sliding-window CV (each fold has identical train and val sizes, e.g. 4-month train + 1-month val rolled by 1 month), or (b) keep the expanding window but report results per-fold rather than as a mean, and update the manuscript to call it "expanding-window walk-forward validation". Option (a) is closer to the production sliding-window retraining pattern referenced in Discussion 4.6. **Priority: MEDIUM.**

### 5. Advanced model has no ablation over the imbalance-handling stack

The pipeline applies *both* 5x positive-edge oversampling *and* focal loss alpha=0.25 / gamma=2 *and* the auxiliary link-prediction loss, with no ablation. The manuscript Discussion (Section 4.1) claims "either alone is insufficient" without evidence. A reviewer will ask: is the lift coming from the sampler, the loss, or the auxiliary head?

**Fix.** Add a `--ablation {sampler_only, loss_only, both, neither}` flag and run all four configurations under the same chronological 80/20 split. Report a 4-row table in Results 3.2 and update Discussion 4.1 with the empirical decomposition. Likely takes <2 hours of GPU time on the available RTX 5090. **Priority: MEDIUM.**

### 6. Concept drift / temporal generalisation is discussed but not measured

The Discussion (Section 4.3) cites Dal Pozzolo 2018 and acknowledges the train-test drift, but the experimental protocol does not measure it. There is no per-month AUC breakdown, no PSI (Population Stability Index) on the engineered features, and no comparison between val-fold AUC and a held-out tail of the train timeline.

**Fix.** Add a `temporal_diagnostics(train, predictions)` function that computes (a) per-month ROC-AUC on the validation fold, (b) PSI between train-period and val-period feature distributions for the top-15 LightGBM features, and (c) the AUC delta between the closest-val and farthest-val month. Save as `deliverables/temporal_drift.png` and a JSON. **Priority: MEDIUM.**

### 7. No SHAP / regulatory-explainability artefact, despite manuscript promising one

Manuscript Section 4.4 commits to TreeSHAP attribution and "a comparable post-hoc attribution scheme for the GNN, such as integrated gradients on edge attributes plus subgraph extraction". The current `model_baseline.py` saves a feature-importance bar but no SHAP, and `model_advanced.py` saves no attribution at all. For a DACH BaFin / EU AI Act narrative, this is the deliverable a regulator will actually inspect.

**Fix.** Add a `compute_shap.py` script that loads `lgbm_baseline.txt`, samples 5000 val rows, runs `shap.TreeExplainer`, and saves `deliverables/shap_summary.png` plus a per-feature mean absolute SHAP CSV. For the GNN, add a `gnn_attribution.py` that uses `torch_geometric.explain.GNNExplainer` on 100 sampled positive edges and saves `deliverables/gnn_subgraph_examples.png`. **Priority: MEDIUM.**

### 8. Stronger tabular baseline option not benchmarked: CatBoost or LightGBM-Dart with target encoding

The fallback CatBoost ensemble is positioned as a GNN-failure-mode safety net rather than a competitor. Public IEEE-CIS solutions show CatBoost with native categorical handling is within 0.001-0.003 ROC-AUC of LightGBM on this dataset, and the LightGBM-Dart variant gives an additional 0.001-0.003 lift through dropout regularisation in the boosting step. Ablating only LightGBM versus a GNN sets up an unfair comparison.

**Fix.** Promote the CatBoost ensemble to a first-class baseline (run unconditionally, not just on GNN failure). Add a third tabular model: LightGBM with `boosting_type='dart'`, `drop_rate=0.1`, otherwise identical hyperparameters. Report all three in the Results 3.1 table alongside the GNN. **Priority: LOW.**

### 9. Fairness / bias check is absent

The manuscript discusses regulatory framing (EU AI Act, BaFin) at length but does not include a fairness audit. Even with anonymised features, group-level AUC slices on `addr2` (country proxy), `card4` (Visa / MC / etc.), and `P_emaildomain` (gmail.com / yahoo.com / corporate domains) are computable and are exactly the slices a model-risk-management review would request.

**Fix.** Add `fairness_audit.py` that computes per-slice ROC-AUC, false-positive rate at the production threshold, and the demographic-parity ratio between the largest two groups for each of the three slicing variables. Save `deliverables/fairness_slices.png` and a JSON. Reference: Aequitas (Saleiro 2018) or Fairlearn for the methodology. **Priority: LOW.**

### 10. EDA notebook has no output cells executed (already flagged as Phase-1 scaffold) but also has no nbval check or smoke-test

The notebook is JSON-valid but has zero executed outputs. There is no `nbval` or `papermill` smoke-test that confirms the notebook runs end-to-end on a small data sample, which means a reviewer can't tell whether it would actually execute against the downloaded Kaggle data without replaying the full ~700 MB pipeline.

**Fix.** Add a `tests/test_notebook_smoke.py` that uses `papermill` to execute `01_EDA.ipynb` against a 1000-row sample of `train_transaction.csv` (synthetic if the real file is absent) and asserts no cell raised. CI-cheap, catches the most common scaffold bug. **Priority: LOW.**

---

## Summary scoring

| # | Improvement | Priority |
|---|-------------|----------|
| 0 | UID feature block + GNN node-id replacement | HIGH |
| 1 | requirements.txt + global seed lock | HIGH |
| 2 | Calibration + threshold sweep | HIGH |
| 3 | Fix GNN node-feature target leak | HIGH |
| 4 | Sliding-window CV (or honest expanding-window naming) | MEDIUM |
| 5 | Imbalance-stack ablation (4 configs) | MEDIUM |
| 6 | Per-month AUC + PSI temporal diagnostics | MEDIUM |
| 7 | SHAP + GNNExplainer artefacts | MEDIUM |
| 8 | CatBoost + LGBM-Dart as first-class baselines | LOW |
| 9 | Per-slice fairness audit | LOW |
| 10 | papermill smoke-test on EDA notebook | LOW |

The four HIGH items together turn this from a credible Phase 1 scaffold into a defensible production-grade fraud benchmark; #3 in particular is a correctness fix, not a polish item.
