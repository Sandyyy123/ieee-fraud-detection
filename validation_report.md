# Validation Report - Project #14 (IEEE-CIS Fraud Detection)

**Overall: PASS-WITH-WARNINGS**

## Compact summary

Project #14 scaffold is structurally sound and reproducible. All code parses cleanly, the notebook is valid JSON, the manuscript hits 4145 words inside the 4000-5000 target, and the presentation HTML is fully self-contained (zero external resources). Methods named in the manuscript Methods section all appear in `model_baseline.py` or `model_advanced.py`. Five randomly selected references resolve live on CrossRef with title-match True. Em-dash count is zero across all artefacts; no AI-tell phrases anywhere; checkpoint schema includes the four required fields. One genuine warning: the inline citation `[Hamilton 2017]` (GraphSAGE, anchoring the advanced model) is referenced in both manuscript and `model_advanced.py` docstring but is NOT listed as an entry in `reports/references.md`. This is an orphan citation and should be added before publication.

---

## Findings (one per line)

### 1. Notebook validity
- [PASS] `notebooks/01_EDA.ipynb` parses as JSON cleanly via `json.load`.

### 2. Python script syntax
- [PASS] `src/model_baseline.py` parses cleanly via `ast.parse`. No syntax errors.
- [PASS] `src/model_advanced.py` parses cleanly via `ast.parse`. No syntax errors.

### 3. Manuscript word count
- [PASS] `wc -w manuscripts/manuscript.md` = **4145 words**. Inside the 4000-5000 target.

### 4. Self-contained HTML
- [PASS] `grep -E 'href="http|src="http' deliverables/presentation.html` returns **0 hits**. No external CSS, JS, or image resources. Presentation is fully inline-only.

### 5. IMRaD completeness
- [PASS] Title (line 1), Abstract (line 11), Introduction (line 19), Materials and Methods (line 31), Results (line 69), Discussion (line 107), Conclusion (line 139), References (line 143). All eight sections present, ordered correctly, IMRaD-compliant.

### 6. Method drift (manuscript Methods vs scripts)
- [PASS] LightGBM with `is_unbalance=True` -> in `model_baseline.py` (`import lightgbm as lgb`, `is_unbalance` flag).
- [PASS] 5-fold chronological CV on `TransactionDT` -> in `model_baseline.py` (`chronological_folds()`).
- [PASS] Engineered features (time decomposition, rolling per-card aggregations, frequency encoding, V-block missingness) -> all present in `model_baseline.py`.
- [PASS] Heterogeneous GraphSAGE (2 layers, hidden 128, dropout 0.3) -> in `model_advanced.py` (`HeteroGraphSAGE`, two message-passing layers).
- [PASS] HGT alternative (4 attention heads) -> referenced in `model_advanced.py` docstring as alternative.
- [PASS] Focal loss (gamma=2, alpha=0.25) -> in `model_advanced.py`.
- [PASS] CatBoost three-seed ensemble fallback (seeds 42, 1337, 7) -> in `model_advanced.py` (`train_catboost_ensemble`, `catboost_seed{seed}.cbm`).
- [PASS] No method named in Methods is missing from the scripts. No drift detected.

### 7. Citation drift (inline cites vs references.md)
- [PASS] 28 unique inline `[Author Year]` citations found across the manuscript.
- [PASS] 27 of 28 map cleanly to entries in `reports/references.md`: Akoglu 2014, Arik 2021, Bahnsen 2016, Bhattacharyya 2011, Breiman 2001, Buda 2018, Carcillo 2021, Chawla 2002, Chen 2016, Davis 2006, Dou 2020, Fernandez 2018, Fiore 2019, Freund 1997, Friedman 2001, Han 2005, He 2008, He 2009, Hu 2020, Krawczyk 2016, Lin 2020, Liu 2021, Lundberg 2020, Ngai 2011, Pang 2021, Provost 2001, Whitrow 2008.
- [WARN] **Orphan citation: `[Hamilton 2017]`**. Cited in manuscript section 2.4 (anchors the GraphSAGE choice) and referenced in `src/model_advanced.py` docstring as "Hamilton 2017", but no Hamilton entry exists in `reports/references.md`. The expected reference is Hamilton, Ying, Leskovec, "Inductive Representation Learning on Large Graphs", NeurIPS 2017 (arXiv:1706.02216). Add this entry before publication.

### 8. Re-verify 5 random references via CrossRef live API
- [PASS] Breiman 2001 (DOI 10.1023/A:1010933404324) -> HTTP 200, title "Random Forests". Match = True.
- [PASS] Chen Guestrin 2016 (DOI 10.1145/2939672.2939785) -> HTTP 200, title "XGBoost". Match = True.
- [PASS] Chawla 2002 (DOI 10.1613/jair.953) -> HTTP 200, title "SMOTE: Synthetic Minority Over-sampling Technique". Match = True.
- [PASS] Hu 2020 (DOI 10.1145/3366423.3380027) -> HTTP 200, title "Heterogeneous Graph Transformer". Match = True.
- [PASS] Davis Goadrich 2006 (DOI 10.1145/1143844.1143874) -> HTTP 200, title "The relationship between Precision-Recall and ROC curves". Match = True.
- 5 of 5 resolve live with title-match True.

### 9. Em-dash scan
- [PASS] Em-dash (U+2014) count across all artefacts: **0**. Per-file: brief.md 0, notebooks/01_EDA.ipynb 0, reports/references.md 0, src/model_baseline.py 0, src/model_advanced.py 0, manuscripts/manuscript.md 0, deliverables/presentation.html 0.

### 10. AI-tell scan
- [PASS] `grep -riE 'verified by [0-9]+ agents|AI-verified|cross-checked by Claude'` over the project folder returns **0 hits**. No AI-tell phrasing found.

### 11. Checkpoint schema
- [PASS] `checkpoint.json` keys: `project_number`, `title`, `methodology`, `phase`, `status`, `needs_main_session_execution`, `blockers`. All four required fields (`project_number`, `title`, `methodology`, `status`) are present. Two extra fields (`phase`, `needs_main_session_execution`, `blockers`) provide useful state context.

### 12. Phase-1 scaffold note (Liora-specific)
- [INFO] Project is Phase 1 scaffold (not in #1-#8 executed band), so absence of saved model artefacts (`lgbm_baseline.txt`, `gnn_advanced.pt`, `metrics_*.json`) under `deliverables/` is expected and not a finding. Manuscript Results section correctly marks numeric fields as `<TBD>` placeholders to be resolved on main-session execution against the downloaded Kaggle dataset.

---

## Action items before publication

1. Add the Hamilton 2017 GraphSAGE reference entry to `reports/references.md` (suggested DOI lookup: search CrossRef for "Inductive Representation Learning on Large Graphs Hamilton" 2017; the canonical NeurIPS 2017 record). Without it, the GraphSAGE methodology citation has no traceable source.

No other changes required for the scaffold to pass QC.

---

Role A (VALIDATOR) complete.
