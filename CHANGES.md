# CHANGES - Project 14: IEEE-CIS Fraud Detection

## 2026-05-08: Fix GNN node-feature target leak (Improver QA #3)

### Summary

The Improver QA report flagged a correctness bug in `src/model_advanced.py`:
`build_graph()` aggregated `fraud_rate` over the full DataFrame before the
chronological train/val split, so validation-fold labels bled into the
card / merchant / device node features (column index 3). Even though the
supervised edge loss was masked, the GNN encoder could still read leaked
val labels through the node-feature pathway, producing optimistic AUC.

This fix computes `fraud_rate` over training-set rows only. The graph
topology (edges, edge_attr, y) and the label-free aggregates (count,
mean amount, std amount) still use the full DataFrame, since they do not
carry the supervised target.

### File modified

`src/model_advanced.py`

### Function signature change

`build_graph(df)` is now `build_graph(df, train_mask=None)`:

- `train_mask` is a row-aligned boolean numpy array flagging the training
  partition. It is consumed only by the label-derived aggregate.
- If `train_mask` is `None` (pure inference time), the function falls back
  to full-df aggregation and prints a warning.

### Diff (high-level)

#### Before (lines 81-140, original)

```python
def build_graph(df: pd.DataFrame):
    ...
    df = df.dropna(subset=["card_node", "merch_node", "dev_node"]).copy()
    ...
    def agg(group_col, n_nodes):
        g = df.groupby(group_col).agg(
            n=("isFraud", "size"),
            mean_amt=("TransactionAmt", "mean"),
            std_amt=("TransactionAmt", "std"),
            fraud_rate=("isFraud", "mean"),   # <-- LEAK: aggregated over full df
        ).fillna(0.0)
        feats = np.zeros((n_nodes, 4), dtype=np.float32)
        feats[g.index.values, 0] = np.log1p(g["n"].values)
        feats[g.index.values, 1] = np.log1p(g["mean_amt"].values)
        feats[g.index.values, 2] = np.log1p(g["std_amt"].values.astype(float))
        feats[g.index.values, 3] = g["fraud_rate"].values
        return torch.from_numpy(feats)
```

#### After (lines 81-187, fixed)

```python
def build_graph(df: pd.DataFrame, train_mask: np.ndarray | None = None):
    ...
    keep = df[["card_node", "merch_node", "dev_node"]].notna().all(axis=1).values
    if train_mask is not None:
        train_mask = np.asarray(train_mask, dtype=bool)
        if len(train_mask) != len(df):
            raise ValueError(...)
        train_mask = train_mask[keep]
    df = df.loc[keep].copy()
    ...
    if train_mask is None:
        print("WARNING: build_graph called without train_mask; ...")
        df_train_for_labels = df
    else:
        df_train_for_labels = df.loc[train_mask]
        print(f"build_graph: fraud_rate aggregated over {len(df_train_for_labels):,} "
              f"training rows out of {len(df):,} total rows (leak-safe).")

    def agg(group_col, n_nodes):
        g_full = df.groupby(group_col).agg(
            n=("isFraud", "size"),
            mean_amt=("TransactionAmt", "mean"),
            std_amt=("TransactionAmt", "std"),
        ).fillna(0.0)
        g_train = df_train_for_labels.groupby(group_col).agg(
            fraud_rate=("isFraud", "mean"),   # <-- FIXED: train rows only
        ).fillna(0.0)
        feats = np.zeros((n_nodes, 4), dtype=np.float32)
        feats[g_full.index.values, 0] = np.log1p(g_full["n"].values)
        feats[g_full.index.values, 1] = np.log1p(g_full["mean_amt"].values)
        feats[g_full.index.values, 2] = np.log1p(g_full["std_amt"].values.astype(float))
        feats[g_train.index.values, 3] = g_train["fraud_rate"].values
        return torch.from_numpy(feats)
```

#### Caller update in `train_gnn()` (lines 222-258 before, 232-269 after)

The chronological 80/20 split is now computed at the row level on
`TransactionDT` BEFORE `build_graph()` is called, and the resulting mask is
passed in. The previous edge-level mask is still derived afterwards (used
for the supervised training/validation indices) since `build_graph` may
drop rows with missing card / merchant / device IDs.

```python
# Before
data, _df_used = build_graph(df)
...
edge_t = ("card", "transacted_at", "merchant")
cm_dt = data[edge_t].edge_attr[:, 1]
order = torch.argsort(cm_dt)
cutoff = int(0.8 * len(order))
train_mask = torch.zeros(len(order), dtype=torch.bool)
train_mask[order[:cutoff]] = True
val_mask = ~train_mask

# After
edge_t = ("card", "transacted_at", "merchant")
dt_vals = df["TransactionDT"].values
row_order = np.argsort(dt_vals)
row_cutoff = int(0.8 * len(row_order))
row_train_mask = np.zeros(len(df), dtype=bool)
row_train_mask[row_order[:row_cutoff]] = True

data, _df_used = build_graph(df, train_mask=row_train_mask)
...
cm_dt = data[edge_t].edge_attr[:, 1]
order = torch.argsort(cm_dt)
cutoff = int(0.8 * len(order))
train_mask = torch.zeros(len(order), dtype=torch.bool)
train_mask[order[:cutoff]] = True
val_mask = ~train_mask
```

### Behavioural impact

- Reported val AUC will drop to its honest value. Cards / merchants /
  devices that appear only in the val partition now have `fraud_rate = 0.0`
  (neutral prior), instead of leaking their own future label rate.
- Label-free node features (count, mean amount, std amount) are unchanged.
- Inference path (no `train_mask`) preserves the previous behaviour and
  prints a warning so it is not used silently in training.

### Caveats

- The recommended unit test from `improvements.md` ("re-run `build_graph`
  on a label-shuffled training partition and assert val AUC drops to ~0.5")
  is not part of this patch. Adding it requires a runnable GPU smoke
  fixture and is out of scope for this targeted correctness fix.
- Node-row alignment: `build_graph` drops rows where card / merchant /
  device IDs are missing. The caller's `train_mask` is sliced with the
  same `keep` filter before being applied to the label aggregate.
- The CatBoost fallback (`train_catboost_ensemble`) was not affected by
  this leak and is unchanged.

### Files NOT modified

- `manuscripts/` (unchanged, per task instruction)
- `notebooks/01_EDA.ipynb` (unchanged, per task instruction)
- `src/model_baseline.py` (out of scope, no GNN node features there)
- `improvements.md`, `brief.md`, `validation_report.md` (advisory only)
