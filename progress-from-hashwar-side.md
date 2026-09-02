# Progress — Hashwar's Side (Modeling & Transfer Learning)

_Last updated: 2026-08-31. Everything below was verified against the actual
files in this repository — not against the docx handoff documents._

---

## 1. Status at a glance

| Item | State |
|---|---|
| Blockers §8.1 (wrong dataset path) | ✅ Fixed — config now points at the complete dataset |
| Blocker §8.2 (SMOTE/type desync) | ✅ Fixed — consumer type now survives SMOTE |
| §8.6 (preprocessing too slow) | ✅ Fixed — fully vectorized |
| §9 lexical day ordering | ✅ Fixed — day columns sorted chronologically |
| Channel nets borrowing residential hyperparameters (§9) | ✅ Fixed — dedicated `channel_training` config block |
| Staged benchmark code (Stages 1–5) | ✅ Written, importable, never trained yet |
| Stage 5 Pakistani-data adapters | ✅ xlsx loader + `prepare_pakistan_target.py` + length adapter (pad/truncate) |
| Preprocessed splits (`data/processed/`) | ❌ Empty — must be regenerated (first run command does this) |
| Trained models (`models/checkpoints/`) | ❌ Empty — nothing has been trained |
| Measured metrics (ROC-AUC / F1 / precision / recall) | ❌ **None exist yet** |
| Transfer learning (Stage 5) | ✅ Unblocked — Maheen's `pakistan_sgcc_format.xlsx` is in the repo (SYNTHETIC theft — see §4.3) |
| IDE terminal | ✅ Recovered via background mode (foreground pane still stuck; restart it when convenient) |
| Missing deps (`xgboost`, `openpyxl`) | ⏳ Installing — `scipy` already present |

**One-line summary:** every known blocker between raw data and the first
training run has been fixed in code, the complete five-stage benchmark is
implemented, and Stage 5 now has both a dataset and an adapter — the only
remaining step is to actually run the pipeline (§5).

---

## 2. Fixes made to the existing pipeline

All fixes target the blockers documented in `PROJECT_OVERVIEW.md`.

### `config/config.yaml`
- `paths.raw_sgcc_csv` repointed from `data/raw/sgcc_data.csv` (**the
  truncated 33,841-row copy — someone had renamed the bad file onto the
  expected path**) to `data/raw/recovered/data.csv` (the complete 42,372-row
  official SGCC dataset). A warning comment marks the truncated copy as
  never-to-be-used.
- New `channel_training` block (epochs 15, lr 1e-3, batch 128) so the three
  auxiliary channel nets are no longer forced to share the residential
  model's hyperparameters.
- The industrial proxy threshold stays at **500 kWh/day deliberately**: the
  benchmark must report how few industrial accounts actually exist, not hide it.

### `src/preprocessing/preprocess.py`
- **Vectorized** the whole per-row transform (impute → cap → scale). The old
  `iterrows()` + pure-Python loop over 42,372 × 1,034 values would have taken
  hours; now it is matrix operations.
- **Day columns sorted chronologically** before use. The raw CSV stores them
  lexically (`2014/1/1, 2014/1/10, …, 2014/1/2, …`), which internally
  shuffled every month before handing the series to a CNN/BiLSTM.
- **§8.2 fix:** the encoded consumer type now rides through SMOTE as an
  auxiliary feature and is re-thresholded afterwards, so `type_train.npy`
  is the same length as `X_train.npy` and every synthetic row carries a
  plausible type. Previously step 4 crashed on a short boolean mask.
- Imputation kept rule-compatible but vectorized (mean of the two immediate
  original neighbours, else 0), the 5 all-NaN rows are excluded outright,
  and the loader prints row count + theft rate and **warns if the truncated
  copy was loaded** (33,841 rows instead of 42,372).
- Splits saved as float32 (half the disk size).

### `src/training/train_channels.py`
- Reads the new `channel_training` config block (falls back to the
  residential block if absent) instead of hard-borrowing it.

---

## 3. What was built: the staged benchmark (`src/experiments/`)

Built exactly to the staged-benchmark requirement: **strict sequential
stages, separate metrics per stage, no skipping ahead, weak results
reported honestly.** Every stage appends to
`experiments_results/benchmark_results.json`; the orchestrator prints the
consolidated table at the end.

| File | Stage | What it does |
|---|---|---|
| `common.py` | — | shared split loader, metric helpers, results JSON writer, consumer-type cross-tab |
| `stage1_xgboost.py` | 1 | 20 engineered tabular features per customer (mean, std, zero-day fraction, max single-day drop, linear + Spearman trend, 30-day windowed volatility, crash ratio, skew/kurtosis, …) → XGBoost classifier. Reports ROC-AUC/F1/precision/recall on the **imbalanced** val split + held-out test + top-10 feature importances. Falls back to sklearn HistGB if xgboost is missing. |
| `stage2_raw.py` | 2 | Residential 1D-CNN + BiLSTM and industrial autoencoder on the **raw daily sequence only** (no boosting). Early stopping on val AUC; industrial τ = 95th percentile of recon error on a held-out normal subset; reports reconstruction-error distributions for normal vs theft. Auto-computes Δ vs Stage 1. |
| `stage3_boosted.py` | 3 | Pretrains the 3 auxiliary channel nets on the train split only (AE on normals; pretext + frequency on all rows), freezes them, stacks to (batch, 4, seq_len), retrains both detectors. Auto-computes Δ vs Stage 2 — the point is to measure whether Channel Boosting helps, not just to add it. Channel + detector checkpoints also land in `models/checkpoints/` under the production names. |
| `stage4_report.py` | 4 | System integration: comparable [0, 1] scores from both detectors (sigmoid probability vs `min(err/τ, 1)` normalized anomaly score), routed by consumer type, ready for the separately-built verification/coordinator layer (deliberately **not** implemented here). Prints the industrial sample-size cross-tabulation and a data-driven reliability flag. |
| `stage5_transfer.py` | 5 | With `--target_csv`: channels + SGCC trunk frozen, fresh trainable dense head fine-tuned on the target data only, evaluated on target val/test **and** re-evaluated on SGCC val so the expected transfer dip is visible. Without it (or if the file is too small): writes an honest **PENDING** entry. A length adapter pads/truncates target series to SGCC's 1,034 days so shorter Pakistani series are accepted. Refuses to fine-tune on fewer than 50 rows / 5 positives per split. |
| `prepare_pakistan_target.py` | — | Converts Maheen's `pakistan_sgcc_format.xlsx` (wide SGCC-like OR long smart-meter layout — auto-detected) into `data/raw/pakistan/pakistan_target.csv` with `CONS_NO, FLAG, <daily kWh>` columns; kW→kWh conversion by detected reading interval; prints the class balance and a synthetic-theft warning. |
| `run_all_benchmark.py` | — | Runs preprocess → 1 → 2 → 3 → 4 → 5 in strict order, **aborting on any stage failure**, then prints the consolidated benchmark table + honesty notes. Supports `--from_stage N` to resume. |

Ground rules enforced throughout:
- Val/test are **never** rebalanced — SMOTE applies to the training split only.
- Class balance and row counts are printed before anything trains.
- Every reported metric is measured by this pipeline on this data; no
  published figure is ever quoted as our own.

---

## 4. What is NOT done yet (and what blocks it)

1. **Nothing has run.** `data/processed/` and `models/checkpoints/` are
   still empty; there are zero measured metrics. The terminal is recovered
   (background mode), so the next action is simply §5.
2. **Maheen's preprocessed-split artifacts vanished.** Her verification
   report shows plausible balance numbers, but the `.npy` outputs are not on
   disk — our first command regenerates them from the complete dataset.
3. **Stage 5 data exists but its theft labels are SYNTHETIC.** Maheen's
   `pakistan_sgcc_format.xlsx` mixes 42 real houses with 42 generated theft
   scenarios (perfectly balanced). Any Stage 5 number must therefore be
   reported as *"synthetic theft injection — exploratory"*, never as
   validated transfer performance. `data_collection_status.md` still (correctly)
   shows 0 confirmed cases.
4. **Industrial detector reliability is expected to be flagged UNRELIABLE:**
   at the 500 kWh/day proxy threshold only ~37 accounts qualify (~12 normal,
   ~8 after the train split). This is reported by design, not papered over.
5. Deps: `scipy` is installed; `xgboost` + `openpyxl` were mid-install at the
   time of writing — `pip install -r requirements.txt` before running.
6. `research/~$pakistan_theft_cases.xlsx` is an Excel lock file — delete it
   before committing.

---

## 5. The single next action

```fish
source venv/bin/activate.fish
pip install -r requirements.txt

# convert Maheen's workbook into the Stage 5 target CSV
python -m src.experiments.prepare_pakistan_target --input pakistan_sgcc_format.xlsx

# full benchmark, Stages 1-5 in strict order
python -m src.experiments.run_all_benchmark --config config/config.yaml \
    --target_csv data/raw/pakistan/pakistan_target.csv
```

Expected console flow: split counts + balances → Stage 1 metrics → Stage 2
metrics (+ Δ vs 1) → Stage 3 metrics (+ Δ vs 2) → Stage 4 unified scores +
industrial counts/reliability → Stage 5 transfer metrics (+ Δ vs Stage 3) →
consolidated table + honesty notes. Everything persists in
`experiments_results/benchmark_results.json`.

Note: no GPU is visible to torch (`cuda: False`), so Stages 2/3/5 train on
CPU — expect hours, not minutes. Early stopping limits the damage; do not
kill the run between stages, each stage's results persist as it finishes.

---

## 6. Known caveats we will state in any report (not optional)

- The consumer-type split is a **magnitude heuristic** — SGCC has no
  consumer-type column. Small industrial users and large households are
  misclassified in both directions.
- Zero-fill imputation of leading NaN runs can mimic the residential theft
  signature (38% of accounts open with ≥30 blank days). Kept identical
  across all stages for comparability; flagged in the honesty notes.
- The industrial score `min(err/τ, 1)` is a normalized ratio, **not** a
  probability, and τ is the 95th percentile of normal error by construction.
- Stage 5 freezes a trunk re-trained on SGCC with the identical Stage 3
  procedure (exact Stage 3 weights cannot accept a new head because their
  head outputs dimension 1). Stated in the results JSON.
- Stage 5 target series shorter than 1,034 days are zero-padded (longer ones
  truncated to the most recent days) to fit the frozen channel nets.
- Stage 5 theft positives in the Pakistani dataset are SYNTHETIC — the result
  is exploratory, labeled as such in the adapter's output and in any report.
- If Channel Boosting does not improve ROC-AUC over Stage 2, that negative
  delta is printed and reported — the table keeps weak numbers.
- Transfer-learning results will be labeled "validated on real target data",
  "synthetic theft injection — exploratory", or "designed, pending data" —
  never anything in between.
