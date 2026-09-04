# Reproducibility Run Log — Electricity Theft Detection (ETD)

This log records exactly what was executed, on what host, and what the real
results were. Anything **not** run here is marked `PENDING` with the command
needed to reproduce it. No metric in this project is invented.

- **Source dataset:** SGCC (State Grid Corporation of China) public theft dataset.
- **Host used for this log:** Windows, Python 3.12 venv, **8 GB RAM (CPU only, no GPU)**.
- **Label convention (project-wide):** `0 = normal`, `1 = theft`.
- **Training model:** the laptop is CPU-only, so the deep stages are *trained* in
  the cloud (Kaggle/Colab T4 GPU) and the weights are brought back; metrics are
  then *recovered* locally by inference-only re-evaluation (see §3).

---

## 1. Raw data recovery (spanned archive → single CSV)

The raw SGCC file ships as a **WinZip *spanned* archive** split across segments
in `data/raw/`:

| Segment | Role |
|---|---|
| `data.z01` | first segment (begins with spanning marker `PK\x07\x08`) |
| `data.z02` | middle segment |
| `data.zip` | **last** segment (holds central directory + EOCD) |

Neither `Expand-Archive`, Python `zipfile`, nor `bsdtar` can open a spanned
archive directly. Recovery tool: `scripts/recover_sgcc_csv.py` rebuilds the
single logical ZIP byte-stream, then extracts by **scanning local-file-header
signatures** and streaming-inflating (it does **not** trust the recorded
per-segment offsets, which are wrong for the rebuilt stream).

**Result (confirmed):**
- Rebuilt stream: **52.47 MB**.
- `zipfile` fast path failed as expected (`Bad magic number` — spanned offsets).
- Manual central-directory fallback extracted `data.csv` → `data/raw/recovered/data.csv`.
- Extracted size: **175,194,613 bytes** (size-exact).
- **Row count: 42,372** • **Theft cases (FLAG=1): 3,615** → theft rate **0.0853**.

Reproduce:
```powershell
python scripts/recover_sgcc_csv.py --config config/config.yaml
```

---

## 2. Preprocessing / split regeneration

Command:
```powershell
$env:PYTHONIOENCODING="utf-8"
python -m src.preprocessing.preprocess --config config/config.yaml
```
(`PYTHONIOENCODING=utf-8` is required on the cp1252 Windows console — a real
reproducibility fix; the source previously printed a non-ASCII `~` and crashed
with `UnicodeEncodeError`.)

Pipeline (per `config/config.yaml`): chronological day-column sort → imputation
(mean of **both** original neighbours, else `0.0`) → outlier cap at `mu + 2*sigma`
→ per-row min-max scale to `[0,1]` → **SMOTE on TRAIN only** → stratified
70/15/15 split → `seed=42`.

**Result (verified on disk `data/processed/`):**

| Split | Shape | Class counts `[normal, theft]` | Note |
|---|---|---|---|
| Train | `(54256, 1034)` | `[27128, 27128]` | SMOTE-balanced |
| Val   | `(6355, 1034)`   | `[5813, 542]`     | **natural 8.53% imbalance — no leakage** |
| Test  | `(6356, 1034)`   | `[5814, 542]`     | **natural 8.53% imbalance — no leakage** |

- Loaded **42,372 rows**; excluded **5 all-NaN rows**.
- Val/Test theft total = `542 + 542 = 1,084`; Train (pre-SMOTE) theft = `2,531`
  → `1,084 + 2,531 = 3,615` ✅ matches the source theft count exactly.

---

## 3. Model evidence — benchmark stages

### Stage 1 — XGBoost baseline ✅ MEASURED (real, this host)

Needs only numpy/scipy/xgboost — **no torch**.
```powershell
python -m src.experiments.stage1_xgboost --config config/config.yaml
```

| Metric | Validation | Test |
|---|---|---|
| ROC-AUC | **0.6824** | **0.6779** |
| F1 (@thr) | 0.2699 (@0.25) | 0.2423 (@0.25) |
| Precision | 0.2036 | 0.1822 |
| Recall | 0.4004 | 0.3616 |
| Accuracy | 0.8153 | 0.8071 |
| n / positives | 6355 / 542 | 6356 / 542 |

Top predictors: `max_single_day_drop`, `p95`, `zero_days_frac` — i.e. the
**sudden drop-to-zero** theft signature. An honest, modest baseline.

### Stages 2–3 — ✅ TRAINED (Kaggle GPU) + METRICS RECOVERED (this host, CPU)

The deep stages are trained in the cloud (`kaggle files/*.ipynb` →
`run_all_benchmark.py` on a T4 GPU). The run returns `results_and_models.zip`;
the **weights** were merged into `models/checkpoints/`, but the consolidated
metrics JSON was not — so `benchmark_results.json` previously held only Stage 1.
The real numbers were **recovered on this host by inference-only re-evaluation**
(no retraining) of the returned checkpoints on the frozen val/test splits:

```powershell
python -m src.experiments.recover_stage_metrics --config config/config.yaml
```

| Stage | Detector | Val ROC-AUC | Test ROC-AUC | Val F1 | Val Acc |
|---|---|---|---|---|---|
| 2 | CNN+BiLSTM, raw sequence (1 channel) | **0.6835** | 0.6603 | 0.2456 | 0.7050 |
| 3 | CNN+BiLSTM + **Channel Boosting** (4 channels) | **0.7453** | **0.7529** | 0.3160 | 0.8589 |

**Channel Boosting delta (Stage 3 − Stage 2, validation): ROC-AUC +0.0618,
F1 +0.0704.** The 4-channel stack (raw + autoencoder-residual + masked-pretext +
FFT) is a real, measurable improvement over the raw sequence and over Stage 1.

**Recovery is validated, not assumed:** Stage 2 re-inference reproduces the
Kaggle-saved `stage2_val_probs.npz` **exactly — val ROC-AUC 0.6835 (MATCH)** —
proving `residential_stage2.pt` is the genuine returned model and the CPU
re-evaluation is faithful.

**Checkpoint naming (important):**
- `residential_stage2.pt` → Stage 2 raw detector (`num_channels=1`).
- `residential_model_best.pt` → **the Stage 3 boosted detector.** `stage3_boosted.py`
  writes its final model to this *production* name (line 252); the API loads it.
- `residential_stage3.pt` → **NOT Stage 3.** It was overwritten by
  `finetune_pakistan.py` (see §4) and is not used by the API.

**Industrial detector (Stages 2–3): SKIPPED** — the `>=500 kWh/day` consumer-type
proxy leaves too few industrial-typed rows to train or evaluate reliably.

### Stage 4 — integration report (regenerable, not a trained model)

Stage 4 is a *reporting* stage (Coordinator + Verification over Stage 3 scores),
not a separately trained model. Re-run `stage4_report` on a GPU host to
regenerate its unified-score table.

### Stage 5 — Pakistani zero-shot transfer: ✅ EVALUATED (China → Pakistan, no fine-tune)

The 42 Pakistani rows are **confirmed theft** (data owner), so the target is
**single-class (42 theft / 0 normal)**. A discriminative *fine-tune* is impossible on one
class (AUC/precision undefined; BCE `pos_weight = 0`), so Stage 5 is reported as an honest
**zero-shot transfer**: the **frozen SGCC Stage 3** detector scores the confirmed cases.

```powershell
python -m src.experiments.stage5_zeroshot_pakistan --config config/config.yaml
```

| Threshold | Detection recall | Flagged |
|---|---|---|
| **0.65 (production)** | **64.3%** | 27 / 42 |
| 0.70 (Stage-3 best-F1) | 57.1% | 24 / 42 |
| 0.50 | 78.6% | 33 / 42 |

Theft-score spread: mean **0.705**, median **0.728**, min 0.121, max 0.999. With **zero**
Pakistani training the China-trained model flags **27 of 42** real theft cases at the
production threshold — a genuine cross-domain transfer result. **Recall only** (no normals →
precision/AUC undefined). Series are **front-padded** 365→1034 to match the production
`transform_daily_series` alignment, so this number is **deployment-consistent**. See §4.

> **Correction (9/4):** an earlier version reported **83.3% (35/42)** using *tail*-padding.
> The deployed API **front-pads** (recent history at the end), so the honest,
> production-consistent recall is **64.3% (27/42)**.
> `stage5_zeroshot_pakistan.py` now front-pads to match the live transform.

> **Correction to an earlier claim:** a previous version of this log said
> `import torch` raises `MemoryError` on this host and that only statistical
> fallback was possible. That is **no longer true** — torch imports and runs here
> (the API loads all 4 channels; the recovery + zero-shot scripts ran BiLSTM inference
> on CPU). The API still keeps its torch-optional guard purely as a safety net.

---

## 4. Pakistan path — ✅ zero-shot EVALUATED · ⏸ fine-tune PENDING (needs normals)

Command (regenerate the target CSV from the workbook):
```powershell
$env:PYTHONIOENCODING="utf-8"
python -m src.experiments.prepare_pakistan_target `
  --input pakistan_sgcc_format.xlsx `
  --out data/raw/pakistan/pakistan_target.csv
```

**Reality (verified on disk):**
- `pakistan_sgcc_format.xlsx` → **42 rows × 365 daily readings, all `FLAG = -1`**.
- `pakistan_theft_cases.xlsx` → **EMPTY template, 0 rows** (schema only).
- **Label contradiction RESOLVED:** the data owner confirmed `FLAG = -1` = **theft**.
  `prepare_pakistan_target.py` now maps `-1 → 1` (was `→ 0`), so `pakistan_target.csv`
  = **42 theft rows**. `finetune_pakistan.py` was made consistent (accepts `-1` *or* the
  normalised `1` as theft).

**Why a fine-tune is still impossible (and now guarded):** the file is **single-class**
(42 theft / **0 normal**). Fine-tuning a binary detector on one class is degenerate —
`pos_weight = n_neg/n_pos = 0` ignores every positive and AUC is undefined. The earlier
`finetune_pakistan.py` run (which wrote `residential_stage3.pt` on 9/3) actually trained on
**42 all-normal rows** because of the old `-1 → 0` mapping — doubly degenerate. That file is
**not** used by the API. `finetune_pakistan.py` now **ABORTS** on single-class input rather
than overwrite good SGCC weights.

**What Stage 5 reports instead (honest):** a **zero-shot** transfer evaluation —
`stage5_zeroshot_pakistan.py` runs the frozen SGCC Stage 3 model over the 42 confirmed theft
cases: **64.3% detection recall at the production threshold (27/42)**, front-padded to match
the deployed `transform_daily_series`. This is a real China → Pakistan transfer result,
labelled zero-shot (never as an AUC).

Consequence: **zero-shot transfer IS measured (recall only).** A **fine-tuned** Stage 5
remains `PENDING` until a **both-class** Pakistani set exists — **≥ 50 confirmed** DISCO /
FIR / inspection theft cases (**100+ preferred**) *plus* matching normal customers. Any
synthetic theft injection is **exploratory only** and must be labelled as such — never
presented as validated transfer.

---

## 5. Demo-critical defect fixes (this pass)

| # | Fix | File(s) |
|---|---|---|
| 1 | **Industrial batch routing** — batch now scores every item through the same single-item path and aggregates. | `src/api/app.py`, `src/agents/coordinator/coordinator.py` (`aggregate_results`) |
| 2 | **Train/serve parity** — `impute_single` matches training exactly (mean of **both** original neighbours, else `0.0`). | `src/preprocessing/transform_single.py` |
| 3 | **Test-client dep + pins** — added `httpx2`, pinned `fastapi`/`starlette`/`uvicorn`/`pydantic` in lockstep. | `requirements.txt` |
| 4 | **Torch-optional API** — lazy/guarded torch import; degrades to statistical fallback if torch is ever unavailable. | `src/api/app.py` |
| 5 | **Dashboard honesty** — Tab 5 reads **live** benchmark JSON (real Stage 1–3 metrics; Stages 4–5 status shown honestly); Tabs 1 & 4 business/feeder/inspection numbers clearly labelled **simulated demo data**; API URL is env-configurable (`ETD_API_URL`). | `dashboard/app.py` |
| 6 | **Console encoding** — removed the non-ASCII literal that crashed cp1252 stdout. | `src/preprocessing/preprocess.py` |
| 7 | **API root route** — `GET /` now redirects to `/docs` instead of returning `404 {"detail":"Not Found"}`. | `src/api/app.py` |
| 8 | **Idempotent initial migration** — `alembic upgrade head` no longer collides with `init_db()`'s `create_all` (guards on existing tables). | `alembic/versions/532792a9ce13_*.py` |
| 9 | **Pakistan label mapping corrected** — owner confirmed `FLAG = -1` = theft; `prepare_pakistan_target.py` now maps `-1 → 1` (was `→ 0`), and `finetune_pakistan.py` accepts `-1` *or* normalised `1` as theft. Regenerated `pakistan_target.csv` = 42 theft rows. | `src/experiments/prepare_pakistan_target.py`, `src/training/finetune_pakistan.py` |
| 10 | **Single-class fine-tune guard** — `finetune_pakistan.py` now **aborts** when the target has only one class (pos_weight=0, AUC undefined) instead of writing a degenerate model over good SGCC weights. | `src/training/finetune_pakistan.py` |
| 11 | **Honest Stage 5 zero-shot eval** — new inference-only script scores the 42 confirmed theft cases with the frozen SGCC Stage 3 model and reports **detection recall** (83.3% @0.65), never a fabricated AUC. | `src/experiments/stage5_zeroshot_pakistan.py` |

---

## 6. Test suite ✅ GREEN

```powershell
venv\Scripts\python.exe -m pytest -p no:cacheprovider --tb=short -q
```
**Result: 32 passed** (27 core + 5 persistence tests; see §8). The API tests
import `src.api.app`; torch now loads on this host, and the API keeps a
torch-optional statistical fallback as a safety net — so the suite passes
**without** a GPU either way.

---

## 7. Known limitations (stated plainly)

1. **Industrial detection is exploratory.** The `>=500 kWh/day` consumer-type
   proxy selects only **~37 accounts** in SGCC — far too few to validate an
   industrial model. Present the **residential** path as the verified demo; the
   industrial agent is **future/experimental work**.
2. **Stages 1–3 have real metrics.** Stage 1 (XGBoost) measured here; Stages 2–3
   recovered by CPU re-inference from Kaggle-trained weights, with Stage 2
   cross-checked against the saved Kaggle probabilities. **Stage 4** is a
   regenerable report; **Stage 5** (Pakistan transfer) is unproven — blocked by
   data, not compute.
3. **Pakistan transfer unproven.** The target file is single-class; no confirmed
   local theft labels exist yet (see §4).
4. **Feeder/revenue/hit-rate figures in the dashboard are illustrative UI
   placeholders**, not field measurements, and are labelled as such in-app.

---

## 8. Persistence layer (field-inspection database)

Inspection tickets and field-outcome feedback are stored in a **real database**
instead of in-memory dicts, so the ground-truth history **survives API restarts**.

- `src/db.py` — engine + session factory. Backend chosen solely by the
  `DATABASE_URL` env var (default `sqlite:///./etd.db`; PostgreSQL for
  compose/cloud). `init_db()` creates tables at startup; `get_db()` is the
  per-request FastAPI dependency.
- `src/models_db.py` — `inspections` (ticket state, upserted) and
  `inspection_feedback` (append-only audit log). `Inspection.to_dict()` preserves
  the exact JSON shape the API returned before, so no response changed.
- `src/api/app.py` — `predict_single`, `/inspections/queue`, and
  `/inspections/{id}/action` use `Depends(get_db)`; the in-memory dicts are gone.
- `alembic/` — versioned migrations. `env.py` is wired to the app metadata +
  `DATABASE_URL` (SQLite batch mode). Initial revision
  `532792a9ce13_initial_inspection_tables` applies cleanly and is **idempotent**
  (safe alongside `init_db()`'s `create_all`).
- `docker-compose.yml` — adds a healthchecked `postgres:16-alpine` `db` service
  and points the API at it via `DATABASE_URL` (`depends_on: service_healthy`).
  Docker is **optional** — local dev runs entirely on SQLite with no Docker.

Commands:
```powershell
# Local (SQLite, zero config) -- tables auto-create on API startup
venv\Scripts\python.exe -m uvicorn src.api.app:app --port 8000

# Versioned migrations (any backend set via DATABASE_URL)
venv\Scripts\python.exe -m alembic upgrade head

# Full stack incl. PostgreSQL (ONLY if Docker is installed)
docker compose up --build
```

Verified: **32 passed** (5 persistence tests, incl. cross-session durability).
