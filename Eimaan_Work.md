# Eimaan — Work Log & Technical Documentation
## Electricity Theft Detection (ETD) & Multi-Agent Verification Ops — Hackathon (AliCloud)

This document is a detailed record of **all work done by Eimaan** on this repository, on the
`Eimaan` branch. It covers the end-to-end system: data pipeline, deep-learning detection
channels and models, the **Coordinator Agent** and **Verification Agent** (both designed and
implemented by Eimaan), the FastAPI inference service, the Streamlit operations dashboard,
persistence, the five-stage benchmark, and the honest cross-domain (China → Pakistan)
transfer evaluation.

---

## 1. Project Goal & High-Level Architecture

**Goal.** Detect electricity theft (non-technical loss) for distribution companies (DISCOs)
by ranking consumer accounts on their daily consumption history, then suppress false
positives with operational context so field teams only raid genuine suspects.

**Architecture.** A multi-agent system wrapped around a channel-boosted deep-learning
detector:

```
 daily kWh series
        │
        ▼
 Preprocessing (impute → outlier-cap → min-max → length-align)
        │
        ▼
 4 Channel Stack ── raw │ autoencoder-residual │ masked-pretext │ FFT
        │
        ├── Residential  → CNN + BiLSTM  → sigmoid theft probability
        └── Industrial   → Autoencoder   → reconstruction error → τ-calibrated probability
        │
        ▼
 Coordinator Agent  (routing, calibration, risk tiers, queue building)
        │  calls
        ▼
 Verification Agent (rule-based false-positive suppression + financial loss estimate)
        │
        ▼
 FastAPI (:8000)  ◄──►  Streamlit Ops Dashboard (:8501 / :8502)
        │
        ▼
 SQLAlchemy / SQLite (PostgreSQL via DATABASE_URL) + Alembic migrations
```

**Tech stack.** Python 3.12, PyTorch (CPU on this host; GPU training on Kaggle/Colab),
NumPy/Pandas, XGBoost, FastAPI + Uvicorn, Streamlit, SQLAlchemy + Alembic, pytest.

---

## 2. Data Pipeline & Preprocessing (`src/preprocessing/`)

Implemented and hardened the single source of truth for feature preparation so that
**training and inference use byte-identical transforms** (train/serve parity):

1. **Missing-value imputation** (`impute_missing_matrix` / `impute_single`): a missing cell
   becomes the mean of its two *original* neighbours only when both exist; otherwise `0.0`.
   No forward-propagation of already-filled values (an earlier single-neighbour fallback
   caused train/serve skew and was removed).
2. **Outlier capping** at `mu + 2·sigma` per account.
3. **Per-account min-max scaling** to `[0, 1]`.
4. **Length alignment** to the model window (SGCC = 1034 days). The production inference
   path **front-pads** zeros (`align_sequence_length`) so recent history occupies the final
   timesteps; longer series keep the most recent window.
5. **Consumer-type proxy**: `≥ 500 kWh/day` ⇒ "industrial" (SGCC has no type label; this
   proxy is disclosed as statistically unreliable / exploratory).
6. **SMOTE applied to the training split only** — validation and test keep the natural
   8.53 % theft rate (no leakage).

`transform_single.py` mirrors the batch pipeline exactly for real-time single-series
scoring, including the front-pad alignment.

---

## 3. Detection Channels & Models (`src/channels/`, `src/agents/*/model.py`)

**Four-channel boosted stack** (the core research contribution):

| Channel | Signal |
|---|---|
| Raw | normalized consumption |
| Autoencoder residual | reconstruction error highlights anomalous shape |
| Masked-pretext | self-supervised masked-segment features |
| FFT / frequency | spectral signature of tampering |

- **Residential detector**: CNN + BiLSTM over the 4-channel stack → sigmoid probability.
- **Industrial detector**: autoencoder; anomaly = reconstruction error, calibrated to a
  probability by a logistic sigmoid centred at `τ` (see Coordinator §4.2).

Stages 2–3 were **trained on a Kaggle/Colab T4 GPU** (this laptop is CPU-only); weights
return via `results_and_models.zip` into `models/checkpoints/`. Metrics were later
**recovered on this CPU host by inference-only re-evaluation** (no retraining).

---

## 4. Coordinator Agent (`src/agents/coordinator/coordinator.py`) — *by Eimaan*

The Coordinator is the arbitration layer that turns a raw model output into an
**operationally actionable decision**. Responsibilities:

### 4.1 Model routing & score extraction (`coordinate`)
- `residential` / `commercial` → uses the residential sigmoid probability directly.
- `industrial` → requires a reconstruction error **and** `τ`; converts them via calibration.
- Raises a clear `ValueError` if the required score for a type is missing (no silent
  defaults).

### 4.2 Industrial anomaly calibration (`calibrate_industrial_anomaly_score`)
```
P(theft) = 1 / (1 + exp(-steepness · (err − τ) / τ)),  steepness = 4.0
```
- Guarantees `P = 0.50` exactly at `err == τ` (the normal/anomaly boundary).
- Deliberately replaces a naive linear `err/τ` ratio, which misclassified normal samples.
- Clips the exponent to `[-20, 20]` to avoid overflow; returns `0.0` for `τ ≤ 0` or NaN.

### 4.3 Risk-tier assignment (`assign_risk_tier`)
| Tier | Condition |
|---|---|
| CRITICAL | meter seal broken **or** reverse-current alert **or** prob ≥ 0.85 |
| HIGH | prob ≥ 0.65 |
| MEDIUM | prob ≥ 0.40 |
| LOW | otherwise |

### 4.4 Verification cross-check & result assembly
`coordinate()` calls the **Verification Agent** (`verify_flag`) on the raw score, adopts the
adjusted probability, then emits a `ScoringResult` dataclass carrying: raw score, verified
probability, risk tier, `is_theft_suspect` (prob ≥ threshold, default 0.65), action
recommendation, human-readable `reasons`, the full `adjustments` audit trace, and the
`financial_impact` estimate.

### 4.5 Feeder aggregation & dispatch queue (`aggregate_results`, `coordinate_batch`)
- Per-feeder stats: meters, suspects, CRITICAL/HIGH counts, total stolen kWh, total loss.
- Theft-rate % per feeder.
- **Priority ranking**: CRITICAL first, then verified probability descending — this is the
  field-inspection dispatch queue.
- `aggregate_results` was split out of `coordinate_batch` so the FastAPI batch endpoint can
  aggregate the *exact* per-account `ScoringResult`s it produced (industrial routed through
  the autoencoder with full context) instead of re-coordinating stripped copies.

---

## 5. Verification Agent (`src/agents/verification/verify.py`) — *by Eimaan*

A **deliberately rule-based, fully transparent** false-positive suppression layer. It
cross-checks a telemetry-based theft probability against non-telemetry operational/grid
context *before* a field ticket is dispatched, and **never silently overrides** the score —
every reason and multiplicative adjustment is returned for audit.

### 5.1 `CustomerContext` (the operational inputs)
feeder/transformer IDs, tariff category, sanctioned load, **feeder loss %**, recent audit
result + months since, billing dispute, grid-topology issue, rooftop solar / net-metering,
seasonal occupancy, tamper-event count, meter-seal broken, reverse-current alert,
historical mean kWh, recent-30-day mean kWh, tariff rate.

### 5.2 The seven suppression / escalation rules
1. **Prior audit outcome & recency decay** — `cleared` ≤ 6 mo: multiply by a linear decay
   `0.25 → 1.0` (fresh clearance suppresses hard, old clearance barely); `confirmed_theft`:
   floor probability at **0.95** (recidivism); `meter_fault`: ×0.40 (replace meter, not raid).
2. **Feeder line-loss context** — loss ≥ 25 % and raw ≥ 0.35: ×1.25 boost (high-loss feeder
   correlates with theft); loss ≤ 6 % with no seal/tamper: ×0.65 suppression (drop likely
   vacancy/efficiency).
3. **Grid hardware / CT-PT topology** — known miscalibration or phase imbalance: ×0.45
   (anomaly is probably metering equipment, not theft).
4. **Rooftop solar / net-metering** — verified PV with no seal/tamper: ×0.40 (daylight drop
   is on-site generation).
5. **Seasonal / agricultural vacancy** — registered seasonal profile: ×0.50.
6. **Hardware tamper (overrides all suppressions)** — seal broken / reverse current /
   ≥ 2 tamper events: floor at **0.98**; exactly 1 event: ×1.35 (min 0.80).
7. **Billing dispute** — adds a reason (flag stands; notify field team of billing review).

Final probability is clamped to `[0, 1]`.

### 5.3 Financial loss estimation
```
drop_kwh_per_day      = max(0, historical_mean_kwh − recent_30d_mean_kwh)
estimated_monthly_kwh = drop_kwh_per_day × 30
estimated_monthly_loss= estimated_monthly_kwh × tariff_rate_per_kwh
```
> **Known limitation (documented honestly):** this windowed estimator only sees a drop that
> sits in the *final 30 days*. Real theft rows whose theft window is earlier (or whose tail
> recovered) yield `drop = 0` → Rs 0.00 even when the verdict is CRITICAL. A window-agnostic
> baseline-vs-lowest-segment estimator is the planned improvement.

### 5.4 Action recommendation
`DISPATCH_IMMEDIATE_RAID` (prob ≥ 0.85 or seal broken) · `SCHEDULE_FIELD_INSPECTION`
(≥ 0.65) · `MAINTENANCE_METER_REPLACE` · `VERIFY_SOLAR_INVERTER` · `MONITOR`.

---

## 6. Inference Service (`src/api/app.py`) — FastAPI on :8000

- `GET /health` — reports `models_loaded`, active channel count, sequence length.
- `POST /api/v1/predict/single` — full pipeline: `transform_daily_series` → 4-channel stack →
  residential **or** industrial agent → Coordinator → Verification → auto-registers an
  `Inspection` ticket when `is_theft_suspect`. Returns raw score, verified probability,
  tier, suspect flag, action, reasons, adjustments, financial impact.
- `POST /api/v1/predict/batch` — per-account routing (industrial via autoencoder) then
  `aggregate_results` for the prioritized queue.
- `GET /api/v1/benchmark/results` — reads `benchmark_results.json` **live per request**.
- Torch-optional guard: a calibrated statistical fallback exists, but on this host the real
  neural path is active (4 channels loaded).

---

## 7. Operations Dashboard (`dashboard/app.py`) — Streamlit on :8501 / :8502

Five tabs: **Grid Overview & Feeders**, **Account Deep-Dive & What-If**, **Batch Scoring &
Dispatch Queue**, **Field Inspector Feedback Loop**, **Benchmark & Transfer Lab**.

- Tab 2 lets an operator paste a daily-kWh series, set the Verification context, and get a
  live verdict with the full rule audit trail.
- **Real-data presets added (7/8/9)** so demos run on *measured* rows, not synthetic curves:
  Pakistan confirmed theft **PK003**, SGCC validation theft **row #2712**, SGCC validation
  normal **row #0** — each verified as a true positive/negative through the live API.
- Tab 5 renders the honest benchmark table and the honesty/governance checklist directly
  from `benchmark_results.json`.

---

## 8. Persistence (`src/db.py`, `src/models_db.py`, `alembic/`)

SQLAlchemy ORM with SQLite by default and PostgreSQL via `DATABASE_URL`; Alembic migration
setup (`alembic.ini`, `alembic/`). Inspection tickets and scoring history are persisted so
the field-inspection queue survives restarts.

---

## 9. Five-Stage Benchmark (`src/experiments/`, `experiments_results/`)

| Stage | What | Honest result |
|---|---|---|
| 1 | XGBoost baseline | val ROC-AUC **0.6824**, test 0.6779 |
| 2 | Raw CNN+BiLSTM (1 channel) | val ROC-AUC **0.6835** (reproduces Kaggle `stage2_val_probs.npz` exactly), test 0.6603 |
| 3 | Channel-Boosted CNN+BiLSTM (4 channels) | val **0.7453** / test **0.7529**; **+0.0618 AUC, +0.0704 F1** over Stage 2 |
| 4 | Multi-agent integration | reporting stage (regenerate via `stage4_report`) |
| 5 | Pakistan transfer | **zero-shot**, see §10 |

Metrics for Stages 2–3 were **recovered on this CPU host** by `recover_stage_metrics.py`
(inference-only re-evaluation of the frozen Kaggle weights on the val/test splits) — nothing
retrained, nothing fabricated.

---

## 10. Pakistan Cross-Domain Transfer (Stage 5) — honest zero-shot

- **Label resolution:** the owner confirmed `FLAG = -1` = **theft**; `prepare_pakistan_target`
  now maps `-1 → 1` (it previously mapped to normal, which had silently produced a
  single-class target and corrupted an earlier fine-tune).
- **Why no fine-tune:** the target is **single-class (42 theft / 0 normal)** — ROC-AUC and
  precision are undefined and BCE `pos_weight = 0` ignores every positive. `finetune_pakistan`
  now **aborts** on single-class input rather than overwrite good SGCC weights.
- **What is reported:** a genuine **zero-shot** transfer — the frozen SGCC Stage 3 detector
  scores the 42 confirmed Pakistani theft cases. **Detection recall @0.65 = 64.3 % (27/42)**;
  57.1 % @0.70; 78.6 % @0.50. Reported as **recall only**, never as AUC.
- **Alignment correctness:** the eval **front-pads** 365→1034 to match the production
  `transform_daily_series`. (An earlier tail-pad run inflated this to 83.3 %; corrected and
  documented in `RUN_LOG.md`.)
- A fine-tuned Stage 5 stays **PENDING** until a both-class Pakistani set exists
  (≥ 50 confirmed theft + matching normals).

---

## 11. Honesty & Scientific Governance

- No data leakage (SMOTE train-only; imbalanced val/test).
- Source = public **SGCC** dataset (42,372 rows / 3,615 theft = 8.53 %).
- Industrial ≥500 kWh/day proxy disclosed as exploratory.
- Logistic τ-calibration instead of a naive ratio.
- Every reported number is reproducible from a script on disk; corrections are dated and
  explained in `experiments_results/RUN_LOG.md`.

---

## 12. Testing & Verification

- **32 pytest tests pass** (`tests/`: api, coordinator, preprocessing, verification, db).
- Live end-to-end checks: real theft presets → CRITICAL / `DISPATCH_IMMEDIATE_RAID`;
  real normal → LOW / `MONITOR`; residential vs industrial routing produce distinct scores.

---

## 13. Known Limitations & Next Steps

1. Financial-loss estimator is windowed (last-30-day drop) — see §5.3; make it
   window-agnostic.
2. Stage 4 unified-score table not yet regenerated on this host.
3. Fine-tuned Pakistan Stage 5 pending a both-class dataset.
4. Industrial model is anomaly-based and was not recovered as a discriminative model.

*End of work log — Eimaan, `Eimaan` branch.*
