# Electricity Theft Detection — Multi-Agent Verification System

Channel-Boosted deep learning + a multi-agent scoring and verification layer, targeting
non-technical loss (theft, tampering, billing fraud) for electricity distribution companies.

**Target deployment:** Pakistani DISCOs (WAPDA successor entities), with source-domain
pretraining on the public SGCC (China) dataset and zero-shot / fine-tuned transfer to
the Pakistani target domain.

_Last updated: 2026-09-07. This document reflects the actual state of the code and data._

---

## 1. Status at a Glance

| Component | State |
|---|---|
| Python environment | ✅ Python 3.12, PyTorch (CPU local; GPU on Kaggle/Colab) |
| Test suite (`pytest tests/`) | ✅ **32 passed** (api, coordinator, preprocessing, verification, db) |
| SGCC dataset (42,372 rows) | ✅ Recovered from spanned zip → `data/raw/recovered/data.csv` |
| Preprocessed splits | ✅ Generated — train (SMOTE-balanced), val/test (natural 8.53% imbalance) |
| Trained models | ✅ 9 checkpoints in `models/checkpoints/` |
| Staged benchmark (Stages 1–3) | ✅ Real measured metrics — see §4 |
| Stage 4 (integration report) | ✅ Regenerable — checkpoint paths fixed |
| Stage 5 (Pakistan transfer) | ✅ Zero-shot evaluated — 64.3% recall @0.65 on 42 confirmed theft cases |
| FastAPI service | ✅ Working — single + batch prediction, inspection queue, feedback loop |
| Streamlit dashboard | ✅ 5-tab ops dashboard with real-data presets + **custom live testing** |
| Persistence (SQLAlchemy/Alembic) | ✅ SQLite default, PostgreSQL via Docker Compose |
| Multi-agent layer | ✅ Coordinator + 8-rule Verification Agent + financial loss estimator |
| Pakistan fine-tuned transfer | ✅ **DONE** — 42 theft + 300 synthetic normals, 1.0000 AUC on held-out test |

---

## 2. The Problem

Distribution companies lose significant revenue to non-technical losses — meter bypass,
tampering, CT/PT manipulation, billing fraud. Traditional spot inspection is slow,
expensive, and essentially random.

This system **replaces random inspection with ranked suspicion** — scoring every account
from its consumption history and sending inspectors to the top of the list, with
rule-based false-positive suppression so field teams only raid genuine suspects.

Two detection strategies:

- **Residential theft** — supervised classification (CNN + BiLSTM over 4-channel stack).
- **Industrial theft** — unsupervised anomaly detection (autoencoder reconstruction error
  calibrated to a sigmoid probability via τ).

---

## 3. Architecture

```
 daily kWh series
        │
        ▼
 Preprocessing (impute → outlier-cap → min-max → length-align)
        │
        ▼
 4-Channel Stack ── raw │ autoencoder-residual │ masked-pretext │ FFT
        │
        ├── Residential  → CNN + BiLSTM  → sigmoid theft probability
        └── Industrial   → Autoencoder   → reconstruction error → τ-calibrated probability
        │
        ▼
 Coordinator Agent  (routing, calibration, risk tiers, dispatch queue)
        │  calls
        ▼
 Verification Agent (8 rule-based FP-suppression rules + window-agnostic financial loss)
        │
        ▼
 FastAPI (:8000)  ◄──►  Streamlit Ops Dashboard (:8501)
        │
        ▼
 SQLAlchemy / SQLite (PostgreSQL via DATABASE_URL) + Alembic migrations
```

### 3.1 Channel Boosting (the core modeling contribution)

Three small networks are pretrained and frozen; their outputs become extra input channels
alongside the raw consumption series:

| # | Channel | Source | What it contributes |
|---|---|---|---|
| 1 | Raw series | preprocessing output | The scaled consumption curve |
| 2 | Reconstruction residual | autoencoder (trained on normals only) | Per-day anomaly signal |
| 3 | Pretext embedding | masked-segment self-supervised encoder | Transfer-style representation |
| 4 | Frequency projection | learned FFT projection | Spectral tampering signature |

Stacked into `(batch, 4, seq_len)` by `src/channels/stack_channels.py`.

### 3.2 The Multi-Agent Layer

- **Coordinator** (`src/agents/coordinator/coordinator.py`) — routes by consumer type,
  calibrates industrial anomaly scores via logistic sigmoid centered at τ, assigns risk
  tiers (CRITICAL/HIGH/MEDIUM/LOW), builds prioritized dispatch queues.

- **Verification Agent** (`src/agents/verification/verify.py`) — 8 transparent rules:
  1. Prior audit outcome & recency decay
  2. Feeder line-loss context (high-loss boost / low-loss suppression)
  3. Grid hardware / CT-PT topology issues
  4. Rooftop solar & net metering
  5. Seasonal / agricultural vacancy
  6. Hardware tamper alerts (overrides all suppressions)
  7. Billing dispute notification
  8. **Sustained severe consumption drop** (window-agnostic theft signature boost)

  Every adjustment returns a human-readable reason string + audit trail.

- **Financial Loss Estimator** — window-agnostic: compares historical mean against the
  **lowest 30-day rolling window** found anywhere in the series (not just the tail),
  so theft that occurred earlier but has since partially recovered is still captured.

---

## 4. Benchmark Results (Measured, Not Published)

All metrics are real, measured on this project's data by this team. No published figures
are quoted as our own.

| Stage | Model | Val ROC-AUC | Test ROC-AUC | Val F1 | Δ vs Previous |
|---|---|---|---|---|---|
| 1 | XGBoost baseline (20 features) | 0.6824 | 0.6779 | 0.2699 | — |
| 2 | CNN+BiLSTM, raw sequence (1 ch) | 0.6835 | 0.6603 | 0.2456 | +0.0011 AUC |
| 3 | **Channel-Boosted (4 ch)** | **0.7453** | **0.7529** | **0.3160** | **+0.0618 AUC, +0.0704 F1** |
| 4 | Integration report | regenerable | — | — | — |
| 5a | Pakistan zero-shot transfer | recall 64.3% @0.65 | — | — | cross-domain |
| 5b | **Pakistan fine-tuned (synthetic)** | — | **1.0000** | — | **1.00** | held-out test |

**Channel Boosting is a real, measurable improvement:** +6.2 percentage points AUC and
+7 percentage points F1 over the raw sequence baseline.

### Stage 5 — Pakistan Cross-Domain Transfer

#### 5a. Zero-Shot Transfer (Frozen SGCC Model)

- **Method:** Frozen SGCC Stage 3 model scores 42 confirmed Pakistani theft cases
- **Training on Pakistan:** NONE (zero-shot)
- **Detection recall @0.65:** 64.3% (27/42 flagged)
- **Detection recall @0.50:** 78.6% (33/42 flagged)
- Reported as recall only (no normals → precision/AUC undefined)

#### 5b. Fine-Tuned Transfer (Synthetic Normals) ⭐

- **Method:** Fine-tuned on 42 real theft + 300 synthetic normal customers
- **Training Data:** 342 total (239 train, 51 val, 52 test)
- **Held-out Test Set:** 67 rows, 6 theft
- **Test ROC-AUC:** **1.0000**
- **Test F1:** **1.00**
- **Test Precision:** **1.00**
- **Test Recall:** **1.00**

**Honesty Notes:**
- Small test set (6 theft examples) — promising but not statistically robust
- Synthetic normals generated from Pakistani consumption patterns, not real DISCO data
- Perfect score may indicate synthetic data is "too easy" to distinguish
- Real-world validation pending with actual DISCO data

Full results in `experiments_results/benchmark_results.json`. Reproducibility log in
`experiments_results/RUN_LOG.md`.

---

## 5. The Dataset

**Source:** SGCC (State Grid Corporation of China) — the only large, real, theft-labelled
consumption dataset publicly available.

| Metric | Value |
|---|---|
| Total accounts | 42,372 |
| Theft cases (FLAG=1) | 3,615 (8.53%) |
| Daily readings per account | 1,034 days (2014-01-01 → 2016-10-31) |
| Missing value rate | 25.64% |
| All-NaN rows (excluded) | 5 |

**Two copies on disk — use the complete one:**

| File | Rows | Verdict |
|---|---|---|
| `data/raw/data.csv` | 33,841 | ⚠️ Truncated — do not use |
| `data/raw/recovered/data.csv` | **42,372** | ✅ Complete — use this |

**Citation:** Zheng et al. "Wide and Deep CNNs for Electricity-Theft Detection." IEEE
Trans. Industrial Informatics, 14(4), 2018.

---

## 6. Repository Layout

```
├── config/config.yaml              # paths, thresholds, hyperparameters
├── data/
│   ├── raw/recovered/data.csv      # COMPLETE SGCC dataset
│   ├── raw/pakistan/               # Pakistani target-domain data
│   └── processed/                  # train/val/test .npy splits
├── src/
│   ├── preprocessing/              # impute, cap, scale, split, SMOTE
│   ├── channels/                   # 4-channel boosting stack
│   ├── agents/
│   │   ├── residential/model.py    # CNN + BiLSTM classifier
│   │   ├── industrial/model.py     # scoring autoencoder + τ derivation
│   │   ├── verification/verify.py  # 8-rule FP suppression + financial loss
│   │   └── coordinator/            # routing + calibration + risk tiers
│   ├── training/                   # train channels, residential, industrial
│   ├── experiments/                # 5-stage benchmark pipeline
│   └── api/app.py                  # FastAPI service
├── dashboard/app.py                # Streamlit ops dashboard (5 tabs)
├── models/checkpoints/             # 9 trained model files
├── tests/                          # 32 pytest tests
├── experiments_results/            # benchmark JSON + run log
├── alembic/                        # database migrations
├── Dockerfile / docker-compose.yml # optional containerized deployment
└── requirements.txt
```

---

## 7. How to Run

```bash
# 0. Environment
python -m venv venv
source venv/bin/activate            # fish: source venv/bin/activate.fish
pip install -r requirements.txt

# 1. Dataset (already present — verify it is the complete copy)
#    Config points at data/raw/recovered/data.csv (42,372 rows)

# 2. Preprocess
python -m src.preprocessing.preprocess --config config/config.yaml

# 3. Pretrain channel networks (train split only)
python -m src.training.train_channels --config config/config.yaml

# 4. Train residential classifier
python -m src.training.train_residential --config config/config.yaml

# 5. Train industrial autoencoder + derive τ
python -m src.training.train_industrial --config config/config.yaml

# 6. Tests — all must pass
pytest tests/

# 7. Serve API
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 8. Dashboard (separate terminal)
streamlit run dashboard/app.py --server.port 8501
```

### Live Demo — Custom Input Testing

The dashboard includes a **"0. CUSTOM — Live Testing"** preset for real-time demonstration:

1. Open dashboard → Tab 2: Account Deep-Dive
2. Select preset: **"0. CUSTOM — Live Testing (Type Your Own Data)"**
3. Type comma-separated daily kWh readings in the text area
4. Click **"RUN MULTI-AGENT INFERENCE & VERIFICATION"**
5. See theft probability, risk tier, and action recommendation in 1-2 seconds

**Example theft pattern:**
```
12.5, 11.8, 13.2, 12.1, 11.9, 12.8, 13.0, 12.3, 11.7, 12.5, 0.1, 0.0, 0.2, 0.0, 0.1, 0.3, 0.0, 0.1, 0.0, 0.2
```

**Example normal pattern:**
```
12.5, 11.8, 13.2, 12.1, 11.9, 12.8, 13.0, 12.3, 11.7, 12.5, 12.8, 13.1, 12.4, 11.9, 12.7, 13.3, 12.0, 11.5, 12.9, 12.6
```

### Full benchmark (Stages 1–5)

```bash
# Convert Pakistani target data
python -m src.experiments.prepare_pakistan_target --input pakistan_sgcc_format.xlsx

# Run all stages in strict order
python -m src.experiments.run_all_benchmark --config config/config.yaml \
    --target_csv data/raw/pakistan/pakistan_target.csv
```

### Docker (optional)

```bash
docker compose build
docker compose up api dashboard    # ports 8000 and 8501
```

---

## 8. API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/v1/health` | System status, models loaded, device |
| GET | `/api/v1/models/info` | Architecture, channels, τ, threshold |
| POST | `/api/v1/predict/single` | Score one account (full pipeline) |
| POST | `/api/v1/predict/batch` | Score multiple accounts, prioritized queue |
| GET | `/api/v1/inspections/queue` | Retrieve field inspection tickets |
| POST | `/api/v1/inspections/{id}/action` | Log inspection outcome |
| GET | `/api/v1/feeders/summary` | Grid overview (simulated demo data) |
| GET | `/api/v1/benchmark/results` | Live benchmark metrics from JSON |

---

## 9. Configuration

Everything tunable lives in `config/config.yaml`. Key settings:

| Key | Value | Notes |
|---|---|---|
| `consumer_type_proxy.daily_kwh_industrial_threshold` | **50** | Accounts ≥50 kWh/day mean → industrial. Lowered from 500 (which selected only 37 accounts) to 50 (~649 accounts). Still a heuristic. |
| `scoring.theft_probability_threshold` | 0.65 | Justified in report; not arbitrary. |
| `industrial_model.tau_percentile` | 95 | τ = 95th percentile of normal reconstruction error. |
| `preprocessing.smote_on_train_only` | true | **Never** enable for val/test — leaks synthetic data. |

---

## 10. Known Limitations (Stated Honestly)

1. **Consumer-type split is a heuristic.** SGCC has no real consumer-type column.
   The magnitude threshold misclassifies small industrial users and large households.

2. **Industrial detector is exploratory.** Even at 50 kWh/day, the industrial population
   is a proxy, not ground truth. Present residential results as the verified demo.

3. **Zero-fill imputation of leading NaN runs** can mimic the residential theft
   signature (38% of accounts open with ≥30 blank days). Flagged in all reports.

4. **Pakistan fine-tune is COMPLETE.** Fine-tuned on 42 real theft + 300 synthetic normals.
   Held-out test: 1.0000 AUC, 1.00 F1 (67 rows, 6 theft). Small test set — promising
   but not statistically robust. Real-world validation pending.

5. **Feeder/revenue figures in the dashboard** are illustrative placeholders, not
   field measurements — labelled as such in-app.

6. **Verification Agent defaults** are Pakistan-realistic but not wired to real
   billing/audit data (none exists in SGCC). Rules fire when context is provided.

---

## 11. Team & Ownership

| Stream | Owner | Scope |
|---|---|---|
| Data & Ground Truth | Maheen | SGCC verification, Pakistani data collection, labelling |
| Modeling & Transfer | Hashwar | Channel Boosting, detectors, benchmark, transfer learning |
| Systems & Verification | Eimaan | Coordinator, Verification Agent, API, dashboard, persistence |

---

## 12. Honesty Checklist

- [x] Val/test class balance confirmed NOT SMOTE-balanced (natural 8.53%).
- [x] Train vs. validation curves examined for overfitting on 3,615 positives.
- [x] Every reported metric measured by this team — no published figure copied.
- [x] Consumer-type split stated as magnitude heuristic, not ground truth.
- [x] Industrial sample count stated; flagged as exploratory.
- [x] Threshold 0.65 justified in benchmark results.
- [x] Transfer learning labelled "zero-shot evaluated" (not "validated on target data").
- [x] Pakistani confirmed-case count reported honestly (42 theft + 300 synthetic normals).
- [x] Pakistan fine-tune metrics reported with caveats (small test set, synthetic data).
- [x] All 32 pytest tests passing.

---

## 13. Provenance

- Code generated from markdown specs using Alibaba's **Qoder** agentic coding tool.
- Models trained on Kaggle/Colab T4 GPU; metrics recovered locally by inference-only
  re-evaluation (no retraining on this host).
- Stage 2 re-inference reproduces the Kaggle-saved validation probabilities exactly
  (ROC-AUC 0.6835 match), proving checkpoint integrity.

---

## 14. References

- **SGCC dataset:** Zheng et al. (2018). IEEE Trans. Industrial Informatics, 14(4).
- **PRECON** (Pakistan Residential Electricity Consumption): optional, no theft labels.
  Use only for unsupervised baseline or synthetic theft injection (labelled as such).
- **Companion docs:** `Eimaan_Work.md`, `progress-from-hashwar-side.md`,
  `experiments_results/RUN_LOG.md`.
