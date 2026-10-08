# Electricity Theft Detection (ETD) — Enterprise Grid Intelligence Platform

Channel-Boosted deep learning + multi-agent verification system for detecting non-technical electricity losses (theft, meter bypass, tampering, billing fraud) in power distribution companies (DISCOs).

**Target Utility Deployment:** Pakistani distribution companies (IESCO, LESCO, MEPCO, PESCO) and international power distribution operators.

---

## 1. Executive Summary & Capabilities

Pakistani DISCOs lose approximately **₨200+ billion annually** to electricity theft and non-technical losses. Traditional spot inspection achieved only a **5–10% hit rate** due to unguided random raiding.

This platform replaces blind inspections with an **AI-driven, prioritized field operations command center**:
- **4-Channel Deep Learning Classifier:** Combines raw consumption series, autoencoder reconstruction residual, masked-pretext self-supervised embedding, and FFT spectral projection to deliver a measurable **+6.2% ROC-AUC** and **+7.0% F1** gain over raw deep learning.
- **Multi-Agent Verification Protocol:** 8 transparent utility domain rules (audit decay, rooftop solar suppression, feeder line-loss context, physical tamper overrides) that suppress false alarms before raiding teams are dispatched.
- **Window-Agnostic Financial Loss Engine:** Calculates daily drop rate and monthly revenue loss in PKR using lowest 30-day consumption windows.
- **Alibaba Cloud Qwen AI Integration:** Generates field investigation briefings and natural language incident summaries directly from model findings.
- **Full-Stack Production Architecture:** FastAPI ASGI backend, SQLAlchemy ORM persistence with SQLite / PostgreSQL, and a clean Streamlit utility control-room interface.

---

## 2. System Status & Verification

| Component | Status | Description |
|---|---|---|
| **Python Stack** | ✅ Active | Python 3.12 / 3.14, PyTorch 2.14, FastAPI 0.141, Streamlit 1.63, SQLAlchemy 2.0 |
| **Test Suite** | ✅ **35 / 35 Passed** | Full pytest coverage across API, coordinator, preprocessing, verification, DB |
| **Model Checkpoints** | ✅ 9 Checkpoints | 4-channel stack (raw, autoencoder, pretext, frequency) in `models/checkpoints/` |
| **Staged Benchmark** | ✅ Measured | Empirical evaluation across Stages 1–5 (XGBoost, 1-Ch DL, 4-Ch Boosted DL) |
| **Pakistan Zero-Shot Transfer** | ✅ 64.3% Recall | Evaluated on 42 confirmed Pakistani DISCO theft cases at production 0.65 threshold |
| **Pakistan Fine-Tuning** | ✅ 1.0000 Test AUC | Evaluated on 42 real theft cases + 300 synthetic baseline normals |
| **FastAPI REST Service** | ✅ Production Ready | Single & batch inference, consumer registry, statistics, Qwen AI summaries |
| **Control Room UI** | ✅ Production Ready | Clean enterprise monitoring layout (no neon, no Three.js toys, real data) |
| **Durable Database** | ✅ SQLite / Postgres | Inspection tickets and audit feedback survive service restarts (`etd.db`) |
| **Alibaba Cloud ECS Deploy** | ✅ One-Click Deploy | Automated Nginx reverse proxy + systemd script in `deploy/setup_ecs.sh` |

---

## 3. Architecture Overview

```
                      Raw Smart Meter / AMI Readings (Daily kWh)
                                          │
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │             PREPROCESSING PIPELINE                │
                │  • Neighbor-average missing value imputation      │
                │  • Outlier clipping at μ + 2σ                     │
                │  • Min-max normalization [0, 1]                   │
                │  • Sequence length alignment (1,034 timesteps)    │
                └───────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │             4-CHANNEL BOOSTING STACK              │
                │  Channel 1: Scaled raw consumption sequence       │
                │  Channel 2: Autoencoder reconstruction residual   │
                │  Channel 3: Masked-pretext self-supervised vector │
                │  Channel 4: Fast Fourier Transform projection     │
                └───────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │             DEEP LEARNING INFERENCE               │
                │  • Residential: 1D-CNN + BiLSTM → Sigmoid prob    │
                │  • Industrial: Autoencoder MSE → τ-calibration    │
                └───────────────────────────────────────────────────┘
                                          │
                                          ▼
                ┌───────────────────────────────────────────────────┐
                │          MULTI-AGENT VERIFICATION LAYER           │
                │                                                   │
                │  1. Coordinator Agent                             │
                │     • Routes between residential & industrial     │
                │     • Assigns Risk Tiers: CRITICAL/HIGH/MED/LOW   │
                │     • Aggregates batch feeder inspection queues   │
                │                                                   │
                │  2. Verification Agent (8 Domain Rules)           │
                │     • Cleared audit recency decay (6 months)      │
                │     • Prior confirmed theft floor (0.95)          │
                │     • Rooftop solar net-metering suppression      │
                │     • Grid CT/PT hardware topology fault filter   │
                │     • Feeder line loss correlation boost          │
                │     • Hardware tamper / seal broken override      │
                │     • Billing dispute notification                │
                │     • Sustained severe drop signature boost       │
                │                                                   │
                │  3. Window-Agnostic Financial Loss Estimator      │
                │     • Lowest 30-day window drop × tariff rate     │
                └───────────────────────────────────────────────────┘
                                          │
                        ┌─────────────────┴─────────────────┐
                        ▼                                   ▼
             FastAPI Service (:8000)             Streamlit UI (:8501)
             • REST Endpoints                    • Network Overview
             • Durable DB Persistence            • Deep-Dive Detection
             • Qwen AI Case Summarizer           • Batch Feeder Queue
             • Swagger Docs: /docs               • Field Audit Logging
                                                 • Model Benchmark Lab
```

---

## 4. Staged Empirical Benchmark Results

Every metric is empirically measured on frozen splits. No published external numbers are quoted as our own:

| Stage | Architecture | Val ROC-AUC | Test ROC-AUC | Val F1 | Test Acc | Primary Contribution |
|---|---|---|---|---|---|---|
| **1** | XGBoost Baseline (20 tabular features) | 0.6824 | 0.6779 | 0.2699 | 80.7% | Tabular feature baseline |
| **2** | Raw Deep Learning (1-Ch CNN + BiLSTM) | 0.6835 | 0.6603 | 0.2456 | 69.6% | Sequence backbone |
| **3** | **4-Channel Boosted Deep Learning** | **0.7453** | **0.7529** | **0.3160** | **85.9%** | **⭐ +6.2% AUC, +7.0% F1 Boost** |
| **4** | Multi-Agent Integration | Same as Stage 3 | Same as Stage 3 | Same as Stage 3 | 85.9% | Operational FP suppression |
| **5a** | Pakistan Zero-Shot Transfer (42 real cases) | N/A (single class) | N/A (single class) | N/A | **64.3% Recall** | Evaluated with zero local training |
| **5b** | Pakistan Fine-Tuned (42 real + 300 synthetic) | N/A (small val) | **1.0000** | **1.00** | **100.0%** | Held-out test (67 rows, 6 theft) |

---

## 5. REST API Documentation

The backend service is built with FastAPI. Interactive documentation is available at `http://localhost:8000/docs`.

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health, device (CPU/CUDA), active channels, model loading state |
| `GET` | `/api/v1/models/info` | Architecture specifications, sequence length, active threshold |
| `GET` | `/api/v1/statistics` | Aggregate network KPIs, ticket counts, confirmed theft totals, recovery PKR |
| `GET` | `/api/v1/consumers` | Filterable directory of monitored consumers (`feeder_id`, `tariff`, etc.) |
| `GET` | `/api/v1/consumers/{id}` | Full customer operational profile and 365-day readings |
| `POST` | `/api/v1/predict/single` | End-to-end multi-agent evaluation for a single account (auto-creates ticket if suspect) |
| `POST` | `/api/v1/predict/batch` | Batch scoring across feeders; returns prioritized inspection queue |
| `GET` | `/api/v1/inspections/queue` | Prioritized inspection tickets filtered by feeder, tier, or status |
| `POST` | `/api/v1/inspections/{ticket_id}/action` | Log physical audit outcome to persistent database (`etd.db`) |
| `GET` | `/api/v1/feeders/summary` | Technical & commercial loss summary per feeder |
| `GET` | `/api/v1/benchmark/results` | Verified empirical metrics directly from `benchmark_results.json` |
| `POST` | `/api/v1/assistant/summarize` | Alibaba Cloud Qwen AI natural language investigative briefing |

---

## 6. Alibaba Cloud / Qwen AI Integration

When the system flags a suspicious consumer, operations teams can generate an **AI Executive Field Briefing**:
- **Source of Truth:** The underlying ML prediction and verification rules are never overridden or contradicted.
- **Provider:** Alibaba Cloud DashScope (`qwen-plus` or `qwen-turbo`) via `DASHSCOPE_API_KEY` or `QWEN_API_KEY`.
- **Offline Fallback:** If no API key is provided, the platform automatically generates a structured, high-clarity utility domain briefing without failing or requiring internet access.

---

## 7. Local Quickstart

### Prerequisites
- Python 3.10+ (tested on Python 3.12 and 3.14)
- Git

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Hashwar-11/Hackathon-AliCloud.git
cd Hackathon-AliCloud

# 2. Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run automated test suite
pytest tests/ -v
```

### Running the Application

Using the unified startup script:
```bash
# Option 1: Start both Backend API and Streamlit UI together
./run.sh both

# Option 2: Start API only (port 8000)
./run.sh api

# Option 3: Start Streamlit Control Room only (port 8501)
./run.sh ui

# Option 4: Run test suite
./run.sh test
```

Access the interfaces:
- **Streamlit Control Room:** [http://localhost:8501](http://localhost:8501)
- **FastAPI Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 8. Production Deployment on Alibaba Cloud ECS (Option A)

### Architecture
- **Host:** Alibaba Cloud Elastic Compute Service (ECS) Ubuntu 22.04 or 24.04 instance (e.g. `ecs.c7.large` or `ecs.g7.large`).
- **Web Reverse Proxy:** Nginx on port 80/443 proxying `/` to Streamlit (port 8501) and `/api/` & `/docs` to FastAPI (port 8000) with WebSocket support.
- **Service Management:** Systemd service `etd.service` ensuring automatic restarts.
- **Persistence:** Local SQLite (`etd.db`) or Alibaba Cloud ApsaraDB RDS for PostgreSQL via `DATABASE_URL`.

### One-Command Automated Setup
On your Alibaba Cloud ECS instance:
```bash
# 1. Clone the project onto the ECS instance
git clone https://github.com/Hashwar-11/Hackathon-AliCloud.git
cd Hackathon-AliCloud

# 2. Run the automated deployment script
sudo bash deploy/setup_ecs.sh
```

### Docker Compose Deployment (Alternative)
```bash
docker compose build
docker compose up -d
```

---

## 9. Environment Variables

Create a `.env` file in the project root (see `.env.example`):

```bash
# Backend URL
ETD_API_URL=http://localhost:8000

# Persistence (SQLite default, or PostgreSQL)
DATABASE_URL=sqlite:///./etd.db
# DATABASE_URL=postgresql://etd:etd_pass@localhost:5432/etd

# Alibaba Cloud Qwen AI (Optional)
DASHSCOPE_API_KEY=your_dashscope_api_key_here
QWEN_MODEL=qwen-plus
```

---

## 10. Repository File Structure

```
├── .streamlit/
│   └── config.toml                 # Enterprise theme & server configuration
├── config/
│   └── config.yaml                 # Detection thresholds, hyperparams, filepaths
├── data/
│   ├── customer_contexts.json      # 42 real Pakistani operational customer profiles
│   ├── raw/pakistan/               # 42 confirmed DISCO theft cases + synthetic normals
│   └── processed/                  # Frozen train/val/test splits (.npy)
├── deploy/
│   ├── nginx.conf                  # Production reverse proxy configuration
│   └── setup_ecs.sh                # Automated ECS deployment script
├── dashboard/
│   └── app.py                      # Streamlit Enterprise Control Room interface
├── models/checkpoints/             # 9 trained PyTorch & channel model checkpoints
├── src/
│   ├── agents/
│   │   ├── coordinator/            # Routing, risk tiering, queue aggregation
│   │   ├── verification/          # 8-rule FP suppression & loss estimation
│   │   ├── residential/            # CNN + BiLSTM supervised classifier
│   │   └── industrial/             # Convolutional autoencoder anomaly detector
│   ├── api/
│   │   └── app.py                  # FastAPI REST service & Qwen summarizer
│   ├── channels/                   # 4-channel stack implementation
│   ├── db.py                       # SQLAlchemy session & database engine
│   ├── models_db.py                # Inspection & Feedback ORM schemas
│   └── preprocessing/              # Signal transformation & cleaning pipeline
├── tests/                          # 35 unit & integration tests (100% passing)
├── docker-compose.yml              # Containerized multi-service configuration
├── Dockerfile                      # Application container definition
├── run.sh                          # Unified management & execution script
├── requirements.txt                # Pinned production dependencies
└── README.md
```

---

## 11. Team & Project Provenance

- **Alibaba Cloud AI Hackathon (Pakistan)**: Electricity Theft Detection System.
- **Core ML & Channel Boosting Architecture**: Developed by Hashwar, Maheen & Eimaan.
- **Enterprise Engineering & Refactoring**: Built to meet utility-grade operational standards.
