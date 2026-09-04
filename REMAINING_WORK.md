# Remaining Work — Electricity Theft Detection Project

_Last updated: 2026-09-04_

This document lists everything that is **designed but not yet implemented**, along with
concrete steps to complete each item. Nothing here is blocked by code — only by external
data, infrastructure, or decisions.

---

## 1. Pakistan Fine-Tuned Transfer (Stage 5)

### Current State
- **Zero-shot transfer evaluated:** 64.3% recall @0.65 on 42 confirmed theft cases
- **Fine-tune:** PENDING — blocked on both-class target data
- **Code:** `src/training/finetune_pakistan.py` exists and works; aborts on single-class input (by design)

### What Is Needed

| Requirement | Minimum | Preferred | Why |
|---|---|---|---|
| Confirmed theft cases | 50 | 100+ | Enough to fine-tune the dense head without overfitting |
| Normal customers | 200 | 500+ | Both classes required for AUC/precision; BCE needs `pos_weight > 0` |
| Data format | SGCC-compatible | Smart-meter long format | `prepare_pakistan_target.py` auto-detects both layouts |
| Source | DISCO inspection records, FIR reports, DSU findings | — | Must be field-confirmed, not synthetic |

### How to Get the Data

1. **Identify a DISCO contact:**
   - Target: LESCO, IESCO, GECO, or any of the 10 successor entities of WAPDA
   - Ask for: "Non-technical loss records" or "theft inspection reports" from a pilot area
   - Specific documents: FIR-registered theft cases, DSU high-loss-feeder findings,
     meter replacement records (tampered meters)

2. **Request template:**
   ```
   Subject: Request for Theft Inspection Data — Research Collaboration

   Dear [DISCO Name] Team,

   We are conducting research on AI-based electricity theft detection for Pakistani
   distribution companies. We request access to:

   1. Confirmed theft cases (FIR-registered or inspection-confirmed) from [pilot area]
   2. Corresponding normal customer records (same feeder/transformer)
   3. Time period: [e.g., 2023-01 to 2025-12]
   4. Format: Consumer ID, daily/monthly kWh readings, theft flag, feeder ID

   This is for academic research under the Alibaba Cloud Hackathon. Results will be
   shared with the DISCO for operational use.

   Contact: [Your name, email, phone]
   ```

3. **Data formatting:**
   - Once you get the data, format it as `pakistan_sgcc_format.xlsx`:
     - Columns: `CONS_NO, FLAG, <daily kWh columns>`
     - `FLAG = -1` for theft, `FLAG = 0` for normal
     - Daily readings in wide format (one column per day)
   - Run: `python -m src.experiments.prepare_pakistan_target --input pakistan_sgcc_format.xlsx`

### How to Run the Fine-Tune

```bash
# 1. Ensure you have both-class data
python -m src.experiments.prepare_pakistan_target --input pakistan_sgcc_format.xlsx
# Should print: "theft: N, normal: M" where both N > 0 and M > 0

# 2. Run the fine-tune (GPU recommended)
python -m src.training.finetune_pakistan --config config/config.yaml

# 3. Evaluate on target val/test
python -m src.experiments.stage5_transfer --config config/config.yaml \
    --target_csv data/raw/pakistan/pakistan_target.csv

# 4. Update benchmark_results.json with the new Stage 5 numbers
```

### Expected Outcome
- Fine-tuned model that improves recall on Pakistani data beyond the 64.3% zero-shot baseline
- Report both SGCC-only baseline and fine-tuned results (expect some dip on SGCC due to domain shift)
- Label clearly: "fine-tuned on N confirmed Pakistani theft cases"

---

## 2. Alibaba Cloud Deployment

### Current State
- **Local deployment works:** FastAPI + Streamlit + SQLite/PostgreSQL + Docker
- **Cloud architecture designed:** SLS → MaxCompute → PAI-EAS → Function Compute
- **Code:** Not yet adapted for cloud services

### Architecture (As Designed)

```
┌─────────────────────────────────────────────────────────────────┐
│                        Alibaba Cloud                             │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌────────┐ │
│  │   SLS    │───▶│  MaxCompute  │───▶│  PAI-EAS │───▶│   FC   │ │
│  │ (logs)   │    │ (batch score)│    │ (serving)│    │(alert) │ │
│  └──────────┘    └──────────────┘    └──────────┘    └────────┘ │
│       │                │                    │               │    │
│       ▼                ▼                    ▼               ▼    │
│  Meter data      OSS / TableStore      Model weights    DingTalk │
│  from AMI        (results)             (OSS bucket)     SMS/Email│
└─────────────────────────────────────────────────────────────────┘
```

### How to Deploy

#### Step 1: Set Up Alibaba Cloud Account
1. Go to [alibabacloud.com](https://www.alibabacloud.com) and create an account
2. Enable these services:
   - **OSS** (Object Storage Service) — for model weights and data
   - **PAI** (Platform for AI) — for model serving (PAI-EAS)
   - **MaxCompute** — for batch processing
   - **SLS** (Log Service) — for log ingestion
   - **Function Compute (FC)** — for alerting
   - **DingTalk** (optional) — for notifications

#### Step 2: Containerize the Application
The `Dockerfile` already exists. Build and push to Alibaba Cloud Container Registry:

```bash
# Login to ACR
docker login --username=<your-username> registry.ap-southeast-1.aliyuncs.com

# Build
docker build -t etd-api:latest .

# Tag and push
docker tag etd-api:latest registry.ap-southeast-1.aliyuncs.com/<namespace>/etd-api:latest
docker push registry.ap-southeast-1.aliyuncs.com/<namespace>/etd-api:latest
```

#### Step 3: Deploy Model Serving (PAI-EAS)
1. Go to PAI console → EAS (Elastic Algorithm Service)
2. Create a service:
   - Image: your ACR image
   - Resource: 2 vCPU, 8 GB RAM (minimum for 4-channel model)
   - Port: 8000
   - Environment variables:
     - `CONFIG_PATH=/app/config/config.yaml`
     - `DATABASE_URL=postgresql://user:pass@host:5432/etd`
3. Deploy and get the endpoint URL

#### Step 4: Set Up Batch Processing (MaxCompute)
1. Create a MaxCompute project
2. Upload the scoring script (`src/experiments/run_all_benchmark.py` adapted for batch)
3. Schedule daily/weekly batch scoring of all accounts
4. Output results to OSS or TableStore

#### Step 5: Set Up Alerting (Function Compute)
1. Create a function that triggers on new high-risk scores
2. Send alerts via DingTalk webhook or SMS:
   ```python
   import requests
   def handler(event, context):
       # Parse the event (new high-risk score)
       consumer_id = event['consumer_id']
       probability = event['theft_probability']
       # Send DingTalk alert
       webhook_url = "https://oapi.dingtalk.com/robot/send?access_token=..."
       requests.post(webhook_url, json={
           "msgtype": "text",
           "text": {"content": f"High-risk theft alert: {consumer_id} (prob={probability:.2f})"}
       })
   ```

#### Step 6: Connect Dashboard
- Update `dashboard/app.py` to point `ETD_API_URL` at the PAI-EAS endpoint
- Deploy dashboard to Alibaba Cloud ECS or use Streamlit Cloud

### Estimated Cost (Monthly)
| Service | Spec | Cost (USD) |
|---|---|---|
| PAI-EAS | 2 vCPU, 8 GB | ~$50-100 |
| MaxCompute | Pay-as-you-go | ~$10-30 |
| OSS | 10 GB | ~$1 |
| SLS | 1 GB/day | ~$5 |
| Function Compute | 1M invocations | ~$1 |
| **Total** | | **~$70-140/month** |

### Alternative: Simpler Deployment
If Alibaba Cloud setup is too complex for the hackathon timeline:
- Deploy on a single **ECS instance** (4 vCPU, 16 GB) with Docker Compose
- Cost: ~$80/month
- No batch processing; just real-time API + dashboard

---

## 3. Real Feeder Telemetry Integration

### Current State
- **Dashboard shows simulated feeder data** (labelled as such in-app)
- **Verification Agent rules** are built but don't fire without real context
- **API accepts** `CustomerContext` with feeder/audit/solar fields

### What Is Needed

| Data Source | What It Provides | How to Get It |
|---|---|---|
| AMI (Advanced Metering Infrastructure) | Daily/hourly kWh per meter | DISCO IT department |
| Feeder SCADA | Feeder-level loss %, load | DISCO grid operations |
| Billing system | Tariff category, disputes, sanctioned load | DISCO commercial department |
| Inspection records | Audit results, confirmed theft, meter faults | DISCO revenue protection team |
| Net-metering registry | Rooftop solar installations | DISCO distributed generation team |

### How to Integrate

#### Option A: CSV/Excel Import (Simplest)
1. Get data exports from DISCO in CSV/Excel format
2. Write a loader script:
   ```python
   # scripts/load_disco_data.py
   import pandas as pd
   from src.db import SessionLocal, init_db
   from src.models_db import CustomerContext as DBContext

   def load_customer_contexts(csv_path):
       df = pd.read_csv(csv_path)
       db = SessionLocal()
       for _, row in df.iterrows():
           ctx = DBContext(
               consumer_id=row['consumer_id'],
               feeder_id=row['feeder_id'],
               tariff_category=row['tariff'],
               feeder_loss_pct=row['feeder_loss_pct'],
               recent_audit_result=row['last_audit'],
               # ... etc
           )
           db.merge(ctx)
       db.commit()
   ```
3. Run the loader to populate the database
4. The API will automatically use the stored context when scoring

#### Option B: API Integration (Production)
1. DISCO exposes a REST API for customer data
2. Create a sync service that pulls context nightly:
   ```python
   # scripts/sync_customer_context.py
   import requests
   from src.db import SessionLocal
   from src.models_db import CustomerContext

   DISCO_API_URL = "https://disco-api.example.com/customers"

   def sync_all():
       response = requests.get(DISCO_API_URL)
       customers = response.json()
       db = SessionLocal()
       for cust in customers:
           ctx = CustomerContext(
               consumer_id=cust['id'],
               feeder_id=cust['feeder'],
               # ... map DISCO fields to our schema
           )
           db.merge(ctx)
       db.commit()
   ```
3. Schedule via cron: `0 2 * * * /path/to/venv/bin/python scripts/sync_customer_context.py`

#### Option C: Demo Mode (Hackathon)
If real data is not available before the demo:
1. Generate realistic synthetic context for the 42 Pakistani cases:
   ```python
   # scripts/generate_demo_context.py
   import numpy as np
   import pandas as pd

   np.random.seed(42)
   n = 42
   contexts = pd.DataFrame({
       'consumer_id': [f'PK{i:03d}' for i in range(n)],
       'feeder_id': np.random.choice(['FEEDER-NORTH-01', 'FEEDER-SOUTH-02'], n),
       'feeder_loss_pct': np.random.uniform(15, 30, n),
       'tariff_category': 'residential',
       'historical_mean_kwh': np.random.uniform(10, 50, n),
       'recent_30d_mean_kwh': np.random.uniform(2, 15, n),
       'recent_audit_result': np.random.choice(['none', 'cleared', 'confirmed_theft'], n, p=[0.7, 0.2, 0.1]),
   })
   contexts.to_csv('data/raw/pakistan/demo_contexts.csv', index=False)
   ```
2. Load into the database before the demo
3. The Verification Agent rules will now fire and show in the dashboard

### Verification Agent Activation Checklist

The Verification Agent currently "never fires" because `CustomerContext` is built with
defaults. To activate it:

- [ ] Populate `feeder_loss_pct` (triggers Rules 2)
- [ ] Populate `recent_audit_result` + `months_since_last_audit` (triggers Rule 1)
- [ ] Populate `known_grid_topology_issue` (triggers Rule 3)
- [ ] Populate `has_rooftop_solar` (triggers Rule 4)
- [ ] Populate `is_seasonal_occupancy` (triggers Rule 5)
- [ ] Populate `tamper_event_count` + `meter_seal_broken` (triggers Rule 6)
- [ ] Populate `billing_dispute_open` (triggers Rule 7)
- [ ] Populate `lowest_window_mean_kwh` (triggers Rule 8 — auto-computed by API)

---

## 4. Additional Improvements (Nice-to-Have)

### 4.1 Industrial Detector Reliability
- **Current:** Threshold at 50 kWh/day selects ~649 accounts (improved from 37)
- **Remaining:** Validate with actual consumer-type labels if available
- **How:** If DISCO provides tariff category data, replace the magnitude heuristic with
  real labels and re-train the industrial autoencoder

### 4.2 Pretext Channel Degeneracy
- **Current:** Pretext channel collapses to a scalar broadcast across timesteps
- **Remaining:** Fix `pretext_channel.py:35` to preserve the 32-dimensional embedding
- **How:** Remove the `.mean(dim=1, keepdim=True).expand(-1, seq_len)` and instead use
  a learned projection or upsampling to maintain temporal variation

### 4.3 Chronological Column Ordering
- **Current:** Fixed — day columns are sorted chronologically
- **Remaining:** Verify the fix is applied everywhere (preprocess, API, transfer scripts)
- **How:** Grep for `sorted_day_columns` and ensure all entry points use it

### 4.4 Imputation Improvement
- **Current:** Zero-fill for leading NaN runs (mimics theft signature)
- **Remaining:** Mask unobserved days instead of zero-filling
- **How:** Add a binary mask channel (1=observed, 0=imputed) as a 5th input channel;
  the model learns to ignore imputed regions

### 4.5 Threshold Justification
- **Current:** `theft_probability_threshold = 0.65` (documented but not empirically justified)
- **Remaining:** Justify via cost-benefit analysis
- **How:** Plot precision-recall curve; choose threshold where marginal cost of false
  positive (customer annoyance) equals marginal benefit of true positive (theft recovered)

---

## 5. Priority Order

If time is limited, tackle in this order:

| Priority | Item | Effort | Impact |
|---|---|---|---|
| **1** | Pakistan data collection | High (external) | Unblocks fine-tuned Stage 5 |
| **2** | Demo context generation | Low (1 hour) | Makes Verification Agent visible in demo |
| **3** | Cloud deployment (ECS simple) | Medium (1 day) | Live demo URL for judges |
| **4** | Pretext channel fix | Low (30 min) | Marginal AUC improvement |
| **5** | Imputation masking | Medium (2 hours) | Reduces false positives on sparse accounts |
| **6** | Full cloud architecture | High (1 week) | Production-ready but not needed for hackathon |

---

## 6. Quick Wins (Can Be Done Today)

### Generate Demo Contexts for Pakistani Cases
```bash
# Create realistic operational context for the 42 confirmed theft cases
python scripts/generate_demo_context.py

# Load into database
python scripts/load_disco_data.py --input data/raw/pakistan/demo_contexts.csv

# Now the API will show Verification Agent rules firing
curl -X POST http://localhost:8000/api/v1/predict/single \
  -H "Content-Type: application/json" \
  -d '{
    "consumer_id": "PK001",
    "client_type": "residential",
    "daily_kwh": [10.0, 10.5, ...]
  }'
# Response will include reasons like "Account sits on high-loss feeder (28.4% ...)"
```

### Deploy to ECS (Simple Cloud Path)
```bash
# On Alibaba Cloud ECS (Ubuntu 22.04, 4 vCPU, 16 GB)
git clone <repo-url>
cd Hackathon-AliCloud
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Start API
nohup uvicorn src.api.app:app --host 0.0.0.0 --port 8000 &

# Start dashboard
nohup streamlit run dashboard/app.py --server.port 8501 &

# Get public IP from ECS console → share with judges
```

---

## 7. Honesty Reminders

When completing any of the above, remember:

1. **Never present synthetic data as real.** If you generate demo contexts, label them "simulated."
2. **Never present zero-shot as fine-tuned.** The 64.3% recall is zero-shot; say so.
3. **Never hide weak industrial numbers.** If the industrial detector underperforms, report it.
4. **Never claim cloud deployment if it's just local Docker.** Be explicit about what's live.
5. **Every metric must be reproducible.** If you can't re-run it, don't report it.

---

_End of remaining work document. Update this file as items are completed._
