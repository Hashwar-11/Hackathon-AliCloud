# 🏆 Final Round 12-Slide Presentation Deck
**Electricity Theft Detection (ETD) — Enterprise Grid Intelligence Platform**  
*Alibaba Cloud AI Hackathon (Pakistan) — Final Round | Team PIEAS (Muhammad Hashwar Haider, Maheen, Eimaan)*

---

## SLIDE 1: Title & Executive Overview
**Slide Title:**  
**Electricity Theft Detection (ETD) — Enterprise Grid Intelligence Platform**  
*Channel-Boosted Deep Learning & Multi-Agent Verification for Non-Technical Loss Reduction*

**On-Slide Content:**
* **Event:** Alibaba Cloud AI Hackathon (Pakistan) — **Grand Finale**
* **Team:** Muhammad Hashwar Haider, Maheen & Eimaan
* **Institution:** Pakistan Institute of Engineering & Applied Sciences (PIEAS), Islamabad
* **Live Control Room URL:** `https://etd-platform-pk.web.app`
* **Live K-Electric / DISCO Sandbox URL:** `https://etd-disco-sandbox-pk.web.app`
* **GitHub Repository:** `https://github.com/Hashwar-11/Hackathon-AliCloud`

> **Speaker Note (20s):** *"Assalam-o-Alaikum judges. We are Team PIEAS, and today we present the ETD Enterprise Grid Intelligence Platform—an end-to-end AI and multi-agent verification system built on Alibaba Cloud to recover billions in stolen electricity across Pakistani DISCOs and K-Electric."*

---

## SLIDE 2: The Problem — Pakistan’s ₨ 200+ Billion Grid Crisis
**Slide Title:**  
**The Problem: Smart Meters Collect Data, But Have Zero Brain to Catch Theft**

**On-Slide Content:**
* **₨ 200+ Billion Annual Deficit:** NEPRA reports 15%–35% AT&C losses across Pakistani utilities (IESCO, LESCO, MEPCO, FESCO, K-Electric) driven by Non-Technical Losses (NTL):
  * Direct phase tapping (*kunda*) & partial meter shunt bypasses
  * Night-only AC timer hooks & neutral-earth return manipulation
  * 3-Phase Industrial CT/PT secondary shorting & DC magnetic tampering
* **Dismal 5%–10% Inspection Hit Rate:** Over 90% of manual utility raids inspect intact meters, wasting millions in transport and enforcement labor.
* **Why Existing Smart Meter Databases Fail:**
  * Utilities use **PITC UDIL** and **K-Electric SAP IS-U / Itron MDM** only to multiply `kWh × Tariff` for monthly bills.
  * Simple SQL drop rules (*"flag >50% bill drop"*) cause **90% False Alarms** because **Rooftop Solar Net-Metering**, seasonal vacations, and transformer faults look identical to meter bypasses in raw kWh data.

> **Speaker Note (30s):** *"Why do utilities still lose 200 billion rupees even after installing smart meters? Because their databases only store numbers for billing. If they flag everyone whose consumption drops by 50%, 9 out of 10 alerts are innocent citizens who installed rooftop solar. Field teams stop trusting the alerts and go back to blind raids."*

---

## SLIDE 3: Our Solution — ETD Enterprise Grid Intelligence Platform
**Slide Title:**  
**Our Solution: 4-Channel Deep Learning + Multi-Agent Domain Arbitration**

**On-Slide Content:**
* **1. Multi-Perspective 4-Channel Deep Learning:** Replaces single-channel raw kWh models with a 4-Channel tensor (Raw Telemetry + Autoencoder Residual + Self-Supervised Pretext + FFT Frequency Spectrum).
* **2. Dual Residential & Industrial Neural Routing:**
  * **Residential (`A-1/A-2`):** Supervised **1D-CNN + BiLSTM** sequence classifier.
  * **Industrial (`B-2/B-3`):** Unsupervised **Convolutional Autoencoder** with Tau-centering ($\tau = 0.0334$).
* **3. 8-Rule Verification Agent (The False-Alarm Killer):** Cross-checks neural predictions against `DLMS/COSEM OBIS` solar export registers, M&S audit history, and hardware tamper logs before dispatching a raid.
* **4. Window-Agnostic PKR Loss Engine & Alibaba Cloud Qwen AI:** Prioritizes field queues by monthly **PKR revenue loss** and generates natural-language raid briefings via **Qwen-Plus**.

> **Speaker Note (25s):** *"Our platform sits directly on top of a utility's existing database. It combines a 4-channel deep learning detector with an 8-rule verification agent that eliminates solar false alarms, calculates exact PKR losses, and writes field briefings using Alibaba Cloud Qwen."*

---

## SLIDE 4: Datasets & Cross-Domain Training Strategy
**Slide Title:**  
**Datasets Used: From 42,372 Benchmark Meters to Pakistani Grid Telemetry**

**On-Slide Content:**
* **1. Primary Training Dataset — State Grid Corporation of China (SGCC):**
  * **Scale:** **42,372 smart meters × 1,034 daily readings** (~2.8 years of daily kWh).
  * **Class Distribution:** `91.3%` Normal (`38,757`) vs. `8.7%` Confirmed Theft (`3,615`).
  * **Strict Frozen Split:** Stratified **70% Train (`29,660`) / 15% Val (`6,356`) / 15% Test (`6,356`)** — zero data leakage.
* **2. Pakistani DISCO Target Dataset (`pakistan_target.csv`):**
  * **42 Pakistani DISCO theft profiles** (365-day sequences aligned to 1,034 steps) + **300 normal baseline profiles** used to evaluate Zero-Shot Transfer (`Stage 5a`) and Fine-Tuning (`Stage 5b`).
* **3. K-Electric SAP IS-U & PITC UDIL Sandbox Cohort:**
  * **120 K-Electric Accounts × 107 Columns** (`0400...` 13-digit Contract Accounts, `LA/LB/AL/BL` Consumer Numbers, 90-day daily load curves, and `OBIS` registers) across 12 Karachi IBCs.

> **Speaker Note (25s):** *"We trained our backbone on 42,372 smart meters over 1,034 days with a frozen 70/15/15 split, evaluated zero-shot and fine-tuned transfer on 42 Pakistani theft profiles, and tested full database compatibility on a 120-account, 107-column K-Electric and PITC sandbox cohort."*

---

## SLIDE 5: Telemetry Preprocessing & Sequence Alignment Pipeline
**Slide Title:**  
**Step 1 of Pipeline: Length-Agnostic Telemetry Cleaning (`src/preprocessing/`)**

**On-Slide Content:**
* **Real Smart Meters Are Noisy:** GPRS/4G packet drops create `NaN` gaps, and voltage surges create extreme outliers.
* **Our 4-Step Mathematical Preprocessing (`transform_daily_series`):**
  1. **Neighbor-Average Imputation:** Linearly interpolates missing `NaN` polling intervals from adjacent days.
  2. **Outlier Clipping ($\mu + 2\sigma$):** Caps extreme spikes at each consumer's historical mean plus two standard deviations:
     $$x_{\text{clipped}}(t) = \min\big(x(t),\; \mu_x + 2\sigma_x\big)$$
  3. **Per-Consumer Min-Max Normalization $[0, 1]$:** Forces the neural network to learn *behavioral shape changes* rather than absolute house size.
  4. **Dynamic Sequence Alignment (`1,034` steps):** Automatically front-pads/aligns 90-day or 365-day Pakistani DISCO exports to the `1,034`-step tensor in **< 2ms**.

> **Speaker Note (20s):** *"Whether a Pakistani utility gives us 90 days, 365 days, or 3 years of meter readings, our preprocessing engine imputes missing cellular packets, clips outliers at mu plus 2 sigma, and aligns the sequence to 1,034 timesteps automatically."*

---

## SLIDE 6: Core AI Innovation — The 4-Channel Boosting Stack
**Slide Title:**  
**Step 2 of Pipeline: 4-Channel Signal Boosting (`src/channels/`)**

**On-Slide Content:**
* **Why 1-Channel Deep Learning Fails:** A raw 1D kWh curve hides subtle bypasses and periodic night-time hooks.
* **Our $(4 \times 1,034)$ Multi-View Tensor Construction:**
  * **Channel 1 — Scaled Raw Sequence ($C_1$):** Normalized daily kWh consumption trajectory.
  * **Channel 2 — Autoencoder Reconstruction Residual ($C_2$):** Trained *exclusively on normal consumers*. Measures how far the consumer deviates from normal grid behavior:
    $$C_2(t) = \big|\, C_1(t) - \hat{C}_1^{\text{normal}}(t) \,\big|$$
  * **Channel 3 — Masked-Pretext Self-Supervised Embedding ($C_3$):** Trained by randomly masking **20% of days** and reconstructing them from surrounding context—capturing seasonal & weekly regularity.
  * **Channel 4 — FFT Spectral Projection ($C_4$):** Computes real Fast Fourier Transform (`torch.fft.rfft`) magnitudes to expose **periodic night-only AC hooks and timer bypasses**.

> **Speaker Note (30s):** *"This is our core ML innovation. Instead of feeding raw kWh alone into a neural network, we construct 4 channels: Raw kWh, Autoencoder reconstruction error from normal behavior, a 20%-masked self-supervised embedding, and an FFT frequency spectrum that immediately catches thieves who hook their ACs every night at 11 PM."*

---

## SLIDE 7: Dual Deep Learning Models & Coordinator Routing
**Slide Title:**  
**Step 3 of Pipeline: Residential CNN-BiLSTM & Industrial Autoencoder**

**On-Slide Content:**
* **Automated Consumer Routing (`Coordinator Agent`):**
  * **Residential / Commercial ($\mu < 50\text{ kWh/d}$ or Tariff `A-1/A-2`):**
    * **Architecture:** `1D-CNN` (64 filters, kernel=3, BatchNorm, ReLU, MaxPool) $\rightarrow$ `2-Layer BiLSTM` (128 hidden units, dropout=0.3) $\rightarrow$ Linear + Sigmoid probability $p \in [0, 1]$.
    * **Training:** Weighted Binary Cross-Entropy (`pos_weight = 10.5`) to counter 1:10 class imbalance.
  * **Industrial ($\mu \ge 50\text{ kWh/d}$ or Tariff `B-2/B-3`):**
    * **Architecture:** 4-Channel **Convolutional Autoencoder** (`latent_dim = 32`).
    * **Tau-Centering Sigmoid Calibration ($\tau = 0.0334$):** Converts unbounded Reconstruction MSE $e$ into a calibrated theft probability aligned with the `0.65` decision threshold:
      $$p_{\text{ind}}(e) = \frac{1}{1 + \exp\big(-k(e/\tau - 1) - \ln(0.65/0.35)\big)}$$

> **Speaker Note (25s):** *"Residential and industrial theft look completely different. Our Coordinator routes residential meters to a 1D-CNN + BiLSTM classifier, and routes industrial factories to an unsupervised Convolutional Autoencoder calibrated via Tau-centering so both models share a unified 0.65 threshold."*

---

## SLIDE 8: Multi-Agent Verification & PKR Financial Loss Engine
**Slide Title:**  
**Step 4 of Pipeline: 8-Rule Verification Agent & PKR Recovery Engine**

**On-Slide Content:**
* **The 8 Utility Domain Rules (`src/agents/verification/verify.py`):**
  1. **Cleared Audit Decay ($\times 0.60$):** Suppresses accounts physically cleared by vigilance within `< 6 months`.
  2. **Prior Confirmed Theft Floor ($\ge 0.95$):** Escalates repeat offenders immediately.
  3. **Rooftop Solar Net-Metering Filter ($\times 0.25$ / $-0.45$):** Reads `OBIS 1.0.2.8.0.255` (Active Export kWh) to **eliminate solar false alarms**.
  4. **Grid CT/PT Fault Filter ($\times 0.40$):** Suppresses known transformer/substation topology faults.
  5. **High-Loss Feeder / PMT Boost ($+0.05$):** Boosts accounts on feeders with $>20\%$ AT&C loss.
  6. **Hardware Tamper Override (Floor $\ge 0.95$):** Escalates `OBIS 0.0.99.98.0.255` Cover Open, DC Magnet, or CT Reversal alerts to **CRITICAL**.
  7. **Billing Dispute Flag:** Annotates open SAP/CRM disputes before dispatch.
  8. **Sustained 30-Day Drop Boost ($+0.08$):** Boosts severe $>60\%$ sustained consumption drops.
* **Window-Agnostic PKR Loss Estimator:** Scans all rolling 30-day windows to compute exact monthly unbilled kWh $\times$ NEPRA Tariff (`₨ 28–38/kWh`).

> **Speaker Note (25s):** *"After the neural network scores a meter, our Verification Agent applies 8 transparent utility rules. If OBIS 1.0.2.8.0 shows solar export, the score is suppressed by 75%. If OBIS 0.0.99.98 shows a broken terminal seal, it escalates to 95%+ Critical Risk and calculates the exact monthly loss in PKR."*

---

## SLIDE 9: Full System Architecture & Alibaba Cloud Integration
**Slide Title:**  
**Production Architecture on Alibaba Cloud (ECS + Qwen-Plus AI)**

**On-Slide Content:**
* **Alibaba Cloud Elastic Compute Service (ECS):**
  * Containerized **Docker Compose** & automated **Ubuntu ECS deployment (`deploy/setup_ecs.sh`)** with **Nginx** reverse proxy and `systemd` auto-recovery (`etd.service`).
* **Alibaba Cloud Model Studio (`Qwen-Plus` via DashScope API):**
  * Endpoint: `POST /api/v1/assistant/summarize`
  * Converts mathematical channel residuals, OBIS event codes, and PKR loss figures into a **3-Paragraph Executive Raid Briefing** for SDOs and vigilance officers (with deterministic offline fallback).
* **Full-Stack Enterprise Persistence:**
  * **Backend:** FastAPI ASGI (`12 REST Endpoints`, `<15ms` inference latency, `35/35 Pytest Suite Passing`).
  * **Database:** SQLAlchemy 2.0 ORM (`SQLite` / `ApsaraDB RDS PostgreSQL`) storing durable `inspections` tickets and `inspection_feedback` ground-truth loops.

> **Speaker Note (20s):** *"Our entire backend runs on FastAPI and SQLAlchemy, packaged for one-click deployment on Alibaba Cloud ECS, and integrates Alibaba Cloud Qwen-Plus via DashScope to generate instant executive raid briefings for field engineers."*

---

## SLIDE 10: Empirical Benchmark Results (Stages 1 to 5)
**Slide Title:**  
**Empirical Benchmark: +6.2% ROC-AUC & +7.0% F1 Gain Over Raw Deep Learning**

**On-Slide Table (100% Measured on Frozen Splits):**

| Stage | Architecture Evaluated | Val ROC-AUC | Test ROC-AUC | Val F1 | Test Acc | Primary Contribution |
|---|---|---|---|---|---|---|
| **Stage 1** | **XGBoost Tabular Baseline** (20 Features) | `0.6824` | `0.6779` | `0.2699` | `80.7%` | Summary stats miss temporal onset |
| **Stage 2** | **1-Channel Deep Learning** (Raw CNN+BiLSTM) | `0.6835` | `0.6603` | `0.2456` | `69.6%` | Raw 1D sequence struggles with noise |
| **Stage 3** | **4-Channel Boosted CNN+BiLSTM (Ours)** | **`0.7453`** | **`0.7529`** | **`0.3160`** | **`85.9%`** | **⭐ +6.2% AUC & +7.0% F1 Boost** |
| **Stage 4** | **4-Ch DL + 8-Rule Verification Agent** | `0.7453` | `0.7529` | `0.3160` | `85.9%` | Eliminates Solar & Audit False Positives |
| **Stage 5a** | **Pakistan Zero-Shot Transfer (42 Cases)** | — | **64.3% Recall** | — | — | Zero local training @ `0.65` threshold (`78.6%` @ `0.50`) |
| **Stage 5b** | **Pakistan Fine-Tuned (42 Theft + 300 Normal)** | — | **`1.0000`** | **`1.000`** | **`100.0%`** | Held-out test split (67 accounts) |

> **Speaker Note (25s):** *"Every metric here is empirically measured. Adding our 4-channel stack jumps Test ROC-AUC from 0.6603 to 0.7529 and F1 by +7%. Even more impressive: when tested zero-shot on 42 Pakistani theft cases without seeing local data, it caught 64.3% at the strict 0.65 threshold, and hit 1.000 AUC after fine-tuning."*

---

## SLIDE 11: K-Electric SAP IS-U & PITC UDIL Sandbox Testing
**Slide Title:**  
**Real-World Compatibility: K-Electric & PITC Sandbox Verification**  
*Live Demo: `https://etd-disco-sandbox-pk.web.app`*

**On-Slide Content:**
* **Why We Built the Sandbox:** To prove that a Pakistani utility can upload its raw database export directly into our platform with **zero ERP modifications**.
* **Unaltered K-Electric Raw CSV (`120 Accounts × 107 Columns`):**
  * **13-Digit Contract Account (`VKONT`):** `0400102945319` ... `0400103816280`
  * **8-Char KE Consumer No:** `LA`, `LB`, `AL`, `BL` zone prefixes (`LB310483`, `AL310566`)
  * **12 Karachi IBCs & PMTs:** `IBC-JOHAR`, `IBC-DHA`, `IBC-CLIFTON`, `IBC-KORANGI`, `IBC-SITE`, `IBC-LYARI`
  * **DLMS/COSEM OBIS Columns + 90-Day Load Profile (`DAY_01_KWH`..`DAY_90_KWH`):**
    * `OBIS 1.0.1.8.0` (Import kWh) | `OBIS 1.0.2.8.0` (Solar Export kWh) | `OBIS 0.0.99.98.0` (Tamper Alarms)
* **120-Account Ground-Truth Scorecard Results:**
  * **Raw 4-Ch DL Alone:** **66.7% Precision** (Caught 40/40 thieves, but **falsely flagged all 20 DHA/Clifton Solar homes** at `83.2%` risk).
  * **4-Ch DL + 8-Rule Verification (Our System):** **100.0% Precision | 100.0% Recall | 100.0% Accuracy**
    * ✅ **40 / 40 True Positives Caught** (20 Shunt/Magnet Tampers + 20 Night AC Hooks).
    * ✅ **80 / 80 Clean Accounts Verified** (**20 / 20 Solar False Alarms Suppressed** from `83.2%` $\rightarrow$ `37.4%`).

> **Speaker Note (35s):** *"To prove Day-1 compatibility with K-Electric and PITC, we built a live Sandbox where we load a raw 120-account by 107-column K-Electric CSV with 0400 Contract Accounts, LA/LB Consumer Numbers, and 90-day OBIS registers. Without our Verification Agent, raw deep learning falsely flags all 20 DHA/Clifton solar homes. With our Verification Agent turned on, it reads the OBIS 1.0.2.8.0 Solar Export column, suppresses all 20 false alarms, and achieves 100% precision and recall across all 120 accounts."*

---

## SLIDE 12: Operational Impact, Deployment Roadmap & Live Demo
**Slide Title:**  
**Impact & Roadmap: Transforming Pakistani Grid Revenue Protection**

**On-Slide Content:**
* **Quantified Utility Impact:**
  * Increases field inspection hit rate from **5–10% (random raids)** to **75%+ (targeted AI dispatch)**.
  * Eliminates customer harassment on **Net-Metering Solar** and **Cleared Audit** accounts.
  * Even a **5% reduction** in Pakistan's NTL recovers **₨ 10+ Billion annually**.
* **Industry Outreach Initiated (PIEAS Official Channel):**
  * Formal sandbox pilot proposals transmitted to **PITC, IESCO, LESCO, MEPCO, FESCO, and K-Electric** (`bscs2453@pieas.edu.pk`).
* **35 / 35 Automated Pytest Suite Passing & Dual Live Deployments:**
  * 🖥️ **Main Enterprise Control Room:** `https://etd-platform-pk.web.app`
  * 🧪 **K-Electric & DISCO Sandbox Lab:** `https://etd-disco-sandbox-pk.web.app`
  * 📦 **Open-Source Repo & Alibaba ECS Scripts:** `https://github.com/Hashwar-11/Hackathon-AliCloud`

> **Speaker Note (20s):** *"Even a 5% reduction in theft recovers over 10 billion rupees annually for Pakistan's power sector. Both our Main Control Room and our K-Electric Sandbox Lab are live right now—let us show you a 60-second live demonstration!"*
