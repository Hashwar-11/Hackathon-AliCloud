"""
FastAPI Enterprise Inference & Operations Service.
Provides REST endpoints for single-account prediction, batch feeder analysis,
prioritized field inspection dispatching, and field outcome feedback logging.

Run with:
    uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
"""
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Query, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from src.agents.verification.verify import CustomerContext
from src.agents.coordinator.coordinator import coordinate, coordinate_batch, aggregate_results, ScoringResult
from src.preprocessing.transform_single import transform_daily_series
from sqlalchemy.orm import Session
from src.db import get_db, init_db, utcnow
from src.models_db import Inspection, InspectionFeedback

# torch and the neural channel/agent modules are imported LAZILY (see
# _try_load_models). On a constrained host -- e.g. an 8 GB demo laptop with little
# free RAM -- `import torch` itself can raise MemoryError. The service is designed to
# degrade gracefully to the calibrated statistical fallback engine in that case
# rather than refusing to boot, so the import is guarded here.
try:
    import torch
    TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR = ""
except Exception as _torch_exc:  # ImportError, MemoryError, OSError, ...
    torch = None
    TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = f"{type(_torch_exc).__name__}: {_torch_exc}"

# ---------------------------------------------------------------------------
# Configuration & Global State
# ---------------------------------------------------------------------------
CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CFG = yaml.safe_load(f)
else:
    CFG = {
        "paths": {"checkpoints_dir": "models/checkpoints/"},
        "channels": {"use_autoencoder_residual": True, "use_pretext_embedding": True, "use_frequency_projection": True},
        "residential_model": {"cnn_filters": 64, "cnn_kernel_size": 3, "lstm_hidden_units": 128, "dropout": 0.3},
        "industrial_model": {"autoencoder_latent_dim": 32},
        "scoring": {"theft_probability_threshold": 0.65},
    }

DEVICE = "cuda" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "cpu"
CKPT_DIR = CFG.get("paths", {}).get("checkpoints_dir", "models/checkpoints/")
SEQ_LEN = 1034  # True SGCC day count

MODELS_LOADED = False
AUTOENCODER = None
PRETEXT = None
FREQ_PROJ = None
RESIDENTIAL_MODEL = None
INDUSTRIAL_MODEL = None
INDUSTRIAL_TAU = 0.0334
ACTIVE_CHANNELS = 1
BUILD_CHANNEL_STACK = None  # lazily bound to src.channels.stack_channels.build_channel_stack


def _try_load_models():
    global MODELS_LOADED, AUTOENCODER, PRETEXT, FREQ_PROJ, RESIDENTIAL_MODEL, INDUSTRIAL_MODEL, INDUSTRIAL_TAU, ACTIVE_CHANNELS, BUILD_CHANNEL_STACK
    if not TORCH_AVAILABLE:
        print(f"[API] torch unavailable ({_TORCH_IMPORT_ERROR}). Statistical fallback engine enabled.")
        MODELS_LOADED = False
        return
    try:
        # Lazy neural imports: only reached when torch imported successfully.
        from src.channels.autoencoder_channel import ResidualAutoencoder
        from src.channels.pretext_channel import MaskedPretextEncoder
        from src.channels.frequency_channel import FrequencyProjection
        from src.channels.stack_channels import build_channel_stack
        from src.agents.residential.model import ResidentialModel
        from src.agents.industrial.model import IndustrialAutoencoder
        BUILD_CHANNEL_STACK = build_channel_stack

        ae_path = os.path.join(CKPT_DIR, "autoencoder_channel.pt")
        pt_path = os.path.join(CKPT_DIR, "pretext_channel.pt")
        fq_path = os.path.join(CKPT_DIR, "frequency_channel.pt")
        res_path = os.path.join(CKPT_DIR, "residential_model_best.pt" if os.path.exists(os.path.join(CKPT_DIR, "residential_model_best.pt")) else "residential_stage3.pt")
        ind_path = os.path.join(CKPT_DIR, "industrial_model.pt" if os.path.exists(os.path.join(CKPT_DIR, "industrial_model.pt")) else "industrial_stage3.pt")
        tau_path = os.path.join(CKPT_DIR, "industrial_tau.txt" if os.path.exists(os.path.join(CKPT_DIR, "industrial_tau.txt")) else "industrial_tau_stage3.txt")

        if os.path.exists(res_path) and os.path.exists(ind_path) and os.path.exists(tau_path):
            # Load auxiliary channels if available
            has_ae = os.path.exists(ae_path)
            has_pt = os.path.exists(pt_path)
            has_fq = os.path.exists(fq_path)

            if has_ae:
                AUTOENCODER = ResidualAutoencoder(seq_len=SEQ_LEN)
                AUTOENCODER.load_state_dict(torch.load(ae_path, map_location=DEVICE, weights_only=True))
                AUTOENCODER.to(DEVICE).eval()

            if has_pt:
                PRETEXT = MaskedPretextEncoder(seq_len=SEQ_LEN)
                PRETEXT.load_state_dict(torch.load(pt_path, map_location=DEVICE, weights_only=True))
                PRETEXT.to(DEVICE).eval()

            if has_fq:
                FREQ_PROJ = FrequencyProjection(seq_len=SEQ_LEN)
                FREQ_PROJ.load_state_dict(torch.load(fq_path, map_location=DEVICE, weights_only=True))
                FREQ_PROJ.to(DEVICE).eval()

            # Determine channel count from saved residential model weights
            res_state = torch.load(res_path, map_location=DEVICE, weights_only=True)
            loaded_channels = res_state["conv.weight"].shape[1]
            ACTIVE_CHANNELS = loaded_channels

            mcfg = CFG.get("residential_model", {})
            RESIDENTIAL_MODEL = ResidentialModel(
                num_channels=ACTIVE_CHANNELS,
                seq_len=SEQ_LEN,
                cnn_filters=mcfg.get("cnn_filters", 64),
                cnn_kernel_size=mcfg.get("cnn_kernel_size", 3),
                lstm_hidden_units=mcfg.get("lstm_hidden_units", 128),
                dropout=mcfg.get("dropout", 0.3),
            )
            RESIDENTIAL_MODEL.load_state_dict(res_state)
            RESIDENTIAL_MODEL.to(DEVICE).eval()

            icfg = CFG.get("industrial_model", {})
            INDUSTRIAL_MODEL = IndustrialAutoencoder(
                num_channels=ACTIVE_CHANNELS,
                seq_len=SEQ_LEN,
                latent_dim=icfg.get("autoencoder_latent_dim", 32),
            )
            INDUSTRIAL_MODEL.load_state_dict(torch.load(ind_path, map_location=DEVICE, weights_only=True))
            INDUSTRIAL_MODEL.to(DEVICE).eval()

            with open(tau_path, "r") as f:
                INDUSTRIAL_TAU = float(f.read().strip())

            # Verify that if 4-channel detector is loaded, auxiliary nets are also ready
            if ACTIVE_CHANNELS == 4 and not (has_ae and has_pt and has_fq):
                print(f"[API] Residential model requires 4 channels but auxiliary channel checkpoints are pending. Fallback enabled.")
                MODELS_LOADED = False
            else:
                MODELS_LOADED = True
                print(f"[API] Neural models loaded successfully from {CKPT_DIR} ({ACTIVE_CHANNELS} channels, Device: {DEVICE})")
        else:
            print(f"[API] Checkpoints not complete in {CKPT_DIR}. Running in statistical fallback mode.")
            MODELS_LOADED = False
    except Exception as e:
        print(f"[API] Model loading warning: {e}. Statistical fallback engine enabled.")
        MODELS_LOADED = False


# ---------------------------------------------------------------------------
# Lifespan Context Manager
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()  # ensure the durable inspection tables exist before serving traffic
    _try_load_models()
    yield


app = FastAPI(
    title="Electricity Theft Detection (ETD) & Verification Engine",
    description="Multi-Agent AI & Rule-Based Coordinator for Revenue Protection in Power Distribution",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inspection tickets & feedback are persisted in the database (src/models_db.py)
# so the ground-truth history survives API restarts. A session is injected into
# each request via Depends(get_db) -- there is no module-level mutable state.

# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class CustomerContextSchema(BaseModel):
    feeder_id: str = "FEEDER-MAIN"
    transformer_id: str = "TX-01"
    tariff_category: str = "residential"
    sanctioned_load_kw: float = 7.0            # typical Pakistani residential sanctioned load
    feeder_loss_pct: float = 18.0              # Pakistani DISCO avg loss 15-25%
    recent_audit_result: str = "none"       # "cleared", "confirmed_theft", "meter_fault", "none"
    months_since_last_audit: int = 999
    billing_dispute_open: bool = False
    known_grid_topology_issue: bool = False
    has_rooftop_solar: bool = False
    solar_net_metering_active: bool = False
    is_seasonal_occupancy: bool = False
    tamper_event_count: int = 0
    meter_seal_broken: bool = False
    reverse_current_alert: bool = False
    historical_mean_kwh: float = 15.0
    recent_30d_mean_kwh: float = 5.0
    lowest_window_mean_kwh: float = 0.0   # 0 = auto-computed from the submitted series
    tariff_rate_per_kwh: float = 28.0      # PKR/kWh — typical Pakistani residential tariff


class PredictRequest(BaseModel):
    consumer_id: str
    client_type: Optional[str] = "residential"  # "residential" or "industrial"
    daily_kwh: List[float]
    context: Optional[CustomerContextSchema] = None


class PredictResponse(BaseModel):
    consumer_id: str
    feeder_id: str
    consumer_type: str
    theft_probability: float
    raw_model_score: float
    risk_tier: str
    is_theft_suspect: bool
    action_recommendation: str
    reasons: List[str]
    adjustments: List[Dict[str, Any]]
    financial_impact: Dict[str, Any]


class BatchPredictRequest(BaseModel):
    items: List[PredictRequest]
    threshold: Optional[float] = 0.65


class InspectionActionRequest(BaseModel):
    ticket_id: str
    inspector_id: str
    action_outcome: str  # "CONFIRMED_THEFT", "DEFECTIVE_METER", "SOLAR_CONFIRMED", "VACANCY_CONFIRMED", "FALSE_POSITIVE"
    actual_theft_found: bool
    penalty_imposed_currency: float = 0.0
    notes: str = ""
    consumer_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Scoring Helper Functions
# ---------------------------------------------------------------------------
def _score_single_internal(req: PredictRequest, threshold_override: Optional[float] = None) -> ScoringResult:
    # 1. Transform sequence with identical train pipeline
    scaled_series, raw_mean, inferred_type = transform_daily_series(
        req.daily_kwh, target_seq_len=SEQ_LEN
    )
    client_type = req.client_type or inferred_type

    # 2. Build CustomerContext
    ctx_kwargs = (
        req.context.model_dump()
        if (req.context and hasattr(req.context, "model_dump"))
        else (req.context.dict() if req.context else {})
    )
    if "historical_mean_kwh" not in ctx_kwargs or ctx_kwargs["historical_mean_kwh"] == 15.0:
        ctx_kwargs["historical_mean_kwh"] = raw_mean
    if "recent_30d_mean_kwh" not in ctx_kwargs or ctx_kwargs["recent_30d_mean_kwh"] == 5.0:
        recent_chunk = req.daily_kwh[-30:] if len(req.daily_kwh) >= 30 else req.daily_kwh
        ctx_kwargs["recent_30d_mean_kwh"] = float(np.nanmean(recent_chunk)) if len(recent_chunk) > 0 else 0.0
    # Window-agnostic lowest 30-day rolling mean (for the financial-loss estimator)
    if "lowest_window_mean_kwh" not in ctx_kwargs or ctx_kwargs.get("lowest_window_mean_kwh", 0.0) == 0.0:
        readings = np.array(req.daily_kwh, dtype=np.float32)
        if len(readings) >= 30:
            cumsum = np.cumsum(np.insert(readings, 0, 0.0))
            window_sums = cumsum[30:] - cumsum[:-30]
            window_means = window_sums / 30.0
            ctx_kwargs["lowest_window_mean_kwh"] = float(np.nanmin(window_means))
        else:
            ctx_kwargs["lowest_window_mean_kwh"] = float(np.nanmean(readings)) if len(readings) > 0 else 0.0

    context = CustomerContext(consumer_id=req.consumer_id, **ctx_kwargs)
    threshold = threshold_override if threshold_override is not None else CFG.get("scoring", {}).get("theft_probability_threshold", 0.65)

    # 3. Model Inference or Calibrated Statistical Engine
    if MODELS_LOADED and torch is not None and BUILD_CHANNEL_STACK is not None and RESIDENTIAL_MODEL is not None and INDUSTRIAL_MODEL is not None:
        x_tensor = torch.from_numpy(np.ascontiguousarray(scaled_series, dtype=np.float32)).unsqueeze(0).to(DEVICE)
        
        if ACTIVE_CHANNELS == 4 and AUTOENCODER and PRETEXT and FREQ_PROJ:
            stacked = BUILD_CHANNEL_STACK(
                x_tensor,
                AUTOENCODER,
                PRETEXT,
                FREQ_PROJ,
                use_autoencoder=True,
                use_pretext=True,
                use_frequency=True,
            )
        else:
            stacked = x_tensor.unsqueeze(1)  # (1, 1, seq_len)

        if client_type == "residential":
            with torch.no_grad():
                raw_prob = float(RESIDENTIAL_MODEL(stacked).item())
            return coordinate(req.consumer_id, "residential", residential_probability=raw_prob, context=context, threshold=threshold)
        else:
            err = float(INDUSTRIAL_MODEL.reconstruction_error(stacked).item())
            return coordinate(req.consumer_id, "industrial", industrial_reconstruction_error=err, industrial_tau=INDUSTRIAL_TAU, context=context, threshold=threshold)
    else:
        # Calibrated Statistical & Heuristic Feature Engine (Zero-Drop & Volatility Ratio).
        # There is NO industrial autoencoder in fallback mode, so an industrial account
        # is scored with the same residential heuristic. We must NOT pass
        # consumer_type="industrial" to coordinate() here -- that path requires a
        # reconstruction error + tau and would raise ValueError (HTTP 500). Score via
        # the probability path, then restore the type label with an honesty note.
        drop_ratio = max(0.0, 1.0 - (context.recent_30d_mean_kwh / max(context.historical_mean_kwh, 1e-4)))
        recent_readings = np.array(req.daily_kwh[-60:]) if len(req.daily_kwh) >= 60 else np.array(req.daily_kwh)
        zero_days_pct = float(np.mean(recent_readings <= 0.05)) if len(recent_readings) > 0 else 0.0
        synthetic_raw_score = min(0.98, max(0.02, 0.65 * drop_ratio + 0.35 * zero_days_pct))

        result = coordinate(
            consumer_id=req.consumer_id,
            consumer_type="residential",   # probability path; industrial AE unavailable
            residential_probability=synthetic_raw_score,
            context=context,
            threshold=threshold,
        )
        if client_type != "residential":
            result.consumer_type = client_type
            result.reasons = list(result.reasons) + [
                f"{client_type.capitalize()} account scored by the statistical fallback "
                "heuristic (industrial autoencoder checkpoint unavailable)."
            ]
        return result


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    """Landing route so http://localhost:8000/ does NOT 404 -- it sends you to the
    interactive API docs (Swagger UI) where every endpoint can be tried live."""
    return RedirectResponse(url="/docs")


@app.get("/api/v1/health", tags=["System"])
def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "device": DEVICE,
        "models_loaded": MODELS_LOADED,
        "active_channels": ACTIVE_CHANNELS,
        "sequence_length": SEQ_LEN,
    }


@app.get("/api/v1/models/info", tags=["System"])
def model_info():
    return {
        "architecture": f"Channel-Boosted ({ACTIVE_CHANNELS}-Ch) 1D-CNN + BiLSTM & Industrial Autoencoder",
        "models_loaded": MODELS_LOADED,
        "active_channels": ACTIVE_CHANNELS,
        "industrial_tau": INDUSTRIAL_TAU,
        "sequence_length": SEQ_LEN,
        "active_threshold": CFG.get("scoring", {}).get("theft_probability_threshold", 0.65),
    }


@app.post("/api/v1/predict/single", response_model=PredictResponse, tags=["Inference"])
@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict_single(req: PredictRequest, db: Session = Depends(get_db)):
    """
    Score a single consumer account using the end-to-end multi-agent pipeline.
    """
    if not req.daily_kwh:
        raise HTTPException(status_code=400, detail="daily_kwh cannot be empty.")

    result = _score_single_internal(req)

    # Auto-register a dispatch ticket (persisted) when the account is a suspect.
    if result.is_theft_suspect:
        ticket_id = f"TCK-{req.consumer_id[:8]}-{int(time.time()) % 10000}"
        exists = db.query(Inspection).filter(Inspection.ticket_id == ticket_id).first()
        if exists is None:
            db.add(Inspection(
                ticket_id=ticket_id,
                consumer_id=result.consumer_id,
                feeder_id=result.feeder_id,
                theft_probability=result.verified_theft_probability,
                risk_tier=result.risk_tier,
                action_recommendation=result.action_recommendation,
                estimated_loss_currency=result.financial_impact.get("estimated_monthly_loss_currency", 0.0),
                status="PENDING_DISPATCH",
            ))
            db.commit()

    return PredictResponse(
        consumer_id=result.consumer_id,
        feeder_id=result.feeder_id,
        consumer_type=result.consumer_type,
        theft_probability=result.verified_theft_probability,
        raw_model_score=result.raw_model_score,
        risk_tier=result.risk_tier,
        is_theft_suspect=result.is_theft_suspect,
        action_recommendation=result.action_recommendation,
        reasons=result.reasons,
        adjustments=result.adjustments,
        financial_impact=result.financial_impact,
    )


@app.post("/api/v1/predict/batch", tags=["Inference"])
def predict_batch(req: BatchPredictRequest):
    """
    Batch score multiple consumer accounts across feeders.
    Returns prioritized inspection queue and feeder loss summary.

    Every account is scored by the SAME single-account pipeline
    (_score_single_internal), so industrial accounts are routed through the
    autoencoder and each account keeps its full verification context. The finished
    ScoringResults are aggregated by coordinate.aggregate_results. We deliberately do
    NOT re-coordinate stripped copies: the previous version dropped the industrial
    reconstruction error (500 on any industrial item) and discarded solar/audit/tamper
    context, silently changing every verified probability.
    """
    if not req.items:
        raise HTTPException(status_code=400, detail="Batch items list cannot be empty.")

    results = [_score_single_internal(item, threshold_override=req.threshold) for item in req.items]
    return aggregate_results(results)


@app.get("/api/v1/inspections/queue", tags=["Operations"])
def get_inspection_queue(
    feeder_id: Optional[str] = None,
    risk_tier: Optional[str] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """Retrieve prioritized field inspection queue with optional filters."""
    query = db.query(Inspection)
    if feeder_id:
        query = query.filter(Inspection.feeder_id == feeder_id)
    if risk_tier:
        query = query.filter(Inspection.risk_tier == risk_tier)
    if status_filter:
        query = query.filter(Inspection.status == status_filter)
    tickets = [t.to_dict() for t in query.all()]

    # Sort critical first, then highest probability
    tickets.sort(key=lambda t: (t["risk_tier"] == "CRITICAL", t["theft_probability"]), reverse=True)
    return {"count": len(tickets), "tickets": tickets}


@app.post("/api/v1/inspections/{ticket_id}/action", tags=["Operations"])
def record_inspection_action(ticket_id: str, action: InspectionActionRequest, db: Session = Depends(get_db)):
    """
    Log field inspection audit outcomes. Feeds the continuous learning / ground truth loop.
    Persisted to the database so the ground-truth history survives API restarts.
    """
    rec = db.query(Inspection).filter(Inspection.ticket_id == ticket_id).first()
    if rec is None:
        # Create an ad-hoc ticket if it is not already in the system.
        rec = Inspection(
            ticket_id=ticket_id,
            consumer_id=action.consumer_id or "UNKNOWN",
            feeder_id="FEEDER-MANUAL",
            theft_probability=0.85,
            risk_tier="HIGH",
        )
        db.add(rec)
    elif action.consumer_id:
        rec.consumer_id = action.consumer_id

    rec.status = "COMPLETED"
    rec.action_outcome = action.action_outcome
    rec.actual_theft_found = action.actual_theft_found
    rec.penalty_imposed = action.penalty_imposed_currency
    rec.inspector_id = action.inspector_id
    rec.notes = action.notes
    rec.resolved_at = utcnow()

    db.add(InspectionFeedback(
        ticket_id=ticket_id,
        consumer_id=rec.consumer_id,
        outcome=action.action_outcome,
        actual_theft=action.actual_theft_found,
        penalty=action.penalty_imposed_currency,
        notes=action.notes,
    ))
    db.commit()
    db.refresh(rec)

    return {"message": "Inspection action logged successfully.", "ticket": rec.to_dict()}


@app.get("/api/v1/feeders/summary", tags=["Analytics"])
def get_feeders_summary():
    """Grid feeder overview metrics.

    HONESTY: these feeder figures are SIMULATED demo data (no live Pakistani feeder
    telemetry is wired in yet). The `simulated` flag lets the dashboard label them
    correctly instead of presenting them as measured field data."""
    feeders = {
        "FEEDER-NORTH-01": {"total_meters": 1240, "suspects": 84, "loss_pct": 28.4, "estimated_monthly_loss_pkr": 1420000},
        "FEEDER-IND-04": {"total_meters": 310, "suspects": 19, "loss_pct": 21.0, "estimated_monthly_loss_pkr": 3890000},
        "FEEDER-SOUTH-02": {"total_meters": 2150, "suspects": 42, "loss_pct": 7.8, "estimated_monthly_loss_pkr": 450000},
        "FEEDER-EAST-07": {"total_meters": 980, "suspects": 65, "loss_pct": 24.5, "estimated_monthly_loss_pkr": 980000},
    }
    return {
        "simulated": True,
        "note": "Illustrative feeder cohort - not measured field data.",
        "feeders": feeders,
    }


BENCHMARK_RESULTS_PATH = os.environ.get(
    "BENCHMARK_RESULTS_PATH", "experiments_results/benchmark_results.json"
)


@app.get("/api/v1/benchmark/results", tags=["Analytics"])
def get_benchmark_results():
    """Serve the REAL staged-benchmark metrics written by
    src.experiments.run_all_benchmark, so the dashboard shows measured numbers
    instead of hard-coded ones. Returns an explicit pending payload when the
    benchmark has not been run yet."""
    import json as _json
    if not os.path.exists(BENCHMARK_RESULTS_PATH):
        return {
            "available": False,
            "path": BENCHMARK_RESULTS_PATH,
            "note": ("Benchmark not run yet. Execute: python -m src.experiments.run_all_benchmark "
                     "--config config/config.yaml"),
            "stages": {},
        }
    with open(BENCHMARK_RESULTS_PATH, "r") as f:
        try:
            data = _json.load(f)
        except _json.JSONDecodeError:
            raise HTTPException(status_code=500, detail="benchmark_results.json is malformed.")
    return {"available": True, "path": BENCHMARK_RESULTS_PATH, "stages": data}
