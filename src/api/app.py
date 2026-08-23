"""
FastAPI inference service. Loads all trained/frozen checkpoints once at
startup and exposes POST /predict, matching the contract:

Request:
    { "consumer_id": str, "client_type": "residential"|"industrial",
      "daily_kwh": [float, ...] }

Response:
    { "consumer_id": str, "theft_probability": float,
      "is_theft_suspect": bool, "reasons": [str, ...] }

Run: uvicorn src.api.app:app --host 0.0.0.0 --port 8000
"""
import os
import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.channels.autoencoder_channel import ResidualAutoencoder
from src.channels.pretext_channel import MaskedPretextEncoder
from src.channels.frequency_channel import FrequencyProjection
from src.channels.stack_channels import build_channel_stack
from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder
from src.agents.verification.verify import CustomerContext
from src.agents.coordinator.coordinator import coordinate

CONFIG_PATH = os.environ.get("CONFIG_PATH", "config/config.yaml")
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_DIR = CFG["paths"]["checkpoints_dir"]
SEQ_LEN = 1035  # matches SGCC day count; override if your data differs

app = FastAPI(title="Electricity Theft Detection API")


class PredictRequest(BaseModel):
    consumer_id: str
    client_type: str  # "residential" or "industrial"
    daily_kwh: list[float]


class PredictResponse(BaseModel):
    consumer_id: str
    theft_probability: float
    is_theft_suspect: bool
    reasons: list[str]


def _load_models():
    autoencoder = ResidualAutoencoder(seq_len=SEQ_LEN)
    autoencoder.load_state_dict(torch.load(os.path.join(CKPT_DIR, "autoencoder_channel.pt"), map_location=DEVICE))
    autoencoder.to(DEVICE).eval()

    pretext = MaskedPretextEncoder(seq_len=SEQ_LEN)
    pretext.load_state_dict(torch.load(os.path.join(CKPT_DIR, "pretext_channel.pt"), map_location=DEVICE))
    pretext.to(DEVICE).eval()

    freq_proj = FrequencyProjection(seq_len=SEQ_LEN)
    freq_proj.load_state_dict(torch.load(os.path.join(CKPT_DIR, "frequency_channel.pt"), map_location=DEVICE))
    freq_proj.to(DEVICE).eval()

    ccfg = CFG["channels"]
    num_channels = 1 + sum([ccfg["use_autoencoder_residual"], ccfg["use_pretext_embedding"], ccfg["use_frequency_projection"]])

    mcfg = CFG["residential_model"]
    residential_model = ResidentialModel(
        num_channels=num_channels, seq_len=SEQ_LEN,
        cnn_filters=mcfg["cnn_filters"], cnn_kernel_size=mcfg["cnn_kernel_size"],
        lstm_hidden_units=mcfg["lstm_hidden_units"], dropout=mcfg["dropout"],
    )
    residential_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "residential_model_best.pt"), map_location=DEVICE))
    residential_model.to(DEVICE).eval()

    icfg = CFG["industrial_model"]
    industrial_model = IndustrialAutoencoder(num_channels=num_channels, seq_len=SEQ_LEN,
                                              latent_dim=icfg["autoencoder_latent_dim"])
    industrial_model.load_state_dict(torch.load(os.path.join(CKPT_DIR, "industrial_model.pt"), map_location=DEVICE))
    industrial_model.to(DEVICE).eval()

    with open(os.path.join(CKPT_DIR, "industrial_tau.txt")) as f:
        tau = float(f.read().strip())

    return autoencoder, pretext, freq_proj, residential_model, industrial_model, tau


MODELS = None  # lazy-loaded on first request to keep import-time light for tests


@app.on_event("startup")
def load_all_models():
    global MODELS
    MODELS = _load_models()


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if MODELS is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet.")
    autoencoder, pretext, freq_proj, residential_model, industrial_model, tau = MODELS

    if len(req.daily_kwh) != SEQ_LEN:
        raise HTTPException(status_code=400,
                             detail=f"Expected {SEQ_LEN} daily readings, got {len(req.daily_kwh)}")

    x = torch.tensor([req.daily_kwh], dtype=torch.float32).to(DEVICE)
    ccfg = CFG["channels"]
    stacked = build_channel_stack(x, autoencoder, pretext, freq_proj,
                                   ccfg["use_autoencoder_residual"],
                                   ccfg["use_pretext_embedding"],
                                   ccfg["use_frequency_projection"])

    context = CustomerContext(consumer_id=req.consumer_id)  # wire real billing/audit lookup here

    if req.client_type == "residential":
        with torch.no_grad():
            prob = residential_model(stacked).item()
        result = coordinate(req.consumer_id, "residential", prob, None, None, context,
                             threshold=CFG["scoring"]["theft_probability_threshold"])
    elif req.client_type == "industrial":
        err = industrial_model.reconstruction_error(stacked).item()
        result = coordinate(req.consumer_id, "industrial", None, err, tau, context,
                             threshold=CFG["scoring"]["theft_probability_threshold"])
    else:
        raise HTTPException(status_code=400, detail="client_type must be 'residential' or 'industrial'")

    return PredictResponse(
        consumer_id=result.consumer_id,
        theft_probability=result.theft_probability,
        is_theft_suspect=result.is_theft_suspect,
        reasons=result.reasons,
    )
