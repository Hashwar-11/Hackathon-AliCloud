"""
Stage 4 — system integration: comparable scores from both detectors.

This stage does NOT train anything new. It loads the Stage 3 (Channel-Boosted)
checkpoints and converts both detectors' outputs into consistent [0, 1]
scores routed by consumer type, so a separately-built verification/coordinator
layer can merge them:

  - Residential: sigmoid probability (already a probability by construction).
  - Industrial : min(recon_error / tau, 1.0) — a NORMALIZED anomaly score.
    Documented caveat (PROJECT_OVERVIEW §9): this ratio is comparable in scale
    but is NOT a probability, and tau is the 95th percentile of normal error,
    so the mapping is miscalibrated by construction.

Also reports the industrial sample-size cross-tabulation and an explicit
reliability flag, per the benchmark requirements.

Run (after stage3_boosted):
    python -m src.experiments.stage4_report --config config/config.yaml
"""
import argparse
import json
import os

import numpy as np
import torch

from src.channels.autoencoder_channel import ResidualAutoencoder
from src.channels.pretext_channel import MaskedPretextEncoder
from src.channels.frequency_channel import FrequencyProjection
from src.channels.stack_channels import build_channel_stack
from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder
from src.experiments.common import (
    load_config, load_split, classification_metrics, count_by_type,
    save_stage_results, print_metrics, STAGE_CKPT_DIR, RESULTS_PATH,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Below this many normal industrial training rows, the industrial detector's
# numbers are reported but flagged unreliable (tau from a handful of points).
INDUSTRIAL_RELIABILITY_MIN = 30


def load_channels(seq_len, ckpt_dir):
    nets = []
    for cls, fname in [(ResidualAutoencoder, "autoencoder_channel.pt"),
                       (MaskedPretextEncoder, "pretext_channel.pt"),
                       (FrequencyProjection, "frequency_channel.pt")]:
        net = cls(seq_len=seq_len)
        net.load_state_dict(torch.load(os.path.join(ckpt_dir, fname),
                                       map_location=DEVICE, weights_only=True))
        net.to(DEVICE).eval()
        for p in net.parameters():
            p.requires_grad = False
        nets.append(net)
    return nets


def stack(X_np, nets, chunk=4096):
    parts = []
    for i in range(0, len(X_np), chunk):
        xb = torch.tensor(X_np[i:i + chunk], dtype=torch.float32).to(DEVICE)
        parts.append(build_channel_stack(xb, *nets).cpu())
    return torch.cat(parts, dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    prod_ckpt = cfg["paths"]["checkpoints_dir"]

    X_va, y_va, ct_va = load_split(processed, "val")
    X_te, y_te, ct_te = load_split(processed, "test")
    seq_len = X_va.shape[1]
    print(f"[stage4] device: {DEVICE}")

    # ---- Industrial sample-size report (requirement of Stage 4) ----
    counts = {"val": count_by_type(ct_va, y_va, "val"),
              "test": count_by_type(ct_te, y_te, "test")}
    tau_path = os.path.join(prod_ckpt, "industrial_tau_stage3.txt")
    ind_ckpt = os.path.join(prod_ckpt, "industrial_stage3.pt")
    industrial_available = os.path.exists(ind_ckpt) and os.path.exists(tau_path)

    # Normal industrial TRAIN rows, as recorded by Stage 3's own results.
    n_ind_train_normals = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            n_ind_train_normals = (json.load(f).get("stage3_boosted", {})
                                   .get("industrial", {}).get("n_train_normals"))

    if not industrial_available:
        reliability = "SKIPPED (too few normal industrial rows to train)"
    elif n_ind_train_normals is not None and n_ind_train_normals < INDUSTRIAL_RELIABILITY_MIN:
        reliability = (f"UNRELIABLE — only {n_ind_train_normals} normal industrial "
                       f"training rows (< {INDUSTRIAL_RELIABILITY_MIN}); tau is derived "
                       "from a handful of accounts. Treat industrial numbers as exploratory.")
    else:
        reliability = "RELIABLE"

    unified = {}
    for split_name, X, y, ct in [("val", X_va, y_va, ct_va), ("test", X_te, y_te, ct_te)]:
        ae, pretext, freq = load_channels(seq_len, prod_ckpt)

        # Residential head
        # Residential head — use residential_model_best.pt (the actual Stage 3 model;
        # residential_stage3.pt was overwritten by finetune_pakistan.py)
        res_ckpt_name = "residential_model_best.pt"
        res_ckpt_path = os.path.join(prod_ckpt, res_ckpt_name)
        if not os.path.exists(res_ckpt_path):
            res_ckpt_path = os.path.join(prod_ckpt, "residential_stage3.pt")
        mcfg = cfg["residential_model"]
        res_model = ResidentialModel(num_channels=4, seq_len=seq_len,
                                     cnn_filters=mcfg["cnn_filters"],
                                     cnn_kernel_size=mcfg["cnn_kernel_size"],
                                     lstm_hidden_units=mcfg["lstm_hidden_units"],
                                     dropout=mcfg["dropout"]).to(DEVICE)
        res_model.load_state_dict(torch.load(
            res_ckpt_path,
            map_location=DEVICE, weights_only=True))
        res_model.eval()

        scores = np.zeros(len(y))
        res_mask = ct == "residential"

        S_res = stack(X[res_mask], (ae, pretext, freq))
        probs = []
        with torch.no_grad():
            for i in range(0, S_res.shape[0], 512):
                probs.append(res_model(S_res[i:i + 512].to(DEVICE)).cpu().numpy())
        scores[res_mask] = np.concatenate(probs)

        ind_metrics = None
        if industrial_available:
            ind_mask = ct == "industrial"
            if ind_mask.sum() > 0:
                icfg = cfg["industrial_model"]
                ind_model = IndustrialAutoencoder(num_channels=4, seq_len=seq_len,
                                                  latent_dim=icfg["autoencoder_latent_dim"]).to(DEVICE)
                ind_model.load_state_dict(torch.load(ind_ckpt, map_location=DEVICE,
                                                     weights_only=True))
                ind_model.eval()
                tau = float(np.loadtxt(tau_path))
                S_ind = stack(X[ind_mask], (ae, pretext, freq))
                err = ind_model.reconstruction_error(S_ind.to(DEVICE)).cpu().numpy()
                scores[ind_mask] = np.minimum(err / tau, 1.0)  # normalized anomaly score
                if len(np.unique(y[ind_mask])) > 1:
                    ind_metrics = classification_metrics(y[ind_mask], err, threshold=None)

        res_metrics = classification_metrics(y[res_mask], scores[res_mask], threshold=None)
        all_metrics = classification_metrics(y, scores, threshold=None)
        unified[split_name] = {
            "residential_auc": res_metrics["roc_auc"],
            "industrial_auc": ind_metrics["roc_auc"] if ind_metrics else None,
            "unified_auc_both_types": all_metrics["roc_auc"],
            "n_residential": int(res_mask.sum()),
            "n_industrial": int((~res_mask).sum()),
        }
        print_metrics(f"Stage 4 unified scores — {split_name} (residential subset)",
                      classification_metrics(y[res_mask], scores[res_mask], 0.5))
        print(f"[stage4] {split_name}: residential AUC={res_metrics['roc_auc']}, "
              f"industrial AUC={ind_metrics['roc_auc'] if ind_metrics else 'n/a'}, "
              f"unified AUC={all_metrics['roc_auc']}")

    print(f"[stage4] INDUSTRIAL RELIABILITY FLAG: {reliability}")

    save_stage_results("stage4_integration", {
        "scoring_convention": {
            "residential": "sigmoid probability from the Stage 3 classifier",
            "industrial": "min(reconstruction_error / tau, 1.0) — normalized anomaly "
                          "score, NOT a probability (see PROJECT_OVERVIEW §9)",
        },
        "industrial_sample_counts": counts,
        "industrial_train_normals": n_ind_train_normals,
        "industrial_reliability": reliability,
        "unified_scores": unified,
        "note": "Verification/coordinator layer is intentionally NOT implemented here "
                "(built separately); these scores are its inputs.",
    })


if __name__ == "__main__":
    main()
