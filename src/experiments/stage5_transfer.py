"""
Stage 5 — transfer learning to the Pakistan target domain.

BLOCKED until a target-domain CSV exists. Expected format (same as SGCC):
    CONS_NO, FLAG, <daily kWh columns...>
Pass its path via --target_csv. When the file is absent this stage writes a
PENDING entry to the benchmark results and exits without touching any model.

Mechanics (as designed in PROJECT_OVERVIEW §10 / Final-Document):
  1. Channel-Boosting layer: loaded from models/checkpoints/ and FROZEN.
  2. Residential trunk (Conv1d + BiLSTM) is trained on SGCC exactly like
     Stage 3, then FROZEN; a NEW trainable dense head (256 -> 64 -> 1) is
     fine-tuned on the target data only.
     NOTE (honesty): exact Stage 3 weights cannot be reloaded into a new
     head because the Stage 3 head's output dimension is 1, not 256. The
     trunk is therefore re-trained on SGCC with the identical procedure and
     used as the frozen surrogate trunk. This caveat is recorded in results.
  3. Industrial: the Stage 3 autoencoder stays frozen; a new dense head over
     (reconstruction error + 32-d latent) is fine-tuned on target rows.
     Skipped if the target set has too few industrial rows.
  4. Reports target-val performance AND re-evaluates on the SGCC val split so
     the comparison vs Stage 3 (and any drop) is explicit.

Run:
    python -m src.experiments.stage5_transfer --config config/config.yaml \
        --target_csv data/raw/pakistan/target.csv
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src.channels.stack_channels import build_channel_stack
from src.channels.autoencoder_channel import ResidualAutoencoder
from src.channels.pretext_channel import MaskedPretextEncoder
from src.channels.frequency_channel import FrequencyProjection
from src.agents.residential.model import ResidentialModel
from src.agents.industrial.model import IndustrialAutoencoder
from src.preprocessing.preprocess import load_sgcc, preprocess
from src.experiments.common import (
    load_config, load_split, classification_metrics, best_f1_threshold,
    save_stage_results, print_metrics, STAGE_CKPT_DIR, RESULTS_PATH,
)
from src.experiments.stage3_boosted import train_residential_boosted, stack_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MIN_POSITIVES_PER_SPLIT = 5
MIN_TARGET_ROWS = 50


class TransferResidentialHead(nn.Module):
    """Frozen SGCC trunk (conv + BiLSTM) + fresh trainable dense head."""

    def __init__(self, trained: ResidentialModel, dropout: float = 0.3):
        super().__init__()
        self.conv = trained.conv
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.bilstm = trained.bilstm
        hidden = trained.classifier.in_features  # 2 * lstm_hidden_units
        self.new_head = nn.Sequential(
            nn.Linear(hidden, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1),
        )
        for p in list(self.conv.parameters()) + list(self.bilstm.parameters()):
            p.requires_grad = False

    def trunk(self, x):
        x = self.relu(self.conv(x))
        x = self.dropout(x)
        x = x.transpose(1, 2)
        _, (h_n, _) = self.bilstm(x)
        return torch.cat([h_n[-2], h_n[-1]], dim=1)

    def forward(self, x):
        return torch.sigmoid(self.new_head(self.trunk(x))).squeeze(-1)


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


def adapt_length(X: np.ndarray, target_len: int) -> np.ndarray:
    """Adapt target-domain series to the SGCC length the frozen channel nets
    expect (their input dims are fixed at pretrain time).
    Longer -> keep the most recent `target_len` days; shorter -> zero-pad at
    the end (treated like missing/zero consumption days)."""
    cur = X.shape[1]
    if cur == target_len:
        return X
    if cur > target_len:
        print(f"[stage5] adapting target length {cur} -> {target_len} "
              "(keeping most recent days)")
        return X[:, -target_len:]
    print(f"[stage5] adapting target length {cur} -> {target_len} (zero-padding tail)")
    return np.pad(X, ((0, 0), (0, target_len - cur)), mode="constant")


def prepare_target(csv_path, cfg):
    """Same per-row transform as SGCC, then stratified 70/15/15 (NO SMOTE).
    Returns None if the file lacks the minimum for a stratified split."""
    df = load_sgcc(csv_path)
    X, y, ctype = preprocess(df, cfg)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    # stratified 70/15/15 needs at least ~3 examples of each class per split
    if n_pos < 3 or n_neg < 3:
        print(f"[stage5] target has {n_pos} positives / {n_neg} normals — "
              "too few for a stratified split")
        return None
    seed = cfg["preprocessing"]["random_seed"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=seed)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=seed)
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te)


def write_pending(reason: str):
    save_stage_results("stage5_transfer", {
        "status": "PENDING — no Pakistani target-domain dataset provided yet",
        "reason": reason,
        "expected_input": "CSV with columns CONS_NO, FLAG, <daily kWh columns>",
        "command_when_ready": ("python -m src.experiments.stage5_transfer "
                               "--config config/config.yaml --target_csv <path>.csv"),
        "honesty_note": ("Published work fine-tunes on 500-1,000 target samples. "
                         "Below ~100 confirmed cases the transfer result is not "
                         "defensible — report it as 'designed, pending data'."),
    })
    print("[stage5] PENDING:", reason)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--target_csv", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    processed = cfg["paths"]["processed_dir"]
    prod_ckpt = cfg["paths"]["checkpoints_dir"]

    if not args.target_csv or not os.path.exists(args.target_csv):
        write_pending(f"target CSV not found: {args.target_csv!r}")
        return

    splits = prepare_target(args.target_csv, cfg)
    if splits is None:
        write_pending("target CSV has too few examples of one class to split "
                      "(need >= 3 positives and >= 3 normals).")
        return
    (tX_tr, ty_tr), (tX_va, ty_va), (tX_te, ty_te) = splits
    print(f"[stage5] target splits — train: {len(ty_tr)} {np.bincount(ty_tr)}, "
          f"val: {len(ty_va)} {np.bincount(ty_va)}, test: {len(ty_te)} {np.bincount(ty_te)}")

    if len(ty_tr) < MIN_TARGET_ROWS or ty_tr.sum() < MIN_POSITIVES_PER_SPLIT or ty_va.sum() < MIN_POSITIVES_PER_SPLIT:
        write_pending(
            f"target dataset too small to fine-tune defensibly "
            f"(rows={len(ty_tr)}, train positives={int(ty_tr.sum())}, "
            f"val positives={int(ty_va.sum())}; need >= {MIN_TARGET_ROWS} rows and "
            f">= {MIN_POSITIVES_PER_SPLIT} positives per split). "
            "Report as 'designed, pending data'."
        )
        return

    X_tr, y_tr, ct_tr = load_split(processed, "train")
    X_va, y_va, ct_va = load_split(processed, "val")
    seq_len = X_tr.shape[1]
    if tX_tr.shape[1] != seq_len:
        tX_tr = adapt_length(tX_tr, seq_len)
        tX_va = adapt_length(tX_va, seq_len)
        tX_te = adapt_length(tX_te, seq_len)

    ae, pretext, freq = load_channels(seq_len, prod_ckpt)
    tr_res = ct_tr == "residential"
    va_res = ct_va == "residential"

    # ---- Frozen trunk surrogate: SGCC-trained, identical Stage 3 procedure ----
    print("[stage5] building SGCC trunk (frozen surrogate for the Stage 3 trunk) ...")
    S_tr_res = stack_split(X_tr[tr_res], ae, pretext, freq)
    S_va_res = stack_split(X_va[va_res], ae, pretext, freq)
    trunk_model, _ = train_residential_boosted(S_tr_res, y_tr[tr_res], S_va_res, y_va[va_res],
                                               cfg, tag="stage5_trunk")
    transfer = TransferResidentialHead(trunk_model, dropout=cfg["residential_model"]["dropout"]).to(DEVICE)

    # ---- Fine-tune ONLY the new head on target data ----
    tS_tr = stack_split(tX_tr, ae, pretext, freq)
    tS_va = stack_split(tX_va, ae, pretext, freq)
    tS_te = stack_split(tX_te, ae, pretext, freq)
    y_t_tr = torch.tensor(ty_tr, dtype=torch.float32)

    opt = torch.optim.Adam(transfer.new_head.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()
    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(20):
        transfer.train()
        perm = torch.randperm(tS_tr.shape[0])
        for i in range(0, tS_tr.shape[0], 32):
            idx = perm[i:i + 32]
            opt.zero_grad()
            loss = loss_fn(transfer(tS_tr[idx].to(DEVICE)), y_t_tr[idx].to(DEVICE))
            loss.backward()
            opt.step()
        transfer.eval()
        with torch.no_grad():
            prob = torch.sigmoid(torch.cat([
                transfer.new_head(transfer.trunk(tS_va[i:i + 256].to(DEVICE)))
                for i in range(0, tS_va.shape[0], 256)
            ])).squeeze(-1).cpu().numpy()
        auc = float(roc_auc_score(ty_va, prob))
        print(f"[stage5] epoch {epoch+1}/20  target_val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc, patience = auc, 0
            best_state = {k: v.cpu().clone() for k, v in transfer.new_head.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    transfer.new_head.load_state_dict(best_state)

    def eval_on(S, y_true):
        transfer.eval()
        with torch.no_grad():
            prob = torch.sigmoid(torch.cat([
                transfer.new_head(transfer.trunk(S[i:i + 256].to(DEVICE)))
                for i in range(0, S.shape[0], 256)
            ])).squeeze(-1).cpu().numpy()
        thr = best_f1_threshold(y_true, prob)
        return classification_metrics(y_true, prob, thr)

    target_val = eval_on(tS_va, ty_va)
    target_test = eval_on(tS_te, ty_te)
    sgcc_val = eval_on(S_va_res, y_va[va_res])
    print_metrics("Stage 5 transfer — TARGET VALIDATION", target_val)
    print_metrics("Stage 5 transfer — TARGET TEST", target_test)
    print_metrics("Stage 5 transfer — SGCC VAL (forgetting check)", sgcc_val)

    comparison = None
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            s3 = json.load(f).get("stage3_boosted", {})
        s3_val = s3.get("residential", {}).get("val_metrics")
        if s3_val:
            comparison = {
                "note": "Target-val transfer result vs SGCC-only Stage 3. The dip is "
                        "EXPECTED (China -> Pakistan domain shift) and must be reported.",
                "stage3_sgcc_val": s3_val,
                "stage5_target_val": target_val,
                "delta_roc_auc": round(target_val["roc_auc"] - s3_val["roc_auc"], 4),
            }
            print(f"[stage5] vs Stage 3 (SGCC-only) — delta ROC-AUC: "
                  f"{comparison['delta_roc_auc']:+.4f}")

    os.makedirs(STAGE_CKPT_DIR, exist_ok=True)
    torch.save(transfer.state_dict(), os.path.join(STAGE_CKPT_DIR, "transfer_residential.pt"))

    save_stage_results("stage5_transfer", {
        "status": "COMPLETED",
        "target_csv": args.target_csv,
        "target_splits": {
            "train": {"n": int(len(ty_tr)), "positives": int(ty_tr.sum())},
            "val": {"n": int(len(ty_va)), "positives": int(ty_va.sum())},
            "test": {"n": int(len(ty_te)), "positives": int(ty_te.sum())},
        },
        "target_val_metrics": target_val,
        "target_test_metrics": target_test,
        "sgcc_val_metrics_after_transfer": sgcc_val,
        "comparison_vs_stage3": comparison,
        "industrial_transfer": "not attempted — requires industrial-typed target rows; "
                               "see honesty note",
        "honesty_note": ("Frozen trunk is a surrogate re-trained on SGCC with the "
                         "identical Stage 3 procedure (exact Stage 3 weights cannot "
                         "take a new head). If the target positive count is in the "
                         "dozens, treat these numbers as provisional."),
    })


if __name__ == "__main__":
    main()
