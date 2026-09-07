# Pakistani Target-Domain Dataset Status

_Last updated: 2026-09-04_

## Current State

| Metric | Value |
|---|---|
| Confirmed theft cases (FIR/DISCO) | **0** |
| Confirmed normal cases | **0** |
| Synthetic theft cases (generated) | **42** (in `pakistan_sgcc_format.xlsx`) |
| Target for fine-tune | **100+ confirmed theft + matching normals** |

## What exists on disk

| File | Contents | Usable for fine-tune? |
|---|---|---|
| `pakistan_sgcc_format.xlsx` | 42 rows × 365 daily readings, all `FLAG = -1` (theft, confirmed by data owner) | **No** — single-class (42 theft / 0 normal). AUC/precision undefined; BCE `pos_weight = 0`. |
| `pakistan_theft_cases.xlsx` | Empty template (schema only, 0 rows) | **No** — placeholder. |
| `data/raw/pakistan/pakistan_target.csv` | 42 theft rows (converted from xlsx by `prepare_pakistan_target.py`) | **No** — same single-class limitation. |

## What has been measured

- **Zero-shot transfer (Stage 5):** The frozen SGCC Stage 3 detector scores the 42 confirmed Pakistani theft cases without any fine-tuning.
  - Detection recall @0.65 (production threshold): **64.3% (27/42)**
  - Detection recall @0.50: **78.6% (33/42)**
  - Reported as recall only (no normals → precision/AUC undefined).
  - Series front-padded 365→1034 to match the production `transform_daily_series`.

## What is needed for fine-tuned Stage 5

1. **≥ 50 confirmed theft cases** from a DISCO (FIR records, inspection reports, DSU findings).
2. **Matching normal customers** (same feeder/region, same time period) — at least 200+ preferred.
3. **Both classes required** — a single-class fine-tune is degenerate and `finetune_pakistan.py` now aborts to protect the good SGCC weights.

## Label convention

- `FLAG = -1` = **theft** (confirmed by data owner; `prepare_pakistan_target.py` maps `-1 → 1`).
- `FLAG = 0` = normal.
- `FLAG = 1` = normal (in SGCC convention).

## Honesty note

The 42 synthetic theft rows are **generated** scenarios, not field-confirmed cases. Any result using them is labeled "synthetic theft injection — exploratory" and is **never** presented as validated transfer performance. The zero-shot result on confirmed theft cases is the only honest cross-domain measurement.

## Data collection contacts

- **Target DISCO:** Not yet identified.
- **Pilot area:** Not yet selected.
- **DSU high-loss-feeder findings:** Not yet requested.

_Status: Data collection not yet started. Zero-shot transfer evaluated as a proof of concept._
