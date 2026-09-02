"""
Runs the full staged benchmark in strict order and prints the consolidated
table. Stages never run out of order: if any stage fails, everything after
it is skipped.

    python -m src.experiments.run_all_benchmark --config config/config.yaml
    # ... and once the Pakistani dataset exists:
    python -m src.experiments.run_all_benchmark --config config/config.yaml \
        --target_csv data/raw/pakistan/target.csv

Order: preprocess -> Stage 1 -> Stage 2 -> Stage 3 -> Stage 4 -> Stage 5.
"""
import argparse
import json
import os
import sys
import traceback

from src.experiments.common import RESULTS_PATH


def run_preprocess(config_path):
    from src.preprocessing import preprocess as pp
    sys.argv = ["preprocess", "--config", config_path]
    pp.main()


def run_stage(module_name, config_path, target_csv=None):
    import importlib
    mod = importlib.import_module(f"src.experiments.{module_name}")
    argv = [module_name, "--config", config_path]
    if target_csv is not None:
        argv += ["--target_csv", target_csv]
    sys.argv = argv
    mod.main()


def fmt(m, key="roc_auc"):
    if not m:
        return "—"
    v = m.get(key)
    return "—" if v is None else f"{v:.4f}"


def print_consolidated_table():
    if not os.path.exists(RESULTS_PATH):
        print("No results file found — nothing ran.")
        return
    with open(RESULTS_PATH) as f:
        r = json.load(f)

    print("\n" + "=" * 100)
    print("CONSOLIDATED BENCHMARK — residential detector, VALIDATION split "
          "(real-world imbalance, never SMOTE'd)")
    print("=" * 100)
    header = f"{'Stage':<42} {'ROC-AUC':>9} {'F1':>8} {'Prec':>8} {'Recall':>8} {'Acc':>8}"
    print(header)
    print("-" * len(header))

    rows = []
    if "stage1_xgboost" in r:
        s = r["stage1_xgboost"]
        rows.append(("1. XGBoost tabular baseline", s.get("val_metrics")))
        rows.append(("1. XGBoost — TEST (held out)", s.get("test_metrics")))
    if "stage2_raw" in r:
        s = r["stage2_raw"]
        rows.append(("2. CNN+BiLSTM raw sequence (no boost)", s.get("residential", {}).get("val_metrics")))
        rows.append(("2. CNN+BiLSTM — TEST", s.get("residential", {}).get("test_metrics")))
    if "stage3_boosted" in r:
        s = r["stage3_boosted"]
        rows.append(("3. CNN+BiLSTM + Channel Boosting", s.get("residential", {}).get("val_metrics")))
        rows.append(("3. Boosted — TEST", s.get("residential", {}).get("test_metrics")))
    if "stage5_transfer" in r and r["stage5_transfer"].get("status") == "COMPLETED":
        s = r["stage5_transfer"]
        rows.append(("5. Transfer — TARGET validation", s.get("target_val_metrics")))
        rows.append(("5. Transfer — SGCC val (forgetting)", s.get("sgcc_val_metrics_after_transfer")))

    for name, m in rows:
        print(f"{name:<42} {fmt(m, 'roc_auc'):>9} {fmt(m, 'f1'):>8} "
              f"{fmt(m, 'precision'):>8} {fmt(m, 'recall'):>8} {fmt(m, 'accuracy'):>8}")

    # Industrial detector lines (AUC only when sample allows; always flagged)
    print("-" * len(header))
    for stage_key, label in [("stage2_raw", "2. Industrial AE (raw)"),
                             ("stage3_boosted", "3. Industrial AE (boosted)")]:
        if stage_key in r:
            ind = r[stage_key].get("industrial", {})
            vm = ind.get("val_metrics")
            status = ind.get("status", "?")
            if status == "SKIPPED":
                print(f"{label:<42} SKIPPED — {ind.get('reason', 'too few rows')}")
            else:
                print(f"{label:<42} {fmt(vm, 'roc_auc'):>9} "
                      f"(tau={ind.get('tau', '—')}, "
                      f"train normals={ind.get('n_train_normals', '?')}, "
                      f"val normals={ind.get('n_val_normals', '?')}/"
                      f"theft={ind.get('n_val_theft', '?')})")

    # Stage 4 + Stage 5 status
    print("-" * len(header))
    if "stage4_integration" in r:
        s4 = r["stage4_integration"]
        print("4. Integration: " + s4.get("industrial_reliability", "?"))
        for split, u in s4.get("unified_scores", {}).items():
            print(f"     {split}: residential AUC={u.get('residential_auc')}  "
                  f"industrial AUC={u.get('industrial_auc')}  "
                  f"unified AUC={u.get('unified_auc_both_types')}")
    if "stage5_transfer" in r:
        s5 = r["stage5_transfer"]
        print(f"5. Transfer: {s5.get('status')}")
        if s5.get("comparison_vs_stage3"):
            print(f"     delta ROC-AUC vs SGCC-only Stage 3: "
                  f"{s5['comparison_vs_stage3']['delta_roc_auc']:+.4f}")
    print("=" * 100)

    # Honesty notes — weak/inconclusive results must be visible here
    print("\nHONESTY NOTES")
    notes = []
    if "stage2_raw" in r and r["stage2_raw"].get("industrial", {}).get("status") == "SKIPPED":
        notes.append("Industrial detector (Stage 2): " + r["stage2_raw"]["industrial"].get("reason", ""))
    if "stage3_boosted" in r:
        ind = r["stage3_boosted"].get("industrial", {})
        if ind.get("status") == "SKIPPED":
            notes.append("Industrial detector (Stage 3): " + ind.get("reason", ""))
        elif ind.get("n_train_normals") is not None:
            notes.append(f"Industrial detector trained on only {ind['n_train_normals']} normal "
                         "accounts (proxy label @500 kWh/day). Its metrics are EXPLORATORY, "
                         "not evidence.")
        cmp23 = r["stage3_boosted"].get("comparison_vs_stage2")
        if cmp23 and cmp23.get("delta_roc_auc") is not None and cmp23["delta_roc_auc"] <= 0:
            notes.append(f"Channel Boosting did NOT improve ROC-AUC over Stage 2 "
                         f"(delta={cmp23['delta_roc_auc']:+.4f}). Report it anyway.")
    if "stage4_integration" in r:
        notes.append("Stage 4: " + r["stage4_integration"].get("industrial_reliability", ""))
        notes.append("Industrial score min(err/tau, 1) is a normalized ratio, NOT a probability.")
    if "stage5_transfer" in r:
        s5 = r["stage5_transfer"]
        if s5.get("status") != "COMPLETED":
            notes.append("Stage 5 pending: " + s5.get("reason", ""))
        notes.append("Consumer-type split is a magnitude heuristic — SGCC has no type column. "
                     "Zero-fill imputation can mimic the theft signature on 38% of accounts. "
                     "All numbers measured on this data; never cite published SGCC figures.")
    for i, note in enumerate(notes, 1):
        print(f"  {i}. {note}")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--target_csv", default=None,
                        help="Pakistan target-domain CSV for Stage 5 (optional)")
    parser.add_argument("--from_stage", type=int, default=0,
                        help="resume from a stage (1-5); earlier results must exist")
    args = parser.parse_args()

    stages = [
        ("preprocess", lambda: run_preprocess(args.config)),
        ("stage1_xgboost", lambda: run_stage("stage1_xgboost", args.config)),
        ("stage2_raw", lambda: run_stage("stage2_raw", args.config)),
        ("stage3_boosted", lambda: run_stage("stage3_boosted", args.config)),
        ("stage4_report", lambda: run_stage("stage4_report", args.config)),
        ("stage5_transfer", lambda: run_stage("stage5_transfer", args.config, args.target_csv)),
    ]
    start = max(0, args.from_stage)  # from_stage=1 skips preprocess, etc.
    for i, (name, fn) in enumerate(stages):
        if i < start:
            print(f"\n>>> skipping {name} (--from_stage)")
            continue
        print(f"\n{'#' * 72}\n>>> RUNNING {name}\n{'#' * 72}")
        try:
            fn()
        except Exception:
            print(f"\n!!! {name} FAILED — stopping. No later stage will run.")
            traceback.print_exc()
            print_consolidated_table()
            sys.exit(1)

    print_consolidated_table()


if __name__ == "__main__":
    main()
