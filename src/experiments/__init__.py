"""Staged benchmark experiments (Stage 1-5) for the SGCC theft pipeline.

These scripts run the sequential model comparison:
    stage1_xgboost    — tabular-feature baseline
    stage2_raw        — raw-sequence deep detectors (no Channel Boosting)
    stage3_boosted    — same detectors with Channel Boosting
    stage4_report     — comparable scores + industrial sample-size report
    stage5_transfer   — Pakistan target-domain fine-tuning (needs target CSV)

Every stage appends its results to experiments_results/benchmark_results.json
so run_all_benchmark.py can print the consolidated table at the end.
"""
