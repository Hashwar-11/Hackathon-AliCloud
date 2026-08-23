#!/usr/bin/env bash
# Downloads the SGCC electricity theft dataset (the only dataset here with real
# theft labels). PRECON is optional and has NO theft labels — see README.
set -euo pipefail

RAW_DIR="data/raw"
mkdir -p "$RAW_DIR"

echo "== SGCC dataset =="
echo "Source: https://github.com/henryRDlab/ElectricityTheftDetection"
echo "This repo requires you to accept the author's terms before downloading."
echo "Manual steps (cannot be scripted due to their access terms):"
echo "  1. Visit the GitHub page above."
echo "  2. Download data.zip, data.z01, data.z02 into $RAW_DIR/"
echo "  3. Run: cd $RAW_DIR && zip -s 0 data.zip --out combined.zip && unzip combined.zip"
echo "  4. Confirm you now have $RAW_DIR/sgcc_data.csv (rename if the extracted file differs)"
echo ""
echo "Citation (required if you publish results):"
echo "Zheng, Z., Yang, Y., Niu, X., Dai, H.N., Zhou, Y. 'Wide and Deep Convolutional"
echo "Neural Networks for Electricity-Theft Detection to Secure Smart Grids.'"
echo "IEEE Trans. Industrial Informatics, 14(4), 2018."
echo ""
echo "== PRECON dataset (optional, NO theft labels) =="
echo "Source: https://opendata.com.pk -- search 'PRECON Pakistan Residential"
echo "Electricity Consumption Dataset'"
echo "Download the per-house CSVs into $RAW_DIR/precon/"
echo "Use this ONLY for: (a) unsupervised normal-baseline training, or"
echo "(b) synthetic theft injection (Jokar et al. 2015 attack functions)."
echo "If you use (b), your report must say the theft examples are synthetic."
