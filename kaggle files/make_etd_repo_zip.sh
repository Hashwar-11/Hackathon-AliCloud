#!/bin/sh
# Builds etd-repo.zip — the upload package for Kaggle / Google Colab.
# Contains code + config + SGCC data + Pakistan files; excludes venv, git,
# caches, and the split archives (the complete copy in data/raw/recovered/
# is what the pipeline actually reads).
#
# Run from anywhere:  sh "kaggle files/make_etd_repo_zip.sh"

cd "$(dirname "$0")/.." || exit 1
OUT="kaggle files/etd-repo.zip"
rm -f "$OUT"

zip -r "$OUT" . \
  -x "venv/*" \
  -x ".git/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x ".pytest_cache/*" \
  -x "data/raw/data.z01" \
  -x "data/raw/data.z02" \
  -x "data/raw/data.zip" \
  -x "research/~\$*" \
  -x "kaggle files/etd-repo.zip" \
  -x "kaggle files/*.ipynb"

echo ""
echo "Created: $OUT"
ls -lh "$OUT"
echo ""
echo "Next:"
echo "  Kaggle -> + New Dataset  -> upload this zip -> name it 'etd-repo'"
echo "  Colab  -> Google Drive   -> upload this zip to the Drive root"
