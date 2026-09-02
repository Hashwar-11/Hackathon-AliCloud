# Cloud Training — Kaggle & Colab

Local machine is CPU-only, so the full benchmark (Stages 2/3/5) is trained in
the cloud on a free GPU. The local laptop only runs **inference/demo** afterwards.

## Files in this folder

| File | Purpose |
|---|---|
| `make_etd_repo_zip.sh` | Builds `etd-repo.zip` (code + config + data) for upload |
| `kaggle_train.ipynb` | **Primary** — full benchmark on Kaggle GPU |
| `colab_train.ipynb` | **Backup** — same run on Colab if Kaggle fails |

## Step 1 — build the upload zip (local)

```sh
sh "kaggle files/make_etd_repo_zip.sh"
```

Produces `kaggle files/etd-repo.zip`.

## Step 2A — Kaggle (primary)

1. kaggle.com → **+ New → New Dataset** → upload `etd-repo.zip` → name it `etd-repo` → Create.
2. **+ New → New Notebook** → attach the `etd-repo` dataset.
3. Notebook Settings: **Accelerator = GPU T4 x2**, **Internet = On**.
4. Upload `kaggle_train.ipynb` (File → Open notebook → upload) or paste its cells.
5. Run All. Expect ~20–40 min total on GPU.
6. Output panel → download `results_and_models.zip`.

## Step 2B — Colab (only if Kaggle fails)

1. Upload `etd-repo.zip` to your Google Drive root.
2. Open `colab_train.ipynb` in Colab (Upload notebook).
3. Runtime → Change runtime type → **T4 GPU**.
4. Run All → `results_and_models.zip` downloads automatically at the end.

⚠️ Run the pipeline on **ONE platform only** — two runs waste quota and give
conflicting weights.

## Step 3 — bring results home (local)

```sh
cd electricity-theft-detection
unzip -o ~/Downloads/results_and_models.zip
```

This merges into the project:
- `models/checkpoints/` — production weights the API/dashboard loads
  (`autoencoder_channel.pt`, `pretext_channel.pt`, `frequency_channel.pt`,
  `residential_model_best.pt`, `industrial_model.pt`)
- `models/stage_checkpoints/` — per-stage models incl. `transfer_residential.pt`
- `experiments_results/benchmark_results.json` — the final consolidated table

## If a run dies mid-way

Each stage persists its result before the next starts. Re-run the copy cell,
then add `--from_stage N` to the benchmark cell:

| index | stage |
|---|---|
| 0 | preprocess |
| 1 | stage1_xgboost |
| 2 | stage2_raw |
| 3 | stage3_boosted |
| 4 | stage4_report |
| 5 | stage5_transfer |

## For the hackathon submission

`models/` is gitignored — explicitly include `models/` +
`experiments_results/` in the submission zip so judges get the trained weights.
