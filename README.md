# Electricity Theft Detection — AI Pipeline

Channel-Boosted deep learning + multi-agent scoring, trained on the real SGCC
theft-labeled dataset. No manual field inspection in the loop.

## Directory structure

```
electricity-theft-detection/
├── data/
│   ├── raw/                     # untouched downloaded CSVs go here
│   └── processed/               # output of preprocessing, gitignored
├── config/
│   └── config.yaml              # every path, threshold, hyperparameter lives here
├── scripts/
│   └── download_data.sh         # pulls SGCC (+ optional PRECON) into data/raw
├── src/
│   ├── preprocessing/
│   │   └── preprocess.py        # impute, 3-sigma cap, min-max scale, SMOTE, train/val/test split
│   ├── channels/
│   │   ├── autoencoder_channel.py   # Channel 2: reconstruction-residual (pretrained on normals)
│   │   ├── pretext_channel.py       # Channel 3: masked-timestep self-supervised embedding
│   │   ├── frequency_channel.py     # Channel 4: learned FFT projection
│   │   └── stack_channels.py        # combines channels 1-4 into one tensor per customer
│   ├── agents/
│   │   ├── residential/
│   │   │   └── model.py         # Channel-Boosted 1D-CNN + BiLSTM classifier
│   │   ├── industrial/
│   │   │   └── model.py         # Channel-Boosted Autoencoder (reconstruction-error scoring)
│   │   ├── verification/
│   │   │   └── verify.py        # rule-based cross-check against billing/audit history
│   │   └── coordinator/
│   │       └── coordinator.py   # fuses agent outputs into final theft_probability
│   ├── training/
│   │   ├── train_channels.py    # pretrains the 3 auxiliary channel networks
│   │   ├── train_residential.py # trains the residential classifier
│   │   └── train_industrial.py  # trains the industrial autoencoder + derives tau
│   └── api/
│       └── app.py               # FastAPI service, POST /predict
├── dashboard/
│   └── app.py                   # Streamlit ops dashboard (flags, tickets)
├── models/checkpoints/          # saved .pt weights land here, gitignored
├── tests/
│   └── test_preprocessing.py
├── notebooks/                   # exploratory analysis only, nothing production runs from here
├── requirements.txt
└── Dockerfile
```

## Exact run order (do not skip steps or reorder)

```bash
# 0. environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. get the data
bash scripts/download_data.sh                # downloads SGCC into data/raw/

# 2. preprocess — this is where impute/cap/scale/SMOTE/split all happen
python -m src.preprocessing.preprocess --config config/config.yaml

# 3. pretrain the auxiliary channel networks on the training split ONLY
python -m src.training.train_channels --config config/config.yaml

# 4. train the residential classifier (channels frozen from step 3)
python -m src.training.train_residential --config config/config.yaml

# 5. train the industrial autoencoder + derive threshold tau
python -m src.training.train_industrial --config config/config.yaml

# 6. run tests before you trust any of it
pytest tests/

# 7. serve
uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# 8. (optional) ops dashboard
streamlit run dashboard/app.py
```

## Honesty checkpoints — do these before writing any number in a report

- After step 2, print the class balance of the **validation** split (must NOT be SMOTE-balanced —
  only the training split gets resampled, or your validation metrics are fiction).
- After step 4/5, log train vs. validation loss curves. A large gap = overfitting on the small
  positive class (3,615 theft accounts total). This is expected, not optional to check.
- There is no consumer-type ("residential" vs "industrial") column in SGCC. Step 2 assigns a
  **proxy** label by consumption magnitude (see `preprocess.py`, `assign_consumer_type_proxy`).
  This is a heuristic, not ground truth — every report you write must say so.
