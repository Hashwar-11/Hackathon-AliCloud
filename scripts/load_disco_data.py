"""
Load customer operational context from a CSV into the database,
so the API's Verification Agent rules fire with real context instead of defaults.

Run:
    python scripts/load_disco_data.py --input data/raw/pakistan/demo_contexts.csv

The CSV must have columns matching CustomerContext fields:
    consumer_id, feeder_id, transformer_id, tariff_category, sanctioned_load_kw,
    feeder_loss_pct, recent_audit_result, months_since_last_audit,
    billing_dispute_open, known_grid_topology_issue, has_rooftop_solar,
    solar_net_metering_active, is_seasonal_occupancy, tamper_event_count,
    meter_seal_broken, reverse_current_alert, historical_mean_kwh,
    recent_30d_mean_kwh, tariff_rate_per_kwh
"""
import argparse
import os
import sys

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.db import SessionLocal, init_db


def load_contexts(csv_path: str):
    """Load customer contexts from CSV into the database.

    Note: This stores contexts in a simple JSON file (data/customer_contexts.json)
    rather than a database table, since the API reads context from the request payload.
    The dashboard can then pre-populate the context form from this file.

    For production use with a real DISCO database, replace this with an ORM model
    and a SQLAlchemy table.
    """
    import json

    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} customer contexts from {csv_path}")

    # Validate required columns
    required = ["consumer_id", "feeder_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert to list of dicts, handling NaN and bool columns
    contexts = []
    for _, row in df.iterrows():
        ctx = {}
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                continue
            # Convert numpy bool to Python bool
            if isinstance(val, (bool, np.bool_)):
                val = bool(val)
            elif isinstance(val, (np.integer,)):
                val = int(val)
            elif isinstance(val, (np.floating,)):
                val = float(val)
            ctx[col] = val
        contexts.append(ctx)

    # Save as JSON for the dashboard/API to load
    out_path = "data/customer_contexts.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(contexts, f, indent=2)

    print(f"Saved {len(contexts)} contexts → {out_path}")
    print(f"\nThe dashboard can now pre-populate the Verification context form")
    print(f"from this file. The API accepts context in the POST /predict/single body.")
    print(f"\nExample API call with context:")
    print(f"""
    curl -X POST http://localhost:8000/api/v1/predict/single \\
      -H "Content-Type: application/json" \\
      -d '{{
        "consumer_id": "{contexts[0]["consumer_id"]}",
        "client_type": "residential",
        "daily_kwh": [10.0, 10.5, 9.8, ...],
        "context": {{
          "feeder_id": "{contexts[0].get("feeder_id", "FEEDER-01")}",
          "feeder_loss_pct": {contexts[0].get("feeder_loss_pct", 18.0)},
          "historical_mean_kwh": {contexts[0].get("historical_mean_kwh", 15.0)},
          "recent_30d_mean_kwh": {contexts[0].get("recent_30d_mean_kwh", 5.0)}
        }}
      }}'
    """)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load customer context CSV into the system")
    parser.add_argument("--input", required=True, help="Path to the context CSV file")
    args = parser.parse_args()

    import numpy as np
    load_contexts(args.input)
