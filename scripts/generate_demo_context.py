"""
Generate realistic operational context for the 42 confirmed Pakistani theft cases,
so the Verification Agent rules fire during the demo instead of being inert.

Run:
    python scripts/generate_demo_context.py

Produces:
    data/raw/pakistan/demo_contexts.csv

Load into the database:
    python scripts/load_disco_data.py --input data/raw/pakistan/demo_contexts.csv
"""
import os
import numpy as np
import pandas as pd

SEED = 42
N = 42
FEEDERS = ["FEEDER-NORTH-01", "FEEDER-SOUTH-02", "FEEDER-EAST-03", "FEEDER-IND-04"]
AUDIT_OPTIONS = ["none", "cleared", "confirmed_theft", "meter_fault"]
TARIFF_CATEGORIES = ["residential", "residential", "residential", "commercial"]  # mostly residential


def main():
    np.random.seed(SEED)

    contexts = pd.DataFrame({
        "consumer_id": [f"PK{i+1:03d}" for i in range(N)],
        "feeder_id": np.random.choice(FEEDERS, N),
        "transformer_id": [f"TX-{np.random.randint(1, 6):02d}" for _ in range(N)],
        "tariff_category": np.random.choice(TARIFF_CATEGORIES, N),
        "sanctioned_load_kw": np.round(np.random.uniform(5.0, 15.0, N), 1),
        # Pakistani DISCO feeders typically have 15-30% loss
        "feeder_loss_pct": np.round(np.random.uniform(15.0, 30.0, N), 1),
        # Most have no audit; some cleared; a couple confirmed theft (recidivism signal)
        "recent_audit_result": np.random.choice(
            AUDIT_OPTIONS, N, p=[0.65, 0.15, 0.10, 0.10]
        ),
        "months_since_last_audit": np.random.choice(
            [999, 1, 2, 3, 6, 12, 18], N, p=[0.65, 0.05, 0.05, 0.05, 0.05, 0.10, 0.05]
        ),
        # ~10% have billing disputes
        "billing_dispute_open": np.random.choice([False, True], N, p=[0.90, 0.10]),
        # ~5% have known grid topology issues
        "known_grid_topology_issue": np.random.choice([False, True], N, p=[0.95, 0.05]),
        # ~8% have rooftop solar (growing in Pakistan)
        "has_rooftop_solar": np.random.choice([False, True], N, p=[0.92, 0.08]),
        "solar_net_metering_active": np.random.choice([False, True], N, p=[0.94, 0.06]),
        # ~5% seasonal (agricultural tube-wells)
        "is_seasonal_occupancy": np.random.choice([False, True], N, p=[0.95, 0.05]),
        # Smart meter tamper events — most have 0, a few have 1-2
        "tamper_event_count": np.random.choice([0, 0, 0, 0, 0, 1, 1, 2], N),
        # ~5% have broken meter seal
        "meter_seal_broken": np.random.choice([False, True], N, p=[0.95, 0.05]),
        # ~3% reverse current alerts
        "reverse_current_alert": np.random.choice([False, True], N, p=[0.97, 0.03]),
        # Consumption profiles — theft cases tend to show drops
        "historical_mean_kwh": np.round(np.random.uniform(10.0, 60.0, N), 1),
        "recent_30d_mean_kwh": np.round(np.random.uniform(1.0, 20.0, N), 1),
        # Tariff rate — PKR/kWh, tiered but use a representative average
        "tariff_rate_per_kwh": np.round(np.random.uniform(25.0, 35.0, N), 1),
    })

    # Make solar cases consistent: if has_rooftop_solar, likely net metering too
    solar_mask = contexts["has_rooftop_solar"]
    contexts.loc[solar_mask, "solar_net_metering_active"] = True

    # Make tamper cases consistent: if seal broken, at least 1 tamper event
    seal_mask = contexts["meter_seal_broken"]
    contexts.loc[seal_mask, "tamper_event_count"] = np.maximum(
        contexts.loc[seal_mask, "tamper_event_count"].values, 1
    )

    out_dir = "data/raw/pakistan"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "demo_contexts.csv")
    contexts.to_csv(out_path, index=False)

    print(f"Generated {N} demo customer contexts → {out_path}")
    print(f"\nSummary:")
    print(f"  Feeders: {contexts['feeder_id'].value_counts().to_dict()}")
    print(f"  Audit results: {contexts['recent_audit_result'].value_counts().to_dict()}")
    print(f"  Rooftop solar: {contexts['has_rooftop_solar'].sum()}")
    print(f"  Billing disputes: {contexts['billing_dispute_open'].sum()}")
    print(f"  Tamper events ≥1: {(contexts['tamper_event_count'] >= 1).sum()}")
    print(f"  Meter seal broken: {contexts['meter_seal_broken'].sum()}")
    print(f"  Feeder loss %: mean={contexts['feeder_loss_pct'].mean():.1f}, "
          f"min={contexts['feeder_loss_pct'].min():.1f}, max={contexts['feeder_loss_pct'].max():.1f}")
    print(f"\nHONESTY: These are SYNTHETIC demo contexts — not real DISCO operational data.")
    print(f"Label them as 'simulated' in any presentation or dashboard.")


if __name__ == "__main__":
    main()
