"""
Generate synthetic normal Pakistani customers for fine-tuning.

Since real normal customer data is confidential/unavailable from DISCOs,
this script generates realistic synthetic normal customers based on:
1. SGCC normal customer patterns (as reference)
2. Pakistani consumption context (lower consumption, different climate)
3. Realistic daily variation patterns

These are SYNTHETIC — clearly labeled as such for honest reporting.

Run:
    python scripts/generate_pakistani_normals.py

Output:
    data/raw/pakistan/pakistan_normals_synthetic.csv
    - 300 normal customers (FLAG = 0)
    - 365 days of daily kWh readings
    - Realistic Pakistani residential consumption patterns
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

SEED = 42
N_NORMALS = 300  # Generate 300 normal customers
N_DAYS = 365  # One year of data

# Pakistani residential consumption characteristics
# (lower than Chinese SGCC due to smaller homes, lower appliance usage)
PK_MEAN_KWH = 12.0  # Average daily consumption (kWh)
PK_STD_KWH = 4.0  # Day-to-day variation
PK_SEASONAL_FACTOR = 0.3  # Summer vs winter variation (30% higher in summer)

# Realistic consumption patterns
PATTERNS = {
    "low_consumption": {"mean": 6.0, "std": 2.0, "weight": 0.25},  # Small households
    "medium_consumption": {"mean": 12.0, "std": 3.5, "weight": 0.50},  # Average households
    "high_consumption": {"mean": 22.0, "std": 6.0, "weight": 0.20},  # Larger households
    "very_high_consumption": {"mean": 35.0, "std": 10.0, "weight": 0.05},  # Wealthy households
}


def generate_seasonal_pattern(n_days: int, base_date: datetime) -> np.ndarray:
    """Generate a seasonal multiplier (higher in summer due to AC/fans)."""
    day_of_year = np.arange(n_days)
    # Pakistan summer: April-September (days 90-270)
    # Peak in June-July (days 150-210)
    seasonal = 1.0 + PK_SEASONAL_FACTOR * np.sin(2 * np.pi * (day_of_year - 60) / 365)
    return np.clip(seasonal, 0.8, 1.3)  # Clamp to [0.8, 1.3]


def generate_weekly_pattern(n_days: int, base_date: datetime) -> np.ndarray:
    """Generate weekly pattern (slightly higher on weekends)."""
    dates = [base_date + timedelta(days=i) for i in range(n_days)]
    day_of_week = np.array([d.weekday() for d in dates])  # 0=Monday, 6=Sunday
    # Weekends (5=Sat, 6=Sun) have ~10% higher consumption
    weekly = np.where(day_of_week >= 5, 1.10, 1.0)
    return weekly


def generate_normal_customer(customer_id: str, n_days: int, pattern: dict) -> dict:
    """Generate one normal customer's consumption series."""
    base_date = datetime(2024, 1, 1)
    
    # Base consumption with random variation
    base_mean = pattern["mean"] * np.random.uniform(0.8, 1.2)  # ±20% variation
    base_std = pattern["std"]
    
    # Generate daily consumption
    daily_kwh = np.random.normal(base_mean, base_std, n_days)
    
    # Apply seasonal pattern
    seasonal = generate_seasonal_pattern(n_days, base_date)
    daily_kwh *= seasonal
    
    # Apply weekly pattern
    weekly = generate_weekly_pattern(n_days, base_date)
    daily_kwh *= weekly
    
    # Add random noise
    noise = np.random.normal(1.0, 0.05, n_days)  # ±5% noise
    daily_kwh *= noise
    
    # Ensure non-negative
    daily_kwh = np.maximum(daily_kwh, 0.0)
    
    # Round to 2 decimal places
    daily_kwh = np.round(daily_kwh, 2)
    
    # Generate column names (Excel serial dates starting from 1900-01-01)
    # 2024-01-01 = 45292 in Excel date system
    start_serial = 45292
    day_columns = [str(start_serial + i) for i in range(n_days)]
    
    # Build row
    row = {"CONS_NO": customer_id, "FLAG": 0}  # FLAG=0 for normal
    for i, col in enumerate(day_columns):
        row[col] = daily_kwh[i]
    
    return row


def main():
    np.random.seed(SEED)
    
    print(f"[generate_pakistani_normals] Generating {N_NORMALS} synthetic normal customers...")
    print(f"  - Based on Pakistani residential consumption patterns")
    print(f"  - {N_DAYS} days of data per customer")
    print(f"  - Seasonal + weekly patterns included")
    print()
    
    # Select patterns for each customer
    pattern_names = list(PATTERNS.keys())
    pattern_weights = [PATTERNS[p]["weight"] for p in pattern_names]
    selected_patterns = np.random.choice(pattern_names, N_NORMALS, p=pattern_weights)
    
    # Generate customers
    rows = []
    for i in range(N_NORMALS):
        customer_id = f"PK-NORM-{i+1:04d}"
        pattern = PATTERNS[selected_patterns[i]]
        row = generate_normal_customer(customer_id, N_DAYS, pattern)
        rows.append(row)
        
        if (i + 1) % 50 == 0:
            print(f"  Generated {i+1}/{N_NORMALS} customers...")
    
    # Create DataFrame
    df = pd.DataFrame(rows)
    
    # Save to CSV
    out_dir = "data/raw/pakistan"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "pakistan_normals_synthetic.csv")
    df.to_csv(out_path, index=False)
    
    print()
    print(f"[generate_pakistani_normals] ✓ Generated {len(df)} normal customers")
    print(f"  Output: {out_path}")
    print(f"  Columns: CONS_NO, FLAG, <{N_DAYS} day columns>")
    print(f"  FLAG distribution: {df['FLAG'].value_counts().to_dict()}")
    print()
    print("⚠️  IMPORTANT: These are SYNTHETIC customers, not real data.")
    print("   Label clearly in any reports: 'synthetic normal customers'")
    print()
    
    # Print consumption statistics
    day_cols = [c for c in df.columns if c not in ("CONS_NO", "FLAG")]
    mean_per_customer = df[day_cols].mean(axis=1)
    print("Consumption statistics (kWh/day):")
    print(f"  Mean: {mean_per_customer.mean():.2f}")
    print(f"  Std: {mean_per_customer.std():.2f}")
    print(f"  Min: {mean_per_customer.min():.2f}")
    print(f"  Max: {mean_per_customer.max():.2f}")
    print()
    
    # Show pattern distribution
    print("Pattern distribution:")
    for pattern_name in pattern_names:
        count = (selected_patterns == pattern_name).sum()
        print(f"  {pattern_name}: {count} customers ({count/N_NORMALS*100:.1f}%)")


if __name__ == "__main__":
    main()
