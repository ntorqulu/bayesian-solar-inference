"""
Merge ESIOS prices + generation + AEMET weather into unified dataset
Output: data/merged/merged_dataset_with_generation.csv
         data/merged/merged_dataset.csv  (if generation not yet available)
"""

import pandas as pd
import numpy as np
import os

# Ensure output directory exists
os.makedirs("data/merged", exist_ok=True)

# ── LOAD SOURCE FILES ─────────────────────────────────────────────────────────
print("Loading source datasets...")

# Prices
prices_df = pd.read_csv("data/raw/esios_prices.csv")
prices_df["date"] = (
    pd.to_datetime(prices_df["date"], utc=True)
    .dt.tz_convert("Europe/Madrid")
    .dt.normalize()
    .dt.tz_localize(None)
)
print(f"Prices: {len(prices_df)} rows, {prices_df.date.min().date()} → {prices_df.date.max().date()}")

# Generation (optional — still downloading?)
try:
    gen_df = pd.read_csv("data/raw/esios_generation.csv")
    # Dates saved as tz-naive strings from the pull script — parse directly
    gen_df["date"] = pd.to_datetime(gen_df["date"]).dt.normalize()
    print(f"Generation: {len(gen_df)} rows, {gen_df.date.min().date()} → {gen_df.date.max().date()}")
    HAS_GEN = True
except FileNotFoundError:
    print("⚠ Generation data not found yet (still downloading?)")
    HAS_GEN = False
    gen_df = None

# Weather
weather_df = pd.read_csv("data/raw/aemet_barcelona_2015_2026.csv")
weather_df["date"] = pd.to_datetime(weather_df["date"]).dt.normalize()
print(f"Weather: {len(weather_df)} rows, {weather_df.date.min().date()} → {weather_df.date.max().date()}")

# ── MERGE ─────────────────────────────────────────────────────────────────────
print("\nMerging datasets...")

result = prices_df.copy()

if HAS_GEN:
    result = result.merge(gen_df, on="date", how="left")
    print(f"✓ Generation merged")

result = result.merge(weather_df, on="date", how="left")
print(f"✓ Weather merged")

# ── FEATURE ENGINEERING ───────────────────────────────────────────────────────
print("\nEngineering features...")

# Price regimes
result["regime"] = "normal"
result.loc[result.price_spain <= 20,  "regime"] = "collapse"
result.loc[result.price_spain >= 100, "regime"] = "spike"
result["is_collapse"] = (result["regime"] == "collapse").astype(int)

# Time features
result["year"]       = result["date"].dt.year
result["month"]      = result["date"].dt.month
result["day_of_year"]= result["date"].dt.dayofyear
result["quarter"]    = result["date"].dt.quarter
result["is_weekend"] = (result["date"].dt.dayofweek >= 5).astype(int)

# Renewable generation ratio (requires generation data and penetration columns)
if HAS_GEN and "wind_penetration" in result.columns:
    print("  - Computing gen_to_demand_ratio (wind + solar vs demand)")
    result["gen_to_demand_ratio"] = (
        (result["wind_gen_mwh_day"] + result["solar_pv_mwh_day"])
        / result["demand_mwh_day"].replace(0, np.nan)
    )

    # ── Penetration validation ────────────────────────────────────────────────
    demand_mean = result["demand_mwh_day"].mean()
    ren_mean    = result["renewables_penetration"].mean() * 100
    demand_ok   = 550_000 < demand_mean < 800_000
    ren_ok      = 15 < ren_mean < 50
    print(f"\n  [VALIDATION] demand_mwh_day mean : {demand_mean:>10,.0f}  "
          f"{'✅ OK' if demand_ok else '❌ INFLATED — re-run pull_all_esios_data.py'}")
    print(f"  [VALIDATION] renewables_pen mean : {ren_mean:>9.1f}%  "
          f"{'✅ OK' if ren_ok else '❌ WRONG — re-run pull_all_esios_data.py'}")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL MERGED DATASET")
print("=" * 60)
print(f"Rows      : {len(result)}")
print(f"Dates     : {result.date.min().date()} → {result.date.max().date()}")
print(f"Columns   : {len(result.columns)}")
print(f"\nRegime distribution:")
print(result["regime"].value_counts().to_string())
print(f"\nCollapse rate: {result['is_collapse'].mean():.4f} ({result['is_collapse'].mean() * 100:.2f}%)")
print(f"\nMissing values:\n{result.isna().sum().to_string()}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
output_file = (
    "data/merged/merged_dataset_with_generation.csv"
    if HAS_GEN
    else "data/merged/merged_dataset.csv"
)
result.to_csv(output_file, index=False)
print(f"\n✓ Saved → {output_file}")

# ── SAMPLE INSPECTION ─────────────────────────────────────────────────────────
print(f"\nSample rows (first 5):")
sample_cols = ["date", "price_spain", "spread_fr_es", "regime", "temp_mean_c",
               "wind_speed_ms", "sunshine_hours"]
print(result.head()[sample_cols].to_string(index=False))

if HAS_GEN and "wind_penetration" in result.columns:
    print(f"\nGeneration columns sample:")
    print(result[["date", "wind_gen_mwh_day", "solar_pv_mwh_day",
                  "wind_penetration", "solar_penetration"]].head().to_string(index=False))

# Sanity check around blackout
if (result["date"] >= "2025-04-25").any() and (result["date"] <= "2025-05-01").any():
    print(f"\nBlackout window (Apr 25 – May 1, 2025):")
    cols = ["date", "price_spain", "is_collapse", "temp_mean_c", "wind_speed_ms"]
    if HAS_GEN and "wind_penetration" in result.columns:
        cols.extend(["wind_penetration", "solar_penetration"])
    mask = (result["date"] >= "2025-04-25") & (result["date"] <= "2025-05-01")
    print(result[mask][cols].to_string(index=False))
