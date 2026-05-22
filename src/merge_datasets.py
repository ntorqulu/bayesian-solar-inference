"""
merge_datasets.py
=================
Merges all data sources into a single daily dataset for the hidden solar model.

Inputs  (data/raw/):
  esios_prices.csv                  — daily prices (Spain, France, Portugal)
  esios_generation.csv              — daily demand, wind, solar PV (utility-scale)
  aemet_barcelona_2015_2026.csv     — daily weather (Barcelona, proxy for national)
  esios_selfconsumption.csv         — monthly self-consumption PV capacity (indicator 1945)
  idae_selfconsumption_capacity.csv — annual IDAE capacity (validation only)

Output (data/merged/):
  merged_dataset.csv                — one row per day, all features aligned

Key additions vs previous version:
  • Self-consumption capacity merged and interpolated monthly → daily
  • sunshine_hours imputed with monthly mean (AEMET gaps)
  • Robust date parsing (handles both tz-aware and tz-naive CSV dates)
  • IDAE annual capacity added as validation column
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

os.makedirs("data/merged", exist_ok=True)

SELFCONS_PV_INDICATOR = 1945   # Potencia instalada autoconsumo solar fotovoltaica


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_date_col(series: pd.Series) -> pd.Series:
    """
    Robustly parse a date column that may be tz-aware or tz-naive.
    Always returns tz-naive, normalised to midnight.
    """
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    return parsed.dt.normalize()


def impute_monthly_mean(series: pd.Series, dates: pd.Series) -> pd.Series:
    """
    Fill NaN values in `series` using the mean of the same (year, month) group.
    Falls back to the overall mean for any month with no valid values.
    """
    s = series.copy()
    month_key = dates.dt.to_period("M")
    monthly_means = s.groupby(month_key).transform("mean")
    overall_mean  = s.mean()
    s = s.fillna(monthly_means).fillna(overall_mean)
    return s


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD ALL SOURCE FILES
# ══════════════════════════════════════════════════════════════════════════════
print("Loading source datasets...")

# ── Prices ────────────────────────────────────────────────────────────────────
prices_df = pd.read_csv("data/raw/esios_prices.csv")
prices_df["date"] = parse_date_col(prices_df["date"])
print(f"  Prices      : {len(prices_df):,} rows  "
      f"{prices_df.date.min().date()} → {prices_df.date.max().date()}")

# ── Generation ────────────────────────────────────────────────────────────────
try:
    gen_df = pd.read_csv("data/raw/esios_generation.csv")
    gen_df["date"] = parse_date_col(gen_df["date"])
    print(f"  Generation  : {len(gen_df):,} rows  "
          f"{gen_df.date.min().date()} → {gen_df.date.max().date()}")
    HAS_GEN = True
except FileNotFoundError:
    print("  Generation  : ⚠ not found (still downloading?)")
    HAS_GEN = False
    gen_df = None

# ── Weather ───────────────────────────────────────────────────────────────────
weather_df = pd.read_csv("data/raw/aemet_barcelona_2015_2026.csv")
weather_df["date"] = parse_date_col(weather_df["date"])
print(f"  Weather     : {len(weather_df):,} rows  "
      f"{weather_df.date.min().date()} → {weather_df.date.max().date()}")

# ── Self-consumption capacity (monthly, indicator 1945) ───────────────────────
try:
    sc_raw = pd.read_csv("data/raw/esios_selfconsumption.csv")
    sc_raw["date"] = parse_date_col(sc_raw["date"])
    sc_raw["value"] = pd.to_numeric(sc_raw["value"], errors="coerce")

    # Keep only solar PV self-consumption capacity (indicator 1945)
    sc_pv = (
        sc_raw[sc_raw["indicator_id"] == SELFCONS_PV_INDICATOR]
        [["date", "value"]]
        .rename(columns={"value": "selfcons_pv_capacity_mw"})
        .dropna(subset=["selfcons_pv_capacity_mw"])
        .sort_values("date")
        .drop_duplicates("date")
        .reset_index(drop=True)
    )
    print(f"  Self-cons   : {len(sc_pv):,} monthly rows  "
          f"{sc_pv.date.min().date()} → {sc_pv.date.max().date()}  "
          f"(indicator {SELFCONS_PV_INDICATOR})")
    HAS_SC = True
except FileNotFoundError:
    print("  Self-cons   : ⚠ not found — run pull_selfconsumption.py first")
    HAS_SC = False
    sc_pv = None

# ── IDAE annual capacity (validation only) ────────────────────────────────────
try:
    idae_df = pd.read_csv("data/raw/idae_selfconsumption_capacity.csv")
    print(f"  IDAE table  : {len(idae_df)} annual rows (validation reference)")
    HAS_IDAE = True
except FileNotFoundError:
    print("  IDAE table  : ⚠ not found (optional, validation only)")
    HAS_IDAE = False
    idae_df = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. INTERPOLATE SELF-CONSUMPTION CAPACITY: MONTHLY → DAILY
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SC:
    print("\nInterpolating self-consumption capacity to daily frequency...")

    # Build a daily date range spanning the full study period
    date_min = prices_df["date"].min()
    date_max = prices_df["date"].max()
    daily_index = pd.DataFrame(
        {"date": pd.date_range(date_min, date_max, freq="D")}
    )

    # Left-join monthly capacity onto daily index
    # Monthly values represent capacity at start of that month
    sc_daily = daily_index.merge(sc_pv, on="date", how="left")

    # Forward-fill then backward-fill:
    #   forward-fill carries the last known monthly value across all days in the month
    #   backward-fill handles the period before the first recorded value (pre-2019 ≈ 0)
    sc_daily["selfcons_pv_capacity_mw"] = (
        sc_daily["selfcons_pv_capacity_mw"]
        .ffill()
        .bfill()
    )

    # For dates before 2019 where no ESIOS data exists, capacity was near-zero.
    # ESIOS starts in 2015 but the sun tax (abolished Oct 2018) kept installations tiny.
    # If bfill still leaves NaN (shouldn't happen but defensive), fill with 0.
    sc_daily["selfcons_pv_capacity_mw"] = (
        sc_daily["selfcons_pv_capacity_mw"].fillna(0)
    )

    # Validation: December values should match IDAE table
    if HAS_IDAE:
        print("\n  Validation — Dec capacity vs IDAE annual figures:")
        dec_vals = (
            sc_daily[sc_daily["date"].dt.month == 12]
            .groupby(sc_daily["date"].dt.year)["selfcons_pv_capacity_mw"]
            .last()
            .reset_index()
            .rename(columns={"date": "year"})
        )
        for _, row in idae_df.iterrows():
            yr  = int(row["year"])
            ref = row["cumulative_capacity_mw"]
            esios_val = dec_vals[dec_vals["year"] == yr]["selfcons_pv_capacity_mw"]
            if not esios_val.empty:
                esios_v = esios_val.values[0]
                diff_pct = abs(esios_v - ref) / ref * 100
                status = "✅" if diff_pct < 20 else "⚠"
                print(f"    {yr}: ESIOS={esios_v:,.0f} MW  IDAE={ref:,.0f} MW  "
                      f"diff={diff_pct:.1f}%  {status}")
            else:
                print(f"    {yr}: ESIOS=n/a  IDAE={ref:,.0f} MW")

    print(f"\n  Daily capacity range: "
          f"{sc_daily['selfcons_pv_capacity_mw'].min():.0f} – "
          f"{sc_daily['selfcons_pv_capacity_mw'].max():.0f} MW")


# ══════════════════════════════════════════════════════════════════════════════
# 3. MERGE ALL DATASETS
# ══════════════════════════════════════════════════════════════════════════════
print("\nMerging datasets...")

result = prices_df.copy()

if HAS_GEN:
    result = result.merge(gen_df, on="date", how="left")
    print(f"  ✓ Generation merged")

result = result.merge(weather_df, on="date", how="left")
print(f"  ✓ Weather merged")

if HAS_SC:
    result = result.merge(sc_daily, on="date", how="left")
    print(f"  ✓ Self-consumption capacity merged (daily interpolated)")

if HAS_IDAE:
    # Add IDAE annual capacity as a validation column (year-level join)
    result["year_int"] = result["date"].dt.year
    idae_map = idae_df.set_index("year")["cumulative_capacity_mw"].to_dict()
    result["idae_capacity_mw"] = result["year_int"].map(idae_map)
    result.drop(columns=["year_int"], inplace=True)
    print(f"  ✓ IDAE annual capacity merged (validation column)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. WEATHER IMPUTATION (sunshine_hours and other AEMET gaps)
# ══════════════════════════════════════════════════════════════════════════════
print("\nImputing weather gaps...")

WEATHER_COLS = [
    "temp_mean_c", "temp_max_c", "temp_min_c",
    "sunshine_hours", "wind_speed_ms", "wind_gust_ms",
    "precip_mm", "humidity_mean_pct",
    "pressure_max_hpa", "pressure_min_hpa",
]

for col in WEATHER_COLS:
    if col not in result.columns:
        continue
    n_missing_before = result[col].isna().sum()
    if n_missing_before > 0:
        result[col] = impute_monthly_mean(result[col], result["date"])
        n_missing_after = result[col].isna().sum()
        print(f"  {col:<25} {n_missing_before:>4} NaN → {n_missing_after:>4} NaN "
              f"(monthly mean imputation)")
    else:
        print(f"  {col:<25} no missing values")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

# Time features
result["year"]        = result["date"].dt.year
result["month"]       = result["date"].dt.month
result["day_of_year"] = result["date"].dt.dayofyear
result["quarter"]     = result["date"].dt.quarter
result["is_weekend"]  = (result["date"].dt.dayofweek >= 5).astype(int)
result["season"]      = result["month"].map({
    12: "winter", 1: "winter", 2: "winter",
     3: "spring", 4: "spring", 5: "spring",
     6: "summer", 7: "summer", 8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
})

# Price regimes (useful for EDA even if not the model target)
result["regime"]      = "normal"
result.loc[result["price_spain"] <= 20,  "regime"] = "collapse"
result.loc[result["price_spain"] >= 100, "regime"] = "spike"
result["is_collapse"] = (result["regime"] == "collapse").astype(int)

# Utility-scale renewables features
if HAS_GEN and "wind_penetration" in result.columns:
    result["gen_to_demand_ratio"] = (
        (result["wind_gen_mwh_day"] + result["solar_pv_mwh_day"])
        / result["demand_mwh_day"].replace(0, np.nan)
    )

# Self-consumption features (key for the hidden solar model)
if HAS_SC and HAS_GEN:
    # Theoretical max generation: capacity × peak sunshine
    # Full-load hours ≈ sunshine_hours (rough proxy, refined by the Bayesian model)
    result["selfcons_theoretical_mwh"] = (
        result["selfcons_pv_capacity_mw"] * result["sunshine_hours"]
    )

    # Apparent demand suppression: how much demand is hidden by self-consumption
    # This is the signal the Bayesian model will decompose
    # (only meaningful post-2019 when capacity was non-negligible)
    result["selfcons_share_of_demand"] = (
        result["selfcons_theoretical_mwh"]
        / result["demand_mwh_day"].replace(0, np.nan)
    ).clip(0, None)

    print(f"  ✓ selfcons_theoretical_mwh computed")
    print(f"  ✓ selfcons_share_of_demand computed")

    # Validation: in 2025, self-cons should be ~4% of demand (REE stated this)
    mask_2025 = result["year"] == 2025
    if mask_2025.any():
        share_2025 = result.loc[mask_2025, "selfcons_share_of_demand"].mean()
        print(f"\n  [VALIDATION] 2025 mean selfcons_share_of_demand: "
              f"{share_2025*100:.2f}%  "
              f"{'✅ ~4% matches REE announcement' if 0.02 < share_2025 < 0.08 else '⚠ check'}")


# ══════════════════════════════════════════════════════════════════════════════
# 6. VALIDATION SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("VALIDATION")
print("=" * 65)

if HAS_GEN:
    demand_mean    = result["demand_mwh_day"].mean()
    demand_mean_mw = result["demand_mean_mw"].mean()
    ren_mean       = result["renewables_penetration"].mean() * 100 if "renewables_penetration" in result.columns else 0
    print(f"\n  demand_mwh_day mean  : {demand_mean:>10,.0f}  "
          f"{'✅' if 550_000 < demand_mean < 800_000 else '❌'}")
    print(f"  demand_mean_mw mean  : {demand_mean_mw:>10,.0f}  "
          f"{'✅' if 22_000 < demand_mean_mw < 32_000 else '❌'}")
    print(f"  renewables_pen mean  : {ren_mean:>9.1f}%  "
          f"{'✅' if 8 < ren_mean < 50 else '❌'}")

if HAS_SC:
    cap_2025 = result.loc[result["year"] == 2025, "selfcons_pv_capacity_mw"].mean()
    print(f"  selfcons_pv cap 2025 : {cap_2025:>10,.0f} MW  "
          f"{'✅ ~8700 MW matches REE' if 7000 < cap_2025 < 10000 else '❌'}")


# ══════════════════════════════════════════════════════════════════════════════
# 7. SUMMARY & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL MERGED DATASET")
print("=" * 65)
print(f"  Rows      : {len(result):,}")
print(f"  Dates     : {result.date.min().date()} → {result.date.max().date()}")
print(f"  Columns   : {len(result.columns)}")
print(f"\n  Missing values (columns with >0 NaN):")
miss = result.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss):
    for col, n in miss.items():
        print(f"    {col:<35} {n:>5} ({n/len(result)*100:.1f}%)")
else:
    print("    none")

output_file = "data/merged/merged_dataset.csv"
result.to_csv(output_file, index=False)
print(f"\n✓ Saved → {output_file}")

# ── Blackout window sanity check ──────────────────────────────────────────────
mask_blackout = (
    (result["date"] >= "2025-04-25") & (result["date"] <= "2025-05-01")
)
if mask_blackout.any():
    print(f"\nBlackout window (Apr 25 – May 1, 2025):")
    cols = ["date", "price_spain", "is_collapse", "sunshine_hours", "temp_mean_c"]
    if HAS_SC:
        cols.append("selfcons_pv_capacity_mw")
    if HAS_GEN and "renewables_penetration" in result.columns:
        cols.append("renewables_penetration")
    print(result[mask_blackout][cols].to_string(index=False))

# ── Self-consumption capacity trajectory ──────────────────────────────────────
if HAS_SC:
    print(f"\nSelf-consumption PV capacity trajectory (annual Dec reading, MW):")
    dec_cap = (
        result[result["date"].dt.month == 12]
        .groupby(result["date"].dt.year)["selfcons_pv_capacity_mw"]
        .last()
    )
    for yr, cap in dec_cap.items():
        bar = "█" * int(cap / 500)
        print(f"  {yr}: {cap:>7,.0f} MW  {bar}")