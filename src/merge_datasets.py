"""
merge_datasets.py
=================
Merges all data sources into a single daily dataset for the hidden solar model.

Inputs  (data/raw/):
  esios_prices.csv                  — daily prices Spain/France/Portugal
  esios_generation.csv              — daily demand, wind, utility solar PV
  aemet_barcelona_2015_2026.csv     — daily weather (Barcelona proxy)
  esios_selfconsumption.csv         — monthly self-consumption PV capacity
                                      (indicator 1945, one row per region)
  idae_selfconsumption_capacity.csv — annual IDAE capacity (validation only)

Output (data/merged/):
  merged_dataset.csv

Known data quality issues (fixed here):
  1. demand_mwh_day constant (3 values) — pull script bug; flagged at runtime
  2. solar_pv_mwh_day decreases over time — geo_id filter bug; flagged + clipped
  3. selfcons capacity per-region not national — FIXED: sum across regions here
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

os.makedirs("data/merged", exist_ok=True)

SELFCONS_PV_INDICATOR = 1945   # Potencia instalada autoconsumo solar fotovoltaica


# ── HELPERS ───────────────────────────────────────────────────────────────────

def parse_date_col(series: pd.Series) -> pd.Series:
    """Parse date column robustly — handles both tz-aware and tz-naive CSV dates."""
    parsed = pd.to_datetime(series, errors="coerce")
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    return parsed.dt.normalize()


def impute_monthly_mean(series: pd.Series, dates: pd.Series) -> pd.Series:
    """Fill NaN with the mean of the same (year, month). Falls back to overall mean."""
    s = series.copy()
    monthly_means = s.groupby(dates.dt.to_period("M")).transform("mean")
    s = s.fillna(monthly_means).fillna(s.mean())
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
    print("  Generation  : ⚠ not found")
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
    sc_raw["date"]  = parse_date_col(sc_raw["date"])
    sc_raw["value"] = pd.to_numeric(sc_raw["value"], errors="coerce")

    # FIX: indicator 1945 returns ONE ROW PER REGION per month (~17 regions).
    # Must SUM across all regions to get the national total before joining.
    sc_national = (
        sc_raw[sc_raw["indicator_id"] == SELFCONS_PV_INDICATOR]
        .groupby("date")["value"]
        .sum()                              # national total MW
        .reset_index()
        .rename(columns={"value": "selfcons_pv_capacity_mw"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    print(f"  Self-cons   : {len(sc_national)} monthly rows  "
          f"{sc_national.date.min().date()} → {sc_national.date.max().date()}")
    print(f"    Dec 2024 national total : "
          f"{sc_national[sc_national.date.dt.year == 2024].tail(1)['selfcons_pv_capacity_mw'].values[0]:,.0f} MW")
    print(f"    Dec 2025 national total : "
          f"{sc_national[sc_national.date.dt.year == 2025].tail(1)['selfcons_pv_capacity_mw'].values[0]:,.0f} MW")
    HAS_SC = True
except FileNotFoundError:
    print("  Self-cons   : ⚠ not found — run pull_selfconsumption.py first")
    HAS_SC = False
    sc_national = None

# ── IDAE annual capacity (validation only) ────────────────────────────────────
try:
    idae_df = pd.read_csv("data/raw/idae_selfconsumption_capacity.csv")
    HAS_IDAE = True
except FileNotFoundError:
    HAS_IDAE = False
    idae_df = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. INTERPOLATE SELF-CONSUMPTION CAPACITY: MONTHLY → DAILY
# ══════════════════════════════════════════════════════════════════════════════
if HAS_SC:
    print("\nInterpolating self-consumption capacity to daily frequency...")

    date_min    = prices_df["date"].min()
    date_max    = prices_df["date"].max()
    daily_index = pd.DataFrame({"date": pd.date_range(date_min, date_max, freq="D")})

    # Left-join monthly national totals onto daily index
    sc_daily = daily_index.merge(sc_national, on="date", how="left")

    # Forward-fill carries each month's value across all days of that month.
    # Backward-fill handles dates before the first recorded value.
    sc_daily["selfcons_pv_capacity_mw"] = (
        sc_daily["selfcons_pv_capacity_mw"].ffill().bfill().fillna(0)
    )

    # Validation against IDAE annual figures
    if HAS_IDAE:
        print("\n  Validation — national Dec capacity vs IDAE annual figures:")
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
            match = dec_vals[dec_vals["year"] == yr]["selfcons_pv_capacity_mw"]
            if not match.empty:
                esios_v  = match.values[0]
                diff_pct = abs(esios_v - ref) / ref * 100
                status   = "✅" if diff_pct < 25 else "⚠"
                print(f"    {yr}: ESIOS={esios_v:>8,.0f} MW  "
                      f"IDAE={ref:>7,.0f} MW  diff={diff_pct:.1f}%  {status}")


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
    print(f"  ✓ Self-consumption capacity merged (national sum, daily interpolated)")

if HAS_IDAE:
    result["year_int"] = result["date"].dt.year
    idae_map = idae_df.set_index("year")["cumulative_capacity_mw"].to_dict()
    result["idae_capacity_mw"] = result["year_int"].map(idae_map)
    result.drop(columns=["year_int"], inplace=True)
    print(f"  ✓ IDAE annual capacity merged (validation column)")


# ══════════════════════════════════════════════════════════════════════════════
# 4. DATA QUALITY FIXES
# ══════════════════════════════════════════════════════════════════════════════
print("\nApplying data quality fixes...")

# FIX: solar_pv_mwh_day has 13 negative values (ESIOS data artefacts)
if "solar_pv_mwh_day" in result.columns:
    n_neg = (result["solar_pv_mwh_day"] < 0).sum()
    if n_neg > 0:
        result["solar_pv_mwh_day"] = result["solar_pv_mwh_day"].clip(lower=0)
        result["solar_pv_mean_mw"]  = result["solar_pv_mean_mw"].clip(lower=0)
        print(f"  ✓ Clipped {n_neg} negative solar_pv_mwh_day values to 0")

# FLAG: demand is constant — this indicates a pull script bug
if "demand_mwh_day" in result.columns:
    n_unique_demand = result["demand_mwh_day"].nunique()
    if n_unique_demand <= 5:
        print(f"\n  ❌ WARNING: demand_mwh_day has only {n_unique_demand} unique values")
        print(f"     This means indicator 469 was not properly aggregated in pull_all_esios_data.py")
        print(f"     The demand column is UNRELIABLE — re-pull required")
        print(f"     See: process_generation() — check that df.groupby('date').sum() ")
        print(f"     is summing across all 17 provincial geo_ids correctly")

# FLAG: solar decreasing over time — this indicates a geo_id filter bug
if "solar_pv_mwh_day" in result.columns:
    solar_2015 = result[result["date"].dt.year == 2015]["solar_pv_mean_mw"].mean()
    solar_2024 = result[result["date"].dt.year == 2024]["solar_pv_mean_mw"].mean()
    if solar_2024 < solar_2015:
        print(f"\n  ❌ WARNING: solar_pv_mean_mw DECREASES over time")
        print(f"     2015 avg: {solar_2015:,.0f} MW  |  2024 avg: {solar_2024:,.0f} MW")
        print(f"     Expected: 2024 >> 2015 (utility solar grew from ~4.5 GW to ~25 GW)")
        print(f"     Cause: geo_id=8741 filter in process_generation may have selected")
        print(f"     wrong series. Check diagnostic output of pull_all_esios_data.py")


# ══════════════════════════════════════════════════════════════════════════════
# 5. WEATHER IMPUTATION (AEMET gaps)
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
    n_before = result[col].isna().sum()
    if n_before > 0:
        result[col] = impute_monthly_mean(result[col], result["date"])
        n_after = result[col].isna().sum()
        print(f"  {col:<25} {n_before:>4} NaN → {n_after:>4} NaN")
    else:
        print(f"  {col:<25} no missing values")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

# Time features
result["year"]        = result["date"].dt.year
result["month"]       = result["date"].dt.month
result["day_of_year"] = result["date"].dt.dayofyear
result["quarter"]     = result["date"].dt.quarter
result["is_weekend"]  = (result["date"].dt.dayofweek >= 5).astype(int)
result["season"]      = result["month"].map({
    12: "winter", 1: "winter",  2: "winter",
     3: "spring", 4: "spring",  5: "spring",
     6: "summer", 7: "summer",  8: "summer",
     9: "autumn", 10: "autumn", 11: "autumn",
})

# Price regimes
result["regime"]      = "normal"
result.loc[result["price_spain"] <= 20,  "regime"] = "collapse"
result.loc[result["price_spain"] >= 100, "regime"] = "spike"
result["is_collapse"] = (result["regime"] == "collapse").astype(int)

# Generation features
if HAS_GEN and "wind_gen_mwh_day" in result.columns:
    result["gen_to_demand_ratio"] = (
        (result["wind_gen_mwh_day"] + result["solar_pv_mwh_day"])
        / result["demand_mwh_day"].replace(0, np.nan)
    )

# Self-consumption features (key for the hidden solar model)
if HAS_SC and "sunshine_hours" in result.columns:
    # Theoretical max generation proxy: capacity × sunshine hours
    # The Bayesian model will refine the efficiency coefficient
    result["selfcons_theoretical_mwh"] = (
        result["selfcons_pv_capacity_mw"] * result["sunshine_hours"]
    )

    if HAS_GEN and "demand_mwh_day" in result.columns:
        result["selfcons_share_of_demand"] = (
            result["selfcons_theoretical_mwh"]
            / result["demand_mwh_day"].replace(0, np.nan)
        ).clip(0, None)

        # Validation: REE stated ~4% share in Jan-Nov 2025
        mask_2025 = result["year"] == 2025
        if mask_2025.any():
            share_2025 = result.loc[mask_2025, "selfcons_share_of_demand"].mean()
            ok = 0.02 < share_2025 < 0.10
            print(f"\n  [VALIDATION] 2025 selfcons_share_of_demand: "
                  f"{share_2025*100:.2f}%  "
                  f"{'✅ consistent with REE ~4%' if ok else '⚠ unexpected — check demand column'}")


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
    print("    none ✅")

print(f"\n  Regime distribution:")
print(result["regime"].value_counts().to_string())

output_file = "data/merged/merged_dataset.csv"
result.to_csv(output_file, index=False)
print(f"\n✓ Saved → {output_file}")

# ── Blackout window sanity check ──────────────────────────────────────────────
mask_blackout = (
    (result["date"] >= "2025-04-25") & (result["date"] <= "2025-05-01")
)
if mask_blackout.any():
    print(f"\nBlackout window (Apr 25 – May 1, 2025):")
    cols = ["date", "price_spain", "regime", "sunshine_hours", "temp_mean_c"]
    if HAS_SC:
        cols.append("selfcons_pv_capacity_mw")
    if HAS_GEN and "renewables_penetration" in result.columns:
        cols.append("renewables_penetration")
    print(result[mask_blackout][cols].to_string(index=False))

# ── Self-consumption trajectory ───────────────────────────────────────────────
if HAS_SC:
    print(f"\nSelf-consumption PV capacity — national total (Dec each year):")
    dec_cap = (
        result[result["date"].dt.month == 12]
        .groupby(result["date"].dt.year)["selfcons_pv_capacity_mw"]
        .last()
    )
    for yr, cap in dec_cap.items():
        bar = "█" * int(cap / 500)
        print(f"  {yr}: {cap:>8,.0f} MW  {bar}")