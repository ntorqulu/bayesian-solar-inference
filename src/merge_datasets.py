"""
merge_datasets.py
=================
Merges all clean source files into a single daily dataset for the Bayesian
hidden solar model.

Inputs (data/raw/):
  esios_prices.csv              — indicator 600  : daily prices Spain/France/Portugal
  esios_demand.csv              — indicator 460  : daily demand (4x fix applied)
  esios_wind.csv                — indicator 10288: daily wind generation
  esios_solar.csv               — indicator 10358: daily utility solar PV (from 2019)
  aemet_barcelona_2015_2026.csv — daily weather (Barcelona proxy for Spain)
  esios_selfconsumption.csv     — indicator 1945 : monthly self-consumption capacity
                                                   (one row per region → summed here)
  idae_selfconsumption_capacity.csv — annual IDAE figures (validation only)

Output (data/merged/):
  merged_dataset.csv
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data/merged", exist_ok=True)
os.makedirs("data/processed/figures", exist_ok=True)

SELFCONS_PV_INDICATOR = 1945

# ── HELPERS ───────────────────────────────────────────────────────────────────

def load(path, label):
    """Load CSV, parse date column, normalise to tz-naive midnight."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    print(f"  {label:<14}: {len(df):>5,} rows  "
          f"{df.date.min().date()} → {df.date.max().date()}")
    return df


def impute_monthly_mean(series, dates):
    """Fill NaN with the same (year, month) mean; fall back to overall mean."""
    s = series.copy()
    monthly = s.groupby(dates.dt.to_period("M")).transform("mean")
    return s.fillna(monthly).fillna(s.mean())


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading source datasets...")

prices  = load("data/raw/esios_prices.csv",              "Prices")
demand  = load("data/raw/esios_demand.csv",              "Demand")
wind    = load("data/raw/esios_wind.csv",                "Wind")
solar   = load("data/raw/esios_solar.csv",               "Solar PV")
weather = load("data/raw/aemet_barcelona_2015_2026.csv", "Weather")

# Self-consumption: sum across regions → national monthly total
sc_raw = pd.read_csv("data/raw/esios_selfconsumption.csv")
sc_raw["date"]  = pd.to_datetime(sc_raw["date"], errors="coerce").dt.normalize()
sc_raw["value"] = pd.to_numeric(sc_raw["value"], errors="coerce")
sc_monthly = (
    sc_raw[sc_raw["indicator_id"] == SELFCONS_PV_INDICATOR]
    .groupby("date")["value"].sum()
    .reset_index()
    .rename(columns={"value": "selfcons_pv_capacity_mw"})
    .sort_values("date")
)
print(f"  {'Self-cons':<14}: {len(sc_monthly):>5,} monthly rows  "
      f"{sc_monthly.date.min().date()} → {sc_monthly.date.max().date()}")
dec24 = sc_monthly[sc_monthly.date.dt.year == 2024].tail(1)["selfcons_pv_capacity_mw"].values
dec25 = sc_monthly[sc_monthly.date.dt.year == 2025].tail(1)["selfcons_pv_capacity_mw"].values
if dec24.size: print(f"    Dec 2024 national total : {dec24[0]:,.0f} MW")
if dec25.size: print(f"    Dec 2025 national total : {dec25[0]:,.0f} MW")

try:
    idae = pd.read_csv("data/raw/idae_selfconsumption_capacity.csv")
    HAS_IDAE = True
except FileNotFoundError:
    HAS_IDAE = False
    idae = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. VALIDATE INDIVIDUAL SOURCES
# ══════════════════════════════════════════════════════════════════════════════
print("\nValidating individual sources...")

# Demand: all years should be 18k-35k MW
demand_by_yr = demand.groupby(demand.date.dt.year)["demand_mean_mw"].mean()
demand_ok = all(18000 < v < 35000 for v in demand_by_yr.values)
print(f"  Demand  all-years mean in 18k-35k: {'ok' if demand_ok else 'error'}")
if not demand_ok:
    for yr, v in demand_by_yr.items():
        if not (18000 < v < 35000):
            print(f"    error {yr}: {v:,.0f} MW — check fetch_demand.py")

# Solar: growing trend
s2022 = solar.groupby(solar.date.dt.year)["solar_mean_mw"].mean().get(2022, 0)
s2024 = solar.groupby(solar.date.dt.year)["solar_mean_mw"].mean().get(2024, 0)
solar_ok = s2024 > s2022 > 0
print(f"  Solar   growing 2022→2024 ({s2022:,.0f}→{s2024:,.0f} MW): "
      f"{'ok' if solar_ok else 'error check fetch_solar.py (use indicator 10358)'}")

# Wind: plausible mean
wind_mean = wind["wind_mean_mw"].mean()
wind_ok = 2000 < wind_mean < 6000
print(f"  Wind    mean {wind_mean:,.0f} MW (expected 2k-6k): {'ok' if wind_ok else 'error'}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTERPOLATE SELF-CONSUMPTION: MONTHLY → DAILY
# ══════════════════════════════════════════════════════════════════════════════
print("\nInterpolating self-consumption capacity to daily frequency...")

date_range = pd.DataFrame({
    "date": pd.date_range(prices.date.min(), prices.date.max(), freq="D")
})
sc_daily = date_range.merge(sc_monthly, on="date", how="left")
sc_daily["selfcons_pv_capacity_mw"] = (
    sc_daily["selfcons_pv_capacity_mw"].ffill().bfill().fillna(0)
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MERGE
# ══════════════════════════════════════════════════════════════════════════════
print("\nMerging datasets...")

# Rename columns for clarity before merging
demand = demand.rename(columns={
    "demand_mwh_day":  "demand_mwh_day",
    "demand_mean_mw":  "demand_mean_mw",
    "demand_peak_mw":  "demand_peak_mw",
    "demand_peak_hour":"demand_peak_hour",
})
wind = wind.rename(columns={
    "wind_mwh_day":   "wind_gen_mwh_day",
    "wind_mean_mw":   "wind_gen_mean_mw",
    "wind_peak_mw":   "wind_gen_peak_mw",
    "wind_peak_hour": "wind_gen_peak_hour",
})
solar = solar.rename(columns={
    "solar_mwh_day":   "solar_pv_mwh_day",
    "solar_mean_mw":   "solar_pv_mean_mw",
    "solar_peak_mw":   "solar_pv_peak_mw",
    "solar_peak_hour": "solar_pv_peak_hour",
})

result = prices.copy()
result = result.merge(demand,   on="date", how="left"); print("  ✓ Demand merged")
result = result.merge(wind,     on="date", how="left"); print("  ✓ Wind merged")
result = result.merge(solar,    on="date", how="left"); print("  ✓ Solar PV merged")
result = result.merge(weather,  on="date", how="left"); print("  ✓ Weather merged")
result = result.merge(sc_daily, on="date", how="left"); print("  ✓ Self-consumption merged")

if HAS_IDAE:
    idae_map = idae.set_index("year")["cumulative_capacity_mw"].to_dict()
    result["idae_capacity_mw"] = result["date"].dt.year.map(idae_map)
    print("  ✓ IDAE annual capacity merged (validation column)")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY FIXES
# ══════════════════════════════════════════════════════════════════════════════
print("\nApplying data quality fixes...")

# Clip negative solar values (ESIOS artefacts)
for col in ["solar_pv_mwh_day", "solar_pv_mean_mw"]:
    if col in result.columns:
        n_neg = (result[col] < 0).sum()
        if n_neg > 0:
            result[col] = result[col].clip(lower=0)
            print(f"  ✓ Clipped {n_neg} negative {col} values to 0")


# ══════════════════════════════════════════════════════════════════════════════
# 6. WEATHER IMPUTATION
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


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

result["year"]        = result["date"].dt.year
result["month"]       = result["date"].dt.month
result["day_of_year"] = result["date"].dt.dayofyear
result["quarter"]     = result["date"].dt.quarter
result["is_weekend"]  = (result["date"].dt.dayofweek >= 5).astype(int)
result["season"]      = result["month"].map({
    12:"winter", 1:"winter",  2:"winter",
     3:"spring", 4:"spring",  5:"spring",
     6:"summer", 7:"summer",  8:"summer",
     9:"autumn",10:"autumn", 11:"autumn",
})

# Price regime labels
result["regime"] = "normal"
result.loc[result["price_spain"] <= 20,  "regime"] = "collapse"
result.loc[result["price_spain"] >= 100, "regime"] = "spike"
result["is_collapse"] = (result["regime"] == "collapse").astype(int)

# Renewables penetration (wind + utility solar ÷ demand)
if all(c in result.columns for c in
       ["wind_gen_mwh_day", "solar_pv_mwh_day", "demand_mwh_day"]):
    result["renewables_penetration"] = (
        (result["wind_gen_mwh_day"].fillna(0) + result["solar_pv_mwh_day"].fillna(0))
        / result["demand_mwh_day"].replace(0, np.nan)
    ).clip(0, None)

# Self-consumption proxy (capacity × sunshine hours)
if "selfcons_pv_capacity_mw" in result.columns and "sunshine_hours" in result.columns:
    result["selfcons_theoretical_mwh"] = (
        result["selfcons_pv_capacity_mw"] * result["sunshine_hours"]
    )
    if "demand_mwh_day" in result.columns:
        result["selfcons_share_of_demand"] = (
            result["selfcons_theoretical_mwh"]
            / result["demand_mwh_day"].replace(0, np.nan)
        ).clip(0, None)

        mask_2025 = result["year"] == 2025
        if mask_2025.any():
            share = result.loc[mask_2025, "selfcons_share_of_demand"].mean()
            ok = 0.02 < share < 0.10
            print(f"\n  [VALIDATION] 2025 selfcons_share_of_demand: {share*100:.2f}%  "
                  f"{'ok consistent with REE ~4%' if ok else 'unexpected'}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY & SAVE
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
        print(f"    {col:<40} {n:>5} ({n/len(result)*100:.1f}%)")
else:
    print("    none ok")

print(f"\n  Regime distribution:")
print(result["regime"].value_counts().to_string())

# Key validation checks
print(f"\n  Key checks:")
dm = result["demand_mean_mw"].mean()
wm = result["wind_gen_mean_mw"].mean()
sm = result["solar_pv_mean_mw"].mean()
rp = result["renewables_penetration"].mean() * 100 if "renewables_penetration" in result.columns else 0
print(f"    demand_mean_mw    : {dm:>8,.0f} MW  {'ok' if 22000<dm<32000 else 'error'}")
print(f"    wind_gen_mean_mw  : {wm:>8,.0f} MW  {'ok' if 2000<wm<6000 else 'error'}")
print(f"    solar_pv_mean_mw  : {sm:>8,.0f} MW  {'ok (post-2019 only)' if sm>0 else 'error'}")
print(f"    renewables_pen    : {rp:>8.1f}%   {'ok' if 8<rp<50 else 'error'}")
print(f"    selfcons_max_mw   : {result['selfcons_pv_capacity_mw'].max():>8,.0f} MW  "
      f"{'ok' if result['selfcons_pv_capacity_mw'].max()>5000 else 'error'}")

result.to_csv("data/merged/merged_dataset.csv", index=False)
print(f"\n✓ Saved → data/merged/merged_dataset.csv")

# Blackout window
mask_bk = (result["date"] >= "2025-04-25") & (result["date"] <= "2025-05-01")
if mask_bk.any():
    print(f"\nBlackout window (Apr 25 – May 1, 2025):")
    cols = ["date", "price_spain", "regime", "demand_mean_mw",
            "wind_gen_mwh_day", "solar_pv_mwh_day",
            "renewables_penetration", "selfcons_pv_capacity_mw"]
    cols = [c for c in cols if c in result.columns]
    print(result[mask_bk][cols].to_string(index=False))

# Self-consumption trajectory
print(f"\nSelf-consumption PV capacity — national total (Dec each year):")
dec = result[result.date.dt.month == 12].groupby(result.date.dt.year)["selfcons_pv_capacity_mw"].last()
for yr, cap in dec.items():
    bar = "*" * int(cap / 500)
    print(f"  {yr}: {cap:>8,.0f} MW  {bar}")