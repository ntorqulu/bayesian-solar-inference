"""
merge_datasets.py
=================
Merges all clean source files into a single daily dataset.

Key change vs previous version:
  Weather now comes from 17 AEMET stations (one per autonomous community).
  Sunshine hours S_t are computed as a capacity-weighted mean across stations,
  where the weight of each region = its share of national self-consumption PV
  capacity in that month from ESIOS indicator 1945.
  This gives a national irradiance proxy that accounts for where the panels
  actually are, rather than using a single Barcelona station as a proxy for all.

Inputs (data/raw/):
  esios_prices.csv              — indicator 600  : daily prices
  esios_demand.csv              — indicator 460  : daily demand (4x fix applied)
  esios_wind.csv                — indicator 10288: daily wind generation
  esios_solar.csv               — indicator 10358: daily utility solar PV
  aemet_all_stations.csv        — 17 AEMET stations, one row per (date, station)
  aemet_national.csv            — static-weighted national aggregate (fallback)
  esios_selfconsumption.csv     — indicator 1945 : monthly capacity by region
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

# CCAA → AEMET station mapping (must match pull_weather_data.py)
CCAA_TO_STATION = {
    "Andalucía": "5783",
    "Cataluña": "0076",
    "C. Valenciana": "8416Y",
    "Murcia": "7031",
    "Madrid": "3195",
    "Castilla-La Mancha": "4121",
    "Castilla y León": "2539",
    "Aragón": "9434",
    "Extremadura": "4452",
    "Canarias": "C029O",
    "País Vasco": "1024E",
    "Galicia": "1387",
    "Baleares": "B228",
    "Navarra": "9263D",
    "Asturias": "1249I",
    "Cantabria": "1109",
    "La Rioja": "9170",
}

# Map ESIOS geo_name values to our CCAA keys
# (ESIOS uses full official names; we normalise here)
ESIOS_TO_CCAA = {
    "Andalucía": "Andalucía",
    "Cataluña": "Cataluña",
    "Comunitat Valenciana": "C. Valenciana",
    "Región de Murcia": "Murcia",
    "Comunidad de Madrid": "Madrid",
    "Castilla - La Mancha": "Castilla-La Mancha",
    "Castilla y León": "Castilla y León",
    "Aragón": "Aragón",
    "Extremadura": "Extremadura",
    "Canarias": "Canarias",
    "País Vasco": "País Vasco",
    "Galicia": "Galicia",
    "Illes Balears": "Baleares",
    "Comunidad Foral de Navarra": "Navarra",
    "Principado de Asturias": "Asturias",
    "Cantabria": "Cantabria",
    "La Rioja": "La Rioja",
}


# ── HELPERS ───────────────────────────────────────────────────────────────────
def load(path, label):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_convert("Europe/Madrid").dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    print(
        f"  {label:<18}: {len(df):>6,} rows  "
        f"{df.date.min().date()} → {df.date.max().date()}"
    )
    return df


def impute_monthly_mean(series, dates):
    s = series.copy()
    monthly = s.groupby(dates.dt.to_period("M")).transform("mean")
    return s.fillna(monthly).fillna(s.mean())


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD
# ══════════════════════════════════════════════════════════════════════════════
print("Loading source datasets...")

prices = load("data/raw/esios_prices.csv", "Prices")
demand = load("data/raw/esios_demand.csv", "Demand")
wind = load("data/raw/esios_wind.csv", "Wind")
solar = load("data/raw/esios_solar.csv", "Solar PV")

# Weather: prefer multi-station file, fall back to single-station
try:
    weather_raw = pd.read_csv("data/raw/aemet_all_stations.csv")
    weather_raw["date"] = pd.to_datetime(weather_raw["date"]).dt.normalize()
    n_stations = weather_raw["station_id"].nunique()
    print(
        f"  {'Weather (stations)':<18}: {len(weather_raw):>6,} rows  "
        f"{weather_raw.date.min().date()} → {weather_raw.date.max().date()}  "
        f"({n_stations} stations)"
    )
    MULTI_STATION = True
except FileNotFoundError:
    print("  ⚠ aemet_all_stations.csv not found — falling back to aemet_national.csv")
    weather_raw = load("data/raw/aemet_national.csv", "Weather (national)")
    MULTI_STATION = False

# Self-consumption: raw regional data
sc_raw = pd.read_csv("data/raw/esios_selfconsumption.csv")
sc_raw["date"] = pd.to_datetime(sc_raw["date"], errors="coerce").dt.normalize()
sc_raw["value"] = pd.to_numeric(sc_raw["value"], errors="coerce")

# National total (for overall capacity column)
sc_national = (
    sc_raw[sc_raw["indicator_id"] == SELFCONS_PV_INDICATOR]
    .groupby("date")["value"]
    .sum()
    .reset_index()
    .rename(columns={"value": "selfcons_pv_capacity_mw"})
    .sort_values("date")
)
print(f"  {'Self-cons':<18}: {len(sc_national):>6,} monthly rows")
dec24 = (
    sc_national[sc_national.date.dt.year == 2024]
    .tail(1)["selfcons_pv_capacity_mw"]
    .values
)
dec25 = (
    sc_national[sc_national.date.dt.year == 2025]
    .tail(1)["selfcons_pv_capacity_mw"]
    .values
)
if dec24.size:
    print(f"    Dec 2024 national: {dec24[0]:,.0f} MW")
if dec25.size:
    print(f"    Dec 2025 national: {dec25[0]:,.0f} MW")

try:
    idae = pd.read_csv("data/raw/idae_selfconsumption_capacity.csv")
    HAS_IDAE = True
except FileNotFoundError:
    HAS_IDAE = False
    idae = None


# ══════════════════════════════════════════════════════════════════════════════
# 2. CAPACITY-WEIGHTED SUNSHINE HOURS
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing capacity-weighted sunshine hours...")

if MULTI_STATION:
    # --- Regional capacity: one row per (month, ccaa) ---
    sc_regional = sc_raw[sc_raw["indicator_id"] == SELFCONS_PV_INDICATOR].copy()
    sc_regional["ccaa"] = sc_regional["geo_name"].map(ESIOS_TO_CCAA)
    sc_regional = sc_regional.dropna(subset=["ccaa"])
    sc_regional = sc_regional.rename(columns={"value": "cap_mw"})
    sc_regional = sc_regional[["date", "ccaa", "cap_mw"]].copy()

    # Map CCAA → station_id
    sc_regional["station_id"] = sc_regional["ccaa"].map(CCAA_TO_STATION)

    # Merge regional capacity with daily sunshine per station
    sun = weather_raw[["date", "station_id", "sunshine_hours"]].copy()
    merged_sun = sun.merge(sc_regional, on=["station_id"], suffixes=("", "_cap"))

    # Align: use the capacity from the same month as the day
    merged_sun["month"] = merged_sun["date"].dt.to_period("M")
    merged_sun["cap_month"] = (
        merged_sun["date_cap"].dt.to_period("M")
        if "date_cap" in merged_sun.columns
        else merged_sun["month"]
    )

    # Simpler approach: for each day, join with the most recent monthly capacity
    # Build a day → monthly capacity lookup per station
    sc_daily_reg = []
    date_range = pd.DataFrame(
        {"date": pd.date_range(prices.date.min(), prices.date.max(), freq="D")}
    )

    for ccaa, station_id in CCAA_TO_STATION.items():
        sc_ccaa = sc_regional[sc_regional["ccaa"] == ccaa][["date", "cap_mw"]].copy()
        if sc_ccaa.empty:
            continue
        # Forward-fill monthly capacity to daily
        tmp = date_range.merge(sc_ccaa, on="date", how="left")
        tmp["cap_mw"] = tmp["cap_mw"].ffill().bfill().fillna(0)
        tmp["station_id"] = station_id
        tmp["ccaa"] = ccaa
        sc_daily_reg.append(tmp)

    sc_daily_reg = pd.concat(sc_daily_reg, ignore_index=True)

    # Join with daily sunshine
    sun_cap = sun.merge(sc_daily_reg, on=["date", "station_id"], how="left")

    # Capacity-weighted sunshine: S_t = Σ(cap_i * sun_i) / Σ(cap_i)
    def cap_weighted_sunshine(group):
        valid = group.dropna(subset=["sunshine_hours", "cap_mw"])
        if valid.empty or valid["cap_mw"].sum() == 0:
            return pd.Series({"sunshine_hours_weighted": np.nan, "n_stations_sun": 0})
        w = valid["cap_mw"] / valid["cap_mw"].sum()
        return pd.Series(
            {
                "sunshine_hours_weighted": (valid["sunshine_hours"] * w).sum(),
                "n_stations_sun": len(valid),
            }
        )

    sun_national = sun_cap.groupby("date").apply(cap_weighted_sunshine).reset_index()
    sun_national["date"] = pd.to_datetime(sun_national["date"]).dt.normalize()

    print(f"  Weighted sunshine computed for {len(sun_national):,} days")
    print(
        f"  Mean weighted sunshine: {sun_national['sunshine_hours_weighted'].mean():.2f} h"
    )
    print(f"  vs Barcelona alone would be: ~7.3 h")

    # Also compute simple (unweighted) national means for all other weather vars
    WEATHER_COLS = [
        "temp_mean_c",
        "temp_max_c",
        "temp_min_c",
        "sunshine_hours",
        "wind_speed_ms",
        "wind_gust_ms",
        "precip_mm",
        "humidity_mean_pct",
        "pressure_max_hpa",
        "pressure_min_hpa",
    ]
    weather_national = weather_raw.groupby("date")[WEATHER_COLS].mean().reset_index()
    weather_national["date"] = pd.to_datetime(weather_national["date"]).dt.normalize()

    # Merge weighted sunshine into national weather
    weather_national = weather_national.merge(
        sun_national[["date", "sunshine_hours_weighted", "n_stations_sun"]],
        on="date",
        how="left",
    )
    # Replace the unweighted sunshine with the capacity-weighted version
    weather_national["sunshine_hours_raw"] = weather_national["sunshine_hours"]
    weather_national["sunshine_hours"] = weather_national["sunshine_hours_weighted"]
    weather_national = weather_national.drop(columns=["sunshine_hours_weighted"])

else:
    # Fallback: single-station national file already has sunshine_hours
    weather_national = weather_raw.copy()
    weather_national["sunshine_hours_raw"] = weather_national["sunshine_hours"]
    weather_national["n_stations_sun"] = 1


# ══════════════════════════════════════════════════════════════════════════════
# 3. INTERPOLATE SELF-CONSUMPTION: MONTHLY → DAILY
# ══════════════════════════════════════════════════════════════════════════════
print("\nInterpolating self-consumption capacity to daily frequency...")

date_range = pd.DataFrame(
    {"date": pd.date_range(prices.date.min(), prices.date.max(), freq="D")}
)
sc_daily = date_range.merge(sc_national, on="date", how="left")
sc_daily["selfcons_pv_capacity_mw"] = (
    sc_daily["selfcons_pv_capacity_mw"].ffill().bfill().fillna(0)
)


# ══════════════════════════════════════════════════════════════════════════════
# 4. MERGE
# ══════════════════════════════════════════════════════════════════════════════
print("\nMerging datasets...")

demand = demand.rename(columns={c: c for c in demand.columns})
wind = wind.rename(
    columns={
        "wind_mwh_day": "wind_gen_mwh_day",
        "wind_mean_mw": "wind_gen_mean_mw",
        "wind_peak_mw": "wind_gen_peak_mw",
        "wind_peak_hour": "wind_gen_peak_hour",
    }
)
solar = solar.rename(
    columns={
        "solar_mwh_day": "solar_pv_mwh_day",
        "solar_mean_mw": "solar_pv_mean_mw",
        "solar_peak_mw": "solar_pv_peak_mw",
        "solar_peak_hour": "solar_pv_peak_hour",
    }
)

result = prices.copy()
result = result.merge(demand, on="date", how="left")
print("  Demand")
result = result.merge(wind, on="date", how="left")
print("  Wind")
result = result.merge(solar, on="date", how="left")
print("  Solar PV")
result = result.merge(weather_national, on="date", how="left")
print("  Weather (17-station weighted)")
result = result.merge(sc_daily, on="date", how="left")
print("  Self-consumption")

if HAS_IDAE:
    idae_map = idae.set_index("year")["cumulative_capacity_mw"].to_dict()
    result["idae_capacity_mw"] = result["date"].dt.year.map(idae_map)
    print("  IDAE validation column")


# ══════════════════════════════════════════════════════════════════════════════
# 5. DATA QUALITY FIXES
# ══════════════════════════════════════════════════════════════════════════════
print("\nData quality fixes...")

for col in ["solar_pv_mwh_day", "solar_pv_mean_mw"]:
    if col in result.columns:
        n = (result[col] < 0).sum()
        if n > 0:
            result[col] = result[col].clip(lower=0)
            print(f"  Clipped {n} negative {col} values")


# ══════════════════════════════════════════════════════════════════════════════
# 6. WEATHER IMPUTATION
# ══════════════════════════════════════════════════════════════════════════════
print("\nImputing weather gaps...")

WEATHER_IMPUTE = [
    "temp_mean_c",
    "temp_max_c",
    "temp_min_c",
    "sunshine_hours",
    "wind_speed_ms",
    "wind_gust_ms",
    "precip_mm",
    "humidity_mean_pct",
    "pressure_max_hpa",
    "pressure_min_hpa",
]
for col in WEATHER_IMPUTE:
    if col not in result.columns:
        continue
    n_before = result[col].isna().sum()
    if n_before > 0:
        result[col] = impute_monthly_mean(result[col], result["date"])
        print(f"  {col:<25} {n_before:>4} NaN → {result[col].isna().sum():>4} NaN")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\nEngineering features...")

result["year"] = result["date"].dt.year
result["month"] = result["date"].dt.month
result["day_of_year"] = result["date"].dt.dayofyear
result["quarter"] = result["date"].dt.quarter
result["is_weekend"] = (result["date"].dt.dayofweek >= 5).astype(int)
result["season"] = result["month"].map(
    {
        12: "winter",
        1: "winter",
        2: "winter",
        3: "spring",
        4: "spring",
        5: "spring",
        6: "summer",
        7: "summer",
        8: "summer",
        9: "autumn",
        10: "autumn",
        11: "autumn",
    }
)

result["regime"] = "normal"
result.loc[result["price_spain"] <= 20, "regime"] = "collapse"
result.loc[result["price_spain"] >= 100, "regime"] = "spike"
result["is_collapse"] = (result["regime"] == "collapse").astype(int)

if all(
    c in result.columns
    for c in ["wind_gen_mwh_day", "solar_pv_mwh_day", "demand_mwh_day"]
):
    result["renewables_penetration"] = (
        (result["wind_gen_mwh_day"].fillna(0) + result["solar_pv_mwh_day"].fillna(0))
        / result["demand_mwh_day"].replace(0, np.nan)
    ).clip(0, None)

# Hidden solar proxy H_t = C_t × S_t (capacity-weighted sunshine)
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
            ok = 0.02 < share < 0.12
            print(
                f"\n  [VALIDATION] 2025 selfcons_share_of_demand: {share*100:.2f}%  "
                f"{'ok consistent with REE ~4%' if ok else '⚠ unexpected'}"
            )


# ══════════════════════════════════════════════════════════════════════════════
# 8. SUMMARY & SAVE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("FINAL MERGED DATASET")
print("=" * 65)
print(f"  Rows      : {len(result):,}")
print(f"  Dates     : {result.date.min().date()} → {result.date.max().date()}")
print(f"  Columns   : {len(result.columns)}")

miss = result.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss):
    print(f"\n  Missing values (>0 NaN):")
    for col, n in miss.items():
        print(f"    {col:<40} {n:>5} ({n/len(result)*100:.1f}%)")

print(f"\n  Key checks:")
dm = result["demand_mean_mw"].mean()
wm = result["wind_gen_mean_mw"].mean()
sm = result["solar_pv_mean_mw"].mean()
print(f"    demand_mean_mw    : {dm:>8,.0f} MW  {'ok' if 22000<dm<32000 else 'error'}")
print(f"    wind_gen_mean_mw  : {wm:>8,.0f} MW  {'ok' if 2000<wm<6000 else 'error'}")
print(f"    solar_pv_mean_mw  : {sm:>8,.0f} MW  {'ok' if sm>0 else 'error'}")
print(
    f"    sunshine_hours    : {result['sunshine_hours'].mean():>8.2f} h   "
    f"(capacity-weighted across {n_stations if MULTI_STATION else 1} stations)"
)
print(
    f"    selfcons_max_mw   : {result['selfcons_pv_capacity_mw'].max():>8,.0f} MW  "
    f"{'ok' if result['selfcons_pv_capacity_mw'].max()>5000 else 'error'}"
)

result.to_csv("data/merged/merged_dataset.csv", index=False)
print(f"\nSaved → data/merged/merged_dataset.csv")

# Blackout window
mask_bk = (result["date"] >= "2025-04-25") & (result["date"] <= "2025-05-01")
if mask_bk.any():
    print(f"\nBlackout window (Apr 25–May 1 2025):")
    cols = [
        "date",
        "price_spain",
        "regime",
        "demand_mean_mw",
        "sunshine_hours",
        "renewables_penetration",
        "selfcons_pv_capacity_mw",
    ]
    cols = [c for c in cols if c in result.columns]
    print(result[mask_bk][cols].to_string(index=False))

print(f"\nSelf-consumption PV capacity (Dec each year):")
dec = (
    result[result.date.dt.month == 12]
    .groupby(result.date.dt.year)["selfcons_pv_capacity_mw"]
    .last()
)
for yr, cap in dec.items():
    bar = "*" * int(cap / 500)
    print(f"  {yr}: {cap:>8,.0f} MW  {bar}")
