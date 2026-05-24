"""
fetch_wind.py
=============
Pulls ESIOS indicator 10288 (Generación eólica peninsular, hourly MW).
geo_id=8741 (Península) only. Canarias (geo_id=8742) excluded.

Output: data/raw/esios_wind.csv
Columns: date, wind_mwh_day, wind_mean_mw, wind_peak_mw, wind_peak_hour
"""

import pandas as pd
from fetch_shared import download_indicator, clean_hourly

INDICATOR_ID = 10288

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
raw = download_indicator(INDICATOR_ID, "wind_gen")

# ── CLEAN ─────────────────────────────────────────────────────────────────────
df = clean_hourly(raw, "wind_gen")

# Geo filter: peninsula only
if "geo_id" in df.columns:
    print(f"\n  geo_ids: {sorted(df['geo_id'].unique())}  → keeping 8741 (Península)")
    df = df[df["geo_id"] == 8741].copy()

print(f"  Sample mean: {df['wind_gen'].mean():,.0f} MW  "
      f"(expected 3,000–5,000 MW)")

# ── DAILY AGGREGATION ─────────────────────────────────────────────────────────
df["_day"] = df["date"].dt.tz_convert(None).dt.normalize()

daily = (
    df.groupby("_day")["wind_gen"]
    .agg(wind_mean_mw="mean", wind_peak_mw="max")
    .reset_index()
    .rename(columns={"_day": "date"})
)
daily["wind_mwh_day"] = daily["wind_mean_mw"] * 24

def _peak_hour(g):
    idx = g["wind_gen"].idxmax()
    return g.loc[idx, "date"].tz_convert(None).hour

peak_h = (
    df.groupby("_day").apply(_peak_hour)
    .reset_index()
    .rename(columns={"_day": "date", 0: "wind_peak_hour"})
)
daily = daily.merge(peak_h, on="date", how="left")
daily = daily[["date","wind_mwh_day","wind_mean_mw","wind_peak_mw","wind_peak_hour"]]

# ── VALIDATE & SAVE ───────────────────────────────────────────────────────────
mean_mw = daily["wind_mean_mw"].mean()
ok = 2_000 < mean_mw < 8_000
print(f"\nWind daily: {len(daily)} days  mean={mean_mw:,.0f} MW  "
      f"{'✅ OK' if ok else '❌ CHECK'}")

daily.to_csv("data/raw/esios_wind.csv", index=False)
print("✓ Saved → data/raw/esios_wind.csv")