"""
fetch_solar.py
==============
Pulls ESIOS indicator 10358 (Generación solar fotovoltaica peninsular, hourly MW).
geo_id=8741 (Península) only. Available from January 2019 onward.
Pre-2019 rows will be NaN in the output — expected and documented.

Output: data/raw/esios_solar.csv
Columns: date, solar_mwh_day, solar_mean_mw, solar_peak_mw, solar_peak_hour

NOTE: Do NOT use indicator 10289 (reversed/decreasing trend from 2023).
      Do NOT use indicator 1159 (403 Forbidden with standard token).
"""

import pandas as pd
from fetch_shared import download_indicator, clean_hourly

INDICATOR_ID = 10358    # ← verified: geo_id=8741, Jan 2019+, growing trend ✅

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
raw = download_indicator(INDICATOR_ID, "solar_pv")

# ── CLEAN ─────────────────────────────────────────────────────────────────────
df = clean_hourly(raw, "solar_pv")

if "geo_id" in df.columns:
    print(f"\n  geo_ids: {sorted(df['geo_id'].unique())}  → keeping 8741 (Península)")
    df = df[df["geo_id"] == 8741].copy()

print(f"  Rows with data: {len(df):,}  "
      f"({df['date'].min().date()} → {df['date'].max().date()})")

solar_2022 = df[df["date"].dt.year == 2022]["solar_pv"].mean()
solar_2024 = df[df["date"].dt.year == 2024]["solar_pv"].mean()
print(f"  2022 mean: {solar_2022:,.0f} MW  |  2024 mean: {solar_2024:,.0f} MW  "
      f"{'✅ growing' if solar_2024 > solar_2022 > 0 else '❌ NOT growing — check indicator'}")

# ── DAILY AGGREGATION ─────────────────────────────────────────────────────────
df["_day"] = df["date"].dt.tz_convert(None).dt.normalize()

daily = (
    df.groupby("_day")["solar_pv"]
    .agg(solar_mean_mw="mean", solar_peak_mw="max")
    .reset_index()
    .rename(columns={"_day": "date"})
)
daily["solar_mwh_day"] = daily["solar_mean_mw"] * 24

def _peak_hour(g):
    idx = g["solar_pv"].idxmax()
    return g.loc[idx, "date"].tz_convert(None).hour

peak_h = (
    df.groupby("_day").apply(_peak_hour)
    .reset_index()
    .rename(columns={"_day": "date", 0: "solar_peak_hour"})
)
daily = daily.merge(peak_h, on="date", how="left")

# Reindex to full date range so pre-2019 days appear as NaN
full_range = pd.DataFrame({
    "date": pd.date_range("2015-01-01", "2026-04-30", freq="D").normalize()
})
daily = full_range.merge(daily, on="date", how="left")
daily = daily[["date","solar_mwh_day","solar_mean_mw","solar_peak_mw","solar_peak_hour"]]

# ── VALIDATE & SAVE ───────────────────────────────────────────────────────────
n_nan    = daily["solar_mwh_day"].isna().sum()
n_data   = daily["solar_mwh_day"].notna().sum()
ok_grow  = solar_2024 > solar_2022 > 0
print(f"\nSolar daily: {n_data} days with data, {n_nan} NaN (pre-2019, expected)  "
      f"{'✅ OK' if ok_grow else '❌ CHECK indicator'}")
print(f"  Peak hour distribution (should cluster 12–14h):")
print(f"  {daily['solar_peak_hour'].value_counts().sort_index().to_dict()}")

daily.to_csv("data/raw/esios_solar.csv", index=False)
print("✓ Saved → data/raw/esios_solar.csv")