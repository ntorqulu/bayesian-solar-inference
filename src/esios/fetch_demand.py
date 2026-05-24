"""
fetch_demand.py
===============
Pulls ESIOS indicator 460 (Demanda real peninsular, hourly MW).

Format change: indicator 460 switched from instantaneous MW to 4x-inflated values
MWh-since-midnight on 2022-05-24. The script detects and corrects this
on 2022-05-24. Fixed by dividing post-boundary values by 4 (confirmed via diagnostic).

Output: data/raw/esios_demand.csv
Columns: date (day), demand_mwh_day, demand_mean_mw, demand_peak_mw, demand_peak_hour
"""

import pandas as pd, numpy as np
from fetch_shared import download_indicator, clean_hourly

INDICATOR_ID = 460
BOUNDARY     = pd.Timestamp("2022-05-24", tz="Europe/Madrid")
PEAK_MW      = 45_000   # physical ceiling for peninsular demand

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
raw = download_indicator(INDICATOR_ID, "demand")

# ── CLEAN ─────────────────────────────────────────────────────────────────────
df = clean_hourly(raw, "demand")

# Indicator 460 has geo_id=8741 (Península) only — no geo filter needed
# Verify:
geo_ids = df["geo_id"].unique() if "geo_id" in df.columns else []
print(f"\n  geo_ids present: {sorted(geo_ids)}")
print(f"  Raw sample — first 3 rows:\n{df.head(3).to_string()}")
print(f"  Pre-boundary mean:  {df[df['date'] <  BOUNDARY]['demand'].mean():>10,.0f} MW")
print(f"  Post-boundary mean: {df[df['date'] >= BOUNDARY]['demand'].mean():>10,.0f} MW  "
      f"(raw cumulative — will be corrected)")

# ── CUMULATIVE → INSTANTANEOUS (post-boundary only) ───────────────────────────
pre  = df[df["date"] <  BOUNDARY].copy()
post = df[df["date"] >= BOUNDARY].copy().sort_values("date")

# ── FORMAT: values are 4× real instantaneous MW ─────────────────────────
# Confirmed by deep-dive diagnostic (00_pull_diagnostics.ipynb):
# Every post-boundary value is exactly 4× real demand.
# Proof: hour 13 clean = 30,042 MW; hour 14 inflated = 120,112 ÷ 4 = 30,028 MW.
# Apply ÷4 only to rows where value exceeds the physical maximum (>45,000 MW).
# This handles 2022-05-24 correctly: hours 0-13 are already clean (~22-30k MW)
# and hours 14-23 are 4× inflated (~120k MW). The threshold distinguishes them.
post2 = post.copy()
inflated = post2["demand"] > PEAK_MW
post2.loc[inflated, "demand"] = post2.loc[inflated, "demand"] / 4
n_div = inflated.sum()
print(f"  Divided {n_div:,} inflated rows by 4 (>{PEAK_MW:,} MW threshold)")

df_clean = pd.concat([pre, post2], ignore_index=True).sort_values("date")
print(f"  Post-boundary mean after conversion: "
      f"{df_clean[df_clean['date'] >= BOUNDARY]['demand'].mean():,.0f} MW")

# Quick validation
pre_mean  = df_clean[df_clean["date"] <  BOUNDARY]["demand"].mean()
post_mean = df_clean[df_clean["date"] >= BOUNDARY]["demand"].mean()
ok = 18_000 < pre_mean < 35_000 and 18_000 < post_mean < 35_000
print(f"  Pre-boundary mean:  {pre_mean:,.0f} MW  {'✅' if 18000<pre_mean<35000 else '❌'}")
print(f"  Post-boundary mean: {post_mean:,.0f} MW  {'✅' if 18000<post_mean<35000 else '❌'}")

# ── DAILY AGGREGATION ─────────────────────────────────────────────────────────
df_clean["_day"] = df_clean["date"].dt.tz_convert(None).dt.normalize()

daily = (
    df_clean.groupby("_day")["demand"]
    .agg(demand_mean_mw="mean", demand_peak_mw="max")
    .reset_index()
    .rename(columns={"_day": "date"})
)
daily["demand_mwh_day"] = daily["demand_mean_mw"] * 24

def _peak_hour(g):
    idx = g["demand"].idxmax()
    return g.loc[idx, "date"].tz_convert(None).hour

peak_h = (
    df_clean.groupby("_day")
    .apply(_peak_hour)
    .reset_index()
    .rename(columns={"_day": "date", 0: "demand_peak_hour"})
)
daily = daily.merge(peak_h, on="date", how="left")
daily = daily[["date","demand_mwh_day","demand_mean_mw","demand_peak_mw","demand_peak_hour"]]

# ── VALIDATE & SAVE ───────────────────────────────────────────────────────────
mean_mw  = daily["demand_mean_mw"].mean()
n_unique = daily["demand_mwh_day"].nunique()
ok_mean  = 22_000 < mean_mw  < 32_000
ok_uniq  = n_unique > 100

print(f"\nDemand daily: {len(daily)} days  "
      f"[{daily.date.min().date()} → {daily.date.max().date()}]")
print(f"  mean={mean_mw:,.0f} MW  unique_vals={n_unique}  "
      f"{'✅ OK' if ok_mean and ok_uniq else '❌ CHECK'}")
print(f"\nBy year:")
daily["yr"] = daily["date"].dt.year
for yr, grp in daily.groupby("yr"):
    m = grp["demand_mean_mw"].mean()
    flag = "✅" if 18000 < m < 35000 else "❌"
    print(f"  {yr}: {m:>8,.0f} MW  {flag}")
daily = daily.drop(columns=["yr"])

daily.to_csv("data/raw/esios_demand.csv", index=False)
print("\n✓ Saved → data/raw/esios_demand.csv")