"""
fetch_prices.py
===============
Pulls ESIOS indicator 600 (day-ahead spot prices: Spain, France, Portugal).
Output: data/raw/esios_prices.csv

Columns: date, price_spain, price_france, price_portugal, spread_fr_es, spread_pt_es
"""

import pandas as pd
from fetch_shared import download_indicator

# ── DOWNLOAD ──────────────────────────────────────────────────────────────────
raw = download_indicator(600, "price")

# ── PROCESS ───────────────────────────────────────────────────────────────────
df = raw.rename(columns={"datetime": "date", "value": "price_eur_mwh"})
df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Madrid")

# Keep Spain, France, Portugal only; on-the-hour only
df = df[df["geo_name"].isin(["España", "Portugal", "Francia"])].copy()
df = df[df["date"].dt.minute == 0].copy()
df = df.sort_values(["geo_name", "date"]).drop_duplicates(
    subset=["geo_name", "date"], keep="first"
)
df["day"] = df["date"].dt.floor("D")

# Require ≥23 h coverage (DST-safe)
coverage  = df.groupby(["day", "geo_name"]).size().reset_index(name="n")
valid     = coverage[coverage["n"] >= 23][["day", "geo_name"]]
df        = df.merge(valid, on=["day", "geo_name"], how="inner")

daily = (
    df.groupby(["day", "geo_name"])["price_eur_mwh"]
    .mean()
    .reset_index()
    .pivot(index="day", columns="geo_name", values="price_eur_mwh")
    .rename(columns={"España": "price_spain", "Portugal": "price_portugal",
                     "Francia": "price_france"})
    .sort_index()
)
daily["spread_fr_es"] = daily["price_france"] - daily["price_spain"]
daily["spread_pt_es"] = daily["price_portugal"] - daily["price_spain"]
daily.index = daily.index.tz_localize(None)
daily = daily.reset_index().rename(columns={"day": "date"})

# ── VALIDATE & SAVE ───────────────────────────────────────────────────────────
print(f"\nPrices: {len(daily)} days  "
      f"[{daily.date.min().date()} → {daily.date.max().date()}]")
print(f"  Spain  mean={daily.price_spain.mean():.1f} €/MWh  "
      f"min={daily.price_spain.min():.1f}  max={daily.price_spain.max():.1f}")
print(f"  Missing days: {daily.price_spain.isna().sum()}  ok" 
      if daily.price_spain.isna().sum() == 0 else "  missing days exist")

daily.to_csv("data/raw/esios_prices.csv", index=False)
print("Saved → data/raw/esios_prices.csv")