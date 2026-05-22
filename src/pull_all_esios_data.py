import requests
import pandas as pd
import time
import os
from pathlib import Path
from datetime import timedelta

def load_env_file(env_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a local .env file if present."""
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(Path(__file__).resolve().parents[1] / ".env")

# ── CONFIG ────────────────────────────────────────────────────────────────────
TOKEN = os.getenv("ESIOS_TOKEN")
if not TOKEN:
    raise EnvironmentError(
        "ESIOS_TOKEN environment variable is not set. "
        "Export it before running: export ESIOS_TOKEN='your_token_here'"
    )

BASE_URL = "https://api.esios.ree.es"
HEADERS = {
    "Accept": "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "Host": "api.esios.ree.es",
    "x-api-key": TOKEN,
}

START = "2015-01-01"
END = "2026-04-30"

# ── INDICATORS ────────────────────────────────────────────────────────────────
# 600   = European day-ahead electricity market prices (España/Portugal/Francia)
# 469   = Demanda real peninsular por provincias (MW, real-time)
#         Returns ~17 provincial series that must be SUMMED to get total demand.
#         DO NOT use 1293 (Demanda prevista D-1): broken unit, ~8x inflated.
# 10288 = Wind generation peninsular (MW) single geo_id=8741 series
# 10289 = Solar PV generation peninsular (MW) single geo_id=8741 series

INDICATORS = {
    "price":    600,    # Prices (hourly, €/MWh)
    "demand":   469,    # Real peninsular demand by province (hourly, MW) NOT 1293
    "wind_gen": 10288,  # Wind generation (hourly, MW)
    "solar_pv": 10289,  # Solar PV generation (hourly, MW)
}

# Ensure output directory exists
os.makedirs("data/raw", exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def generate_monthly_chunks(start, end):
    """Generate date ranges for monthly API requests."""
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    chunks = []
    current = start
    while current <= end:
        next_month = current + pd.offsets.MonthBegin(1)
        chunk_end = min(next_month - timedelta(days=1), end)
        chunks.append((
            current.strftime("%Y-%m-%dT00:00:00"),
            chunk_end.strftime("%Y-%m-%dT23:59:59"),
        ))
        current = next_month
    return chunks


session = requests.Session()


def fetch_chunk(indicator_id, start_date, end_date, max_retries=5):
    """Fetch data chunk from ESIOS API with retry logic."""
    url = f"{BASE_URL}/indicators/{indicator_id}"
    params = {"start_date": start_date, "end_date": end_date, "time_trunc": "hour"}

    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=HEADERS, params=params, timeout=120)
            r.raise_for_status()
            values = r.json()["indicator"]["values"]
            return pd.DataFrame(values)
        except requests.exceptions.ReadTimeout:
            wait = 20 * (attempt + 1)
            print(f"    timeout — retry in {wait}s ({attempt + 1}/{max_retries})")
            time.sleep(wait)
        except requests.exceptions.HTTPError as e:
            print(f"    HTTP ERROR: {e}")
            return pd.DataFrame()
        except Exception as e:
            print(f"    ERROR: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════════════
# 1. DOWNLOAD ALL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ESIOS Data Puller — Starting download")
print("=" * 70)

chunks = generate_monthly_chunks(START, END)
dfs = {}

for name, indicator_id in INDICATORS.items():
    print(f"\n[{name.upper()}] Downloading indicator {indicator_id}")
    parts = []
    for start_date, end_date in chunks:
        print(f"  {start_date[:10]} → {end_date[:10]}", end="  ", flush=True)
        df_chunk = fetch_chunk(indicator_id, start_date, end_date)
        print(f"{len(df_chunk):>5} rows")
        if not df_chunk.empty:
            parts.append(df_chunk)
        time.sleep(1)

    if not parts:
        print(f"  ⚠ FAILED: no data retrieved for {name}")
        continue

    dfs[name] = pd.concat(parts, ignore_index=True)
    print(f"  TOTAL {name}: {len(dfs[name]):,} rows")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PROCESS PRICES
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PROCESSING PRICES")
print("=" * 70)

price_df = dfs["price"].copy()
price_df = price_df.rename(columns={"datetime": "date", "value": "price_eur_mwh"})
price_df["date"] = pd.to_datetime(price_df["date"], utc=True).dt.tz_convert("Europe/Madrid")

price_df = price_df[price_df["geo_name"].isin(["España", "Portugal", "Francia"])].copy()
price_df = price_df[price_df["date"].dt.minute == 0].copy()
price_df = price_df.sort_values(["geo_name", "date"]).drop_duplicates(
    subset=["geo_name", "date"], keep="first"
)

price_df["day"] = price_df["date"].dt.floor("D")

# Require ≥23 hours of coverage per day (handles DST transitions)
coverage = price_df.groupby(["day", "geo_name"]).size().reset_index(name="n_hours")
valid_days = coverage[coverage["n_hours"] >= 23][["day", "geo_name"]]
price_df = price_df.merge(valid_days, on=["day", "geo_name"], how="inner")

daily_prices = (
    price_df.groupby(["day", "geo_name"])["price_eur_mwh"]
    .mean()
    .reset_index()
    .pivot(index="day", columns="geo_name", values="price_eur_mwh")
    .rename(columns={
        "España":   "price_spain",
        "Portugal": "price_portugal",
        "Francia":  "price_france",
    })
    .sort_index()
)

daily_prices["spread_fr_es"] = daily_prices["price_france"] - daily_prices["price_spain"]
daily_prices["spread_pt_es"] = daily_prices["price_portugal"] - daily_prices["price_spain"]

# Drop timezone before reset_index so the date column saves cleanly to CSV
daily_prices.index = daily_prices.index.tz_localize(None)
daily_prices = daily_prices.reset_index().rename(columns={"day": "date"})

# Do NOT ffill — keep NaN for missing days so data gaps are visible in the merge
print(f"\nPrice data: {len(daily_prices)} days")
print(f"Missing price days: {daily_prices['price_spain'].isna().sum()}")
print("\nPrice sample:")
print(daily_prices.head(3).to_string())

daily_prices.to_csv("data/raw/esios_prices.csv", index=False)
print("Saved → data/raw/esios_prices.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PROCESS GENERATION (Demand + Wind + Solar)
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("PROCESSING GENERATION DATA")
print("=" * 70)


def process_generation(raw_df, value_col, date_col="datetime"):
    """
    Process hourly national generation/demand indicators into daily aggregates.
    Returns: date, {col}_mwh_day, {col}_mean_mw, {col}_peak_mw, {col}_peak_hour.

    INDICATOR BEHAVIOUR (confirmed from diagnostic output):
    ─────────────────────────────────────────────────────
    • 469  (demand): returns ~17 provincial series (geo_id 23,26,32,...).
                     Must SUM across all provinces per timestamp to get
                     total peninsular demand. No geo filter needed — all
                     returned series are peninsular provinces.
    • 10288 (wind):  returns geo_id=8741 (Península) + geo_id=8742 (Canarias).
                     Filter to geo_id=8741 only.
    • 10289 (solar): returns geo_id=8741 (Península) + geo_id=8743 (Baleares).
                     Filter to geo_id=8741 only.

    UNIT: All indicators return instantaneous MW.
    Daily MWh = mean_hourly_MW × 24h.
    """
    df = raw_df.copy()
    df = df.rename(columns={date_col: "date", "value": value_col})
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert("Europe/Madrid")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    df = df[df["date"].dt.minute == 0].copy()

    # ── GEO FILTER ────────────────────────────────────────────────────────────
    if "geo_id" in df.columns:
        island_ids = {8742, 8743}
        has_peninsula = (df["geo_id"] == 8741).any()

        if has_peninsula:
            # Wind/solar: keep Península only
            df = df[df["geo_id"] == 8741].copy()
        else:
            # Demand: sum all provincial series, exclude islands
            df = df[~df["geo_id"].isin(island_ids)].copy()
    elif "geo_name" in df.columns:
        island_names = {"Canarias", "Baleares", "Ceuta", "Melilla"}
        df = df[~df["geo_name"].isin(island_names)].copy()

    if len(df) == 0:
        raise ValueError(
            f"process_generation: no rows for '{value_col}' after geo filter."
        )

    # ── Sum across geo series per timestamp (critical for demand) ─────────────
    df["day"] = df["date"].dt.tz_localize(None).dt.normalize()

    hourly = (
        df.groupby("date")[value_col]
        .sum()
        .reset_index()
    )
    hourly["day"] = hourly["date"].dt.tz_localize(None).dt.normalize()

    # ── Daily aggregation ─────────────────────────────────────────────────────
    daily = (
        hourly.groupby("day")[value_col]
        .agg(**{
            f"{value_col}_mean_mw": "mean",
            f"{value_col}_peak_mw": "max",
        })
        .reset_index()
        .rename(columns={"day": "date"})
    )
    # MW × 24h = MWh/day
    daily[f"{value_col}_mwh_day"] = daily[f"{value_col}_mean_mw"] * 24

    # ── Peak hour ─────────────────────────────────────────────────────────────
    # groupby("day").apply().reset_index() produces a "day" column, not "date".
    # Rename it so the merge key matches daily["date"].
    peak_hours = (
        hourly.groupby("day")
        .apply(lambda g: g.loc[g[value_col].idxmax(), "date"].tz_localize(None).hour)
        .reset_index()
        .rename(columns={"day": "date", 0: f"{value_col}_peak_hour"})
    )
    daily = daily.merge(peak_hours, on="date", how="left")

    cols = [
        "date",
        f"{value_col}_mwh_day",
        f"{value_col}_mean_mw",
        f"{value_col}_peak_mw",
        f"{value_col}_peak_hour",
    ]
    return daily[cols]


# ── GEO DIAGNOSTIC ────────────────────────────────────────────────────────────
print("Geo series available per indicator (confirms routing logic):")
for name in ["demand", "wind_gen", "solar_pv"]:
    raw = dfs[name].rename(columns={"datetime": "date", "value": "val"})
    raw["val"] = pd.to_numeric(raw["val"], errors="coerce")
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert("Europe/Madrid")
    raw = raw[raw["date"].dt.minute == 0]
    geo_cols = [c for c in ["geo_id", "geo_name"] if c in raw.columns]
    if geo_cols:
        grp = raw.groupby(geo_cols)["val"].agg(["count", "mean"]).reset_index()
        print(f"\n  [{name}]  expected sum ~25k MW for demand, ~3–8k MW for wind/solar")
        print(grp.to_string(index=False))
        print(f"  Sum of all series: {grp['mean'].sum():,.0f} MW")
    else:
        print(f"\n  [{name}] no geo columns found")
print()

# Process each generation component
demand_daily = process_generation(dfs["demand"], "demand")
print(
    f"Demand:          {len(demand_daily)} days  "
    f"mean={demand_daily['demand_mean_mw'].mean():,.0f} MW  "
    f"mwh/day={demand_daily['demand_mwh_day'].mean():,.0f}"
)

wind_daily = process_generation(dfs["wind_gen"], "wind_gen")
print(f"Wind generation: {len(wind_daily)} days")

solar_daily = process_generation(dfs["solar_pv"], "solar_pv")
print(f"Solar PV:        {len(solar_daily)} days")

# Merge all generation
result = demand_daily.copy()
for df_extra in [wind_daily, solar_daily]:
    df_extra["date"] = pd.to_datetime(df_extra["date"]).dt.normalize()
    result = result.merge(df_extra, on="date", how="left")

result["date"] = pd.to_datetime(result["date"]).dt.normalize()

# ── Penetration ratios ────────────────────────────────────────────────────────
if "demand_mwh_day" in result.columns:
    if "wind_gen_mwh_day" in result.columns:
        result["wind_penetration"] = (
            result["wind_gen_mwh_day"] / result["demand_mwh_day"]
        ).clip(0, None)

    if "solar_pv_mwh_day" in result.columns:
        result["solar_penetration"] = (
            result["solar_pv_mwh_day"] / result["demand_mwh_day"]
        ).clip(0, None)

if "wind_penetration" in result.columns and "solar_penetration" in result.columns:
    result["renewables_penetration"] = (
        result["wind_penetration"] + result["solar_penetration"]
    )

print("\nGeneration data sample:")
print(result.head(3).to_string())

result.to_csv("data/raw/esios_generation.csv", index=False)
print("Saved → data/raw/esios_generation.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. SUMMARY + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("DOWNLOAD SUMMARY")
print("=" * 70)

print(f"\nesios_prices.csv")
print(f"  Rows    : {len(daily_prices)}")
print(f"  Columns : {', '.join(daily_prices.columns.tolist())}")
print(f"  Dates   : {daily_prices.date.min().date()} → {daily_prices.date.max().date()}")

print(f"\nesios_generation.csv")
print(f"  Rows    : {len(result)}")
print(f"  Columns : {', '.join(result.columns.tolist())}")
print(f"  Dates   : {result.date.min().date()} → {result.date.max().date()}")

# Sanity check: Apr 25 – May 1, 2025 (blackout window)
print("\n" + "=" * 70)
print("SANITY CHECK: April 25 – May 1, 2025 (Blackout window)")
print("=" * 70)

mask_p = (daily_prices.date >= "2025-04-25") & (daily_prices.date <= "2025-05-01")
mask_g = (result.date >= "2025-04-25") & (result.date <= "2025-05-01")

print("\nPrices:")
print(daily_prices[mask_p][["date", "price_spain", "price_france", "spread_fr_es"]].to_string(index=False))

print("\nGeneration:")
print(result[mask_g][[
    "date", "demand_mwh_day", "wind_gen_mwh_day",
    "solar_pv_mwh_day", "renewables_penetration",
]].to_string(index=False))

# Self-validating checks
print("\n" + "=" * 70)
print("VALIDATION — expected vs actual values")
print("=" * 70)
demand_mean    = result["demand_mwh_day"].mean()
demand_mean_mw = result["demand_mean_mw"].mean()
ren_mean       = result["renewables_penetration"].mean() * 100
demand_ok  = 550_000 < demand_mean    < 800_000   # peninsular ~600–720k MWh/day
mw_ok      = 22_000  < demand_mean_mw < 32_000    # peninsular ~25–28k MW avg
ren_ok     = 8       < ren_mean       < 50         # wind+solar 2015–2026 avg ~15–35%

print(f"\n  demand_mwh_day mean : {demand_mean:>10,.0f}  {'✅ OK' if demand_ok else '❌ WRONG'}")
print(f"  demand_mean_mw mean : {demand_mean_mw:>10,.0f}  {'✅ OK' if mw_ok     else '❌ WRONG'}")
print(f"  renewables_pen mean : {ren_mean:>9.1f}%  {'✅ OK' if ren_ok    else '❌ WRONG'}")

print("\n" + "=" * 70)
print("COMPLETE")
print("=" * 70)