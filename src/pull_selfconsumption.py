"""
pull_selfconsumption.py
=======================
Step 1: Discovers ESIOS indicator IDs for self-consumption (autoconsumo) data
         published by REE in December 2025.
Step 2: Pulls monthly generation and installed capacity data.
Step 3: Saves to data/raw/esios_selfconsumption.csv

Run from project root:
    python src/pull_selfconsumption.py

Output columns:
    date (month start), selfcons_gen_gwh, selfcons_capacity_mw, source
"""

import os
import time
import requests
import pandas as pd
from pathlib import Path

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
TOKEN    = os.getenv("ESIOS_TOKEN")
if not TOKEN:
    raise EnvironmentError(
        "ESIOS_TOKEN not set. Run: export ESIOS_TOKEN='your_token'"
    )

BASE_URL = "https://api.esios.ree.es"
HEADERS  = {
    "Accept":       "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "Host":         "api.esios.ree.es",
    "x-api-key":    TOKEN,
}

OUT_DIR  = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── STEP 1: DISCOVER INDICATOR IDs ───────────────────────────────────────────
# The self-consumption data was published December 2025.
# We search for "autoconsumo" to find the relevant indicators.

print("=" * 60)
print("STEP 1 — Searching ESIOS for autoconsumo indicators")
print("=" * 60)

search_url = f"{BASE_URL}/indicators"
params = {"text": "autoconsumo", "lang": "es"}

r = requests.get(search_url, headers=HEADERS, params=params, timeout=30)
r.raise_for_status()
results = r.json().get("indicators", [])

print(f"\nFound {len(results)} indicators matching 'autoconsumo':\n")
for ind in results:
    print(f"  ID={ind['id']:>6}  {ind['name'][:80]}")

print()

# Also search for "fotovoltaico" to catch any differently named indicators
params2 = {"text": "fotovoltaico autoconsumo", "lang": "es"}
r2 = requests.get(search_url, headers=HEADERS, params=params2, timeout=30)
r2.raise_for_status()
results2 = r2.json().get("indicators", [])

# Merge and deduplicate
all_results = {ind["id"]: ind for ind in results + results2}
print(f"Total unique indicators (autoconsumo + fotovoltaico): {len(all_results)}\n")

# ── STEP 2: IDENTIFY THE RIGHT INDICATORS ────────────────────────────────────
# Based on REE announcement (Dec 2025), we expect:
#   - Monthly generation of self-consumption (GWh)
#   - Installed capacity of self-consumption (MW)
#   - Possibly broken down by technology (PV, wind, etc.)
#
# We'll try to fetch a small sample from each candidate to confirm units and coverage.

print("=" * 60)
print("STEP 2 — Probing candidate indicators")
print("=" * 60)

TEST_START = "2025-01-01T00:00:00"
TEST_END   = "2025-03-31T23:59:59"

candidate_ids = []
for ind_id, ind in all_results.items():
    name_lower = ind["name"].lower()
    # Focus on generation and capacity indicators, skip price indicators
    if any(kw in name_lower for kw in ["generaci", "potencia", "energía", "energia"]):
        if "precio" not in name_lower and "coste" not in name_lower:
            candidate_ids.append(ind_id)

print(f"\nProbing {len(candidate_ids)} generation/capacity candidates...\n")

confirmed = {}  # id -> {name, unit, n_rows, sample_value}

for ind_id in candidate_ids:
    url    = f"{BASE_URL}/indicators/{ind_id}"
    params = {
        "start_date":  TEST_START,
        "end_date":    TEST_END,
        "time_trunc":  "month",
    }
    try:
        r = requests.get(url, headers=HEADERS, params=params, timeout=20)
        r.raise_for_status()
        data   = r.json()["indicator"]
        values = data.get("values", [])
        if len(values) > 0:
            sample = values[0].get("value", None)
            unit   = data.get("magnitud", {})
            if isinstance(unit, dict):
                unit_name = unit.get("name", "?")
            else:
                unit_name = str(unit)
            confirmed[ind_id] = {
                "name":   data["name"],
                "unit":   unit_name,
                "n_rows": len(values),
                "sample": sample,
            }
            print(f"  ✓ ID={ind_id:>6}  n={len(values):>3}  "
                  f"sample={str(sample)[:10]:>12}  {unit_name:>8}  "
                  f"{data['name'][:60]}")
        time.sleep(0.5)
    except Exception as e:
        pass  # silently skip indicators with no data in test period

print(f"\n{len(confirmed)} indicators have data in Jan–Mar 2025\n")


# ── STEP 3: SELECT AND PULL ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 — Pulling full history for confirmed indicators")
print("=" * 60)

# If confirmed indicators were found, pull them all.
# If nothing was found (indicator IDs may differ in your environment),
# the script prints guidance to identify the right IDs manually.

if not confirmed:
    print("\n⚠ No autoconsumo indicators found with data in 2025.")
    print("  This likely means the data was published under different indicator IDs.")
    print("  To find them manually:")
    print("  1. Go to https://www.esios.ree.es/es/analisis")
    print("  2. Search for 'autoconsumo fotovoltaico'")
    print("  3. Note the indicator ID from the URL")
    print("  4. Add the ID to KNOWN_IDS below and re-run")
    print()
    # Fallback: try known candidate IDs from REE documentation
    # These are the most likely IDs based on REE's Dec 2025 announcement
    KNOWN_IDS = {
        # Format: id: description
        # Add here if you find the IDs manually
    }
    print("  Known candidate IDs to try (add manually if found):")
    print("  https://www.esios.ree.es/es/generacion-y-consumo")
    print()
else:
    # Pull full history for all confirmed indicators
    FULL_START = "2015-01-01T00:00:00"
    FULL_END   = "2026-04-30T23:59:59"

    all_dfs = []
    for ind_id, meta in confirmed.items():
        print(f"\nPulling ID={ind_id}: {meta['name'][:60]}...")
        url    = f"{BASE_URL}/indicators/{ind_id}"
        params = {
            "start_date": FULL_START,
            "end_date":   FULL_END,
            "time_trunc": "month",
        }
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=60)
            r.raise_for_status()
            values = r.json()["indicator"]["values"]
            if values:
                df = pd.DataFrame(values)
                df["indicator_id"]   = ind_id
                df["indicator_name"] = meta["name"]
                df["unit"]           = meta["unit"]
                all_dfs.append(df)
                print(f"  → {len(df)} monthly records  "
                      f"({df['datetime'].min()[:7]} – {df['datetime'].max()[:7]})")
        except Exception as e:
            print(f"  ✗ Error: {e}")
        time.sleep(1)

    if all_dfs:
        raw = pd.concat(all_dfs, ignore_index=True)
        raw["date"] = pd.to_datetime(raw["datetime"], utc=True)\
                        .dt.tz_convert("Europe/Madrid")\
                        .dt.tz_convert(None)\
                        .dt.to_period("M")\
                        .dt.to_timestamp()
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")

        out_path = OUT_DIR / "esios_selfconsumption.csv"
        raw.to_csv(out_path, index=False)
        print(f"\n✓ Raw data saved → {out_path}")
        print(f"  {len(raw)} rows across {raw['indicator_id'].nunique()} indicators")

        # Print pivot for inspection
        print("\nMonthly totals by indicator (2025):")
        sample = raw[raw["date"].dt.year == 2025]\
                    .groupby(["indicator_name", "date"])["value"]\
                    .sum().unstack("date")
        print(sample.to_string())
    else:
        print("\n⚠ No data pulled. Check indicator IDs.")