"""
AEMET Daily Climate Data Puller
Station: 0149X (Barcelona city)
Period:  2015-01-01 to 2026-04-30

Usage:
    export AEMET_API_KEY='your_key_here'
    python pull_weather_data.py

Output:
    data/raw/aemet_barcelona_2015_2026.csv
"""

import requests
import pandas as pd
import time
import os
from pathlib import Path
from datetime import datetime, timedelta


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
API_KEY = os.getenv("AEMET_API_KEY")
if not API_KEY:
    raise EnvironmentError(
        "AEMET_API_KEY environment variable is not set. "
        "Export it before running: export AEMET_API_KEY='your_key_here'"
    )

STATION  = "0149X"
START    = "2015-01-01"
END      = "2026-04-30"

# Output path consistent with merge_datasets.py expectation
OUT_FILE = "data/raw/aemet_barcelona_2015_2026.csv"

BASE_URL = "https://opendata.aemet.es/opendata/api"
HEADERS  = {"api_key": API_KEY, "Accept": "application/json"}

# Ensure output directory exists
os.makedirs("data/raw", exist_ok=True)


# ── HELPERS ───────────────────────────────────────────────────────────────────

def fetch_chunk(date_from: str, date_to: str) -> list:
    """
    Fetch daily climate data for STATION between date_from and date_to.
    AEMET uses a two-step redirect: first call returns a data URL,
    second call returns the actual JSON records.
    """
    d_from = f"{date_from}T00:00:00UTC"
    d_to   = f"{date_to}T23:59:59UTC"

    url = (
        f"{BASE_URL}/valores/climatologicos/diarios/datos/"
        f"fechaini/{d_from}/fechafin/{d_to}/estacion/{STATION}/"
    )

    # Step 1 — get redirect URL
    r1 = requests.get(url, headers=HEADERS, timeout=30)
    r1.raise_for_status()
    meta = r1.json()

    estado = meta.get("estado")
    if estado != 200:
        print(f"  ⚠ API warning for {date_from}→{date_to}: "
              f"estado={estado} — {meta.get('descripcion', '')}")
        return []

    data_url = meta.get("datos")
    if not data_url:
        print(f"  ⚠ No data URL returned for {date_from}→{date_to}")
        return []

    # Step 2 — fetch actual data
    r2 = requests.get(data_url, timeout=30)
    r2.raise_for_status()
    records = r2.json()

    return records if isinstance(records, list) else []


def generate_chunks(start: str, end: str, months: int = 6):
    """Split date range into chunks of `months` months (AEMET max is 6 months)."""
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end,   "%Y-%m-%d")

    chunks = []
    current = s
    while current <= e:
        month = current.month - 1 + months
        year  = current.year + month // 12
        month = month % 12 + 1
        chunk_end = min(datetime(year, month, 1) - timedelta(days=1), e)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return chunks


def parse_spanish_float(val):
    """AEMET returns numbers with comma as decimal separator."""
    if val is None or val == "":
        return None
    try:
        return float(str(val).replace(",", "."))
    except ValueError:
        return None


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    chunks = generate_chunks(START, END, months=6)
    print(f"Fetching {len(chunks)} six-month chunks for station {STATION}")
    print(f"Period: {START} → {END}\n")

    all_records = []

    for date_from, date_to in chunks:
        print(f"  Fetching {date_from} → {date_to} ...", end=" ", flush=True)

        for attempt in range(4):
            try:
                records = fetch_chunk(date_from, date_to)
                print(f"{len(records)} records")
                all_records.extend(records)
                break
            except requests.HTTPError as e:
                if e.response is not None and e.response.status_code == 429:
                    wait = 30 * (attempt + 1)
                    print(f"rate limited — waiting {wait}s (retry {attempt + 1}/3) ...",
                          end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"HTTP error: {e}")
                    break
            except Exception as e:
                print(f"Error: {e}")
                break

        time.sleep(3)  # be polite between requests

    if not all_records:
        print("\nNo data retrieved. Check your API key or station ID.")
        return

    # ── Parse into DataFrame ──────────────────────────────────────────────────
    df = pd.DataFrame(all_records)

    column_map = {
        "fecha":    "date",
        "tmed":     "temp_mean_c",
        "tmax":     "temp_max_c",
        "tmin":     "temp_min_c",
        "prec":     "precip_mm",
        "velmedia": "wind_speed_ms",
        "racha":    "wind_gust_ms",
        "sol":      "sunshine_hours",
        "presMax":  "pressure_max_hpa",
        "presMin":  "pressure_min_hpa",
        "hrMedia":  "humidity_mean_pct",
    }

    keep = {k: v for k, v in column_map.items() if k in df.columns}
    df = df[list(keep.keys())].rename(columns=keep)

    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")

    numeric_cols = [c for c in df.columns if c != "date"]
    for col in numeric_cols:
        df[col] = df[col].apply(parse_spanish_float)

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\nDone! {len(df)} daily records retrieved")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nMissing values per column:")
    print(df.isnull().sum().to_string())

    # Check for missing date ranges
    full_range = pd.date_range(START, END)
    missing_dates = full_range.difference(df["date"])
    if len(missing_dates) > 0:
        print(f"\n  ⚠ {len(missing_dates)} dates missing from output:")
        gaps = []
        start_gap = missing_dates[0]
        prev = missing_dates[0]
        for d in missing_dates[1:]:
            if (d - prev).days > 1:
                gaps.append((start_gap, prev))
                start_gap = d
            prev = d
        gaps.append((start_gap, prev))
        for g in gaps:
            print(f"    {g[0].date()} → {g[1].date()}")
    else:
        print(f"\nNo missing dates — complete series!")

    print(f"\nSample (around April 28, 2025):")
    mask = (df["date"] >= "2025-04-25") & (df["date"] <= "2025-05-01")
    print(df[mask].to_string(index=False))

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUT_FILE, index=False)
    print(f"\nSaved → {OUT_FILE}")


if __name__ == "__main__":
    main()
