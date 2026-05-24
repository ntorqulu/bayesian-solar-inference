"""
fetch_shared.py
===============
Shared utilities for all ESIOS fetch scripts:
  - Environment loading
  - Session management with retry
  - Monthly chunk generation
  - Raw data persistence
"""

import os, time, requests, pandas as pd
from pathlib import Path

# ── ENV ───────────────────────────────────────────────────────────────────────
def load_env(env_path: Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k not in os.environ:
            os.environ[k] = v

load_env(Path(__file__).resolve().parents[2] / ".env")

TOKEN = os.getenv("ESIOS_TOKEN")
if not TOKEN:
    raise EnvironmentError("ESIOS_TOKEN not set.")

BASE_URL = "https://api.esios.ree.es"
HEADERS  = {
    "Accept":       "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "Host":         "api.esios.ree.es",
    "x-api-key":    TOKEN,
}

START = "2015-01-01"
END   = "2026-04-30"

os.makedirs("data/raw", exist_ok=True)

# ── SESSION ───────────────────────────────────────────────────────────────────
_session = requests.Session()

def fetch_month(indicator_id: int, start_date: str, end_date: str,
                max_retries: int = 5) -> pd.DataFrame:
    """Fetch one monthly chunk from ESIOS. Returns raw DataFrame."""
    global _session
    url    = f"{BASE_URL}/indicators/{indicator_id}"
    params = {"start_date": start_date, "end_date": end_date, "time_trunc": "hour"}

    for attempt in range(max_retries):
        try:
            r = _session.get(url, headers=HEADERS, params=params, timeout=120)
            r.raise_for_status()
            values = r.json()["indicator"]["values"]
            return pd.DataFrame(values) if values else pd.DataFrame()

        except requests.exceptions.ReadTimeout:
            wait = 30 * (attempt + 1)
            print(f"    timeout — retry in {wait}s ({attempt+1}/{max_retries})")
            time.sleep(wait)

        except requests.exceptions.HTTPError as e:
            print(f"    HTTP {e.response.status_code}: {e}")
            return pd.DataFrame()

        except (requests.exceptions.SSLError,
                requests.exceptions.ConnectionError) as e:
            wait = 45 * (attempt + 1)
            print(f"    SSL/connection error — refreshing session, retry in {wait}s")
            _session.close()
            _session = requests.Session()
            time.sleep(wait)

        except Exception as e:
            print(f"    ERROR: {e}")
            return pd.DataFrame()

    return pd.DataFrame()


def monthly_chunks(start: str = START, end: str = END):
    """Yield (start_str, end_str) for every month in the range."""
    current = pd.Timestamp(start).replace(day=1)
    end_ts  = pd.Timestamp(end)
    while current <= end_ts:
        month_end  = current + pd.offsets.MonthEnd(0)
        chunk_end  = min(month_end, end_ts)
        yield (
            current.strftime("%Y-%m-%dT00:00:00"),
            chunk_end.strftime("%Y-%m-%dT23:59:59"),
        )
        current += pd.offsets.MonthBegin(1)


def download_indicator(indicator_id: int, name: str) -> pd.DataFrame:
    """Download all months for one indicator. Prints progress."""
    print(f"\n[{name.upper()}] Downloading indicator {indicator_id}")
    parts = []
    for s, e in monthly_chunks():
        print(f"  {s[:10]} → {e[:10]}", end="  ", flush=True)
        chunk = fetch_month(indicator_id, s, e)
        print(f"{len(chunk):>5} rows")
        if not chunk.empty:
            parts.append(chunk)
        time.sleep(1)

    if not parts:
        raise RuntimeError(f"No data retrieved for indicator {indicator_id} ({name})")

    df = pd.concat(parts, ignore_index=True)
    print(f"  TOTAL {name}: {len(df):,} rows")
    return df


def clean_hourly(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """
    Shared cleaning applied to every indicator's raw data:
      1. Parse datetime → UTC → floor to hour → Madrid tz   (collapses sub-second offsets)
      2. Keep only on-the-hour rows (minute == 0)
      3. Drop extra time columns (datetime_utc, tz_time) that cause phantom duplicates
      4. Deduplicate on (date, geo_id/geo_name)
    Returns a DataFrame with columns: date, {value_col}, [geo_id], [geo_name]
    """
    df = df.rename(columns={"datetime": "date", "value": value_col})

    # Floor in UTC (no DST ambiguity), then convert to Madrid
    df["date"] = (
        pd.to_datetime(df["date"], utc=True)
        .dt.floor("h")
        .dt.tz_convert("Europe/Madrid")
    )
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Keep only essential columns
    keep = ["date", value_col] + [c for c in ["geo_id", "geo_name"] if c in df.columns]
    df   = df[keep].copy()

    # Minute filter (after floor this is always 0, but defensive)
    df = df[df["date"].dt.minute == 0].copy()

    # Dedup on (date, geo_id)
    dedup = ["date"] + [c for c in ["geo_id", "geo_name"] if c in df.columns]
    df = df.sort_values("date").drop_duplicates(subset=dedup, keep="first")

    return df