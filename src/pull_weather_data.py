"""
pull_weather_data.py
====================
AEMET Daily Climate Data — 17 non-airport stations, one per autonomous community.

Airport stations are excluded because they systematically underreport sunshine
hours (aviation cloud-cover sensors, not Campbell-Stokes recorders) and show
heat-island temperature bias. City-centre synoptic stations and observatories
are used instead.

Output:
    data/raw/aemet_all_stations.csv   ← one row per (date, station)
    data/raw/aemet_national.csv       ← static-weighted national daily mean

Usage:
    export AEMET_API_KEY='your_key'
    python src/pull_weather_data.py
"""

import requests, pandas as pd, time, os
from pathlib import Path
from datetime import datetime, timedelta


def load_env(p):
    if not Path(p).exists():
        return
    for line in Path(p).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k not in os.environ:
            os.environ[k] = v


load_env(Path(__file__).resolve().parents[1] / ".env")

API_KEY = os.getenv("AEMET_API_KEY")
if not API_KEY:
    raise EnvironmentError("AEMET_API_KEY not set.")

START = "2015-01-01"
END = "2026-04-30"
BASE_URL = "https://opendata.aemet.es/opendata/api"
HEADERS = {"api_key": API_KEY, "Accept": "application/json"}
os.makedirs("data/raw", exist_ok=True)

# ── STATION TABLE (non-airport) ───────────────────────────────────────────────
# One synoptic/observatory/urban station per autonomous community.
# Approx 2025 self-consumption capacity (MW) for static weighting.
STATIONS = [
    # (ccaa,                  station_id, city/site,                     approx_cap_mw)
    ("Andalucía", "5783", "Sevilla Centro", 1800),
    ("Cataluña", "0149X", "Barcelona", 1400),  # 0201D has no sol
    ("C. Valenciana", "8293X", "Valencia Viveros", 1200),
    ("Murcia", "7031", "Murcia San Javier", 700),  # 7012C has no sol
    ("Madrid", "3196", "Madrid Retiro", 650),
    ("Castilla-La Mancha", "4121", "Ciudad Real", 500),
    ("Castilla y León", "2465", "Salamanca Matacán", 420),
    ("Aragón", "9390", "Zaragoza Aula Dei", 380),
    ("Extremadura", "4452", "Cáceres", 340),  # 4386 invalid
    ("Canarias", "C447A", "Las Palmas Escaleritas", 320),
    ("País Vasco", "1082", "San Sebastián Igueldo", 250),
    ("Galicia", "1484C", "Santiago Campus", 220),
    ("Baleares", "B954", "Palma Puerto", 210),
    ("Navarra", "9263D", "Pamplona", 180),  # 9240E invalid
    ("Asturias", "1208H", "Gijón", 140),
    ("Cantabria", "1111X", "Santander Ciudad", 110),
    ("La Rioja", "9091O", "Logroño Ciudad", 95),
]

STATIONS_DF = pd.DataFrame(
    STATIONS, columns=["ccaa", "station_id", "city", "approx_cap_mw"]
)
STATIONS_DF["static_weight"] = (
    STATIONS_DF["approx_cap_mw"] / STATIONS_DF["approx_cap_mw"].sum()
)


# ── HELPERS ───────────────────────────────────────────────────────────────────
def parse_float(v):
    if v is None or v == "":
        return None
    try:
        return float(str(v).replace(",", "."))
    except:
        return None


def fetch_chunk(station_id, date_from, date_to):
    url = (
        f"{BASE_URL}/valores/climatologicos/diarios/datos/"
        f"fechaini/{date_from}T00:00:00UTC/"
        f"fechafin/{date_to}T23:59:59UTC/estacion/{station_id}/"
    )
    r1 = requests.get(url, headers=HEADERS, timeout=30)
    r1.raise_for_status()
    meta = r1.json()
    if meta.get("estado") != 200:
        return []
    data_url = meta.get("datos")
    if not data_url:
        return []
    r2 = requests.get(data_url, timeout=30)
    r2.raise_for_status()
    rec = r2.json()
    return rec if isinstance(rec, list) else []


def date_chunks(start, end, months=6):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    chunks, cur = [], s
    while cur <= e:
        mo = cur.month - 1 + months
        yr = cur.year + mo // 12
        mo = mo % 12 + 1
        ce = min(datetime(yr, mo, 1) - timedelta(days=1), e)
        chunks.append((cur.strftime("%Y-%m-%d"), ce.strftime("%Y-%m-%d")))
        cur = ce + timedelta(days=1)
    return chunks


COLUMN_MAP = {
    "fecha": "date",
    "tmed": "temp_mean_c",
    "tmax": "temp_max_c",
    "tmin": "temp_min_c",
    "prec": "precip_mm",
    "velmedia": "wind_speed_ms",
    "racha": "wind_gust_ms",
    "sol": "sunshine_hours",
    "presMax": "pressure_max_hpa",
    "presMin": "pressure_min_hpa",
    "hrMedia": "humidity_mean_pct",
}

# ── MAIN ──────────────────────────────────────────────────────────────────────
chunks = date_chunks(START, END, months=6)
print(f"Pulling {len(STATIONS)} non-airport stations × {len(chunks)} chunks\n")

all_dfs = []
for ccaa, station_id, city, cap_mw in STATIONS:
    print(f"\n── {ccaa} ({station_id} / {city}) ──")
    records = []
    for d_from, d_to in chunks:
        print(f"  {d_from} → {d_to}", end="  ", flush=True)
        for attempt in range(4):
            try:
                r = fetch_chunk(station_id, d_from, d_to)
                print(f"{len(r)} records")
                records.extend(r)
                break
            except requests.HTTPError as e:
                code = e.response.status_code if e.response else "?"
                if code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"rate-limited {wait}s ...", end=" ", flush=True)
                    time.sleep(wait)
                else:
                    print(f"HTTP {code} — skip")
                    break
            except Exception as e:
                print(f"error: {e} — skip")
                break
        time.sleep(3)

    if not records:
        print(f"  ⚠ No data for {ccaa}")
        continue

    df = pd.DataFrame(records)
    keep = {k: v for k, v in COLUMN_MAP.items() if k in df.columns}
    df = df[list(keep)].rename(columns=keep)
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    for col in [c for c in df.columns if c != "date"]:
        df[col] = df[col].apply(parse_float)
    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["station_id"] = station_id
    df["ccaa"] = ccaa
    df["city"] = city
    all_dfs.append(df)
    n_sun = df["sunshine_hours"].notna().sum() if "sunshine_hours" in df.columns else 0
    pct = n_sun / len(df) * 100 if len(df) > 0 else 0
    sun_flag = (
        "" if pct > 50 else "  ⚠ LOW sunshine coverage — consider alternate station"
    )
    print(f"  → {len(df)} rows  sunshine: {pct:.0f}%{sun_flag}")

# ── COMBINE ───────────────────────────────────────────────────────────────────
if not all_dfs:
    raise RuntimeError("No data retrieved — check API key.")

raw = pd.concat(all_dfs, ignore_index=True)
raw.to_csv("data/raw/aemet_all_stations.csv", index=False)
print(
    f"\n✓ Saved aemet_all_stations.csv  ({len(raw):,} rows, {raw.station_id.nunique()} stations)"
)

# ── STATIC-WEIGHTED NATIONAL AGGREGATE ───────────────────────────────────────
weights = STATIONS_DF.set_index("station_id")["static_weight"].to_dict()
raw["weight"] = raw["station_id"].map(weights)
WCOLS = [
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


def wmean(group):
    out = {"date": group["date"].iloc[0]}
    for col in WCOLS:
        if col not in group.columns:
            continue
        v = group[[col, "weight"]].dropna(subset=[col])
        if v.empty:
            out[col] = None
        else:
            w = v["weight"] / v["weight"].sum()
            out[col] = (v[col] * w).sum()
    out["n_stations"] = (
        group["sunshine_hours"].notna().sum()
        if "sunshine_hours" in group.columns
        else 0
    )
    return pd.Series(out)


national = raw.groupby("date").apply(wmean).reset_index(drop=True)
national["date"] = pd.to_datetime(national["date"])
national = national.sort_values("date").reset_index(drop=True)
national.to_csv("data/raw/aemet_national.csv", index=False)
print(f"✓ Saved aemet_national.csv  ({len(national):,} daily rows)")

print(f"\nSunshine coverage by station:")
for ccaa, sid, city, _ in STATIONS:
    s = raw[raw.station_id == sid].get("sunshine_hours", pd.Series(dtype=float))
    if len(s):
        pct = s.notna().mean() * 100
        bar = "*" * int(pct / 10)
        print(f"  {ccaa:<25} {sid:<8} {pct:>5.1f}%  {bar}")

print(f"\nBlackout window Apr 25–May 1 2025:")
mask = (national.date >= "2025-04-25") & (national.date <= "2025-05-01")
print(national[mask][["date", "sunshine_hours", "temp_mean_c"]].to_string(index=False))
