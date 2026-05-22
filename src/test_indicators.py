"""
find_solar_indicator.py
=======================
Tests candidate solar PV indicators against your ESIOS token to find one
that returns hourly peninsular data with correct growing trend 2015→2024.

Run:  python src/find_solar_indicator.py
Then: copy the working ID into pull_all_esios_data.py → INDICATORS["solar_pv"]
"""

import os, time, requests, pandas as pd
from pathlib import Path

def load_env(path):
    if not Path(path).exists(): return
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1)
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k not in os.environ: os.environ[k] = v

load_env(Path(__file__).resolve().parents[1] / ".env")

TOKEN = os.getenv("ESIOS_TOKEN")
if not TOKEN:
    raise EnvironmentError("ESIOS_TOKEN not set")

HEADERS = {
    "Accept":       "application/json; application/vnd.esios-api-v1+json",
    "Content-Type": "application/json",
    "Host":         "api.esios.ree.es",
    "x-api-key":    TOKEN,
}

# ── Candidate solar indicators ────────────────────────────────────────────────
# All are some variant of "generación solar fotovoltaica peninsular"
CANDIDATES = {
    1159:  "Generación solar fotovoltaica (classic)",
    10289: "Solar PV generation (newer)",
    10355: "Solar PV generation peninsular (alt)",
    1060:  "Solar thermal generation (CSP, peninsular)",
    10358: "Solar photovoltaic peninsular (alt2)",
    76:    "Generación solar (old aggregate)",
    77:    "Generación solar fotovoltaica (old)",
    10245: "Solar PV real-time (SCADA)",
    1160:  "Solar thermal (CSP)",
    10359: "Solar PV peninsular balance",
}

# Test window: one representative summer month 2024 (should be high)
# and one winter month 2015 (should be lower but non-zero)
TEST_PERIODS = [
    ("2024-07-01T00:00:00", "2024-07-31T23:59:59", "Jul 2024"),
    ("2015-07-01T00:00:00", "2015-07-31T23:59:59", "Jul 2015"),
    ("2022-07-01T00:00:00", "2022-07-31T23:59:59", "Jul 2022"),
]

print("=" * 70)
print("ESIOS SOLAR INDICATOR SEARCH")
print("=" * 70)
print(f"Testing {len(CANDIDATES)} candidate IDs across {len(TEST_PERIODS)} periods\n")

results = {}

for ind_id, desc in CANDIDATES.items():
    print(f"\n── ID {ind_id}: {desc}")
    period_results = {}

    for start, end, label in TEST_PERIODS:
        url    = f"https://api.esios.ree.es/indicators/{ind_id}"
        params = {"start_date": start, "end_date": end, "time_trunc": "hour"}

        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)

            if r.status_code == 403:
                print(f"  {label}: 403 FORBIDDEN (access restricted)")
                period_results[label] = None
                break

            if r.status_code == 404:
                print(f"  {label}: 404 NOT FOUND")
                period_results[label] = None
                break

            r.raise_for_status()
            values = r.json()["indicator"]["values"]

            if not values:
                print(f"  {label}: 0 rows")
                period_results[label] = 0
                continue

            df = pd.DataFrame(values)
            df["value"] = pd.to_numeric(df["value"], errors="coerce")

            # Filter to peninsula only if geo_id present
            if "geo_id" in df.columns:
                peninsula = df[df["geo_id"] == 8741]
                if len(peninsula) > 0:
                    df = peninsula
                geo_ids = sorted(df["geo_id"].unique().tolist())
            else:
                geo_ids = ["n/a"]

            mean_mw = df["value"].mean()
            max_mw  = df["value"].max()
            n_rows  = len(df)

            period_results[label] = mean_mw
            print(f"  {label}: {n_rows:>5} rows  mean={mean_mw:>8,.0f} MW  "
                  f"max={max_mw:>8,.0f} MW  geo_ids={geo_ids}")

        except Exception as e:
            print(f"  {label}: ERROR — {e}")
            period_results[label] = None

        time.sleep(0.8)

    results[ind_id] = period_results

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — candidates with data in ALL test periods")
print("=" * 70)
print(f"\n{'ID':<8} {'Jul 2015':>12} {'Jul 2022':>12} {'Jul 2024':>12}  trend  notes")

working = []
for ind_id, pr in results.items():
    v15 = pr.get("Jul 2015")
    v22 = pr.get("Jul 2022")
    v24 = pr.get("Jul 2024")

    if v15 is None or v24 is None or v15 == 0 or v24 == 0:
        continue

    # Good indicator: 2024 > 2022 > 2015, values physically plausible
    # July 2015: ~4500 MW installed → expect ~1500–2500 MW average hourly
    # July 2024: ~25000 MW installed → expect ~5000–12000 MW average hourly
    growing     = v24 > v15
    plausible15 = 500 < v15 < 8000
    plausible24 = 2000 < v24 < 20000

    trend = "✅ growing" if growing else "❌ decreasing"
    flags = []
    if not plausible15: flags.append(f"2015 val suspicious ({v15:,.0f} MW)")
    if not plausible24: flags.append(f"2024 val suspicious ({v24:,.0f} MW)")
    notes = ", ".join(flags) if flags else "✅ plausible"

    v22_str = f"{v22:>12,.0f}" if v22 else "         n/a"
    print(f"  {ind_id:<6} {v15:>12,.0f} {v22_str} {v24:>12,.0f}  {trend}  {notes}")

    if growing and plausible15 and plausible24:
        working.append(ind_id)

if working:
    print(f"\n✅ RECOMMENDED INDICATOR(S): {working}")
    print(f"\nIn pull_all_esios_data.py, set:")
    print(f'    "solar_pv": {working[0]},')
else:
    print("\n⚠ No single indicator passed all checks.")
    print("  Try the ESIOS web interface to find the correct solar PV series:")
    print("  https://www.esios.ree.es/es/analisis/1159")
    print()
    print("  Also try searching the API directly:")
    print("  curl -H 'x-api-key: YOUR_TOKEN' \\")
    print("    'https://api.esios.ree.es/indicators?text=solar+fotovoltaica'")