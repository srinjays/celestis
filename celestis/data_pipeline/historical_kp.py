"""
data_pipeline/historical_kp.py
Downloads and caches historical 3-hourly Kp data from NASA GSFC SPDF OMNI2 archive.

Source (confirmed working):
  https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{YYYY}.dat

OMNI2 Column layout (1-indexed):
  Col 1  : Year (cols[0])
  Col 2  : Day of year (cols[1])
  Col 3  : Hour (cols[2])
  Col 39 : Kp index × 10  (cols[38], integer, fill=99)
  Col 51 : F10.7 index    (cols[50], float, fill=999.9)

  Verified against: omni2_2017.dat column inspection (55 columns per row)

This is the same OMNI2 dataset used by NRLMSISE-00 model inputs and
is citable as: King & Papitashvili (2005), JGR, doi:10.1029/2004JA010649
"""

import os
import sys
import ssl
import json
import logging
import urllib.request
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(__file__))
try:
    from config import OUTPUT_DIR
except ImportError:
    OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

KP_CACHE_PATH  = os.path.join(OUTPUT_DIR, "kp_omni_cache.json")
F107_CACHE_PATH = os.path.join(OUTPUT_DIR, "f107_cache.json")
SPDF_BASE      = "https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni"
_SSL_CTX       = ssl._create_unverified_context()   # needed on some Windows installs


def _load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(path: str, data: dict):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f)


def _fetch_omni_year(year: int, cache: dict) -> dict:
    """
    Download one year of OMNI2 hourly data and parse Kp (col 24) + F10.7 (col 50).
    Returns dict: {"YYYY-DOYHH": {"kp": float, "f107": float}}
    """
    yr_key = str(year)
    if yr_key in cache:
        return cache[yr_key]

    url = f"{SPDF_BASE}/omni2_{year}.dat"
    log.info(f"[OMNI2] Downloading {year} from NASA SPDF ({url})")
    try:
        r    = urllib.request.urlopen(url, context=_SSL_CTX, timeout=30)
        data = r.read().decode("ascii", errors="ignore")
    except Exception as e:
        log.error(f"[OMNI2] Download failed for {year}: {e}")
        return {}

    records = {}
    for line in data.splitlines():
        parts = line.split()
        if len(parts) < 51:   # need at least 51 columns
            continue
        try:
            yr_    = int(parts[0])
            doy    = int(parts[1])
            hr     = int(parts[2])
            kp_raw = int(parts[38])    # col 39: Kp × 10 (fill = 99)
            f107   = float(parts[50])  # col 51: F10.7 (fill = 999.9)
            if kp_raw >= 99:  kp_raw = None
            if f107  >= 999.9: f107  = None
            dt  = datetime(yr_, 1, 1) + timedelta(days=doy - 1, hours=hr)
            key = dt.strftime("%Y-%m-%d-%H")
            records[key] = {
                "kp":   round(kp_raw / 10.0, 1) if kp_raw is not None else None,
                "f107": round(f107, 1) if f107 is not None else None,
            }  
        except (ValueError, IndexError):
            continue

    if records:
        cache[yr_key] = records
        _save_cache(KP_CACHE_PATH, cache)
        log.info(f"[OMNI2] Parsed {len(records)} hourly records for {year}")
    return records


def get_kp_max_for_storm(onset_date: str, end_date: str) -> dict:
    """
    Get peak and mean Kp for a geomagnetic storm window.

    Parameters  onset_date, end_date : "YYYY-MM-DD"
    Returns     dict(kp_max, kp_mean, storm_class, n_readings)
    """
    cache = _load_cache(KP_CACHE_PATH)

    dt0 = datetime.strptime(onset_date, "%Y-%m-%d")
    dt1 = datetime.strptime(end_date,   "%Y-%m-%d")
    years = list(range(dt0.year, dt1.year + 1))

    kp_vals = []
    for yr in years:
        records = _fetch_omni_year(yr, cache)
        # Walk every hour in the window
        dt = dt0
        while dt <= dt1 + timedelta(days=1):
            key = dt.strftime("%Y-%m-%d-%H")
            if key in records and records[key]["kp"] is not None:
                kp_vals.append(records[key]["kp"])
            dt += timedelta(hours=1)

    if not kp_vals:
        log.warning(f"[Kp] No data for {onset_date}–{end_date}")
        return {"kp_max": None, "kp_mean": None, "storm_class": "G0", "n_readings": 0}

    kp_max  = max(kp_vals)
    kp_mean = round(sum(kp_vals) / len(kp_vals), 2)
    klass   = ("G5" if kp_max >= 9 else "G4" if kp_max >= 8 else
               "G3" if kp_max >= 7 else "G2" if kp_max >= 6 else
               "G1" if kp_max >= 5 else "G0")
    return {
        "kp_max":      round(kp_max, 2),
        "kp_mean":     kp_mean,
        "storm_class": klass,
        "n_readings":  len(kp_vals),
    }


def get_f107_for_date(date_str: str) -> float | None:
    """Get F10.7 from OMNI2 data (hourly average for the date)."""
    cache = _load_cache(KP_CACHE_PATH)
    yr = int(date_str[:4])
    records = _fetch_omni_year(yr, cache)
    f107_vals = []
    for hr in range(24):
        dt = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(hours=hr)
        key = dt.strftime("%Y-%m-%d-%H")
        if key in records and records[key]["f107"]:
            f107_vals.append(records[key]["f107"])
    if not f107_vals:
        return None
    return round(sum(f107_vals) / len(f107_vals), 1)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    events = [
        ("Halloween 2003",   "2003-10-28", "2003-10-31"),
        ("Sep 2017 X9.3",    "2017-09-06", "2017-09-10"),
        ("Starlink 2022",    "2022-02-03", "2022-02-04"),
        ("Gannon 2024",      "2024-05-10", "2024-05-12"),
        ("Quiet 2022",       "2022-08-10", "2022-08-11"),
    ]
    print("\nHistorical Kp from NASA OMNI2:")
    print("-" * 60)
    for name, onset, end in events:
        r = get_kp_max_for_storm(onset, end)
        f107d = get_f107_for_date(onset)
        print(f"  {name:<22} Kp_max={r['kp_max']}  "
              f"class={r['storm_class']}  F10.7={f107d}  "
              f"n={r['n_readings']}")
