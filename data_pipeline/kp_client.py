"""
data_pipeline/kp_client.py
Fetches historical Kp index and F10.7 solar flux from free public APIs.

Sources:
  Kp  : GFZ Potsdam (definitive, 3-hourly, global standard)
        https://kp.gfz-potsdam.de/app/json/
  F10.7: LASP LISIRD / NOAA SWPC observed solar cycle indices
         https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json
         (monthly averages; daily from LASP API)

No authentication required for either source.
"""

import requests
import logging
import json
import os
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

try:
    from config import OUTPUT_DIR
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from config import OUTPUT_DIR

_KP_CACHE_PATH  = os.path.join(OUTPUT_DIR, "kp_cache.json")
_F107_CACHE_PATH = os.path.join(OUTPUT_DIR, "f107_cache.json")


def _load_json_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _save_json_cache(path: str, data: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


# ── GFZ Potsdam Kp API ───────────────────────────────────────────────────────

def fetch_kp_for_storm(onset_date: str, end_date: str) -> dict:
    """
    Fetch Kp index from GFZ Potsdam for a storm window.

    Parameters
    ----------
    onset_date : str  "YYYY-MM-DD"
    end_date   : str  "YYYY-MM-DD"

    Returns
    -------
    dict with keys:
      - kp_max      : maximum Kp in window (float)
      - kp_mean     : mean Kp in window
      - storm_class : G0–G5 string
    """
    cache = _load_json_cache(_KP_CACHE_PATH)
    key   = f"{onset_date}_{end_date}"
    if key in cache:
        return cache[key]

    url = (
        f"https://kp.gfz-potsdam.de/app/json/"
        f"?startdate={onset_date}&enddate={end_date}&index=Kp"
    )
    log.info(f"[GFZ] Fetching Kp {onset_date} → {end_date}")
    kp_vals = []
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data    = r.json()
        kp_vals = [float(v) for v in data.get("Kp", []) if v is not None]
    except Exception as e:
        log.warning(f"[GFZ] Primary failed ({e}), trying NOAA SWPC archive...")

    # ── Fallback: NOAA SWPC 3-hourly Kp JSON ────────────────────────────
    if not kp_vals:
        try:
            noaa_url = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index-1-minute.json"
            r2 = requests.get(noaa_url, timeout=15)
            r2.raise_for_status()
            raw = r2.json()
            for row_data in raw[1:]:  # skip header row
                ts = str(row_data[0])[:10]  # "YYYY-MM-DD"
                if onset_date <= ts <= end_date:
                    try:
                        kp_vals.append(float(row_data[1]))
                    except (ValueError, IndexError):
                        pass
            log.info(f"[NOAA-fallback] Got {len(kp_vals)} Kp readings in window")
        except Exception as e2:
            log.warning(f"[NOAA-fallback] Also failed: {e2}")

    # ── No data from either source ────────────────────────────────────────
    if not kp_vals:
        log.warning(f"[Kp] No data for {onset_date}–{end_date}, returning None")
        return {"kp_max": None, "kp_mean": None, "storm_class": "G0"}

    kp_max  = max(kp_vals)
    kp_mean = sum(kp_vals) / len(kp_vals)
    result  = {
        "kp_max":      round(kp_max, 2),
        "kp_mean":     round(kp_mean, 2),
        "storm_class": _kp_to_class(kp_max),
    }
    cache[key] = result
    _save_json_cache(_KP_CACHE_PATH, cache)
    return result


def _kp_to_class(kp: float) -> str:
    if kp >= 9.0: return "G5"
    if kp >= 8.0: return "G4"
    if kp >= 7.0: return "G3"
    if kp >= 6.0: return "G2"
    if kp >= 5.0: return "G1"
    return "G0"


# ── NOAA SWPC F10.7 ──────────────────────────────────────────────────────────

def fetch_f107_monthly() -> dict:
    """
    Download the NOAA SWPC monthly observed solar F10.7 index.
    Returns dict: {"YYYY-MM": f107_value}

    Cached to disk permanently — monthly data only changes once a month.
    """
    cache = _load_json_cache(_F107_CACHE_PATH)
    if cache:
        return cache

    url = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
    log.info("[NOAA] Fetching F10.7 monthly index")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        raw = r.json()
        result = {}
        for entry in raw:
            month = entry.get("time-tag", "")[:7]   # "YYYY-MM"
            f107  = entry.get("f10.7", None)
            if month and f107 is not None:
                result[month] = round(float(f107), 1)

        _save_json_cache(_F107_CACHE_PATH, result)
        log.info(f"[NOAA] Downloaded {len(result)} months of F10.7 data")
        return result
    except Exception as e:
        log.error(f"[NOAA] F10.7 fetch failed: {e}")
        return {}


def get_f107_for_date(date_str: str, f107_monthly: dict | None = None) -> float | None:
    """
    Look up F10.7 for a given "YYYY-MM-DD" date from the monthly cache.
    Returns the monthly average (standard practice for input to NRLMSISE-00).
    """
    if f107_monthly is None:
        f107_monthly = fetch_f107_monthly()
    month_key = date_str[:7]   # "YYYY-MM"
    return f107_monthly.get(month_key)
