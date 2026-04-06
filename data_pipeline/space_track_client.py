"""
data_pipeline/space_track_client.py
Authenticated REST client for Space-Track.org TLE history API.

Docs: https://www.space-track.org/documentation
API endpoint used: /basicspacedata/query/class/gp_history/

Usage:
    client = SpaceTrackClient()
    client.login()
    tles = client.get_tles_around_date(norad=25544, date_str="2024-05-10", window_days=1)
    client.logout()
"""

import requests
import json
import time
import os
import logging
from datetime import datetime, timedelta

log = logging.getLogger(__name__)

try:
    from config import SPACETRACK_USER, SPACETRACK_PASS, SPACETRACK_BASE, RAW_TLE_CACHE, OUTPUT_DIR
except ImportError:
    import sys; sys.path.insert(0, os.path.dirname(__file__))
    from config import SPACETRACK_USER, SPACETRACK_PASS, SPACETRACK_BASE, RAW_TLE_CACHE, OUTPUT_DIR


class SpaceTrackClient:
    """Thin authenticated wrapper around the Space-Track.org REST API."""

    LOGIN_URL  = f"{SPACETRACK_BASE}/ajaxauth/login"
    LOGOUT_URL = f"{SPACETRACK_BASE}/ajaxauth/logout"
    BASE_URL   = f"{SPACETRACK_BASE}/basicspacedata/query"

    def __init__(self):
        self.session = requests.Session()
        self._logged_in = False
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Load disk cache to avoid hammering the API
        if os.path.exists(RAW_TLE_CACHE):
            with open(RAW_TLE_CACHE) as f:
                self._cache = json.load(f)
        else:
            self._cache = {}

    # ── Authentication ───────────────────────────────────────────────────────

    def login(self):
        if not SPACETRACK_USER or not SPACETRACK_PASS:
            raise ValueError(
                "Space-Track credentials missing. "
                "Add SPACETRACK_USER and SPACETRACK_PASS to backend/.env"
            )
        r = self.session.post(self.LOGIN_URL, data={
            "identity": SPACETRACK_USER,
            "password": SPACETRACK_PASS,
        })
        r.raise_for_status()
        self._logged_in = True
        log.info("[SpaceTrack] Logged in successfully")

    def logout(self):
        if self._logged_in:
            self.session.get(self.LOGOUT_URL)
            self._logged_in = False
            log.info("[SpaceTrack] Logged out")
        self._save_cache()

    def _save_cache(self):
        with open(RAW_TLE_CACHE, "w") as f:
            json.dump(self._cache, f, indent=2)

    # ── TLE Fetching ─────────────────────────────────────────────────────────

    def get_tles_around_date(self, norad: int, date_str: str,
                              window_days: int = 1) -> list[dict]:
        """
        Fetch all TLEs for a given NORAD ID within ±window_days of date_str.

        Returns list of dicts, each with:
            { "EPOCH": str, "TLE_LINE1": str, "TLE_LINE2": str, "NORAD_CAT_ID": str }
        """
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=window_days)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=window_days)).strftime("%Y-%m-%d")
        return self._fetch_tle_range(norad, start, end)

    def get_closest_tle_before(self, norad: int, date_str: str,
                                lookback_days: int = 7) -> dict | None:
        """Get the single TLE closest to but before date_str."""
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        end   = (dt - timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
        tles  = self._fetch_tle_range(norad, start, end)
        if not tles:
            return None
        return tles[-1]

    def get_closest_tle_after(self, norad: int, date_str: str,
                               lookahead_days: int = 10) -> dict | None:
        """Get the single TLE closest to but after date_str + storm recovery."""
        dt    = datetime.strptime(date_str, "%Y-%m-%d")
        start = (dt + timedelta(hours=12)).strftime("%Y-%m-%d")
        end   = (dt + timedelta(days=lookahead_days)).strftime("%Y-%m-%d")
        tles  = self._fetch_tle_range(norad, start, end)
        if not tles:
            return None
        return tles[0]

    def _fetch_tle_range(self, norad: int, start: str, end: str) -> list[dict]:
        """
        Core fetch method. Caches results to avoid redundant API calls.
        Rate-limits to 1 request/second per Space-Track ToS.
        """
        cache_key = f"{norad}_{start}_{end}"
        if cache_key in self._cache:
            log.debug(f"[SpaceTrack] Cache hit: {cache_key}")
            return self._cache[cache_key]

        if not self._logged_in:
            raise RuntimeError("Call client.login() before fetching TLEs")

        url = (
            f"{self.BASE_URL}"
            f"/class/gp_history"
            f"/NORAD_CAT_ID/{norad}"
            f"/orderby/EPOCH asc"
            f"/EPOCH/{start}--{end}"
            f"/format/json"
        )

        log.info(f"[SpaceTrack] Fetching NORAD {norad} | {start} → {end}")
        time.sleep(1)  # Respect rate limit (ToS: ≤ 1 req/s)

        r = self.session.get(url, timeout=30)
        r.raise_for_status()

        data = r.json()

        # ── Detect quota exceeded / API error responses ─────────────────────
        if (isinstance(data, list) and len(data) == 1
                and isinstance(data[0], dict) and "error" in data[0]):
            err_msg = data[0]["error"]
            if "quota" in err_msg.lower() or "violated" in err_msg.lower():
                raise RuntimeError(
                    f"Space-Track QUOTA EXCEEDED. Wait 30-60 min, then retry.\n"
                    f"Msg: {err_msg[:120]}"
                )
            else:
                log.error(f"[SpaceTrack] API error: {err_msg[:120]}")
                return []

        records = []
        for item in data:
            line1 = item.get("TLE_LINE1", "")
            line2 = item.get("TLE_LINE2", "")

            # If no explicit TLE lines, reconstruct from GP element fields
            if not line2 and item.get("MEAN_MOTION"):
                try:
                    mean_motion  = float(item["MEAN_MOTION"])
                    inclination  = float(item.get("INCLINATION", 0))
                    raan         = float(item.get("RA_OF_ASC_NODE", 0))
                    ecc_str      = str(item.get("ECCENTRICITY", "0")).replace("0.", "")
                    ecc_str      = ecc_str[:7].ljust(7, "0")
                    arg_perigee  = float(item.get("ARG_OF_PERICENTER", 0))
                    mean_anomaly = float(item.get("MEAN_ANOMALY", 0))
                    rev_num      = int(float(item.get("REV_AT_EPOCH", 0)))
                    norad_id     = str(norad).zfill(5)
                    line2 = (f"2 {norad_id} {inclination:8.4f} {raan:8.4f} "
                             f"{ecc_str} {arg_perigee:8.4f} {mean_anomaly:8.4f} "
                             f"{mean_motion:11.8f}{rev_num:5d}0")
                    line2 = line2[:69].ljust(69)
                    epoch = item.get("EPOCH", "")[:10].replace("-", "")
                    line1 = (f"1 {norad_id}U          {epoch[:8]}"
                             f"         0.00000000  00000-0  00000-0 0  9990")
                except Exception as e:
                    log.debug(f"GP element TLE reconstruction failed: {e}")

            if line2 and item.get("MEAN_MOTION"):
                records.append({
                    "EPOCH":        item.get("EPOCH", ""),
                    "TLE_LINE1":    line1,
                    "TLE_LINE2":    line2,
                    "NORAD_CAT_ID": item.get("NORAD_CAT_ID", str(norad)),
                    "MEAN_MOTION":  item.get("MEAN_MOTION", ""),
                })

        self._cache[cache_key] = records
        self._save_cache()
        log.info(f"[SpaceTrack] Got {len(records)} TLEs for NORAD {norad}")
        return records
