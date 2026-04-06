"""
data_pipeline/build_dataset.py
Main pipeline — fetches real TLEs from Space-Track, computes TLE-derived
ground-truth altitude drops, augments with real Kp/F10.7 from NASA OMNI2,
and outputs ground_truth_dataset.csv.

Features:
  - Incremental CSV writes  → no data lost if quota hit mid-run
  - Resume support          → re-run safely, already-collected rows skipped
  - Graceful quota handling → stops cleanly, logs remaining work
"""

import os
import sys
import csv
import math
import logging
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from config import (SATELLITES, STORM_EVENTS, OOD_LABELS,
                    TLE_BEFORE_OFFSET_DAYS, TLE_AFTER_OFFSET_DAYS,
                    GROUND_TRUTH_CSV, OUTPUT_DIR, SAT_PARAMS_EXTENDED)
from space_track_client import SpaceTrackClient
from tle_utils import tle_pair_to_altitude, compute_delta_h, physics_predict_dh
from historical_kp import get_kp_max_for_storm, get_f107_for_date, KP_CACHE_PATH, _load_cache, _fetch_omni_year

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger(__name__)

# ── Satellite physical parameters — imported from config ──────────────────────
SAT_PARAMS = SAT_PARAMS_EXTENDED


# ── Quiet-time density (kg/m³) and Kp multipliers ────────────────────────────
QUIET_DENSITY = {
    200: 4.0e-10, 250: 5.5e-11, 300: 9.0e-12, 350: 2.0e-12,
    400: 4.7e-13, 450: 1.1e-13, 500: 3.0e-14, 550: 1.8e-14,
    600: 9.0e-15, 700: 4.0e-15, 800: 2.5e-15, 900: 1.5e-15,
}
KP_MULT = {0: 1.0, 1: 1.1, 2: 1.2, 3: 1.5, 4: 2.0, 5: 3.0,
           6: 5.0, 7: 8.0, 8: 15.0, 9: 30.0}


def _quiet_density(alt_km):
    alts = sorted(QUIET_DENSITY)
    if alt_km <= alts[0]:  return QUIET_DENSITY[alts[0]]
    if alt_km >= alts[-1]: return QUIET_DENSITY[alts[-1]]
    for i in range(len(alts)-1):
        if alts[i] <= alt_km <= alts[i+1]:
            t = (alt_km - alts[i]) / (alts[i+1] - alts[i])
            return QUIET_DENSITY[alts[i]] ** (1-t) * QUIET_DENSITY[alts[i+1]] ** t
    return 1e-13


def _storm_density(alt_km, kp, cme_speed):
    k    = min(int(kp), 9)
    mult = KP_MULT.get(k, 30.0)
    mult *= 1.0 + (cme_speed / 3000.0) * 0.6
    return _quiet_density(alt_km) * mult


def _classify_phase(kp_max, duration_h):
    if kp_max >= 7.0 and duration_h <= 12:  return "sudden_commencement"
    if kp_max >= 5.0 and duration_h <= 36:  return "main_phase"
    return "recovery_phase"


def run_pipeline():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("=" * 60)
    log.info("CELESTIS Real-Data Ground Truth Pipeline")
    log.info("=" * 60)

    # ── Step 1: OMNI2 cache warm-up ──────────────────────────────────────────
    log.info("[1/4] Pre-downloading NASA OMNI2 Kp/F10.7 data...")
    unique_years = sorted({int(ev[1][:4]) for ev in STORM_EVENTS} |
                          {int(ev[2][:4]) for ev in STORM_EVENTS})
    log.info(f"      Years needed: {unique_years}")
    cache = _load_cache(KP_CACHE_PATH)
    for yr in unique_years:
        _fetch_omni_year(yr, cache)
    log.info("      OMNI2 cache ready.")

    # ── Step 2: Space-Track login ─────────────────────────────────────────────
    log.info("[2/4] Authenticating with Space-Track.org...")
    client = SpaceTrackClient()
    try:
        client.login()
    except ValueError as e:
        log.error(f"Login failed: {e}")
        return False

    # ── Step 3: Resume support — load existing rows ───────────────────────────
    existing_keys = set()
    existing_rows = []
    if os.path.exists(GROUND_TRUTH_CSV):
        with open(GROUND_TRUTH_CSV, newline="") as f:
            for r in csv.DictReader(f):
                existing_rows.append(r)
                existing_keys.add((r["storm_label"], r["satellite"]))
        log.info(f"Resume: {len(existing_rows)} rows already collected, skipping those.")

    # Open in append mode — writes each row immediately (crash-safe)
    csv_fh     = open(GROUND_TRUTH_CSV, "a", newline="")
    writer     = None
    new_count  = 0
    skipped    = 0
    quota_hit  = False

    log.info(f"[3/4] Processing {len(STORM_EVENTS)} events x {len(SATELLITES)} satellites...")

    for (label, onset, end, kp_enc, f107_enc, cme, flare) in STORM_EVENTS:
        if quota_hit:
            break

        # Real Kp + F10.7 from OMNI2
        kp_data     = get_kp_max_for_storm(onset, end)
        kp_max_real = kp_data["kp_max"] or kp_enc
        f107_real   = get_f107_for_date(onset) or f107_enc

        dt0 = datetime.strptime(onset, "%Y-%m-%d")
        dt1 = datetime.strptime(end,   "%Y-%m-%d")
        dur = max(6.0, (dt1 - dt0).total_seconds() / 3600)

        phase = _classify_phase(kp_max_real, dur)
        split = "ood" if label in OOD_LABELS else "train"

        for sat_name, (norad, launch_yr) in SATELLITES.items():
            if quota_hit:
                break

            # Skip pre-launch combinations
            if int(onset[:4]) < launch_yr:
                skipped += 1
                continue

            # Skip already-collected combinations (resume)
            if (label, sat_name) in existing_keys:
                continue

            # Fetch TLEs with quota-graceful error handling
            try:
                tle_b = client.get_closest_tle_before(
                    norad, onset, lookback_days=TLE_BEFORE_OFFSET_DAYS + 2)
                tle_a = client.get_closest_tle_after(
                    norad, end,   lookahead_days=TLE_AFTER_OFFSET_DAYS + 1)
            except RuntimeError as e:
                if "QUOTA" in str(e):
                    log.error(str(e))
                    log.warning("Saving progress. Re-run in 30-60 min.")
                    quota_hit = True
                    break
                raise

            if not tle_b or not tle_a:
                log.warning(f"  x No TLE: {label} / {sat_name}")
                skipped += 1
                continue
            if not tle_b.get("TLE_LINE2") or not tle_a.get("TLE_LINE2"):
                log.warning(f"  x Empty TLE lines: {label} / {sat_name}")
                skipped += 1
                continue

            # Ground-truth ΔH from TLE mean motion
            obs_b = tle_pair_to_altitude(tle_b["TLE_LINE1"], tle_b["TLE_LINE2"])
            obs_a = tle_pair_to_altitude(tle_a["TLE_LINE1"], tle_a["TLE_LINE2"])
            gt    = compute_delta_h(obs_b, obs_a)

            # Physics-only baseline
            p      = SAT_PARAMS[norad]
            rho    = _storm_density(obs_b["altitude_km"], kp_max_real, cme)
            dh_phy = physics_predict_dh(
                rho_kg_m3=rho, Cd=p["Cd"], area_m2=p["area_m2"],
                mass_kg=p["mass_kg"], alt_km=obs_b["altitude_km"],
                duration_hours=dur)
            resid = gt["dh_true_km"] - dh_phy

            row = {
                "storm_label":      label,
                "satellite":        sat_name,
                "norad":            norad,
                "onset_date":       onset,
                "end_date":         end,
                "kp_max_real":      kp_max_real,
                "f107_real":        f107_real,
                "cme_km_s":         cme,
                "flare_enc":        flare,
                "storm_duration_h": round(dur, 1),
                "alt_before_km":    gt["alt_before_km"],
                "alt_after_km":     gt["alt_after_km"],
                "dh_true_km":       gt["dh_true_km"],
                "dh_physics_km":    dh_phy,
                "residual_km":      round(resid, 4),
                "delta_n":          gt["delta_n"],
                "epoch_before":     gt["epoch_before"],
                "epoch_after":      gt["epoch_after"],
                "storm_phase":      phase,
                "storm_class":      kp_data["storm_class"],
                "split":            split,
            }

            # Write immediately (incremental, crash-safe)
            if writer is None:
                writer = csv.DictWriter(csv_fh, fieldnames=list(row.keys()))
                if not existing_keys:  # only write header if starting fresh
                    writer.writeheader()
            writer.writerow(row)
            csv_fh.flush()
            new_count += 1
            existing_keys.add((label, sat_name))

            status = "OOD" if split == "ood" else "train"
            log.info(f"  + {label} / {sat_name}: "
                     f"dH_true={gt['dh_true_km']:+.3f}km  "
                     f"dH_phys={dh_phy:+.3f}km  "
                     f"resid={resid:+.4f}km  [{status}]")

    csv_fh.close()
    client.logout()

    total = len(existing_rows) + new_count
    log.info(f"[4/4] Done.  Total rows: {total}  (new this run: {new_count})")
    if quota_hit:
        log.warning("Quota hit — re-run in 30-60 min to collect remaining events.")
        return False
    log.info("Complete! Run benchmark.py next.")
    return True


if __name__ == "__main__":
    run_pipeline()
