"""
data_pipeline/tle_utils.py
Parse TLE strings → orbital elements → altitude.

Ground Truth Definition (paper-grade):
  "True altitude drop" = semi-major axis derived from TLE mean motion
  before storm onset minus semi-major axis after storm end.

  h = a - R_Earth
  a = (GM / (2π·n/86400)²)^(1/3)
  where n = mean motion in rev/day (TLE Line 2, cols 52–63)

This definition matches Doornbos & Klinkrad (2006) and is reproducible
from any NORAD TLE archive.
"""

import math
from config import R_EARTH_KM, GM_KM3_S2


def parse_mean_motion(tle_line2: str) -> float:
    """
    Extract mean motion n (rev/day) from TLE Line 2.
    TLE Line 2 format: cols 52-63 (0-indexed) = mean motion rev/day
    """
    return float(tle_line2[52:63])


def parse_epoch(tle_line1: str) -> str:
    """Extract epoch string from TLE Line 1, columns 18-32."""
    return tle_line1[18:32].strip()


def mean_motion_to_altitude(n_rev_day: float) -> float:
    """
    Convert mean motion (rev/day) → altitude (km).

    Steps:
      1. Convert rev/day → rad/s
      2. Apply Kepler's third law: a = (GM / ω²)^(1/3)
      3. h = a - R_Earth

    Returns altitude in km above mean Earth surface.
    """
    omega = (2 * math.pi * n_rev_day) / 86400.0   # rad/s
    a_km  = (GM_KM3_S2 / omega**2) ** (1.0 / 3.0)  # km
    h_km  = a_km - R_EARTH_KM
    return round(h_km, 3)


def tle_pair_to_altitude(tle_line1: str, tle_line2: str) -> dict:
    """
    Given a TLE line pair, return a dict with:
      - altitude_km  : computed from mean motion
      - mean_motion  : raw rev/day value
      - epoch        : epoch string from Line 1
    """
    n   = parse_mean_motion(tle_line2)
    alt = mean_motion_to_altitude(n)
    ep  = parse_epoch(tle_line1)
    return {
        "altitude_km":   alt,
        "mean_motion":   n,
        "epoch":         ep,
    }


def compute_delta_h(before: dict, after: dict) -> dict:
    """
    Compute true altitude drop (ground truth) from two TLE observations.

    Parameters
    ----------
    before : dict  — output of tle_pair_to_altitude() from pre-storm TLE
    after  : dict  — output of tle_pair_to_altitude() from post-storm TLE

    Returns
    -------
    dict with:
      - dh_true_km      : altitude dropped (positive = lost altitude)
      - alt_before_km   : altitude before storm
      - alt_after_km    : altitude after storm
      - delta_n         : mean motion change (rev/day), positive = decayed
    """
    dh = before["altitude_km"] - after["altitude_km"]
    dn = after["mean_motion"]  - before["mean_motion"]
    return {
        "dh_true_km":    round(dh, 4),
        "alt_before_km": before["altitude_km"],
        "alt_after_km":  after["altitude_km"],
        "delta_n":       round(dn, 8),
        "epoch_before":  before["epoch"],
        "epoch_after":   after["epoch"],
    }


def physics_predict_dh(rho_kg_m3: float, Cd: float, area_m2: float,
                        mass_kg: float, alt_km: float,
                        duration_hours: float) -> float:
    """
    Pure physics baseline prediction of altitude drop (km).
    Uses standard drag equation integrated over storm duration.

    F_drag = 0.5 * rho * Cd * A * v²
    a_drag = F_drag / m
    dv_per_orbit = a_drag * (pi * r / v)
    dh_per_orbit = -2 * r * dv_per_orbit / v
    n_orbits = duration_sec / T
    dh_total = dh_per_orbit * n_orbits
    """
    r_m    = (R_EARTH_KM + alt_km) * 1000.0   # orbital radius in metres
    v_ms   = math.sqrt(GM_KM3_S2 * 1e9 / r_m)  # orbital velocity m/s
    rho_km = rho_kg_m3 / 1e9                   # kg/km³

    # Drag force N → acceleration m/s²
    F_drag   = 0.5 * rho_kg_m3 * Cd * area_m2 * v_ms**2
    a_drag   = F_drag / mass_kg  # m/s²

    # Velocity lost per orbit
    T_s         = 2 * math.pi * r_m / v_ms        # orbital period (s)
    dv_per_orb  = a_drag * (math.pi * r_m / v_ms) # m/s

    # Height drop per orbit
    dh_per_orb_km = 2 * (r_m / 1000) * dv_per_orb / (v_ms / 1000) / 1000

    # Total orbits during storm
    n_orbits = (duration_hours * 3600) / T_s

    return round(dh_per_orb_km * n_orbits, 4)
