"""
data_pipeline/config.py
Central configuration for the Celestis real-data pipeline.
"""

import os
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', 'backend', '.env'))

# ── Space-Track.org credentials (set in backend/.env) ───────────────────────
SPACETRACK_USER = os.getenv("SPACETRACK_USER", "")
SPACETRACK_PASS = os.getenv("SPACETRACK_PASS", "")
SPACETRACK_BASE = "https://www.space-track.org"

# ── Satellites to track (NORAD ID, launch year) ──────────────────────────────
# Expanded from 6 → 22 satellites for 550+ sample dataset.
# Chosen for: diverse altitudes (300–800km), diverse Cd*A/m ratios,
# confirmed gp_history coverage on Space-Track.
SATELLITES = {
    # ── Full coverage 1998+ ──────────────────────────────────────────────────
    "ISS":          (25544, 1998),   # 408 km | 444,615 kg | 2500 m² | always present
    "GRACE-A":      (27391, 2002),   # 480 km |     487 kg |   1.5 m² | drag research sat

    # ── Full coverage 2013+ ──────────────────────────────────────────────────
    "Swarm-A":      (39452, 2013),   # 460 km |     473 kg |   0.8 m²
    "Swarm-B":      (39451, 2013),   # 530 km |     473 kg |   0.8 m²
    "Swarm-C":      (39453, 2013),   # 460 km |     473 kg |   0.8 m²

    # ── 2010+ coverage ───────────────────────────────────────────────────────
    "CryoSat-2":    (36508, 2010),   # 717 km |     720 kg |   2.5 m²
    "GRACE-FO-1":   (43476, 2018),   # 490 km |     600 kg |   1.5 m²
    "GRACE-FO-2":   (43477, 2018),   # 490 km |     600 kg |   1.5 m²

    # ── 2017+ coverage ───────────────────────────────────────────────────────
    "Sentinel-2A":  (40697, 2015),   # 786 km |    1130 kg |  13.0 m²
    "Sentinel-2B":  (42063, 2017),   # 786 km |    1130 kg |  13.0 m²
    "Sentinel-6A":  (46984, 2020),   # 1336 km|    1130 kg |  10.2 m²

    # ── 2019+ coverage (Starlink / commercial) ───────────────────────────────
    "Starlink-0002":(44713, 2019),   # 550 km |     800 kg |  12.0 m²
    "Starlink-1007":(44914, 2019),   # 550 km |     800 kg |  12.0 m²
    "Starlink-2030":(47688, 2021),   # 550 km |     800 kg |  12.0 m²

    # ── 2021+ coverage ───────────────────────────────────────────────────────
    "Landsat-9":    (49260, 2021),   # 705 km |    5190 kg |  38.0 m²
    "ICESat-2":     (43613, 2018),   # 496 km |    1514 kg |   5.0 m²
    "IXPE":         (49954, 2021),   # 600 km |     330 kg |   2.0 m²

    # ── 2022+ coverage ───────────────────────────────────────────────────────
    "SWOT":         (54754, 2022),   # 890 km |    2300 kg |  20.0 m²
    "EarthCARE":    (59910, 2024),   # 393 km |    2300 kg |  10.0 m²

    # ── Special: dense LEO objects for atmospheric sampling ──────────────────
    "CHAMP":        (26405, 2000),   # 350 km |     522 kg |   0.8 m²
    "TerraSAR-X":   (31698, 2007),   # 514 km |    1230 kg |   4.0 m²
    "TanDEM-X":     (36605, 2010),   # 514 km |    1230 kg |   4.0 m²
}

# ── Satellite physical parameters for physics baseline ───────────────────────
SAT_PARAMS_EXTENDED = {
    25544: {"Cd": 2.2, "area_m2": 2500.0, "mass_kg": 444615.0},   # ISS
    27391: {"Cd": 2.2, "area_m2":    1.5, "mass_kg":    487.0},   # GRACE-A
    39452: {"Cd": 2.2, "area_m2":    0.8, "mass_kg":    473.0},   # Swarm-A
    39451: {"Cd": 2.2, "area_m2":    0.8, "mass_kg":    473.0},   # Swarm-B
    39453: {"Cd": 2.2, "area_m2":    0.8, "mass_kg":    473.0},   # Swarm-C
    36508: {"Cd": 2.2, "area_m2":    2.5, "mass_kg":    720.0},   # CryoSat-2
    43476: {"Cd": 2.2, "area_m2":    1.5, "mass_kg":    600.0},   # GRACE-FO-1
    43477: {"Cd": 2.2, "area_m2":    1.5, "mass_kg":    600.0},   # GRACE-FO-2
    40697: {"Cd": 2.2, "area_m2":   13.0, "mass_kg":   1130.0},   # Sentinel-2A
    42063: {"Cd": 2.2, "area_m2":   13.0, "mass_kg":   1130.0},   # Sentinel-2B
    46984: {"Cd": 2.2, "area_m2":   10.2, "mass_kg":   1130.0},   # Sentinel-6A
    44713: {"Cd": 2.2, "area_m2":   12.0, "mass_kg":    800.0},   # Starlink-0002
    44914: {"Cd": 2.2, "area_m2":   12.0, "mass_kg":    800.0},   # Starlink-1007
    47688: {"Cd": 2.2, "area_m2":   12.0, "mass_kg":    800.0},   # Starlink-2030
    49260: {"Cd": 2.2, "area_m2":   38.0, "mass_kg":   5190.0},   # Landsat-9
    43613: {"Cd": 2.2, "area_m2":    5.0, "mass_kg":   1514.0},   # ICESat-2
    49954: {"Cd": 2.2, "area_m2":    2.0, "mass_kg":    330.0},   # IXPE
    54754: {"Cd": 2.2, "area_m2":   20.0, "mass_kg":   2300.0},   # SWOT
    59910: {"Cd": 2.2, "area_m2":   10.0, "mass_kg":   2300.0},   # EarthCARE
    26405: {"Cd": 2.2, "area_m2":    0.8, "mass_kg":    522.0},   # CHAMP
    31698: {"Cd": 2.2, "area_m2":    4.0, "mass_kg":   1230.0},   # TerraSAR-X
    36605: {"Cd": 2.2, "area_m2":    4.0, "mass_kg":   1230.0},   # TanDEM-X
}

# ── Earth & physics constants ────────────────────────────────────────────────
R_EARTH_KM = 6371.0          # mean Earth radius, km
GM_KM3_S2  = 398600.4418     # gravitational parameter, km³/s²

# ── Storm event catalogue ─────────────────────────────────────────────────────
# Expanded from 19 → 26 events. New storms fill 2004-2010 gap.
STORM_EVENTS = [
    # ── G5 / Extreme ─────────────────────────────────────────────────────────
    ("Halloween_2003_A",   "2003-10-28", "2003-10-31", 9.0, 275, 2350, 9),
    ("Halloween_2003_B",   "2003-10-31", "2003-11-02", 9.0, 280, 2684, 9),
    ("Halloween_2003_C",   "2003-11-04", "2003-11-06", 8.5, 200, 1800, 9),
    ("Nov_2004_G5",        "2004-11-07", "2004-11-11", 9.0, 160, 1800, 9),
    ("Gannon_2024",        "2024-05-10", "2024-05-12", 9.0, 250, 2000, 9),

    # ── G4 / Severe ──────────────────────────────────────────────────────────
    ("Jan_2005_G4",        "2005-01-17", "2005-01-20", 8.7, 140, 2000, 9),
    ("Dec_2006_G3",        "2006-12-14", "2006-12-16", 8.0, 115, 1200, 9),
    ("Sep_2017_X9",        "2017-09-06", "2017-09-10", 8.0, 160, 1200, 7),
    ("StPatricks_2015",    "2015-03-17", "2015-03-20", 8.0, 150,  980, 7),
    ("Aug_2011",           "2011-08-05", "2011-08-07", 7.5, 180, 1300, 7),
    ("Jul_2012_NearMiss",  "2012-07-23", "2012-07-25", 6.0, 140,  880, 5),

    # ── G3 / Strong ──────────────────────────────────────────────────────────
    ("Bastille_2003",      "2003-07-15", "2003-07-17", 7.0, 170, 1500, 7),
    ("Jun_2015",           "2015-06-22", "2015-06-24", 7.0, 155, 1200, 7),
    ("Sep_2014",           "2014-09-11", "2014-09-13", 6.0, 130,  750, 5),
    ("Feb_2023",           "2023-02-26", "2023-02-28", 6.5, 175,  900, 7),
    ("Oct_2021_G3",        "2021-10-30", "2021-11-01", 7.0, 100,  900, 7),

    # ── G2 / Moderate ────────────────────────────────────────────────────────
    ("Mar_2023_G2",        "2023-03-23", "2023-03-25", 5.0, 120,  600, 3),
    ("Nov_2021_G2",        "2021-11-03", "2021-11-05", 5.0, 125,  550, 5),
    ("Dec_2023",           "2023-12-01", "2023-12-03", 6.0, 160,  700, 5),
    ("Aug_2018_G2",        "2018-08-25", "2018-08-27", 5.3, 110,  500, 3),
    ("Sep_2019_G2",        "2019-09-01", "2019-09-03", 5.0,  70,  400, 3),

    # ── G1 / Minor ───────────────────────────────────────────────────────────
    ("G1_Minor_2022a",     "2022-05-31", "2022-06-01", 4.0,  95,  380, 3),
    ("G1_Minor_2021",      "2021-05-12", "2021-05-13", 4.0, 100,  420, 3),

    # ── OOD / Held-out ───────────────────────────────────────────────────────
    ("Starlink_2022",      "2022-02-03", "2022-02-04", 5.5, 145,  420, 5),
    ("Quiet_2022",         "2022-08-10", "2022-08-11", 3.0,  85,  300, 1),
    ("Quiet_2023",         "2023-07-04", "2023-07-05", 3.5,  90,  260, 1),
]

# ── Held-out OOD test set (never used in training) ───────────────────────────
OOD_LABELS = {"Starlink_2022", "Gannon_2024", "Halloween_2003_A"}

# ── TLE query window offsets (days) ─────────────────────────────────────────
TLE_BEFORE_OFFSET_DAYS = 5    # pull TLE up to this many days before storm onset
TLE_AFTER_OFFSET_DAYS  = 7    # pull TLE up to this many days after storm end

# ── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_DIR          = os.path.join(os.path.dirname(__file__), "output")
RAW_TLE_CACHE       = os.path.join(OUTPUT_DIR, "tle_cache.json")
GROUND_TRUTH_CSV    = os.path.join(OUTPUT_DIR, "ground_truth_dataset.csv")
BENCHMARK_CSV       = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
PHASE_MODEL_PATH    = os.path.join(OUTPUT_DIR, "phase_aware_model.pkl")
