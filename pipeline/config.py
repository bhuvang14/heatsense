"""HeatSense study-area constants (Bengaluru only, no multi-city config)."""

from __future__ import annotations

from pathlib import Path

# --- Study area ---------------------------------------------------------------
CITY = "Bengaluru"
MAP_CENTER = (12.97, 77.59)  # lat, lon
BBOX = {
    "lon_min": 77.45,
    "lon_max": 77.78,
    "lat_min": 12.83,
    "lat_max": 13.14,
}

# Target grid is ~100 m. Generating the full bbox at 100 m gives ~122k cells,
# but only cells inside BBMP wards are kept (~70k), which stays inside the
# 60k-80k cap for the saved dataset. The GEE export uses the same ~100 m grid.
GRID_SIZE_M_SYNTHETIC = 100
GRID_SIZE_M_GEE = 100

SUMMER_WINDOW = ("03-01", "05-31")  # summer composite, March 1 to May 31
RANDOM_SEED = 42

# --- Paths --------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
WARD_FILE = RAW_DIR / "bengaluru_wards.geojson"
GRID_CSV = DATA_DIR / "grid_features.csv"
WARD_AGG_CSV = DATA_DIR / "ward_features.csv"
WARDS_GEOJSON = DATA_DIR / "wards.geojson"
METRICS_JSON = DATA_DIR / "model_metrics.json"
SHAP_GLOBAL_CSV = DATA_DIR / "shap_global.csv"
SHAP_WARD_JSON = DATA_DIR / "shap_by_ward.json"
COOLING_CSV = DATA_DIR / "cooling_by_ward.csv"
TOP10_HOTTEST_JSON = DATA_DIR / "top10_hottest.json"
TOP10_PLANTING_JSON = DATA_DIR / "top10_planting.json"
SUMMARY_CACHE = DATA_DIR / "summary_cache.json"

FEATURES = [
    "ndvi",
    "ndbi",
    "ndwi",
    "building_density",
    "road_density",
    "population",
]

DATA_LABEL = "LST is surface heat, not felt heat. Cooling estimates are correlational, not field-measured."

for _d in (DATA_DIR, RAW_DIR):
    _d.mkdir(parents=True, exist_ok=True)
