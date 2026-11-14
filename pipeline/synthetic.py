"""Synthetic ~150 m LST / land-cover / population grid for Bengaluru.

This exists so the whole pipeline and app can run end to end today. Nothing here
is a measurement: every row is labelled data_source="synthetic". The spatial
structure is hand-tuned to look like Bengaluru (hot compact core, industrial
pockets, parks and lakes, dense fringe belts) so the downstream ranking is
meaningful as a demo, not as a claim about the real city.

LST is generated as a noisy function of NDVI (cooling), NDBI / building
density / road density (heating) and NDWI (cooling), with spatially correlated
noise and a spatially varying NDVI->LST slope, so local cooling efficiency
genuinely differs between wards.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from pipeline import config
from pipeline.wards import load_wards

# --- Landmarks (lat, lon, radius_m, strength) ---------------------------------
URBAN_BUMPS = [
    (12.972, 77.578, 3800, 0.85),  # Majestic / Shivajinagar / KR Market core
    (13.030, 77.520, 1700, 0.55),  # Peenya industrial
    (12.980, 77.750, 2300, 0.55),  # Whitefield / ITPL
    (12.845, 77.670, 1700, 0.50),  # Electronic City
    (12.930, 77.620, 2000, 0.40),  # Koramangala / BTM
    (13.000, 77.640, 1800, 0.40),  # Indiranagar
    (12.955, 77.700, 1600, 0.35),  # Marathahalli
    (13.070, 77.640, 1500, 0.35),  # Yelahanka / Thanisandra
    (12.905, 77.560, 1800, 0.30),  # JP Nagar
    (13.050, 77.500, 1200, 0.30),  # Nagasandra / Peenya north
]

PARKS = [
    (12.9763, 77.5929, 750, 0.95),   # Cubbon Park
    (12.9507, 77.5848, 900, 0.95),   # Lalbagh
    (13.0190, 77.5660, 1000, 0.85),  # IISc
    (13.0700, 77.5800, 2000, 0.90),  # GKVK
    (12.8600, 77.5800, 5200, 0.90),  # Bannerghatta
    (12.9400, 77.5000, 1400, 0.80),  # Turahalli
    (12.9080, 77.4850, 1300, 0.75),  # Kengeri / Uttarahalli edge
    (12.9950, 77.6800, 900, 0.70),   # Indiranagar defence land
]

LAKES = [
    (12.9825, 77.6190, 520, 1.0),  # Ulsoor
    (12.9300, 77.6700, 950, 1.0),  # Bellandur
    (13.0450, 77.5900, 620, 1.0),  # Hebbal
    (13.0070, 77.5700, 420, 1.0),  # Sankey Tank
    (12.9180, 77.6180, 380, 1.0),  # Madiwala
    (12.9400, 77.7400, 800, 1.0),  # Varthur
    (12.9220, 77.6420, 300, 1.0),  # Agara
    (12.8950, 77.5100, 400, 1.0),  # Kengeri
]

ROAD_ANGLES_DEG = [0, 32, 68, 104, 140, 176, 212, 248, 284, 320]
RING_RADIUS_M = 7500.0  # outer ring road


# --- helpers ------------------------------------------------------------------
def build_grid() -> tuple[np.ndarray, np.ndarray, int, int]:
    """Cell-centre lon/lat arrays (flattened) plus grid shape."""
    b = config.BBOX
    step = config.GRID_SIZE_M_SYNTHETIC
    lat0 = (b["lat_min"] + b["lat_max"]) / 2
    dlat = step / 110_574.0
    dlon = step / (111_320.0 * np.cos(np.radians(lat0)))
    lats = np.arange(b["lat_min"], b["lat_max"], dlat) + dlat / 2
    lons = np.arange(b["lon_min"], b["lon_max"], dlon) + dlon / 2
    lon2d, lat2d = np.meshgrid(lons, lats)
    return lon2d.ravel(), lat2d.ravel(), len(lats), len(lons)


def metric_xy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local equirectangular metres around the study-area centre."""
    lat0 = (config.BBOX["lat_min"] + config.BBOX["lat_max"]) / 2
    lon0 = (config.BBOX["lon_min"] + config.BBOX["lon_max"]) / 2
    x = (lon - lon0) * 111_320.0 * np.cos(np.radians(lat0))
    y = (lat - lat0) * 110_574.0
    return x, y


def bumps(x: np.ndarray, y: np.ndarray, specs: list[tuple[float, float, float, float]]) -> np.ndarray:
    """Sum of isotropic Gaussian bumps."""
    total = np.zeros_like(x)
    for clat, clon, radius, amp in specs:
        cx, cy = metric_xy(np.array([clon]), np.array([clat]))
        d2 = (x - cx[0]) ** 2 + (y - cy[0]) ** 2
        total += amp * np.exp(-d2 / (2.0 * radius**2))
    return total


def soft_clip(values: np.ndarray, lo: float, hi: float, k: float = 2.0) -> np.ndarray:
    """Compress values into [lo, hi] with tanh tails instead of hard clipping."""
    hi_knee, lo_knee = hi - k, lo + k
    out = np.where(values > hi_knee, hi_knee + k * np.tanh((values - hi_knee) / k), values)
    out = np.where(out < lo_knee, lo_knee - k * np.tanh((lo_knee - out) / k), out)
    return out


def smooth(rng: np.random.Generator, rows: int, cols: int, sigma_cells: float, amp: float) -> np.ndarray:
    """Standardised spatially correlated noise with std == amp."""
    raw = gaussian_filter(rng.normal(size=(rows, cols)), sigma=sigma_cells, mode="nearest")
    flat = raw.ravel()
    flat = (flat - flat.mean()) / (flat.std() + 1e-9)
    return flat * amp


def road_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Distance (m) to the nearest radial arterial or the ring road."""
    dist = np.full_like(x, 1e9)
    for ang in ROAD_ANGLES_DEG:
        ux, uy = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        proj = np.maximum(x * ux + y * uy, 0.0)  # half-line, clipped at centre
        dist = np.minimum(dist, np.hypot(x - proj * ux, y - proj * uy))
    dist = np.minimum(dist, np.abs(np.hypot(x, y) - RING_RADIUS_M))
    return dist


def generate() -> pd.DataFrame:
    """Build the full synthetic cell table (before the ward join)."""
    rng = np.random.default_rng(config.RANDOM_SEED)
    lon, lat, rows, cols = build_grid()
    x, y = metric_xy(lon, lat)
    r = np.hypot(x, y)

    # Urban intensity: named hotspots + diffuse decay from the centre.
    urban = bumps(x, y, URBAN_BUMPS) + 0.35 * np.exp(-r / 6500.0)
    urban += smooth(rng, rows, cols, 6, 0.05)
    urban = np.clip(urban, 0.0, 1.0)

    parks = np.clip(bumps(x, y, PARKS), 0.0, 1.0)
    water_field = np.clip(bumps(x, y, LAKES), 0.0, 1.0)
    is_water = water_field > 0.45

    # Roads follow burnt-in radial corridors + ring road.
    road_field = np.exp(-road_distance(x, y) / 110.0)
    road_density = np.clip(0.85 * road_field + 0.30 * urban + smooth(rng, rows, cols, 2, 0.05), 0, 1)

    # Vegetation: greener at the fringe, stripped from the dense core, boosted in parks.
    base_green = 0.30 + 0.18 * (1.0 - np.exp(-r / 8000.0))
    ndvi = (
        base_green * (1.0 - 0.55 * urban)
        + 0.62 * parks
        - 0.22 * water_field
        + smooth(rng, rows, cols, 8, 0.05)   # neighbourhood-scale variation
        + smooth(rng, rows, cols, 2.5, 0.10)  # plot/street-scale heterogeneity
        + smooth(rng, rows, cols, 1.5, 0.035)
    )
    ndvi = np.clip(ndvi, 0.02, 0.88)

    ndbi = (
        -0.32
        + 0.70 * urban
        + 0.20 * road_density
        + 0.10 * smooth(rng, rows, cols, 3, 1.0)
        - 0.25 * parks
        - 0.18 * water_field
        + smooth(rng, rows, cols, 2, 0.04)
    )
    ndbi = np.clip(ndbi, -0.65, 0.65)

    ndwi = np.clip(-0.28 + 0.85 * water_field - 0.05 * urban + smooth(rng, rows, cols, 2, 0.04), -0.7, 0.7)

    building_density = 0.05 + 0.78 * urban + 0.12 * road_density - 0.30 * parks
    building_density += smooth(rng, rows, cols, 2, 0.05)
    building_density = np.clip(building_density, 0, 1) * np.where(is_water, 0.02, 1.0)

    # Population: dense core plus a residential belt at ~5 km, empty water.
    belt = np.exp(-(((r - 5200.0) / 3000.0) ** 2))
    # People per 100 m cell (~10,000 m2): ~0-1,600, total ~8-9 M over BBMP.
    pop = 12.0 + 385.0 * urban**1.5 + 260.0 * belt
    pop *= (1.0 - 0.85 * parks) * (1.0 - 0.97 * is_water)
    pop *= 1.0 + 0.25 * smooth(rng, rows, cols, 2, 1.0)
    population = np.clip(np.round(pop), 0, 1600).astype(int)

    # Spatially varying NDVI->LST sensitivity (C per +1 NDVI). Negative = cooling.
    # Low-frequency field so neighbouring cells share a regime but wards differ.
    sens_field = smooth(rng, rows, cols, 26, 1.0)
    sens = 3.2 + 1.5 * (sens_field + 1.0)  # ~3.2 to ~6.2 C per unit NDVI
    slope_field = -np.clip(sens, 3.0, 6.5)

    ambient = smooth(rng, rows, cols, 12, 1.0) + smooth(rng, rows, cols, 3, 0.4)
    lst = (
        36.6
        + 6.5 * ndbi
        + 4.8 * building_density
        + 2.4 * road_density
        + slope_field * ndvi
        - 3.4 * ndwi
        + 2.0 * urban**2
        + ambient
    )
    # Soft-clip to the realistic summer band (~30-48 C) without a flat plateau.
    lst = np.round(soft_clip(lst, 30.0, 48.0, k=2.0), 2)

    plantable = (
        (~is_water)
        & (building_density < 0.55)
        & (ndbi < 0.25)
        & (road_density < 0.50)
        & (ndvi < 0.72)
    ).astype(int)

    df = pd.DataFrame(
        {
            "lon": np.round(lon, 5),
            "lat": np.round(lat, 5),
            "lst": lst,
            "ndvi": np.round(ndvi, 3),
            "ndbi": np.round(ndbi, 3),
            "ndwi": np.round(ndwi, 3),
            "building_density": np.round(building_density, 3),
            "road_density": np.round(road_density, 3),
            "population": population,
            "plantable": plantable,
            "data_source": "synthetic",
        }
    )
    df.insert(0, "cell_id", np.arange(1, len(df) + 1))
    return df


def assign_wards(df: pd.DataFrame, wards: gpd.GeoDataFrame) -> pd.DataFrame:
    """Spatial join cells to wards; cells outside every ward are dropped."""
    cells = gpd.GeoDataFrame(
        df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326
    )
    joined = gpd.sjoin(
        cells,
        wards[["ward_id", "ward_name", "geometry"]],
        how="inner",
        predicate="intersects",
    )
    joined = joined.drop(columns=["geometry", "index_right"])
    joined = joined.drop_duplicates(subset="cell_id", keep="first")
    return pd.DataFrame(joined)


def main() -> None:
    wards = load_wards()
    df = generate()
    n_all = len(df)
    print(f"[synthetic] grid cells before ward join: {n_all}")
    print(f"[synthetic] LST min/mean/p95/max: "
          f"{df['lst'].min():.1f} / {df['lst'].mean():.1f} / "
          f"{df['lst'].quantile(0.95):.1f} / {df['lst'].max():.1f} C")
    print(f"[synthetic] NDVI mean {df['ndvi'].mean():.3f}; "
          f"plantable {df['plantable'].mean() * 100:.1f}% of cells; "
          f"population total {df['population'].sum() / 1e6:.2f} M")

    out = assign_wards(df, wards)
    out = out.sort_values("cell_id").reset_index(drop=True)
    out.to_csv(config.GRID_CSV, index=False)

    print(f"[synthetic] cells inside wards: {len(out)} (dropped {n_all - len(out)})")
    print(f"[synthetic] wards covered: {out['ward_id'].nunique()} / {len(wards)}")
    print(f"[synthetic] wrote {config.GRID_CSV} "
          f"({config.GRID_CSV.stat().st_size / 1e6:.2f} MB, {len(out.columns)} cols)")
    print(out.head(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
