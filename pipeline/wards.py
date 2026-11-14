"""Ward boundaries for Bengaluru.

Order of preference:
1. data/raw/bengaluru_wards.geojson (243-ward BBMP/KGIS set, used here)
2. a public download of BBMP ward boundaries
3. a clearly labelled synthetic Voronoi tessellation (~200 cells)

All three are normalised to `ward_id` (string) and `ward_name` (string).
The ward source actually used is recorded in README.md and in `data_source`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
from shapely.geometry import MultiPolygon, Polygon, box

from pipeline import config

# Public mirrors tried in order when no local ward file exists.
WARD_URLS = [
    "https://raw.githubusercontent.com/datameet/Municipal_Spatial_Data/master/Bangalore/BBMP.geojson",
    "https://raw.githubusercontent.com/datameet/municipal_spatial_data/master/BBMP/ward.geojson",
]

ID_CANDIDATES = ("KGISWardNo", "ward_id", "ward_no", "WardNo", "Ward_No", "id", "WARD_ID")
NAME_CANDIDATES = ("KGISWardName", "ward_name", "wardname", "WardName", "Ward_Name", "name")


def _bbox_polygon() -> Polygon:
    b = config.BBOX
    return box(b["lon_min"], b["lat_min"], b["lon_max"], b["lat_max"])


def _normalise(raw: gpd.GeoDataFrame, source: str) -> gpd.GeoDataFrame:
    """Rename id/name columns, repair geometry, keep polygons only."""
    id_col = next((c for c in ID_CANDIDATES if c in raw.columns), None)
    name_col = next((c for c in NAME_CANDIDATES if c in raw.columns), None)
    if id_col is None or name_col is None:
        raise ValueError(f"cannot find ward id/name columns in {list(raw.columns)}")

    gdf = raw[[id_col, name_col, "geometry"]].rename(
        columns={id_col: "ward_id", name_col: "ward_name"}
    )
    gdf["ward_id"] = gdf["ward_id"].astype(str)
    gdf["ward_name"] = gdf["ward_name"].astype(str)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    gdf = gdf.to_crs(4326)
    # Some KGIS polygons self-intersect; buffer(0) is enough for a demo map.
    gdf["geometry"] = gdf.geometry.apply(lambda g: g if g.is_valid else g.buffer(0))
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    gdf = gdf[gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"])].copy()
    gdf["data_source"] = source
    return gdf.reset_index(drop=True)


def _download_wards() -> gpd.GeoDataFrame | None:
    import requests

    for url in WARD_URLS:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code != 200 or len(resp.content) < 10_000:
                continue
            config.RAW_DIR.mkdir(parents=True, exist_ok=True)
            config.WARD_FILE.write_bytes(resp.content)
            print(f"[wards] downloaded {url} -> {config.WARD_FILE.name}")
            return _normalise(gpd.read_file(config.WARD_FILE), "bbmp_wards_real")
        except Exception as exc:  # noqa: BLE001 - network is best effort
            print(f"[wards] download failed ({url}): {exc}")
    return None


def _synthetic_wards(n: int = 200, seed: int = config.RANDOM_SEED) -> gpd.GeoDataFrame:
    """Jittered Voronoi tessellation of the bbox -> ~n fake wards."""
    from shapely.ops import voronoi_diagram

    rng = np.random.default_rng(seed)
    b = config.BBOX
    pts = gpd.GeoSeries(
        gpd.points_from_xy(
            rng.uniform(b["lon_min"], b["lon_max"], n),
            rng.uniform(b["lat_min"], b["lat_max"], n),
        ),
        crs=4326,
    )
    cells = voronoi_diagram(pts.union_all(), envelope=_bbox_polygon())
    polys = list(getattr(cells, "geoms", []))
    gdf = gpd.GeoDataFrame(
        {
            "ward_id": [f"S{i:03d}" for i in range(1, len(polys) + 1)],
            "ward_name": [f"Ward {i:03d} (synthetic)" for i in range(1, len(polys) + 1)],
        },
        geometry=polys,
        crs=4326,
    )
    gdf = gpd.clip(gdf, _bbox_polygon())
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["data_source"] = "synthetic"
    return gdf


def load_wards(force_synthetic: bool = False) -> gpd.GeoDataFrame:
    """Return ward polygons normalised to ward_id/ward_name."""
    if not force_synthetic and config.WARD_FILE.exists():
        gdf = _normalise(gpd.read_file(config.WARD_FILE), "bbmp_243_real")
        print(f"[wards] using local {config.WARD_FILE} ({len(gdf)} wards, 243-ward BBMP set)")
        return gdf
    if not force_synthetic:
        gdf = _download_wards()
        if gdf is not None:
            print(f"[wards] using downloaded ward set ({len(gdf)} wards)")
            return gdf
    print("[wards] !! NO REAL WARD DATA - using SYNTHETIC Voronoi wards !!")
    gdf = _synthetic_wards()
    config.WARD_FILE.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(config.WARD_FILE, driver="GeoJSON")
    return gdf


def main() -> None:
    gdf = load_wards()
    print(f"wards={len(gdf)} source={gdf['data_source'].iloc[0]}")
    print(f"area_km2={gdf.to_crs(32643).area.sum() / 1e6:.1f}")
    print(gdf[["ward_id", "ward_name"]].head().to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
