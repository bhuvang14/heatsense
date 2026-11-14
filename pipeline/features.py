"""Load the cell grid, validate it, and compute ward-level aggregates.

Works unchanged on the synthetic CSV and on the GEE export (same schema).
If `ward_id` is missing (fresh GEE export), cells are joined to wards here.
"""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from pipeline import config
from pipeline.wards import load_wards

CELL_AREA_M2 = config.GRID_SIZE_M_SYNTHETIC**2  # 100 m x 100 m = 10,000 m2
REQUIRED = ["cell_id", "lon", "lat", "lst", "ndvi", "ndbi", "ndwi",
            "building_density", "road_density", "population", "plantable", "data_source"]
NUMERIC = ["lon", "lat", "lst", "ndvi", "ndbi", "ndwi",
           "building_density", "road_density", "population"]

# Physical bounds used to clip impossible values (none should occur in synthetic).
CLIP = {
    "lst": (15.0, 65.0),
    "ndvi": (-1.0, 1.0),
    "ndbi": (-1.0, 1.0),
    "ndwi": (-1.0, 1.0),
    "building_density": (0.0, 1.0),
    "road_density": (0.0, 1.0),
    "population": (0.0, 5000.0),  # people per ~100 m cell
}


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema, coerce types, drop invalid rows, clip outliers."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"grid CSV is missing required columns: {missing}")

    df = df.copy()
    df["ward_id"] = df.get("ward_id", pd.Series(index=df.index, dtype=str)).astype(str)
    df["ward_name"] = df.get("ward_name", pd.Series(index=df.index, dtype=str)).astype(str)
    # Accept both a 0/1 flag (synthetic) and a 0-1 fraction (raw GEE export).
    plantable = pd.to_numeric(df["plantable"], errors="coerce").fillna(0)
    df["plantable"] = (plantable >= 0.5).astype(int) if plantable.max() <= 1.0 else plantable.astype(int)

    n0 = len(df)
    df = df.dropna(subset=NUMERIC + ["cell_id"])
    df = df[(df["lon"].between(77.0, 78.3)) & (df["lat"].between(12.5, 13.5))]
    df = df[df["plantable"].isin([0, 1])]
    df = df.drop_duplicates(subset="cell_id")
    print(f"[features] dropped {n0 - len(df)} invalid/duplicate rows of {n0}")

    for col, (lo, hi) in CLIP.items():
        n_clipped = int((~df[col].between(lo, hi)).sum())
        if n_clipped:
            print(f"[features] clipped {n_clipped} {col} outliers into [{lo}, {hi}]")
        df[col] = df[col].clip(lo, hi)

    sources = sorted(df["data_source"].dropna().astype(str).unique())
    print(f"[features] data_source={sources}")
    if len(sources) > 1:
        print("[features] WARNING: mixed data sources in one grid")
    return df.reset_index(drop=True)


def attach_wards(df: pd.DataFrame) -> pd.DataFrame:
    """Join cells to wards when ward_id is absent or empty."""
    if "ward_id" in df.columns and df["ward_id"].notna().all() and (df["ward_id"] != "nan").all():
        return df
    print("[features] ward_id missing -> joining cells to wards")
    wards = load_wards()[["ward_id", "ward_name", "geometry"]]
    cells = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs=4326)
    joined = gpd.sjoin(cells, wards, how="inner", predicate="intersects")
    joined = joined.drop(columns=["geometry", "index_right"]).drop_duplicates("cell_id")
    return pd.DataFrame(joined).reset_index(drop=True)


def ward_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    """Per-ward means, totals, plantable area and population density."""
    wards = load_wards()
    area = wards.copy()
    area["area_km2"] = area.to_crs(32643).area / 1e6

    agg = (
        df.groupby("ward_id", as_index=False)
        .agg(
            lst_mean=("lst", "mean"),
            ndvi_mean=("ndvi", "mean"),
            ndbi_mean=("ndbi", "mean"),
            population=("population", "sum"),
            plantable_cells=("plantable", "sum"),
            n_pixels=("cell_id", "size"),
            data_source=("data_source", "first"),
        )
    )
    agg["plantable_area_m2"] = agg["plantable_cells"] * CELL_AREA_M2
    out = agg.merge(area[["ward_id", "ward_name", "area_km2"]], on="ward_id", how="left")
    out["pop_density"] = out["population"] / out["area_km2"].replace(0, np.nan)
    out = out.drop(columns=["plantable_cells", "area_km2"])
    out = out.sort_values("lst_mean", ascending=False).reset_index(drop=True)

    print(f"[features] wards aggregated: {len(out)}; "
          f"cells {out['n_pixels'].sum()}, "
          f"population {out['population'].sum() / 1e6:.2f} M")
    return out


def load_grid() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Public helper: validated cell table + ward aggregates."""
    if not config.GRID_CSV.exists():
        raise FileNotFoundError(f"{config.GRID_CSV} not found - run pipeline/synthetic.py first")
    df = validate(pd.read_csv(config.GRID_CSV))
    df = attach_wards(df)
    return df, ward_aggregates(df)


def main() -> None:
    df, wards = load_grid()
    df.to_csv(config.GRID_CSV, index=False)
    wards.to_csv(config.WARD_AGG_CSV, index=False)
    print(f"[features] wrote {config.GRID_CSV.name} ({len(df)} rows) and "
          f"{config.WARD_AGG_CSV.name} ({len(wards)} rows)")
    print(wards.head(5)[["ward_id", "ward_name", "lst_mean", "ndvi_mean", "population"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
