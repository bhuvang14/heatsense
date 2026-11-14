"""HeatSense: real-data export from Google Earth Engine (OPTIONAL upgrade path).

Produces the SAME CSV schema as pipeline/synthetic.py:
    cell_id, lon, lat, lst, ndvi, ndbi, ndwi, building_density,
    road_density, population, plantable, data_source
so the rest of the pipeline runs unchanged.

Usage (needs a Google Earth Engine account):
    pip install earthengine-api
    earthengine authenticate
    python gee/export.py                       # submits a Drive export task
    # then download the CSV from Drive to data/grid_features.csv
    ./run_all.sh                               # rerun pipeline + app

Notes / documented substitutes (no OSM road layer exists in the GEE catalogue):
* building_density = fraction of the cell covered by ESA WorldCover built-up
  (class 50). This replaces a building-footprint rasterisation (Open Buildings
  polygons are not rasterised here to keep the export tractable).
* road_density = a PROXY, not a road survey: Dynamic World "built" probability
  minus the built-up fraction, i.e. largely transport/other impervious surface.
* plantable = open vegetation classes from WorldCover, minus water/built-up,
  AND NDBI below a threshold (per the brief).
"""

from __future__ import annotations

import sys

try:
    import ee
except ImportError:  # optional dependency, only needed for this script
    ee = None

# --- Study area (must match pipeline/config.py) --------------------------------
BBOX = {"lon_min": 77.45, "lon_max": 77.78, "lat_min": 12.83, "lat_max": 13.14}
GRID_SCALE_M = 100  # ~100 m cells
YEARS = ("2019-01-01", "2025-06-01")  # Landsat 8/9 + S2 era
SUMMER = ("-03-01", "-05-31")  # summer composite window: Mar 1 - May 31
OUT_DESCRIPTION = "heatsense_bengaluru_grid"


def aoi() -> ee.Geometry:
    return ee.Geometry.Rectangle(
        [BBOX["lon_min"], BBOX["lat_min"], BBOX["lon_max"], BBOX["lat_max"]]
    )


def mask_landsat_l2(image: ee.Image) -> ee.Image:
    """Cloud-mask Landsat C2 L2 with QA_PIXEL; return surface temperature in C."""
    qa = image.select("QA_PIXEL")
    # bits 1 (dilated cloud), 3 (cloud), 4 (cloud shadow)
    clear = qa.bitwiseAnd(0b11010).eq(0)
    lst_c = (
        image.select("ST_B10")
        .multiply(0.00341802)
        .add(149.0)
        .subtract(273.15)  # Kelvin -> Celsius
        .rename("LST")
    )
    return lst_c.updateMask(clear).copyProperties(image, ["system:time_start"])


def landsat_lst(region: ee.Geometry) -> ee.Image:
    l8 = (
        ee.ImageCollection("LANDSAT/LC08/C02/T1_L2")
        .filterBounds(region)
        .filterDate(YEARS[0], YEARS[1])
        .filter(ee.Filter.calendarRange(3, 5, "month"))
        .map(mask_landsat_l2)
    )
    l9 = (
        ee.ImageCollection("LANDSAT/LC09/C02/T1_L2")
        .filterBounds(region)
        .filterDate(YEARS[0], YEARS[1])
        .filter(ee.Filter.calendarRange(3, 5, "month"))
        .map(mask_landsat_l2)
    )
    return l8.merge(l9).median().rename("LST")


def mask_s2_sr(image: ee.Image) -> ee.Image:
    """Cloud/shadow mask Sentinel-2 SR with the SCL band."""
    scl = image.select("SCL")
    # 1 saturated, 3 cloud shadow, 8/9 cloud medium/high, 10 cirrus, 11 snow
    bad = scl.remap([1, 3, 8, 9, 10, 11], [1, 1, 1, 1, 1, 1], 0)
    return image.updateMask(bad.eq(0))


def sentinel2_indices(region: ee.Geometry) -> ee.Image:
    s2 = (
        ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
        .filterBounds(region)
        .filterDate(YEARS[0], YEARS[1])
        .filter(ee.Filter.calendarRange(3, 5, "month"))
        .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 60))
        .map(mask_s2_sr)
        .median()
    )
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    ndbi = s2.normalizedDifference(["B11", "B8"]).rename("NDBI")
    ndwi = s2.normalizedDifference(["B3", "B8"]).rename("NDWI")
    return ee.Image.cat([ndvi, ndbi, ndwi])


def worldpop(region: ee.Geometry) -> ee.Image:
    """People per ~100 m pixel from the latest WorldPop year available."""
    pop = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filterBounds(region)
        .sort("system:time_start", False)
        .first()
    )
    return pop.select("population").rename("POP")


def surface_fractions(region: ee.Geometry) -> ee.Image:
    """Built-up fraction, road proxy and plantable mask (see module docstring)."""
    worldcover = ee.ImageCollection("ESA/WorldCover/v200").first().select("Map")
    built_frac = worldcover.eq(50).rename("BUILT_FRAC")
    open_veg = worldcover.remap([20, 30, 40, 60], [1, 1, 1, 1], 0).rename("OPEN_VEG")

    dw_built = (
        ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
        .filterBounds(region)
        .filterDate(YEARS[0], YEARS[1])
        .select("built")
        .median()
        .rename("DW_BUILT")
    )
    road_proxy = dw_built.subtract(built_frac).clamp(0, 1).rename("ROAD_PROXY")
    return ee.Image.cat([built_frac, open_veg, road_proxy])


def predictor_stack(region: ee.Geometry) -> ee.Image:
    lst = landsat_lst(region)
    idx = sentinel2_indices(region)
    pop = worldpop(region)
    fracs = surface_fractions(region)
    # Plantable: open vegetation, low built fraction and low NDBI (per brief).
    plantable = (
        fracs.select("OPEN_VEG").gt(0.5)
        .And(fracs.select("BUILT_FRAC").lt(0.2))
        .And(idx.select("NDBI").lt(0.1))
        .rename("PLANTABLE")
    )
    return ee.Image.cat([lst, idx, pop, fracs, plantable]).float()


def grid_cells(region: ee.Geometry) -> ee.FeatureCollection:
    """One ~100 m polygon per cell, labelled with a unique integer id."""
    proj = ee.Projection("EPSG:32643")  # UTM 43N over Bengaluru
    coords = ee.Image.pixelCoordinates(proj)
    col = coords.select("x").divide(GRID_SCALE_M).floor()
    row = coords.select("y").divide(GRID_SCALE_M).floor()
    key = col.multiply(1e5).add(row).toInt64().rename("cell_id")
    return key.reduceToVectors(
        geometry=region,
        crs=proj,
        scale=GRID_SCALE_M,
        geometryType="polygon",
        labelProperty="cell_id",
        reducer=ee.Reducer.first(),
        maxPixels=1e9,
        tileScale=4,
    )


def build_export() -> ee.batch.Task:
    region = aoi()
    stack = predictor_stack(region)
    stats = stack.reduceRegions(
        collection=grid_cells(region),
        reducer=ee.Reducer.median().combine(ee.Reducer.mean(), sharedInputs=True),
        scale=GRID_SCALE_M,
        crs=ee.Projection("EPSG:32643"),
        tileScale=8,
    )

    def to_row(f: ee.Feature) -> ee.Feature:
        centroid = f.geometry().centroid(1)
        return ee.Feature(
            None,
            {
                "cell_id": f.get("cell_id"),
                "lon": centroid.coordinates().get(0),
                "lat": centroid.coordinates().get(1),
                "lst": f.get("LST_median"),
                "ndvi": f.get("NDVI_median"),
                "ndbi": f.get("NDBI_median"),
                "ndwi": f.get("NDWI_median"),
                "building_density": f.get("BUILT_FRAC_mean"),
                "road_density": f.get("ROAD_PROXY_mean"),
                "population": f.get("POP_mean"),
                # reduceRegions gives a fraction here; discretise to 0/1 (>0.5 = plantable).
                "plantable": ee.Algorithms.If(
                    ee.Number(f.get("PLANTABLE_mean")).gt(0.5), 1, 0
                ),
                "data_source": "gee",
            },
        )

    rows = stats.map(to_row).filter(
        ee.Filter.notNull(["lst", "ndvi", "ndbi", "ndwi", "population"])
    )
    return ee.batch.Export.table.toDrive(
        collection=rows,
        description=OUT_DESCRIPTION,
        folder="heatsense",
        fileNamePrefix=OUT_DESCRIPTION,
        fileFormat="CSV",
        selectors=[
            "cell_id", "lon", "lat", "lst", "ndvi", "ndbi", "ndwi",
            "building_density", "road_density", "population", "plantable", "data_source",
        ],
    )


def main() -> int:
    if ee is None:
        print(
            "[gee] the Earth Engine SDK is not installed.\n"
            "      fix: pip install earthengine-api==1.6.4 && earthengine authenticate\n"
            "      Nothing was submitted."
        )
        return 1
    try:
        ee.Initialize()
    except Exception as exc:  # noqa: BLE001 - auth is the usual failure
        print(
            "[gee] Earth Engine is not initialised.\n"
            f"      reason: {exc}\n"
            "      fix: pip install earthengine-api && earthengine authenticate\n"
            "      (or set a service-account key with ee.Initialize(credentials)).\n"
            "      Nothing was submitted."
        )
        return 1

    task = build_export()
    task.start()
    print(f"[gee] export task submitted: {task.id}")
    print(f"[gee] description={OUT_DESCRIPTION}; monitor at "
          "https://code.earthengine.google.com/tasks")
    print("[gee] when it finishes, download the CSV to data/grid_features.csv "
          "and rerun ./run_all.sh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
