# HeatSense — Bengaluru

**One-line claim:** A spatially validated planting prioritiser that ranks where a tree cools most
for the most exposed people.

LST is surface heat, not felt heat. Cooling estimates are correlational, not field-measured.

---

## What it does

1. Builds a ~100 m grid of Bengaluru (70,910 cells inside BBMP wards) with land surface temperature,
   vegetation/built-up/water indices, building and road density, population and a plantability mask.
2. Predicts LST from those drivers with XGBoost, validated with **spatial** block CV (whole wards
   held out), and explains it with SHAP.
3. Estimates a **local cooling efficiency** per ward (OLS slope of LST on NDVI) with a documented
   fallback when a ward's sample is too small or too noisy.
4. Ranks planting candidates by `cooling efficiency × plantable area × exposed population`, enforces
   1 km spacing, and flags heat-equity wards (hot + dense + low green cover).
5. Serves an offline-capable Leaflet app with a ward choropleth, top-10 lists, planting markers,
   an NDVI "what-if" slider, and the model card.

## Study area and ward set

- Bengaluru only, hardcoded: centre `12.97, 77.59`; bbox `77.45–77.78 E`, `12.83–13.14 N`;
  summer composite window **1 March – 31 May**.
- **Ward set used: the real 243-ward BBMP delimitation** (KGIS ward codes, published via DataMeet,
  downloaded once from `datameet/Municipal_Spatial_Data` and cached at
  `data/raw/bengaluru_wards.geojson`). 704 km² total, 243 wards, all cells fall inside a ward.
  No synthetic wards were needed.

## Quickstart

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt      # macOS: `brew install libomp` first if xgboost import fails
./run_all.sh                         # generates data + trains model + writes GeoJSON/JSON
python app/app.py                    # http://localhost:5000
```

Windows: use `run_all.bat`. Python 3.10+ is fine; this was built and verified on 3.12.

Run time on a laptop: ~15 s for the whole pipeline from scratch (synthetic 2 s, model ~8 s, rest < 2 s).

If port 5000 is already taken (macOS AirPlay Receiver often listens there), use
`PORT=5001 python app/app.py`.

## What the pipeline produces (this run)

| Artefact | Contents |
|---|---|
| `data/grid_features.csv` | 70,910 cells, 100 m, columns: `cell_id, lon, lat, lst, ndvi, ndbi, ndwi, building_density, road_density, population, plantable, data_source, ward_id, ward_name` |
| `data/ward_features.csv` | per-ward means/totals + population density |
| `data/model_metrics.json` | spatial CV per fold + pooled, grouping method, city-wide NDVI slope, fallback rule |
| `data/shap_global.csv` | mean |SHAP| per driver (C) and share |
| `data/shap_by_ward.json` | top-3 drivers per ward with direction (`hot` / `cooling`) and share |
| `data/cooling_by_ward.csv` | per-ward LST~NDVI slope, R², n, fallback flag, slope used |
| `data/top10_planting.json` | 10 pixel-level planting spots with scores and expected cooling |
| `data/top10_hottest.json` | 10 hottest wards by mean LST |
| `data/wards.geojson` | 243 ward polygons + all ward properties + summary text |
| `data/meta.json` | app header/banner data, model card facts |

Model (synthetic data): **pooled out-of-fold R² 0.953, RMSE 0.84 °C** over 5 ward-grouped folds
(fold R² 0.960/0.942/0.939/0.948/0.957). City-wide NDVI slope **−13.4 °C per +1.0 NDVI** (R² 0.53);
**129 of 243 wards** use the city-wide fallback under the stated rule.

LST range in the synthetic grid: 30–48 °C; population total 8.95 M over the 243 wards.

### How the ranking works

- **Candidates:** `plantable = 1`, LST above the ward's 75th percentile, NDVI below the city's 40th
  percentile (6,143 cells across 154 wards). Cells are 100 m.
- **Pixel score:** min-max normalised `|local slope|` × `plantable area` × `population`, multiplied,
  so no factor dominates by units. Population is smoothed over a 3×3 (~300 m) neighbourhood.
- **Top 10 spots:** greedy by score with a **1 km minimum spacing** so they are not all neighbours.
- **Ward priority:** **sum** of its candidate pixel scores (rewards both intensity and the number of
  opportunities); ranks all 243 wards.
- **Equity flag:** ward LST in the top quartile **and** population density in the top quartile
  **and** mean NDVI in the bottom quartile (31 wards).

## API

| Route | Purpose |
|---|---|
| `GET /` | the app |
| `GET /data/<file>` | whitelisted files from `data/` |
| `GET /whatif?ward=ID&dndvi=0.1` | predicted cooling for an NDVI increase |

```bash
curl "http://localhost:5000/whatif?ward=133&dndvi=0.1"
# {"clamped": false, "dndvi": 0.1, "label": "Correlational scenario, not a causal or field-measured result.",
#  "predicted_cooling_c": 0.75, "slope": -7.4757, "slope_fallback": false,
#  "slope_r2": 0.488, "ward_id": "133", "ward_name": "Gandhinagar"}
```

`dndvi` is clamped to [0, 0.5]; non-numeric input returns 400, unknown ward returns 404 (JSON).

## Switching to real Google Earth Engine data

The pipeline accepts any CSV with the same schema; nothing else changes.

```bash
pip install earthengine-api
earthengine authenticate              # one-time browser login
python gee/export.py                  # submits a Drive export task (~100 m grid, MAM 2019-2025)
# download heatsense_bengaluru_grid.csv from Drive to data/grid_features.csv
./run_all.sh                          # detects data_source=gee and does NOT overwrite it
python app/app.py
```

`run_all.sh` checks the first row's `data_source` and skips the synthetic generator when the grid is
real. `pipeline/features.py` joins wards if the export has no `ward_id`.

**GEE inputs:** Landsat 8/9 Collection 2 Level 2 `ST_B10` (scale 0.00341802, offset 149.0, K→°C,
QA_PIXEL cloud/shadow mask), Sentinel-2 SR Harmonized medians for NDVI/NDBI/NDWI (SCL mask),
WorldPop `GP/100m/pop` (latest year), ESA WorldCover v200 for built-up fraction and plantability.

**Documented substitutes (no proper OSM road layer exists in the GEE catalogue):**
`building_density` = WorldCover built-up class fraction of the cell; `road_density` = a **proxy**
(Dynamic World `built` probability minus built-up fraction), not a road network measurement.
`plantable` = open vegetation classes, low built fraction, NDBI < 0.1. These are flagged here and in
`gee/export.py` because they are the weakest part of the real-data path.

## Assumptions (made without asking, as instructed)

1. **100 m synthetic grid.** The bbox at 100 m is ~122k cells; clipping to BBMP wards leaves 70,910,
   so the saved dataset stays inside the 60–80k cap. No 150 m fallback was needed.
2. **Population** in the synthetic grid totals 8.95 M over the 243 wards (BBMP-scale). It is entirely
   generated; no census or WorldPop values are used in the demo.
3. **Spatial CV grouping = `ward_id`** via GroupKFold (not 2 km blocks). Whole wards are held out;
   ward areas are uneven, so fold sizes differ slightly. Random splits are never used.
4. **Fallback threshold** is applied exactly as specified: n < 100 cells, R² < 0.10, or slope ≥ 0 →
   city-wide slope. Under this rule 129/243 wards fall back, mostly small dense-core wards.
5. **Pixel plantable area** is the full 100 m cell (10,000 m²) for every candidate, so that factor is
   constant and does not change today's ranking. It is kept for schema completeness.
6. **Ward priority = sum** (not mean) of candidate pixel scores.
7. **Population exposure** is a 3×3 cell (~300 m) mean of cell population.
8. **SHAP** is computed on the final model fitted to all cells, using ≤40 sampled cells per ward
   (~9.7k rows). It is descriptive, not out-of-sample.
9. **Ward summaries** are deterministic templates in this run (no `ANTHROPIC_API_KEY` was set).
   Set the key (and optionally `HEATSENSE_MODEL`) and rerun `pipeline/summaries.py` for Claude-written
   summaries; results are cached in `data/summary_cache.json`.

## Limitations

- **Synthetic caveat:** the demo LST is generated from NDVI/NDBI/NDWI/built density plus smooth noise.
  The model therefore looks better than it would on real data (R² ≈ 0.95 is inflated by construction),
  and the "hot spots" are the landmarks we encoded, not findings.
- **LST is surface temperature**, not air temperature and not felt heat (no humidity, wind or shade
  modelling). Daytime Landsat overpasses only.
- **Cooling slopes are correlational.** `−slope × ΔNDVI` is a cross-sectional scenario; it is not a
  measured cooling effect, and greening a specific site may behave differently.
- **Ward-level regression is noisy** in small, homogeneous wards (hence the fallback rule); the
  city-wide slope mixes between-ward variation (an ecological-fallacy risk).
- **SHAP attributions** lean on built-up structure because NDVI/NDBI/building density are correlated.
- **Planting spots are 100 m candidate cells**, not sited parcels: no ownership, utilities, soil,
  water or community consultation.
- **Basemap tiles are the only online dependency** (CARTO/OSM). Leaflet is vendored in
  `app/static/vendor/leaflet`, and if tiles fail the map keeps working on a plain background.
- The GEE export script is compile-checked only; its runtime behaviour has not been verified here.

## Project structure

```
heatsense/
├── PLAN.md
├── README.md
├── requirements.txt
├── run_all.sh / run_all.bat
├── app/
│   ├── app.py                 Flask: static UI, /data/<file>, /whatif
│   └── static/
│       ├── index.html, app.js, style.css
│       └── vendor/leaflet/    vendored Leaflet 1.9.4 (+ images)
├── data/
│   ├── raw/bengaluru_wards.geojson   real 243-ward BBMP/KGIS boundaries
│   ├── grid_features.csv, ward_features.csv
│   ├── wards.geojson, meta.json
│   ├── model_metrics.json, shap_global.csv, shap_by_ward.json
│   ├── cooling_by_ward.csv, ward_priority.csv, summaries.json
│   └── top10_hottest.json, top10_planting.json, summary_cache.json
├── gee/
│   └── export.py              Earth Engine export (same CSV schema, data_source="gee")
└── pipeline/
    ├── config.py              study area constants and paths
    ├── wards.py               ward loader (real download -> synthetic Voronoi fallback)
    ├── synthetic.py           synthetic grid generator + ward join
    ├── features.py            schema validation, clipping, ward aggregates
    ├── model.py               XGBoost + ward GroupKFold + SHAP
    ├── cooling.py             per-ward OLS slopes, fallback rule, whatif()
    ├── priority.py            candidates, score, spacing, equity flags
    ├── summaries.py           template / Claude build-time summaries + cache
    └── outputs.py             wards.geojson, top-10 lists, meta.json
```

## Notes

- Summaries are written at **build time**, never per request; the app only reads precomputed files.
- `pipeline/wards.py` still supports downloading the KGIS set, and falls back to a labelled synthetic
  Voronoi tessellation if both the local file and the network are unavailable.
- Model card (R², RMSE, folds, grouping, fallback rule) is in the app's "About the model" dialog.
