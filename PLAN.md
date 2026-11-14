# HeatSense — build plan (hackathon, one day)

1. Synthetic ~100 m grid over Bengaluru bbox with realistic hotspots, green patches, lakes; ward join; `data/grid_features.csv`.
2. `gee/export.py` real-data upgrade path, same CSV schema (compile-checked only, no auth).
3. `features.py` schema validation + ward aggregates.
4. `model.py` XGBoost LST model, GroupKFold by ward (spatial CV), metrics + SHAP global/per-ward.
5. `cooling.py` per-ward OLS LST~NDVI slope with fallback rules + `whatif()`.
6. `priority.py` plantable/hot/low-green candidates, normalised score, top-10 spots with 1 km spacing, equity flags.
7. `summaries.py` template summaries with optional Claude build-time call + cache.
8. `outputs`: wards.geojson, top10_hottest.json, top10_planting.json, model_metrics.json.
9. Flask app + Leaflet UI (choropleth, popups, top-10 lists, what-if slider, markers, equity layer, model panel, synthetic banner).
10. README (claim, assumptions, run commands, GEE switch, limits), requirements.txt, run_all.sh/.bat.
11. Full clean run, start app, curl /whatif, then demo script.

Assumption: synthetic data first so the whole demo works offline; real GEE is opt-in via the same schema.
