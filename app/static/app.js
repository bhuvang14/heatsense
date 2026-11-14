/* HeatSense front end: Leaflet + precomputed GeoJSON from the Flask backend. */
"use strict";

const CENTER = [12.97, 77.59];
const PALETTE = ["#ffffb2", "#fed976", "#feb24c", "#fd8d3c", "#e31a1c", "#800026"];
const FEATURE_LABELS = {
  ndvi: "NDVI (vegetation)",
  ndbi: "NDBI (built-up)",
  ndwi: "NDWI (water)",
  building_density: "Building density",
  road_density: "Road density",
  population: "Population",
};

const state = {
  meta: null,
  metrics: null,
  wards: null,
  spots: [],
  hottest: [],
  selected: null,
  layerById: new Map(),
  selectedLayer: null,
};

const els = {
  map: document.getElementById("map"),
  banner: document.getElementById("synthetic-banner"),
  aboutBtn: document.getElementById("about-btn"),
  aboutDialog: document.getElementById("about-dialog"),
  aboutBody: document.getElementById("about-body"),
  slider: document.getElementById("ndvi-slider"),
  sliderValue: document.getElementById("ndvi-value"),
  whatifWard: document.getElementById("whatif-ward"),
  whatifCooling: document.getElementById("whatif-cooling"),
  whatifNote: document.getElementById("whatif-note"),
  spotsToggle: document.getElementById("spots-toggle"),
  equityToggle: document.getElementById("equity-toggle"),
  listHottest: document.getElementById("list-hottest"),
  listPlanting: document.getElementById("list-planting"),
  legendMin: document.getElementById("legend-min"),
  legendMax: document.getElementById("legend-max"),
  mapFallback: document.getElementById("map-fallback"),
};

let map;
let wardLayer;
let spotsLayer;
let equityLayer;
let lstDomain = [30, 48];

/* ---------- helpers ---------- */
const esc = (s) =>
  String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]
  );

function quantile(sorted, q) {
  const pos = (sorted.length - 1) * q;
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
}

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function heatColor(value) {
  const [min, max] = lstDomain;
  const t = Math.min(Math.max((value - min) / (max - min || 1), 0), 1);
  const scaled = t * (PALETTE.length - 1);
  const i = Math.min(Math.floor(scaled), PALETTE.length - 2);
  const f = scaled - i;
  const a = hexToRgb(PALETTE[i]);
  const b = hexToRgb(PALETTE[i + 1]);
  const rgb = a.map((v, k) => Math.round(v + (b[k] - v) * f));
  return `rgb(${rgb.join(",")})`;
}

function wardStyle(props, selected = false) {
  return {
    fillColor: heatColor(props.lst_mean),
    fillOpacity: selected ? 0.9 : 0.76,
    color: selected ? "#16202a" : "rgba(255,255,255,0.85)",
    weight: selected ? 3 : 1,
  };
}

/* ---------- popup ---------- */
function popupHtml(p) {
  const drivers = (p.top_drivers || [])
    .map(
      (d) =>
        `<li>${esc(FEATURE_LABELS[d.feature] || d.feature)}
         <span class="dir ${d.direction}">${esc(d.direction)}</span>
         <span class="dir">${Math.round(d.share * 100)}%</span></li>`
    )
    .join("");
  const coolingPer01 = (-p.slope * 0.1).toFixed(1);
  const fallback = p.slope_fallback
    ? `<p class="fallback">City-wide fallback slope (ward sample too small or too noisy).</p>`
    : "";
  const equity = p.equity_flag
    ? `<span class="equity">Equity priority: hot, dense, low green cover</span>`
    : "";
  return `
    <div class="popup">
      <h3>${esc(p.ward_name)}</h3>
      <div class="row"><span>Mean surface LST</span><b>${p.lst_mean.toFixed(1)} °C</b></div>
      <div class="row"><span>Mean NDVI</span><b>${p.ndvi_mean.toFixed(2)}</b></div>
      <div class="row"><span>Population</span><b>${Number(p.population).toLocaleString()}</b></div>
      <div class="row"><span>Plantable area</span><b>${Math.round(p.plantable_area_m2 / 1000).toLocaleString()} k m²</b></div>
      <div class="divider"></div>
      <div class="row"><span>Top heat drivers</span><b></b></div>
      <ul class="drivers">${drivers || "<li>no SHAP data</li>"}</ul>
      <div class="divider"></div>
      <div class="row"><span>Cooling efficiency</span><b>${coolingPer01} °C per +0.1 NDVI</b></div>
      ${fallback}
      <p class="summary">${esc(p.summary || "")}</p>
      ${equity}
    </div>`;
}

/* ---------- selection + what-if ---------- */
function selectWard(props, layer) {
  state.selected = props;
  if (state.selectedLayer && state.selectedLayer !== layer) {
    state.selectedLayer.setStyle(wardStyle(state.selectedLayer.feature.properties));
  }
  state.selectedLayer = layer || null;
  if (layer) layer.setStyle(wardStyle(props, true));

  els.slider.disabled = false;
  els.whatifWard.textContent = props.ward_name;
  els.whatifCooling.textContent = "Predicted cooling: …";
  els.whatifNote.textContent = "";
  runWhatIf();
}

async function runWhatIf() {
  if (!state.selected) return;
  const dndvi = els.slider.value;
  els.sliderValue.textContent = Number(dndvi).toFixed(2);
  const url = `/whatif?ward=${encodeURIComponent(state.selected.ward_id)}&dndvi=${dndvi}`;
  try {
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    els.whatifCooling.textContent = `Predicted cooling: ${data.predicted_cooling_c.toFixed(1)} °C`;
    els.whatifNote.textContent = data.slope_fallback
      ? `City-wide slope ${data.slope.toFixed(1)} °C per +1.0 NDVI (R² ${data.slope_r2.toFixed(2)}).`
      : `Local slope ${data.slope.toFixed(1)} °C per +1.0 NDVI (R² ${data.slope_r2.toFixed(2)}, n=${state.selected.slope_n}).`;
  } catch (err) {
    els.whatifCooling.textContent = "Predicted cooling: unavailable";
    els.whatifNote.textContent = String(err.message || err);
  }
}

/* ---------- sidebar lists ---------- */
function renderLists() {
  els.listHottest.innerHTML = state.hottest
    .map(
      (w, i) => `<li><button type="button" data-ward="${esc(w.ward_id)}">
        ${i + 1}. ${esc(w.ward_name)}
        <span class="meta">${w.lst_mean.toFixed(1)} °C · NDVI ${w.ndvi_mean.toFixed(2)}${
        w.equity_flag ? " · equity" : ""
      }</span></button></li>`
    )
    .join("");

  els.listPlanting.innerHTML = state.spots
    .map(
      (s, i) => `<li><button type="button" data-spot="${i}">
        ${i + 1}. ${esc(s.ward_name)}
        <span class="meta">${s.lst_c.toFixed(1)} °C · +0.1 NDVI ≈ ${s.expected_cooling_c_per_0_1_ndvi.toFixed(
        1
      )} °C</span></button></li>`
    )
    .join("");

  els.listHottest.querySelectorAll("button[data-ward]").forEach((btn) =>
    btn.addEventListener("click", () => focusWard(btn.dataset.ward))
  );
  els.listPlanting.querySelectorAll("button[data-spot]").forEach((btn) =>
    btn.addEventListener("click", () => focusSpot(Number(btn.dataset.spot)))
  );
}

function focusWard(wardId) {
  const layer = state.layerById.get(String(wardId));
  if (!layer) return;
  map.fitBounds(layer.getBounds(), { padding: [30, 30] });
  layer.openPopup();
  selectWard(layer.feature.properties, layer);
}

function focusSpot(index) {
  const spot = state.spots[index];
  if (!spot) return;
  if (!els.spotsToggle.checked) {
    els.spotsToggle.checked = true;
    spotsLayer.addTo(map);
  }
  map.setView([spot.lat, spot.lon], 15);
  const ward = state.layerById.get(String(spot.ward_id));
  if (ward) selectWard(ward.feature.properties, ward);
  const marker = spotMarkers[index];
  if (marker) marker.openPopup();
}

/* ---------- map init ---------- */
function initMap() {
  map = L.map(els.map, { zoomControl: true }).setView(CENTER, 11);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors &copy; CARTO',
  }).addTo(map);

  map.on("tileerror", () => els.mapFallback.classList.remove("hidden"));

  const values = state.wards.features.map((f) => f.properties.lst_mean).sort((a, b) => a - b);
  lstDomain = [quantile(values, 0.02), quantile(values, 0.98)];
  els.legendMin.textContent = `${lstDomain[0].toFixed(0)}`;
  els.legendMax.textContent = `${lstDomain[1].toFixed(0)}`;

  wardLayer = L.geoJSON(state.wards, {
    style: (f) => wardStyle(f.properties),
    onEachFeature: (feature, layer) => {
      state.layerById.set(String(feature.properties.ward_id), layer);
      layer.bindPopup(popupHtml(feature.properties), { maxWidth: 320 });
      layer.on("click", () => selectWard(feature.properties, layer));
    },
  }).addTo(map);

  // Equity outline layer (off by default)
  const equityFeatures = state.wards.features.filter((f) => f.properties.equity_flag);
  equityLayer = L.geoJSON(
    { type: "FeatureCollection", features: equityFeatures },
    { style: { color: "#7a0f1e", weight: 2.5, dashArray: "5 4", fill: false } }
  );

  // Planting spot markers
  spotMarkers.length = 0;
  spotsLayer = L.layerGroup(
    state.spots.map((s, i) => {
      const marker = L.circleMarker([s.lat, s.lon], {
        radius: 7,
        color: "#12351d",
        weight: 1.5,
        fillColor: "#2f6b3a",
        fillOpacity: 0.95,
      }).bindPopup(
        `<div class="popup"><h3>#${i + 1} planting spot</h3>
         <div class="row"><span>Ward</span><b>${esc(s.ward_name)}</b></div>
         <div class="row"><span>Surface LST</span><b>${s.lst_c.toFixed(1)} °C</b></div>
         <div class="row"><span>NDVI</span><b>${s.ndvi.toFixed(2)}</b></div>
         <div class="row"><span>Population (≈300 m)</span><b>${Math.round(
           s.population_300m
         ).toLocaleString()}</b></div>
         <div class="row"><span>Expected cooling / +0.1 NDVI</span><b>${s.expected_cooling_c_per_0_1_ndvi.toFixed(
           1
         )} °C</b></div>
         <p class="fallback">${
           s.slope_fallback ? "Uses city-wide cooling slope." : "Uses local ward cooling slope."
         }</p></div>`,
        { maxWidth: 300 }
      );
      // Clicking a spot keeps its own popup open but also targets that ward.
      marker.on("click", () => {
        const ward = state.layerById.get(String(s.ward_id));
        if (ward) selectWard(ward.feature.properties, ward);
      });
      spotMarkers.push(marker);
      return marker;
    })
  ).addTo(map);
}

const spotMarkers = [];

/* ---------- wiring ---------- */
function wireUi() {
  els.spotsToggle.addEventListener("change", () => {
    if (els.spotsToggle.checked) spotsLayer.addTo(map);
    else map.removeLayer(spotsLayer);
  });

  els.equityToggle.addEventListener("change", () => {
    if (els.equityToggle.checked) equityLayer.addTo(map);
    else map.removeLayer(equityLayer);
  });

  els.slider.addEventListener("input", runWhatIf);

  document.querySelectorAll(".tab").forEach((tab) =>
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
      const showHottest = tab.dataset.tab === "hottest";
      els.listHottest.classList.toggle("hidden", !showHottest);
      els.listPlanting.classList.toggle("hidden", showHottest);
    })
  );

  els.aboutBtn.addEventListener("click", () => {
    els.aboutBody.innerHTML = aboutHtml();
    els.aboutDialog.showModal();
  });
}

function aboutHtml() {
  const m = state.metrics;
  const meta = state.meta;
  const folds = m.folds
    .map((f) => `<tr><th>Fold ${f.fold}</th><td>R² ${f.r2.toFixed(3)} · RMSE ${f.rmse_c.toFixed(2)} °C · ${f.n_wards_val} wards held out</td></tr>`)
    .join("");
  const synthetic =
    meta.data_source === "synthetic"
      ? `<div class="warn">These results come from a SYNTHETIC generator, not measurements. R² is inflated because the generator builds LST from the same predictors the model uses. Do not cite these numbers as findings about Bengaluru.</div>`
      : "";
  return `
    ${synthetic}
    <table>
      <tr><th>Data source</th><td>${esc(meta.data_source)}${meta.data_source === "synthetic" ? " (demo)" : " (Google Earth Engine export)"}</td></tr>
      <tr><th>Ward set</th><td>${esc(meta.ward_set)}</td></tr>
      <tr><th>Wards / grid</th><td>${meta.n_wards} wards · ${meta.grid_size_m} m cells · ${m.n_samples.toLocaleString()} cells</td></tr>
      <tr><th>Spatial validation</th><td>${esc(m.grouping)}</td></tr>
      <tr><th>Overall out-of-fold</th><td>R² ${m.overall_r2.toFixed(3)} · RMSE ${m.overall_rmse_c.toFixed(2)} °C</td></tr>
      ${folds}
      <tr><th>City-wide NDVI slope</th><td>${Number(meta.city_ndvi_slope_c_per_unit).toFixed(1)} °C per +1.0 NDVI</td></tr>
      <tr><th>Cooling slopes</th><td>${m.cooling_fallback_wards} of ${m.n_wards} wards use the city-wide fallback (${esc(m.cooling_fallback_rule)})</td></tr>
      <tr><th>Ward summaries</th><td>${esc(meta.summary_source === "claude" ? "Claude API at build time" : "deterministic template")}</td></tr>
    </table>
    <p class="note">The model predicts Landsat surface temperature (LST) from NDVI, NDBI, NDWI,
    building and road density and population. Validation holds out whole wards, so scores
    reflect performance on places the model has never seen.</p>
    <p class="note">Limits: LST is the temperature of the surface in the satellite pixel, not the
    air people feel; cooling slopes are cross-sectional associations, not measured outcomes of
    planting; planting spots are 100 m candidate cells, not sited, surveyed parcels.</p>`;
}

async function boot() {
  try {
    const [meta, metrics, wards, hottest, planting] = await Promise.all([
      fetch("/data/meta.json").then((r) => r.json()),
      fetch("/data/model_metrics.json").then((r) => r.json()),
      fetch("/data/wards.geojson").then((r) => r.json()),
      fetch("/data/top10_hottest.json").then((r) => r.json()),
      fetch("/data/top10_planting.json").then((r) => r.json()),
    ]);
    state.meta = meta;
    state.metrics = metrics;
    state.wards = wards;
    state.hottest = hottest.wards;
    state.spots = planting.spots;

    if (meta.data_source === "synthetic") els.banner.classList.remove("hidden");
    if (meta.footer) document.querySelector(".footer").textContent = meta.footer;

    initMap();
    renderLists();
    wireUi();
  } catch (err) {
    els.map.innerHTML = `<p style="padding:24px;font-size:14px">Could not load data: ${esc(
      err.message || err
    )}. Run ./run_all.sh and reload.</p>`;
  }
}

boot();
