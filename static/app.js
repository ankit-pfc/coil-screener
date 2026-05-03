const statusText = document.getElementById("statusText");
const tickersInput = document.getElementById("tickersInput");
const universeSelect = document.getElementById("universeSelect");
const limitInput = document.getElementById("limitInput");
const runButton = document.getElementById("runButton");
const savedRunsSelect = document.getElementById("savedRunsSelect");
const loadSavedButton = document.getElementById("loadSavedButton");
const resultsTableBody = document.querySelector("#resultsTable tbody");
const resultsTitle = document.getElementById("resultsTitle");
const resultCount = document.getElementById("resultCount");
const selectedTicker = document.getElementById("selectedTicker");
const chartSubtitle = document.getElementById("chartSubtitle");
const chartMount = document.getElementById("chartMount");
const featureGrid = document.getElementById("featureGrid");
const expandChartButton = document.getElementById("expandChartButton");
const chartOverlay = document.getElementById("chartOverlay");
const overlayTitle = document.getElementById("overlayTitle");
const overlayChartMount = document.getElementById("overlayChartMount");
const closeOverlayButton = document.getElementById("closeOverlayButton");

let currentResults = [];
let activeTicker = null;
let activeBars = [];

const amrutTickers = new Set([
  "AER",
  "AVT",
  "BDC",
  "DD",
  "EWY",
  "LAZ",
  "PPC",
  "PTCT",
  "STLD",
  "TEX",
  "UTHR"
]);

const featureLabels = {
  score_total: "Total Score",
  score_long_coil: "Long Coil",
  score_tight_resistance: "Tight Resistance",
  score_ascending_compression: "Ascending Compression",
  age_years: "Age Years",
  last_close: "Last Close",
  pos_in_10y_range: "Position In 10Y Range",
  dist_to_10y_high_pct: "Distance To 10Y High",
  range_ratio_24_120: "24M / 120M Range",
  range_ratio_24_60: "24M / 60M Range",
  low_36m_above_10y_low_pct: "36M Low Above 10Y Low",
  slope_high_60m: "60M High Slope",
  slope_low_60m: "60M Low Slope",
  trend_r2_60m: "60M Trend R²",
  peak_age_months: "Peak Age Months",
  old_peak_similarity: "Old Peak Similarity"
};

function setStatus(message) {
  statusText.textContent = message;
}

function formatNumber(value, digits = 3) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }
  return Number(value).toFixed(digits);
}

function splitTickers(value) {
  return value
    .split(/[\s,]+/)
    .map((item) => item.trim().toUpperCase())
    .filter(Boolean);
}

async function fetchJson(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const payload = await response.text();
    throw new Error(payload || `Request failed: ${response.status}`);
  }
  return response.json();
}

function renderFeatureGrid(data) {
  featureGrid.innerHTML = "";

  if (!data) {
    featureGrid.innerHTML = '<p class="muted-note">Feature data will appear after you select a ticker.</p>';
    return;
  }

  Object.entries(featureLabels).forEach(([key, label]) => {
    const card = document.createElement("div");
    card.className = "feature-card";
    card.innerHTML = `<span>${label}</span><strong>${formatNumber(data[key])}</strong>`;
    featureGrid.appendChild(card);
  });
}

function renderEmptyChart(message, mount = chartMount) {
  mount.innerHTML = `<div class="chart-empty">${message}</div>`;
}

function svgEl(name, attrs = {}) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", name);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function renderChart(ticker, bars, mount = chartMount, options = {}) {
  if (!bars || !bars.length) {
    renderEmptyChart("No monthly bars available.", mount);
    return;
  }

  const width = options.width || 960;
  const height = options.height || 420;
  const labelSize = options.labelSize || 11;
  const lineWidth = options.lineWidth || 1.15;
  const candleScale = options.candleScale || 0.58;
  const margin = options.margin || { top: 18, right: 56, bottom: 34, left: 16 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const highs = bars.map((bar) => bar.high);
  const lows = bars.map((bar) => bar.low);
  const highest = Math.max(...highs);
  const lowest = Math.min(...lows);
  const padding = (highest - lowest) * 0.05 || 1;
  const yMax = highest + padding;
  const yMin = lowest - padding;
  const candleSlot = plotWidth / bars.length;
  const candleWidth = Math.max(options.minCandleWidth || 2, candleSlot * candleScale);

  const yScale = (value) => margin.top + ((yMax - value) / (yMax - yMin)) * plotHeight;
  const xScale = (index) => margin.left + index * candleSlot + candleSlot / 2;

  const svg = svgEl("svg", {
    viewBox: `0 0 ${width} ${height}`,
    class: "chart-svg",
    role: "img",
    "aria-label": `${ticker} monthly candlestick chart`
  });

  for (let i = 0; i <= 4; i += 1) {
    const y = margin.top + (plotHeight / 4) * i;
    svg.appendChild(svgEl("line", {
      x1: margin.left,
      y1: y,
      x2: width - margin.right,
      y2: y,
      stroke: "rgba(46, 38, 28, 0.08)",
      "stroke-width": 1
    }));

    const value = yMax - ((yMax - yMin) / 4) * i;
    const label = svgEl("text", {
      x: width - margin.right + 8,
      y: y + 4,
      fill: "rgba(46, 38, 28, 0.68)",
      "font-size": labelSize
    });
    label.textContent = formatNumber(value, 2);
    svg.appendChild(label);
  }

  bars.forEach((bar, index) => {
    const x = xScale(index);
    const openY = yScale(bar.open);
    const closeY = yScale(bar.close);
    const highY = yScale(bar.high);
    const lowY = yScale(bar.low);
    const rising = bar.close >= bar.open;
    const color = rising ? "#1d6b57" : "#b4473e";
    const bodyTop = Math.min(openY, closeY);
    const bodyHeight = Math.max(1.5, Math.abs(closeY - openY));

    svg.appendChild(svgEl("line", {
      x1: x,
      y1: highY,
      x2: x,
      y2: lowY,
      stroke: color,
      "stroke-width": lineWidth
    }));

    svg.appendChild(svgEl("rect", {
      x: x - candleWidth / 2,
      y: bodyTop,
      width: candleWidth,
      height: bodyHeight,
      fill: color,
      rx: 1
    }));
  });

  const labelIndexes = [0, Math.floor(bars.length * 0.25), Math.floor(bars.length * 0.5), Math.floor(bars.length * 0.75), bars.length - 1];
  [...new Set(labelIndexes)].forEach((index) => {
    const bar = bars[index];
    const label = svgEl("text", {
      x: xScale(index),
      y: height - 10,
      fill: "rgba(46, 38, 28, 0.65)",
      "font-size": labelSize,
      "text-anchor": "middle"
    });
    label.textContent = bar.date.slice(0, 7);
    svg.appendChild(label);
  });

  chartMount.innerHTML = "";
  mount.innerHTML = "";
  mount.appendChild(svg);
}

function renderResults(results, title) {
  currentResults = results || [];
  resultsTitle.textContent = title;
  resultCount.textContent = String(currentResults.length);
  resultsTableBody.innerHTML = "";
  activeTicker = null;
  activeBars = [];
  expandChartButton.textContent = "⤢";
  selectedTicker.textContent = "None";
  chartSubtitle.textContent = "Select a ticker to load candles.";
  renderEmptyChart("Select a result row to inspect the monthly chart.");
  renderFeatureGrid(null);

  currentResults.forEach((row, index) => {
    const tr = document.createElement("tr");
    tr.dataset.ticker = row.ticker;
    const isAmrutPick = amrutTickers.has(row.ticker);
    tr.innerHTML = `
      <td>${index + 1}</td>
      <td><strong>${row.ticker}</strong>${isAmrutPick ? '<span class="tag-chip">test</span>' : ""}</td>
      <td>${formatNumber(row.score_total)}</td>
      <td>${formatNumber(row.score_long_coil)}</td>
      <td>${formatNumber(row.score_tight_resistance)}</td>
      <td>${formatNumber(row.score_ascending_compression)}</td>
      <td>${formatNumber(row.age_years, 1)}</td>
    `;
    tr.addEventListener("click", () => selectTicker(row.ticker, row));
    resultsTableBody.appendChild(tr);
  });
}

async function selectTicker(ticker, summaryRow) {
  activeTicker = ticker;
  selectedTicker.textContent = ticker;
  chartSubtitle.textContent = `Loading ${ticker} monthly history...`;
  document.querySelectorAll("#resultsTable tbody tr").forEach((row) => {
    row.classList.toggle("active", row.dataset.ticker === ticker);
  });

  try {
    const payload = await fetchJson(`/api/history/${ticker}`);
    activeBars = payload.bars || [];
    chartSubtitle.textContent = `${ticker} monthly candles (${payload.bars.length} bars shown)`;
    renderChart(ticker, payload.bars);
    renderFeatureGrid(payload.features || summaryRow);
    setStatus(`Loaded ${ticker} chart.`);
  } catch (error) {
    chartSubtitle.textContent = `Failed to load ${ticker}`;
    renderEmptyChart("Could not load monthly history for this ticker.");
    renderFeatureGrid(summaryRow);
    setStatus(`Chart load failed for ${ticker}.`);
  }
}

function openChartOverlay() {
  if (!activeTicker || !activeBars.length) {
    setStatus("Select a ticker before expanding the chart.");
    return;
  }

  overlayTitle.textContent = `${activeTicker} Monthly Chart`;
  chartOverlay.classList.add("open");
  chartOverlay.setAttribute("aria-hidden", "false");
  document.body.classList.add("modal-open");
  renderChart(activeTicker, activeBars.slice(-120), overlayChartMount, {
    width: 1600,
    height: 900,
    labelSize: 28,
    lineWidth: 3,
    candleScale: 1.08,
    minCandleWidth: 8,
    margin: { top: 36, right: 128, bottom: 82, left: 32 }
  });
}

function closeChartOverlay() {
  chartOverlay.classList.remove("open");
  chartOverlay.setAttribute("aria-hidden", "true");
  document.body.classList.remove("modal-open");
  overlayChartMount.innerHTML = "";
}

async function loadSavedRuns() {
  const payload = await fetchJson("/api/saved-runs");
  savedRunsSelect.innerHTML = "";
  payload.runs.forEach((run) => {
    const option = document.createElement("option");
    option.value = run.name;
    option.textContent = run.name;
    savedRunsSelect.appendChild(option);
  });
}

async function loadSavedRun(name) {
  if (!name) {
    return;
  }

  setStatus(`Loading ${name}...`);
  const payload = await fetchJson(`/api/saved-runs/${encodeURIComponent(name)}`);
  renderResults(payload.results, `Saved Run · ${name}`);
  setStatus(`Loaded ${payload.count} saved rows from ${name}.`);
}

async function loadDefaultTickers() {
  const payload = await fetchJson("/api/default-tickers");
  tickersInput.value = payload.tickers.join("\n");
}

async function runScreen() {
  setStatus("Running live screen...");
  runButton.disabled = true;

  try {
    const tickers = splitTickers(tickersInput.value);
    const body = {
      tickers,
      universe: universeSelect.value || null,
      limit: limitInput.value ? Number(limitInput.value) : null
    };

    const payload = await fetchJson("/api/screen", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });

    renderResults(payload.results, "Live Screen Results");
    setStatus(`Live screen finished with ${payload.count} ranked rows.`);
  } catch (error) {
    console.error(error);
    setStatus("Live screen failed.");
  } finally {
    runButton.disabled = false;
  }
}

runButton.addEventListener("click", runScreen);
loadSavedButton.addEventListener("click", () => loadSavedRun(savedRunsSelect.value));
expandChartButton.addEventListener("click", openChartOverlay);
closeOverlayButton.addEventListener("click", closeChartOverlay);
chartOverlay.addEventListener("click", (event) => {
  if (event.target === chartOverlay) {
    closeChartOverlay();
  }
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && chartOverlay.classList.contains("open")) {
    closeChartOverlay();
  }
});

async function boot() {
  try {
    await Promise.all([loadDefaultTickers(), loadSavedRuns()]);

    const benchmarkName = Array.from(savedRunsSelect.options).find((option) =>
      option.value === "sp50_plus_amrut_results.csv"
    )?.value || savedRunsSelect.value;

    if (benchmarkName) {
      savedRunsSelect.value = benchmarkName;
      await loadSavedRun(benchmarkName);
    } else {
      renderEmptyChart("Run a screen or load a saved CSV.");
      renderFeatureGrid(null);
      setStatus("No saved runs found.");
    }
  } catch (error) {
    console.error(error);
    renderEmptyChart("The UI could not load its initial data.");
    renderFeatureGrid(null);
    setStatus("Initial load failed.");
  }
}

boot();
