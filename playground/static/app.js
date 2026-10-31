"use strict";

async function apiCall(path, body) {
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(`${path} failed: ${response.status}`);
  }
  return response.json();
}

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

function computeLayout(root) {
  const positions = new Map();
  let nextX = 0;

  function visit(node, depth) {
    if (!node) return;
    visit(node.left, depth + 1);
    positions.set(node.id, { x: nextX++, y: depth, node });
    visit(node.right, depth + 1);
  }

  visit(root, 0);
  return positions;
}

// Renders a tree into `svg`. `newIds` marks nodes drawn as new/copied
// (amber) rather than shared (pine outline); `inRange(value)` marks
// nodes that additionally get a range_query highlight ring.
function renderTreeInto(svg, treeJson, { newIds = new Set(), inRange = null } = {}) {
  svg.innerHTML = "";

  if (!treeJson) {
    svg.setAttribute("viewBox", "0 0 100 60");
    const text = svgEl("text", {
      x: "50",
      y: "32",
      "text-anchor": "middle",
      "font-family": "ui-monospace, monospace",
      "font-size": "10",
      fill: "var(--ink-soft)",
    });
    text.textContent = "Empty()";
    svg.appendChild(text);
    return;
  }

  const positions = computeLayout(treeJson);
  const spacingX = 56;
  const spacingY = 64;
  const padding = 28;
  const xs = [...positions.values()].map((p) => p.x);
  const ys = [...positions.values()].map((p) => p.y);
  const width = Math.max(...xs) * spacingX + padding * 2;
  const height = Math.max(...ys) * spacingY + padding * 2;
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const pixel = (p) => ({ cx: p.x * spacingX + padding, cy: p.y * spacingY + padding });

  function drawEdges(node) {
    if (!node) return;
    const here = pixel(positions.get(node.id));
    for (const child of [node.left, node.right]) {
      if (!child) continue;
      const there = pixel(positions.get(child.id));
      svg.appendChild(
        svgEl("line", {
          x1: here.cx,
          y1: here.cy,
          x2: there.cx,
          y2: there.cy,
          stroke: "var(--pine-soft)",
          "stroke-width": "1.5",
        }),
      );
      drawEdges(child);
    }
  }
  drawEdges(treeJson);

  for (const { x, y, node } of positions.values()) {
    const { cx, cy } = pixel({ x, y });
    const isNew = newIds.has(node.id);
    const group = svgEl("g", {});

    if (inRange && inRange(node.value)) {
      group.appendChild(
        svgEl("circle", {
          cx,
          cy,
          r: "20",
          fill: "none",
          stroke: "var(--sky)",
          "stroke-width": "2",
        }),
      );
    }

    group.appendChild(
      svgEl("circle", {
        cx,
        cy,
        r: "16",
        fill: isNew ? "var(--amber)" : "var(--paper)",
        stroke: isNew ? "var(--amber)" : "var(--pine-soft)",
        "stroke-width": isNew ? "2" : "1.5",
      }),
    );
    const label = svgEl("text", {
      x: cx,
      y: cy + 4,
      "text-anchor": "middle",
      "font-family": "ui-monospace, monospace",
      "font-size": "12",
      fill: isNew ? "var(--paper)" : "var(--ink)",
    });
    label.textContent = node.value;
    group.appendChild(label);
    svg.appendChild(group);
  }
}

function createTreePanel(treeId) {
  const root = document.querySelector(`.tree-panel[data-tree-id="${treeId}"]`);
  const els = {
    input: root.querySelector(".value-input"),
    insertBtn: root.querySelector(".insert-btn"),
    deleteBtn: root.querySelector(".delete-btn"),
    randomBtn: root.querySelector(".random-btn"),
    resetBtn: root.querySelector(".reset-btn"),
    timeline: root.querySelector(".timeline"),
    svg: root.querySelector(".tree-svg"),
    statSize: root.querySelector(".stat-size"),
    statDepth: root.querySelector(".stat-depth"),
    statMin: root.querySelector(".stat-min"),
    statMax: root.querySelector(".stat-max"),
    statVersion: root.querySelector(".stat-version"),
    values: root.querySelector(".values"),
    kthInput: root.querySelector(".kth-input"),
    kthResult: root.querySelector(".kth-result"),
    rangeLo: root.querySelector(".range-lo"),
    rangeHi: root.querySelector(".range-hi"),
    rangeLoValue: root.querySelector(".range-lo-value"),
    rangeHiValue: root.querySelector(".range-hi-value"),
  };

  const state = { versions: [], selected: 0 };

  function applyState(history) {
    state.versions = history.versions;
    state.selected = state.versions.length - 1;
    render();
  }

  function render() {
    renderTimeline();
    renderStats();
    syncRangeBounds();
    renderTreeAndHighlight();
    renderKth();
  }

  function renderTimeline() {
    els.timeline.innerHTML = "";
    for (const version of state.versions) {
      const chip = document.createElement("button");
      const classes = ["chip"];
      if (version.index === state.selected) classes.push("current");
      if (version.rotated) classes.push("rotated");
      chip.className = classes.join(" ");
      chip.textContent = `v${version.index} ${version.label}`;
      chip.addEventListener("click", () => {
        state.selected = version.index;
        render();
      });
      els.timeline.appendChild(chip);
    }
    els.timeline.lastElementChild?.scrollIntoView({ block: "nearest", inline: "end" });
  }

  function renderStats() {
    const current = state.versions[state.selected];
    els.statSize.textContent = current.size;
    els.statDepth.textContent = current.depth;
    els.statMin.textContent = current.minimum ?? "–";
    els.statMax.textContent = current.maximum ?? "–";
    els.statVersion.textContent = `v${current.index}`;
    els.values.textContent = `in_order: [${current.values.join(", ")}]`;
  }

  function syncRangeBounds() {
    const current = state.versions[state.selected];
    const lo = current.minimum ?? 0;
    const hi = current.maximum ?? 0;
    for (const input of [els.rangeLo, els.rangeHi]) {
      input.min = String(lo);
      input.max = String(hi);
    }
    // Keep existing handle positions if still in bounds; otherwise reset
    // to the full range, so switching to an empty/smaller tree doesn't
    // leave the sliders stuck out of range.
    if (Number(els.rangeLo.value) < lo || Number(els.rangeLo.value) > hi) {
      els.rangeLo.value = String(lo);
    }
    if (Number(els.rangeHi.value) < lo || Number(els.rangeHi.value) > hi) {
      els.rangeHi.value = String(hi);
    }
    renderRangeLabel();
  }

  function renderRangeLabel() {
    els.rangeLoValue.textContent = els.rangeLo.value;
    els.rangeHiValue.textContent = els.rangeHi.value;
  }

  function renderTreeAndHighlight() {
    const current = state.versions[state.selected];
    const previous = state.selected > 0 ? state.versions[state.selected - 1] : null;
    const previousIds = new Set(previous ? previous.node_ids : []);
    const newIds = new Set(current.node_ids.filter((id) => !previousIds.has(id)));

    const lo = Number(els.rangeLo.value);
    const hi = Number(els.rangeHi.value);
    const hasRange = current.values.length > 0;
    const inRange = hasRange ? (value) => value >= lo && value <= hi : null;

    renderTreeInto(els.svg, current.tree, { newIds, inRange });
  }

  function renderKth() {
    const current = state.versions[state.selected];
    const k = Number.parseInt(els.kthInput.value, 10);
    els.kthResult.textContent =
      Number.isInteger(k) && k >= 0 && k < current.values.length ? current.values[k] : "out of range";
  }

  function readValue() {
    const value = Number.parseInt(els.input.value, 10);
    return Number.isNaN(value) ? null : value;
  }

  els.insertBtn.addEventListener("click", async () => {
    const value = readValue();
    if (value === null) return;
    applyState(await apiCall(`/api/${treeId}/insert`, { value }));
    els.input.value = "";
  });

  els.deleteBtn.addEventListener("click", async () => {
    const value = readValue();
    if (value === null) return;
    applyState(await apiCall(`/api/${treeId}/delete`, { value }));
    els.input.value = "";
  });

  els.randomBtn.addEventListener("click", async () => {
    const value = Math.floor(Math.random() * 100);
    applyState(await apiCall(`/api/${treeId}/insert`, { value }));
  });

  els.resetBtn.addEventListener("click", async () => {
    applyState(await apiCall(`/api/${treeId}/reset`));
  });

  els.kthInput.addEventListener("input", renderKth);

  for (const input of [els.rangeLo, els.rangeHi]) {
    input.addEventListener("input", () => {
      renderRangeLabel();
      renderTreeAndHighlight();
    });
  }

  apiCall(`/api/state/${treeId}`).then(applyState);
}

createTreePanel("a");
createTreePanel("b");

const combineResult = document.getElementById("combine-result");

async function combine(op) {
  const result = await apiCall("/api/combine", { op });
  combineResult.innerHTML = "";

  const label = document.createElement("p");
  label.className = "values";
  label.textContent = `${result.label} = {${result.values.join(", ")}}`;

  const treeWrap = document.createElement("div");
  treeWrap.className = "tree-wrap";
  const svg = svgEl("svg", { viewBox: "0 0 100 60", class: "tree-svg" });
  treeWrap.appendChild(svg);

  combineResult.appendChild(label);
  combineResult.appendChild(treeWrap);
  renderTreeInto(svg, result.tree, {});
}

document.getElementById("union-btn").addEventListener("click", () => combine("union"));
document.getElementById("intersection-btn").addEventListener("click", () => combine("intersection"));
document.getElementById("difference-btn").addEventListener("click", () => combine("difference"));
