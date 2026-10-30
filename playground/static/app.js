"use strict";

const state = {
  versions: [],
  selected: 0,
};

const els = {
  input: document.getElementById("value-input"),
  insertBtn: document.getElementById("insert-btn"),
  deleteBtn: document.getElementById("delete-btn"),
  randomBtn: document.getElementById("random-btn"),
  resetBtn: document.getElementById("reset-btn"),
  timeline: document.getElementById("timeline"),
  svg: document.getElementById("tree-svg"),
  statSize: document.getElementById("stat-size"),
  statDepth: document.getElementById("stat-depth"),
  statVersion: document.getElementById("stat-version"),
  values: document.getElementById("values"),
};

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

function applyState(history) {
  state.versions = history.versions;
  state.selected = state.versions.length - 1;
  render();
}

function render() {
  renderTimeline();
  renderTree();
  renderStats();
}

function renderTimeline() {
  els.timeline.innerHTML = "";
  for (const version of state.versions) {
    const chip = document.createElement("button");
    chip.className = "chip" + (version.index === state.selected ? " current" : "");
    chip.textContent = `v${version.index} ${version.label}`;
    chip.addEventListener("click", () => {
      state.selected = version.index;
      render();
    });
    els.timeline.appendChild(chip);
  }
  els.timeline.lastElementChild?.scrollIntoView({ block: "nearest", inline: "end" });
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

function svgEl(tag, attrs) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs)) {
    el.setAttribute(key, value);
  }
  return el;
}

function renderTree() {
  const current = state.versions[state.selected];
  const previous = state.selected > 0 ? state.versions[state.selected - 1] : null;
  const previousIds = new Set(previous ? previous.node_ids : []);

  els.svg.innerHTML = "";

  if (!current.tree) {
    els.svg.setAttribute("viewBox", "0 0 100 60");
    const text = svgEl("text", {
      x: "50",
      y: "32",
      "text-anchor": "middle",
      "font-family": "ui-monospace, monospace",
      "font-size": "10",
      fill: "var(--ink-soft)",
    });
    text.textContent = "Empty()";
    els.svg.appendChild(text);
    return;
  }

  const positions = computeLayout(current.tree);
  const spacingX = 56;
  const spacingY = 64;
  const padding = 28;
  const xs = [...positions.values()].map((p) => p.x);
  const ys = [...positions.values()].map((p) => p.y);
  const width = Math.max(...xs) * spacingX + padding * 2;
  const height = Math.max(...ys) * spacingY + padding * 2;
  els.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  const pixel = (p) => ({ cx: p.x * spacingX + padding, cy: p.y * spacingY + padding });

  function drawEdges(node) {
    if (!node) return;
    const here = pixel(positions.get(node.id));
    for (const child of [node.left, node.right]) {
      if (!child) continue;
      const there = pixel(positions.get(child.id));
      els.svg.appendChild(
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
  drawEdges(current.tree);

  for (const { x, y, node } of positions.values()) {
    const { cx, cy } = pixel({ x, y });
    const isNew = !previousIds.has(node.id);
    const group = svgEl("g", {});
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
    els.svg.appendChild(group);
  }
}

function renderStats() {
  const current = state.versions[state.selected];
  els.statSize.textContent = current.size;
  els.statDepth.textContent = current.depth;
  els.statVersion.textContent = `v${current.index}`;
  els.values.textContent = `in_order: [${current.values.join(", ")}]`;
}

function readValue() {
  const value = Number.parseInt(els.input.value, 10);
  return Number.isNaN(value) ? null : value;
}

els.insertBtn.addEventListener("click", async () => {
  const value = readValue();
  if (value === null) return;
  applyState(await apiCall("/api/insert", { value }));
  els.input.value = "";
});

els.deleteBtn.addEventListener("click", async () => {
  const value = readValue();
  if (value === null) return;
  applyState(await apiCall("/api/delete", { value }));
  els.input.value = "";
});

els.randomBtn.addEventListener("click", async () => {
  const value = Math.floor(Math.random() * 100);
  applyState(await apiCall("/api/insert", { value }));
});

els.resetBtn.addEventListener("click", async () => {
  applyState(await apiCall("/api/reset"));
});

apiCall("/api/state").then(applyState);
