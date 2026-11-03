"""The model as one self-contained page: a tree to fold, a graph to read, a diff to spot.

Everything the page needs is inlined -- the graph library, the layout engine and the model
itself -- so it opens from the filesystem with no server and no network, and it keeps
working on a machine that has neither.

The fold is the whole point. A node the reader has not opened stands for everything under
it, and an edge is drawn between the nearest opened ancestors of its two ends, so the same
graph reads as four repos or as two hundred classes without a second model behind it.
"""
import json
import os
from typing import Any, Optional

VENDOR_SCRIPTS = ("cytoscape.min.js", "dagre.min.js", "cytoscape-dagre.js")

_PAGE_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f6f7f9; --panel: #ffffff; --ink: #14181f; --muted: #6b7280;
    --line: #d8dde5; --accent: #1d4ed8;
    --repo-app: #dbeafe; --repo-core: #dcfce7; --repo-yolox: #fef3c7;
    --repo-train: #f3e8ff; --repo-other: #e5e7eb;
  }
  * { box-sizing: border-box; }
  html, body { height: 100%; margin: 0; }
  body {
    font: 13px/1.45 ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
    color: var(--ink); background: var(--bg);
    display: grid; grid-template-rows: auto 1fr; height: 100vh;
  }
  header {
    display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
    padding: 8px 12px; background: var(--panel); border-bottom: 1px solid var(--line);
  }
  header .title { font-weight: 700; margin-right: 6px; }
  header .meta { color: var(--muted); font-size: 12px; }
  .group { display: flex; gap: 6px; align-items: center; }
  .group > label { color: var(--muted); }
  button, select, input[type=text] {
    font: inherit; padding: 3px 8px; border: 1px solid var(--line);
    border-radius: 6px; background: #fff; color: var(--ink);
  }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); color: var(--accent); }
  button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  button:disabled { color: #c4c9d2; cursor: default; border-color: var(--line); }
  button:disabled:hover { color: #c4c9d2; border-color: var(--line); }
  input[type=text] { min-width: 220px; }
  main { display: grid; grid-template-columns: 320px 1fr 340px; min-height: 0; }
  aside, section { min-height: 0; overflow: auto; background: var(--panel); }
  aside { border-right: 1px solid var(--line); }
  section.details { border-left: 1px solid var(--line); padding: 10px 12px; }
  #cy { height: 100%; width: 100%; background: var(--bg); }
  .members-head {
    font-weight: 700; margin: 10px 0 4px; display: flex; gap: 8px; align-items: baseline;
  }
  .members-head .count { font-weight: 400; }
  .member {
    display: block; padding: 2px 6px; border-radius: 4px; cursor: pointer;
    border-left: 3px solid transparent;
  }
  .member:hover { background: #eef2ff; }
  .member.current { background: #dbeafe; border-left-color: var(--accent); }
  .member .sig-line {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    white-space: pre-wrap; word-break: break-word;
  }
  .member .sig-doc { color: var(--muted); font-size: 12px; padding-left: 14px; }
  .member .sig-line { user-select: text; cursor: text; }
  .member .copy {
    float: right; margin-left: 6px; border: 1px solid var(--line); border-radius: 4px;
    background: #fff; color: var(--muted); font-size: 10px; padding: 0 5px; cursor: pointer;
    visibility: hidden;
  }
  .member:hover .copy { visibility: visible; }
  .member .copy:hover { color: var(--accent); border-color: var(--accent); }
  .details .member { margin-left: -6px; }
  .tree { padding: 6px 4px 40px 4px; }
  .row {
    display: flex; align-items: center; gap: 4px; padding: 1px 4px; border-radius: 4px;
    white-space: nowrap; cursor: default;
  }
  .row:hover { background: #eef2ff; }
  .row.selected { background: #dbeafe; }
  .caret {
    width: 14px; text-align: center; color: var(--muted); cursor: pointer;
    user-select: none;
  }
  .caret.leaf { visibility: hidden; }
  .label { cursor: pointer; }
  .label .kind { color: var(--muted); font-size: 11px; margin-right: 4px; }
  .count { color: var(--muted); font-size: 11px; }
  .status-added { color: #15803d; font-weight: 700; }
  .status-changed { color: #b45309; font-weight: 700; }
  .status-removed { color: #b91c1c; text-decoration: line-through; }
  .status-dirty { color: #7c3aed; font-weight: 700; }
  .details h2 { font-size: 14px; margin: 4px 0 2px; }
  .details .path { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                   font-size: 12px; color: var(--muted); word-break: break-all; }
  .details .doc { margin: 8px 0; }
  .details .sig {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
    background: #f3f4f6; border: 1px solid var(--line); border-radius: 6px;
    padding: 6px 8px; margin: 8px 0; white-space: pre-wrap; word-break: break-word;
  }
  .details ul { margin: 4px 0 10px; padding-left: 16px; }
  .details li { cursor: pointer; }
  .details li:hover { color: var(--accent); }
  /* A type this checkout defines is a box on the picture, so it is written as somewhere
     to go rather than as more of the same text. */
  .typeref {
    color: var(--accent); cursor: pointer; text-decoration: underline dotted;
    text-underline-offset: 2px;
  }
  .typeref:hover { background: #dbeafe; }
  /* What a Ctrl held over a signature in the graph is pointing at: the canvas cannot
     underline one word of a label, so the word says itself where the cursor is. */
  .typehint {
    position: fixed; z-index: 20; pointer-events: none; padding: 1px 6px;
    font: 11px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    background: var(--accent); color: #fff; border-radius: 4px;
  }
  .badge {
    display: inline-block; padding: 0 6px; border-radius: 10px; font-size: 11px;
    background: var(--repo-other); margin-right: 4px;
  }
  .legend { display: flex; gap: 10px; flex-wrap: wrap; font-size: 11px; color: var(--muted); }
  .legend span::before {
    content: ""; display: inline-block; width: 10px; height: 10px; margin-right: 4px;
    border-radius: 2px; vertical-align: -1px;
  }
  .legend .app::before { background: var(--repo-app); }
  .legend .core::before { background: var(--repo-core); }
  .legend .yolox::before { background: var(--repo-yolox); }
  .legend .train::before { background: var(--repo-train); }
</style>
</head>
<body>
<header>
  <span class="group">
    <button id="back" title="back (Alt+Left, or the mouse back button)">&larr;</button>
    <button id="forward" title="forward (Alt+Right)">&rarr;</button>
  </span>
  <span class="title">__TITLE__</span>
  <span class="meta">__META__</span>
  <span class="group">
    <label>open to</label>
    <button data-level="repo">repos</button>
    <button data-level="package">packages</button>
    <button data-level="module">modules</button>
    <button data-level="class">classes</button>
  </span>
  <span class="group">
    <label>edges</label>
    <button class="edge on" data-edge="constructs">constructs</button>
    <button class="edge" data-edge="calls">calls</button>
    <button class="edge" data-edge="imports">imports</button>
    <button class="edge" data-edge="inherits">inherits</button>
  </span>
  <span class="group">
    <label>focus</label>
    <button id="focus-toggle">off</button>
    <select id="focus-depth">
      <option value="1" selected>1 hop</option>
      <option value="2">2 hops</option>
      <option value="3">3 hops</option>
    </select>
  </span>
  <span class="group">
    <input type="text" id="search" placeholder="search symbol, Enter to open" list="symbols">
    <datalist id="symbols"></datalist>
  </span>
  <span class="group">
    <button id="signatures-toggle">signatures in graph</button>
    <button id="changed-only">changed only</button>
    <button id="relayout">re-layout</button>
    <button id="reset-diff" title="make the current code the baseline: the green/amber/red
colouring starts over from here (needs --serve)">reset colours</button>
  </span>
  <span class="legend" id="legend"></span>
</header>
<main>
  <aside><div class="tree" id="tree"></div></aside>
  <section><div id="cy"></div></section>
  <section class="details" id="details"></section>
</main>
<script>__CYTOSCAPE__</script>
<script>__DAGRE__</script>
<script>__CYTOSCAPE_DAGRE__</script>
<script>
const MODEL = __MODEL__;
</script>
<script>__APP__</script>
</body>
</html>
"""

_APP_JS = r"""
cytoscape.use(window.cytoscapeDagre);

const KIND_ORDER = {repo: 0, package: 1, folder: 1, module: 2, file: 3, class: 4,
                    function: 5, method: 6};
const KIND_GLYPH = {repo: "■", package: "▢", folder: "▤", module: "▤", class: "◆",
                    function: "ƒ", method: "·", file: "▪"};
const DRAWN_KINDS = {repo: 1, package: 1, module: 1, class: 1, function: 1, method: 1};

function glyphOf(node) {
  if (node.kind !== "file") return KIND_GLYPH[node.kind];
  return node.flags.indexOf("csv") >= 0 ? "▦" : "{}";
}
function isDrawn(id) {
  const node = nodesById.get(id);
  return node ? DRAWN_KINDS[node.kind] === 1 : false;
}
function hasTypes(node) {
  /* Nothing to reveal when the def carries no annotations: the two renderings agree. */
  return !!node.signature_typed && node.signature_typed !== node.signature_short;
}
function typesOf(node) {
  /* Which of the names in a signature this checkout defines. Absent on most defs and on
     every node a snapshot handed back as a ghost, so never assumed present. */
  return (node && node.types) || {};
}
function drawnSignature(node) {
  /* The label a box wears: types only for the defs the reader has asked about, and a dot
     on the ones still holding some back, so the click has something to aim at. */
  if (!node.signature_short) return node.name;
  if (state.typed.has(node.id)) return node.signature_typed;
  return hasTypes(node) ? node.signature_short + " ·" : node.signature_short;
}
const STORAGE_KEY = "arch-view-" + MODEL.root;

const nodesById = new Map();
for (const node of MODEL.nodes) nodesById.set(node.id, node);
const childrenOf = new Map();
for (const node of MODEL.nodes) {
  if (!childrenOf.has(node.id)) childrenOf.set(node.id, []);
  if (node.parent === null || node.parent === undefined) continue;
  if (!childrenOf.has(node.parent)) childrenOf.set(node.parent, []);
  childrenOf.get(node.parent).push(node.id);
}
for (const list of childrenOf.values()) {
  list.sort((a, b) => {
    const left = nodesById.get(a), right = nodesById.get(b);
    const byKind = KIND_ORDER[left.kind] - KIND_ORDER[right.kind];
    if (byKind !== 0) return byKind;
    return left.name.localeCompare(right.name);
  });
}
const roots = MODEL.nodes.filter(n => !n.parent).map(n => n.id);
const descendantCount = new Map();
function countDescendants(id) {
  if (descendantCount.has(id)) return descendantCount.get(id);
  let total = 0;
  for (const child of childrenOf.get(id) || []) total += 1 + countDescendants(child);
  descendantCount.set(id, total);
  return total;
}
for (const id of nodesById.keys()) countDescendants(id);

/* Opened at first: the roots, and below them any node that is its parent's only child -- a
   repo holding one package is a wrapper, and stopping there shows one box and nothing else. */
function initiallyExpanded() {
  const expanded = new Set();
  const pending = [...roots];
  while (pending.length) {
    const id = pending.pop();
    expanded.add(id);
    const children = childrenOf.get(id) || [];
    if (children.length === 1 && (childrenOf.get(children[0]) || []).length) pending.push(children[0]);
  }
  return expanded;
}

const state = {
  expanded: initiallyExpanded(),
  edges: {constructs: true, calls: false, imports: false, inherits: false},
  focus: true,
  focusDepth: 1,
  changedOnly: false,
  signatures: true,
  showPrivate: false,
  /* Which defs are currently showing their parameter types, one box at a time: types are
     what a reader wants for the one function they are asking about, and what turns the
     other twenty boxes into a wall. */
  typed: new Set(),
  history: [],
  historyIndex: -1,
  selected: null,
};
restoreState();

function saveState() {
  const payload = {
    expanded: Array.from(state.expanded), edges: state.edges, focus: state.focus,
    focusDepth: state.focusDepth, changedOnly: state.changedOnly, signatures: state.signatures,
    showPrivate: state.showPrivate, typed: Array.from(state.typed), history: state.history,
    historyIndex: state.historyIndex,
    selected: state.selected,
  };
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(payload)); } catch (e) {}
}
function restoreState() {
  let raw = null;
  try { raw = localStorage.getItem(STORAGE_KEY); } catch (e) { return; }
  if (!raw) return;
  const saved = JSON.parse(raw);
  const kept = (saved.expanded || []).filter(id => nodesById.has(id));
  if (kept.length) state.expanded = new Set(kept);
  if (saved.edges) Object.assign(state.edges, saved.edges);
  if (saved.focus !== undefined) state.focus = !!saved.focus;
  if (saved.focusDepth) state.focusDepth = saved.focusDepth;
  state.changedOnly = !!saved.changedOnly;
  if (saved.signatures !== undefined) state.signatures = !!saved.signatures;
  if (saved.showPrivate !== undefined) state.showPrivate = !!saved.showPrivate;
  if (saved.typed) state.typed = new Set(saved.typed.filter(id => nodesById.has(id)));
  if (saved.history) {
    const kept = saved.history.filter(id => nodesById.has(id));
    state.history = kept;
    state.historyIndex = Math.min(saved.historyIndex, kept.length - 1);
  }
  if (saved.selected && nodesById.has(saved.selected)) state.selected = saved.selected;
}

function parentOf(id) {
  const node = nodesById.get(id);
  if (!node || !node.parent) return null;
  return nodesById.has(node.parent) ? node.parent : null;
}
function ancestorsOpen(id) {
  let parent = parentOf(id);
  while (parent) {
    if (!state.expanded.has(parent)) return false;
    parent = parentOf(parent);
  }
  return true;
}
function hasChildren(id) { return (childrenOf.get(id) || []).length > 0; }
function isVisible(id) {
  if (state.expanded.has(id) && hasChildren(id)) return false;
  return ancestorsOpen(id);
}
function represent(id) {
  let current = id;
  while (current && !isVisible(current)) current = parentOf(current);
  return current;
}
function expandTo(id) {
  let parent = parentOf(id);
  while (parent) { state.expanded.add(parent); parent = parentOf(parent); }
}
function openToKind(kind) {
  state.expanded = new Set();
  for (const node of MODEL.nodes) {
    if (KIND_ORDER[node.kind] < KIND_ORDER[kind] && hasChildren(node.id)) {
      state.expanded.add(node.id);
    }
  }
  refresh();
}

/* Repos take the palette's colours in the order they were read; past the palette, grey. */
const REPO_PALETTE = ["app", "core", "yolox", "train"];
const repoColour = new Map();
for (const node of MODEL.nodes) {
  if (node.kind !== "repo") continue;
  repoColour.set(node.repo, REPO_PALETTE[repoColour.size] || "other");
}
function repoClass(node) {
  return repoColour.get(node.repo) || "other";
}
(function drawLegend() {
  const legend = document.getElementById("legend");
  for (const [repo, colour] of repoColour) {
    const item = document.createElement("span");
    item.className = colour;
    item.textContent = repo;
    legend.appendChild(item);
  }
})();
function statusOf(id) { return MODEL.status[id] || (nodesById.get(id).flags.includes("dirty") ? "dirty" : ""); }
function subtreeStatus(id) {
  const own = statusOf(id);
  if (own) return own;
  for (const child of childrenOf.get(id) || []) {
    const found = subtreeStatus(child);
    if (found) return found;
  }
  return "";
}

function moduleNameOf(id) {
  let current = id;
  while (current) {
    const node = nodesById.get(current);
    if (!node) return "";
    if (node.kind === "module") return node.name;
    current = parentOf(current);
  }
  return "";
}

function symbolOf(id) {
  /* A method's edges are its class's business: `TunnelNode.__init__ -> UnifiedSource`
     reads as `TunnelNode -> UnifiedSource`, which is the sentence a plan is made of. */
  const node = nodesById.get(id);
  if (!node) return id;
  if (node.kind !== "method") return id;
  const owner = parentOf(id);
  return owner ? owner : id;
}

function subtreeOf(id) {
  const collected = [id];
  for (const child of childrenOf.get(id) || []) collected.push(...subtreeOf(child));
  return collected;
}

function neighbourhoodElements() {
  /* What the selected symbol itself touches, N hops out, whatever is folded.
     The overview answers "what is there"; this answers "what would I break", so it
     works on the symbols the source names rather than on the boxes they sit in. */
  /* A def picked on purpose is the subject: the picture answers what IT calls, rather
     than folding it back into its class the way every other method is folded. */
  const anchorId = state.selected;
  const anchor = nodesById.get(anchorId);
  const anchorIsDef = anchor.kind === "method" || anchor.kind === "function";
  const project = (id) => (id === anchorId ? id : symbolOf(id));
  const seeds = new Set();
  if (anchorIsDef) {
    seeds.add(anchorId);
  } else {
    for (const id of subtreeOf(anchorId)) seeds.add(symbolOf(id));
  }
  const reachable = new Set(seeds);
  let frontier = new Set(seeds);
  const usable = [];
  const degree = new Map();
  const projected = new Set();
  for (const edge of MODEL.edges) {
    const source = project(edge.source), target = project(edge.target);
    // The subject's own edges are shown whatever the kind buttons say: they are the
    // answer to the question that picking it asked.
    const isSubjects = source === anchorId || target === anchorId;
    if (!state.edges[edge.kind] && !isSubjects) continue;
    if (source === target) continue;
    const key = source + "|" + target + "|" + edge.kind;
    if (projected.has(key)) continue;
    projected.add(key);
    usable.push({source, target, kind: edge.kind, line: edge.line});
    degree.set(source, (degree.get(source) || 0) + 1);
    degree.set(target, (degree.get(target) || 0) + 1);
  }
  /* A utility everything touches -- LOG, ensure_folder -- says nothing about who its
     users are to each other, and walking through one drags in the whole checkout. It is
     shown when something reached leads to it, and the walk stops there. */
  const HUB_DEGREE = 20;
  const passable = (id) => seeds.has(id) || (degree.get(id) || 0) <= HUB_DEGREE;
  for (let hop = 0; hop < state.focusDepth; hop++) {
    const next = new Set();
    for (const edge of usable) {
      if (frontier.has(edge.source) && passable(edge.source) && !reachable.has(edge.target)) {
        next.add(edge.target);
      }
      if (frontier.has(edge.target) && passable(edge.target) && !reachable.has(edge.source)) {
        next.add(edge.source);
      }
    }
    for (const id of next) reachable.add(id);
    frontier = next;
  }
  const elements = [];
  /* The subject says which module it is in, and the rest say so only when they are
     somewhere else: a neighbour from the same file repeating its name is noise. */
  const home = moduleNameOf(anchorId);
  for (const id of reachable) {
    const node = nodesById.get(id);
    if (!node) continue;
    if (seeds.has(id) === false && node.kind === "package") continue;
    let context = node.kind === "module" || node.kind === "package" ? "" : moduleNameOf(id);
    if (context === home && id !== anchorId) context = "";
    let title = node.name;
    if (id === anchorId && anchorIsDef) {
      const owner = parentOf(id);
      const ownerNode = owner ? nodesById.get(owner) : null;
      const written = drawnSignature(node);
      title = ownerNode && ownerNode.kind === "class" ? ownerNode.name + "." + written
                                                      : written;
    }
    const plain = context ? title + "\n" + context : title;
    elements.push({data: {
      id: node.id, label: KIND_GLYPH[node.kind] + " " + plain, kind: node.kind,
      repo: repoClass(node), status: subtreeStatus(node.id), size: 30,
    }});
  }
  const seen = new Set();
  for (const edge of usable) {
    if (!reachable.has(edge.source) || !reachable.has(edge.target)) continue;
    const id = edge.source + ">" + edge.target + ">" + edge.kind;
    if (seen.has(id)) continue;
    seen.add(id);
    elements.push({data: {id, source: edge.source, target: edge.target, kind: edge.kind,
                          width: 1.5, label: ""}});
  }
  memberElements(elements, anchorId);
  return elements;
}

function buildElements() {
  const visible = [];
  for (const node of MODEL.nodes) {
    if (!isDrawn(node.id)) continue;
    if (isVisible(node.id)) visible.push(node);
  }
  const keep = new Set(visible.map(n => n.id));
  if (state.changedOnly) {
    for (const id of Array.from(keep)) if (!subtreeStatus(id)) keep.delete(id);
  }
  const aggregated = new Map();
  for (const edge of MODEL.edges) {
    if (!state.edges[edge.kind]) continue;
    const source = represent(edge.source), target = represent(edge.target);
    if (!source || !target || source === target) continue;
    if (!keep.has(source) || !keep.has(target)) continue;
    const key = source + "|" + target + "|" + edge.kind;
    const found = aggregated.get(key);
    if (found) { found.weight += 1; continue; }
    aggregated.set(key, {source, target, kind: edge.kind, weight: 1});
  }
  if (state.focus && state.selected) return neighbourhoodElements();
  const elements = [];
  for (const node of visible) {
    if (!keep.has(node.id)) continue;
    const total = descendantCount.get(node.id);
    const suffix = state.expanded.has(node.id) || total === 0 ? "" : " +" + total;
    elements.push({data: {
      id: node.id, label: KIND_GLYPH[node.kind] + " " + node.name + suffix,
      kind: node.kind, repo: repoClass(node), status: subtreeStatus(node.id),
      size: 26 + Math.min(26, Math.round(Math.log2(1 + total) * 5)),
    }});
  }
  for (const edge of aggregated.values()) {
    elements.push({data: {
      id: edge.source + ">" + edge.target + ">" + edge.kind,
      source: edge.source, target: edge.target, kind: edge.kind,
      width: Math.min(6, 1 + Math.log2(edge.weight)), label: edge.weight > 1 ? String(edge.weight) : "",
    }});
  }
  if (state.selected) memberElements(elements, state.selected);
  return elements;
}

const cy = cytoscape({
  container: document.getElementById("cy"),
  wheelSensitivity: 0.2,
  style: [
    {selector: "node", style: {
      "label": "data(label)", "font-size": 11, "text-valign": "center",
      "text-wrap": "wrap", "text-max-width": 160, "shape": "round-rectangle",
      "width": "label", "height": "label", "padding": "6px",
      "border-width": 1, "border-color": "#94a3b8", "background-color": "#e5e7eb",
      "color": "#14181f",
    }},
    {selector: "node[repo = 'app']", style: {"background-color": "#dbeafe", "border-color": "#1d4ed8"}},
    {selector: "node[repo = 'core']", style: {"background-color": "#dcfce7", "border-color": "#15803d"}},
    {selector: "node[repo = 'yolox']", style: {"background-color": "#fef3c7", "border-color": "#b45309"}},
    {selector: "node[repo = 'train']", style: {"background-color": "#f3e8ff", "border-color": "#7e22ce"}},
    {selector: "node[kind = 'repo']", style: {"font-weight": "bold", "font-size": 14}},
    {selector: "node[kind = 'package']", style: {"font-weight": "bold"}},
    {selector: "node[kind = 'method'], node[kind = 'function']", style: {"shape": "round-tag"}},
    {selector: "node[status = 'added']", style: {"border-width": 4, "border-color": "#15803d"}},
    {selector: "node[status = 'changed']", style: {"border-width": 4, "border-color": "#b45309"}},
    {selector: "node[status = 'dirty']", style: {"border-width": 4, "border-color": "#7c3aed"}},
    {selector: "node[status = 'removed']", style: {"border-width": 3, "border-style": "dashed",
      "border-color": "#b91c1c", "opacity": 0.55}},
    {selector: "node[summary = 1]", style: {
      "font-family": "ui-sans-serif, system-ui, sans-serif", "font-size": 11,
      "background-color": "#eef2f7", "border-color": "#9ca3af", "color": "#6b7280",
    }},
    {selector: "node[member = 1]", style: {
      "font-family": "ui-monospace, SFMono-Regular, Menlo, monospace", "font-size": 10,
      "text-wrap": "wrap", "text-max-width": 620, "text-justification": "left",
      "shape": "round-rectangle", "padding": "5px", "border-width": 1,
      "border-style": "dashed", "background-opacity": 0.92,
    }},
    {selector: "node[typeref = 1]", style: {
      "border-style": "solid", "font-size": 10, "padding": "4px", "opacity": 0.95,
    }},
    {selector: "edge[kind = 'typeref']", style: {
      "curve-style": "straight", "line-color": "#9ca3af", "line-style": "dotted",
      "target-arrow-shape": "none", "opacity": 0.6,
    }},
    {selector: "edge[kind = 'member']", style: {
      "curve-style": "taxi", "taxi-direction": "downward", "taxi-turn": 18,
      "taxi-turn-min-distance": 6, "line-color": "#9ca3af", "line-style": "dashed",
      "target-arrow-shape": "none", "opacity": 0.7,
    }},
    {selector: "node.selected", style: {"border-width": 5, "border-color": "#1d4ed8"}},
    {selector: "edge", style: {
      "curve-style": "bezier", "target-arrow-shape": "triangle", "width": "data(width)",
      "arrow-scale": 0.8, "font-size": 9, "label": "data(label)", "color": "#6b7280",
      "text-background-color": "#f6f7f9", "text-background-opacity": 1,
      "line-color": "#9ca3af", "target-arrow-color": "#9ca3af", "opacity": 0.8,
    }},
    {selector: "edge[kind = 'constructs']", style: {"line-color": "#1d4ed8", "target-arrow-color": "#1d4ed8"}},
    {selector: "edge[kind = 'calls']", style: {"line-color": "#15803d", "target-arrow-color": "#15803d"}},
    {selector: "edge[kind = 'imports']", style: {"line-style": "dashed", "opacity": 0.45}},
    {selector: "edge[kind = 'inherits']", style: {"line-color": "#7e22ce",
      "target-arrow-color": "#7e22ce", "target-arrow-shape": "triangle-tee"}},
  ],
});

function makeRoomForMembers(anchor, membersRight) {
  /* The hung boxes are wider than the class they hang from, and dagre put the next rank
     right beside it. Everything to the right moves out by what the widest of them needs,
     which keeps dagre's arrangement and only stretches the gap. */
  const gap = 70;
  const rightward = [];
  let closest = null;
  cy.nodes("[!member]").forEach((node) => {
    if (node.id() === anchor.id()) return;
    const nodeLeft = node.position().x - node.width() / 2;
    if (nodeLeft <= anchor.position().x) return;
    rightward.push(node);
    if (closest === null || nodeLeft < closest) closest = nodeLeft;
  });
  if (closest === null) return;
  const shift = membersRight + gap - closest;
  if (shift <= 0) return;
  for (const node of rightward) {
    const position = node.position();
    node.position({x: position.x + shift, y: position.y});
  }
}

function placeMembers(members) {
  /* Straight down from the anchor, in the order the file declares them, their left edges
     under its left edge: signatures are read from the left like the code they are. The
     types a box has open go beside it on its own line, where they were written. */
  if (members.empty()) return;
  const anchor = cy.getElementById(members.first().data("anchor"));
  if (anchor.empty()) return;
  const chipsOf = new Map();
  members.forEach((member) => {
    if (!member.data("typeref")) return;
    const owner = member.data("owner");
    if (!chipsOf.has(owner)) chipsOf.set(owner, []);
    chipsOf.get(owner).push(member);
  });
  const origin = anchor.position();
  const left = origin.x - anchor.outerWidth() / 2;
  let cursor = origin.y + anchor.outerHeight() / 2 + 34;
  let widest = anchor.outerWidth();
  /* Outer, not inner: a box is its text plus its padding and border, and stacking them by
     the text alone overlapped every pair of neighbours by three pixels -- which is a click
     landing on the box above the one it was aimed at. */
  members.forEach((member) => {
    if (member.data("typeref")) return;
    const height = member.outerHeight();
    cursor += height / 2;
    member.position({x: left + member.outerWidth() / 2, y: cursor});
    let reach = member.outerWidth();
    for (const chip of chipsOf.get(member.id()) || []) {
      reach += 18;
      chip.position({x: left + reach + chip.outerWidth() / 2, y: cursor});
      reach += chip.outerWidth();
    }
    cursor += height / 2 + 10;
    if (reach > widest) widest = reach;
  });
  makeRoomForMembers(anchor, left + widest);
}

function runLayout() {
  const members = cy.nodes("[member = 1]");
  const hung = members.union(members.connectedEdges());
  const ranked = cy.elements().difference(hung);
  const layout = ranked.layout({name: "dagre", rankDir: "LR", nodeSep: 24, rankSep: 90,
                                edgeSep: 8, animate: false, fit: false, padding: 20});
  layout.run();
  placeMembers(members);
  cy.fit(30);
  if (cy.zoom() > 1.4) { cy.zoom(1.4); cy.center(); }
}

function refresh(relayout = true) {
  const elements = buildElements();
  cy.batch(() => {
    cy.elements().remove();
    cy.add(elements);
    if (state.selected) {
      const anchor = state.focus ? state.selected : represent(state.selected);
      const selected = cy.getElementById(anchor);
      if (selected.nonempty()) selected.addClass("selected");
    }
  });
  if (relayout) runLayout();
  renderTree();
  renderDetails();
  syncButtons();
  saveState();
}

function goTo(id) {
  /* Somewhere else on the picture: the tree has to be opened to it before the graph can
     be asked to draw it, which is what the search box does too. */
  if (!nodesById.has(id)) return;
  expandTo(id);
  refresh();
  select(id, true);
}

cy.on("tap", "node", (event) => {
  const id = event.target.id();
  /* A type chip is the one member box that is about somewhere else, so it navigates. */
  if (event.target.data("typeref")) {
    goTo(event.target.data("goto"));
    return;
  }
  if (event.target.data("summary")) {
    state.showPrivate = !state.showPrivate;
    refresh();
    return;
  }
  /* A def hanging under its class is read, not navigated to: the click the reader has in
     hand is worth more spent on its types than on re-anchoring the picture around it --
     the details panel's own list still makes any of them the subject. Boxes with nothing
     to reveal keep the old behaviour; they carry no dot, so no toggle was aimed at. */
  const node = nodesById.get(id);
  if (event.target.data("member") && node && hasTypes(node)) {
    if (state.typed.has(id)) state.typed.delete(id); else state.typed.add(id);
    refresh();
    return;
  }
  select(id, false);
});
function typeAtPoint(target, point) {
  /* Which type name a point inside a hung signature box lands on. The box is drawn in a
     monospace font, so one character is as wide as the label over its longest line and a
     point is a row and a column rather than a search through geometry. Cytoscape draws
     labels into a canvas -- a word in one cannot be a link -- so this is what stands in
     for the panel's underlined names, behind Ctrl so an ordinary click still opens types. */
  const element = target[0] || target;
  if (!element || !element.data("member") || element.data("typeref")) return null;
  const types = typesOf(nodesById.get(element.id()));
  const names = Object.keys(types);
  if (!names.length) return null;
  const drawn = element._private.rscratch;
  const lines = drawn && drawn.labelWrapCachedLines;
  if (!lines || !lines.length || !drawn.labelWidth) return null;
  let widest = 0;
  for (const line of lines) if (line.length > widest) widest = line.length;
  const charWidth = drawn.labelWidth / widest;
  const rowHeight = drawn.labelHeight / lines.length;
  /* A box is a good deal taller than the text it holds -- ten pixels of signature inside
     twenty-three of box -- and a click in that margin is aimed at the nearest character,
     not at nothing. Missing it was most of the box. Vertically there is nothing else the
     margin could mean, so the row is simply clamped; horizontally only as much slack as
     the margin is actually wide is forgiven, or a click in the empty right-hand end of a
     short wrapped line would land on that line's last word. */
  const raw = Math.floor((point.y - (drawn.labelY - drawn.labelHeight / 2)) / rowHeight);
  const row = Math.min(lines.length - 1, Math.max(0, raw));
  const text = lines[row];
  const margin = Math.max(0, (element.outerWidth() - drawn.labelWidth) / 2) / charWidth;
  let column = (point.x - (drawn.labelX - drawn.labelWidth / 2)) / charWidth;
  const written = text.replace(/\s+$/, "").length;
  if (column < 0 && column >= -margin) column = 0;
  if (column > written && column <= written + margin) column = written - 0.001;
  for (const name of names) {
    let at = text.indexOf(name);
    while (at >= 0) {
      const before = at === 0 ? "" : text[at - 1];
      const after = text[at + name.length] || "";
      const whole = !/[A-Za-z0-9_.]/.test(before) && !/[A-Za-z0-9_]/.test(after);
      if (whole && column >= at && column < at + name.length) return types[name];
      at = text.indexOf(name, at + 1);
    }
  }
  return null;
}

const typeHint = document.createElement("div");
typeHint.className = "typehint";
typeHint.style.display = "none";
document.body.appendChild(typeHint);

function showTypeHint(targetId, mouse) {
  const node = targetId ? nodesById.get(targetId) : null;
  if (!node) {
    typeHint.style.display = "none";
    document.getElementById("cy").style.cursor = "";
    return;
  }
  typeHint.textContent = "\u2192 " + node.name;
  typeHint.style.left = (mouse.clientX + 12) + "px";
  typeHint.style.top = (mouse.clientY - 24) + "px";
  typeHint.style.display = "block";
  document.getElementById("cy").style.cursor = "pointer";
}

function modelPoint(mouse) {
  /* Where the mouse is in the graph's own coordinates, which is what a label was laid
     out in. */
  const box = document.getElementById("cy").getBoundingClientRect();
  const pan = cy.pan(), zoom = cy.zoom();
  return {x: (mouse.clientX - box.left - pan.x) / zoom,
          y: (mouse.clientY - box.top - pan.y) / zoom};
}

function memberAt(point) {
  /* Cytoscape keeps a ctrl+click to itself -- it is its add-to-selection gesture -- so
     the box under the pointer is found here rather than asked for. */
  let found = null, nearest = Infinity;
  cy.nodes("[member = 1]").forEach((node) => {
    if (node.data("typeref")) return;
    const box = node.boundingBox();
    if (point.x < box.x1 || point.x > box.x2) return;
    if (point.y < box.y1 || point.y > box.y2) return;
    /* Boxes stack close enough to touch, so the one meant is the one the point sits
       deepest inside rather than whichever came first. */
    const away = Math.abs(point.y - (box.y1 + box.y2) / 2);
    if (away < nearest) { nearest = away; found = node; }
  });
  return found;
}

const graph = document.getElementById("cy");
graph.addEventListener("mousemove", (event) => {
  if (!(event.ctrlKey || event.metaKey)) { showTypeHint(null); return; }
  const point = modelPoint(event);
  const node = memberAt(point);
  showTypeHint(node ? typeAtPoint(node, point) : null, event);
});
graph.addEventListener("mouseleave", () => showTypeHint(null));
graph.addEventListener("click", (event) => {
  /* Ctrl says the click is aimed at a word rather than at the box. Missing one does
     nothing: folding the types away was not what the reader asked for. */
  if (!(event.ctrlKey || event.metaKey)) return;
  const point = modelPoint(event);
  const node = memberAt(point);
  const wanted = node ? typeAtPoint(node, point) : null;
  showTypeHint(null);
  if (!wanted) return;
  event.preventDefault();
  event.stopPropagation();
  goTo(wanted);
});
document.addEventListener("keyup", (event) => {
  if (event.key === "Control" || event.key === "Meta") showTypeHint(null);
});

cy.on("dbltap", "node", (event) => {
  const id = event.target.id();
  if (!hasChildren(id)) return;
  state.expanded.add(id);
  refresh();
});
cy.on("cxttap", "node", (event) => {
  const parent = parentOf(event.target.id());
  if (!parent) return;
  state.expanded.delete(parent);
  refresh();
});

function pushHistory(id) {
  if (state.history[state.historyIndex] === id) return;
  state.history = state.history.slice(0, state.historyIndex + 1);
  state.history.push(id);
  if (state.history.length > 60) state.history.shift();
  state.historyIndex = state.history.length - 1;
}

function stepHistory(delta) {
  /* Where the reader came from, which a graph that redraws itself under every click
     otherwise loses. */
  const wanted = state.historyIndex + delta;
  if (wanted < 0 || wanted >= state.history.length) return;
  state.historyIndex = wanted;
  const id = state.history[wanted];
  if (!nodesById.has(id)) return;
  expandTo(id);
  state.selected = id;
  refresh();
}

function select(id, center) {
  pushHistory(id);
  state.selected = id;
  if (!isDrawn(id)) {
    /* A config is read, not drawn: the graph keeps whatever question it was answering. */
    renderTree();
    renderDetails();
    syncButtons();
    saveState();
    return;
  }
  cy.nodes().removeClass("selected");
  const representative = state.focus ? id : represent(id);
  const element = cy.getElementById(representative);
  if (element.nonempty()) {
    element.addClass("selected");
    /* Focus has just laid the neighbourhood out and fitted it; centring on the box
       after that only pushes the neighbours off the edge. */
    if (center && !state.focus) cy.animate({center: {eles: element}}, {duration: 200});
  }
  if (state.focus) refresh(); else { renderTree(); renderDetails(); saveState(); }
}

/* ------------------------------------------------------------------ the tree */
function renderTree() {
  const container = document.getElementById("tree");
  container.textContent = "";
  for (const id of roots) container.appendChild(treeRow(id, 0));
}
function treeRow(id, depth) {
  const node = nodesById.get(id);
  const wrapper = document.createElement("div");
  const row = document.createElement("div");
  row.className = "row" + (state.selected === id ? " selected" : "");
  row.style.paddingLeft = (depth * 12 + 4) + "px";
  const caret = document.createElement("span");
  caret.className = "caret" + (hasChildren(id) ? "" : " leaf");
  caret.textContent = state.expanded.has(id) ? "▾" : "▸";
  caret.onclick = () => {
    if (state.expanded.has(id)) state.expanded.delete(id); else state.expanded.add(id);
    refresh();
  };
  row.appendChild(caret);
  const label = document.createElement("span");
  label.className = "label";
  const kind = document.createElement("span");
  kind.className = "kind";
  kind.textContent = glyphOf(node);
  label.appendChild(kind);
  const text = document.createElement("span");
  text.textContent = node.name;
  const status = statusOf(id);
  if (status) text.className = "status-" + status;
  label.appendChild(text);
  const total = descendantCount.get(id);
  if (total) {
    const count = document.createElement("span");
    count.className = "count";
    count.textContent = " " + total;
    label.appendChild(count);
  }
  label.onclick = () => select(id, true);
  row.appendChild(label);
  wrapper.appendChild(row);
  if (state.expanded.has(id)) {
    for (const child of childrenOf.get(id) || []) wrapper.appendChild(treeRow(child, depth + 1));
  }
  return wrapper;
}

/* --------------------------------------------------------------- the members */
function containerOf(id) {
  /* What a signature list is about: the class a method belongs to, the module a
     free function belongs to, or the thing itself when it is one of those. */
  let current = id;
  while (current) {
    const node = nodesById.get(current);
    if (!node) return null;
    if (node.kind === "class" || node.kind === "module") return current;
    current = parentOf(current);
  }
  return null;
}

function memberRows(containerId) {
  const container = nodesById.get(containerId);
  const rows = [];
  for (const childId of childrenOf.get(containerId) || []) {
    const child = nodesById.get(childId);
    if (!child.signature) continue;
    rows.push(child);
  }
  rows.sort((left, right) => left.line - right.line);
  if (container.kind === "class") return rows;
  const classes = [], functions = [];
  for (const row of rows) {
    if (row.kind === "class") classes.push(row); else functions.push(row);
  }
  return classes.concat(functions);
}

function isPrivate(name) {
  /* One leading underscore and no second one: `_project` is the class's own business,
     `__call__` is how everyone else uses it. */
  return name.startsWith("_") && !name.startsWith("__");
}

function splitMembers(rows) {
  const open = [], hidden = [];
  for (const row of rows) {
    if (isPrivate(row.name)) hidden.push(row); else open.push(row);
  }
  return {open, hidden};
}

function memberElements(elements, anchorId) {
  /* Every method as a box of its own, hung below the class. Dagre ranks the graph left
     to right and knows only one direction, so these are kept out of it and placed by
     hand under the anchor -- what a class uses goes right, what it is made of goes down. */
  if (!state.signatures) return;
  const anchor = nodesById.get(anchorId);
  if (!anchor || (anchor.kind !== "class" && anchor.kind !== "module")) return;
  const rows = memberRows(anchorId);
  if (!rows.length) return;
  const split = splitMembers(rows);
  const drawn = state.showPrivate ? rows : split.open;
  const taken = new Set();
  for (const element of elements) taken.add(element.data.id);
  for (const row of drawn) {
    if (taken.has(row.id)) continue;
    const written = drawnSignature(row);
    /* Types push a signature well past what a bare parameter list needed; the box wraps
       at 620px, so it has the room for several lines before this has to bite. */
    const label = written.length > 200 ? written.slice(0, 199) + "\u2026" : written;
    elements.push({data: {
      id: row.id, label: label, kind: row.kind, member: 1, anchor: anchorId,
      repo: repoClass(row), status: statusOf(row.id), size: 24,
    }});
    elements.push({data: {
      id: anchorId + ">member>" + row.id, source: anchorId, target: row.id,
      kind: "member", width: 1.5, label: "",
    }});
    typeChips(elements, row, written, taken);
  }
  if (!split.hidden.length) return;
  const summaryId = anchorId + ">private";
  const mark = state.showPrivate ? "\u2212 hide " : "+ ";
  elements.push({data: {
    id: summaryId, label: mark + split.hidden.length + " private", member: 1,
    summary: 1, anchor: anchorId, kind: "private", repo: repoClass(nodesById.get(anchorId)),
  }});
  elements.push({data: {
    id: anchorId + ">member>private", source: anchorId, target: summaryId,
    kind: "member", width: 1.5, label: "",
  }});
}

function typeChips(elements, row, written, taken) {
  /* A type the reader has just asked to see is a thing on the picture, so it is given a
     box of its own beside the def that names it: reading `model_config: ModelConfig` and
     going to ModelConfig become the same gesture. Only the box whose types are open gets
     them -- the same reason the types themselves are per box. */
  if (!state.typed.has(row.id)) return;
  const types = typesOf(row);
  for (const name of Object.keys(types)) {
    if (written.indexOf(name) < 0) continue;
    const target = nodesById.get(types[name]);
    if (!target) continue;
    const id = row.id + ">type>" + name;
    if (taken.has(id)) continue;
    taken.add(id);
    elements.push({data: {
      id: id, label: KIND_GLYPH.class + " " + target.name, kind: "class", member: 1,
      typeref: 1, owner: row.id, goto: target.id, repo: repoClass(target),
      status: statusOf(target.id), size: 22,
    }});
    elements.push({data: {
      id: id + ">edge", source: row.id, target: id, kind: "typeref", width: 1, label: "",
    }});
  }
}

function signatureInto(box, types, text) {
  /* The panel writes a signature as text except for the names this checkout defines,
     which become the way to get to them: the reader asks "what is a ModelConfig" exactly
     where the word is, and the answer is one click rather than a search. */
  const names = Object.keys(types).filter(name => text.indexOf(name) >= 0);
  if (!names.length) { box.textContent = text; return; }
  /* Longest first, so `tunnel.ModelConfig` is not eaten by its own suffix. */
  names.sort((left, right) => right.length - left.length);
  const escaped = names.map(name => name.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp("(?<![A-Za-z0-9_.])(" + escaped.join("|") + ")\\b", "g");
  let cursor = 0;
  let match;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      box.appendChild(document.createTextNode(text.slice(cursor, match.index)));
    }
    const name = match[1];
    const link = document.createElement("span");
    link.className = "typeref";
    link.textContent = name;
    link.title = "go to " + name;
    link.onclick = (event) => { event.stopPropagation(); goTo(types[name]); };
    box.appendChild(link);
    cursor = match.index + name.length;
  }
  if (cursor < text.length) box.appendChild(document.createTextNode(text.slice(cursor)));
}

function memberSection(panel, title, count, note) {
  const heading = document.createElement("div");
  heading.className = "members-head";
  const label = document.createElement("span");
  label.textContent = title;
  heading.appendChild(label);
  const tally = document.createElement("span");
  tally.className = "count";
  tally.textContent = note ? count + " · " + note : String(count);
  heading.appendChild(tally);
  panel.appendChild(heading);
}

function memberRow(panel, row, selectedId) {
  const entry = document.createElement("div");
  entry.className = "member" + (row.id === selectedId ? " current" : "");
  const copy = document.createElement("button");
  copy.className = "copy";
  copy.textContent = "copy";
  copy.title = "copy the signature";
  copy.onclick = (event) => {
    event.stopPropagation();
    navigator.clipboard.writeText(row.signature);
    copy.textContent = "copied";
    setTimeout(() => { copy.textContent = "copy"; }, 900);
  };
  entry.appendChild(copy);
  const signature = document.createElement("div");
  signature.className = "sig-line";
  const status = statusOf(row.id);
  if (status) signature.classList.add("status-" + status);
  signatureInto(signature, typesOf(row), row.signature);
  entry.appendChild(signature);
  if (row.doc) {
    const doc = document.createElement("div");
    doc.className = "sig-doc";
    doc.textContent = row.doc;
    entry.appendChild(doc);
  }
  /* Selecting redraws this panel, which would drop a half-made text selection --
     so dragging over a signature to copy it is left alone. */
  entry.onclick = () => {
    const dragged = window.getSelection().toString();
    if (dragged) return;
    select(row.id, false);
  };
  entry.ondblclick = () => navigator.clipboard.writeText(row.signature);
  panel.appendChild(entry);
}

function renderMembers(panel, selectedId) {
  /* Every definition of the thing the selection sits in, in the order the file has
     them, public first: what the rest of the system may call is read before what the
     thing does to itself. */
  const containerId = containerOf(selectedId);
  if (!containerId) return;
  const container = nodesById.get(containerId);
  const rows = memberRows(containerId);
  if (!rows.length) return;
  const split = splitMembers(rows);
  const where = containerId === selectedId ? "" : "in " + container.name;
  if (split.open.length) {
    const title = container.kind === "class" ? "methods" : "defines";
    memberSection(panel, title, split.open.length, where);
    for (const row of split.open) memberRow(panel, row, selectedId);
  }
  if (split.hidden.length) {
    memberSection(panel, "private", split.hidden.length, "");
    for (const row of split.hidden) memberRow(panel, row, selectedId);
  }
}

/* --------------------------------------------------------------- the details */
function renderDetails() {
  const panel = document.getElementById("details");
  panel.textContent = "";
  if (!state.selected) {
    panel.innerHTML = "<p class='doc'>Click a node. Double-click opens it, right-click " +
      "closes its parent. The tree on the left folds the same state.</p>";
    return;
  }
  const node = nodesById.get(state.selected);
  const heading = document.createElement("h2");
  heading.textContent = node.name;
  panel.appendChild(heading);
  const badges = document.createElement("div");
  const kindBadge = document.createElement("span");
  kindBadge.className = "badge";
  kindBadge.textContent = node.kind;
  badges.appendChild(kindBadge);
  for (const flag of node.flags) {
    const badge = document.createElement("span");
    badge.className = "badge";
    badge.textContent = flag;
    badges.appendChild(badge);
  }
  const status = statusOf(node.id);
  if (status) {
    const badge = document.createElement("span");
    badge.className = "badge status-" + status;
    badge.textContent = status;
    badges.appendChild(badge);
  }
  panel.appendChild(badges);
  if (node.file) {
    const path = document.createElement("div");
    path.className = "path";
    path.textContent = node.line ? node.file + ":" + node.line : node.file;
    path.title = "click to copy";
    path.onclick = () => navigator.clipboard.writeText(path.textContent);
    panel.appendChild(path);
  }
  /* A package/folder/repo has no single file (node.file is null) but is still a real
     selection -- the id alone is enough for Claude to look it up in MODEL.nodes /
     MODEL.edges, so the button belongs on every kind, not only ones with a file. */
  const copyButton = document.createElement("button");
  copyButton.textContent = "copy for prompt";
  copyButton.onclick = () => {
    /* What it does is the reader's business, not the prompt's: a prompt needs to say
       which thing is meant, and the docstring is already in the file. The leading tag
       names the page and the node id so Claude, reading this in a prompt, knows to
       open that file and look up MODEL.nodes / MODEL.edges by id for the graph
       context (who constructs/calls/imports this) instead of guessing from the name. */
    const pageName = "architecture-" + MODEL.root.split("/").pop() + ".html";
    const named = node.signature ? node.signature : node.kind + " " + node.name;
    let reference;
    if (node.kind === "file") reference = node.file;
    else if (node.file) reference = named + " -- " + node.file + ":" + node.line;
    else reference = named;
    const tagged = "[" + pageName + " node:" + node.id + "] " + reference;
    navigator.clipboard.writeText(tagged);
  };
  panel.appendChild(copyButton);
  if (node.signature) {
    /* A class is declared in one line and used through another: what a caller has to
       satisfy is __init__, so the constructor is shown beside the declaration. */
    let text = node.signature;
    let types = typesOf(node);
    if (node.kind === "class") {
      for (const child of childrenOf.get(node.id) || []) {
        const method = nodesById.get(child);
        if (method.name === "__init__" && method.signature) {
          text = node.signature + "\n" + method.signature;
          /* The constructor shown beside the declaration brings its own types with it. */
          types = Object.assign({}, types, typesOf(method));
          break;
        }
      }
    }
    const signature = document.createElement("div");
    signature.className = "sig";
    signatureInto(signature, types, text);
    signature.title = "click to copy";
    signature.onclick = () => navigator.clipboard.writeText(text);
    panel.appendChild(signature);
  }
  if (node.doc) {
    const doc = document.createElement("p");
    doc.className = "doc";
    doc.textContent = node.doc;
    panel.appendChild(doc);
  }
  if (node.bases.length) {
    panel.appendChild(listOf("inherits from", node.bases.map(name => ({label: name}))));
  }
  renderMembers(panel, node.id);
}
function shortName(id) {
  const node = nodesById.get(id);
  return node ? node.name : id;
}
function listOf(title, items) {
  const box = document.createElement("div");
  const heading = document.createElement("h2");
  heading.textContent = title;
  box.appendChild(heading);
  const list = document.createElement("ul");
  for (const item of items) {
    const entry = document.createElement("li");
    entry.textContent = item.label;
    if (item.id) {
      entry.onclick = () => { expandTo(item.id); refresh(); select(item.id, true); };
    }
    list.appendChild(entry);
  }
  box.appendChild(list);
  return box;
}

/* -------------------------------------------------------------- the controls */
function syncButtons() {
  for (const button of document.querySelectorAll("button.edge")) {
    button.classList.toggle("on", state.edges[button.dataset.edge]);
  }
  const focusButton = document.getElementById("focus-toggle");
  focusButton.classList.toggle("on", state.focus);
  focusButton.textContent = state.focus ? "on" : "off";
  document.getElementById("changed-only").classList.toggle("on", state.changedOnly);
  document.getElementById("signatures-toggle").classList.toggle("on", state.signatures);
  document.getElementById("focus-depth").value = String(state.focusDepth);
  document.getElementById("back").disabled = state.historyIndex <= 0;
  document.getElementById("forward").disabled =
    state.historyIndex >= state.history.length - 1;
}
for (const button of document.querySelectorAll("button[data-level]")) {
  button.onclick = () => openToKind(button.dataset.level);
}
for (const button of document.querySelectorAll("button.edge")) {
  button.onclick = () => {
    state.edges[button.dataset.edge] = !state.edges[button.dataset.edge];
    refresh();
  };
}
document.getElementById("focus-toggle").onclick = () => { state.focus = !state.focus; refresh(); };
document.getElementById("focus-depth").onchange = (event) => {
  state.focusDepth = Number(event.target.value);
  if (state.focus) refresh();
};
document.getElementById("signatures-toggle").onclick = () => {
  state.signatures = !state.signatures;
  refresh();
};
document.getElementById("changed-only").onclick = () => {
  state.changedOnly = !state.changedOnly;
  refresh();
};
document.getElementById("relayout").onclick = () => runLayout();
document.getElementById("reset-diff").onclick = async () => {
  /* Only the server can move the baseline -- a file:// page has no process behind it. The
     reload afterwards is what makes the colouring disappear: the page is rebuilt against
     the snapshot that was just written. */
  const button = document.getElementById("reset-diff");
  if (window.location.protocol !== "http:") {
    button.textContent = "needs --serve";
    return;
  }
  button.disabled = true;
  button.textContent = "resetting...";
  const response = await fetch("/reset-diff", {method: "POST"});
  if (!response.ok) {
    button.disabled = false;
    button.textContent = "reset failed";
    return;
  }
  window.location.reload();
};
document.getElementById("back").onclick = () => stepHistory(-1);
document.getElementById("forward").onclick = () => stepHistory(1);
document.addEventListener("keydown", (event) => {
  if (!event.altKey) return;
  if (event.key === "ArrowLeft") { stepHistory(-1); event.preventDefault(); }
  if (event.key === "ArrowRight") { stepHistory(1); event.preventDefault(); }
});
document.addEventListener("mouseup", (event) => {
  if (event.button === 3) { stepHistory(-1); event.preventDefault(); }
  if (event.button === 4) { stepHistory(1); event.preventDefault(); }
});

const datalist = document.getElementById("symbols");
const searchable = MODEL.nodes.filter(n => n.kind !== "repo");
for (const node of searchable.slice(0, 4000)) {
  const option = document.createElement("option");
  option.value = node.name;
  datalist.appendChild(option);
}
document.getElementById("search").addEventListener("keydown", (event) => {
  if (event.key !== "Enter") return;
  const needle = event.target.value.trim().toLowerCase();
  if (!needle) return;
  let best = null;
  for (const node of MODEL.nodes) {
    const name = node.name.toLowerCase();
    if (name === needle) { best = node; break; }
    if (!best && name.includes(needle)) best = node;
  }
  if (!best) return;
  expandTo(best.id);
  refresh();
  select(best.id, true);
});

refresh();
"""


def _read_vendor(vendor_dir: str) -> dict[str, str]:
    scripts: dict[str, str] = {}
    for name in VENDOR_SCRIPTS:
        path = os.path.join(vendor_dir, name)
        with open(path, "r", encoding="utf-8") as script_file:
            scripts[name] = script_file.read()
    return scripts


def render_page(model: dict[str, Any], status: dict[str, str], title: str, meta: str,
                vendor_dir: str, ghosts: Optional[list[dict[str, Any]]] = None) -> str:
    """The whole page as one string: model, library and layout engine inlined."""
    payload = dict(model)
    if ghosts:
        payload["nodes"] = model["nodes"] + ghosts
    payload["status"] = status
    scripts = _read_vendor(vendor_dir)
    model_json = json.dumps(payload, separators=(",", ":"))
    page = _PAGE_TEMPLATE
    page = page.replace("__TITLE__", title)
    page = page.replace("__META__", meta)
    page = page.replace("__CYTOSCAPE__", scripts["cytoscape.min.js"])
    page = page.replace("__DAGRE__", scripts["dagre.min.js"])
    page = page.replace("__CYTOSCAPE_DAGRE__", scripts["cytoscape-dagre.js"])
    page = page.replace("__MODEL__", model_json)
    page = page.replace("__APP__", _APP_JS)
    return page
