// Browser-only version of Mappy: runs the model and the similarity search
// in the visitor's browser, so the site can be hosted on GitHub Pages.
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

const d3 = window.d3;

// ---------- settings (same meaning as in app.py) ----------
const MIN_SIM = 0.40;       // drop weak matches
const MAX_SIM = 0.90;       // drop near-duplicates
const LINK_SIM = 0.50;      // line between two generated words if at least this similar
const CLUSTER_DIST = 0.5;   // lower = smaller, tighter color groups
const NUM_WORDS = 16;
const SCALE = 500;          // pixels per 1.0 of distance
const DIM = 384;            // vector size for all-MiniLM-L6-v2

const statusEl = document.getElementById("status");
function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.hidden = !msg;
}

// ---------- model + vocabulary ----------
let extractor, WORDS, VECS;
const nodeVecs = new Map();   // word -> vector, for words currently on the map

async function init() {
  setStatus("Loading model and vocabulary… (the first visit can take a minute)");
  const [ext, words, buf] = await Promise.all([
    pipeline("feature-extraction", "Xenova/all-MiniLM-L6-v2"),
    fetch("data/words.json").then(r => r.json()),
    fetch("data/embeddings.bin").then(r => r.arrayBuffer()),
  ]);
  extractor = ext;
  WORDS = words;
  VECS = new Int8Array(buf);
}

async function embed(text) {
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return Float32Array.from(out.data);
}

function poolVec(i) {
  const v = new Float32Array(DIM);
  const off = i * DIM;
  for (let k = 0; k < DIM; k++) v[k] = VECS[off + k] / 127;
  return v;
}

function dot(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}

// cosine similarity between q and every vocabulary word
function scoreAll(q) {
  const n = WORDS.length;
  const scores = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    let acc = 0;
    const off = i * DIM;
    for (let k = 0; k < DIM; k++) acc += VECS[off + k] * q[k];
    scores[i] = acc / 127;
  }
  return scores;
}

// average-linkage clustering on cosine distance (same idea as sklearn's AgglomerativeClustering)
function cluster(vecs, threshold) {
  let clusters = vecs.map((_, i) => [i]);
  const dist = (a, b) => 1 - dot(vecs[a], vecs[b]);
  while (clusters.length > 1) {
    let best = Infinity, bi = -1, bj = -1;
    for (let i = 0; i < clusters.length; i++) {
      for (let j = i + 1; j < clusters.length; j++) {
        let total = 0;
        for (const a of clusters[i]) for (const b of clusters[j]) total += dist(a, b);
        const avg = total / (clusters[i].length * clusters[j].length);
        if (avg < best) { best = avg; bi = i; bj = j; }
      }
    }
    if (best >= threshold) break;
    clusters[bi] = clusters[bi].concat(clusters[bj]);
    clusters.splice(bj, 1);
  }
  const labels = new Array(vecs.length);
  clusters.forEach((members, c) => members.forEach(m => { labels[m] = c; }));
  return labels;
}

async function buildGraph(word) {
  const q = await embed(word);
  const scores = scoreAll(q);
  const order = Array.from(scores.keys()).sort((a, b) => scores[b] - scores[a]);

  const chosen = [];
  for (const i of order) {
    const w = WORDS[i], s = scores[i];
    if (w === word || s > MAX_SIM) continue;
    if (s < MIN_SIM) break;
    chosen.push(i);
    if (chosen.length === NUM_WORDS) break;
  }

  const words = [word, ...chosen.map(i => WORDS[i])];
  const vecs = [q, ...chosen.map(poolVec)];
  nodeVecs.clear();
  words.forEach((w, k) => nodeVecs.set(w, vecs[k]));

  // color groups: biggest cluster first, single words get -1 (gray)
  const labels = cluster(vecs.slice(1), CLUSTER_DIST);
  const sizes = {};
  labels.forEach(g => { sizes[g] = (sizes[g] || 0) + 1; });
  const multi = Object.keys(sizes).map(Number)
    .filter(g => sizes[g] > 1)
    .sort((a, b) => sizes[b] - sizes[a]);
  const remap = new Map(multi.map((g, n) => [g, n]));

  const nodes = [{ id: word, center: true, group: -1, sim: 1 }];
  chosen.forEach((i, k) => {
    nodes.push({
      id: WORDS[i],
      center: false,
      group: remap.has(labels[k]) ? remap.get(labels[k]) : -1,
      sim: dot(q, vecs[k + 1]),
    });
  });

  const links = [];
  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) {
      const s = dot(vecs[i], vecs[j]);
      if (i === 0 || s >= LINK_SIM) links.push({ source: words[i], target: words[j], sim: s });
    }
  }
  return { nodes, links };
}

// ---------- drawing ----------
const width = window.innerWidth;
const height = window.innerHeight;

const svg = d3.select("#wordcloud").attr("width", width).attr("height", height);
const linkLayer = svg.append("g");
const nodeLayer = svg.append("g");

let nodes = [];
let links = [];
let selected = null;

const families = [d3.interpolateGreens, d3.interpolateReds, d3.interpolateBlues,
                  d3.interpolatePurples, d3.interpolateOranges];
const shade = d3.scaleLinear().domain([MIN_SIM, MAX_SIM]).range([0.45, 0.95]).clamp(true);

function wordColor(d) {
  if (d.center) return "#000";
  if (d.group < 0) return d3.interpolateGreys(shade(d.sim));
  return families[d.group % families.length](shade(d.sim));
}

const simulation = d3.forceSimulation()
  .force("link", d3.forceLink()
    .id(d => d.id)
    .distance(d => (1 - d.sim) * SCALE)
    .strength(d => (d.source.center || d.target.center) ? 1 : 0.2))
  .force("collide", d3.forceCollide(35))
  .on("tick", ticked);

// drag moves a word and leaves it where you drop it
const drag = d3.drag()
  .on("start", (e) => { if (!e.active) simulation.alphaTarget(0.3).restart(); })
  .on("drag",  (e, d) => { d.fx = e.x; d.fy = e.y; hideMenu(); })
  .on("end",   (e) => { if (!e.active) simulation.alphaTarget(0); });

function linkKey(d) {
  const s = d.source.id || d.source;
  const t = d.target.id || d.target;
  return s + "|" + t;
}

function update(alpha = 0.3) {
  linkLayer.selectAll("line")
    .data(links, linkKey)
    .join("line")
    .attr("stroke", "#999")
    .attr("stroke-width", d => ((d.source.center || d.target.center) ? 1.5 : 1));

  nodeLayer.selectAll("text")
    .data(nodes, d => d.id)
    .join(enter => enter.append("text")
      .attr("text-anchor", "middle")
      .attr("dominant-baseline", "middle")
      .style("cursor", "pointer")
      .call(drag)
      .on("click", onWordClick))
    .text(d => d.id)
    .attr("fill", wordColor)
    .attr("font-size", d => d.center ? 32 : 18)
    .attr("font-weight", d => d.center ? "bold" : "normal");

  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(alpha).restart();
}

function ticked() {
  linkLayer.selectAll("line")
    .attr("x1", d => d.source.x).attr("y1", d => d.source.y)
    .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
  nodeLayer.selectAll("text")
    .attr("x", d => d.x).attr("y", d => d.y);
}

function loadGraph(graph) {
  nodes = graph.nodes.map(d => ({ ...d }));
  links = graph.links.map(d => ({ ...d }));
  const center = nodes.find(n => n.center);
  center.fx = width / 2;
  center.fy = height / 2;
  hideMenu();
  update(1);
}

async function showWord(word) {
  setStatus(`Finding words related to "${word}"…`);
  const graph = await buildGraph(word);
  loadGraph(graph);
  setStatus(graph.nodes.length > 1 ? "" : `No related words found for "${word}". Try lowering MIN_SIM.`);
  history.replaceState(null, "", "?word=" + encodeURIComponent(word));
  document.title = word + " · Mappy";
}

// ---------- menu ----------
const menu = document.getElementById("menu");
const menuTitle = document.getElementById("menu-title");
const deleteBtn = document.getElementById("delete-btn");
const releaseBtn = document.getElementById("release-btn");
const recenterBtn = document.getElementById("recenter-btn");
const addForm = document.getElementById("add-form");
const addInput = document.getElementById("add-input");

function onWordClick(event, d) {
  event.stopPropagation();
  selected = d;
  menuTitle.textContent = d.id;
  deleteBtn.disabled = d.center;
  recenterBtn.disabled = d.center;
  releaseBtn.disabled = d.center;
  addInput.value = "";
  menu.style.left = (event.pageX + 12) + "px";
  menu.style.top = (event.pageY + 12) + "px";
  menu.hidden = false;
}

function hideMenu() {
  menu.hidden = true;
  selected = null;
}

svg.on("click", hideMenu);

// 1. delete
deleteBtn.onclick = () => {
  if (!selected || selected.center) return;
  const gone = selected;
  nodes = nodes.filter(n => n !== gone);
  links = links.filter(l => l.source !== gone && l.target !== gone);
  nodeVecs.delete(gone.id);
  hideMenu();
  update();
};

// 2. add a connected word
addForm.onsubmit = async (e) => {
  e.preventDefault();
  const word = addInput.value.trim().toLowerCase();
  if (!word || !selected) return;
  if (nodes.some(n => n.id === word)) {
    alert("That word is already on the map.");
    return;
  }
  const anchor = selected;
  hideMenu();

  const v = await embed(word);
  const center = nodes.find(n => n.center);
  const simTo = n => dot(v, nodeVecs.get(n.id));

  const newNode = {
    id: word,
    center: false,
    group: anchor.center ? -1 : anchor.group,   // takes the clicked word's color family
    sim: simTo(center),
    x: anchor.x + 30,
    y: anchor.y + 30,
  };

  const newLinks = [{ source: anchor, target: newNode, sim: simTo(anchor) }];
  nodes.forEach(n => {
    if (n !== anchor) {
      const s = simTo(n);
      if (s >= LINK_SIM) newLinks.push({ source: n, target: newNode, sim: s });
    }
  });

  nodeVecs.set(word, v);
  nodes.push(newNode);
  links.push(...newLinks);
  update();
};

// 3. let a dragged word float again
releaseBtn.onclick = () => {
  if (!selected || selected.center) return;
  selected.fx = null;
  selected.fy = null;
  hideMenu();
  simulation.alpha(0.3).restart();
};

// 4. make the clicked word the new center
recenterBtn.onclick = async () => {
  if (!selected) return;
  const word = selected.id;
  hideMenu();
  await showWord(word);
};

// ---------- start ----------
const startWord = (new URLSearchParams(location.search).get("word") || "").trim().toLowerCase();

if (!startWord) {
  location.href = "homepage.html";
} else {
  try {
    await init();
    await showWord(startWord);
  } catch (err) {
    console.error(err);
    setStatus("Something went wrong loading the map. Open the browser console for details.");
  }
}
