// Mappy (combined version): scores words with two models in the visitor's
// browser, so the site can be hosted on GitHub Pages.
//   GloVe  -> everyday associations (leash, kennel, hungry)
//   MiniLM -> "kinds of" words (poodle, terrier) and phrases
import { pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";
import { saveMap } from "../firebase/firebase.js";

const d3 = window.d3;

// ---------- settings ----------
const NUM_WORDS = 16;
const GLOVE_WEIGHT = 0.5;     // how much GloVe counts (0–1); MiniLM gets the rest
                              // higher = more associations, lower = more "kinds of"
const MIN_SIM = 0.30;         // a word must be at least this similar to appear
const MAX_SIM = 0.99;         // only blocks exact duplicates (plurals are already removed)
const LINK_SIM = 1.1;         // above 1 = lines only go to the center word
const CLUSTER_DIST = 0.55;    // lower = smaller, tighter color groups
const HUB_WEIGHT = 1.0;       // higher = generic words ("thing", "know") pushed out more
const ANTONYM_MARGIN = 0.1;   // drop words that are this much closer to an opposite than to the input
const SPREAD = 0.7;           // raise to spread words out, lower to pull them in
const CENTER_FONT = 36;
const WORD_FONT = 24;
const MINILM_MODEL = "Xenova/all-MiniLM-L6-v2";   // must match MINILM_MODEL in export_vocab.py
const STEP = 2;          // how many words each button adds or removes
const MAX_WORDS = 30;

const statusEl = document.getElementById("status");
function setStatus(msg) {
  statusEl.textContent = msg;
  statusEl.hidden = !msg;
}

// ---------- word data ----------
let WORDS, INDEX, ANTONYMS, POOL, HUB, HUBM;
let VECS, DIM, QSCALE;        // GloVe, every typeable word
let MVECS, MDIM, MSCALE;      // MiniLM, map words only
let extractor;
const nodeVecs = new Map();   // word -> { g, m } for words currently on the map
let numWords = NUM_WORDS;   // changes when "see more" / "see less" is clicked
let currentWord = null;
let currentForce = false;

async function init() {
  setStatus("Loading words… (the first visit can take a minute)");
  const [ext, meta, words, buf, pool, mbuf, antonyms] = await Promise.all([
    pipeline("feature-extraction", MINILM_MODEL),
    fetch("data/meta.json").then(r => r.json()),
    fetch("data/words.json").then(r => r.json()),
    fetch("data/vectors.bin").then(r => r.arrayBuffer()),
    fetch("data/pool.json").then(r => r.json()),
    fetch("data/pool_minilm.bin").then(r => r.arrayBuffer()),
    fetch("data/antonyms.json").then(r => r.json()),
  ]);
  extractor = ext;
  DIM = meta.dim;   QSCALE = meta.scale;  VECS = new Int8Array(buf);
  MDIM = meta.mdim; MSCALE = meta.mscale; MVECS = new Int8Array(mbuf);
  WORDS = words;
  INDEX = new Map(words.map((w, i) => [w, i]));
  POOL = pool.index;
  HUB = pool.hub;
  HUBM = pool.hubm;
  ANTONYMS = antonyms;
}

function unpack(arr, i, dim, scale) {
  const v = new Float32Array(dim);
  const off = i * dim;
  for (let k = 0; k < dim; k++) v[k] = arr[off + k] / scale;
  return v;
}
const gloveOf = i => unpack(VECS, i, DIM, QSCALE);       // i = word number
const minilmOf = p => unpack(MVECS, p, MDIM, MSCALE);    // p = position in POOL

function dot(a, b) {
  let s = 0;
  for (let k = 0; k < a.length; k++) s += a[k] * b[k];
  return s;
}

function normalize(v) {
  const n = Math.sqrt(dot(v, v)) || 1;
  return v.map(x => x / n);
}

// combined similarity of two items ({ g, m }); uses MiniLM alone if GloVe doesn't know a word
function simBetween(a, b) {
  const m = dot(a.m, b.m);
  if (!a.g || !b.g) return m;
  return GLOVE_WEIGHT * dot(a.g, b.g) + (1 - GLOVE_WEIGHT) * m;
}

// Vectors for a word or short phrase.
// g = GloVe (phrases average their words), or null if GloVe doesn't know a word
// m = MiniLM (always works, even for typos and rare words)
async function vectorsFor(text) {
  const tokens = text.toLowerCase().split(/[^a-z]+/).filter(Boolean);
  const unknown = tokens.filter(t => !INDEX.has(t));
  let g = null;
  if (tokens.length && !unknown.length) {
    const sum = new Float32Array(DIM);
    tokens.forEach(t => gloveOf(INDEX.get(t)).forEach((x, k) => { sum[k] += x; }));
    g = normalize(sum);
  }
  const out = await extractor(text, { pooling: "mean", normalize: true });
  return { g, m: Float32Array.from(out.data), tokens, unknown };
}

// ---------- "did you mean" ----------
// edit distance that counts swapped letters ("dgo" -> "dog") as one mistake
function editDistance(a, b, max) {
  if (Math.abs(a.length - b.length) > max) return max + 1;
  const d = [];
  for (let i = 0; i <= a.length; i++) d.push([i]);
  for (let j = 1; j <= b.length; j++) d[0][j] = j;
  for (let i = 1; i <= a.length; i++) {
    let rowMin = Infinity;
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      d[i][j] = Math.min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost);
      if (i > 1 && j > 1 && a[i - 1] === b[j - 2] && a[i - 2] === b[j - 1]) {
        d[i][j] = Math.min(d[i][j], d[i - 2][j - 2] + 1);
      }
      rowMin = Math.min(rowMin, d[i][j]);
    }
    if (rowMin > max) return max + 1;
  }
  return d[a.length][b.length];
}

const letters = w => w.split("").sort().join("");

function suggestionsFor(word, count = 5) {
  const max = word.length <= 4 ? 1 : 2;
  const wordLetters = letters(word);
  const found = [];
  for (let i = 0; i < WORDS.length; i++) {      // WORDS is ordered most-common first
    const dist = editDistance(word, WORDS[i], max);
    if (dist <= max) found.push({ w: WORDS[i], dist, len: Math.abs(WORDS[i].length - word.length),
                                swap: letters(WORDS[i]) === wordLetters ? 0 : 1, i });
  }
  // fewest mistakes first, then swapped letters ("dgo" -> "dog"), then same length, then most common
  found.sort((a, b) => a.dist - b.dist || a.swap - b.swap || a.len - b.len || a.i - b.i);
  return found.slice(0, count).map(f => f.w);
}

const suggestBox = document.getElementById("suggest");
function showSuggestions(badWord, onPick, onAnyway) {
  const options = suggestionsFor(badWord);
  suggestBox.innerHTML = "";
  const msg = document.createElement("p");
  msg.textContent = options.length
    ? `We don't know "${badWord}". Did you mean:`
    : `We don't know "${badWord}". Try another word.`;
  suggestBox.appendChild(msg);
  options.forEach(w => {
    const btn = document.createElement("button");
    btn.textContent = w;
    btn.onclick = () => { suggestBox.hidden = true; onPick(w); };
    suggestBox.appendChild(btn);
  });
  if (onAnyway) {
    const anyway = document.createElement("button");
    anyway.textContent = `use "${badWord}" anyway`;
    anyway.onclick = () => { suggestBox.hidden = true; onAnyway(); };
    suggestBox.appendChild(anyway);
  }
  const close = document.createElement("button");
  close.textContent = "✕";
  close.className = "close";
  close.onclick = () => { suggestBox.hidden = true; };
  suggestBox.appendChild(close);
  suggestBox.hidden = false;
}

// ---------- finding related words ----------
// true if a is a plural of b (dogs/dog, boxes/box, puppies/puppy)
function isPlural(a, b) {
  return a === b + "s" || a === b + "es" ||
         (b.endsWith("y") && a === b.slice(0, -1) + "ies");
}
function sameWord(a, b) {
  return a === b || isPlural(a, b) || isPlural(b, a);
}

// average-linkage clustering on combined similarity
function cluster(items, threshold) {
  let clusters = items.map((_, i) => [i]);
  const dist = (a, b) => 1 - simBetween(items[a], items[b]);
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
  const labels = new Array(items.length);
  clusters.forEach((members, c) => members.forEach(m => { labels[m] = c; }));
  return labels;
}

function buildGraph(word, item) {
  const { g, m, tokens } = item;
  const W = g ? GLOVE_WEIGHT : 0;   // no GloVe vector -> MiniLM only

  // opposites of the input (and of each word in a phrase)
  const opposites = new Set(tokens.flatMap(t => ANTONYMS[t] || []));
  const oppVecs = g ? [...opposites].filter(a => INDEX.has(a)).map(a => gloveOf(INDEX.get(a))) : [];

  // rank every map word: combined similarity, minus a penalty for generic words
  const cand = POOL.map((wi, p) => {
    let accM = 0;
    const offM = p * MDIM;
    for (let k = 0; k < MDIM; k++) accM += MVECS[offM + k] * m[k];
    const simM = accM / MSCALE;
    let simG = 0;
    if (g) {
      let accG = 0;
      const offG = wi * DIM;
      for (let k = 0; k < DIM; k++) accG += VECS[offG + k] * g[k];
      simG = accG / QSCALE;
    }
    const sim = W * simG + (1 - W) * simM;
    const hub = W * HUB[p] + (1 - W) * HUBM[p];
    return { wi, p, sim, simG, rank: 2 * sim - HUB_WEIGHT * hub };
  });
  cand.sort((a, b) => b.rank - a.rank);

  const chosen = [];
  for (const c of cand) {
    const w = WORDS[c.wi];
    if (c.sim < MIN_SIM || c.sim > MAX_SIM) continue;
    if (tokens.some(t => sameWord(w, t)) || sameWord(w, word)) continue;
    if (chosen.some(x => sameWord(WORDS[x.wi], w))) continue;
    if (opposites.has(w)) continue;
    if (oppVecs.length) {
      const v = gloveOf(c.wi);
      if (oppVecs.some(a => dot(v, a) > c.simG + ANTONYM_MARGIN)) continue;
    }
    chosen.push(c);
      if (chosen.length === numWords) break;
  }

  const words = [word, ...chosen.map(c => WORDS[c.wi])];
  const items = [{ g, m }, ...chosen.map(c => ({ g: gloveOf(c.wi), m: minilmOf(c.p) }))];
  nodeVecs.clear();
  words.forEach((w, k) => nodeVecs.set(w, items[k]));

  // color groups: biggest cluster first, single words get -1 (gray)
  const labels = cluster(items.slice(1), CLUSTER_DIST);
  const sizes = {};
  labels.forEach(gr => { sizes[gr] = (sizes[gr] || 0) + 1; });
  const multi = Object.keys(sizes).map(Number)
    .filter(gr => sizes[gr] > 1)
    .sort((a, b) => sizes[b] - sizes[a]);
  const remap = new Map(multi.map((gr, n) => [gr, n]));

  const nodes = [{ id: word, center: true, group: -1, sim: 1 }];
  chosen.forEach((c, k) => {
    nodes.push({
      id: WORDS[c.wi],
      center: false,
      group: remap.has(labels[k]) ? remap.get(labels[k]) : -1,
      sim: c.sim,
    });
  });

  const links = [];
  for (let i = 0; i < words.length; i++) {
    for (let j = i + 1; j < words.length; j++) {
      const sim = i === 0 ? nodes[j].sim : simBetween(items[i], items[j]);
      if (i === 0 || sim >= LINK_SIM) links.push({ source: words[i], target: words[j], sim });
    }
  }
  return { nodes, links };
}

// ---------- drawing ----------
const width = window.innerWidth;
const height = window.innerHeight;
const SCALE = Math.min(width, height) * SPREAD;   // pixels per 1.0 of distance (fits the screen)

const svg = d3.select("#wordcloud").attr("width", width).attr("height", height)
  .style("font-family", "sans-serif");
const linkLayer = svg.append("g");
const nodeLayer = svg.append("g");

let nodes = [];
let links = [];
let selected = null;

const families = [d3.interpolateGreens, d3.interpolateReds, d3.interpolateBlues,
                  d3.interpolatePurples, d3.interpolateOranges];
const shade = d3.scaleLinear().domain([0.3, 0.75]).range([0.5, 0.95]).clamp(true);

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
  .force("collide", d3.forceCollide(75))
  .force("charge", d3.forceManyBody().strength(-60))   // spreads words evenly around the center
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
    .attr("font-size", d => d.center ? CENTER_FONT : WORD_FONT)
    .attr("font-weight", d => d.center ? "bold" : "normal");

  simulation.nodes(nodes);
  simulation.force("link").links(links);
  simulation.alpha(alpha).restart();
}

function ticked() {
  // keep every word on screen
  nodes.forEach(d => {
    d.x = Math.max(60, Math.min(width - 60, d.x));
    d.y = Math.max(30, Math.min(height - 30, d.y));
  });
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
  center.fx = center.x = width / 2;
  center.fy = center.y = height / 2;
  // start words in a circle around the center, at their target distance
  // (sorted by color group, so related words sit next to each other)
  const others = nodes.filter(n => !n.center).sort((a, b) => a.group - b.group);
  others.forEach((n, k) => {
    const angle = (2 * Math.PI * k) / others.length;
    const r = (1 - n.sim) * SCALE;
    n.x = center.x + r * Math.cos(angle);
    n.y = center.y + r * Math.sin(angle);
  });
  hideMenu();
  update(1);
}

async function showWord(text, force = false) {
  const word = text.trim().toLowerCase();
  setStatus(`Finding words related to "${word}"…`);
  const item = await vectorsFor(word);
  if (item.unknown.length && !force) {
    // GloVe doesn't know it: offer spellings, or search with MiniLM alone
    setStatus("");
    const bad = item.unknown[0];
    showSuggestions(bad,
      picked => showWord(word.replace(bad, picked)),
      () => showWord(word, true));
    return;
  }
  suggestBox.hidden = true;
  currentWord = word;
  currentForce = force;
  const graph = buildGraph(word, item);
  loadGraph(graph);
  setStatus(graph.nodes.length > 1 ? "" : `No related words found for "${word}".`);
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
  clearTimeout(hideTimer);
  guide.classList.remove("visible"); 
  guideWord = null;
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
async function addWord(word, anchor, force = false) {
  if (nodes.some(n => n.id === word)) {
    alert("That word is already on the map.");
    return;
  }
  if (!nodes.includes(anchor)) return;   // anchor was deleted meanwhile
  const item = await vectorsFor(word);
  if (item.unknown.length && !force) {
    const bad = item.unknown[0];
    showSuggestions(bad,
      picked => addWord(word.replace(bad, picked), anchor),
      () => addWord(word, anchor, true));
    return;
  }
  const v = { g: item.g, m: item.m };
  const center = nodes.find(n => n.center);
  const simTo = n => simBetween(v, nodeVecs.get(n.id));

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
}

addForm.onsubmit = (e) => {
  e.preventDefault();
  const word = addInput.value.trim().toLowerCase();
  if (!word || !selected) return;
  const anchor = selected;
  hideMenu();
  addWord(word, anchor);
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
// (asks first, offers to save, and remembers the old map so it can be reopened)
recenterBtn.onclick = () => {
  if (!selected) return;
  pendingWord = selected.id;
  hideMenu();
  confirmText.textContent =
    `This will make a new map for "${pendingWord}". ` +
    `Your map for "${currentWord}" will be replaced, but you can come back to it ` +
    `with the "previous map" button. Do you want to save it to My maps first?`;
  confirmBox.hidden = false;
};

// ---------- warning, saving, and going back ----------
const confirmBox = document.getElementById("confirm-box");
const confirmText = document.getElementById("confirm-text");
const prevBtn = document.getElementById("prev-btn");
const mapHistory = [];      // earlier maps, newest last
let pendingWord = null;

// everything needed to bring a map back exactly as it was, including edits
function snapshot() {
  return {
    word: currentWord,
    force: currentForce,
    count: numWords,
    nodes: nodes.map(n => ({ id: n.id, center: n.center, group: n.group, sim: n.sim,
                             x: n.x, y: n.y, fx: n.fx, fy: n.fy })),
    links: links.map(l => ({ source: l.source.id || l.source, target: l.target.id || l.target, sim: l.sim })),
    vecs: new Map(nodeVecs),
  };
}

function restore(saved) {
  currentWord = saved.word;
  currentForce = saved.force;
  numWords = saved.count;
  nodeVecs.clear();
  saved.vecs.forEach((v, w) => nodeVecs.set(w, v));
  nodes = saved.nodes.map(n => ({ ...n }));
  links = saved.links.map(l => ({ ...l }));
  hideMenu();
  update(0.05);             // tiny nudge, so words stay where they were
  history.replaceState(null, "", "?word=" + encodeURIComponent(currentWord));
  document.title = currentWord + " · Mappy";
  prevBtn.hidden = mapHistory.length === 0;
}

// draw the current map onto a picture; returns it as a data URL (or null)
// scale 2 = sharp full-size picture, smaller = thumbnail
function mapPicture(scale = 2, type = "image/png", quality) {
  return new Promise(resolve => {
    const clone = svg.node().cloneNode(true);
    clone.setAttribute("xmlns", "http://www.w3.org/2000/svg");
    let bg = getComputedStyle(document.body).backgroundColor;
    if (!bg || bg === "rgba(0, 0, 0, 0)" || bg === "transparent") bg = "#ffffff";
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("width", "100%");
    rect.setAttribute("height", "100%");
    rect.setAttribute("fill", bg);
    clone.insertBefore(rect, clone.firstChild);

    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.drawImage(img, 0, 0);
      resolve(canvas.toDataURL(type, quality));
    };
    img.onerror = () => resolve(null);
    img.src = "data:image/svg+xml;charset=utf-8," +
              encodeURIComponent(new XMLSerializer().serializeToString(clone));
  });
}

// download the current map as a PNG picture
async function saveMapImage(name) {
  const picture = await mapPicture(2);
  if (!picture) return;
  const a = document.createElement("a");
  a.download = `mappy-${name}.png`;
  a.href = picture;
  a.click();
}

// save the current map to "My maps" on the homepage (kept in this browser)
// save the current map to "My maps" (your account if signed in, otherwise this browser)
async function saveToLibrary() {
  const map = {
    word: currentWord,
    date: new Date().toISOString(),
    code: encodeMap(),                                        // the map itself (same as share links)
    thumb: await mapPicture(320 / width, "image/jpeg", 0.7),  // small preview picture
  };
  try {
    const where = await saveMap(map);
    showToast(where === "account"
      ? `Saved "${currentWord}" to My maps.`
      : `Saved "${currentWord}" to My maps on this device. Sign in to keep maps on every device.`);
  } catch (err) {
    console.error(err);
    showToast("Couldn't save this map. Please try again.");
  }
}

async function goToNewCenter(save) {
  confirmBox.hidden = true;
  if (save) await saveToLibrary();
  mapHistory.push(snapshot());
  prevBtn.hidden = false;
  showWord(pendingWord);
}

document.getElementById("confirm-save").onclick = () => goToNewCenter(true);
document.getElementById("confirm-go").onclick = () => goToNewCenter(false);
document.getElementById("confirm-cancel").onclick = () => { confirmBox.hidden = true; };

prevBtn.onclick = () => {
  const saved = mapHistory.pop();
  if (saved) restore(saved);
};

// hover
const guide = document.createElement("div");
guide.id = "guide";
guide.innerHTML = `
  <img id="picture" alt="" hidden referrerpolicy="no-referrer">
  <div id="word"></div>
  <div id="def"></div>
  <div id="def-buttons">
    <button id="normal">say it normal</button>
    <button id="slowed">say it slowed</button>
  </div>`;
document.body.appendChild(guide);

const guideImg = document.getElementById("picture");
const guideWordElement = document.getElementById("word");
const guideDef = document.getElementById("def");
let guideWord = null;
let hideTimer = null;
let wikiphotos = {}
let senses = {}
fetch("data/photos.json").then(response => response.json()).then(data => {wikiphotos = data; }).catch(() => {});
fetch("data/senses.json").then(response => response.json()).then(data => {senses = data; }).catch(() => {});

const SKIP = new Set("a an the of or and in to with by for as on at from that which is are be any one two some its it into used".split(" "));

// avg glove vector of words
function textVector(text) {
  const sum = new Float32Array(DIM);  // assuming 300-dimensional glove vectors
  let n = 0;
  for (const w of text.toLowerCase().split(/[^a-z]/)) {
    if(!w || SKIP.has(w) || !INDEX.has(w)) continue;
    gloveOf(INDEX.get(w)).forEach((x, k) => { sum[k] += x; });
    n++;
  }
  return n ? normalize(sum) : null;
}

// find meaning of word that best fits other words + definitions that are more frequently used
function senseFor(word){
  const options = senses[word];
  if(!options || !options.length) return null;
  if(options.length === 1) return options[0];
  const context = new Float32Array(DIM);
  let n = 0;
  nodeVecs.forEach((item, w) => {
    if (w !== word && item.g) { item.g.forEach((x, k) => { context[k] += x; }); n++; }
  });
  if (!n) return options[0];
  const c = normalize(context);
  let best = options[0], bestScore = -Infinity;
  for (const s of options) {
    const v = textVector(s.d + " " + s.h);
    const score = (v ? dot(v, c) : 0) + 0.1 * s.c;
    if (score > bestScore) {bestScore = score; best = s;}
  }
  return best;
}

// 1. pictures
const searchCache = new Map();
async function searchArasaac(query) {
  if (searchCache.has(query)) return searchCache.get(query);
  let results = [];
  try {
    const response = await fetch(`https://api.arasaac.org/v1/pictograms/en/search/` + encodeURIComponent(query));
    if (response.ok) results = await response.json();
  } catch (error) {}
  searchCache.set(query, results);
  return results;
}

const pictureDoc = new Map();
async function pictureFor(word) {
  const sense = senseFor(word);
  if (!sense) return null;
  const key = sense.p[0][1];
  if (pictureDoc.has(key)) return pictureDoc.get(key);
  let url = null;
  for (const [query, code] of sense.p) {
    const results = await searchArasaac(query);
    const match = results.find(p => (p.synsets || []).includes(code) ||
                                    (p.synsets || []).includes(code.replace(/-a$/, "-s")));
    if (match) {
      url = `https://api.arasaac.org/v1/pictograms/${match._id}?download=false`;
      break;
    }
    if (wikiphotos[code]) {
      url = `https://commons.wikimedia.org/wiki/Special:FilePath/${encodeURIComponent(wikiphotos[code])}?width=300`;
      break;
    }
  }
  pictureDoc.set(key, url);
  return url; 
}

// 2. pronunciation
function say(rate) {
  if (!guideWord) return;
  const utterance = new SpeechSynthesisUtterance(guideWord);
  utterance.lang = "en-US";
  utterance.rate = rate;
  if (speechSynthesis.speaking) speechSynthesis.cancel();
  setTimeout(() => speechSynthesis.speak(utterance), 100);
}
document.getElementById("normal").onclick = () => say(1);
document.getElementById("slowed").onclick = () => say(0.5);

function hideGuide () {
  clearTimeout(hideTimer);
  hideTimer = setTimeout(() => {
    guide.classList.remove("visible"); guideWord = null;
  }, 300);
}

svg.on("mouseover", async (event) => {
  if (event.target.tagName !== "text") return;
  clearTimeout(hideTimer);
  const d = event.target.__data__;
  if (guideWord == d.id) return;
  guideWord = d.id;
  guideWordElement.textContent = d.id;
  guideDef.textContent = senseFor(d.id)?.d || "";
  guideImg.hidden = true;
  guide.style.left = (event.pageX + 14) + "px";
  guide.style.top = (event.pageY + 14) + "px";
  guide.classList.add("visible");
  const url = await pictureFor(d.id);
  if (guideWord == d.id && url) {
    guideImg.hidden = false;
    guideImg.src = url;
  }
});
svg.on("mouseout", (event) => {
  if (event.target.tagName === "text") hideGuide();
});
guide.addEventListener("mouseleave", hideGuide);
guide.addEventListener("mouseenter", () => clearTimeout(hideTimer));
svg.on("mousedown", () => { guide.classList.remove("visible"); guideWord = null; });

// see less / see more
const lessBtn = document.getElementById("less-btn");
const moreBtn = document.getElementById("more-btn");

function changeCount(delta) {
  if (!currentWord) return;
  numWords = Math.max(STEP, Math.min(MAX_WORDS, numWords + delta));
  lessBtn.disabled = numWords <= STEP;
  moreBtn.disabled = numWords >= MAX_WORDS;
  showWord(currentWord, currentForce);
}

lessBtn.onclick = () => changeCount(-STEP);
moreBtn.onclick = () => changeCount(STEP);

// ---------- save and share ----------
document.getElementById("save-btn").onclick = () => saveMapImage(currentWord);
document.getElementById("share-btn").onclick = shareMap;

const toast = document.getElementById("toast");
function showToast(msg) {
  toast.textContent = msg;
  toast.hidden = false;
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => { toast.hidden = true; }, 2500);
}

// the whole map (words, colors, positions, lines) packed into a short code for the link
function encodeMap() {
  const index = new Map(nodes.map((n, i) => [n.id, i]));
  const r = x => Math.round(x * 1000) / 1000;
  const state = {
    w: currentWord, f: currentForce ? 1 : 0, n: numWords,
    // [word, color group, similarity, x and y as a share of the screen, pinned?]
    nodes: nodes.map(n => [n.id, n.group, r(n.sim), r(n.x / width), r(n.y / height), n.fx != null && !n.center ? 1 : 0]),
    links: links.map(l => [index.get(l.source.id || l.source), index.get(l.target.id || l.target), r(l.sim)]),
  };
  const json = JSON.stringify(state);
  return btoa(unescape(encodeURIComponent(json))).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function decodeMap(code) {
  try {
    const b64 = code.replace(/-/g, "+").replace(/_/g, "/");
    return JSON.parse(decodeURIComponent(escape(atob(b64))));
  } catch (err) {
    return null;
  }
}

async function shareMap() {
  const url = location.origin + location.pathname +
              "?word=" + encodeURIComponent(currentWord) + "#map=" + encodeMap();
  if (navigator.share) {
    try {
      await navigator.share({ title: `Mappy: ${currentWord}`, text: `My word map for "${currentWord}"`, url });
      return;
    } catch (err) {
      if (err.name === "AbortError") return;     // they closed the share window
    }
  }
  try {
    await navigator.clipboard.writeText(url);
    showToast("Link copied! Paste it anywhere to share this map.");
  } catch (err) {
    prompt("Copy this link to share your map:", url);
  }
}

// open a map that someone shared (rebuilds it exactly, including their edits)
async function loadSharedMap(state) {
  setStatus("Opening shared map…");
  currentWord = state.w;
  currentForce = !!state.f;
  numWords = state.n || NUM_WORDS;
  nodeVecs.clear();
  for (const [id] of state.nodes) {
    const v = await vectorsFor(id);
    nodeVecs.set(id, { g: v.g, m: v.m });
  }
  nodes = state.nodes.map(([id, group, sim, x, y, pinned]) => {
    const n = { id, group, sim, center: id === state.w, x: x * width, y: y * height };
    if (n.center) { n.fx = n.x = width / 2; n.fy = n.y = height / 2; }
    else if (pinned) { n.fx = n.x; n.fy = n.y; }
    return n;
  });
  links = state.links.map(([i, j, sim]) => ({ source: state.nodes[i][0], target: state.nodes[j][0], sim }));
  hideMenu();
  update(0.05);
  setStatus("");
  document.title = currentWord + " · Mappy";
}

// ---------- start ----------
const startWord = (new URLSearchParams(location.search).get("word") || "").trim().toLowerCase();
const sharedMap = location.hash.startsWith("#map=") ? decodeMap(location.hash.slice(5)) : null;

if (!startWord && !sharedMap) {
  location.href = "homepage.html";
} else {
  init()
    .then(() => sharedMap ? loadSharedMap(sharedMap) : showWord(startWord))
    .catch(err => {
      console.error(err);
      setStatus("Something went wrong loading the map. Open the browser console for details.");
    });
}
