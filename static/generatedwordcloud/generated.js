const width = window.innerWidth;
const height = window.innerHeight;
const SCALE = 500;      // distance can change
const LINK_SIM = 0.5;   // match LINK_SIM in app.py

const svg = d3.select("#wordcloud").attr("width", width).attr("height", height);
const linkLayer = svg.append("g");
const nodeLayer = svg.append("g");

let nodes = [];
let links = [];
let selected = null;

// ---------- colors ----------
const families = [d3.interpolateGreens, d3.interpolateReds, d3.interpolateBlues,
                  d3.interpolatePurples, d3.interpolateOranges];
const shade = d3.scaleLinear().domain([0.4, 0.9]).range([0.45, 0.95]).clamp(true);

function wordColor(d) {
  if (d.center) return "#000";
  if (d.group < 0) return d3.interpolateGreys(shade(d.sim));
  return families[d.group % families.length](shade(d.sim));
}

// ---------- simulation ----------
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

// ---------- menu ----------
const menu = document.getElementById("menu");
const menuTitle = document.getElementById("menuTitle");
const deleteBtn = document.getElementById("deleteButton");
const releaseBtn = document.getElementById("moveButton");
const recenterBtn = document.getElementById("newmapButton");
const addForm = document.getElementById("add-form");
const addInput = document.getElementById("addInput");

function onWordClick(event, d) {
  event.stopPropagation();
  selected = d;
  menuTitle.textContent = d.id;
  deleteBtn.disabled = d.center;     // can't delete the center word
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

  const res = await fetch("/api/similarity", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ word, others: nodes.map(n => n.id) })
  });
  const sims = await res.json();
  const center = nodes.find(n => n.center);

  const newNode = {
    id: word,
    center: false,
    group: anchor.center ? -1 : anchor.group,   // takes the clicked word's color family
    sim: sims[center.id],
    x: anchor.x + 30,
    y: anchor.y + 30
  };
  nodes.push(newNode);

  links.push({ source: anchor, target: newNode, sim: sims[anchor.id] });
  nodes.forEach(n => {
    if (n !== anchor && n !== newNode && sims[n.id] >= LINK_SIM) {
      links.push({ source: n, target: newNode, sim: sims[n.id] });
    }
  });

  hideMenu();
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
  const res = await fetch("/api/graph", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ word })
  });
  loadGraph(await res.json());
};

// ---------- start ----------
loadGraph(JSON.parse(document.getElementById("graph-data").textContent));