// "My maps": shows maps saved from the map page (kept in this browser only)
const LIBRARY_KEY = "mappy-library";
const librarySection = document.getElementById("library");
const libraryList = document.getElementById("library-list");

function loadLibrary() {
  try { return JSON.parse(localStorage.getItem(LIBRARY_KEY)) || []; } catch (err) { return []; }
}

function saveLibrary(list) {
  try { localStorage.setItem(LIBRARY_KEY, JSON.stringify(list)); } catch (err) {}
}

function showLibrary() {
  const library = loadLibrary();
  librarySection.hidden = library.length === 0;
  libraryList.innerHTML = "";

  // newest first
  library.slice().reverse().forEach(map => {
    const card = document.createElement("div");
    card.className = "map-card";

    const link = document.createElement("a");
    link.href = `map.html?word=${encodeURIComponent(map.word)}#map=${map.code}`;
    if (map.thumb) {
      const img = document.createElement("img");
      img.src = map.thumb;
      img.alt = `map for ${map.word}`;
      link.appendChild(img);
    }
    const title = document.createElement("div");
    title.className = "map-word";
    title.textContent = map.word;
    link.appendChild(title);
    card.appendChild(link);

    const date = document.createElement("div");
    date.className = "map-date";
    date.textContent = new Date(map.date).toLocaleDateString(undefined,
      { month: "short", day: "numeric", year: "numeric" });
    card.appendChild(date);

    const del = document.createElement("button");
    del.className = "map-delete";
    del.textContent = "Delete";
    del.onclick = () => {
      if (!confirm(`Delete your map for "${map.word}"?`)) return;
      saveLibrary(loadLibrary().filter(m => m.id !== map.id));
      showLibrary();
    };
    card.appendChild(del);

    libraryList.appendChild(card);
  });
}

showLibrary();
