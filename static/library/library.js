// "My maps": maps saved to your account (signed in) or this browser (not signed in)
import { loadMaps, deleteMap } from "../firebase/firebase.js";

const libraryList = document.getElementById("library-list");
const emptyMessage = document.getElementById("library-empty");

async function showLibrary() {
  let library = [];
  try {
    library = await loadMaps();          // newest first
  } catch (err) {
    console.error(err);
    emptyMessage.textContent = "Couldn't load your maps. Please check your connection and try again.";
  }
  emptyMessage.hidden = library.length > 0;
  libraryList.innerHTML = "";

  library.forEach(map => {
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
    del.onclick = async () => {
      if (!confirm(`Delete your map for "${map.word}"?`)) return;
      await deleteMap(map.id);
      showLibrary();
    };
    card.appendChild(del);

    libraryList.appendChild(card);
  });
}

showLibrary();
