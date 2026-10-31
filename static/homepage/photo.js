// "Use a picture": the person picks or takes a photo, the CLIP model (running in
// the browser) compares it to the photo words, and they pick the right word.
// The photo never leaves their device.
import {
  AutoProcessor, CLIPVisionModelWithProjection, RawImage,
} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3";

const PHOTO_MODEL = "Xenova/clip-vit-base-patch32";   // must match CLIP_MODEL in export_photo_labels.py
const CHOICES = 5;                                     // how many guesses to show

const photoBtn = document.getElementById("photo-btn");
const photoInput = document.getElementById("photo-input");
const photoBox = document.getElementById("photo-box");
const preview = document.getElementById("photo-preview");
const photoStatus = document.getElementById("photo-status");
const photoChoices = document.getElementById("photo-choices");

// load the model and word data only the first time a photo is used
let loading = null;
function loadPhotoModel() {
  if (!loading) {
    loading = Promise.all([
      AutoProcessor.from_pretrained(PHOTO_MODEL),
      CLIPVisionModelWithProjection.from_pretrained(PHOTO_MODEL),
      fetch("data/photo_labels.json").then(r => r.json()),
      fetch("data/photo_meta.json").then(r => r.json()),
      fetch("data/photo_text.bin").then(r => r.arrayBuffer()),
    ]).then(([processor, model, labels, meta, buf]) =>
      ({ processor, model, labels, meta, text: new Int8Array(buf) }));
    loading.catch(() => { loading = null; });   // allow a retry after a failure
  }
  return loading;
}

function showChoices(words) {
  photoChoices.innerHTML = "";
  words.forEach(w => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "photo-choice";
    btn.textContent = w;
    btn.onclick = () => { location.href = "map.html?word=" + encodeURIComponent(w); };
    photoChoices.appendChild(btn);
  });
  const again = document.createElement("button");
  again.type = "button";
  again.className = "photo-again";
  again.textContent = "📷 try another picture";
  again.onclick = () => photoInput.click();
  photoChoices.appendChild(again);
}

async function guessWords(file) {
  const { processor, model, labels, meta, text } = await loadPhotoModel();
  const image = await RawImage.fromBlob(file);
  const inputs = await processor(image);
  const { image_embeds } = await model(inputs);
  const v = image_embeds.data;
  let norm = 0;
  for (let k = 0; k < v.length; k++) norm += v[k] * v[k];
  norm = Math.sqrt(norm) || 1;

  // compare the photo to every photo word
  const scores = labels.map((w, i) => {
    let s = 0;
    const off = i * meta.dim;
    for (let k = 0; k < meta.dim; k++) s += text[off + k] * v[k];
    return { w, s: s / (meta.scale * norm) };
  });
  scores.sort((a, b) => b.s - a.s);
  return scores.slice(0, CHOICES).map(x => x.w);
}

photoBtn.onclick = () => photoInput.click();

photoInput.onchange = async () => {
  const file = photoInput.files[0];
  photoInput.value = "";            // so choosing the same photo again still works
  if (!file) return;

  photoBox.hidden = false;
  photoChoices.innerHTML = "";
  preview.src = URL.createObjectURL(file);
  preview.onerror = () => {
    photoStatus.textContent = "This picture type can't be opened here. Please try a JPG or PNG photo.";
    showChoices([]);
  };
  photoStatus.textContent = loading
    ? "Looking at your picture…"
    : "Looking at your picture… (the first time can take a minute)";

  try {
    const words = await guessWords(file);
    photoStatus.textContent = "Is this a picture of:";
    showChoices(words);
  } catch (err) {
    console.error(err);
    photoStatus.textContent = "Sorry, I couldn't read that picture. Please try another one (JPG or PNG works best).";
    showChoices([]);
  }
};
