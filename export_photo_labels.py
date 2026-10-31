"""
Prepares the words the "use a picture" button can recognize.
Run it once (with .venv active), after export_vocab.py:

    python export_photo_labels.py

It picks the map words that are things you can take a picture of
(dog, cup, car, tree...) and saves what the CLIP picture model
"thinks" each one looks like, so the website can compare a photo to them.

Creates:
  data/photo_labels.json  - the words a photo can be matched to
  data/photo_text.bin     - CLIP's description of each word (8-bit)
  data/photo_meta.json    - size and scale of those descriptions

The first run downloads the CLIP model (~600 MB, one time only).
"""
import json
import os

import numpy as np
import nltk
import torch
from transformers import CLIPModel, CLIPTokenizer

nltk.download('wordnet', quiet=True)
from nltk.corpus import wordnet as wn

CLIP_MODEL = "openai/clip-vit-base-patch32"   # must match PHOTO_MODEL in photo.js
PROMPT = "a photo of a {}"                     # CLIP works best with a short sentence

words = json.load(open('data/words.json'))
pool = json.load(open('data/pool.json'))['index']

PHYSICAL = wn.synset('physical_entity.n.01')


def count(s, w):
    return sum(l.count() for l in s.lemmas() if l.name().lower() == w)


def photographable(w):
    """True if w is mainly used as a noun for a physical thing (dog, cup), not an idea."""
    syns = wn.synsets(w)
    nouns = [s for s in syns if s.pos() == 'n']
    if not nouns:
        return False
    noun_uses = sum(count(s, w) for s in nouns)
    other_uses = sum(count(s, w) for s in syns if s.pos() != 'n')
    if noun_uses < other_uses or (noun_uses == 0 and syns[0].pos() != 'n'):
        return False
    best = max(nouns, key=lambda s: count(s, w))
    return any(PHYSICAL in path for path in best.hypernym_paths())


labels = [words[i] for i in pool if photographable(words[i])]
print(f"{len(labels)} words can be matched to photos.")

print(f"Loading {CLIP_MODEL} (downloads ~600 MB the first time)...")
model = CLIPModel.from_pretrained(CLIP_MODEL)
tokenizer = CLIPTokenizer.from_pretrained(CLIP_MODEL)

vecs = []
with torch.no_grad():
    for start in range(0, len(labels), 256):
        batch = [PROMPT.format(w) for w in labels[start:start + 256]]
        tokens = tokenizer(batch, padding=True, return_tensors="pt")
        vecs.append(model.get_text_features(**tokens).numpy())
        print(f"  {min(start + 256, len(labels))}/{len(labels)}")
vecs = np.concatenate(vecs).astype(np.float32)
vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)

scale = 127.0 / float(np.abs(vecs).max())
quantized = np.clip(np.round(vecs * scale), -127, 127).astype(np.int8)

os.makedirs('data', exist_ok=True)
json.dump(labels, open('data/photo_labels.json', 'w'))
json.dump({"dim": int(vecs.shape[1]), "scale": scale}, open('data/photo_meta.json', 'w'))
quantized.tofile('data/photo_text.bin')
print(f"Done! {os.path.getsize('data/photo_text.bin') / 1e6:.1f} MB.")
