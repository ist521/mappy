"""
Builds the word data for Mappy (combined GloVe + MiniLM version).
Run it ONCE on your computer, with your .venv active:

    pip install gensim wordfreq nltk sentence-transformers
    python export_vocab.py

The first run downloads GloVe (~380 MB, one time only). It creates a `data/`
folder that the website loads:

    data/meta.json        - settings the website needs (vector sizes, scales)
    data/words.json       - every word the site can look up (50k)
    data/vectors.bin      - GloVe vector for each of those words (8-bit)
    data/pool.json        - which words may APPEAR on a map, plus "generic word" scores
    data/pool_minilm.bin  - MiniLM vector for each map word (8-bit)
    data/antonyms.json    - WordNet opposites, so "hot" never shows "cold"

Commit the `data/` folder to GitHub.

To block extra words from ever appearing, put them in a file named
`blocklist.txt` next to this script (one word per line) and re-run.
"""
import json
import os
import urllib.request

import numpy as np
import gensim.downloader as api
import nltk
from nltk.corpus import wordnet as wn
from sentence_transformers import SentenceTransformer
from wordfreq import top_n_list

GLOVE_MODEL = 'glove-wiki-gigaword-300'
MINILM_MODEL = 'all-MiniLM-L6-v2'      # must match the model name in map.js
LOOKUP_SIZE = 50000   # words people can TYPE (includes plurals, -ing forms, etc.)
POOL_SIZE = 50000     # how common a word must be to APPEAR on the map
                      # (50,000 lets in specific words like "poodle" and "terrier";
                      #  lower it, e.g. 20000, for only very everyday words)

nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))

print(f"Loading {GLOVE_MODEL} (downloads ~380 MB the first time)...")
glove = api.load(GLOVE_MODEL)
print(f"Loading {MINILM_MODEL}...")
minilm = SentenceTransformer(MINILM_MODEL)

# ---------- offensive-word blocklist ----------
blocked = set()
BLOCKLIST_URL = ("https://raw.githubusercontent.com/LDNOOBW/"
                 "List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en")
try:
    with urllib.request.urlopen(BLOCKLIST_URL, timeout=30) as r:
        blocked |= {w.strip().lower() for w in r.read().decode().splitlines() if w.strip()}
    print(f"Loaded {len(blocked)} blocked words.")
except Exception as e:
    print(f"WARNING: could not download the blocklist ({e}). Continuing without it.")
if os.path.exists('blocklist.txt'):
    with open('blocklist.txt') as f:
        extra = {w.strip().lower() for w in f if w.strip()}
    blocked |= extra
    print(f"Added {len(extra)} words from blocklist.txt.")


def is_proper_noun(w):
    """True if WordNet only knows w as a name (Paris, Einstein...)."""
    names = [l.name() for s in wn.synsets(w) for l in s.lemmas() if l.name().lower() == w]
    return bool(names) and all(n[0].isupper() for n in names)


def mainly_adverb(w):
    """True if w's most common meaning is an adverb (really, always, maybe...)."""
    syns = wn.synsets(w)
    return bool(syns) and syns[0].pos() == 'r'


def quantize(v):
    """Compress unit vectors to 8-bit integers. Returns (int8 array, scale)."""
    scale = 127.0 / float(np.abs(v).max())
    return np.clip(np.round(v * scale), -127, 127).astype(np.int8), scale


def hub_scores(rows, against):
    """Average similarity of each row to its 10 nearest neighbours in `against`.
    Generic words ("thing", "know") score high; the website subtracts this
    when ranking so they sink and specific words rise."""
    out = np.zeros(len(rows), dtype=np.float32)
    for start in range(0, len(rows), 2000):
        sims = rows[start:start + 2000] @ against.T
        out[start:start + 2000] = np.sort(sims, axis=1)[:, -11:-1].mean(axis=1)
    return out


# ---------- words people can type ----------
lookup = []
for w in top_n_list('en', LOOKUP_SIZE * 2):
    if w.isalpha() and w in glove.key_to_index and w not in blocked:
        lookup.append(w)
    if len(lookup) == LOOKUP_SIZE:
        break
index = {w: i for i, w in enumerate(lookup)}

# ---------- words that may appear on the map ----------
common = set(top_n_list('en', POOL_SIZE))
pool = [
    i for i, w in enumerate(lookup)
    if w in common
    and len(w) >= 3
    and wn.morphy(w) == w          # dictionary base form: drops "dogs", slang like "dogg"
    and not is_proper_noun(w)      # drops names
    and not mainly_adverb(w)       # drops filler like "really", "always"
    and w not in STOPWORDS         # drops "the", "would", "very"...
]

# ---------- antonyms ----------
antonyms = {}
for w in lookup:
    ants = set()
    seen_pos = set()
    for s in wn.synsets(w):
        pos = 'a' if s.pos() == 's' else s.pos()
        if pos in seen_pos:          # only the most common meaning per part of speech,
            continue                 # so "hot" (stolen) doesn't make "legal" an opposite
        seen_pos.add(pos)
        # "freezing" -> similar to "cold" -> antonym "hot";
        # "happy" -> also see "glad" -> antonym "sad"
        heads = [s] + s.similar_tos() + s.also_sees()
        for h in heads:
            for l in h.lemmas():
                for a in l.antonyms():
                    name = a.name().lower()
                    if name in index and name != w:
                        ants.add(name)
    if ants:
        antonyms[w] = ants
# make it two-way: if "sad" is the opposite of "happy", "happy" is the opposite of "sad"
for w, ants in list(antonyms.items()):
    for a in ants:
        antonyms.setdefault(a, set()).add(w)
antonyms = {w: sorted(a) for w, a in antonyms.items()}

# ---------- GloVe vectors (all typeable words) ----------
gvecs = np.array([glove[w] for w in lookup], dtype=np.float32)
gvecs /= np.linalg.norm(gvecs, axis=1, keepdims=True)
gq, gscale = quantize(gvecs)

# ---------- MiniLM vectors (map words only) ----------
print(f"Encoding {len(pool)} map words with {MINILM_MODEL} (a minute or two)...")
mvecs = minilm.encode([lookup[i] for i in pool], normalize_embeddings=True,
                      show_progress_bar=True).astype(np.float32)
mq, mscale = quantize(mvecs)

# ---------- generic-word scores ----------
print("Scoring generic words...")
hub_g = hub_scores(gvecs[pool], gvecs)
hub_m = hub_scores(mvecs, mvecs)

# ---------- save ----------
os.makedirs('data', exist_ok=True)
with open('data/meta.json', 'w') as f:
    json.dump({"dim": int(gvecs.shape[1]), "scale": gscale,
               "mdim": int(mvecs.shape[1]), "mscale": mscale}, f)
with open('data/words.json', 'w') as f:
    json.dump(lookup, f)
with open('data/pool.json', 'w') as f:
    json.dump({"index": pool,
               "hub": [round(float(h), 3) for h in hub_g],
               "hubm": [round(float(h), 3) for h in hub_m]}, f)
with open('data/antonyms.json', 'w') as f:
    json.dump(antonyms, f)
gq.tofile('data/vectors.bin')
mq.tofile('data/pool_minilm.bin')

# the old version's file isn't used anymore
if os.path.exists('data/embeddings.bin'):
    os.remove('data/embeddings.bin')

print(f"Done! {len(lookup)} typeable words, {len(pool)} map words, "
      f"{len(antonyms)} words with antonyms.")
