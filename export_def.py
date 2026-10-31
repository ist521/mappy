"""
Saves a short definition and picture-matching info for every word.
Run it once (with .venv active):   python export_definitions.py

Creates:
  data/senses.json       - word -> up to 3 meanings, each with a short definition,
                           how common it is, hint words, and which ARASAAC pictogram
                           meanings count as a match (own meaning first, then broader
                           ones, e.g. poodle -> dog). The website picks the meaning
                           that best fits the other words on the map, so "canine"
                           next to "dog" means the animal, not the tooth.
  data/photos.json       - meaning -> a Wikimedia Commons photo, for words ARASAAC
                           has no pictogram for (poodle, terrier...). Matched by
                           meaning through Wikidata, so "bulldog" never gets the game.

Uses WordNet 3.1, because ARASAAC labels its pictograms with WordNet 3.1 codes.
"""
import json
import re
import urllib.parse
import urllib.request

import nltk
nltk.download('wordnet31', quiet=True)
from nltk.corpus import wordnet31 as wn

PARENTS = 2   # how many broader meanings to try for a picture (poodle -> dog -> canine)


def code(s):
    """WordNet 3.1 code in ARASAAC's format, e.g. '02086723-n'."""
    return f"{s.offset():08d}-{'a' if s.pos() == 's' else s.pos()}"


SENSES = 3   # how many meanings to keep per word; the website picks the one that fits the map

words = json.load(open('data/words.json'))
senses = {}
for w in words:
    base = wn.morphy(w) or w            # "dogs" -> "dog"
    syns = wn.synsets(base)
    if not syns:
        continue
    count = lambda s: sum(l.count() for l in s.lemmas() if l.name().lower() == base)
    syns = sorted(syns, key=count, reverse=True)[:SENSES]     # most commonly used meanings first
    total = sum(count(s) for s in syns) or 1
    entry = []
    for s in syns:
        tries = [[base, code(s)]]
        p = s
        for _ in range(PARENTS):
            parents = p.hypernyms() + p.instance_hypernyms()
            if not parents:
                break
            p = parents[0]
            tries.append([p.lemmas()[0].name().replace('_', ' '), code(p)])
        # words that describe this meaning: other names for it + its broader category
        # ("canine" the animal -> "carnivore"; "canine" the tooth -> "tooth")
        hints = [l.name().replace('_', ' ') for l in s.lemmas() if l.name().lower() != base]
        hints += [h.lemmas()[0].name().replace('_', ' ') for h in s.hypernyms()]
        entry.append({
            "d": s.definition().split(';')[0],    # definition, first part only
            "c": round(count(s) / total, 2),       # how common this meaning is (0-1)
            "h": ' '.join(hints),
            "p": tries,                            # meanings to try for a picture
        })
    senses[w] = entry

json.dump(senses, open('data/senses.json', 'w'), separators=(',', ':'))
print(f"Saved meanings for {len(senses)} words.")
pics = {w: [t for sense in e for t in sense["p"]] for w, e in senses.items()}   # for the photo step

# ---------- Wikimedia photos, matched by meaning ----------
# Ask Wikidata for every entry that has a WordNet 3.1 code AND a photo.
print("Asking Wikidata for photos (can take up to a minute)...")
needed = {c for tries in pics.values() for _, c in tries}
query = "SELECT ?syn ?img WHERE { ?item wdt:P8814 ?syn ; wdt:P18 ?img . }"
url = "https://query.wikidata.org/sparql?" + urllib.parse.urlencode({"query": query, "format": "json"})
req = urllib.request.Request(url, headers={"User-Agent": "Mappy word map (class project)"})
photos = {}
try:
    with urllib.request.urlopen(req, timeout=120) as r:
        rows = json.load(r)["results"]["bindings"]
    for row in rows:
        value = row["syn"]["value"]             # e.g. "02115987-n"
        digits = re.search(r"\d{8}", value)
        pos = re.findall(r"[nvasr]", re.sub(r"\d", "", value))
        if not digits or not pos:
            continue
        c = f"{digits.group()}-{'a' if pos[-1] == 's' else pos[-1]}"
        if c in needed and c not in photos:
            # keep just the file name, e.g. "Poodle.jpg"
            photos[c] = urllib.parse.unquote(row["img"]["value"].rsplit("/", 1)[-1])
    print(f"Found photos for {len(photos)} meanings.")
except Exception as e:
    print(f"WARNING: could not reach Wikidata ({e}). Saving without photos.")
json.dump(photos, open('data/photos.json', 'w'), separators=(',', ':'))
