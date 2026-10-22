"""
Run this ONCE on your computer (with the same Python setup you use for app.py):

    python3 export_vocab.py

It creates a `data/` folder with:
    data/words.json      - the vocabulary list
    data/embeddings.bin  - every word's vector, compressed to 8-bit (~8 MB)

Commit the `data/` folder to GitHub. The website loads these files instead of
running Python.
"""
import json
import os
import re

import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

# same vocabulary rules as app.py
pattern = re.compile(r'^[a-zA-Z]{2,}$')
words = set()
for token in model.tokenizer.vocab.keys():
    if token.startswith('##') or token.startswith('['):
        continue
    if pattern.match(token):
        words.add(token.lower())
words = sorted(words)
print(f"Encoding {len(words)} words... (takes a minute or two)")

emb = model.encode(words, normalize_embeddings=True, show_progress_bar=True)

# compress each number from 32-bit float to 8-bit integer (-127..127)
quantized = np.clip(np.round(emb * 127), -127, 127).astype(np.int8)

os.makedirs('data', exist_ok=True)
with open('data/words.json', 'w') as f:
    json.dump(words, f)
quantized.tofile('data/embeddings.bin')

size_mb = os.path.getsize('data/embeddings.bin') / 1e6
print(f"Done! Saved {len(words)} words, {quantized.shape[1]} dimensions, {size_mb:.1f} MB.")
