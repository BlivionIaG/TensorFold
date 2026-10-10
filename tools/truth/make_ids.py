#!/usr/bin/env python3
"""Tokenize a text file with the model's tokenizer.json and save a window as ids.npy.

Usage: make_ids.py <model_dir> <text> <tokens> <out.npy> [--offset N]   (offset counts tokens)
"""
import argparse
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("model"), ap.add_argument("text"), ap.add_argument("tokens", type=int), ap.add_argument("out")
ap.add_argument("--offset", type=int, default=0)
a = ap.parse_args()
ids = Tokenizer.from_file(str(Path(a.model) / "tokenizer.json")).encode(Path(a.text).read_text()).ids
win = np.array(ids[a.offset:a.offset + a.tokens], np.int64)
if len(win) < a.tokens:
    raise SystemExit(f"only {len(win)} tokens available after offset {a.offset}")
np.save(a.out, win)
print(f"{a.out}: {len(win)} of {len(ids)} tokens")
