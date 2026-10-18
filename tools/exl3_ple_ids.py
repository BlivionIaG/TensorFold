#!/usr/bin/env python3
"""Independent numpy check of the PLE n-gram row ids (tensorfold qwen4_exp/cuda/ngram.py NGram.ids), constants read from the n-gram file.
  exl3_ple_ids.py DIR ids   -> prints 'IDS <token>: 16 row ids' for the 16-token teacher-forced sentence (compare with tf-cuda-test exl3-forward ple-ids)
"""
import json, struct, sys
import numpy as np

d = sys.argv[1]
fh = open(d + "/ngram_embedding.safetensors", "rb")
n = struct.unpack("<Q", fh.read(8))[0]
hdr = json.loads(fh.read(n))
base = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
def arr(name, dt):
    e = hdr[base + name]
    fh.seek(8 + n + e["data_offsets"][0])
    return np.frombuffer(fh.read(e["data_offsets"][1] - e["data_offsets"][0]), dtype=dt)
offs, sizes, mult = arr("head_offsets", "<i8"), arr("head_vocab_sizes", "<i8"), arr("layer_multipliers", "<i8")
eos = 248044
tokens = [760, 6511, 314, 9338, 369, 11751, 13, 1049, 369, 3750, 364, 279, 242476, 300, 21262, 13]
seq = np.concatenate([[eos, eos], tokens]).astype(np.int64)[None]
pos = np.arange(seq.shape[1])
eos_at = np.where(seq == eos, pos[None], -1)
before = np.concatenate([np.full((1, 1), -1), np.maximum.accumulate(eos_at, axis=1)[:, :-1]], axis=1)
in_segment = pos[None] - (before + 1)
shifted = []
for shift in range(3):
    source = pos - shift
    taken = np.take_along_axis(seq, np.broadcast_to(np.maximum(source, 0)[None], seq.shape), axis=1)
    shifted.append(np.where((in_segment >= shift) & (source[None] >= 0), taken, eos))
blocks = []
with np.errstate(over="ignore"):
    for ngram in range(2, 4):
        first = (ngram - 2) * 8
        mixed = shifted[0] * mult[0]
        for p in range(1, ngram):
            mixed = np.bitwise_xor(mixed, shifted[p] * mult[p])
        blocks.append(mixed[..., None] % sizes[first:first + 8] + offs[first:first + 8])
out = np.concatenate(blocks, axis=-1)[0, 2:]
for t, row in zip(tokens, out):
    print("IDS %d: %s" % (t, " ".join(str(int(v)) for v in row)))
