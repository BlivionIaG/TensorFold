#!/usr/bin/env python3
"""Minimal byte-level BPE for the Qwen3.8-Flash-Next tokenizer.json (no tokenizers/transformers needed).

  exl3_tok.py DIR encode "some ASCII text"   -> comma separated ids
  exl3_tok.py DIR decode 1,2,3               -> the text, then one repr per token

The pretokenizer regex is the tokenizer.json one with \\p{L}\\p{M} -> [A-Za-z] and \\p{N} -> [0-9], so it is exact for ASCII text only.
"""
import json, re, sys

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return dict(zip(bs, map(chr, cs)))

PAT = re.compile(r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\nA-Za-z0-9]?[A-Za-z]+|[0-9]| ?[^\sA-Za-z0-9]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+")

def load(d):
    t = json.load(open(d + "/tokenizer.json"))
    vocab = t["model"]["vocab"]
    ranks = {}
    for i, m in enumerate(t["model"]["merges"]):
        a, b = m.split(" ") if isinstance(m, str) else m
        ranks[(a, b)] = i
    return vocab, ranks

def bpe(word, ranks):
    parts = list(word)
    while len(parts) > 1:
        best, at = None, -1
        for i in range(len(parts) - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None and (best is None or r < best):
                best, at = r, i
        if best is None:
            break
        parts[at:at + 2] = [parts[at] + parts[at + 1]]
    return parts

def main():
    d, cmd, arg = sys.argv[1:4]
    vocab, ranks = load(d)
    b2u = bytes_to_unicode()
    u2b = {v: k for k, v in b2u.items()}
    inv = {i: s for s, i in vocab.items()}
    if cmd == "encode":
        ids = []
        for piece in PAT.findall(arg):
            word = "".join(b2u[b] for b in piece.encode())
            ids += [vocab[p] for p in bpe(word, ranks)]
        print(",".join(map(str, ids)))
    else:
        ids = [int(x) for x in arg.split(",") if x]
        toks = []
        for i in ids:
            s = inv.get(i, "<%d>" % i)
            toks.append(bytes(u2b[c] for c in s).decode("utf-8", "replace") if all(c in u2b for c in s) else s)
        print("".join(toks))
        for i, t in zip(ids, toks):
            print(i, repr(t))

main()
