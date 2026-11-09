#!/usr/bin/env python3
"""prof_report.py TABLE.txt [TOKENS]: a hipprof table of a decode run as a per-token breakdown.

Per kernel launch shape: launches a token (count / TOKENS, the tokens the profiled requests decoded; default: the most frequent count),
mean time of one launch (timed with a sync after it: kernel plus the runtime's fixed cost), the pipelined time from the replays (rep_us),
the weight bytes it reads and the GB/s of both against the card's 409.6 GB/s, and the share of the token's time.
"""
import re, sys, collections

rows = []
for line in open(sys.argv[1]):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    rows.append(dict(name=f[0], count=int(f[1]), grid=f[2], block=f[3], shared=int(f[4]), m=int(f[5]), n=int(f[6]), k=int(f[7]), sides=int(f[8]), avg=float(f[9]), min=float(f[10]), total=float(f[11]), rep=float(f[12]), bytes=float(f[13]), gap=float(f[14]) if len(f) > 14 else 0.0))
decode = [r for r in rows if r["m"] in (0, 1)]
head = [r for r in decode if r["n"] >= 200000]  # the lm_head launch: once a token
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else (head[0]["count"] if head else collections.Counter(r["count"] for r in decode).most_common(1)[0][0])
def short(n):
    if n.startswith("_ZN2tf4rocm"):
        return re.sub(r"^_ZN2tf4rocm\d+", "", n).split("INS0_")[0] + "<" + ",".join(re.findall(r"Li(\d+)", n)) + ">"
    return n
live = [r for r in decode if r["count"] >= tokens // 2]
for r in live:
    r["t"] = r["rep"] if r["rep"] else r["avg"]  # pipelined where replayed
tot = sum(r["t"] * r["count"] for r in live) / tokens
launches = sum(r["count"] for r in live) / tokens
gaps = sum(r["gap"] * r["count"] for r in live) / tokens
print(f"{tokens} decode tokens, {launches:.0f} launches a token; sum of kernel times (replayed, else synced) {tot / 1000:.2f} ms a token, mean gaps between kernels {gaps / 1000:.2f} ms a token ({gaps / launches:.1f} us a launch)")
print(f"{'kernel':34s} {'grid':>10s} {'blk':>5s} {'n':>6s} {'k':>5s} {'/tok':>5s} {'us':>7s} {'min':>6s} {'rep_us':>7s} {'gap':>5s} {'MB':>6s} {'GB/s':>6s} {'ms/tok':>7s} {'%':>5s}")
for r in sorted(live, key=lambda r: -r["t"] * r["count"]):
    per = r["count"] / tokens
    us = r["t"]
    ms = us * per / 1000
    gbs = r["bytes"] / us / 1000 if r["bytes"] and us else 0
    print(f"{short(r['name'])[:34]:34s} {r['grid']:>10s} {r['block'].split(',')[0]:>5s} {r['n']:>6d} {r['k']:>5d} {per:5.1f} {us:7.1f} {r.get('min', 0):6.1f} {r['rep']:7.1f} {r['gap']:5.1f} {r['bytes'] / 1e6:6.1f} {gbs:6.0f} {ms:7.3f} {100 * us * per / tot:5.1f}")
