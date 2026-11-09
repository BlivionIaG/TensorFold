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
    rows.append(dict(name=f[0], count=int(f[1]), grid=f[2], block=f[3], shared=int(f[4]), m=int(f[5]), n=int(f[6]), k=int(f[7]), sides=int(f[8]), avg=float(f[9]), total=float(f[11]), rep=float(f[12]), bytes=float(f[13])))
decode = [r for r in rows if r["m"] in (0, 1)]
tokens = int(sys.argv[2]) if len(sys.argv) > 2 else collections.Counter(r["count"] for r in decode).most_common(1)[0][0]
def short(n):
    if n.startswith("_ZN2tf4rocm"):
        return re.sub(r"^_ZN2tf4rocm\d+", "", n).split("INS0_")[0] + "<" + ",".join(re.findall(r"Li(\d+)", n)) + ">"
    return n
tot = sum(r["total"] for r in decode if r["count"] >= tokens // 2) / tokens
print(f"{tokens} decode tokens; sum of mean launch times a token {tot / 1000:.2f} ms (each launch synced)")
print(f"{'kernel':34s} {'grid':>10s} {'blk':>7s} {'m':>2s} {'n':>6s} {'k':>5s} {'/tok':>5s} {'us':>7s} {'rep_us':>7s} {'MB':>6s} {'GB/s':>6s} {'repGB/s':>7s} {'ms/tok':>7s} {'%':>5s}")
rep_sum = 0.0
for r in sorted(decode, key=lambda r: -r["total"]):
    if r["count"] < tokens // 2:
        continue
    per = r["count"] / tokens
    ms = r["total"] / tokens / 1000
    gbs = r["bytes"] / r["avg"] / 1000 if r["bytes"] and r["avg"] else 0
    rgbs = r["bytes"] / r["rep"] / 1000 if r["bytes"] and r["rep"] else 0
    rep_sum += (r["rep"] if r["rep"] else r["avg"]) * per
    share = 100 * (r['total'] / tokens) / tot
    print(f"{short(r['name'])[:34]:34s} {r['grid']:>10s} {r['block'].split(',')[0]:>7s} {r['m']:>2d} {r['n']:>6d} {r['k']:>5d} {per:5.1f} {r['avg']:7.1f} {r['rep']:7.1f} {r['bytes'] / 1e6:6.1f} {gbs:6.0f} {rgbs:7.0f} {ms:7.3f} {share:5.1f}")
print(f"pipelined estimate (rep time where measured, else the synced time): {rep_sum / 1000:.2f} ms a token")
