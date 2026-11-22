#!/usr/bin/env python3
"""presets_from_bench.py: turn llmbench.py JSONL (ttft, decode and conc runs of ONE backend) into a router config override.

  presets_from_bench.py --name S  s-ttft.jsonl s-conc.jsonl [s-hol.jsonl] > s.json
  presets_from_bench.py --name M  m-ttft.jsonl m-conc.jsonl [m-hol.jsonl] > m.json     (merge both under "backends" by hand)

Writes {"backends": {NAME: {"prefill_pts": ..., "decode": {...}}}} to stdout and a table to stderr. It fills only what a bench measures:
  prefill_pts            median TTFT of a one-request-at-a-time ttft run, per prompt size (prompt tokens, seconds)
  decode.round_ms_pts    median stream-event gap of the n concurrent short-prompt streams of a conc run, per n (streams, ms)
  decode.tokens_per_round_pts  mean tokens per event, per n
`prefill.mode` (interleave or block), the guard and the cache sizes are judgments: the hol run's longest stall is printed to decide them.
"""
import argparse
import json
import statistics
import sys


def records(paths):
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def build(recs):
    ttft, conc, hol, floor = {}, {}, [], []
    for r in recs:
        m = r.get("mode")
        if m == "ttft" and r.get("ok") and r.get("pt") and r.get("ttft"):
            ttft.setdefault(r["want"], []).append((r["pt"], r["ttft"]))
        elif m == "decode" and r.get("ok") and not r.get("ctx") and r.get("pt") and r.get("ttft"):
            floor.append((r["pt"], r["ttft"]))                 # short prompt: the fixed first-token cost
        elif m == "conc" and r.get("ok") and not r.get("prompt") and r.get("itl_p50") is not None:
            conc.setdefault(r["n"], []).append(r)
        elif m and m.startswith("hol"):
            hol.append(r)
    pf = [[round(med([p for p, _ in floor])), round(med([t for _, t in floor]), 3)]] if floor else []
    for want in sorted(ttft):
        pf.append([round(med([p for p, _ in ttft[want]])), round(med([t for _, t in ttft[want]]), 3)])
    rounds, tpr = [], []
    for n in sorted(conc):
        rounds.append([n, round(med([r["itl_p50"] for r in conc[n]]) * 1000.0, 2)])
        tpr.append([n, round(med([r["tok_per_event"] for r in conc[n]]), 3)])
    return pf, rounds, tpr, hol


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True, help="backend name in the router config (S or M)")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    pf, rounds, tpr, hol = build(records(a.files))
    if not pf or not rounds:
        sys.exit("need ttft and conc records (decode records at n=1 come from `conc --n 1,...`)")
    ov = {"backends": {a.name: {"prefill_pts": pf, "decode": {"round_ms_pts": rounds, "tokens_per_round_pts": tpr}}}}
    json.dump(ov, sys.stdout, indent=1)
    print()
    sys.stderr.write("prefill (tokens, s): %s\nround (streams, ms): %s\ntokens per round: %s\n" % (pf, rounds, tpr))
    for r in hol:
        sys.stderr.write("hol: %s\n" % json.dumps({k: r[k] for k in r if k in ("long", "b_first_token", "max_stall", "a_tokens_during", "during")}))


if __name__ == "__main__":
    main()
