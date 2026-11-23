#!/usr/bin/env python3
"""predcheck.py: how well did the router's cost model predict what each request got?

  predcheck.py R.json explain.jsonl [--skip N]

R.json is `livebench.py --json` output of a run through the router; explain.jsonl is the router's --explain-file log. The i-th arrival of the run
is matched with the i-th decision record after the first N records of the file (the records are in arrival order). Predicted TTFT is the chosen
backend's `pred.ttft` at decision time; measured TTFT is first token minus arrival minus the time the request waited in the router (`held_s`),
because the prediction is made after that wait. Prints the error by prompt size and the share within the phase-0 gate (20 %).
"""
import argparse
import json
import statistics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("explain")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--gate", type=float, default=0.20)
    a = ap.parse_args()
    run = json.load(open(a.run))
    reqs = sorted((r for r in run["requests"] if r["t_arr"] is not None), key=lambda r: r["t_arr"])
    recs = [json.loads(l) for l in open(a.explain) if l.strip()][a.skip:]
    recs.sort(key=lambda r: r["ts"])
    n = min(len(reqs), len(recs))
    if len(reqs) != len(recs):
        print("note: %d arrivals vs %d decision records after --skip; matching the first %d" % (len(reqs), len(recs), n))
    rows = []
    for r, d in zip(reqs[:n], recs[:n]):
        if r["t_first"] is None or not d.get("choice"):
            continue
        pred = d["candidates"][d["choice"]]["pred"]["ttft"]
        meas = r["t_first"] - r["t_arr"] - (d.get("held_s") or 0.0)
        rows.append((r["n_in"], d["choice"], pred, meas))
    if not rows:
        raise SystemExit("no matched requests with a first token")
    bins = [(0, 4000), (4000, 12000), (12000, 10 ** 9)]
    print("%-14s %-4s %-5s %-10s %-10s %-10s %-10s" % ("prompt tokens", "to", "n", "pred med s", "meas med s", "med |err|", "within %d%%" % round(a.gate * 100)))
    for lo, hi in bins:
        for b in sorted({x[1] for x in rows}):
            sel = [x for x in rows if lo <= x[0] < hi and x[1] == b]
            if not sel:
                continue
            err = [abs(p - m) / max(m, 0.05) for _, _, p, m in sel]
            print("%-14s %-4s %-5d %-10.2f %-10.2f %-10.2f %-10.0f%%" % ("%d-%s" % (lo, hi if hi < 10 ** 9 else "+"), b, len(sel), statistics.median(p for _, _, p, _ in sel),
                                                                      statistics.median(m for _, _, _, m in sel), statistics.median(err), 100.0 * sum(e <= a.gate for e in err) / len(err)))
    err = [abs(p - m) / max(m, 0.05) for _, _, p, m in rows]
    print("all: %d requests, median |err| %.0f%%, within %.0f%%: %.0f%%" % (len(rows), 100 * statistics.median(err), 100 * a.gate, 100.0 * sum(e <= a.gate for e in err) / len(err)))


if __name__ == "__main__":
    main()
