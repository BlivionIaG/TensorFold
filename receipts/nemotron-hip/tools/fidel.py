"""fidel.py OUTDIR LABEL=RIG...: per-row top-1 table and every top-token difference against the fp32 reference.
Pooled rows: story_full (32), chat_long (224), code_long (224); a long sequence holds its full one's rows."""
import csv, hashlib, json, os, sys
import numpy as np
from tokenizers import Tokenizer
A = os.path.expanduser("$REFS"); N = os.path.expanduser("$REFS"); W = os.path.expanduser("$WORK/logs"); V = 131072
P = {"story_full": 15, "chat_long": 27, "code_long": 15}
tok = Tokenizer.from_file(os.path.expanduser("~/models/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-MLX-4bit/tokenizer.json"))
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
def ref(seq): return np.load(f"{A}/ref_{seq}.npy" if seq.endswith("long") else f"{N}/ref_{seq}.npy", mmap_mode="r")
summary = {}
diffs = []
for arg in sys.argv[2:]:
    label, rig = arg.split("=")
    for kind, path in (("dec", "decode"), ("pro", "prompt")):
        rows = []
        files = {}
        for seq, p in P.items():
            f = f"{W}/{kind}_{label}_final_{seq}.bin"
            files[seq] = hashlib.sha256(open(f, "rb").read()).hexdigest()
            h = np.fromfile(f, dtype=np.float32).reshape(-1, V).astype(np.float64)
            if kind == "pro": h = h[p:]
            r = np.asarray(ref(seq)[p:p + len(h)], dtype=np.float64)
            assert len(r) == len(h)
            for i in range(len(r)):
                rb, hb = int(r[i].argmax()), int(h[i].argmax())
                rs = np.partition(r[i], -2)[-2:]; hs = np.partition(h[i], -2)[-2:]
                e = np.abs(r[i] - h[i])
                row = {"rig": rig, "path": path, "seq": seq, "row": i, "position": p + i, "ref_top": rb, "native_top": hb,
                       "agree": rb == hb, "ref_top1_minus_top2": round(float(rs[1] - rs[0]), 4),
                       "native_top1_minus_top2": round(float(hs[1] - hs[0]), 4),
                       "ref_margin_ref_over_native_pick": round(float(r[i, rb] - r[i, hb]), 4),
                       "native_margin_native_over_ref_pick": round(float(h[i, hb] - h[i, rb]), 4),
                       "max_abs_err": round(float(e.max()), 4), "mean_abs_err": round(float(e.mean()), 5)}
                rows.append(row)
                if rb != hb:
                    diffs.append({**row, "ref_top_text": tok.decode([rb]), "native_top_text": tok.decode([hb])})
        with open(f"{out}/rows_{rig}_{path}.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        summary[f"{rig} {path}"] = {"agree": sum(r["agree"] for r in rows), "rows": len(rows),
                                     "max_abs_err": max(r["max_abs_err"] for r in rows),
                                     "mean_abs_err": round(float(np.mean([r["mean_abs_err"] for r in rows])), 5),
                                     "logits_sha256": files}
json.dump({"summary": summary, "differences": diffs}, open(f"{out}/fidelity.json", "w"), indent=1)
for k, v in summary.items(): print(k, f"{v['agree']}/{v['rows']}", "max", v["max_abs_err"], "mean", v["mean_abs_err"])
for d in diffs: print(d["rig"], d["path"], d["seq"], "pos", d["position"], "ref", d["ref_top"], repr(d["ref_top_text"]), "native", d["native_top"], repr(d["native_top_text"]), "ref margin", d["ref_margin_ref_over_native_pick"], "native margin", d["native_margin_native_over_ref_pick"])
