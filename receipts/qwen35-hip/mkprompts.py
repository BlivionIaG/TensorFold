# mkprompts.py MODEL_DIR REPO OUT N...: exact token arrays from the repo's public docs, tokenized by the checkpoint
import glob, json, os, sys
from tokenizers import Tokenizer
model, repo, out, ns = sys.argv[1], sys.argv[2], sys.argv[3], [int(x) for x in sys.argv[4:]]
tok = Tokenizer.from_file(os.path.join(model, "tokenizer.json"))
files = sorted(glob.glob(os.path.join(repo, "docs", "**", "*.md"), recursive=True)) + [os.path.join(repo, f) for f in ("README.md", "CONTRIBUTING.md", "LICENSE")]
text = "\n\n".join(open(f, encoding="utf-8").read() for f in files if os.path.exists(f))
ids = tok.encode(text, add_special_tokens=False).ids
need = max(ns); reps = -(-need // len(ids)); ids = (ids * reps)[:need]
os.makedirs(out, exist_ok=True)
json.dump({"source": [os.path.relpath(f, repo) for f in files], "tokens_per_pass": len(tok.encode(text, add_special_tokens=False).ids)}, open(os.path.join(out, "prompts_source.json"), "w"), indent=1)
for n in ns: json.dump(ids[:n], open(os.path.join(out, f"prefill_{n}.json"), "w"))
print("source tokens", len(tok.encode(text, add_special_tokens=False).ids), "written", ns)
