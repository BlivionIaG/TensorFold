"""mkids.py: token arrays from the TensorFold v1.1.0 Markdown files (76aafcf2), path order, each array a prefix of the next."""
import hashlib, json, subprocess, sys
from tokenizers import Tokenizer
repo, tok, out = sys.argv[1], sys.argv[2], sys.argv[3]
C = "76aafcf2"
git = ["git", "-c", "gc.auto=0", "-c", "maintenance.auto=false", "-C", repo]
paths = sorted(p for p in subprocess.check_output(git + ["ls-tree", "-r", "--name-only", C]).decode().split("\n") if p.endswith(".md"))
text = "".join(f"\n\n# {p}\n\n" + subprocess.check_output(git + ["show", f"{C}:{p}"]).decode() for p in paths)
ids = Tokenizer.from_file(tok).encode(text, add_special_tokens=False).ids
print("files", len(paths), "chars", len(text), "tokens", len(ids))
man = {"source": f"TensorFold {C}: every tracked *.md file in path order, each after a line \"# <path>\"", "files": paths,
       "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "tokenizer": "the checkpoint tokenizer.json, add_special_tokens false", "arrays": {}}
for n in (2048, 8192, 32768, 65536, 131072, 262080):
    a = ids[:n]; assert len(a) == n
    s = json.dumps(a)
    open(f"{out}/prompt{n}.json", "w").write(s)
    man["arrays"][str(n)] = {"file": f"prompt{n}.json", "ids_sha256": hashlib.sha256(s.encode()).hexdigest(),
                              "file_sha256": hashlib.sha256(s.encode()).hexdigest()}
json.dump(man, open(f"{out}/ids_manifest.json", "w"), indent=1)
print(json.dumps(man["arrays"], indent=1))
