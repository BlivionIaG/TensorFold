"""annotate.py FILE KEY=VALUE...: add receipt fields (source_commit, binary_sha256, ...) to a bench JSON; a list is wrapped."""
import json, sys
f = sys.argv[1]; d = json.load(open(f))
meta = dict(a.split("=", 1) for a in sys.argv[2:])
d = {**meta, **d} if isinstance(d, dict) else {**meta, "results": d}
json.dump(d, open(f, "w"), indent=2); open(f, "a").write("\n")
