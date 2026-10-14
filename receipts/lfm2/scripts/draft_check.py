"""Drafted replies against "draft": false on a served TensorFold: 256 greedy tokens per prompt, compared as text."""
import json, sys, urllib.request
url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
def comp(prompt, draft):
    body = {"model": "m", "prompt": prompt, "max_tokens": 256, "temperature": 0}
    if draft is False: body["draft"] = False
    req = urllib.request.Request(url + "/v1/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["choices"][0]["text"]
prompts = ["def fibonacci(n):\n", "The following is a list of the planets in order:\n1.", "import json\n\ndef load_config(path):\n    with open(path) as f:\n"]
eq = 0
for p in prompts:
    a = comp(p, None); b = comp(p, False); eq += a == b
    print(repr(p[:30]), "equal" if a == b else "DIFFERENT", len(a))
print(f"{eq} of {len(prompts)} drafted replies equal to draft:false")
