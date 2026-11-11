#!/usr/bin/env python3
"""The gfx900 gate and decode benchmark: serve Qwen3.5-4B MLX 4-bit through a HIP runtime and compare it with mlx_lm.

    tools/vega-macos/vega_gate.py [--lib hipmac/build/libamdhip64.dylib] [--tag NAME] [--speed N] [--logits] [--prof each|graph]

One server (--parallel 1, greedy, no drafts) answers
  nothink   "What is the capital of France? Answer in one short sentence."  8 tokens, token sha must be the mlx_lm one
  think96   the same with thinking on, 96 tokens, sha must be the mlx_lm one
  speed     N runs of a 128-token story: TensorFold's own decode tokens/s (and prefill) per run
then, with --logits, `tf-qwen35-test logits --decode` over 87 ids (the decode path row by row) against mlx_lm's float32 logits: cosine
per row, argmax equality. With --prof the server runs behind tools/vega-macos/hipprof.c: `each` (graphs off, every kernel launched and
synced, timed; the table of the decode run is written to the output directory), or `graph` (graphs on, every graph replay timed).
The mlx_lm references come from `uvx --from mlx-lm python tools/vega-macos/mlx_ref.py` (cached in ~/.cache/tensorfold-vega/ref).
Env TF_POLICY / --policy and --env K=V reach the engine, which is how a kernel variant is switched on (see docs/recipes/vega-macos.md).
Exit status 0 when every check passed. Results are also appended as JSON lines to OUT/results.jsonl.
"""
import argparse, hashlib, json, os, signal, subprocess, sys, time, urllib.request, re, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
HOME = os.path.expanduser("~")
CACHE = os.path.join(HOME, ".cache", "tensorfold-vega")
QUESTION = "What is the capital of France? Answer in one short sentence."
STORY = "Write a long story about a dragon who learns to code."

ap = argparse.ArgumentParser()
ap.add_argument("--lib", default=os.environ.get("TENSORFOLD_HIP") or os.path.join(os.environ.get("UT4G_USBGPU", os.path.join(HOME, "Projects/Infra/ut4g-usbgpu")), "hipmac/build/libamdhip64.dylib"),
                help="libamdhip64.dylib of hipmac (default $TENSORFOLD_HIP, else $UT4G_USBGPU/hipmac/build/libamdhip64.dylib)")
ap.add_argument("--backend", default="ut4g", help="HIPMAC_BACKEND (ut4g card, mock offline)")
ap.add_argument("--bin", default=os.path.join(ROOT, "zig-out/native-hip/bin/tensorfold-native"))
ap.add_argument("--test-bin", default=os.path.join(ROOT, "zig-out/bin/tf-qwen35-test"))
ap.add_argument("--model", default=os.path.join(HOME, "models/Qwen3.5-4B-MLX-4bit"))
ap.add_argument("--ref", default=os.path.join(CACHE, "ref"))
ap.add_argument("--out", default=os.path.join(CACHE, "bench"))
ap.add_argument("--tag", default="run")
ap.add_argument("--port", type=int, default=18099)
ap.add_argument("--context", type=int, default=4096)
ap.add_argument("--speed", type=int, default=3, help="decode speed runs (0 skips)")
ap.add_argument("--story-tokens", type=int, default=128)
ap.add_argument("--logits", action="store_true", help="also compare decode-path logits with mlx_lm (needs uv, numpy)")
ap.add_argument("--prof", choices=("each", "events", "graph"), help="run behind hipprof")
ap.add_argument("--rep", type=int, default=16, help="hipprof: replays of each matching launch (pipelined time)")
ap.add_argument("--rep-match", default="affine_", help="hipprof: kernel names to replay ('' all: state is garbage afterwards, the timing is not)")
ap.add_argument("--policy", default="", help="TF_POLICY, e.g. stream=reference")
ap.add_argument("--env", action="append", default=[], help="K=V for the engine")
ap.add_argument("--no-gate", action="store_true", help="skip the token checks (speed / profile only)")
a = ap.parse_args()

os.makedirs(a.out, exist_ok=True)
ref = json.load(open(os.path.join(a.ref, "ref.json")))
results = {"tag": a.tag, "policy": a.policy, "env": a.env, "ok": True, "time": time.strftime("%Y-%m-%d %H:%M:%S")}


def fail(msg):
    results["ok"] = False
    print("FAIL", msg, flush=True)


def engine_env(shim=None):
    e = dict(os.environ)
    e["HIPMAC_BACKEND"] = a.backend
    e["TENSORFOLD_HIP"] = shim or a.lib
    if a.policy or a.prof in ("each", "events"):
        pol = [p for p in (a.policy, "graphs=off" if a.prof in ("each", "events") else "") if p]
        e["TF_POLICY"] = ",".join(pol)
    for kv in a.env:
        k, v = kv.split("=", 1)
        e[k] = v
    return e


def build_shim():
    out = os.path.join(a.out, "libhipprof.dylib")
    src = os.path.join(HERE, "hipprof.c")
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.check_call(os.environ.get("TF_CC", "clang").split() + ["-O2", "-dynamiclib", "-I", HERE, "-o", out, src])
    return out


def post(body, timeout=900):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def chat(prompt, tokens, think=False):
    return post({"messages": [{"role": "user", "content": prompt}], "max_tokens": tokens, "temperature": 0, "chat_template_kwargs": {"enable_thinking": think}})


def sha_of(log, reply_id):
    for line in open(log).read().splitlines():
        if "done " + reply_id in line:
            m = re.search(r"sha=([0-9a-f]{12})", line)
            return m.group(1) if m else None
    return None


# ---- the server ----
shim = None
env = engine_env()
if a.prof:
    shim = build_shim()
    env = engine_env(shim)
    env["HIPPROF_REAL"] = a.lib
    env["HIPPROF_REP"] = str(a.rep if a.prof == "each" else 0)
    env["HIPPROF_REP_MATCH"] = a.rep_match
    env["HIPPROF_START"] = "1"
    env["HIPPROF_LOG"] = os.path.join(a.out, f"{a.tag}.prof.txt")
    if a.prof == "graph":
        env["HIPPROF_GRAPH"] = "1"
    if a.prof == "events":
        env["HIPPROF_EVENTS"] = "1"
log = os.path.join(a.out, f"{a.tag}.server.log")
cmd = [a.bin, "serve", a.model, "--parallel", "1", "--context", str(a.context), "--port", str(a.port), "--no-thinking", "--temperature", "0", "--no-drafts"]
print("$", " ".join(cmd), "  env:", {k: v for k, v in env.items() if k.startswith(("HIP", "TF_"))}, flush=True)
srv = subprocess.Popen(cmd, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)
try:
    t0 = time.time()
    while "serving " not in open(log).read():
        if srv.poll() is not None:
            print(open(log).read())
            sys.exit("server exited")
        if time.time() - t0 > 300:
            sys.exit("server did not start")
        time.sleep(0.5)
    results["load_seconds"] = round(time.time() - t0, 1)
    print(f"server up in {results['load_seconds']} s", flush=True)

    if not a.no_gate:
        for name, think, n in (("nothink", False, 8), ("think96", True, 96)):
            r = chat(QUESTION, n, think)
            time.sleep(0.3)
            sha = sha_of(log, r["id"])
            want = ref[name].get("sha") or hashlib.sha256(",".join(map(str, ref[name]["tokens"])).encode()).hexdigest()[:12]
            results[name] = {"sha": sha, "want": want, "tokens": r["usage"]["completion_tokens"]}
            if sha != want:
                fail(f"{name}: token sha {sha} != mlx_lm {want}")
            else:
                print(f"PASS {name}: {r['usage']['completion_tokens']} tokens identical to mlx_lm (sha {sha})", flush=True)

    if a.speed:
        if a.prof:
            time.sleep(0.5)
            os.kill(srv.pid, signal.SIGUSR1)  # the table starts here
        runs = []
        for i in range(a.speed):
            t1 = time.time()
            r = chat(STORY, a.story_tokens)
            wall = time.time() - t1
            tf = r.get("tensorfold", {})
            runs.append({"completion_tokens": r["usage"]["completion_tokens"], "wall": round(wall, 3), "tensorfold": tf})
            print(f"speed run {i}: {r['usage']['completion_tokens']} tokens, wall {wall:.2f} s, tensorfold {tf}", flush=True)
        results["speed"] = runs
        tps = [x["tensorfold"].get("tokens_per_second") for x in runs if x["tensorfold"].get("tokens_per_second")]
        if tps:
            results["decode_tok_s"] = round(sorted(tps)[len(tps) // 2], 2)
            print(f"decode tok/s (median of {len(tps)}): {results['decode_tok_s']}", flush=True)
        if a.prof:
            os.kill(srv.pid, signal.SIGUSR2)
            time.sleep(1.0)
finally:
    srv.send_signal(signal.SIGINT)
    try:
        srv.wait(10)
    except subprocess.TimeoutExpired:
        srv.kill()
        srv.wait()

if a.logits and not a.prof:
    ids = os.path.join(a.ref, "ids.npy")
    ours = os.path.join(a.out, f"{a.tag}.logits.npy")
    e = engine_env()
    t2 = time.time()
    p = subprocess.run([a.test_bin, "logits", a.model, ids, ours, "--decode"], env=e, capture_output=True, text=True)
    results["logits_seconds"] = round(time.time() - t2, 1)
    if p.returncode != 0:
        fail("logits run failed: " + (p.stdout + p.stderr)[-500:])
    else:
        c = subprocess.run(["uv", "run", "--quiet", "--with", "numpy", "python", os.path.join(HERE, "cosine.py"), ours, os.path.join(a.ref, "mlx_logits.npy")], capture_output=True, text=True)
        print(c.stdout.strip(), c.stderr.strip()[-300:], flush=True)
        results["logits"] = c.stdout.strip()
        if c.returncode != 0:
            fail("logits cosine/argmax gate")

with open(os.path.join(a.out, "results.jsonl"), "a") as f:
    f.write(json.dumps(results) + "\n")
print("GATE", "PASS" if results["ok"] else "FAIL", flush=True)
sys.exit(0 if results["ok"] else 1)
