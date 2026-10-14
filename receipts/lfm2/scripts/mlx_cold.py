import sys, time, mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import stream_generate
from mlx_lm.sample_utils import make_sampler
m, tok = load(sys.argv[1])
ids = tok.encode(open(sys.argv[2]).read() * 20)
for n in [int(x) for x in sys.argv[3].split(",")]:
    last = None
    for r in stream_generate(m, tok, ids[:n], max_tokens=int(sys.argv[4]), sampler=make_sampler(0.0)):
        last = r
    print(n, "prompt %.1f tok/s (%.2f s)" % (last.prompt_tps, n / last.prompt_tps), "decode %.1f tok/s" % last.generation_tps, "peak %.1f GB" % last.peak_memory, flush=True)
    mx.clear_cache()
