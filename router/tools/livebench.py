#!/usr/bin/env python3
"""livebench.py: replay a simulator workload (chat / RAG / agent / long-doc / mixed) in real time against REAL servers or the router.

    python3 tools/livebench.py --target 127.0.0.1:8000 --label R --wl chat --rate 0.12 --horizon 300 --corpus corpus.txt \
        --model DeepSeek-V4.1-Flash --key-file keys.txt --json r.json

Run it once per setup (A = the Sparks server, B = the M5 server, R = the router) with the same --wl/--rate/--seed/--horizon: the trace is the
simulator's generator with that seed, so the three runs see identical arrivals, prompt sizes and output lengths.
Prompts are paragraphs of --corpus behind a nonce and end with a request for a long answer. The text differs in every run (--salt, default the clock) so
no run is served from a prefix cache another run filled; temperature 0, thinking off, so the greedy-only M5 and the Sparks serve the same kind of
request. The summary is the simulator's own (TTFT, ITL, E2E, goodput at the workload's SLO).
"""
import argparse
import asyncio
import json
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import replay                                                    # noqa: E402
from llmbench import Corpus, story_prompt                        # noqa: E402


class CorpusPrompter(object):
    """chat messages of about n_target prompt tokens built from corpus paragraphs (chars = tokens x cpt)"""

    def __init__(self, corpus, cpt, salt):
        self.corp, self.cpt, self.salt = corpus, cpt, salt
        self.corp.cpt = cpt

    def _rng(self, rng):
        """same trace, other text in every run: a prefix cache that one setup filled must not serve the next setup"""
        return random.Random(rng.randrange(1 << 62) ^ self.salt)

    def first(self, n_target, rng):
        rng = self._rng(rng)
        material = self.corp.prompt(n_target, "lb", rng).rsplit("\n\nQuestion:", 1)[0]      # the corpus' own question would end the reply after one word
        return [{"role": "user", "content": material + "\n\n" + story_prompt(rng.randrange(1000))}]

    def next_turn(self, hist, reply, n_target, rng):
        rng = self._rng(rng)
        msgs = hist + [{"role": "assistant", "content": reply}]
        have = sum(len(m["content"]) for m in msgs)
        nchars = max(200, int(n_target * self.cpt) - have)
        msgs.append({"role": "user", "content": "More material:\n\n" + self.corp.body(nchars, rng) + "\n\n" + story_prompt(rng.randrange(1000))})
        return msgs


def read_key(path):
    if not path:
        return None
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if line and not line.startswith("#"):
            return line.split(":", 1)[-1].strip()
    return None


async def amain(a):
    host, _, port = a.target.rpartition(":")
    client = replay.Client(int(port), 1.0, a.model, replay.class_of(a.wl), host=host or "127.0.0.1", key=read_key(a.key_file),
                           temperature=a.temperature, extra={"chat_template_kwargs": {"enable_thinking": False}})
    prompter = CorpusPrompter(Corpus(a.corpus), a.cpt, a.salt)
    results = await replay.run_workload(client, a.wl, a.rate, a.seed, a.horizon, a.drain, prompter)
    stats = {n: {"busy_pf_v": 0.0, "busy_dec_v": 0.0} for n in ("S", "M")}
    summary = replay.summarize(a.wl, results, stats, a.horizon)
    split = replay.backend_split(results, a.horizon)
    summary["M_share"] = split["M_share"]
    replay.print_table("%s: %s @ %g/s seed %d horizon %g s (live, real time)" % (a.label, a.wl, a.rate, a.seed, a.horizon), [(a.label, summary)])
    print("errors %d, unfinished %d, of %d requests" % (split["errors"], split["unfinished"], len(results)))
    if a.json:
        rows = [{"rid": str(r.rid), "t_arr": r.t_arr, "t_first": r.t_first, "t_done": r.t_done, "n_in": r.n_in, "n_out": r.n_out, "tokens": r.tokens,
                 "cached": r.cached, "backend": r.backend, "status": r.status, "err": r.err, "gaps_max": max(r.gaps) if r.gaps else None} for r in results]
        json.dump({"label": a.label, "wl": a.wl, "rate": a.rate, "seed": a.seed, "horizon": a.horizon, "summary": summary, "split": split, "requests": rows},
                  open(a.json, "w"), default=str)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, help="host:port of a server or of the router")
    ap.add_argument("--label", default="X")
    ap.add_argument("--model", default="DeepSeek-V4.1-Flash")
    ap.add_argument("--key-file", default=None)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--wl", default="chat")
    ap.add_argument("--rate", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=1, help="the trace: arrivals, prompt sizes, output lengths")
    ap.add_argument("--salt", type=int, default=None, help="prompt text (default: the clock, so every run has fresh text and no cached prefix)")
    ap.add_argument("--horizon", type=float, default=300.0)
    ap.add_argument("--drain", type=float, default=300.0)
    ap.add_argument("--cpt", type=float, default=3.3, help="chars per token of the corpus text (llmbench.py calib measures it)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    if a.salt is None:
        a.salt = int(time.time() * 1000)
    print("salt %d" % a.salt)
    asyncio.run(amain(a))


if __name__ == "__main__":
    main()
