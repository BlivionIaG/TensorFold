# Serving DeepSeek-V4.1-Flash on two Sparks and an M5 Ultra together

Goal: better aggregate throughput, TTFT, decode stalls, cached-turn latency and availability than either machine alone, without changing the tokens
a request gets. Fork-only; nothing here is proposed upstream, which declined any mode that changes output tokens.

## 1. Machines and link

| | Sparks pair (S) | M5 Ultra (M) |
|---|---|---|
| role | TP=2 CUDA server | Metal server |
| prefill | ~2.0k tok/s (1.2k at 1k) | ~0.97k tok/s (0.65k at 1k) |
| decode, 1 stream | 42.5 tok/s (drafts on, 89 of 238 accepted) | 53.9 tok/s (drafts off) |
| decode, aggregate | 110 tok/s at 4 streams (older engine) | 60 tok/s from 4 slots |
| running decoders when a long prompt arrives | ~10 tok/s, worst stall 1.2 s | stop for the whole prefill (2.5 s worst gap at 4k chunks, ~10 s at 9k) |

Link: Spark to Spark is 200G (NCCL round trip 6 us). Each Spark has a 100G cable to the Mac, but the Mac end is a Thunderbolt-5 PCIe Gen4 x4 tunnel:
writes ~29 Gbit/s, reads ~50 Gbit/s, 4 KiB round trip 17 us, measured hand-offs 0.4-1 GB/s including the exporter's build. State is small
(3.7 KB per token, a 32k prompt is ~120 MB), so moving it is cheap; what stays unsolved across engines is exactness, not bandwidth.
Older figures are in `router/docs/baselines.md`; section 7 there says which are stale.

## 2. Principle: a request stays on one engine

Splitting one request across CUDA and Metal gives a third numeric that equals neither engine. Measured: a Metal-landed prefix from a CUDA producer leaves the
local tokens on 1 prompt of 6; even the Metal round trip through the neutral bf16 format has logit NRMS 0.08-0.11 (worst 0.22-0.69). So:

* **Tier 1, exact, default:** whole requests are routed; every reply is what that engine alone would have produced.
* **Tier 2, labelled, opt-in per request:** lanes that cross engines. Reply carries which lane ran; clients and the router can pin `exact`.

The model's own speculative decoding stays exact (every draft is verified), so draft-acceptance work is a tier-1 lever.

## 3. Architecture

```
clients -> router: cost model, prefix map, admission, health -+-> S  (whole request)
                                                              +-> M  (whole request)
                                                              +-> tier-2 lanes only for requests that opt in
```

`router/` is the existing stdlib-only OpenAI-compatible proxy (`python3 -m tfrouter`). Per request it predicts queue wait + prefill of uncached rows +
decode at the forecast load on each backend, adds an interference term for the decoders it would stall, and picks the cheapest healthy backend.
Prefix affinity keeps a conversation on the engine holding its cache; admission control holds the excess in the router (interactive before background)
and answers 503 + Retry-After when both are full; a circuit breaker fails over before the first byte. `X-Route-To: S|M` pins an engine.
Details, config and the offline validation tables are in `router/README.md`.

## 4. What is already built and what is evidenced

| item | state |
|---|---|
| router, admission, affinity, breaker, failover, explain/shadow/dry-run | built, 147 offline tests pass (`cd router && python3 -m unittest discover -s tests -t .`) |
| capability gate: sampled, grammar/logprobs and image requests stay off the greedy-only M | fixed in this PR: it only ran with the hand-off on, so with it off (the default) such a request could reach M and be served greedy |
| cost model checked against the simulator | `tests/test_core.py` asserts equal predictions on live simulator states |
| DES: chat @0.12 req/s TTFT p95 17.0 s (S alone) vs 3.0 s routed; agent @0.05 goodput 4.8 vs 15.6 req/min | simulator, 6 seeds x 900 s |
| live gate E2 (2026-11-11): chat, one run per rate | thin, mixed: goodput 0.109 vs 0.048 req/s at 0.12 (2.3x), but 0.064 vs 0.085 at 0.09, tie at saturation; ITL p95 1.4 s at 0.15 |
| admission control, relative primary bias, output warm-up | added after E2, DES only, not re-run live |
| presets | `router/presets/current-engines.json`, measured 2026-11-22 on today's engines (`router/docs/phase0.md`); the built-in tables stay the Nov-11 values the simulator and fakes use |
| phase 0, current engines measured | done: M's prefill is ~1.8x the Nov-11 figure and close to S at 16k+, M's drafts do not fire, a running stream stalls ~2.9 s on M and ~1.2 s on S behind a long prompt |
| phase 1, router live on both engines (chat 0.12 / 0.25, mixed 0.10, one run per setup) | done once: TTFT p50 2.1 vs 14.4 / 27.5 s, 100 % finished at 0.25 req/s where S / M alone finish 73 / 59 %, failover with no error; one trace and one seed per point, see `router/docs/phase0.md` for what it does not show |

Nothing here claims more than those runs: one seed per point, 4-5 minute traces, a workstation client, admission control at its defaults.

## 5. Plan and gates

Each phase ends with numbers in this file or in the PR; a gate that fails kills or reshapes the next phase.

**Phase 0, measure current engines (exact, no code).** Done 2026-11-22, see `router/docs/phase0.md`: per engine prefill by size, decode at 1/2/4 streams, decode ITL while a 4k/16k prompt
arrives, concurrent prefill; presets generated by `router/tools/presets_from_bench.py`.
Gate: the cost model's predicted TTFT within 20 % of measured. Holds for most class medians (2-22 %), not per request (median error 17-22 %).

**Phase 1, router live (tier 1).** Done once for chat 0.12 / 0.25 and mixed 0.10 against each engine alone (not yet round-robin, and shadow mode was only a smoke request).
Gate: at least 1.4x the better single engine's goodput where one engine misses its SLO (met: 3.5 vs 1.2 good req/min at 0.12, 3.2 vs 0.9 on mixed, 138 vs 99 tok/s and 100 % vs 73 % finished at 0.25),
ITL p95 not worse than the engine alone (below S alone, 60-71 against 74-78 ms, above M alone at 47-48 ms: not met against M), zero lost requests on failover (0 of 15; no stream was cut mid-answer, not covered).

**Phase 2, per-engine stall bounds (tier 1, engine work).** The M5 already runs one decode round between prefill chunks, so the stall is one chunk. Probed 2026-11-22 (`router/docs/phase0.md`): `TF_DSV41_PREFILL_STEP`
4096 / 2048 / 1024 gives 2.0 / 1.0 / 0.65 s per chunk for 8.5 / 9.0 / 12.2 s TTFT at 16k, so 2048 is a free halving and below that TTFT pays one for one. The worst gap stays 1.7-2.4 s at every step because the last gap
(ending at the newcomer's first token) is not chunk-bound; that is the engine work, plus the router's guard (`prefill.mode`, `chunk_rows`) following the chosen step.
Gate: worst decode gap under a 16k arrival below 0.5 s on M with TTFT of the newcomer not worse than 1.3x. Not met; the knob alone cannot meet it.

**Phase 3, M draft acceptance (tier 1, engine work).** Probed 2026-11-22: the drafts accept (1.7 tokens per round with the governor off, replies byte-identical) but a 3-4 row verify window costs 1.69x a one-row round on
Metal (31.3 vs 18.5 ms), so drafting nets 52.0 against 53.7 tok/s: it wins on lists / JSON / code (up to 59.6) and loses on prose (43.8). The governor is right to stay off by default. The M5 drafts with MTP + copies and the Sparks with DSpark + copies.
Gate: acceptance above break-even (1.75 tokens per round) on the benchmark prompts, or a documented reason it cannot be. The reason is documented; what would change it is a cheaper verify window on Metal or a stronger drafter, both engine work.

**Phase 4, cache and capacity (tier 1).** Sticky sessions, prefix-map self-correction from `cached_tokens`, load-aware thresholds from phase-0 curves,
`/v1/load` as the load signal, a `served_by` field in replies.
Gate: a multi-turn trace shows cached-turn TTFT near the engine's floor and no flip of a conversation between engines without cause.

**Phase 5, tier-2 lanes (labelled, only if phase 1-4 leave a gap).** In order of measured value:

1. Chunk-pipelined layer split for prompts of 32k and up: model says +15 % at 32k, +23 % at 64k over S alone, -20 % at 8k; host-tested only, needs one hardware run.
2. Prefill on S, decode on M as an opt-in (`prefill: force`, router path SM): isolation rather than speed; TTFT was at parity (1.01-1.07x of M alone).
3. 1M-class contexts: prefill on S (~9-10 min against ~19 min), decode on M.

These live on the fork's mixed lines, which sit 2,357 commits behind main with conflicts in 49 files (build files, lane host, prompt cache, server CLI,
environment table, Metal sources), so they come as separate stacked PRs per lane after phase 1 shows whether a gap remains.
Not planned: heterogeneous expert/tensor parallel across engines and attention/FFN disaggregation (modelled at +5 to +20 % single stream, below the
replica sum on aggregate, 20-30 engineer-weeks), cross-machine speculative decoding (a per-round hop costs more than it saves).

## 6. Expected effect

All estimates; replaced by measurements as phases land.

| metric | one engine | target |
|---|---|---|
| aggregate decode | S 110 or M 60 tok/s | sum at the load where both are busy, about 1.5x S alone (measured 110 + 60 on older engines) |
| TTFT under load | queues behind one prefill FIFO | about half at 2x load, by balancing |
| decode stall during a long prefill | M 2.5-10 s, S 1.2 s | below 0.5 s on both (phase 2) |
| cached turns | engine-local | near the engine's floor via affinity |
| availability | one machine | either machine can take everything |

## 7. Risks

* Capacity limits differ: S pool at 4 slots vs M context at `--parallel 4`; the router excludes by context and reports 413/503 with the reason.
* A router hop adds one TCP connect per request on the LAN (connections are not kept alive); negligible against 0.15 s first token, measured in phase 1.
* Prompt-cache sizes are modelled, not observed; evictions the router did not cause cost one cold prefill.
* Sampled requests: M is greedy-only, so with tier 2 on they stay on S; in tier 1 the router sends sampling requests to S when M ignores temperature.
* The router is a single point of failure for the front door; it is stateless apart from the affinity table, so a second instance can take over.
