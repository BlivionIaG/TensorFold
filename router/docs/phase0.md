# Phase 0: the current engines, measured (2026-11-22)

Exact, greedy, thinking off, unique prompt per request, `tools/llmbench.py` on each serving host through loopback (python stdlib), the Sparks pair with
4 slots and the M5 with `--parallel 4`. Raw per-request records are not shipped (they hold per-event arrays); the tables below and
`presets/current-engines.json` are what was derived from them.

| | S: Sparks pair, CUDA engine `sd2`, 4 slots | M: M5 Ultra, Metal engine `fin2`, `--parallel 4` |
|---|---|---|
| context served | 131072 | 40960 |
| first token, 39-token prompt | 0.23 s | 0.73 s |
| prefill tok/s at 1k / 4k / 8k / 16k / 32k | 1.22k / 1.79k / 1.94k / 2.05k / 2.04k | 0.77k / 1.42k / 1.38k / 1.67k / 2.0k |
| TTFT at 4k / 8k / 16k / 32k | 2.4 / 4.3 / 8.4 / 16.6 s | 2.8 / 5.8 / 9.9 / 18.3 s |
| decode, 1 stream, story prompt (ctx ~0 / 8k / 32k) | 45 / 44 / 43 tok/s | 53 / 50 / 47 tok/s |
| tokens per stream event (story prompt) | 1.5 | 1.0 |
| tokens per stream event, 8 natural prompts | ~3.0 (earlier sd2 runs; 72 tok/s) | 1.11 (53.5 tok/s) |
| aggregate decode, 2 / 4 streams (story prompts) | 65 / 91 tok/s | 66 / 79 tok/s (94 once all four decode) |
| event ms at 1 / 2 / 4 streams | 33 / 41 / 61 | 18.5 / 27 / 42 |
| aggregate prefill, 2 and 4 simultaneous 4k prompts | 1.77k / 1.74k tok/s (one at a time) | 1.31k / 1.24k tok/s (one at a time) |
| a running stream while another request prefills 8k / 16k | worst gap 1.1 / 1.2 s, ~14-17 tok/s kept | worst gap 2.85 / 2.86 s, 3 / 6 events in the whole prefill |

Findings that change the plan:

* **The machines are closer than the November tables.** M's prefill at 16k is ~1.8x faster (about 0.9k to 1.67k tok/s; 9.9 s against 17.7 s to first token), its decode is 53 tok/s with a flat
  curve, and a long prompt now costs a running stream one stall of ~2.9 s per 4096-row chunk instead of the whole prefill. S keeps the lower first-token cost and the lower stall.
* **M's drafts do not fire.** 10 drafted, 2 accepted per 256 tokens on story text; 1.11 tokens per event on natural prompts, where S gets ~3. With drafts working M would
  be well past S on single-stream decode; this is the phase-3 lever and it is exact.
* **The `popt3` engine (replay producer, prefill 2.5k tok/s at 32k) failed two simultaneous short decode streams** with `HistoryLost` in the shared round ("a round failed (HistoryLost):
  the round's streams fail"); `sd2` ran 1/2/4 streams cleanly, so the pair served with `sd2`. The `prod` profile's `TF_DSV41_EXPERT_TOPP=0.85` prunes routed experts and is not exact;
  the run uses the parity profile that drops it. Both are engine findings for whoever owns those lines, not router work.

# Phase 1: the router in front of both engines, live (2026-11-22)

`tools/livebench.py`, the simulator's `chat` and `mixed` workloads in real time (seed 1, 240 s of arrivals, 120 s drain), greedy, thinking off, fresh prompt text in every run
(`--salt`; a first attempt reused identical text and the router's run hit the prefix caches the two single-engine runs had just filled, so it was thrown away).
A = the Sparks server alone, B = the M5 server alone, R = the router in front of both (`presets/current-engines.json`, defaults otherwise). Client on the workstation, S over
the LAN, M over an ssh tunnel (all three setups pay the same path).

| workload | setup | TTFT p50 / p95 s | ITL p50 / p95 ms | E2E p50 / p95 s | out tok/s | finished | M share |
|---|---|---|---|---|---|---|---|
| chat 0.12 req/s (36 req) | A | 14.4 / 30.7 | 63 / 74 | 34.5 / 51.4 | 61.9 | 100 % | |
| | B | 27.5 / 56.4 | 45 / 48 | 51.9 / 78.2 | 61.9 | 100 % | |
| | **R** | **2.1 / 4.5** | 32 / 68 | **15.8 / 27.9** | 61.9 | 100 % | 41 % |
| chat 0.25 req/s (66 req) | A | 81.1 / 160.5 | 66 / 78 | 110.7 / 182.2 | 99.1 | 73 % | |
| | B | 99.0 / 167.3 | 45 / 48 | 125.9 / 204.9 | 77.0 | 59 % | |
| | **R** | **19.1 / 37.8** | 47 / 71 | **46.5 / 69.9** | **138.3** | **100 %** | 45 % |
| mixed 0.10 req/s (26 req, prompts up to 37k) | A | 20.9 / 47.2 | 59 / 76 | 42.6 / 87.6 | 31.2 | 100 % | |
| | B | 26.0 / 45.7 | 45 / 47 | 55.3 / 115.1 | 31.2 | 100 % | |
| | **R** | **3.9 / 30.6** | 29 / 60 | **22.6 / 35.8** | 31.2 | 100 % | 33 % |

(Output tok/s is equal where every setup finished the same work in the same window; it only separates under overload. "Finished" counts requests that completed inside the drain.)

What the numbers say:

* Same trace, same engines: the router's TTFT p50 is 2.1 s against 14.4 / 27.5 s (S / M alone) at 0.12 req/s and 3.9 s against 20.9 / 26.0 s on the mixed trace. At 0.25 req/s, beyond
  either engine alone, 100 % finished against 73 % / 59 % and aggregate output was 138 tok/s against 99 / 77 (1.4x the better engine).
* ITL p95 is below S alone in all three runs (68 / 71 / 60 ms against 74 / 78 / 76) and above M alone (47-48 ms). [INFERENCE] The router puts most streams on S, whose four-stream rounds are
  slower per stream than M's plain rounds; an ITL term in the cost (not only TTFT / E2E) would trade some of the TTFT win for it. Not changed here.
* [INFERENCE] The single-engine TTFTs are high because a four-slot engine with ~470-token answers keeps its slots busy for 15-20 s each and arrival bursts exceed four. The simulator pointed the
  same way (chat 0.12 TTFT p95 17 s alone, 3 s routed; measured 30.7 s alone on S, 4.5 s routed).
* Prediction check (`tools/predcheck.py`: the router's explain log joined to the measured first tokens of the three R runs). Predicted against measured TTFT medians per class, M up to 4k
  tokens: 3.08 / 3.03 s, 2.56 / 2.49, 2.98 / 2.72; S up to 4k: 1.73 / 1.63, 1.78 / 1.55, 2.01 / 1.64; S 12k+: 15.2 / 17.5 (differences 2-22 %). Per request the median error is 17-22 % and
  47-56 % of requests are inside 20 %. The phase-0 gate (within 20 %) holds for most class medians, not yet per request; at 2-3 s TTFTs 0.4 s is already 20 %.
* Failover: M killed 61 s into a 180 s chat run (15 requests): all 11 requests that arrived afterwards were served by S with no error, the breaker opened for M (4 trips, 80 s cooldown),
  0 of 15 failed. No stream was in flight on M at the kill, so a stream cut mid-answer (which cannot be resumed) is not covered by this run; failover before the first byte is covered by the unit tests.
