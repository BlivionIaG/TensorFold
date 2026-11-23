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
  curve, and a long prompt now costs a running stream one stall of ~2-2.9 s per 4096-row chunk (2.85 s on the 40k-context server with a 4 GiB prompt cache, 2.0 s on the 20k-context probe server) instead of the whole prefill. S keeps the lower first-token cost and the lower stall.
* **M's drafts do not pay.** The governor drops them after 8 rounds (10 drafted, 2 accepted per 256 tokens on story text; 1.11 tokens per event on natural prompts, where S gets ~3). The probe below shows
  why: they accept at ~1.7 tokens per round but the verify window costs 1.69x a plain round, so always-on drafting nets 52.0 against 53.7 tok/s.
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

# Probes for phases 2 and 3 on the M5 (2026-11-22)

Same engine and pack, private server, 1 stream unless noted, greedy. **Engine lines:** `sd2` (Sparks) and `fin2` (M5) are builds of the fork's dev line (`fin2` = `ace515402`, "Merge the streamed
hand-off into the dev line"), not of `main`: `main` has no `--engram-stream-dir` and no `native/deepseek_v41_lane_host.zig`. The router does not care, but every engine number in this file is for those builds.

**Draft governor (phase 3).** The M5 drafts with MTP + copies (the Sparks with DSpark + copies). Its governor samples the first 8 rounds of a request and drops to plain decode below 1.75 tokens per round.
Eight natural prompts, 256 tokens, reply hashes identical in both runs (drafting is verified):

| M5 | tok/s | tokens per event | event ms p50 |
|---|---|---|---|
| default (governor on) | 53.7 | 1.11 | 18.5 |
| `TF_DSV41_DRAFT_OFF=0` (drafts always on) | 52.0 | 1.70 | 31.3 |

Per prompt with drafts on: 43.8 to 59.6 tok/s (prime list 59.6, JSON 58.0, C function 54.7, history essay 43.8). The governor's log lines show first-draft acceptance 0.29-0.69 and 1.4-1.75 tokens per round at round 8.
So the drafts are not failing to accept (1.7 tokens per round) but a 3-4 row verify window costs 1.69x a one-row round (+13 ms on 18.5 ms), which cancels the gain; it pays on lists, JSON and code
and loses on prose. [INFERENCE] The earlier "accepts 2 of 10" counted only the rounds the governor allowed. What would change this is a cheaper verify window on Metal or the stronger drafter the Sparks use, not the governor.
Re-arming the governor mid-request would gain on structured output and lose the 8-round probe elsewhere; not tried.

**Prefill step (phase 2).** `TF_DSV41_PREFILL_STEP` on the M5, a running stream while a 16k prompt prefills (the stream gets one decode event per chunk):

| step | stall per chunk | worst gap (16k arrival) | TTFT 4k / 16k alone |
|---|---|---|---|
| 4096 (used above) | 2.0 s | 1.99 s | 2.8 / 8.5 s |
| 2048 | 1.0 s | 1.73 s | 3.1 / 9.0 s |
| 1024 | 0.65 s | 1.91 s | 4.4 / 12.2 s |

With drafts and taps off at step 2048 the gaps are the same (worst 1.57 / 1.98 s). A step of 2048 halves the stall for +6 % TTFT at 16k, 1024 costs +43 %. [INFERENCE] Per-chunk cost does not shrink with the chunk (a MoE
chunk reads every expert), so smaller steps trade TTFT for stall about one for one. The worst gap did not follow the step: it was the last gap, which ends at the newcomer's first token and was 1.7-2.4 s at every step.

**The last gap (phase 2 engine work).** Lane trace at step 2048, 16.9k-token prompt: the 2048-row chunks take about 0.96 s each (the cone prefill runs only the encoder layers on non-final chunks), but the
last call, with 475 rows, took 1.79 s. Final-call time against rows: 73 rows 0.49 s, 688 rows 1.07 s, 1915 rows 1.99 s. The cone prefill's last chunk also replays the decoder layers over about 2.7k rows
in layer-group programs, all in one call: roughly 0.8-1.0 s of fixed cost per long prompt that no knob shortens. Fix (fork dev line, branch `perf/dsv41-cone-replay-yield`, `TF_DSV41_CONE_YIELD`, default 1): the last chunk runs its
encoder half and the replay runs one program per step call, so the other streams' decode rounds run between programs (the five programs take 0.37 / 0.24 / 0.17 / 0.13 / 0.10 s). Same programs in the same order.

| step | worst gap, before | worst gap, yield | tokens emitted during the prefill, before / yield | newcomer TTFT 16k, before / yield |
|---|---|---|---|---|
| 4096 | 2.0-2.3 s | 2.0 s | 4-5 / 9-10 | 7.9-9.2 / 8.2-9.3 s |
| 2048 | 1.69-1.81 s | 1.13-1.17 s | 8-9 / 13-14 | 8.9-9.9 / 9.0-9.8 s |
| 1024 | not run | 0.71-0.78 s | not run / 21-22 | not run / 11.9-12.5 s |

Three repetitions each, one 16k arrival over one decoding stream. Exactness: four interleaved streams (a decoder, a 16k and an 8k prompt arriving 0.7 s apart so their runs park behind each other, and a short
one) give byte-identical replies on the previous build and on the new build, each with the governor and with drafts forced on. The gap that remains is one encoder chunk, so the gate
stays unmet at 1.3x TTFT. Use 2048 with the yield (1.1 s gaps, TTFT unchanged); 1024 gives 0.7 s gaps for about +35 % TTFT. Step 512 was not run.

**Decode share (`TENSORFOLD_PREFILL_SHARE=p`, the lane host's existing knob; a step of t owes the decoders t(1-p)/p of rounds).** Step 2048 with the yield, one 16k arrival, 3 repetitions:

| p | worst gap | tokens the decoder emits during the prefill | newcomer TTFT 16k |
|---|---|---|---|
| 1 (default) | 1.13-1.15 s | 13-14 | 9.1-10.0 s |
| 0.9 | 1.12-1.34 s | 58-65 | 10.1-11.0 s |
| 0.8 | 1.11-1.21 s | 122-134 | 11.2-12.2 s |
| 0.7 | 1.15-1.19 s | 212-228 | 12.8-13.8 s |
| 0.6 | 1.16-1.17 s | 339-350 | 15.6-16.2 s |

The share moves the decoder's rate during a long prefill from about 1.4 to 6 / 11 / 17 / 21 tok/s and costs the newcomer 10 / 22 / 40 / 70 % TTFT; it never shortens the worst gap (one chunk).
At step 1024 the same knob costs more (0.8: 15.1 s TTFT at 16k) for 0.70-0.74 s gaps, 0.5: 22.9 s.

**Router live, new M engine (2026-11-23, one seed each, 240 s, client on the workstation).** M = the fork dev line with the yield, step 2048, context 40960; S = the pair, parity profile.

| run | TTFT p50 / p95 | ITL p50 / p95 ms | out tok/s | M share | finished |
|---|---|---|---|---|---|
| chat 0.25, previous M (step 4096) | 19.1 / n.a. | n.a. / 71 | 138 | n.a. | 100 % |
| chat 0.25, yield + step 2048 | 22.2 / 40.8 | 48 / 73 | 138 | 43 % | 100 % |
| chat 0.25, yield + step 2048 + share 0.8 | 21.9 / 41.1 | 47 / 73 | 138 | 45 % | 100 % |
| chat 0.25, `policy.mode` ll (predicted TTFT only) | 21.2 / 44.0 | 48 / 73 | 138 | 46 % | 100 % |
| mixed 0.10, previous M | 3.9 / n.a. | n.a. / 60 | n.a. | n.a. | 100 % |
| mixed 0.10, yield + step 2048 | 3.8 / 30.4 | 40 / 52 | 31 | 33 % | 100 % |
| mixed 0.10, yield + step 2048 + share 0.8 | 5.4 / 31.1 | 33 / 66 | 31 | 28 % | 100 % |

Read: at these loads the router's behaviour is set by what S and M can sustain together (138 tok/s at 0.25 either way), so the M engine changes are within one-seed noise on TTFT and visible only in
ITL p95 on the mixed workload (60 to 52 ms). The share does not pay under the router: it lengthens the long prompts the router sends to M (mixed p50 3.8 to 5.4 s) and the router's model does not know it,
while ITL p95 does not improve. Keep the share at 1. The cost mode is not worse than least-loaded (ll: p95 44.0 against 40.8 s). The previous-M rows are the phase-1 runs (different boot of S, different salt).
