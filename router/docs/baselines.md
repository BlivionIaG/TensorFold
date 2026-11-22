# Baselines the router's presets come from

DeepSeek-V4.1-Flash (EXL3 2.9 bpw), measured 2026-11-11 through each server's own API with a stdlib load generator, thinking off, greedy,
unique prompt prefix per request, client on the serving host. **S** = two DGX Sparks (TP=2, CUDA); **M** = one M5 Ultra (Metal, `serve --parallel N`).
Source comments in `tfrouter/config.py` and `tools/sim.py` cite this file by section number (`baselines.md s2`, `s4`, ...). These are the engines as
they were on that date, not today's: section 7 records a newer single-stream reading and what is stale.

## 1. Configuration

| item | S | M |
|---|---|---|
| slots | 4 streams, a 5th queues | `--parallel 4`; throughput stops growing at 4 slots |
| drafts | on, acceptance 47-60 %, 1.7-1.9 tokens per event | auto-off after 8 rounds in this build (accepted 4/10), 1 token per event |
| prompt processing | 2048-row chunks, 250 ms of decode rounds between two chunks, one prompt at a time | one prompt at a time, decode of the other slots stops |
| max context | 1,048,576 (model window), pool shared by the streams | accepted to 1,048,576, booted at 262,144, 4 slots tested at 40,960 |
| prompt cache | 512 MiB | 4 GiB |

## 2. Prefill, one request at a time

| prompt tokens | S tok/s (median) | S TTFT s | M tok/s (median) | M TTFT s |
|---|---|---|---|---|
| ~1k | 1242 | 0.84 | 647 | 1.62 |
| ~4k | 1803 | 2.22 | 914 | 4.58 |
| ~8k | 1894 | 4.47 | 948 | 8.74 |
| ~16k | 1969 | 8.49 | 964 | 17.05 |
| ~32k | 2015 | 16.36 | 973 | 34.09 |
| ~64k | 2022 | 33.6 | 967 | 67.2 |
| ~128k | 1920 | 69.8 | 913 | 144.5 |

First token for a 39-token prompt: S 0.15-0.16 s, M 0.72-0.78 s (a constant ~0.7 s).

## 3. Decode, one stream, 256 new tokens

| context before | S tok/s | S ITL p50 / p95 ms | M tok/s | M ITL p50 / p95 ms |
|---|---|---|---|---|
| ~40 | 58.2 | 30.8 / 37.1 | 39.1 | 25.2 / 25.9 |
| ~8-9k | 56.6 | 31.4 / 37.9 | 37.3 | 26.3 / 27.8 |
| ~33-37k | 53.1 | 31.4 / 38.2 | 35.9 | 26.6 / 46.5 |
| ~132-147k | 53.1 | 31.9 / 39.1 | 33.7 | 27.8 / 54.9 |

## 4. Concurrency, aggregate decode tok/s (short prompts)

| streams | S | M `--parallel 1` | M `--parallel 2` | M `--parallel 4` | M `--parallel 8` |
|---|---|---|---|---|---|
| 1 | 59 | 38 | 39 | 39 | |
| 2 | 82 | 35 | 48 | 48 | |
| 4 | 110 (ITL p50 56.5 ms, worst gap 0.5 s) | 33 | 46 | 53 (worst gap 2.3 s) | |
| 6 | | | | | 45 |
| 8 | 103 (4 run, 4 queue 5-6 s) | | | 45 (4 queue 12-15 s) | 52 |

One M decode round of N rows costs 25 / 39 / 66 / 104 / 135 ms for N = 1 / 2 / 4 / 6 / 8, so M saturates near 60 tok/s from 4 slots.
S round time: 30.8 / 41.3 / 56.5 ms at 1 / 2 / 4 streams, 1.79 / 1.69 / 1.55 tokens per stream per round.

Aggregate prefill under concurrency is flat at the single-request rate (S 1.75k tok/s, M 0.90k): prompts are served one at a time, FIFO.
Four 4k prompts at once on S: TTFT 2.2, 4.4, 7.0, 9.6 s; on M `--parallel 4`: 5.6, 10.4, 15.4, 20.3 s.

## 5. Decode of a running stream while another request's long prompt prefills

| setup | newcomer prompt | newcomer TTFT | running stream ITL p95 during | longest stall | running stream rate during |
|---|---|---|---|---|---|
| S | 8.7k | 5.65 s (4.0-4.5 alone) | 1.06 s | 1.2 s | ~10 tok/s (alone 50-58) |
| S | 16.9k | 10.85 s (8.5 alone) | 1.06 s | 1.2 s | ~10 tok/s |
| M `--parallel 2/4` | 9.0-9.2k | 9.6-9.8 s | no events | 9.6-9.8 s | 0 |
| M `--parallel 2/4` | 17.1-17.4k | 17.7-18.3 s | no events | 17.7-18.3 s | 0 |

S shows a burst of ~6-8 events every ~1.25 s (one 2048-row chunk, then 250 ms of decode). M stops every running stream for the whole prefill.

## 6. Context, slots, memory

| | S | M |
|---|---|---|
| served here | prompts to 134k (TTFT 69.8 s), decode at 132k 53 tok/s | prompts to 147k (TTFT 172 s), decode 33.7 tok/s |
| KV per token | ~1.78 KB for the pair (replicated pool, preallocated) | 1402 B; each extra slot costs ~11.4 MB + 890 B x context |
| weights | ~105 GiB per rank + 12 GiB reserve | 196 GiB resident, ~13 GB headroom under the load limit |
| concurrent sequences | 4 | tested 1, 2, 4, 8 |

Power (GPU-side only): S ~156 W prefill, ~126 W decode; M ~138 W prefill, ~104 W decode.

## 7. What is stale (read before trusting the presets)

* These are the engines of 2026-11-11. A later single-stream reading with the fork's own CUDA engine on the pair (`exact` mode, drafts on, 256 tokens,
  greedy, 15-token prompt) gave 42.5 tok/s, 0.156 s first token, 89 of 238 drafts accepted; the M server with drafts off gave 53.9 tok/s.
  So the decode ordering of section 3 (S 58 vs M 39) no longer holds: both engines are within about 25 % of each other.
* The cross-machine link is not a design input for the two-server router (every request stays on one machine). For the mixed lanes (tier 2) it is:
  Spark-to-Spark 200G, Mac-to-Spark 100G cables behind a Thunderbolt-5 PCIe x4 tunnel (writes ~29 Gbit/s, reads ~50 Gbit/s, 4 KiB round trip 17 us);
  measured hand-offs reach 0.4-1 GB/s including the exporter's build time.
* `docs/fleet-serving.md` phase 0 re-measures sections 3-5 on the current engines and regenerates the presets from it.
