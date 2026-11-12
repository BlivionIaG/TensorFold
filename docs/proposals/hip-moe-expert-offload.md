# MoE expert offload on the Vega 56 (gfx900) over Thunderbolt: Qwen3.6-35B-A3B MLX 4-bit

Branch `feat/moe-expert-offload` on `vega-macos-up`. The 35B-A3B checkpoint is 19.0 GiB (16.9 GiB of routed experts, 40 layers x 256, 1.69 MiB each); the card has 7.9 GiB.

## The limit that shaped the design (measured 2026-11-11)

The card can map about 2047 MiB of Mac memory in total, hipmac's own ~64 MiB among it (the Thunderbolt PCIe DART aperture, `dart-apciec* vm-size 0x7ff00000`; the dext's
`PrepareForDMA` answers `kIOReturnNoMemory`, 0xE00002BD, past it), whatever the block size, with the Mac idle. The first design
(17 GiB pinned host tier read in place by the GPU) therefore cannot exist. Written up in `ut4g-usbgpu/docs/DMA_LIMIT.md` and `HIPMAC.md` (Known limits).

## What was built

Three tiers: experts in pageable Mac memory (16.9 GiB, packed per expert), a 1.5 GiB pinned window the card reads in place, and VRAM cache lines
(4.74 GiB = 2876 lines, 28 %, LRU, core/expert_offload.zig = the CUDA EXL3 host cache's policy: Decision, Policy, Line, Vram, stage, schedule, reap,
touch, observe, seedHottest). Per layer, in the stream (eager, no graph: value waits cannot be captured): router/select/route -> `tf_expert_trace` (picks to
the window) -> `hipStreamWriteValue32(req)` -> `hipStreamWaitValue32(ack)` -> expert GEMVs. A host worker thread spins on `req`, runs the policy, copies the
missing experts' blobs into window slots (two halves alternate by layer), writes the layer's pointer-table row (in the window), stores `ack`. The pointer-table
GEMV/GEMM reads (gfx900 only, `TF_EXPERT_PTRS`) take a VRAM line (hit) or a window slot (miss). The first version admitted by SDMA copies window -> line (`TF_EXPERT_COPIES` a round);
see "Update 2" for what replaced it. `TF_EXPERT_OFFLOAD=0|1|auto`, `--ssd-experts GIB`, `TF_EXPERT_WINDOW_MIB`.

The pointer-table code is built into its own code object (group `affine_tiles_ptr`, `tiles/dot2_tiles_ptr.hip`, `-DTF_EXPERT_PTRS`) and only the routed launches of
a projection with a table (`Launcher.affineFor(ptab)`) take it. Built into every tile it cost a dense stack 5 %: Qwen3.5-4B decode 81.4 -> 77.0 tok/s on the Vega
(`tf-vega-gate`): 3.2 tok/s from the code (more VGPRs and lower occupancy in 150 of 347 dot2 tiles), 1.2 tok/s from the 24 bytes more of `Affine` argument every
launch carried. Raw data: ut4g-usbgpu `artifacts/zig-stack/regress4b/`.

## Results (Vega 56, DPM auto, hipmac 16dde97+f656643)

* Correctness: greedy tokens vs mlx_lm (6 prompts, 8-96 tokens, `tools/vega-macos/moe_ref.py` / `moe_gate.py`, now `tf-moe-gate`): 3 of 6 identical (incl. a 96-token one);
  the other 3 diverge at tokens where mlx_lm's own top-2 are within 0 / 0.25 / 0.37 nats (near ties; bf16 noise). Cache on and `TF_EXPERT_COPIES=0` are token-identical on
  all 6 (so no L2 staleness after SDMA into reused lines).
* Decode (one stream, 200 tokens, COPIES=8): 4.4-6.6 tok/s, hit rate 36-42 %, 190-240 misses/token (315-400 MiB/token over the link), round 110-150 ms/token;
  the host stage is 10-21 ms/token, the rest is the card reading misses at link speed (about 3.7 GB/s): decode is link-bound, and tok/s scales with 1/(1 - hit rate).
* Prefill: 25 tokens 3.6 s; a span reads every non-resident expert of every layer (about 12 GiB of link traffic per 1024-token span).

## Not done

16k/1k serve_bench rows at c=4 and c=8 (c=1 and c=2 are in the bench file: TTFT 98.7 s / 174.5 s, decode 11.5 / 6.6 tok/s agg; c=8 would leave the cache too small: a 17k stream's KV is ~0.6 GiB of VRAM), prefill prestage of layer L+1, a
router-lookahead prefetch, an in-kernel hit/miss split to hide the per-layer handshake, Metal miss path.

## Update: admissions 32 a round, value-wait timeout, serve_bench 1k/512

`TF_EXPERT_COPIES` default is now 32 (8 gave 36-42 % hits and 4.4-6.6 tok/s; 32 gives 75-91 % and 11-22 tok/s on story / salted-filler prompts). The 10 s hipmac
value-wait timeout counts from submission and a forward queues every layer's wait at once, so a prompt span longer than ~10 s was released early (HipFailed): the store now
sets `HIPMAC_WAIT_VALUE_TIMEOUT_MS=0`. serve_bench 1k in / 512 out, c = 1, 2, 4, 8, one round a cell: `hip-moe-expert-offload-bench-1k.md` (c=1: prefill 30 tok/s, TTFT 34 s,
decode 11.0 tok/s, ITL 120 ms; aggregate decode 8.6 / 8.1 / 7.9 tok/s at c = 2 / 4 / 8). Prefill is the weak spot: a prompt span's GEMM tiles read the window in place at a
fraction of link speed (460 tokens 23.8 s); the 16k/1k load was not run (about 14 min a 16k prefill at 19-30 tok/s). Next: stage a span's experts into a VRAM scratch by SDMA
instead of reading them in place, prefill prestage of layer L+1, then the 16k rows.

## Update 2: read-through admission, prompt-span staging, SLRU (commits c22ba92ac8, 4c671bbcc8, and the SLRU commit)

Where the time went (hipprof, `HIPPROF_EVENTS=1`, story decode, c=1, per token): with SDMA admissions 48 ms = kernels 42.7 + gaps 4.9; the two expert GEMV kernels were 31 ms of it
(gate_up avg 519 us vs 62 us min, down 268 vs 43): the cost of reading misses at link speed, plus the admission copies competing for the same link. After the change below:
36.7 ms = dense kernels ~11 (qkv 2.3, lm_head 0.8, router 1.2, out 1.2, select/tail/plan ~2.5, GDN rest), expert compute floor ~4.4, miss reads ~16.6 (27 misses x 0.6 ms: link-saturated,
~2.9 GB/s), per-layer handshake ~4.8 (about 120 us a layer: flag 19 us + host stage ~30 us + release/launch). Host stage is 1.2-2.5 ms/token and host memcpy (59 GB/s) is not a bottleneck.

* Read-through admission: the stream tile that reads a missing expert from the window also stores it into the cache line chosen by the host (`ptab` entries are 16 bytes: source, destination line).
  A miss's one trip over the link is its admission, so every miss is admitted (`TF_EXPERT_COPIES` default 4096); the SDMA admission machinery, its copy stream and its landing events are gone.
  Story decode 20.9 -> 26-28.7 tok/s (hit 88 -> 91 %, 44 -> 27 misses/token).
* Prompt spans: each non-resident expert a span names is copied window -> a one-layer VRAM scratch (256 experts, 453 MB) by SDMA (3.16-3.19 GB/s) before the layer's GEMMs, which then read VRAM.
  466-token prompt 23.8 -> 4.3-4.6 s; serve_bench 1k prefill 30 -> 154 tok/s, TTFT 34 -> 6.7 s. The scratch costs 256 of ~2875 lines and keeps the decode working set intact.
* SLRU (`TF_EXPERT_PROTECT`, default 75 %): a line hit again after its admission is protected; eviction takes the oldest unprotected line. Card A/B (serve_bench filler 1k/256, par 8): c=1 decode
  12.7 -> 14.1 tok/s (ITL 102 -> 91 ms), c=4 tie (16.4 both), c=8 decode-window hit 62.8 -> 80.1 % (128 -> 69 misses/token, round 218 -> 159 ms) though its aggregate figure is within noise.
* Frequency policies, by simulation first (`tools/vega-macos/expert_cache_sim.py`, now `tf-expert-cache-sim`, on the study's real traces, 2876 lines, cold start per prompt): LRU 82.9 %; SLRU 83.6 (+0.8); LFU aged W=10k 83.7 (fragile: W=5k +0.3, 40k -0.3);
  ARC +0.4; per-layer split -1.1; TinyLFU -0.02 to -5.4; 2Q worse. Seeding or warm start (cross-category profile, or prefill counts) changes only the first ~30 decode tokens (misses/token at tokens 0-8 / 8-32: cold 158 / 70,
  seeded 99 / 59, prefill-derived 57 / 51; identical afterwards): about 1300 reads (~0.7 s) per prompt, less than the ~1.6 s a 2876-line seed read costs, so it is not implemented. Static profiles transfer poorly across domains.
* Rejected without building: a second pinned in-place cache tier (a window-resident expert still costs link bytes to read in place; it would only save the host memcpy, which is cheap); speculative all-256 prestage with
  double buffering (moves more bytes than the picks-aware path at 466 rows); skipping the handshake on all-hit layers (~1.4 ms/token); router-lookahead prefetch of layer L+1 (estimated net gain 3-5 ms/token
  after the extra router launches and false-positive traffic, since the link is already ~45 % busy and misses are the experts a predictor finds hardest; needs a second handshake channel and SDMA into lines).
* Gates: cache on (default) vs `TF_EXPERT_COPIES=0` token-identical on 8 prompts (6 short + a 466- and a ~1100-token one, `moe_ref.py` regenerated with mlx_lm); vs mlx_lm the same bf16 near-ties
  (top-2 within 0-0.37 nats) diverge, including both long prompts after ~20 identical tokens.
