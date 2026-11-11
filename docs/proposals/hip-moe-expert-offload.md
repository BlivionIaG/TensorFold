# MoE expert offload on the Vega 56 (gfx900) over Thunderbolt: Qwen3.6-35B-A3B MLX 4-bit

Branch `feat/moe-expert-offload` on `vega-macos-up`. The 35B-A3B checkpoint is 19.0 GiB (16.9 GiB of routed experts, 40 layers x 256, 1.69 MiB each); the card has 7.9 GiB.

## The limit that shaped the design (measured 2026-11-11)

The card can map about 1.96 GiB of Mac memory in total (TinyGPU dext, `PrepareForDMA` answers `kIOReturnNoSpace`), whatever the block size
(30 x 64 MiB, 15 x 128, 7 x 256, 4 x 432, one 1900 MiB all succeed on a fresh server; 2000 MiB or +100 MiB fail), with the Mac idle. The first design
(17 GiB pinned host tier read in place by the GPU) therefore cannot exist. Written up in `ut4g-usbgpu/docs/HIPMAC.md` (Known limits).

## What was built

Three tiers: experts in pageable Mac memory (16.9 GiB, packed per expert), a 1.5 GiB pinned window the card reads in place, and VRAM cache lines
(4.74 GiB = 2876 lines, 28 %, LRU, core/expert_offload.zig = the CUDA EXL3 host cache's policy: Decision, Policy, Line, Vram, stage, schedule, reap,
touch, observe, seedHottest). Per layer, in the stream (eager, no graph: value waits cannot be captured): router/select/route -> `tf_expert_trace` (picks to
the window) -> `hipStreamWriteValue32(req)` -> `hipStreamWaitValue32(ack)` -> expert GEMVs. A host worker thread spins on `req`, runs the policy, copies the
missing experts' blobs into window slots (two halves alternate by layer), writes the layer's pointer-table row (in the window), stores `ack`. The pointer-table
GEMV/GEMM reads (gfx900 only, `TF_EXPERT_PTRS`) take a VRAM line (hit) or a window slot (miss). Admissions (`TF_EXPERT_COPIES` a round, default 8) are SDMA copies
window -> line queued at the round's end. `TF_EXPERT_OFFLOAD=0|1|auto`, `--ssd-experts GIB`, `TF_EXPERT_WINDOW_MIB`.

## Results (Vega 56, DPM auto, hipmac 16dde97+f656643)

* Correctness: greedy tokens vs mlx_lm (6 prompts, 8-96 tokens, `tools/vega-macos/moe_ref.py` / `moe_gate.py`): 3 of 6 identical (incl. a 96-token one);
  the other 3 diverge at tokens where mlx_lm's own top-2 are within 0 / 0.25 / 0.37 nats (near ties; bf16 noise). Cache on and `TF_EXPERT_COPIES=0` are token-identical on
  all 6 (so no L2 staleness after SDMA into reused lines).
* Decode (one stream, 200 tokens, COPIES=8): 4.4-6.6 tok/s, hit rate 36-42 %, 190-240 misses/token (315-400 MiB/token over the link), round 110-150 ms/token;
  the host stage is 10-21 ms/token, the rest is the card reading misses at link speed (about 3.7 GB/s): decode is link-bound, and tok/s scales with 1/(1 - hit rate).
* Prefill: 25 tokens 3.6 s; a span reads every non-resident expert of every layer (about 12 GiB of link traffic per 1024-token span).

## Not done

serve_bench tables (1k/512, 16k/1k at c=1,2,4,8), copy-budget tuning, prefill prestage of layer L+1, seeding the cache from prefill counts, Metal miss path.

## Update: admissions 32 a round, value-wait timeout, serve_bench 1k/512

`TF_EXPERT_COPIES` default is now 32 (8 gave 36-42 % hits and 4.4-6.6 tok/s; 32 gives 75-91 % and 11-22 tok/s on story / salted-filler prompts). The 10 s hipmac
value-wait timeout counts from submission and a forward queues every layer's wait at once, so a prompt span longer than ~10 s was released early (HipFailed): the store now
sets `HIPMAC_WAIT_VALUE_TIMEOUT_MS=0`. serve_bench 1k in / 512 out, c = 1, 2, 4, 8, one round a cell: `hip-moe-expert-offload-bench-1k.md` (c=1: prefill 30 tok/s, TTFT 34 s,
decode 11.0 tok/s, ITL 120 ms; aggregate decode 8.6 / 8.1 / 7.9 tok/s at c = 2 / 4 / 8). Prefill is the weak spot: a prompt span's GEMM tiles read the window in place at a
fraction of link speed (460 tokens 23.8 s); the 16k/1k load was not run (about 14 min a 16k prefill at 19-30 tok/s). Next: stage a span's experts into a VRAM scratch by SDMA
instead of reading them in place, prefill prestage of layer L+1, then the 16k rows.
