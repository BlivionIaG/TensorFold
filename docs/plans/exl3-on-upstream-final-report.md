# EXL3 Qwen3.8-Flash-Next on upstream TensorFold v1.1.0: final report

Branch `exl3-on-upstream-wip` (staging branch `exl3-c-main`), based on upstream tag `v1.1.0`.
Machine: par1-cs13, two RTX 5060 Ti 16 GiB (sm_120), 251 GB RAM, pack
`Qwen3.8-Flash-Next-exl3-3.05bpw`. All measurements are test measurements on idle cards under a lock
wrapper, not benchmarks.

## What this adds

The EXL3 pack now runs through upstream's own Flash Next CUDA family and server (`tensorfold-native serve`),
on one card and at tp=2. Hooks in upstream files are one line each; the rest is new files under
`zig/src/families/flashnext/cuda_exl3_*.zig`, `zig/kernels/cuda/flashnext_exl3_*.cu` and `zig/tests/cuda/`.

| Area | What exists | Main files |
|---|---|---|
| Weights | EXL3 dense, shared, routed experts, 5-bit lm_head, PLE n-gram table read from the pack | `cuda_exl3_load.zig`, `cuda_exl3_store.zig`, `cuda_exl3_ple.zig` |
| Expert offload | Per-layer device line pool fed from a pinned host bank; LRU plus use count; misses copied on a side stream | `cuda_exl3_pool.zig`, `exl3_offload.zig`, `exl3_round.zig` |
| tp=2 | Expert parallel: rank 0 routed 0-255 (+MTP head), rank 1 routed 256-511 + shared expert; one fp32 all-reduce per layer; per-rank pools and half-banks; vocabulary-split lm_head | `cuda_exl3_tp.zig`, `flashnext_exl3_tp.cu` |
| Prefill | Expert-major deep chunks (thousands of rows), one ring of lines across all layers, 64-member tile kernel for the mul1 3-bit experts, tp=2 version with overlapped per-sub-block reduction, expert-major MTP head | `cuda_exl3_prefill*.zig`, `flashnext_exl3_prefill.cu` |
| MTP | Draft rows, drafts with the prompt cache at tp=2, rank 1 keeps no head cache in its windows | `cuda_exl3_mtp.zig`, `cuda_rank_engine.zig` hooks |
| Long context | Attention head split across ranks (12 query heads + 1 KV head each), indexer keys in mapped pinned host memory, 1024-row chunk, prompt states parked in host memory | `cuda_exl3_heads.zig`, `cuda_exl3_kv.zig`, `cuda_snap.zig` |
| KV paging | Full K/V of the 12 attention layers in pinned host memory with a device window and block cache; default off | `cuda_exl3_kvpage.zig`, `flashnext_exl3_kvpage.cu` |
| CUDA graphs | Pool planned on the device, so a decode step (48 layers, NCCL sums, head gathers, MTP windows) is one graph; prefill layer graphs; MTP draft chain as one graph; background instantiation | `cuda_exl3_dplan.zig`, `cuda_exl3_graph.zig`, `cuda_exl3_chain.zig`, `cuda_exl3_recycle.zig` |
| int8 KV | Kernels built and measured, not used by default (decision: bf16 KV is the main path) | `cuda_exl3_kv.zig`, `flashnext_exl3_kv.cu` |

## Verification

References: a float64 oracle (`exl3_*_cpu.zig`, kept as test code) and an independent Python run of the
upstream-quality reference (`tools/exl3_ref_golden.py`).

- **Single card, served over HTTP:** drafted output equals undrafted on every prompt; lm_head matches float64 to
  2.95e-3 of peak; teacher-forced top-1 equals the reference at 16 of 16 positions.
- **tp=2, served over HTTP** (`--tp 2 --rank R`): drafted equals undrafted, bench hash `2c5d814cfa80` in every
  configuration; layers 0, 3 and 17 are bit-identical to tp=1 when run all-resident.
- **200k context, 2 concurrent requests, tp=2, bf16 KV:** two needle requests of about 196k tokens sent together
  were both answered correctly, with drafts on and off, drafted equal to undrafted.
- **KV paging:** at 200k on one card, last-row logits, 128 greedy tokens and every K/V row are bit-equal to keeping
  K/V on the card.
- **Graphs:** logits of every graphed step are bit-equal to eager on both ranks; the device planner equals the host
  planner on 1,600 random rounds.

### Test measurements (ms per token, tp=2 serve gate, two 128-token runs each)

| Configuration | drafts off | drafts on |
|---|---|---|
| Before the stall fixes, eager | 29.7 - 30.1 | 14.6 - 14.9 |
| Defaults (n-gram gather inline) | 24.7 - 26.2 | 11.9 - 12.6 |
| Blocking NCCL communicator | 21.2 - 23.3 | 11.5 - 12.2 |
| Graphs + blocking NCCL (final branch) | 21.1 - 23.4 | 11.5 - 13.7 |

The three lower rows were reproduced on the final branch head (`44d65597`) by a separate gate run: bench hash
`2c5d814cfa80` and drafted equal to undrafted in every configuration; two of three golden prompts match.

Prefill, real text, rows per second: tp=2 deep 1,555 at 8k, 1,486 at 32k, 1,421 at 65k, 1,225 for a real 200k
prompt (tp=1 deep: 1,531 / 1,461 / 1,395 / 818). The old batched pass ran at 100 - 210. A 200k prompt prefills in
168 s on one card; two 196k prompts interleaved took 304 s and 554 s at tp=2.

## What graphs do and do not give

Graphs work and match eager bit for bit, but they are **not faster** than eager on this hardware. The decode
speedup came from removing host stalls: the pool's per-layer host wait (replaced by the device planner), thread
spawns in the n-gram gather, and the non-blocking NCCL communicator's host cost. Once those are gone a step is bound
by GPU kernels (profile: about 20 ms of kernels per token; dense EXL3 projections 6 ms, bf16 hyper-connection
lanes 4.5 ms, all-reduces 2 - 3.5 ms, experts 3 ms). With drafts on, the first drafted request after start is
slower with graphs because new window shapes are captured on the request path. Graphs stay opt-in
(`TF_FLASHNEXT_EXL3_GRAPH=1`). Prefill layer graphs at tp=2 are bit-equal but 10% slower, so they are opt-in too
(`TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1`).

## Switches

| Variable | Effect | Default |
|---|---|---|
| `TF_FLASHNEXT_EXL3_GRAPH=1` | Decode graphs (and prefill dense-half graphs where enabled) | off |
| `TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1` | Prefill layer graphs at tp=2 | off |
| `TF_FLASHNEXT_NCCL_BLOCKING=1` | Blocking NCCL communicator (init has no deadline if the peer never joins) | off |
| `TF_FLASHNEXT_ATTN_SPLIT=0` | Replicated attention at tp=2 (the split is on by default) | split on |
| `TF_FLASHNEXT_IKC_HOST=1` | Indexer keys in mapped pinned host memory (needed for 200k x 2) | off |
| `TF_FLASHNEXT_KVPAGE=1` | K/V paging to host memory; takes precedence over the head split | off |
| `TF_FLASHNEXT_KV=int8` / `--kv-cache fp8` | int8 KV | off |
| `TF_FLASHNEXT_EXL3_PREFILL=0` | Turn the deep prefill off | on |
| `TF_FLASHNEXT_SHARD_HEAD=0` | Whole lm_head on both ranks | split on |
| `TF_FLASHNEXT_DRAFT_HEAD=affine4` | Cheaper draft head; changes which drafts are proposed (88/111 accepted instead of 90/113), output stays bit-equal | off |

200k x 2 at tp=2 was run with `TF_FLASHNEXT_EXL3_STREAMS=2 TF_FLASHNEXT_IKC_HOST=1
TF_FLASHNEXT_EXL3_PREFILL_ROWS=2048 TENSORFOLD_MEMORY_RESERVE_GIB=0.75 TF_FLASHNEXT_NCCL_BLOCKING=1 --parallel 2`;
the pool had 11 lines per layer on rank 0 and 29 on rank 1 (floor 10).

## Known gaps and risks

1. **Golden gate fails on one prompt.** The 16-token prompt leaves the Python reference at step 3 (ours 9564,
   reference 6124, reference margin 0.125, one bf16 step) on one card and at tp=2. Two of three golden prompts
   match exactly. Treated as a bf16 near-tie; not proven. A teacher-forced margin study would settle it.
2. **The head split is not bit-identical to replicated attention** (different fp32 order in the o_proj sum).
   Greedy continuations after 8k and 32k real tokens differ after 3 - 4 tokens; not shown to be near-ties because
   the server returns no logprobs.
3. **tp=2 logits are not bit-identical to tp=1** beyond the first row (max difference 0.14 mean, up to 1.7 on the
   16-token teacher-forced run); top-1 is equal there. At 65k and 200k the deep-prefill last row swaps a near-tie
   pair between tp=1 and tp=2.
4. **200k x 2 is tight on rank 0** (11 lines per layer, floor 10) and slow while both requests prefill: decode ran
   at 0.4 - 10 tokens per second while the other request prefilled. Prompt-cache reuse with drafts did not work at
   196k (a state kept without the MTP head's rows cannot serve a drafting stream).
5. **KV paging** is default off: incompatible with CUDA graphs (window start and slot base are host arguments),
   refuses prompt-cache snapshots (needs `--prompt-cache-gib 0`), and is not exercised with MTP windows or several
   lanes. At 200k, consecutive rows share only 49% of their selected blocks, hit rate 0.83 with a 384 MiB cache.
6. **Graph path:** warm-up of all window shapes at load, rank-1 independent enqueue and async readback are not
   done; a rank-0 shutdown hang was seen once with graphs on after rank 1 exited.
7. **Prefill at tp=2 is only about 2% faster than tp=1 at equal chunk sizes:** the replicated dense work
   (projections, GDN, attention, hyper-connections) is the bottleneck, not the experts.
8. **Blocking NCCL is opt-in** until its init has a timeout.
9. `zig build test-cuda-host` fails on clean upstream v1.1.0 on macOS as well; `test-flashnext-exl3-host`
   passes. Upstream PR #1046, which touches our hook points, could not be read.

## Next steps

1. Finish the graph work: warm up window shapes at load, graph the chain's absorb step, rank-1 independent enqueue,
   device-side token feed. Flip the default only if graphs win in both modes.
2. Add an init timeout to the blocking NCCL communicator and make it the tp=2 default.
3. Margin study for the golden near-tie and for the head split (needs logprobs from the server).
4. Spread the MTP head across both ranks (rank 1 idles about 28% with drafts on); split the replicated dense
   work to speed up prefill at tp=2.
5. Optional: exact same-layer CPU execution of low-weight pool misses (AVX2 on the EPYC 7452); design notes and
   the KTransformers review are in the session record, nothing is built.
6. Split this branch into reviewable PRs.

## Branch map

`exl3-c-main` merges: core adapter, pool, PLE, draft rows, expert-parallel tp=2 (earlier streams), single-card
serving, tp=2 pools + split lm_head + MTP, KV memory levers (head split, host indexer keys, chunk, host snapshots),
KV paging, CUDA graphs and the stall fixes, tp=2 deep prefill. Design notes: `docs/plans/exl3-on-upstream-1.1.0.md`,
`docs/plans/exl3-kv-memory.md`, `docs/plans/exl3-kv-paging.md`, `docs/plans/exl3-graphs.md`.

Not done on purpose: no change to `docs/recipes`, no family or model registration (held until the strict golden
gate passes).
