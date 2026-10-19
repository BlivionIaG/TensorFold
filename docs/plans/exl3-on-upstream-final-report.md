# EXL3 Qwen3.8-Flash-Next on upstream TensorFold v1.1.0: final report

Branch `exl3-on-upstream-wip` (staging branch `exl3-c-main`), based on upstream tag `v1.1.0`.
Machine: par1-cs13, two RTX 5060 Ti 16 GiB (sm_120), 251 GB RAM, pack
`Qwen3.8-Flash-Next-exl3-3.05bpw`. All measurements are test measurements on idle cards under a lock
wrapper, not benchmarks.

## Current handover (2026-10-19)

Draft PR [#50](https://github.com/BlivionIaG/TensorFold/pull/50) is open from
`exl3-on-upstream-wip` (current report head `79a3db12`; runtime code head `44d65597`) into the fork's
`upstream-v1.1.0` base, copied from upstream tag
`v1.1.0`. It is intentionally one large draft to split later. No final merge-readiness claim is made here.

The user's acceptance target is TP=2 with CUDA graphs on both decode and prefill, and graphs must improve
performance. BF16 is the main KV format; dynamic KV paging to RAM is also required. CPU expert execution is
deferred. Decode graphs match eager for eligible captured shapes, but input preparation, the MTP chain's absorb
step, and host readback remain outside capture. No repeatable graph speedup over optimized blocking-NCCL eager
decode has been demonstrated. Prefill graphs capture only each layer's dense sub-blocks; expert scheduling,
ring copies and reductions remain eager. The recorded TP=2 prefill result fell from 1,474 to 1,324 rows/s
(about 10%), as capture removed reduction overlap. Graph performance and full prefill coverage are therefore open
acceptance gates.

The 200k × 2 TP=2 BF16 needle test passed on this tree before the final agent changes; it was not rerun on the
exact PR head. Treat that result as demonstrated on a near-final tree, not as a final-head gate. Separately,
single-card KV paging at 200k matched resident BF16 K/V, but KV paging has not been combined with TP=2 serving,
CUDA graphs, multiple lanes, or prompt snapshots. Its changing host-side window offsets currently prevent safe
graph replay. There is also an allocation/budget mismatch when paging is combined with host indexer keys: the
sequence can take the resident-KV allocation path while the planner budgets it as paged, risking VRAM
under-reservation. Paging overflow sets a device flag, but serving does not yet reject or retry the request.

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
| CUDA graphs | Eligible decode rounds/windows capture the 48-layer pass, NCCL sums and head gathers; prefill captures dense layer sub-blocks; greedy MTP proposals have a conditional graph; graph instantiation runs on a worker | `cuda_exl3_dplan.zig`, `cuda_exl3_graph.zig`, `cuda_exl3_chain.zig`, `cuda_exl3_recycle.zig` |
| int8 KV | Kernels built and measured, not used by default (decision: bf16 KV is the main path) | `cuda_exl3_kv.zig`, `flashnext_exl3_kv.cu` |

## Verification

References: a float64 oracle (`exl3_*_cpu.zig`, kept as test code) and an independent Python run of the
upstream-quality reference (`tools/exl3_ref_golden.py`).

- **Single card, served over HTTP:** drafted output equals undrafted on every prompt; lm_head matches float64 to
  2.95e-3 of peak; teacher-forced top-1 equals the reference at all 16 tested prompt positions. This is a narrow
  sample, not an end-to-end strict golden pass.
- **tp=2, served over HTTP** (`--tp 2 --rank R`): drafted equals undrafted, bench hash `2c5d814cfa80` in every
  configuration; layers 0, 3 and 17 are bit-identical to tp=1 when run all-resident. The serving script's hash
  checks are truncated and it can exit successfully after a golden mismatch; interpret its status as evidence,
  not a strict pass/fail gate.
- **200k context, 2 concurrent requests, tp=2, bf16 KV:** two needle requests of about 196k tokens sent together
  were both answered correctly, with drafts on and off, drafted equal to undrafted.
- **KV paging:** at 200k on one card, last-row logits, 128 greedy tokens and every K/V row are bit-equal to keeping
  K/V on the card; sequence VRAM fell from 5.484 GiB to 1.391 GiB. TP=2 paging has only been tried sequentially
  at shorter contexts, without token hashes or paging counters.
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

For eligible shapes, decode graphs match eager bit for bit, but a repeatable performance win over the optimized
eager path has **not been demonstrated** on this hardware. The decode
speedup came from removing host stalls: the pool's per-layer host wait (replaced by the device planner), thread
spawns in the n-gram gather, and the non-blocking NCCL communicator's host cost. Once those are gone a step is bound
by GPU kernels (profile: about 20 ms of kernels per token; dense EXL3 projections 6 ms, bf16 hyper-connection
lanes 4.5 ms, all-reduces 2 - 3.5 ms, experts 3 ms). With drafts on, the first drafted request after start is
slower in the recorded gate; graph instantiation now runs on a worker, but graph execution still synchronizes for
results and the recorded timings do not isolate first-use capture from replay. Graphs stay opt-in
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

The 200k x 2 tp=2 run used resident BF16 KV (not dynamic KV paging) and was run with
`TF_FLASHNEXT_EXL3_STREAMS=2 TF_FLASHNEXT_IKC_HOST=1
TF_FLASHNEXT_EXL3_PREFILL_ROWS=2048 TENSORFOLD_MEMORY_RESERVE_GIB=0.75 TF_FLASHNEXT_NCCL_BLOCKING=1 --parallel 2`;
the pool had 11 lines per layer on rank 0 and 29 on rank 1 (floor 10).

## Known gaps and risks

1. **Golden gate fails on one prompt.** The 16-token prompt leaves the Python reference at step 3 (ours 9564,
   reference 6124, reference margin 0.125, one bf16 step) on one card and at tp=2. Two of three golden prompts
   match exactly. Treated as a bf16 near-tie; not proven. A teacher-forced margin study would settle it.
2. **The head split is not bit-identical to replicated attention** (different fp32 order in the o_proj sum).
   Greedy continuations after 8k and 32k real tokens differ after 3 - 4 tokens; not shown to be near-ties because
   the server returns no logprobs.
3. **tp=2 parity is not established.** A saved 39-position teacher-forced run matched tp=1 at 37/39 positions
   and failed at positions 18 and 33; logits also differ due to fp32 reduction order. At 65k and 200k the
   deep-prefill last row swaps a near-tie pair between tp=1 and tp=2. The single-layer 16-position top-1 check
   does not cover this case.
4. **200k x 2 is tight on rank 0** (11 lines per layer, floor 10) and slow while both requests prefill: decode ran
   at 0.4 - 10 tokens per second while the other request prefilled. Prompt-cache reuse with drafts did not work at
   196k (a state kept without the MTP head's rows cannot serve a drafting stream).
5. **KV paging** is default off. The 200k bit-equality gate is single-card only; TP=2 checks are shorter and
   sequential, and the 200k × 2 TP=2 gate used resident KV. Paging is not graph-safe (window/epoch are host state),
   refuses prompt-cache snapshots, and is not exercised with MTP windows or multiple lanes. With host indexer keys,
   sequence placement can allocate resident K/V while budget accounting assumes paging; fix this before combining
   the modes. Serving also needs to fail safely on paging-cache overflow instead of continuing with missing
   mappings. At 200k, consecutive rows share only 49% of their selected blocks, hit rate 0.83 with a 384 MiB cache.
6. **200k reference parity is not established:** a saved 200k prefill comparison exited 1 with max logit
   difference 1.1406 and top-1 6009 versus 19251 (reference margin 0.25). The successful 200k × 2 needle answers
   establish retrieval/serving behavior for those prompts, not strict full-logit parity.
7. **Graph path:** warm-up of all window shapes at load, rank-1 independent enqueue and async readback are not
   done; a rank-0 shutdown hang was seen once with graphs on after rank 1 exited.
8. **Prefill at tp=2 is only about 2% faster than tp=1 at equal chunk sizes:** the replicated dense work
   (projections, GDN, attention, hyper-connections) is the bottleneck, not the experts.
9. **Blocking NCCL is opt-in** until its init has a timeout.
10. Existing automated gates need stricter pass/fail behavior before serving as acceptance gates: the HTTP gate can
    print golden mismatches without failing, and compares only truncated drafted/undrafted hash prefixes; the
    prefill comparison allows a close-margin top-1 and can skip missing reference dumps. The passing summaries
    above must be read with those limits.
11. **MTP draft-token parity is not directly asserted for graph replay.** The graph test checks main-model logits,
    not the draft proposals or replay count. Final drafted/undrafted output equality does not prove the proposed
    tokens matched. The draft-row test covers limited inputs and does not assert argmax equality.
12. `zig build test-cuda-host` fails on clean upstream v1.1.0 on macOS as well; `test-flashnext-exl3-host`
   passes. Upstream PR #1046, which touches our hook points, could not be read.

## Next steps

1. Meet the graph acceptance gate: reduce decode host boundaries (including the MTP absorb/readback path), make
   decode graphs measurably faster than optimized eager decode, and graph prefill expert passes while preserving
   reduction overlap. Recheck prefill and decode with repeatable TP=2 gates before changing defaults.
2. Fix the paging/indexer allocation-budget mismatch and add a serving overflow guard. Then make dynamic BF16 KV
   paging replay-safe and rerun the 200k × 2 TP=2 test on the exact PR head with paging, graphs, and concurrent
   requests enabled together.
3. Add an init timeout to the blocking NCCL communicator before making it the tp=2 default.
4. Make the HTTP and prefill gates fail on strict mismatches, compare full hashes, and require reference artifacts
   for long-context parity runs; investigate the 200k reference-logit failure.
5. Assert actual graph replay and compare full MTP draft-token sequences against eager; strengthen the draft-row
   test to assert the decision-critical top-1 output.
6. Margin study for the golden near-tie and for the head split (needs logprobs from the server).
7. Spread the MTP head across both ranks (rank 1 idles about 28% with drafts on); split the replicated dense
   work to speed up prefill at tp=2.
8. Optional later: exact same-layer CPU execution of low-weight pool misses (AVX2 on the EPYC 7452); design notes
   and the KTransformers review are in the session record, nothing is built.
9. Split this branch into reviewable PRs after the TP=2 graph gates pass.

## Branch map

`exl3-c-main` merges: core adapter, pool, PLE, draft rows, expert-parallel tp=2 (earlier streams), single-card
serving, tp=2 pools + split lm_head + MTP, KV memory levers (head split, host indexer keys, chunk, host snapshots),
KV paging, CUDA graphs and the stall fixes, tp=2 deep prefill. Design notes: `docs/plans/exl3-on-upstream-1.1.0.md`,
`docs/plans/exl3-kv-memory.md`, `docs/plans/exl3-kv-paging.md`, `docs/plans/exl3-graphs.md`.

Not done on purpose: no change to `docs/recipes`, no family or model registration (held until the strict golden
gate passes).
