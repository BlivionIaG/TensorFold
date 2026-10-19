# CUDA graphs for EXL3 Flash Next with the expert offload pool

Branch `exl3-c-graphs` (base `exl3-c-main` c5bfe87b). Opt-in: `TF_FLASHNEXT_EXL3_GRAPH=1`.

## What blocked graphs: the pool's host cut

Every offloaded MoE layer ended its segment with a D2H of the picks and a stream synchronize (`cuda_exl3_pass.pooled`),
the host planned hits/misses/victims (`exl3_round.plan`), staged the misses on the pool's copy stream and only then
launched the experts. 48 host waits a token, and a step that cannot be captured.

## Design chosen: the pool planned on the device

Measured first, then chosen over "graph A / host plan / graph B per layer":

- `flashnext_exl3_dplan.cu` `fn_exl3_dplan` (one block a layer) runs the host planner's rules for a one-pass round
  without prefetch or double-buffer protection on device copies of each decoder layer's state (line -> expert, stamps,
  expert -> line, lifetime counts, clock): observe, touch held, misses hottest first, first empty line else the
  coldest resident not in the round. It rewrites the pool's own pointer/width tables in place (fixed addresses) and
  writes a copy list. `fn_exl3_dcopy` copies the misses from the pinned host bank through its mapped (unified)
  address into the lines: 12.6 GB/s vs 13.6 GB/s for the copy engine (`flashnext-exl3-graph plan`).
- Equality: device planner == host planner over 400/600/600 random rounds (10/47/63 lines, 1/4/6 rows): lines,
  stamps, counts, hit/copy/eviction counts, all 1536 table entries, every resident line's bytes.
- Ownership (`cuda_exl3_dplan.zig`): the host (prefill, the MTP layer's pool, eager rounds) and the device take turns;
  `toDevice` uploads the host state after draining the copy streams, `toHost` downloads and rebuilds the Vram, counts
  and statistics before any host plan of a decoder pool. A decode step then has no host wait between its first launch
  and its logits.
- `cuda_exl3_graph.zig` captures that step: `Engine.round` (one row a stream) and `Engine.forward` windows with every
  row's logits (MTP verify windows) of at most lines/10 rows. At tp=2 the graph holds the step's 48 NCCL sums and the
  head's gathers (both ranks capture the same calls). Pinned uploads (ids, positions, n-gram rows) become copy nodes
  read at replay; `prepare` rewrites them before each launch. Keys: rows, the sequences' buffers, attention chunk and
  score-tile seams, scratch address and capacity, scores, logits, keep/streams, planner state. A key is captured at
  its 6th sighting (`TF_FLASHNEXT_EXL3_GRAPH_AFTER`, prefill layers at the 2nd: `..._PREFILL_AFTER`); a new key of a
  cached shape updates that exec in place (`cuGraphExecUpdate`, 2-3 ms instead of a 20-35 ms instantiate), at tp=2
  too: the captured template is kept alive with the exec because NCCL's captured resources belong to it
  (`TF_FLASHNEXT_EXL3_GRAPH_TP_UPDATE=0` instantiates instead).
- A round with more misses than lines free of its own experts reads the extra experts in place from the mapped bank
  (same bytes; slower) and clears their entries afterwards (`fn_exl3_drestore`), so any round up to
  `TF_FLASHNEXT_EXL3_GRAPH_ROWS` (16, the MTP window) plans on the device.
- Fallbacks: more than 16 rows, media, or no pools take the host cut (eager), with a residency pull;
  a failed capture turns graphs off for the process and runs the step eagerly on the device plan
  (`TF_FLASHNEXT_EXL3_GRAPH_STRICT=1` makes it an error instead).
- `TF_FLASHNEXT_EXL3_DPLAN=1` alone: the device plan launched eagerly (no capture).
- `TF_FLASHNEXT_EXL3_OVERLAP=1` (default off, not validated on the pack): held experts compute while the misses' copy
  runs on a side stream.

## Prefill

The expert-major deep prefill's dense half of each layer (sub-blocks: HC, DeltaNet/attention, write-backs, router,
top-k) is captured per layer (`prefillBegin/End` hook in `cuda_exl3_prefill.chunk`): a DeltaNet layer's graph depends
only on the chunk's shape and replays for later chunks; an attention layer's launch geometry follows the position, so
it is re-captured each chunk and updated in place. The experts' half stays eager: its passes, ring copies and member
lists are planned on the host from each layer's picks (data-dependent shapes).

## Measurements (par1-cs13, 2x 5060 Ti, idle cards under gpurun)

Engine test `flashnext-exl3-graph eq` (3 golden prompts, 48 greedy rounds + the same continuation as windows 3,1,4,2;
every step's bf16 logits bit-equal to the host cut, both ranks), decode ms/token:

| tp | prompt | host cut | device plan | graph |
|----|--------|----------|-------------|-------|
| 2 | ids16 | 26.2 | 21.6 | 21.7 |
| 2 | France | 27.9 | 23.7 | 23.7 |
| 2 | Once upon | 28.9 | 25.9 | 25.1 |
| 1 | ids16 | 31.5 | 31.1 | 30.7 |
| 1 | France | 35.5 | 35.0 | 34.3 |
| 1 | Once upon | 36.8 | 36.0 | 35.3 |

Two streams in one round (tp=2): 32.7 ms a round both ways, bit-equal.

Why graphs add little over the device plan: a step is GPU-bound. nsys (tp=1): ~20.6 ms of kernels a token (EXL3
grouped experts 5.5 ms, rows=1 EXL3 linears ~7 ms, native HC lanes ~4.4 ms) plus ~10 ms of expert copies at 86%
hits; host launch gaps were already hidden. The host cut's cost was the waits (both ranks stall at tp=2).

Serve gate at tp=2 (tpgate template, two BENCH runs of 128 tokens, ms/token; DRAFTED==UNDRAFTED and BENCH sha
2c5d814cfa80, the eager one, in every run):

| config | drafts off | drafts on |
|--------|-----------|-----------|
| eager (host cut) | 29.1 / 27.4 | 16.1 / 15.2 |
| device plan, no capture (`DPLAN=1`) | 26.9 / 24.5 | 16.3 / 15.6 |
| graphs, capture at 2nd sighting, tp=2 instantiate | 26.6 / 24.1 | 20.3 / 19.8 |
| graphs, 2nd sighting, in-place updates | 26.6 / 24.1 | 18.8 / 17.6 |
| graphs, 6th sighting, in-place updates (default) | 26.7 / 24.2 | 17.4 / 16.3 |

Replays: 301 of 417 graph-eligible steps in the gate (72%), 11 captures (9 of them in-place updates), 105 steps eager
on the device plan (keys not yet seen 6 times: every request brings new sequence/draft buffers, so new keys). With
drafts on, a capture (15-20 ms of host work the GPU waits for) costs more than the replays save on a GPU-bound step.

Prefill (deep chunks of 1024 rows, 4096-row prompt, 5 runs, last-row logits and 8 decode steps bit-equal): tp=2
(with exl3-c-tpprefill's uncommitted work applied) eager 770.2 rows/s, graphs 770.3-772.6; tp=1 498.7 vs 499.1.
A prompt after the first: 144 DeltaNet layer replays and 48 attention-layer in-place updates.

## Not proven / open

- Prefill graphs at tp=2 need the tp=2 deep prefill (branch `exl3-c-tpprefill`, not merged here).
- The experts' half of prefill is not graphed (host-planned passes).
- In-place reads are proven bit-equal (tp=2, 24 lines, 2936 reads) but slow; windows at few lines lose to the host
  cut's DMA passes.
- The overlap split (`TF_FLASHNEXT_EXL3_OVERLAP=1`) is unvalidated on the pack.
- Graphs stay opt-in (`TF_FLASHNEXT_EXL3_GRAPH=1`); `TF_FLASHNEXT_EXL3_DPLAN=1` alone gives the same decode speed.
- 200k contexts, int8 KV and 2+ concurrent streams in the server under graphs were not run (the engine test covers
  two streams in one round, bit-equal).
