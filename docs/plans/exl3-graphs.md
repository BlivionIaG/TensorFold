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
it is re-captured each chunk and updated in place. For TP=2, the graph captures each sub-block's wait on its incoming
NCCL sum as an external event-wait node. This is intended to preserve the eager schedule where ready sub-blocks begin
while later sub-block reductions are still running. Graphs are cleared before their reduction events are destroyed.
The experts' half stays eager: its passes, ring copies and member lists are planned on the host from each layer's picks
(data-dependent shapes).

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

Prefill graph overlap validation (TP=2, 8192-row prompt, last-row logits plus 8 decode steps bit-equal): graphs
capture and replay the dense half of each layer, with external event-wait nodes for each expert-parallel reduction.
The graphs are cleared before those events are destroyed. With 1024-row chunks, eager measured 770.5 rows/s and
steady graph runs 771.0-772.0; with 2048-row chunks, eager measured 1116.3 and graphs 1112.1-1117.1; with
4096-row chunks, eager measured 1468.0 and graph replays 1469.5-1471.1. These small differences are within run
variation; a graph speedup is not demonstrated. An 8192-row chunk OOMed and fell back to the non-deep path, so its
output is not a valid graph measurement. The larger fitting chunk improves prefill throughput for both modes; it
does not establish a graph-specific win.

Decode fork experiment (TP=2, same kernels and bit-equal logits): toggling multipart projection stream forking on
helped DPLAN eager in these runs, while graph changes were mixed and near measurement noise. This does not explain
or fix the graph-vs-eager decode gap. The earlier controlled serve gate remains the relevant result: graphs did not
reliably beat DPLAN eager.

Local, uncommitted MTP verify-tail A/B (TP=2, blocking NCCL, graph mode): moving the existing greedy `Head.best`
selection and its small result copy onto the main forward stream before the graph's completion boundary preserved
the serving outputs and bench hash, but the paired 128-token gate showed no observable decode-time change at its
reported precision. This is one A/B pair, not a repeatability proof; the optimization remains opt-in and is not
claimed as a graph speedup.

## Not proven / open

- Prefill graph capture/replay is now validated with TP=2 deep prefill at chunk sizes 1024, 2048 and 4096, including
  bit-identical logits and post-prefill decode. It restores the lost reduction overlap but has not shown a repeatable
  throughput win over eager.
- The experts' half of prefill is not graphed (host-planned passes).
- In-place reads are proven bit-equal (tp=2, 24 lines, 2936 reads) but slow; windows at few lines lose to the host
  cut's DMA passes.
- The overlap split (`TF_FLASHNEXT_EXL3_OVERLAP=1`) is unvalidated on the pack.
- Graphs stay opt-in (`TF_FLASHNEXT_EXL3_GRAPH=1`); a repeatable decode speed win over `TF_FLASHNEXT_EXL3_DPLAN=1`
  remains unproven.
- The greedy verify-tail experiment has no demonstrated benefit and stays default-off pending a repeated profile or
  a measured GPU/host critical-path reduction.
- 200k contexts, int8 KV and 2+ concurrent streams in the server under graphs were not run (the engine test covers
  two streams in one round, bit-equal).

## Making tp=2 decode faster (branch exl3-c-fast, base exl3-c-main 7988c6fa)

Profile (nsys, `flashnext-exl3-graph prof`: cuProfilerStart/Stop over 16 graphed rounds of 1 row and 16 verify
windows of 3 rows, both ranks, idle cards, ~99% hits), per round: rank 0 wall 20.2 ms (GPU busy 19.6), rank 1 19.6;
EXL3 dense projections 6.1 (GDN in_proj q|k|v 2.2, z / attention 1.4, GDN out_proj 1.1, lm_head half 0.56),
native bf16 lanes (HC down/up, mixer) 4.5, NCCL 3.5 on rank 0 vs 2.0 on rank 1 (60 all-reduces; rank 0 waits ~1.5 ms
for rank 1's shared expert and extra misses), experts 2.9 / 3.3, HC glue 0.9, router/top-k 0.6, GDN 0.5, device
plan 0.25, miss copies 0.24 / 0.74, attention 0.1. Serve profile with drafts on: the GPU idled about half the time;
every kernel ran eagerly (verify windows rarely repeat a graph key) and each eager all-reduce cost ~294 us of host
time (6801 cudaStreamWaitEvent calls = 2.0 s on rank 0 over one 128-token request).

What helped (all bit-identical: same kernels on the same inputs):

1. The n-gram rows a step gathers on the host (`Ngram.rows`, EXL3 table): inline instead of items/4 threads, and a
   one-pass row unpack: 0.6-0.8 ms a token -> 0.05-0.13 ms (3-row window 2.35 -> 0.15-0.36 ms). It sat on the
   critical path in both modes (graph: before launch; eager: joined at layer 1).
2. `TF_FLASHNEXT_NCCL_BLOCKING=1`: a blocking NCCL communicator. Eager all-reduce 109 us -> 20.6 us (the captured
   one is 20 us); `Collective.wait` also spins before its 50 us sleeps. Opt-in: a blocking init has no deadline if
   the peer never joins.
3. Multi-part EXL3 projections' parts on parallel streams (`cuda_exl3_fork.zig`, default on): GPU busy 19.6 -> 18.8
   ms a round, within noise end to end.

Serve gate at tp=2 (BENCH 128 tokens twice, ms/token; DRAFTED==UNDRAFTED and sha 2c5d814cfa80 in every run):

| tree / config | drafts off | drafts on |
|---|---|---|
| 7988c6fa eager | 30.1 / 29.7 | 14.9 / 14.6 |
| 7988c6fa `GRAPH=1` | 25.7 / 23.5 | 15.4 / 14.8 |
| exl3-c-fast `GRAPH=1 NCCL_BLOCKING=1` | 23.7 / 21.4 | 12.9 / 12.0 |
| exl3-c-fast `NCCL_BLOCKING=1` (eager) | 23.4 / 21.3 | 12.2 / 11.6 |

With cheap eager collectives the step is GPU-bound and graphs no longer add speed (drafts on they lose a little to
captures of keys that do not repeat).

Tried, no gain or rejected:
- upstream's EXL3 linear body re-instantiated at 2-4 programs an SM (128 registers, small spills): bit-equal,
  19.67 -> 19.71 ms/token (the 1-row projection is bound by trellis decode and its serial k walk, not occupancy);
- the device plan's held/miss overlap split (`TF_FLASHNEXT_EXL3_OVERLAP=1`): 22.3 -> 22.1 ms rounds, 29.4 -> 29.9 ms
  windows; stays off;
- resident MTP experts (`TF_FLASHNEXT_EXL3_MTP_RESIDENT=1`): costs rank 0 17 pool lines, no change;
- `NCCL_PROTO=LL`: captured all-reduce 19.8 -> 11.4 us, serve drafts off 24.3/22.0 -> 23.8/21.1 ms, but tp=2 deep
  prefill 1473 -> 1288 rows/s (-12%): not set by default (it is process-wide);
- prefill layer graphs at tp=2 before external event waits: bit-equal, 1474 -> 1324 rows/s. The measured external
  event-wait change restores the overlap and bit parity; it does not produce a repeatable graph-specific speedup.

What remains is kernel time: ~6 ms of 1-3-row EXL3 projections (decode-bound at ~190-290 GB/s), ~4.5 ms of bf16 HC
lanes (already 290-390 GB/s of 448), ~3 ms of experts, ~2 ms of collectives, ~1.5 ms of rank-0 wait. Faster
projection kernels with a different split or fewer HC bytes change the fp32 order (numerics) and would have to be
default-off with logit diffs; the MTP head runs on rank 0 alone (rank 1 idle ~28% with drafts on).
