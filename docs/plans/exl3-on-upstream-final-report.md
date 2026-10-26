# EXL3 Qwen3.8-Flash-Next on upstream TensorFold: final report

Branch `exl3-on-upstream-wip` (staging branch `exl3-c-main`), originally based on upstream tag `v1.1.0` and now merged with current upstream `main`.
Machine: par1-cs13, two RTX 5060 Ti 16 GiB (sm_120), 251 GB RAM, pack
`Qwen3.8-Flash-Next-exl3-3.05bpw`. All measurements are test measurements on idle cards under a lock
wrapper, not benchmarks.

## Current handover (2026-10-23)

Upstream draft PR [#1113](https://github.com/ashhart/TensorFold/pull/1113) is open from the fork branch
`exl3-on-upstream-wip` into upstream `main`. The branch integrates current upstream at `d1018b318`; exact upstream
commit ancestry is retained. The PR description contains the proposed dependency-ordered split plan. This remains
one large draft and is not merge-ready.

The user's acceptance target is TP=2 with CUDA graphs on both decode and prefill, and graphs must improve
performance. BF16 is the main KV format; dynamic KV paging to RAM is also required. CPU expert execution is
deferred. Decode graphs match eager for eligible captured shapes, but input preparation, the MTP chain's absorb
step, and host readback remain outside capture. A new serving ABBA shows a narrow no-draft decode gain but a
material MTP-drafted regression; graphs therefore do not yet meet the performance acceptance gate. Prefill graphs
capture each layer's dense sub-blocks; expert scheduling and ring
copies remain eager. A new event-wait-node path preserves overlap with TP=2 sub-block reductions and passes the
bit-equality gate. Across 8192-row prompts with 1024-, 2048- and 4096-row chunks, graph throughput tracks eager
within run variation; a graph-specific prefill win is not demonstrated. An 8192-row chunk OOMed and fell back to
the non-deep path, so it is excluded from graph results. Graph performance remains an open acceptance gate.

A TP=2 real-text HTTP ABBA on the same native binary, with graphs and shared BF16 paging enabled, measured a small
prefill-only improvement from setting expert-ring lookahead to one layer: 5.1851 s versus 5.2499 s for an 8,192-token
prompt (+1.25% throughput). All 12 measured one-token outputs matched. Lookahead 1 becomes the default, but this is
not a decode gain, concurrent-throughput result, or meaningful progress toward the 2× target by itself. The isolated
synthetic-ID profiler shows a similar ~1% result at 8k and 16k rows, with about 2.8% more expert copies.

The near-200k × 2 TP=2 BF16 needle test (195,938 and 195,957 prompt tokens) passed on this tree before the final
agent changes; it was not rerun on the exact PR head. Treat that result as demonstrated on a near-final tree, not
as a final-head gate or an exact-200k test. The follow-up implementation now combines BF16 paging, snapshots, TP=2,
graphs, MTP and two concurrent HTTP chat histories at ordinary lengths. Restored graph state/logits match exactly;
warm, branch and rewind cache phases each reuse 4,122/4,139 prompt tokens with exact response hashes versus cold
cache. The 200k × 2 gate remains explicitly deferred. Live-rank overflow/failure injection and graph speedup across
the intended drafted workload remain open. BF16 is required; int8 KV is out of scope.

A follow-up on `e22c4552a` passes this gate on both RTX 5060 Ti ranks: shared BF16 paging, recycled request allocation,
snapshot restore, exact state/logit continuation, and graph-mode decode. The earlier zero-capture result came from a
128-entry FIFO warmup history evicting the 192-shape working set (two lanes × two chunk banks × 48 layers). Raising
history capacity to 224 allowed each rank to capture 112 graph entries; the bounded graph cache then replayed all 112
on the next run, including 28 paged-attention replays and four safe generation/origin rebinds. The other 80 shapes ran
eagerly as intended by the 112-entry cache bound. This proves cache reuse and correctness, not a latency gain; graph
speedup remains an open measured-performance gate.

## 2026-10-23 score-grid graph-key ABBA follow-up

An existing four-leg TP=2 HTTP ABBA (bucket off/on/on/off) used both RTX 5060 Ti cards, `--parallel 2`, 8,192 real-text
prompt IDs, 512 generated tokens, graph-enabled decode and prefill, shared BF16 KV paging, 38 expert lines, prompt
cache disabled, and drafts both off and on. Each leg discarded a warm-up request, then measured two requests per
draft mode. All requests returned the same full token hash
`600815b14b64d917c86eebf8279154089d014500a6d213246692cf86d918680f`; drafted/undrafted parity held.

Mean prefill was 5.2437 s with bucketing off and 5.2473 s on (0.07% slower with the candidate). Mean undrafted decode
was 26.911 versus 27.007 tok/s (+0.36%); drafted decode was 25.214 versus 25.381 tok/s (+0.66%). These deltas are
within run variation and do not qualify as an improvement. The candidate reduced main-round graph instantiations
from 8 to 6 in the server summary, while prefill graph counters were unchanged; the additional capture reuse did not
translate into measured end-to-end speed. The score-grid bucket implementation is therefore rejected and removed
from the source tree. This result rules out this bucketing approach as a meaningful performance step; it does not
close the separate shared-snapshot/prefill-replay correctness gate.

## 2026-10-23 MTP verify-best ABBA

An exclusive TP=2 HTTP ABBA (`TF_FLASHNEXT_EXL3_VERIFY_BEST` unset/on/on/unset) tested the existing default-off
path that moves each verify row's top-1 selection and asynchronous record copy into the verify tail, avoiding a
second selection in `draws`. Both RTX 5060 Ti cards ran the same binary
(`e3bf1015f9e0d1847d83f5b651bdabefa48c5ecfc0c45d665e7c5197da6827e3`), 8,192 real-text prompt IDs, 512-token
outputs, TP=2, `--parallel 2`, CUDA graphs, shared BF16 KV paging, 38 expert lines, and no prompt cache. Each leg
discarded warm-up requests and measured two plain and two drafted requests. Drafted requests actually proposed 522
tokens and accepted 403 on every leg; undrafted requests proposed/accepted none. Every output had the same full token
hash `600815b14b64d917c86eebf8279154089d014500a6d213246692cf86d918680f`.

Mean prefill was 5.1738 s with the option off and 5.1771 s on (0.06% slower). Undrafted decode was 26.960 versus
26.950 tok/s (-0.04%); drafted decode was 25.261 versus 25.259 tok/s (-0.01%). This is no measurable improvement;
leave the option off by default. Graph logs show stable main-round replay, but reported zero MTP-chain graph captures
in that workload despite the accepted drafts, so that run did not establish that the draft chain itself was graphed.
The following short-decode ABBA separately confirms MTP-chain graph replay for its own workload; the difference
between those run configurations remains unexplained.

## 2026-10-23 clean TP2 short-decode graph/eager ABBA

A follow-up clean-binary ABBA (`graph/eager/eager/graph`) repeated the short HTTP decode gate on source head
`7f8b7a194`, without temporary chain-tracing instrumentation. Binary SHA-256 was
`a20e5165c138dbbf0192c3fb62aa091acf277aae40eddbfc66c2247efba36385`. Both RTX 5060 Ti cards ran TP=2 with
`--parallel 2`, a 32,768-token window, shared BF16 KV paging, 38 expert lines/layer, blocking NCCL, and prompt cache
disabled. Each leg ran the same golden checks and two 128-token measurements per draft mode. Output hash prefix
`2c5d814cfa80` matched across all runs; drafted runs accepted 87 of 115 proposals in every leg.

Undrafted decode averaged 30.03 tok/s eager versus 33.72 tok/s with graphs (**+12.3%**). Drafted decode averaged
41.96 tok/s eager versus 43.91 tok/s with graphs (**+4.6%**). Rank 0 reported 188 main-round graph replays and 54
MTP-chain replays after four captures; rank 1 has no MTP head. This confirms the MTP chain graph captures and
replays on this short greedy workload. It does not explain why the separate 8,192-ID/512-output verify-best ABBA
reported zero chain replays, and it does not supersede that longer drafted regression (eager was 15.6% faster).
Treat this as a workload-specific short-decode improvement, not a general graph win or progress toward 2×. The
prefill timer was about 0.25 s in graph legs versus 1.39 s eager, but prefill graph counters were zero and repeated
fixture requests may reuse warmed request state; exclude this from any prefill claim. TP=2 prefill graph speedup
remains unproven.

## 2026-10-20 upstream-main integration and release checks

The merge resolves the shared Flash Next hooks against current upstream `main`. On par1-cs13, Zig 0.17
`zig build test-cuda-host` passes after fixing a test fixture that left the new paging option undefined;
`zig build native -Dnvcc=nvcc -Dsm=120 -j16` also passes. `zig fmt --check zig/src zig/tests` passes after formatting
the three reported Zig files. These checks validate compilation and host tests only; the final-head TP=2 serving,
graph-performance, BF16 paging/snapshot/cache-reuse, and 200k × 2 gates remain to be rerun.

The lean checker does not pass: current upstream `main` has 444 findings and this branch has 1,761, leaving
1,317 new findings. Treat this as an explicit CONTRIBUTING blocker; the PR stays draft. On 2026-10-20, removed
118 tool co-author/session trailer lines from the 76 feature commits using a commit-object rewrite scoped to the
feature range. Original author identities, subjects, complete tree, and exact upstream `main` commit IDs are
preserved. The branch history is now grouped into five implementation commits, one upstream integration merge,
and one focused documentation commit. The draft omits the obsolete v1.1.0 and int8-first planning memos.

## 2026-10-20 prefill graph profile addendum

The new `prefill-prof` gate ran TP=2 on both GPUs with BF16 KV paging enabled, 8,192 rows, 4,096-row chunks, two
warm-ups, and one measured forward per mode. Both eager and graph runs returned the same cross-rank logits hash
(`ffcf463b229c2aa6`); graph mode replayed 96 dense-layer graphs. One matched untraced pair measured 5,597.1 ms
eager and 5,639.5 ms graph. In the matched Nsight pair, rank 0 measured 5,693.6 ms eager and 5,696.4 ms graph.
Neither pair establishes a speedup.

Nsight shows the graph replacing eager kernel submissions (20,559 `cuLaunchKernel` calls versus 2,103 plus 96
`cuGraphLaunch` calls) while expert-copy volume and kernel/collective totals remain nearly unchanged. Each rank
transferred about 28.8 GB H2D in approximately 48k operations; rank 0 scheduled 14,426 expert copies (13,993 counted
as prefetch) and 570 passes, with about 35 ms CPU planning. The measured picks-D2H-to-synchronize interval is about
5.2 s cumulatively, but includes the preceding dense kernels and NCCL dependencies; it is not pure CPU stall time.
The next prefill performance work is therefore a proper expert-routing/transfer pipeline, not another capture-threshold
change. The design investigation should first establish staging-arena memory fit after BF16 paging, pools, scratch,
and snapshots; then prototype GPU-built stable member lists and fixed-buffer double-buffered transfers while keeping
existing expert arithmetic and reduction order.

## 2026-10-22 TP2 prefill profile and reduction-overlap A/B

At head `5a94359c`, Nsight Systems profiled a 16,384-row TP2 prefill with graphs, BF16 KV paging, 4,096-row deep
chunks, two warm-ups and one measured forward. Both ranks replayed 168 dense-layer graphs, captured 24 shapes, and
reported the same logits hash (`662339a95f0de2d7`). The measured test-harness wall time was about 11.3 s per rank;
this is not a production HTTP benchmark. In the per-rank summed kernel durations, NCCL ring all-reduce accounts for
24–27%, `tf_exl3_prompt_2` about 22%, GDN about 8%, and attention about 6%. The CUDA memory-operation report shows
about 5.0–5.1 s of cumulative host-to-device memcpy durations per rank. Those intervals can overlap other GPU work;
they are not isolated expert-miss stall time. The kernel-duration sums are likewise not additive wall time.

A follow-up, non-Nsight ABBA test compared only `TF_FLASHNEXT_EXL3_PREFILL_OVERLAP` (1, 0, 0, 1) using the same
binary and graph/paging configuration. Each run used 16,384 rows, 4,096-row chunks, two graph warm-ups, and the
same output hash. Graph replay counts, expert-copy counts and planning time were unchanged. Overlap-on wall times
were 11,215.0 and 11,212.1 ms; overlap-off times were 12,475.7 and 12,472.0 ms. The on-pair mean is 11,213.6 ms
versus 12,473.9 ms off, a 10.1% wall-time reduction (11.2% higher rows/s). This is a repeatable targeted prefill
optimization, not a 2× result and not proof of eager-vs-graph speedup. It supports keeping sub-block reductions
overlapped; next compare collective grouping and remove the post-reduction placement/writeback work only behind
exact rank and graph parity gates.

An opt-in reduced-branch follow-up (commit `99265db1`, `TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH=1`) skips the
post-all-reduce `tpPlace` and feeds the contiguous FP32 sum directly to HC writeback with one unit-weight slot.
The default remains off. The targeted `flashnext-hc` GPU regression passed the new exact hidden/FP32-sum,
signed-zero, BF16-tie and short-tail checks. A separate graph-on, BF16-paged 16k TP2 ABBA run (flag 0, 1, 1, 0)
measured 11,207.6 ms off and 11,033.4 ms on, about 1.6% higher rows/s. All four runs replayed 168 graphs per rank
and returned hash `662339a95f0de2d7`; expert-copy counts and planner time were unchanged. This is a small measured
gain, not a 2× result. Snapshot/cache continuation under the opt-in mode and eager-vs-graph parity remain to be
qualified before enabling it by default.

A four-run 4,096/6,144/6,144/4,096 ABBA comparison used 16,384 rows, TP=2, graphs, BF16 paging, overlap on, and
reduced writeback off. Mean wall time was 11,210.7 ms at 4,096 rows and 10,757.7 ms at 6,144 rows (4.0% lower wall
time, 4.2% higher rows/s). All eight rank-runs returned hash `662339a95f0de2d7`; replay counts were 168 versus 120.
Rank 0 expert copies fell from 25,588 to 21,648 (-15.4%) and passes from 1,144 to 845 (-26.1%). This is a stable,
small chunk-size gain, not a 2× result. Larger chunks consume more scratch/memory and 8,192 already OOMed on this
configuration.

## 2026-10-22 prompt chunk and decode-miss profile

On fork head `06d58b2e`, increasing the EXL3 prompt chunk from 256 to 2,048 rows reduced the matched 8,192-row
TP=2 graph + BF16-paged prefill from the prior ~5.48 s baseline to 5.206 s (about 5% lower wall time). Two current
repeats measured 5,206.4/5,206.1 ms and 5,205.9/5,205.5 ms on ranks 0/1, with the same hash
`ffcf463b229c2aa6`, 72 graph replays, and no eager layers. A 32,768-row run completed in 21,853.8/21,853.6 ms
with matching hash `5894636da2406e08` and 360 graph replays. These results demonstrate the chunk-size setting works
at both shapes; only the 8,192-row result has a matched prior-chunk timing. This is not a graph-on/off comparison.

A separate 16,384-context, 64-step TP=2 decode profile with BF16 paging compared eager DPLAN and graph replay in the
same harness. It passed output-hash parity (`fba5a0bab565af03`) on both ranks. Graph measured 24.30 ms/token versus
24.26 ms/token eager (+0.16% latency, effectively no change); it reused an existing graph for all 64 steps with no
captures or executable updates. The measured graph segment recorded 14,017 expert hits, 1,135 misses/copies across
3,072 passes, and 4,575 KV-page misses (300,405 required page slots; 768 calls; no rebases or overflow). These
counters identify expert and KV miss frequency, not their exposed stall time. Graph replay correctness is confirmed;
a decode graph speedup is not.

## 2026-10-23 TP2 prefill reduction-overlap and chunk ABBA

On fork source head `85df2b43d`, an exclusive four-leg ABBA compared the existing overlapped per-sub-block NCCL
reduction path (default) against `TF_FLASHNEXT_EXL3_PREFILL_OVERLAP=0`. Both RTX 5060 Ti cards ran the same 8,192-row
TP2 prefill with BF16 KV paging and prefill graphs enabled. Each leg replayed 72 dense-layer graphs, had zero eager
fallbacks, and returned cross-rank logits hash `ffcf463b229c2aa6`. Rank wall times were 5,056.2/5,056.1 ms and
5,059.2/5,059.2 ms with overlap on, versus 5,886.9/5,886.6 ms and 5,884.8/5,884.6 ms with overlap off. The paired
means are 5,057.7 ms and 5,885.7 ms: overlap reduced latency 14.1% (16.4% more rows/s). Expert copies, passes and
planning time were unchanged. This confirms the existing overlap implementation is valuable; it is not a new code
change, a graph-on/off comparison, or a 2x result.

A second four-leg ABBA on the same source and workload tested `TF_FLASHNEXT_CHUNK=2048` against the default 1,024.
All legs passed graph/paging and output-hash checks. The 1,024-row legs averaged 5,055.6 ms; 2,048-row legs
averaged 5,300.7 ms (4.85% higher latency). Reject this larger chunk setting for this 8k workload. This is
consistent with the earlier 256-to-2,048 improvement being workload/starting-point specific; it does not justify
globally increasing the current default.

An additional counter-only telemetry run attributed rank 0/1 cross-layer prefetches to 13,098/13,719 copies, of
which 4,712/4,778 (36.0%/34.8%) were evicted unused; their logical payloads were 8.69/8.81 GB. The telemetry
sampling mode adds a synchronization boundary, so its wall time is not used. A separate uninstrumented four-leg
ABBA compared current `ring_ahead=2` with `TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD=0`: mean wall was 5,060.0 ms with
lookahead and 6,409.9 ms without it (+26.7% latency when disabled), despite 6.5–6.9% fewer expert copies. Every
run replayed 72 graphs and returned the same logits hash. Thus prefetch traffic is not safely removable wholesale;
the next opportunity is to improve which experts are prefetched, while retaining the asynchronous lookahead.

## 2026-10-23 TP2 expert-prefetch lookahead ABBA

An exclusive TP2 `prefill-prof` ABBA on both RTX 5060 Ti cards compared `ring_ahead=1` against `ring_ahead=2` at
8,192 and 16,384 rows. Both tests enabled BF16 KV paging and CUDA prefill graphs; warm-up/capture ran before the
measured forward. Inputs are deterministic PRNG IDs (seed `0x9e3779b9`), so this is an isolated schedule profile, not
a real-text performance receipt. Each rank returned the same logits hash for both settings (`ffcf463b229c2aa6` at 8k,
`662339a95f0de2d7` at 16k), with 72 and 168 graph replays respectively and zero eager fallbacks.

At 8k, mean slower-rank wall time was 5,005.6 ms with lookahead 1 and 5,058.0 ms with lookahead 2 (+1.05% rows/s for 1).
At 16k, it was 10,257.1 ms and 10,367.4 ms (+1.07% rows/s for 1). The result repeats across these two tested
shapes. Lookahead 1 made about 2.8% more expert copies, so the profile suggests improved overlap/wait time rather than
less traffic or fewer misses.

A real-text HTTP follow-up used source head `60c92357b` plus the one-line default change, the same native binary
throughout (`e3bf1015f9e0d1847d83f5b651bdabefa48c5ecfc0c45d665e7c5197da6827e3`), Qwen3.8-Flash-Next EXL3 pack
fingerprint `e98f5301e6eb58ab48a117e892f71aecd19b449cec5513a620c53cdcb95ddce4`, two RTX 5060 Ti 16 GiB (SM 12.0),
Linux 6.8.0-147-generic, NVIDIA driver 615.71.09, CUDA 13.4, and Zig 0.17.0. Each server was TP=2, context 32k,
parallel capacity 2, 38 expert lines, shared BF16 KV paging, CUDA decode/prefill graphs, drafts off, and prompt cache
off. The request was serial (not concurrent): first 8,192 IDs from the same locally held real-text fixture
(SHA-256 `b74a9216bb35cf88763989f4fd30ce93dfd726d033403b5c8c268b2b96c2d9b8`), one generated token. The first
request per server was discarded as graph warm-up; three fresh measured requests followed. Four exclusive server
legs ran lookahead 1/2/2/1. Mean server-reported prefill was 5.1851 s at 1 and 5.2499 s at 2: **1.25% higher
prefill throughput** at lookahead 1. All 12 measured outputs had token hash
`c6f3ac57944a531490cd39902d0f777715fd005efac9a30622d5f5205e7f6894`; both graph flags were enabled and server logs
confirm shared BF16 paging. The HTTP server does not emit per-request replay counts. The same binary's synthetic
`prefill-prof` gate separately logged 72/168 graph replays at 8k/16k. This confirms a small, repeatable prefill gain on one real-text prompt,
not a decode win, concurrency throughput result, 2× result, or snapshot/prefix-reuse gate. The runtime default is
reduced to 1; `TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD` remains the override. Snapshot/cache behavior was not changed
and is not re-claimed by this benchmark.

## 2026-10-23 TP2 reduced HC writeback experiment

An exclusive four-leg ABBA compared the existing TP2 HC writeback (`TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH=0`)
with the reduced-writeback path (`=1`) on both RTX 5060 Ti cards. The workload was an 8,192-row graph-on prefill
with BF16 KV paging, 1,024-row chunks, overlapped NCCL reductions, and identical rank/output hashes
(`ffcf463b229c2aa6`). Default legs averaged 5,062.2 ms; reduced-writeback legs averaged 4,958.6 ms: **2.05% lower
latency / 2.09% higher rows/s**. All legs replayed 72 prefill graphs with zero eager layer fallbacks; expert copies,
pass counts, and planner behavior matched. This is a promising measured opt-in prefill improvement, not a 2x gain,
not a serving/decode result, and not yet the default. The reduced path changes HC writeback arithmetic; snapshot/
restore on private BF16-paged lanes also passed exact continuation and graph replay. The matching shared-paging
cache-reuse gate failed with `GraphBudgetExceeded` during decode graph replay after restoring shared snapshots;
both ranks matched the paired BF16 logits through step 2 before the failure. The option remains opt-in, and shared
snapshot/cache reuse with graphs is not yet validated. The failure is being investigated as graph-budget/cache
admission behavior, not treated as evidence of a numerical mismatch.

## 2026-10-23 TP2 HTTP graph/eager decode ABBA

An exclusive off/on/on/off serving comparison used the same rebuilt binary on both RTX 5060 Ti cards, TP=2,
`--parallel 2`, BF16 KV paging with shared pages, 38 expert-pool lines per layer, prompt cache disabled, and a
16-token completion prompt with a 512-token output. Each server was restarted per leg; the gate first ran its three
golden fixtures, then measured two undrafted and two drafted completions. Thus this is a two-stream-capacity server
but serial HTTP timing (not concurrent-request aggregate throughput). All four measured outputs used hash
`b170e8433207`, and drafted/undrafted results matched. Graph-on runs replayed the stable round graph 477 times,
the MTP chain 21 times, and captured four shapes; prefill graph counters stayed zero for this short prompt.

Mean decode throughput was 32.36 tok/s eager versus 34.79 tok/s graph-on without drafts (+7.5%), but 49.17 tok/s
eager versus 41.51 tok/s graph-on with drafts (−15.6%). The four ABBA legs consistently show that the current
graph-enabled drafted path is slower, despite replay. The server reported first-completion prefill/TTFT around
1.39–1.43 s in eager legs and 0.26–0.32 s in graph legs; because this short prompt triggered no prefill graph and
these values varied with server warm state, do not treat them as a prefill-graph result. This test does not establish
a prefill graph speedup or overall multi-request improvement. Retain graphs as opt-in and profile the MTP
verify/absorb and host boundaries before changing defaults; the TP=2 graph performance requirement is not met yet.

## Cumulative performance context

The earliest recorded one-row kernel/decode-step optimization reduced a test-harness decode step from 2.4–3.0 s to
0.41–0.45 s (roughly 5.3–7.3×); that work is already part of the pushed baseline and is not a new gain from the
current chunk-size change. Historical prefill notes report 100–210 rows/s for the old batched pass, versus 1,225–
1,555 rows/s in the later TP=2 deep-prefill runs (nominally 5.8–15.6×). The old and new numbers were not collected
as a controlled same-binary ABBA experiment, so this range is directional, not a causal speedup claim. The named
`5a94359c`→`03d4dbf3` controlled TP=2 prefill ABBA measured **4.90% more rows/s** (16,384 rows, 4,096-row chunks,
BF16 paging, graphs on) with identical output hashes. A later lookahead ABBA measured +1.25% 8k prefill throughput;
reduced HC writeback measured +2.09% 8k prefill rows/s but remains opt-in because shared-snapshot graph replay
failed. The clean short-decode ABBA measured graph-on gains of 12.3% undrafted and 4.6% drafted at 128 output
tokens; a separate 512-token ABBA measured +7.5% undrafted but −15.6% drafted. These are workload-specific, not
cumulative gains and do not establish 2×. Historical prefill records are roughly 5.8–15.6× directional and
one-row harness decode 5.3–7.3×, but those are not controlled same-workload comparisons. There is no defensible
single cumulative end-to-end decode speedup from project start.

## 2026-10-23 batched EXL3 dispatch and TP=2 two-stream gate

The batched decode keep-buffer path was calling the affine expert path for EXL3 layers, whose affine pointers are not
initialized by the EXL3 loader. The focused host test also requires an unattached expert pool to fail before any
keep-buffer copy. The fix now preflights `needsPool()`, copies the requested keep buffers, and dispatches through
`exl3_pass.moe`. On the current fork head plus this change, `zig build test-flashnext-exl3-host` passes and
`zig build native -Dnvcc=nvcc -Dsm=120 -j16` passes (one existing unused-variable nvcc warning).

The rebuilt binary passed an HTTP TP=2 `--parallel 2` decode-graph test on both RTX 5060 Ti cards at 16,384 context
with BF16 KV paging, 68 expert lines/rank, two planned streams, a 0.5 GiB memory reserve, drafts off, and prompt
cache disabled. Two 128-token concurrent requests for each of two prompts, repeated twice, matched their solo token
hashes (4/4 per prompt); the serial controls also matched (1/1 per prompt), with no failed or unmeasured requests.
Both ranks logged decode graph replay. This test had no prefill graph replays and is not a performance comparison;
the line-count/reserve settings are admission accommodations, not defaults. Attempts at 76 lines and at the 76-line
single-stream graph budget were rejected at startup for insufficient memory; reducing the pool to 68 lines admitted
the two-stream run.

A separate 1 GiB prompt-cache run through `tools/bench_cache.py` used the same TP=2/two-stream BF16-paged graph setup,
a 2,571-token completion prompt, 64 output tokens, two repeated requests and two concurrent requests. All outputs
matched the first response hashes, but every request reported `cache_state=cold`, `cached_tokens=0`, zero stored
states and zero hits; the strict cache gate therefore failed qualification. This completion fixture does not
invalidate the previously passing chat warm/branch/rewind cache gate, but cache reuse for this fixture remains
unproven and the difference in request path needs investigation. No speedup is claimed for this correctness fix.

## Matched 5a94359c TP2 prefill comparison (2026-10-22)

An exclusive four-run ABBA comparison (baseline/current/current/baseline) tested the pushed `5a94359c` source and
current `03d4dbf3` source on both RTX 5060 Ti cards. Each sample measured one 16,384-row deterministic prefill
forward with TP=2, CUDA prefill graphs, BF16 KV paging, blocking NCCL, 4,096-row deep chunks and overlap enabled.
The baseline averaged 13,165.4 ms; current averaged 12,550.1 ms: **4.67% lower latency / 4.90% higher rows/s**.
All four rank-pairs replayed 168 graphs and produced the same cross-revision logits hash
`662339a95f0de2d7`; expert copies and pass counts were unchanged. This is a measured prefill gain over the named
baseline, not a serving or multi-context result, not attributable to one isolated commit, and far below 2×. It
does not establish decode improvement. The matched graph-decode ABBA completed: the steady phase measured
22.50/22.47 ms per round on baseline and 22.45/22.50 ms on current (means 22.485 vs 22.475 ms); four-row windows
measured 36.66/36.59 ms baseline and 36.65/36.64 ms current. Each run captured two graphs and replayed 252 times
in the steady phase. The average round-time difference is 0.04%, noise-level rather than an improvement. This
profile used a fixed 16-token prompt and 64 rounds; BF16 paging was enabled, but the short context caused no KV page
misses. An earlier exploratory run selected eager mode by mistake and is excluded from this graph-on comparison.

### Quantified gap to the 2× objective

On these matched workloads, doubling prefill throughput against `5a94359c` requires reducing 16k prefill latency from
13.1654 s to at most 6.5827 s (about 2,489 rows/s); the measured `03d4dbf3` result is 12.5501 s, so another 47.5%
latency reduction is required. Doubling graph decode throughput requires reducing the measured 22.485 ms/round to at
most 11.2425 ms; `03d4dbf3` measured 22.475 ms, essentially unchanged. The decode fixture is short-context with no
KV page misses, and the prefill fixture is a warmed synthetic forward; these thresholds quantify those fixtures only,
not long-context HTTP serving. No current result establishes the 2× objective.

## 2026-10-22 16k paged graph/eager miss diagnostic

A current-head diagnostic on `d467247b` used TP=2, BF16 paging, a 16,384-token repeated-QBench-ID fixture, 64
decode steps, and an eager/graph/graph/eager sequence. This is a synthetic repeated-token stress profile, not a
real-text or serving benchmark. Mean measured latency was 34.735 ms/token for eager and 34.650 ms/token for graph
(0.25% lower, within noise). All four runs returned hash `4c2d8973e43477cd`; graph runs replayed an already captured
graph 64 times with no captures or updates. Rank 0 recorded 3,616 expert misses/copies, 230,795 KV-page misses in
768 page calls, and no rebases or overflow in every run. The controlled ordering confirms separate miss telemetry
and graph/eager output parity under paging, but no graph speedup.

## 2026-10-22 persisted expert-placement A/B

An orderly-shutdown TP=2 HTTP gate verified that both ranks save and restore their lifetime expert-placement profiles
across process restarts. Rank 0 warmed 2,942 expert lines and rank 1 warmed 3,140. A short-prompt ABBA comparison
then used the same 128-token input and 64-token greedy output, TP=2 on both RTX 5060 Ti cards, CUDA graphs enabled,
BF16 KV paging enabled, and identical full output hashes (`c538c2913c77b8713a3f73bf289d20e253144aa010ae46452329fe19d5fdb74a`).
The warm-profile runs averaged 1,767 ms TTFT and 27.72 ms/token decode; the two independent cold-profile runs averaged
1,943 ms TTFT and 29.63 ms/token. This is 9.0% lower TTFT and 6.5% lower decode latency for this short-prompt,
no-drafts workload. It is a measured cache-placement gain, not a 2x result and not evidence for long-context or
drafted decode. Each restart also spent about 16.6 s per rank fingerprinting routed weights before serving; that
startup cost is outside request latency and remains an optimization target. No production default or source change
was made for this experiment; the existing opt-in persisted placement path was measured as implemented.

Two follow-up optimization screens do not qualify for production integration. First, an experimental GPU-built
worklist on an 8,192-row TP2 graph prefill compared host/GPU/GPU/host planning. Mean wall time was 5,159.7 ms for
the host planner and 5,114.0 ms for the GPU worklist (0.89% lower latency); all four runs and both ranks matched
hash `ffcf463b229c2aa6`, with 72 graph replays, 12,513 copies and 570 passes per sample. CPU planning fell from
18.35 to 2.30 ms, but the path still downloads counts and synchronizes every layer; measured pick-wait intervals
remained about 4.77–4.81 s and include GPU dependencies. KV paging was enabled but had zero page misses at this
8k shape. The small end-to-end delta does not justify integrating the prototype as the requested staging pipeline.

Second, an isolated SM120 asynchronous activation double-buffer kernel candidate was tested in TP2 graph prefill
at 16,384 rows and 4,096-row deep chunks, with `prompt_chunk=2048`, BF16 paging enabled and greedy decode disabled.
The baseline/candidate/candidate/baseline means were 10,711.4 ms and 10,677.4 ms (0.32% lower latency). All runs
reported hash `4b9f5c87b87ad84a`, with 96 graph replays and 48 captures per rank. The candidate dispatched only for
the 2,560×10,240 mul1 5-bit projection; other shapes fell back to the existing kernel. The delta is too small to
integrate, and the isolated prototype remains outside the pushed branch.

Third, an experimental exact-order TP2 prefill collective-submission grouping was tested on the same 16,384-row,
graph-enabled workload with BF16 paging and four exclusive ABBA samples (group size 1, 2, 2, 1). Both ranks
preserved hash `662339a95f0de2d7`; each run replayed 168 graphs and retained identical expert-copy/pass counts.
Group size 1 averaged 12,538.7 ms, while group size 2 averaged 13,391.1 ms (6.8% higher latency). The profiled
pick-wait interval also rose from about 11.87 s to 12.71 s. Host tests and CUDA build passed, but GPU results do
not support the change, so the prototype remains unintegrated and is not pushed.

Fourth, an exact-order 32-column-fragment EXL3 linear candidate for K=2560, N=10240, 5-bit mul1 was validated on
both RTX 5060 Ti cards. Four ABBA trials per card, with 128 iterations per sample, covered M=1 and M=4, graph and
eager replay, and resident and rotating weights. Exact output parity passed for F16/BF16/FP32 inputs and outputs,
bias, and both supported trellis layouts. In graph replay, resident-weight latency fell from 44.82 to 26.74 us at
M=1 (40.3%) and 45.35 to 27.46 us at M=4 (39.4%); rotating-weight latency fell from 61.23 to 42.89 us at M=1
(30.0%) and 61.58 to 43.16 us at M=4 (29.9%). This isolated kernel gain did not translate to serving: a TP2
HTTP ABBA (baseline/candidate/candidate/baseline), with graphs and BF16 K/V RAM paging enabled, generated 128
greedy tokens from the 16-id prompt. All runs had hash `2c5d814cfa80`, and drafted/undrafted outputs matched.
Undrafted decode averaged 42.81 tok/s baseline versus 43.41 candidate (+1.4%); drafted decode averaged 56.33
versus 56.28 tok/s (-0.1%). Prefill graphs were not exercised by this short-prompt gate. The end-to-end differences
are within run variation. A follow-up used the same four-server ABBA design and 16-id prompt but generated 512
tokens per measured request. With the same decode-graph/BF16-paging configuration, all output hashes were
`b170e8433207` and drafted/undrafted outputs still matched. Undrafted decode averaged 43.08 tok/s baseline versus
43.53 candidate (+1.0%); drafted decode averaged 56.85 versus 57.01 tok/s (+0.3%). Decode graphs replayed in every
leg; prefill graphs did not run. The active BF16 page window was 4096 tokens, so this was not a test of KV page
misses. The longer output does not change the conclusion: the end-to-end differences remain within run variation,
so the runtime dispatch stays out of the PR; the isolated candidate is not promoted.

The CPU AVX2 investigation is not ready for integration. A single-core, pinned feasibility harness executed a full
synthetic 3-bit expert and matched the established F64 oracle bit-for-bit, but took a median 2.35 s/expert; most time
was materializing rotated F64 weights. That is about 17,400× the 135 us estimated H2D payload floor, so this
materializing implementation is rejected. It is not a CUDA-parity result or an impossibility result for a fused
packed AVX2 kernel. A TP2 decode-gate PCIe telemetry sample on PCIe 4.0 x8 links measured peak Rx at 3.4 GB/s on
rank 0 and 2.5 GB/s on rank 1; during the active decode segment the averages were 1.8 and 0.85 GB/s respectively.
These are well below nominal x8 link bandwidth, so the sample does not support PCIe saturation as the current decode
bottleneck. Nsight Compute counters required admin access (`ERR_NVGPUCTRPERM` for the unprivileged run); collected
on card 0 for the synthetic 2560x10240 dense linear, baseline and no-prefetch each read about 16.44 MB from DRAM.
The reported elapsed DRAM-throughput metric was 56.69% of peak for baseline and 66.35% for the candidate. This
does not show HBM saturation for that kernel, but it is not a whole-model or KV-paging profile. A routed expert is
about 1.758 MiB packed and 9.83 MFLOPs per decode row. Any further CPU candidate must reproduce CUDA's
FP32/MMA/activation rounding, include activation round-trip, scheduling and synchronization, and beat the current
GPU miss route end-to-end. Start with sparse decode
misses, not prefill (where an uploaded expert is reused across many rows); never skip or approximate a routed expert.
If RAM bandwidth is already occupied by KV paging, CPU execution may make things worse.

## Decode critical-path profile and next kernel experiment

An Nsight Systems trace from the current tree (`a939413c`, with the inspected EXL3 linear, expert and TP2 collective
sources unchanged from `5a94359c`) measured 64 warmed single-token rounds and 64 four-row verification windows.
The short-context single-token phase averaged 23.10 ms/round, with about 22.01 ms between first and last GPU kernels;
the verification windows averaged 38.73 ms. These test-harness intervals exclude MTP draft generation/acceptance.
Although KV paging was configured, the trace contained no `kvp_*` fetch kernels, so it does not characterize long-
context paging misses.

The largest active decode costs were approximately 5.7 ms/round in dense EXL3 projections and 4.3 ms in BF16 HC
matrix products. The repeated `tf_exl3_linear_10_2_8` launch (K=2560, N=10240, 5-bit mul1, one row) used 80 CTAs,
256 threads and 147 registers/thread, at roughly 69 us per call across 36 calls/round. A kernel-only no-prefetch
candidate keeps launch geometry, MMA, split/reduction and rounding unchanged. On both RTX 5060 Ti cards it passed
byte-exact output checks at 1 and 4 rows. Seven ABBA trials measured graph replay at 63.5 us baseline versus
58.0-58.2 us candidate for one row (-8.4 to -8.7%), and 63.7-63.8 versus 58.2-58.3 us for four rows
(-8.5 to -8.7%). Rotating weights improved 9.2-9.4% and 9.6-9.8%, respectively; warm-weight latency improved
about 28%. The default-off runtime dispatch was then exercised through the TP2 serving gate with graphs enabled.
Baseline and candidate preserved the same `2c5d814cfa80` output hash, and drafted/undrafted outputs matched on every
prompt. Across the two 128-token samples, plain decode was 41.93/46.41 tok/s baseline and 42.05/46.81 candidate;
drafted decode was 63.40/66.47 baseline and 63.02/66.39 candidate. This is within run variation and provides no
reliable end-to-end improvement. Keep the candidate default-off and do not promote it into the PR absent a larger,
repeatable serving win; the isolated kernel gain does not move the 2x objective materially.

Per-rank mapped-copy kernels occupied 2.11/2.58 ms per round, grouped expert kernels 2.25/2.67 ms, while all-reduce
kernel sums were 3.60 ms on rank 0 and 2.66 ms on rank 1. This suggests rank waiting/imbalance, but does not isolate
NCCL transport. Recorded H2D DMA was only about 349 KB / 0.19 ms during the single-token phase because mapped
expert loads occur inside kernels; those DMA counters do not establish PCIe saturation. Even halving both dense
EXL3 and HC active time would indicate only roughly 1.3× before secondary effects. The 2× objective therefore
requires a combined system-level improvement; no single bottleneck or current optimization proves it.

A 128-row dense-prompt kernel prototype was also tested directly on the RTX 5060 Ti. Its GPU fixture passed exact
output, FP32/BF16, tail and memory-guard checks across all four traced projection shapes; the candidate used 128
registers and 34,816 bytes of shared memory with no local spill. However, paired timings (seven samples, 32 launches
per window, eager and graph, warm and rotating weights) were slower than the existing 64-row tile on every shape:
about 1.01× to 1.31× its latency. It is rejected and is not integrated or pushed. Reusing more decoded weights per
CTA increased pressure enough to erase the expected benefit. A 64-row activation-pipeline alternative remains a
prototype investigation; no dense-kernel speedup is claimed.

An opt-in BF16 HC up-projection/mix fusion was screened with a standalone CUDA exactness test on sm_120. It passed
byte-exact projection, rounded-product, mixed-output and FP32 reduction checks, including changed-input graph replay.
Four eager/captured ABBA pairs on synthetic nonzero BF16 weights measured baseline versus fused per-call latency at
rows 1/4/8/16: eager 7.45/7.64/8.13/10.09 us versus 9.15/9.43/10.82/16.39 us; graph 8.22/8.22/10.25/12.28 us
versus 10.27/10.41/12.31/17.55 us. The candidate lost at every width and is rejected; these are isolated synthetic
kernel timings, not model-weight or serving measurements. The one-launch design is not sufficient to offset its
resource/occupancy cost on the RTX 5060 Ti.

## 2026-10-23 expert-copy and overlap follow-up

An isolated `fn_exl3_dcopy` grid-size screen used the existing CUDA test fixture on both RTX 5060 Ti cards. For 10
expert lines (17.58 MiB), 36–576 blocks measured 1.458–1.466 ms (12.6 GB/s); the equivalent pinned H2D DMA fixture
measured 1.352 ms (13.6 GB/s). Results were effectively identical across cards and block counts, so changing copy
kernel grid size is rejected. The transfer path is near the measured link ceiling; reducing bytes/misses is more
promising than adding blocks.

The existing side-stream split that computes resident experts while mapped misses copy was tested graph-on in a
four-leg TP2 HTTP ABBA (off/on/on/off), with shared BF16 paging, 38 lines/layer, `--parallel 2`, blocking NCCL,
prompt cache disabled and two 128-token samples per draft mode per leg. All outputs matched hash `2c5d814cfa80`,
golden responses matched and MTP acceptance stayed 87/115. Undrafted means were 35.07 tok/s baseline versus 34.96
overlap; drafted means were 44.75 versus 45.13 tok/s. The -0.3% / +0.8% differences are within run variation, so
overlap remains opt-in and is not promoted. The short prompts produced no prefill graph replays.

An opt-in `TF_FLASHNEXT_CACHE_TELEMETRY=1` run on the same short-context configuration collected per-step cache
counters (the boundary synchronizations add overhead, so its tok/s values are not performance measurements). Across
323 decode and 107 verify samples per rank, device expert-pick miss fractions were 15.4% on rank 0 and 17.5% on
rank 1; BF16 KV block misses were zero. This confirms meaningful expert-cache misses but does not exercise long-context
KV paging. Next, measure per-layer miss distributions and compare fixed versus frequency-informed line budgets under
the same graph-on TP2 ABBA; retain only a repeatable serving gain with exact outputs and unchanged BF16 paging/cache
semantics. CPU AVX2 is not the next step: transfer utilization and the prior exact-but-slow CPU prototype do not
support it.

A separate opt-in telemetry diagnostic then submitted two concurrent 30,000-ID prompts at TP2 with shared BF16 KV
paging, context 32,768 and 38 expert lines/layer. The prompt IDs were formed by repeating two golden-fixture ID lists;
this tests paging mechanics, not natural-text quality. Both requests completed and both ranks reported page fetches and
rebases. Across 60,000 prompt rows, each rank counted 4,284,422 BF16 block requests and about 302,000 misses (7.05%,
about 2.47 GB logical fetch payload per rank); the two-token verify phase counted 678/684 misses and about 5.55/5.60
MB per rank. This is direct evidence that BF16 KV paging reaches the RAM-backed sparse-fetch path at long context.
Telemetry adds synchronization, so no throughput claim is made. Decode graph mode was enabled, but this short output
did not produce a replay counter; prefill graphs were not enabled, so this run is not graph-performance evidence.

An opt-in per-layer extension to that telemetry reused the existing planner counters and reported each rank's
routed hit/miss picks by decoder layer. On a TP2 graph-on HTTP run with `--parallel 2`, 32,768-token configured
context, the 128-token HTTP gate, and prompt cache disabled, aggregate pick-miss rates were rank 0: prefill
20.05%, decode 9.72%, verify 7.56%; rank 1: prefill 18.79%, decode 8.47%, verify 5.80%. Highest decode miss counts
in this short fixture were rank 0 layer 1 (339) and layer 6 (248), and rank 1 layer 0 (328) and layer 21 (240).
These are aggregate diagnostic counts, not a natural long-context distribution; opt-in telemetry synchronization
means the run is not a performance benchmark, and it did not measure KV paging misses. Drafted and undrafted outputs
matched on every prompt; two of six prompts did not match the strict golden check. Next use these counts to screen a
fixed-total, minimum-floor per-layer line-budget allocator, then validate under an uninstrumented TP2 ABBA with exact
output and BF16 paging/cache-reuse gates before keeping any policy change.

## 2026-10-23 persisted expert placement under BF16 paging

A fresh per-rank placement profile was collected from the same six-prompt fixture and then compared against cold
placement in an uninstrumented TP2 ABBA. Both runs used the same server binary, `--parallel 2`, 32,768-token context,
graphs enabled, BF16 KV paging/shared allocation enabled, prompt cache disabled, and fixed pool budgets of 38 lines
on rank 0 and 49 on rank 1. Each gate requested 128 output tokens; the client benchmark requests were sequential,
so this is not a concurrent-request benchmark. Full serving output hashes were represented by gate hash
`2c5d814cfa80` in all four runs; drafted output equaled undrafted output, and two of six prompts failed the strict
golden check in every configuration.

| Decode mode | Cold map | Fresh map | Throughput change |
|---|---:|---:|---:|
| Drafts off | 35.19 tok/s | 38.04 tok/s | +8.1% |
| Drafts on | 47.07 tok/s | 48.92 tok/s | +3.9% |

Separate opt-in telemetry runs with the same prompt fixture measured combined decode/verify device-pick misses at
15.43% → 14.22% on rank 0 and 11.89% → 10.44% on rank 1. These telemetry runs include boundary synchronization
and are used only for miss counts, not throughput. Decode graphs replayed in the throughput runs; this short fixture
did not establish long-context KV-page misses, prefill-graph benefit, or two simultaneous live requests.

The next miss-reduction candidate is a fixed-total per-layer line budget informed by these measurements, not a larger
rank-wide allocation. The device planner and prefill-ring layout require variable-sized planner-state offsets and
graph-safe launch/state handling. A local ragged-pool prototype was screened below, but is not committed or promoted.
The exact startup budget must preserve BF16 KV, graph scratch, and the minimum pool floor.

A profile from an older workload was also tested and regressed this fixture: misses rose to 19.50% on rank 0 and
17.13% on rank 1; its uninstrumented ABBA averaged 1.7% lower undrafted and 4.0% lower drafted throughput than cold
placement. Profile reuse is therefore workload-sensitive. This experiment validates the existing opt-in profile
mechanism for a matched prompt mix; it does not justify enabling historical profiles by default or claim a general
serving speedup. Next work: make profile freshness/recency safe, then repeat with a longer representative workload
and simultaneous requests before promoting a placement policy.

## 2026-10-24 profile-sized line-budget screen

The ragged-pool prototype fixes the device picks-buffer capacity from the minimum line floor, so profile-weighted line
counts preserve the modeled variable CUDA allocation exactly. On the two TP2 ranks, startup logged 4,806,791,424 and
5,780,005,248 bytes allocated, equal to each rank's uniform baseline; pool sizes ranged 30–73 lines (rank 0, uniform
54) and 44–84 (rank 1, uniform 65). This is an allocator-budget equality, not a full independent VRAM audit.

An uninstrumented graph-on cold/profile/profile/cold ABBA used two concurrent mixed requests, 256 generated tokens,
32,768 configured context, prompt cache off, and shared BF16 KV. Aggregate results were 34.5/34.8/34.7/34.5 tok/s;
each leg had zero failures and exact alone/serial equality, with the same two output hashes. The profile arms averaged
34.75 tok/s versus 34.50 for the controls (+0.7%), which is not a reliable serving gain.

A separate matched telemetry profile/control pair used the same workload and output hashes. Verify device-pick misses
changed from 110,253 to 101,117 on rank 0 (−8.3%) and from 88,309 to 91,727 on rank 1 (+3.9%): combined verify misses
fell 2.9%. Combined decode misses increased 3.5%. Telemetry synchronizes at boundaries and is only miss evidence;
these counters are rank-asymmetric and do not establish a robust total-miss or throughput improvement. Keep the
allocator experimental. Next, score candidate budgets by replaying ordered demand through the production admission and
eviction policy, then repeat with held-out concurrent prompts before considering promotion.

The held-out check used two different concurrent code/chat prompts with the profile trained on the original pair.
Under telemetry, profile versus uniform measured verify misses of 121,127 versus 113,923 on rank 0 (+6.3%) and
117,102 versus 98,834 on rank 1 (+18.5%); combined verify misses rose 11.9%, and combined decode misses rose 26.8%.
The uninstrumented aggregate was 33.1 versus 33.9 tok/s (−2.4%). All legs had zero failures and identical per-request
hashes. This falsifies generalization for this profile-sizing heuristic; do not promote it. Review also found two
contradictory/stale expectations in the new allocator tests and that profile loading would activate sizing without an
explicit opt-in. The prototype remains an experiment; the next candidate is ordered-demand replay, not another
histogram tweak.

The separately committed recent-demand eviction policy (`7d0970d35`, default-off) was then tested on the same held-out
pair in a graph-on TP2/c=2 ABBA. Uniform legs averaged 34.45 aggregate tok/s; recent-demand legs averaged 33.50 tok/s
(−2.8%). A matched telemetry pair measured 270,350 combined decode/verify misses with recent-demand versus 259,479
with uniform placement (+4.2%); verify misses rose 3.6% and decode misses 6.8%. All four throughput legs and both
telemetry legs had zero failures and identical per-request hashes. Keep recent-demand default-off; it helped the original
pair but did not generalize to this held-out mix.

Review also identified a startup-ledger undercount after adding the recent-demand state: the old estimate omitted 4,610
u32 words per planner pool (18,440 bytes), or about 0.84–0.86 MiB/rank for the current pool counts. The local startup
ledger now derives state bytes from the planner layout. This is a correctness/accounting fix, not a cache-speed claim.

### Next expert-miss experiment

On 2026-10-24, an opt-in DPLAN v3 diagnostic captured 1,024 decode rounds per pool in a 1 MiB trace and exactly replayed them. The TP2 graph-on run used BF16 paged KV, a 10-line expert pool, and one golden prompt on the direct Engine path (not the shared-arena server path). All 96 decoder-pool traces (48 per rank) had zero drops and 1,024/1,024 verified transitions; both ranks produced output hash `da8b74115bc2f3bc`. Decode graphs replayed 1,018/1,024 rounds per rank. TP2's rank masks split 491,520 routed picks across both ranks (236,927 on rank 0; 254,593 on rank 1). The production lifetime policy recorded 122,984 misses/copies on rank 0 and 137,392 on rank 1: 260,376/491,520 combined picks (53.0%), with 231,144 hits. The offline evaluator's lifetime copy count exactly matched production. Combined LRU replayed 293,031 copies (+12.5%); decayed frequency replayed 258,797 (−0.61%, 1,579 fewer). The previous interim note mistakenly doubled the denominator; the rank-specific counts above are verified from all 96 trace files.

A second exact TP2 graph-on trace used 56 expert lines/layer, matching the recent serving configuration, with the same 1,024-token single-prompt fixture. All 96 traces replayed 1,024/1,024 transitions with matching rank hashes. Production misses/copies were 17,629 on rank 0 and 21,477 on rank 1 (39,106/491,520 combined picks, 7.95%). Offline decayed-frequency replay reduced copies to 37,753 (−3.46%); guarded admission reduced copies to 33,720 (−13.8%) while mapping 4,023 cold picks rather than copying them. These are trace counterfactuals, not measured throughput wins; mapped-read cost is not represented.

The v4 trace format adds strict checkpoints for device-planner ownership transfers. A sequential two-prompt run (512 decode rounds each, 10 lines/layer) passed exact replay on all 96 pools with two checkpoints and 1,024/1,024 transitions; the per-segment TP2 output hashes matched across ranks, with combined hash `1f63a7addea9e585`. Production recorded 81,671 misses on rank 0 and 89,854 on rank 1 (171,525/491,520 combined picks, 34.9%). This demonstrates reuse across sequential prompt segments only—not concurrent serving, throughput, or 200k context. No policy is promoted until checkpoint-aware counterfactuals and uninstrumented serving tests pass.

Checkpoint-aware offline replay now cross-checks each trace's lifetime-policy rounds, hit/miss picks, copies, evictions, mapped experts, and mapped picks against the exact device-planner replay; the 96-pool run passed this guard. Across its recorded device segments, LRU caused 182,921 copies (+6.64%); decayed-frequency caused 165,699 (−3.40%, rank 0 −3.55%, rank 1 −3.25%). Guarded admission cut simulated copies to 17,478 (−89.8%) only by mapping 167,186 cold expert picks. These are segment-conditioned counterfactuals: at each checkpoint candidates restart from the production state, and host-prefill intervals/costs are excluded. The mapped-read byte traffic and latency are not modeled, so guarded admission is not a speedup claim. Next, evaluate a default-off decayed-frequency runtime candidate and quantify actual mapped traffic on held-out concurrent prompts; retain the current policy unless both-rank misses and uninstrumented TP2 graph-on serving improve.

The held-out regressions reject both histogram-sized pools and recent-only eviction as general policies. Current DPLAN
telemetry records aggregate/per-layer hit, miss-pick, copy, eviction, and in-place totals, but those totals cannot
reconstruct planner order or reuse distance. The ring contains only a bounded recent window and is not a complete
trace. The planner also admits every miss when a line is available; it does not compare a newcomer against a resident
before evicting, so cache-pollution/bypass is not measured yet.

The bounded, opt-in DPLAN trace now records ordered picks and exact planner transitions, including strict checkpoints
when host prefill transfers pool ownership back to the device planner. Replay verifies every device segment and the
recorded checkpoint states; it deliberately excludes the host-prefill interval. Candidate policy replay is therefore
segment-conditioned, not continuous across requests. Keep tracing out of throughput runs. Next evaluate a default-off
decayed-frequency runtime candidate, count actual mapped-read traffic, and test on held-out concurrent prompts. Reject
dropped or discontinuous traces; preserve TP2 budgets, the 10-line floor, graph capture/replay, snapshots, cache isolation,
and BF16 KV paging. Promotion requires miss improvement on both ranks plus an uninstrumented graph-on serving gain
outside baseline variation. Experimental policies remain opt-in/default-off until then.

## What this adds

The EXL3 pack now runs through upstream's own Flash Next CUDA family and server (`tensorfold-native serve`),
on one card and at tp=2. Integration touches existing upstream files and adds new files under
`zig/src/families/flashnext/cuda_exl3_*.zig`, `zig/kernels/cuda/flashnext_exl3_*.cu` and `zig/tests/cuda/`.

| Area | What exists | Main files |
|---|---|---|
| Weights | EXL3 dense, shared, routed experts, 5-bit lm_head, PLE n-gram table read from the pack | `cuda_exl3_load.zig`, `cuda_exl3_store.zig`, `cuda_exl3_ple.zig` |
| Expert offload | Per-layer device line pool fed from a pinned host bank; LRU plus use count; misses copied on a side stream | `cuda_exl3_pool.zig`, `exl3_offload.zig`, `exl3_round.zig` |
| tp=2 | Expert parallel: rank 0 routed 0-255 (+MTP head), rank 1 routed 256-511 + shared expert; one fp32 all-reduce per layer; per-rank pools and half-banks; vocabulary-split lm_head | `cuda_exl3_tp.zig`, `flashnext_exl3_tp.cu` |
| Prefill | Expert-major deep chunks (thousands of rows), one ring of lines across all layers, 64-member tile kernel for the mul1 3-bit experts, tp=2 version with overlapped per-sub-block reduction, expert-major MTP head | `cuda_exl3_prefill*.zig`, `flashnext_exl3_prefill.cu` |
| MTP | Draft rows, drafts with the prompt cache at tp=2, rank 1 keeps no head cache in its windows | `cuda_exl3_mtp.zig`, `cuda_rank_engine.zig` hooks |
| Long context | Attention head split across ranks (12 query heads + 1 KV head each), indexer keys in mapped pinned host memory, 1024-row chunk, prompt states parked in host memory | `cuda_exl3_heads.zig`, `cuda_exl3_kv.zig`, `cuda_snap.zig` |
| KV paging | Full BF16 K/V of the 12 attention layers in pinned host memory with a device window and block cache; allocation/budget geometry now unified, default off | `cuda_exl3_kvpage.zig`, `cuda_exl3_kv.zig`, `flashnext_exl3_kvpage.cu` |
| CUDA graphs | Eligible decode rounds/windows capture the 48-layer pass, NCCL sums and head gathers; prefill captures dense layer sub-blocks; greedy MTP proposals have a conditional graph; graph instantiation runs on a worker | `cuda_exl3_dplan.zig`, `cuda_exl3_graph.zig`, `cuda_exl3_chain.zig`, `cuda_exl3_recycle.zig` |

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
- **Near-200k context, 2 concurrent requests, tp=2, bf16 KV:** 195,938- and 195,957-token needle requests sent together
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
prompt (tp=1 deep: 1,531 / 1,461 / 1,395 / 818). The old batched pass ran at 100 - 210 rows/s. A prior note's
168 s “on one card” duration is near the tp=2 rate (about 163 s by the reported rate), not the tp=1 rate (about
244 s); its configuration is unverified, so it is excluded. Saved concurrent-run logs show 312.1 / 562.5 s without drafts
and 380.1 / 693.2 s with drafts; the earlier 304 / 554 s figures are not reconciled to those logs.

## What graphs do and do not give

For eligible shapes, decode graphs match eager bit for bit, but a repeatable performance win over the optimized
eager path has **not been demonstrated** on this hardware. The decode speedup came from removing host stalls: the
pool's per-layer host wait (replaced by the device planner), thread spawns in the n-gram gather, and the non-blocking
NCCL communicator's host cost. Once those are gone a step is bound by GPU kernels (profile: about 20 ms of kernels
per token; dense EXL3 projections 6 ms, bf16 hyper-connection lanes 4.5 ms, all-reduces 2 - 3.5 ms, experts 3 ms).
With drafts on, the first drafted request after start is slower in the recorded gate; graph instantiation now runs
on a worker, but graph execution still synchronizes for results and timings do not isolate first-use capture from
replay. A TP=2 projection-fork experiment preserved logits but did not produce a repeatable graph-specific gain.
Graphs stay opt-in (`TF_FLASHNEXT_EXL3_GRAPH=1`). Prefill graph captures now preserve sub-block reduction overlap
with external event-wait nodes and pass bit parity. At chunk sizes 1024, 2048 and 4096, graph throughput remained
within run variation of eager; the former 10% regression is removed, but a speedup is not proven. An 8192-row
chunk request OOMed and used fallback, so it is not counted. Prefill graphs remain opt-in
(`TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1`).

## Prefill graph replay follow-up (2026-10-20)

A TP=2, two-lane HTTP run at 16,040 prompt tokens used BF16 paging, MTP, snapshots and the default
`TF_FLASHNEXT_EXL3_GRAPH_PREFILL_AFTER=2`. Both ranks reported 72 prefill graph replays, 216 eager layer executions,
96 captures and about 314 ms cumulative capture time. Warm/rewind reused 16,033 tokens and branch reused 16,023;
all response hashes matched the cold baselines. The earlier 12k run's zero replays are explained by the threshold:
the available same-shape chunks only reached capture, with no later same-shape chunk to replay.

The DPLAN run and two graph runs measured cold prefill at 27.85/16.47 s, 27.92/16.39 s and 28.00/16.51 s across
the two concurrent lanes. This is within run/lane variation and is not a demonstrated graph speedup. A follow-up
keeps the complete exact replay key, but normalizes only the position and paging-window-origin hash for capture
sightings and driver-checked executable-update candidates; sequence/scratch/event ownership remains strict. The
updated gate retained exact output/cache hashes and reported 72 replays, 192 eager layer executions, 120 captures and
24 successful `Exec.update` calls per rank. Capture time was 325 ms; DPLAN/graph cold prefill measured 16.44/27.80 s
and 16.51/27.99 s. This exercises executable reuse across page-window changes, but shows no speedup; incompatible
topology still uses the instantiate fallback. Next profile the expert/pool host boundary and dense graph replay
versus eager time before making more topology changes.

## Switches

| Variable | Effect | Default |
|---|---|---|
| `TF_FLASHNEXT_EXL3_GRAPH=1` | Decode graphs (and prefill dense-half graphs where enabled) | off |
| `TF_FLASHNEXT_EXL3_GRAPH_PREFILL=1` | Prefill layer graphs at tp=2 | off |
| `TF_FLASHNEXT_NCCL_BLOCKING=1` | Blocking NCCL communicator (init has no deadline if the peer never joins) | off |
| `TF_FLASHNEXT_ATTN_SPLIT=0` | Replicated attention at tp=2 (the split is on by default) | split on |
| `TF_FLASHNEXT_IKC_HOST=1` | Indexer keys in mapped pinned host memory (needed for 200k x 2) | off |
| `TF_FLASHNEXT_KVPAGE=1` | K/V paging to host memory; takes precedence over the head split | off |
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
4. **200k x 2 remains deferred.** Rank-0 pool headroom is tight and simultaneous prefill can hurt decode. The prior
   near-200k resident-BF16 retrieval gate is not a claim for paged 200k capacity. Ordinary-length prompt-cache reuse
   now works with TP=2, BF16 paging, MTP and graph mode: two concurrent chat histories passed warm, branch and rewind
   phases with 4,122/4,139 prompt tokens reused and exact response-hash parity against cold-cache baselines.
5. **BF16 KV paging and snapshots now pass ordinary-length TP=2 correctness gates.** A two-GPU graph test restored
   two paged lanes and preserved exact state/logits; each rank replayed prefill graphs 72 times and decode graphs 5
   times. The HTTP cache gate proves actual snapshot reuse, MTP continuation and two concurrent requests. Still open:
   repeatable graph speedup/profiling, live-rank overflow and failure injection, and long-context paged capacity.
   200k × 2 is deliberately not the next gate. The user’s priority is to make graph execution and cache reuse robust
   and fast first. Int8 KV is out of scope; BF16 remains the primary and required format.
6. **200k reference parity is not established:** a saved 200k prefill comparison exited 1 with max logit
   difference 1.1406 and top-1 6009 versus 19251 (reference margin 0.25). The successful 200k × 2 needle answers
   establish retrieval/serving behavior for those prompts, not strict full-logit parity.
7. **Graph path:** graphs execute and replay for both prefill and decode, including TP2 BF16 paged snapshot restore.
   However, neither decode nor prefill graphs have a repeatable measured speedup yet. MTP verification also has a redundant GPU best-selection/readback
   boundary after full-logit readback. An opt-in verify-tail experiment preserved output/hash but showed no
   observable speed change in one TP=2 A/B pair; no graph speedup is claimed.
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

1. Meet the graph acceptance gate: reduce decode host boundaries (first target is MTP verify's redundant best-
   selection/readback), make decode graphs measurably faster than optimized eager decode, and extend prefill capture
   beyond dense sub-blocks without losing reduction overlap. Recheck with repeated TP=2 gates before changing defaults.
2. Exercise fail-closed paging overflow and collective cleanup on live TP2 ranks; snapshots/MTP rollback and
   ordinary-context cache reuse are implemented and pass their current correctness gates. Defer the 200k × 2
   acceptance run until graph execution and cache reuse are optimized.
3. Add an init timeout to the blocking NCCL communicator before making it the tp=2 default.
4. Make the HTTP and prefill gates fail on strict mismatches, compare full hashes, and require reference artifacts
   for long-context parity runs; investigate the 200k reference-logit failure.
5. Assert actual graph replay and compare full MTP draft-token sequences against eager; strengthen the draft-row
   test to assert the decision-critical top-1 output.
6. Margin study for the golden near-tie and for the head split (needs logprobs from the server).
7. Spread the MTP head across both ranks (rank 1 idles about 28% with drafts on); split the replicated dense
   work to speed up prefill at tp=2.
8. The single-core materializing F64/AVX2 prototype is rejected (2.35 s/expert). A fused packed CPU expert path
   remains unbuilt; only reopen it after the CUDA kernel experiments and only if a bounded CUDA-semantics-correct
   cold-decode test beats the GPU miss route end-to-end without harming BF16 paging. Never omit routed experts.
9. Split this branch into reviewable PRs after the TP=2 graph gates pass.

## Branch map

`exl3-c-main` merges: core adapter, pool, PLE, draft rows, expert-parallel TP=2, serving, split `lm_head`, deep
prefill, BF16 KV paging, snapshots/cache reuse, and CUDA graphs. Current design and validation notes:
`docs/plans/exl3-kv-paging.md` and `docs/plans/exl3-graphs.md`. Superseded v1.1.0 and int8-first planning memos
were removed from the draft because they describe obsolete decisions and conflict with the BF16 paging scope.

Not done on purpose: no change to `docs/recipes`, no family or model registration (held until the strict golden
gate passes).

## 2026-10-24 recent-demand device eviction

The graph/DPLAN path now has an opt-in 256-row, per-layer expert-demand ring (`TF_FLASHNEXT_EXL3_RECENT_EVICTION=1`).
Lifetime counts still seed initial placement and remain available to profile persistence; the recent score only affects
decode miss ordering and device-pool victim selection. The policy does not skip routed experts, alter logits, or enter
request snapshots. Host prefill allocation is unchanged. It is off by default.

Validation on par1-cs13 used the `Qwen3.8-Flash-Next-exl3-3.05bpw` pack, both RTX 5060 Ti cards (sm_120), TP=2,
32,768 configured context, CUDA graphs on, `--parallel 2`, shared BF16 KV paging on, and prompt cache off. Two concurrent
HTTP prompts generated 512 output IDs each with drafting disabled. With synchronized per-layer telemetry, baseline decode
misses were 31,844 on rank 0 and 25,446 on rank 1 (57,290 total); recent-demand measured 29,012 and 22,060 (51,072,
−10.9%). Both per-request output hashes matched between policies. This count comparison is diagnostic and is not used
for throughput.

Two uninstrumented base/recent pairs in opposite order used the same two concurrent prompts and output length. Mean
per-request decode was 27.97 tok/s baseline versus 28.53 tok/s recent-demand (+2.0%); all four output hashes matched.
This is a small result from one prompt pair, not a broad speedup claim. A separate graph-on, shared-BF16-paged, concurrent
512-output MTP-on run also matched both no-draft output hashes; telemetry misses were 729→726, so the larger no-draft
miss reduction does not apply to the MTP path in this fixture.

Validation also passed `zig build test-flashnext-exl3-host`, `zig fmt --check` on the changed Zig files, `git diff --check`,
and the native sm_120 build. The option remains experimental and default-off pending held-out prompts, longer sessions,
and broader MTP/profile interactions.

## 2026-10-24 prefill-graph churn attribution and two-chunk window ABBA

Setup for every leg below: par1-cs13, two RTX 5060 Ti, TP=2, `--parallel 2`, shared BF16 KV paging, decode and prefill
graphs on, MTP drafts on, prompt cache off, two concurrent 16,384-token prompts, 128 generated tokens, two warm-up and
two measured batches per leg. All 16 measured requests across the legs below produced the same token hash
`44f75f563088…`.

**Prefill graph churn is not the bottleneck.** With 38 expert lines and prefill graphs on, the server counters read
`shared lanes reassigned 0`, `admission misses 0`, and prefill graphs replay/eager/instantiated/updated 1200/96/240/0 with
`capture 440 ms` (rank 0) and `401 ms` (rank 1) for the whole run, against about 14 s of prefill per batch. The 96 eager
runs and 240 captures are a one-time fill (240 sighted, 96 run eagerly before instantiation); they did not recur in the
measured batches. The lane-reassignment hypothesis is refuted; even a perfect graph policy bounds the saving at well under
1 s.

**Prefill is bound by the expert stream over PCIe.** Opt-in cache telemetry (`TF_FLASHNEXT_CACHE_TELEMETRY=1`) showed rank 0
moving 427 GB through the expert ring for four batches (about 107 GB per batch, about 13 GB per 4096-row chunk), with
13% of ring copies unused. At 14.4 s per batch that is about 7.4 GB/s per rank over a PCIe 4.0 x8 link.

**The two-chunk window (`TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK=1`) shortens prefill when it is admitted.** At 38 lines per
layer it is declined (`extra_device=588070912`, free 6.36 GB against 6.60 GB protected). At 26 lines per layer it is
admitted. ABBA on one binary (two-chunk, one-chunk, one-chunk, two-chunk), 26 lines per layer:

| leg | mean prefill, measured batches | aggregate decode tok/s |
|---|---|---|
| two-chunk (A) | 12.503, 12.511 | 15.75, 15.74 |
| one-chunk (B) | 14.428, 14.422 | 14.92, 14.94 |
| one-chunk (B) | 15.732, 14.420 | 14.92, 14.97 |
| two-chunk (A) | 12.537, 12.543 | 15.71, 15.76 |

Two-chunk prefill is about 13% shorter (12.52 s against 14.42 s, ignoring the one 15.73 s outlier in a one-chunk leg), with the
same direction in both orders, about 15% more prefill throughput. Reference points at 38 lines with one chunk (separate
runs, not part of the ABBA): prefill 14.41 s, decode 15.87 tok/s. Two-chunk at 26 lines therefore costs about 0.8% decode
against the 38-line configuration. The higher decode throughput of the two-chunk legs over the one-chunk legs at 26 lines
(about +5.5%) is not a decode speedup: see the correction below.

This is a measured, correctness-preserving prefill gain for this workload, not the 2× target. It depends on trading 12 of 38
expert lines per layer for the window's extra device memory, and it was measured on one prompt shape and one machine. The
option stays default-off until held-out prompt shapes, longer prompts, and the MTP and snapshot paths are re-checked.

### Prompt-length and MTP sweep of the two-chunk window (same day)

Same binary, machine, 26 lines per layer and flags as the ABBA above; one server start per leg and order two-chunk,
one-chunk, one-chunk, two-chunk (A B B A). Per leg one warm-up batch (discarded) and two measured batches of two
concurrent prompts, 128 generated tokens. Every measured request returned the same token hash across all four legs in each
row (8,192 and 30,000 tokens without drafts, 16,384 tokens with request-level MTP drafts).

| prompt tokens | drafts | mean prefill s, two-chunk (4 batches) | mean prefill s, one-chunk (4 batches) | change |
|---|---|---|---|---|
| 8,192 | off | 4.925 | 6.47 (one 7.76 outlier excluded) | −24% |
| 30,000 | off | 25.92 | 29.50 | −12% |
| 16,384 | on | 15.37 | 15.23 (one 16.58 outlier excluded) | +1% (no gain) |

The window shortens prefill for every no-draft shape tried, with the same direction in both orders. With MTP drafts on it
did not at first: drafted prompts also ran as 8,192-row paired passes, but `head()` in `cuda_exl3_prefill.zig` declined any
call with more rows than the deep chunk (`dp.rows < n`), so the MTP head silently took the slow per-chunk path for every
paired pass. Aggregate decode for drafted 16,384-token requests was 11.5 tok/s in both modes, below the undrafted 15.7 tok/s,
which matches the earlier finding that MTP-drafted decode regresses on this setup.

**Head fix.** `head()` now splits a paired window's rows into two deep-sized slices (`headSplit`: the first aligned up to
`att_rows` and at most one deep chunk, the second no larger and not under `min_rows`, so the second slice cannot decline after
the first has run). The log line `the MTP head runs expert-major over deep chunks (4096 rows first)` confirms the deep head path
engages. Re-measured with the same ABBA procedure (two-chunk, one-chunk, one-chunk, two-chunk), 16,384 prompt tokens,
drafts on, 26 lines per layer, all 16 measured requests with token hash `44f75f56…`:

| leg | mean prefill s, two measured batches | aggregate decode tok/s |
|---|---|---|
| two-chunk (A1) | 13.177, 13.231 | 12.80, 12.78 |
| one-chunk (B1) | 15.214, 15.215 | 11.65, 11.66 |
| one-chunk (B2) | 15.211, 15.232 | 11.67, 11.67 |
| two-chunk (A2) | 13.179, 13.211 | 12.77, 12.81 |

Drafted prefill is 13.2% shorter (13.20 s against 15.22 s) and drafted aggregate decode is 9.6% higher (12.79 against 11.66
tok/s); the decode difference is again not isolated to a cause. One-chunk drafted prefill is unchanged by the fix (15.2 s).
Host validation: `zig fmt --check` passes and `zig build test-flashnext-exl3-host` passes on par1-cs25 including a new
`headSplit` test. These sweeps do not cover the snapshot path or prompts above 30,000 tokens; the window stays default-off.

**Snapshot gate.** `run-snapshot-gate.sh` (`tf-cuda-test flashnext-exl3-graph … snapshot-shared`, TP=2, shared BF16 paging, graphs
on, `TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK=1`) was rebuilt from the tree with the head change and passed on both ranks (exit
codes 0/0): shared-slot snapshot remapping, recycled generation, foreign windows/history/continuation, exact eager/graph
state and logits, prefill replays 112 and decode replays 3. This gate's prompts are short, so it shows the window-on build
keeps snapshot behavior intact; it does not by itself prove the snapshot path on 8,192-row paired passes.

### Admission thresholds and a 60,000-token prompt

The window needs `extra_device=588070912` bytes and is admitted only when `free >= protected + extra`. Free memory stayed at about
6.35–6.37 GB in every probe; what moved was `protected`, which falls about 69 MB per expert line per layer. Probes
(TP=2, 32,768-token context unless noted, window on): 38 lines declined (protected 6.575 GB), 38 lines with a 17,408-token
context declined (6.575 GB; a smaller window does not help), 29 lines declined (5.953 GB, still about 0.18 GB short), 26 lines
admitted. At a 65,536-token context 26 lines is declined (protected 6.089 GB) and 21 lines is admitted. The practical rule on
this card is therefore a loss of 12 lines per layer at a 32k context and 17 at a 64k context.

A 60,000-token prompt at a 65,536-token context and 21 lines, ABBA on one binary (two concurrent prompts, 128 generated tokens,
drafts off, one discarded warm-up batch and two measured batches per leg), all measured requests with token hash
`566ecfab…`:

| leg | mean prefill s, two measured batches | aggregate decode tok/s |
|---|---|---|
| two-chunk (A1) | 60.44, 60.59 | 5.30, 5.29 |
| one-chunk (B1) | 65.37, 65.45 | 5.30, 5.30 |
| one-chunk (B2) | 65.37, 64.05 | 5.30, 5.30 |
| two-chunk (A2) | 60.62, 60.67 | 5.29, 5.29 |

Prefill is 7.3% shorter with the window (60.58 s against 65.0 s; 65.0 is the mean of all four one-chunk batches). The gain
shrinks with prompt length (24% at 8k, 13% at 16k, 12% at 30k, 7% at 60k), consistent with the fixed dense and attention work
growing relative to the expert stream. At this length the prefill graph counters show 1330 replays, 1438 eager runs and 1141
admission misses against a 112-entry graph cache, so very long prompts mostly run eagerly. Decode is unchanged by the window.

### Correction: the window gives no decode gain

The decode figures above that favored the window (about +5% to +10%) are most likely a prefill/decode interleaving effect, not a
faster decode. With two concurrent prompts, the first request decodes while the second is still prefilling, and the
reported per-request and aggregate decode rates include that overlap. A shorter prefill ends the overlap sooner, so both
requests report a higher rate (7.86/20.5 tok/s with the window against 7.47/19.5 without, TTFT 9.9/20.0 s against 10.4/21.0 s).
Supporting evidence: decode-phase expert cache counters were identical between window-on and window-off server logs
(25,152 plans, 176,210 hit picks, 68,222 miss picks, 66,693 device copies, a 72.1% device hit rate at 26 lines), and decode
graph coverage was the same (468 round-2 replays, 8 eager). This was not confirmed with a decode-only run. Treat the window as
a prefill optimization only; no decode improvement from it is claimed.

### How much of decode is exposed expert-copy time (estimate)

Two decode-phase telemetry runs on the same binary and workload (TP=2, graphs, shared BF16 paging, two concurrent 16,384-token
prompts, 128 generated tokens, window off) differ only in expert lines per layer:

| lines per layer | device copies per decode step (rank 0) | device miss rate | sum of per-request decode tok/s |
|---|---|---|---|
| 38 | 95.3 | 20.8% | 29.7 |
| 26 | 127.3 | 27.9% | 26.95 |

Treating step time as proportional to 1/tok/s, 32 more copies per step cost about 3.4 ms, or about 0.106 ms per copy. A copy is
one 1.76 MiB line, so that is roughly 17 GB/s: close to the PCIe 4.0 x8 link rate, which means these copies are almost entirely
exposed (not hidden behind compute). At 38 lines that puts about 10 ms of a roughly 34 ms step (about 30%) on expert misses.

Consequences (estimates from two operating points, not a controlled sweep): removing every miss would cut step time by at most
about 30% (about 1.4× decode), so cache policy alone cannot reach 2×, and a 6% reduction in copies, the size of the best offline
replay result, is worth about 2% of step time, which is inside run variation. Large decode gains need either hiding the copies
(a router-independent prefetch of the next layer's likely experts) or shrinking the 23–24 ms non-miss part of the step. Neither
has been attempted.

### Next-layer expert prefetch: predictability from a DPLAN trace (negative result)

To size a prefetch that hides decode expert copies, a default-off dump was added to `cuda_exl3_dplan.zig`: with
`TF_FLASHNEXT_EXL3_DPLAN_TRACE=1` and `TF_FLASHNEXT_EXL3_DPLAN_TRACE_DIR=<dir>`, each pool's raw v4 trace buffer is written to
`dplan-<pid>-pool<N>.bin` when the planner is destroyed (only a gracefully stopped rank dumps; a rank killed with SIGKILL does not).
`tools/exl3_dplan_predict.py <dir>` parses the files and replays them.

Capture: rank 0 of a TP=2 graph-on server (26 lines per layer, shared BF16 paging, window off), two concurrent 4,096-token
prompts, 700 generated tokens each, drafts off. 48 decoder pools retained 1,074 aligned rounds (rows per round equal across all
layers in every round); the trace has a discontinuity from request ownership transfers, and 1,026 later rounds were dropped by the
bounded buffer.

Method: train a co-occurrence table on the first half of the rounds (picks of layer L, same round, to picks of layer L+1), then on
the second half predict the top-N experts of layer L+1 from layer L's picks, excluding experts already resident in a simulated
26-line LRU cache, and compare against that layer's real misses:

| prefetched experts per layer | share of misses covered | share of issued copies that were needed |
|---|---|---|
| 2 | 30.9% | 13.0% |
| 4 | 46.8% | 9.8% |
| 8 | 63.6% | 6.7% |
| 16 | 78.7% | 4.1% |
| 32 | 88.9% | 2.3% |

Even the best precision wastes 87% of the copies, and at the measured 0.106 ms per copy (near PCIe line rate) the wasted
traffic costs more than the saved misses. For example top-2 issues 2 copies to save about 0.26 misses per layer-round: about
0.21 ms of link time against about 0.03 ms saved. This predictor does not support a prefetch. The next-layer prefetch is
rejected as a decode optimization on this evidence.

Limits: the prompt is a repeated golden-fixture ID list, so generation is repetitive and the simulated LRU missed only 10.9%
of needed experts against 20.8–27.9% in the serving telemetry; a diverse workload would lower predictability further, not raise
it. One rank, one capture, a plain LRU model rather than the real planner, and a single co-occurrence predictor were tried.

### Two-chunk window on real, different prompts per request

All earlier window measurements sent the same repeated-ID prompt to both requests. This ABBA (two-chunk, one-chunk, one-chunk,
two-chunk; same binary, 26 lines per layer, 32,768 context, TP=2, graphs, shared BF16 paging, drafts off, 128 generated tokens)
sent two different text prompts built from the checkout's own files: request 1 is Zig source (16,008 tokens), request 2 is
markdown docs and test sources (12,758 tokens). One discarded warm-up batch and two measured batches per leg. Token hashes
were identical across all four legs for both requests (`1c2777f5…`, `a65bd936…`).

| leg | request 1 prefill s | request 2 prefill s | batch wall s |
|---|---|---|---|
| two-chunk (A1) | 17.95, 17.95 | 8.14, 8.14 | 26.52, 26.51 |
| one-chunk (B1) | 21.19, 21.25 | 10.66, 10.70 | 30.35, 30.35 |
| one-chunk (B2) | 21.20, 21.27 | 10.68, 10.71 | 30.33, 30.33 |
| two-chunk (A2) | 13.16*, 17.99 | 8.16*, 8.17 | 26.62, 26.62 |

(*one measured A2 batch reported a lower prefill for each request, but the same batch wall time; the lower figures come from the
two requests' prefills overlapping differently in that batch and are not used.) Two-chunk shortens the whole batch by 12.5%
(26.5 s against 30.3 s), request 1 prefill by 15% and request 2 prefill by 24%, with identical outputs. The gain therefore holds
for different, non-repeating prompt content and unequal prompt lengths, not only for the repeated-ID fixture.

### Trimming the graph reservation to fit the window at more lines (dead end)

The window's admission gap at 38 lines is about 0.8 GB. The graph budget is a heuristic (reservation 660 MiB plus headroom 128 MiB;
`exec/template/transient` 16/16/32 MiB, "not measured exec bytes"), so two env overrides were tried on the 38/32-line server:
`TF_FLASHNEXT_EXL3_GRAPH_RESERVATION_MIB=128` with `..._HEADROOM_MIB=32` fails startup with `InvalidGraphPolicy`, and
`..._RESERVATION_MIB=400` with `..._HEADROOM_MIB=64` (32 lines) starts but the window is declined with
`incremental graph reservation unavailable (required=0 supplied=419430400)`: an explicit reservation override disables the
window's own graph reservation. Closing the gap would need code that applies `withWindowReservation` to manual values and a measured
(not heuristic) graph footprint; neither was attempted. The window therefore still fits only after trimming 12 lines per layer at a
32k context (17 at 64k). Making it the default is a decode-versus-prefill trade that depends on output length: at 26 lines decode is
about 6% slower (14.9 against 15.87 tok/s aggregate) while a 16k prefill is 1.9 s shorter, so the window pays back only when generated
tokens are fewer than about 300 per 16k-token prompt. It is left opt-in.

### Decode profile under nsys and a copy-kernel sweep (no change found)

nsys 2026.3 on each rank in turn (`--trace=cuda --cuda-graph-trace=node`), TP=2, 38 lines per layer, graphs on, shared BF16 paging,
window off, two concurrent 2,048-token prompts, 256 generated tokens, three batches. Rank 0 and rank 1 kernel totals:

| | rank 0 | rank 1 |
|---|---|---|
| total kernel time | 31.79 s | 31.12 s |
| NCCL all-reduce | 5.68 s | 5.05 s |
| everything else | 26.11 s | 26.07 s |

Rank 0's biggest kernels: all-reduce 17.9%, `fn_exl3_dcopy` (expert miss copy, on a side stream) 12.6%, `tf_exl3_linear_10_2_8` 9.9%,
`tf_exl3_linear_10_2_4` 7.8%, `tf_exl3_grouped_2` 7.6%, `tf_qmmf` kernels about 10.5% together. The GPU is busy about 82% of the
decode window. The all-reduce median is 28 µs but its mean is 142 µs, so most of its time is waiting for the other rank. The
non-all-reduce work is equal on both ranks and both wait about the same, so the skew is symmetric (per-layer variation in misses on
whichever rank is unlucky), not one structurally heavier rank. A per-instance arrival comparison was attempted and discarded: the two
ranks' all-reduce counts differ (37,446 against 36,192, prefill all-reduces included), so the pairing is invalid.

Sweep of `TF_FLASHNEXT_EXL3_DCOPY_BLOCKS` (default 144 blocks of 256 threads on a 36-SM card) at 12, 24, 48 and 144, ABBA-style
repeats, same workload: summed per-request decode 58.3/58.9 tok/s for every value, identical to the digit, with identical hashes. The
copy kernel's block count does not matter; the copies are link-bound. No default change.

Consequence: about 16% of kernel time is all-reduce wait caused by miss variance, and eliminating that variance would be worth at most a
few percent per step. The decode step has no single large inefficiency to remove with a tuning knob.

### Hit/miss overlap on by default (+4% decode without drafts)

`TF_FLASHNEXT_EXL3_OVERLAP` (the held experts compute while the misses' copies run on a side stream, then the misses' experts
follow) was compiled but default-off and unmeasured. It is active in graph mode (same graph coverage). With the stock 144-block copy
kernel it gave no gain (summed per-request decode 58.1/58.6 against 58.3-58.5/59.0-59.1 tok/s at 2,048-token prompts), because the
copy kernel's blocks fill the 36 SMs and the held experts' kernels cannot run beside them. With few copy blocks the gain appears and
plateaus (2,048-token prompts, 256 generated tokens, 38 lines, summed per-request decode, two measured batches each):

| overlap | copy blocks | decode tok/s |
|---|---|---|
| off | 144 | 58.3-58.5, 59.0 |
| on | 144 | 58.0-58.1, 58.6 |
| on | 32 | 59.2, 59.9 |
| on | 16 | 60.4, 61.0 |
| on | 2, 4, 6, 8 | 60.5-60.6, 61.0-61.1 |

The copy is link-bound (a block-count sweep without overlap was identical to the digit), so 2 blocks already saturate it. Default
change: the split is on unless `TF_FLASHNEXT_EXL3_OVERLAP=0`, and with it on the copy kernel uses a quarter of the SMs, at least 8
blocks (9 on the 36-SM RTX 5060 Ti); `TF_FLASHNEXT_EXL3_DCOPY_BLOCKS` still overrides.

Validation on the exact new defaults, ABBA against `OVERLAP=0` with 144 blocks, 16,384-token prompts, 128 generated tokens, 38 lines,
window off, TP=2, graphs, shared BF16 paging; summed per-request decode tok/s (aggregate in parentheses), all hashes `44f75f…`:

| drafts | new default (two legs, two batches each) | old behavior (two legs, two batches each) | change |
|---|---|---|---|
| off | 31.99, 32.34, 32.01, 32.16 (16.07-16.12) | 30.79, 30.90, 30.68, 30.93 (15.79-15.83) | +4.4% (+1.6% aggregate) |
| on | 19.87, 19.88, 19.79, 19.83 (12.37-12.40) | 20.12, 20.04, 20.00, 20.04 (12.44-12.48) | -1.0% |

The MTP chain does not use the split and loses about 1% to the smaller copy-block count. The snapshot gate
(`flashnext-exl3-graph … snapshot-shared`) passes on both ranks with the new defaults, `zig fmt --check` and
`zig build test-flashnext-exl3-host` pass.

Configuration note: on this setup drafted decode (about 20 tok/s summed, 71% draft acceptance, 62 rounds for 128 tokens) is far slower
than undrafted decode (about 32 tok/s), so serving with `--no-drafts` is the faster setting for these two cards today.

### Drafts versus no drafts on real text (new defaults)

Same server (38 lines, overlap default, window off, TP=2, graphs, shared BF16 paging), two different text prompts (16,008 and 12,758
tokens, built as in the real-text window test), 192 generated tokens, one discarded warm-up batch and two measured batches per
mode. Summed per-request decode: drafts off 25.45 and 26.80 tok/s, drafts on 20.06 and 21.13 tok/s (about 21% lower with drafts).
Draft acceptance was 77% (161/208) on the first prompt and 50% (130/260) on the second. Token hashes were identical between the
two modes for both requests, so drafting is exact but slower here: each drafted round verifies more rows, needs more distinct
experts, and the extra expert misses cost more than the accepted extra tokens save. `--no-drafts` is the faster setting on this
setup.

### Requests decode plain by default when experts are offloaded

Drafted rounds lose to plain decode on offloaded experts (previous section). The lane engine now ignores a request's draft flag when
the engine has offloaded expert layers (`plainByDefault()` in `cuda_exl3_pass.zig`, used where `cuda_lanes.zig` decides `drafting`);
`TF_FLASHNEXT_EXL3_DRAFTS=1` restores drafting. Engines without offloaded experts are unchanged. The MTP head and its draft stream
memory are still loaded; only the per-request drafting is skipped (a later change could also skip loading them).

Validation on the new binary, 16,384-token prompts, 128 generated tokens, 38 lines, overlap default, window off, TP=2, graphs, shared
BF16 paging, two measured batches per row, summed per-request decode tok/s, all hashes `44f75f…`:

| server setting | request draft=false | request draft=true |
|---|---|---|
| new default | 32.01, 32.14 | 32.30, 32.32 (127 rounds, no drafts accepted) |
| `TF_FLASHNEXT_EXL3_DRAFTS=1` | 32.02, 32.22 | 19.83, 19.81 (62 rounds, 65/92 accepted) |

A request that asks for drafts therefore decodes at the plain rate by default (about 63% faster than before on this prompt and about
21% faster on real text), with identical output. The snapshot gate passes on both ranks on a freshly built runner, and
`zig fmt --check`, `zig build test-flashnext-exl3-host` and `zig build compile-cuda-tests` pass.

### Second, independent ABBA of the default-path changes (real text, drafts requested)

Different content from the repeated-ID runs: two different text prompts (16,008 and 12,758 tokens), 192 generated tokens, 38 lines,
TP=2, graphs, shared BF16 paging, window off, one discarded warm-up batch and two measured batches per leg and request mode. New =
the branch defaults (overlap on with few copy blocks, plain decode on offloaded experts); old = `TF_FLASHNEXT_EXL3_OVERLAP=0`,
`TF_FLASHNEXT_EXL3_DCOPY_BLOCKS=144`, `TF_FLASHNEXT_EXL3_DRAFTS=1`. Order new, old, old, new. Token hashes were identical for both
requests in every leg and mode (`b143a9…`, `2b3c44…`).

| requests ask for | new: summed decode tok/s, batch wall s | old: summed decode tok/s, batch wall s |
|---|---|---|
| drafts (a normal client) | 25.80, 26.96, 26.95, 27.02; 32.3, 32.2, 32.2, 32.2 | 20.50, 20.78, 20.52, 19.75; 39.2, 39.0, 39.2, 39.1 |
| no drafts | 26.65, 26.81, 25.47, 26.83; 32.4, 32.4, 32.4, 32.3 | 25.90, 26.07, 25.92, 26.07; 32.8, 32.8, 32.8, 32.8 |

For a client that asks for drafts the batch finishes in 32.2 s instead of 39.1 s (17.6% shorter, 21% more throughput) and summed
decode is about 30% higher. For a client that already disabled drafts the overlap alone gives about 2-3% on summed decode and 1.3%
on batch wall time. Together with the repeated-ID ABBA (+4.4% undrafted, +61% for drafted requests) this is two independent
workloads in both orders.

### Comparison with Strata on the same two RTX 5060 Ti (single request)

Strata 0.2.4.2 (`7dd5cf1`, engine 0.1.48 for CUDA 13) was installed on cs13 with `./setup.sh --model IQ3_XXS --context 32768 --gpus all
--yes`: the Qwen3.8-Flash-Next IQ3_XXS GGUF (47.0 GB, about 3.06 bits per weight), automatic layer split across both cards, its MTP draft
layer with `--spec 4 --spec-min-p 0.5`, int8 KV, expert cache `auto` (about 15.7 GiB of VRAM used on each card), one request at a time
(its default). Neither Strata nor project-maya has an EXL3 path for this model, so the weights differ in format (a GGUF IQ3_XXS against our
EXL3 3.05 bpw) and quality was not compared. Ours was the branch defaults (overlap on, plain decode, window off, 38 lines per layer, TP=2,
graphs, shared BF16 paging, 32,768 context). Both were driven by `tools/exl3_single_request_bench.py` (chat endpoint, thinking off,
temperature 0, `ignore_eos`, 192 requested tokens, a unique prefix per run so no prefix cache applies), three requests per prompt. The
prompts are a 70-token story request and the two real-text prompts used earlier (Zig source and markdown docs). Strata stopped some code
answers early despite `ignore_eos` (41-96 tokens), so its decode figures for those rows cover fewer tokens.

| prompt | metric | ours (EXL3, TP=2) | Strata (IQ3_XXS) |
|---|---|---|---|
| 70 tokens | time to first token | 0.73-0.76 s | 0.57-0.78 s |
| 70 tokens | decode | 33.7-36.5 tok/s | 75.1-87.1 tok/s |
| about 16,050 tokens (Zig) | time to first token | 10.38-10.50 s | 7.47-7.62 s |
| about 16,050 tokens (Zig) | decode | 26.6-29.1 tok/s | 83.1-103.6 tok/s |
| about 12,800 tokens (docs) | time to first token | 9.45-9.65 s | 6.74-6.85 s |
| about 12,800 tokens (docs) | decode | 28.7-29.3 tok/s | 101.1-102.1 tok/s |

On this hardware Strata decodes 2.3 to 3.5 times faster and prefills long prompts about 1.4 times faster (about 2,100 against 1,550 tokens/s
at 16k); short-prompt first-token time is the same. Our two-chunk window (26 lines) closes part of the prefill gap (about 12.5 s per 16k
prompt in the two-request runs) but not the decode gap. From the design documents and the profile, the likely causes of the decode gap are
that Strata computes cold experts on the CPU instead of copying them over PCIe (our exposed miss copies are about 30% of a step), splits layers
across the cards (pipeline) instead of expert-parallel with an all-reduce every layer (about 16% of our kernel time is all-reduce wait), and
makes its MTP drafting pay off (ours loses to plain decode). These causes were not isolated by experiment.

### Why Strata decodes faster: the expert cache hit rate (measured in its log; causes inferred)

Strata prints per request: `expert cache 94.1% hit (+1.3% of the routed experts over PCIe)`, 95.2%, 97.5%, 97.2%, 97.9% and 98.9% (+0.1%) on the
same prompts used above. Ours at 38 lines per layer measured a 20.8% device miss rate (a 79% hit rate) in the decode telemetry, and 27.9% at 26
lines (72% hit). So Strata moves roughly 5 to 20 times fewer experts over PCIe per token, and the exposed miss copies that take about 30% of our
decode step (previous sections) are close to absent for it. This alone accounts for a large part of the decode gap, before any CPU lane or MTP effect.

What is visible in Strata's setup log and files explains how it gets there:
- It fills each GPU's expert cache with 7,507 experts in 10.95 GiB of VRAM (about 1.5 MB per IQ3_XXS expert), roughly 15,000 experts across the two
  cards, about 61% of the model's 24,576. Ours holds 38 lines x 49 layers per rank, 3.23 GiB per rank, about 3,700 experts across both ranks (15%).
- It splits layers across the two cards, so each card holds the dense weights of only its own layers. Ours is expert-parallel: each rank's ledger
  shows `persistent=7.53 GB` (dense weights, replicated or mostly replicated, and other fixed state), `scratch`, `deep_prefill`, `pool_scratch`
  and `graph_reservation`, leaving 3.47 GB for expert pools out of a 16 GB card. Why persistent is 7.5 GB was not broken down in this work.
- It ships a pretrained hot-expert profile (`data/expert-profile.bin`, passed as `--expert-profile`) that seeds the cache. Our profile-sized pools
  failed to generalize on held-out prompts earlier (-2.4% throughput), but those were sizing experiments, not seeding the same cache.
- Its CPU lane runs cold experts from host RAM. The vendored kernel `src/dsv41/vendor/exl3_cpu.cpp` (MIT, from project-maya) implements the EXL3 mul1
  codebook for K bits 1-8 with AVX-512 or AVX2+FMA. cs13's EPYC 7452 has AVX2+FMA only (no AVX-512), so the AVX2 tier would run there. With 94-99%
  hits the CPU lane serves only the remaining 1-6%.

Consequences for the plan (hypotheses to test, not results): raising our expert cache from 3.2 GiB to near 10 GiB per rank is worth more than a CPU
lane or MTP tuning. At 26 to 38 lines, every 12 added lines measured about +10% decode (summed per-request 26.95 to 29.7 tok/s), and the hit rate
rises faster than linearly as the cache approaches the hot set. The first steps are to find what the 7.5 GB `persistent` holds and what can move to
host memory or be sharded across the ranks, then to seed the pools from a hot-expert profile at startup.

### Where the 16 GB goes, and how many expert lines fit

Per-rank memory ledger at 38 lines (default features, `--parallel 2`, 32,768 context, shared BF16 KV paging), from the two ranks' startup logs:

| item | rank 0 | rank 1 |
|---|---|---|
| weights resident on the device | about 5.0 GB | about 5.0 GB |
| shared BF16 KV window (`common allocation`) | 2.02 GB | 2.02 GB |
| MTP head (loaded unless `--no-drafts`) | 0.50 GB | 0 |
| prompt scratch | 0.76 GB | 0.76 GB |
| deep prefill rows (4096) | 0.73 GB | 0.73 GB |
| pool scratch | 0.52 GB | 0.31 GB |
| two streams | 1.09 GB | 0.81 GB |
| graph reservation | 0.39 GB | 0.39 GB |
| startup reserve floor | 1.07 GB (`TF_FLASHNEXT_EXL3_RESERVE_MB`, default 1024) | same |
| expert pools (38 lines) | 3.47 GB | 3.54 GB |

The ledger line `free_after_pools` (2.8 GB and 3.4 GB) is not spare room: admission also holds back the reserve and the future bytes (streams, graphs,
planner), so 38 lines is already near the limit; 44 lines at the default settings and 56 lines were refused with `InsufficientStartupMemory`.
Real headroom comes from not loading the head (`--no-drafts` frees 0.5 GB on rank 0) and from a smaller reserve floor.

Sweep with `--no-drafts` and `TF_FLASHNEXT_EXL3_RESERVE_MB=512`, branch defaults otherwise (overlap on, window off, TP=2, graphs), single request,
`tools/exl3_single_request_bench.py`, 192 forced tokens, three requests per prompt (decode tok/s):

| lines per layer | 70-token prompt | 16k Zig prompt | 12.8k docs prompt | result |
|---|---|---|---|---|
| 38 (default, drafts loaded) | 33.7-36.5 | 26.6-29.1 | 28.7-29.3 | baseline |
| 44 | 34.5-36.4 | 28.6-30.1 | 30.0-33.9 | fits |
| 48 | 34.1-36.7 | 29.3-34.7 | 29.9-31.4 | fits |
| 52 | 35.8-38.2 | 30.9-31.5 | 30.6-32.7 | fits |
| 56 | | | | starts loading, never serves |

Going from 38 to 52 lines (+37% cache) is worth about +6% to +12% decode, and time to first token is unchanged (10.4 s at 16k). That is a
configuration change that needs no code, but it is far short of the gap: Strata's cache holds about four times as many experts and reaches a
94-99% hit rate. To get there the resident state per rank has to shrink by several GB (weights, the 2 GB KV window, prefill scratch), which means a layer
split or other sharding of the dense weights rather than another tuning knob.

### What the pack holds, and a sized plan for a larger expert cache

Tensor sizes by role in `Qwen3.8-Flash-Next-exl3-3.05bpw` (84.95 GB total, from the safetensors headers, `tools/exl3_pack_sizes.py`):

| role | size | where it lives today |
|---|---|---|
| routed experts | 45.77 GB | pinned host banks (about 22 GB per rank), expert pools on the device |
| per-layer n-gram embedding shards | 32.64 GB | host memory |
| DeltaNet attention | 1.32 GB | device, both ranks |
| hyper-connection mixes | 1.28 GB | device, both ranks |
| embed_tokens | 1.27 GB | device |
| MTP head | 1.04 GB (0.5 GB on the device, rank 0) | device unless `--no-drafts` |
| lm_head | 0.40 GB | device (split by vocabulary) |
| full attention | 0.38 GB | device (heads split) |
| shared expert, router, projections, other | about 0.6 GB | device |

The on-device dense set is therefore about 5.8 GB, and the measured 5.0 GB resident on each rank says most of it is replicated, not sharded.
A layer split would roughly halve it (-2.9 GB per rank). Against Strata's 10.95 GiB of expert cache per card, our 3.5 GB of pools is paying for:
12.5 GB of other state per rank at 38 lines (weights 5.0, KV window 2.0, prompt scratch 0.76, deep prefill 0.73, pool scratch 0.3-0.5, streams 0.8-1.1,
graph reservation 0.39, reserve floor 1.07, head 0.5 on rank 0).

Sized estimate of what could be recovered per rank (arithmetic from the ledger above, not tested): layer-split dense weights 2.9 GB, a smaller KV
device window about 1.0 GB, the reserve floor 0.5 GB, prefill rows and scratch about 0.4 GB, no MTP head 0.5 GB on rank 0. Together about 5 GB, which would take
the expert pools from 3.5 GB to roughly 8.5 GB (about 2.4 times the cache; at 69-84 MB per line that is on the order of 100 lines). From the 38 to 52
line sweep, +37% cache gave +6% to +12% decode, so even a 2.4-fold cache is more likely to land decode in the 40s of tokens per second than at
Strata's 75-104, which also benefits from its CPU lane and from MTP drafting that pays off at a 95% hit rate. The expert-cache enlargement is the first
and largest step; the CPU lane (the vendored `exl3_cpu.cpp` runs on cs13's AVX2) and revisiting MTP follow it.

### 68 lines per layer: the shared KV window was sized for 8 lanes

With shared BF16 KV paging on, `TF_FLASHNEXT_KVPAGE_SHARED_LANES` defaults to 8 lanes (`cuda_exl3_kvpage.zig`), while the server runs
`--parallel 2`. Setting it to 2 shrinks the `shared BF16 KV common allocation` from 2,023,234,372 to 809,961,172 bytes per rank (1.21 GB), and the pools can
take it. Sweep with `--no-drafts`, `TF_FLASHNEXT_EXL3_RESERVE_MB=512` and `TF_FLASHNEXT_KVPAGE_SHARED_LANES=2`, branch defaults otherwise, single
request, `tools/exl3_single_request_bench.py`, 192 forced tokens, three requests per prompt (decode tok/s; the 70-token row excludes the first,
cold request):

| lines per layer | 70-token prompt | 16k Zig prompt | 12.8k docs prompt | result |
|---|---|---|---|---|
| 38 (defaults, drafts loaded, 8 lanes) | 34.2-36.5 | 26.6-29.1 | 28.7-29.3 | baseline |
| 52 (no drafts, 512 MB reserve) | 36.2-38.2 | 30.9-31.5 | 30.6-32.7 | fits |
| 60 (plus 2 lanes) | 38.6-39.6 | 32.6-34.4 (first 27.9) | 31.9-34.1 | fits |
| 68 (plus 2 lanes) | 40.4-40.9 | 35.1-35.4 (first 32.2) | 32.9-34.0 | fits |
| 76 (plus 2 lanes) | | | | refused |

`tools/exl3_serve_tp2.sh` now defaults to this configuration (68 lines, no drafts, 512 MB reserve, KV lanes equal to `PARALLEL`) and was run end to end on cs13:
38.8-40.7 tok/s on the 70-token prompt, 33.3-34.7 on the 16k prompt, 32.9-35.4 on the 12.8k prompt, time to first token unchanged (0.57 s,
10.4 s, 9.5 s). Against the defaults that is about +13% to +20% decode on short prompts and +16% to +26% on long ones, for no code change. Limits: the
KV window sized for two lanes means a third concurrent request would not fit, 72 lines was not tried, the 512 MB reserve was tested only in these
runs, and the gap to Strata (75-104 tok/s) is still 2.1 to 3 times.

### One user: 80 lines, and where the curve flattens

For a single-user server (`--parallel 1`, `TF_FLASHNEXT_KVPAGE_SHARED_LANES=1`, `TF_FLASHNEXT_EXL3_RESERVE_MB=256`) the shared KV allocation is 0.61 GB per rank. At 76
lines the stream check then refused to start ("the CUDA memory budget fits no stream"), because it applies a separate 1.0 GiB device reserve. That reserve is
`TENSORFOLD_MEMORY_RESERVE_GIB` (default the larger of 1 GiB and 1/32 of the card; the code allows 0.5 GiB at the least). With it set to 0.5, 72, 76 and 80 lines serve.
Single request, `tools/exl3_single_request_bench.py`, 192 forced tokens, three requests per prompt, decode tok/s (the 70-token row excludes the cold first request):

| setup | lines | 70-token | 16k Zig | 12.8k docs |
|---|---|---|---|---|
| defaults (drafts loaded, 8 KV lanes, `--parallel 2`) | 38 | 34.2-36.5 | 26.6-29.1 | 28.7-29.3 |
| no drafts, 2 KV lanes, reserve 512 MB, `--parallel 2` | 68 | 40.4-40.9 | 35.1-35.4 | 32.9-34.0 |
| plus `--parallel 1`, 1 KV lane, reserves 256 MB and 0.5 GiB | 72 | 40.7-40.7 | 34.6-35.7 | 33.3-35.1 |
| same | 76 | 42.1-42.5 | 34.3-35.6 | 34.9-36.2 |
| same | 80 | 41.7-42.8 | 34.7-36.1 | 34.8-36.8 |

Beyond about 68 lines each further four lines add only 1% to 3%, and the long-prompt rows barely move (35 tok/s, about 28.5 ms per token). That matches the
nsys profile: the roughly 23 ms of each step that is not expert-miss copying (all-reduce waiting, the dense and expert kernels) is now most of the step. Strata's 75-104 tok/s means 10 to 13 ms per
token, so its per-step cost is also less than half of our non-miss time even before drafting. The cache enlargement is therefore done (about +20% to +30% over the defaults); the next gains have to come from the
step itself (the per-layer all-reduce of expert-parallel TP, kernel efficiency) and from making drafting pay, not from more lines.
`tools/exl3_serve_tp2.sh` uses 80 lines when `PARALLEL=1` and 68 otherwise. The 80-line single-user setup was run through the sweep script, not through the launch script, and the 0.5 GiB reserve is the
supported minimum with no headroom for other GPU users.

### The decode step at 80 lines, from nsys (rank 0)

nsys trace of rank 0 with the single-user setup (80 lines, `--parallel 1`, no drafts, graphs; a short story prompt, 256 generated tokens, three runs at
40.3, 44.1 and 44.7 tok/s under the profiler), counted per decode step (765 steps, `tools/exl3_nsys_step_breakdown.py`). Wall time was 24.2 ms per step under the profiler
and the non-copy kernels kept the GPU 81% busy.

| category | ms per step | launches per step |
|---|---|---|
| EXL3 matmuls (`tf_exl3_linear_*`, `tf_exl3_grouped_*`, gate-up and down epilogues) | 10.85 | 819 |
| `qmmf` matmuls | 4.56 | 234 |
| NCCL all-reduce and gather | 3.07 | 49 |
| expert miss copy (side stream, overlapped) | 2.07 | 47 |
| hyper-connection kernels | 0.95 | 390 |
| router | 0.65 | 96 |
| DeltaNet kernels | 0.42 | 108 |
| planner, attention and KV paging, other | 0.85 | about 400 |

Largest single kernels: `tf_exl3_linear_10_2_8` 3.87 ms (49 launches, 79 us each, 80-96 blocks), `tf_exl3_linear_10_2_4` 3.41 ms (108 launches, 31 us), NCCL all-reduce 2.94 ms
(48 launches, 61 us each, mostly waiting for the other rank), `qmmf` 2.68 and 1.87 ms, `tf_exl3_grouped_2` 2.57 ms (192 launches, 13 us).

What this says: the expert misses are no longer the cost (2.07 ms of copy, hidden on a side stream); the step is about 15.4 ms of matmul kernels, 3 ms of all-reduce wait and the rest small kernels.
The dense matmuls run identically on both ranks. Pack inspection (`tools/exl3_pack_sizes.py` and the trellis shapes): routed experts are 3-bit, the DeltaNet projections (1.3 GB) and `lm_head` are 5-bit,
and the on-device dense set is about 5.8 GB, so a rank streams a few GB of dense weights per token (roughly 3.6 GB per step including experts and the head, which is about 8 ms at the card's 448 GB/s).
That mapping from kernel names to weights is an inference from sizes and launch counts, not confirmed by a per-kernel trace of the weights each reads. Implications that follow (hypotheses): sharding the dense
projections across the ranks would cut that per-rank streaming roughly in half but adds collectives per layer, and each NCCL call already costs 61 us here, so it only pays if the
gathers are folded into the existing per-layer all-reduce; a layer split removes the all-reduce (about 3 ms) and the replication but leaves one card idle per token for a single stream.
Neither was prototyped.

### Which weights the decode kernels read, and how close they run to the memory limit

Kernel names decode as `tf_exl3_linear_{2 x bits}_{codebook}_{warps}` (`exl3_linear.cu`: `K2` is twice the width in bits, `CB` 2 is the mul1 codebook, `WK` the warps per block), so
`tf_exl3_linear_10_2_8` is 5-bit mul1 with 8 warps and `tf_exl3_linear_10_2_4` is 5-bit with 4 warps. The pack's 5-bit tensors (trellis shapes, `tools/exl3_pack_sizes.py`
and the bit listing) are the DeltaNet projections (`in_proj_qkv` 0.59 GB, `in_proj_z` 0.35 GB, `out_proj` 0.35 GB over 36 layers), full attention (`q_proj` 0.24 GB, `o_proj` 0.12 GB,
`k_proj` and `v_proj` 0.01 GB each over 12 layers), the shared expert (0.15 GB over 48 layers; rank 1 only) and `lm_head` (0.40 GB, vocabulary split). Routed experts are 3-bit.
Matching launches per step and grid sizes (grid x is N / 128) from the 80-line nsys trace of rank 0:

| kernel | launches per step | grid | likely weights (inferred) | bytes read per step | ms per step | achieved bandwidth |
|---|---|---|---|---|---|---|
| `tf_exl3_linear_10_2_8` | 49 | 80 or 96 blocks | DeltaNet `in_proj_qkv` (36, N 10240) + full-attention `q_proj` (12, N 12288) + one more | about 0.83 GB | 3.87 | about 215 GB/s |
| `tf_exl3_linear_10_2_4` | 108 | 16 to 160 blocks | DeltaNet `in_proj_z` and `out_proj` (72) + full-attention `o_proj`, `k_proj`, `v_proj` (36) | about 0.85 GB | 3.41 | about 250 GB/s |
| `qmmf_kernel` (two variants) | 234 | small grids | 4 hyper-connection mix matrices per layer (192) plus others; 1.28 GB in the pack | about 1.3 GB | 4.56 | about 280 GB/s |
| `tf_exl3_grouped_2` | 192 | 11 x 5 x 8 etc. | routed experts (3-bit), the rank's share | small | 2.57 | not computed |

The card's memory bandwidth is 448 GB/s, so these dense kernels run at roughly 50% to 62% of the limit, and about 11.8 ms of the 24 ms step is spent streaming dense weights that both ranks read in full. The kernel-to-weight match
rests on launch counts, grid sizes and the pack's tensor sizes; it was not confirmed by tracing which tensor each launch reads, and the bandwidths assume each weight is read once.

Two ways to cut that 11.8 ms while keeping tensor parallelism (neither prototyped): (1) make the kernels faster, by moving the 5-bit EXL3 dense matmuls and the `qmmf` matrices from about 55% to about 85%
of the memory limit, worth roughly 4 ms of the step, with no structural change; (2) shard the dense projections across the two ranks (column-parallel input projections, row-parallel
`out_proj`, and the hyper-connection mixes folded into the existing per-layer reduction), which would halve each rank's streaming (up to about 5 to 6 ms) but adds a second
collective per layer, which costs 25 to 60 us per call here, so it pays only if the extra reductions are fused into the one we already do.

### Why the dense kernels run at about half the memory limit, and what stops a quick fix

The launch shape of every EXL3 linear comes from `plan()` in `zig/src/cuda/exl3.zig`, called as `plan(k, n, 192, 8)`: when the output has at least 64 blocks of 128 columns (N >= 8192) it uses 8 warps and no split of K, so the grid is just N / 128 blocks
(80 or 96 for the DeltaNet and full-attention input projections), about 2.2 blocks of 256 threads per SM on a 36-SM card; only narrower outputs (under 64 blocks) are split along K until they reach 192 blocks. That under-occupies the memory system
for the widest matrices, which fits the achieved 215 GB/s of the 448 GB/s limit. The obvious change is to split K for those shapes too (and so use about 192 blocks), but the function's comment says the plan is chosen from the shape alone "so a row's
bits never depend on the row count (linear.py plan)": the split changes the order of the fp32 accumulation, so the outputs would no longer be bit-identical to the current kernels or to the reference, and the branch's earlier verification
(layers 0, 3 and 17 bit-identical to TP=1, graphs bit-equal to eager) depends on that. Any speedup of the dense kernels therefore has to either keep the accumulation order (deeper prefetch or more loads in flight per
thread inside the existing grid) or be accepted as near-exact and checked by token hashes instead of bit equality. The same applies to sharding the dense projections across the ranks, which splits sums between cards. Neither was built.

### Result: splitting K for the wide 5-bit matmuls does not speed decode (rejected)

To test the under-occupancy explanation in the previous section, `plan()` was given an opt-in (`TF_EXL3_WIDE_SPLIT=1`) that splits K for outputs of 64 to 191 column blocks, which moves the 80- and 96-block
projections to 160-320 blocks. It built and passed the host tests. A/B on cs13 at 72 lines per layer, single-user (`PARALLEL=1`, no drafts), the launch script, three requests per prompt
(decode tok/s; legs in the order off, on, on; the second "off" leg did not finish and is not used):

| prompt | split off | split on (two legs) |
|---|---|---|
| 70-token | 39.2, 40.7, 40.6 | 38.8, 40.5, 41.6 and 38.8, 41.8, 41.8 |
| 16k Zig | 33.9, 34.4, 35.3 | 33.9, 33.7, 36.4 and 34.8, 33.3, 36.2 |
| 12.8k docs | 34.4, 34.4, 35.2 | 33.2, 35.6, 35.2 and 33.4, 34.2, 34.6 |

No gain outside run variation (about 3% on these rows), so more blocks in flight for the widest matrices is not what limits them; the extra partial-sum scratch also made
80 lines no longer fit (rank 1 was 0.02 GiB short of its stream budget). The change was removed. The output-hash comparison planned for this test was not usable (the bench prefixes each request with a unique id).
The 215-280 GB/s the dense kernels reach is therefore not explained by launch shape; the trellis decode work per byte, instruction issue, or the replicated small kernels around them (390 hyper-connection
launches and 819 matmul launches per step) are the remaining suspects, none profiled at instruction level (ncu was not run).

### The cards are not throttled; the dense kernels are not bandwidth-bound

`nvidia-smi` sampled every 0.5 s on both 5060 Ti while the single-user setup (72 lines, `PARALLEL=1`, no drafts) decoded 512-token replies (567 busy samples at 50% utilization or more, per card):

| | GPU 0 | GPU 1 |
|---|---|---|
| SM clock | 2,898 MHz average (max 3,210) | 2,879 MHz (max 3,210) |
| memory clock | 13,801 MHz average (max 14,001) | 13,801 MHz |
| power | 102 W average (limit 180) | 103 W |
| GPU utilization / memory utilization | 95% / 45% | 95% / 47% |
| performance state | P1 | P1 |

There is no power cap, no memory down-clock (no compute-state P2 cap) and no thermal limit (59-63 C). The GPU is busy 95% of the time but its memory system only 45%, which agrees with the nsys estimate that the dense 5-bit
and `qmmf` matmuls reach about 215-280 GB/s of the 448 GB/s available: their time goes to latency or instruction issue inside the kernels, not to waiting on DRAM. Decode in this run was 40.6-43.0 tok/s on the short prompt and
29.9-38.3 tok/s on the 16k prompt (512 tokens, three requests each; the long-prompt spread is wide). Finding out which of latency, instruction issue or occupancy is responsible needs the performance counters (`ncu`), which this account cannot read.


## 2026-10-25: upstream merge validation, place-free TP combine, MTP drafting on code

Branch `exl3-on-upstream-merge` (merge of upstream at 98796c566, then fixes). Two merge regressions found and fixed on the GPU:

- The prefill graph binding only knew the one-row indexer `scores` kernel; upstream now launches the 8-row, 256-block `scores_tile`
  when there are four index heads, so every paged attention graph was refused at capture and the snapshot gate failed
  ("paged-attention graph replay after snapshot restore"). `Request.tile` now selects the tile kernel, grid and block divisor.
  The gate passes on both ranks (replays +112, paged +28, rebinds +4, as before the merge).
- `cuda_mtp.zig` read the dense `lm_head.weight` tensor to type the draft artifact, which an EXL3 pack does not have
  (MissingTensor at start with drafts on). The head's format now follows its role when the tensor is absent.
  Also fixed: a dangling vtable address returned by `backend()` (the ranked table is now comptime-static).

A/B, merged build against the build before the upstream merge (same server settings, 72 lines, plain decode, 3 prompts, A/B/B/A):
identical output hashes on all three prompts, decode +0.7% to +1.3%, TTFT about 2% slower on 12-16k prompts (cause not found).

Place-free tensor-parallel combine (`TF_FLASHNEXT_EXL3_TP_DIRECT`, default 1; 0 restores the old path): the write-back reads the all-reduced
sum in place with fp32 ones as weights, so `fn_exl3_tp_place` (48 launches a step) is gone. A first version changed the text of two prompts
because the head's last-row compaction moved y and the weights but not the sum pointer; fixed, and all three hashes match the baseline.
Decode +0.35% to +0.5% (45.0 against 44.99 tok/s short, 34.2 against 34.1 zig, 35.4 against 35.2 docs).

MTP drafting on the merged build (64 lines, `TF_FLASHNEXT_EXL3_DRAFTS=1`, no `--no-drafts`; outputs identical to plain decode):

| prompt | plain tok/s | drafted tok/s | accepted drafts |
|---|---|---|---|
| story (40 tokens in) | 45.0 | 36.0 | 102 of 190 |
| Zig source (16k) | 34.2 | 96.2 | 180 of 185 |
| markdown docs (12.8k) | 35.2 | 36.4 | 130 of 192 |

Drafting is worth 2.8 times on predictable text (it matches Strata's 83-104 tok/s on the same Zig prompt) and loses 20% on free text. A round costs
about 22 ms plain, 60 ms for a 3-wide window and 180 ms at 16 rows (each row brings its own experts over PCIe), so it only pays above about 2.7
tokens a round. Depth 2, 3 or confidence 0.85 changed story throughput by less than 8% and did not remove the loss. Turning on upstream's
`plain_guard` with a fitted window-cost table made the story prompt slower (32 against 36 tok/s) and was reverted; a per-stream rule that falls
back to plain rounds below the break-even acceptance is the open item. Drafts therefore stay opt-in for now.

Follow-up, same day: a per-stream rule in `cuda_lanes.zig` draftFn (smoothed kept share of the window's drafts below 0.8 gives plain rounds, a probe
every 12) was measured against always drafting on the same three prompts (hashes identical): story 40.6 against 36.1 tok/s (+12%, still below the
45.0 of a server without the head), Zig 92.0 against 96.5 (-5%), docs 34.5 against 36.6 (-6%). A plain round still absorbs its kept rows into the head,
so the loss is only partly recovered, and the rule costs the predictable prompts. Reverted. What would make drafting the default is a plain round
that skips the head entirely and catches the head's cache up when drafting resumes; not built.

Prefill, merged build, 72 lines (two-chunk off) against 64 lines with `TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK=1` (72 lines does not start with it:
its 588 MB window leaves the pools no room), temperature 0, 64 tokens: 16k Zig prompt TTFT 10.64 s against 10.21 s (-4%, same text);
12.8k docs prompt 10.54 s against 8.52 s (-19%) but the text differs (hash 0c279a3ea5d3 against 091349946f19), and decode on that prompt is
about 8% lower at 64 lines (31.2-32.0 against 34.1-35.1). The window is therefore not a default: it changes the bits on a prompt whose last chunk
is short, and it costs decode through the lines it takes.

Resident memory per rank at 72 lines (ledger): dense weights and fixed state 5.62 GB, expert pools 6.40 GB, prompt scratch 0.78 GB, deep prefill
0.73 GB, pool scratch 0.32 GB, graph reservation 0.32 GB, stream 0.24 GB, 1.99 GB free after the pools. Shrinking scratch frees about 1 GB (about 14
lines, 2-3% decode by the 4-line sweep); a larger step needs the dense weights sharded across the ranks. Not done.

Second attempt, same day: plain rounds that skip the head, stashing their kept rows (tokens, and the streams in the unused tail of the draft's
`streams` buffer) and catching the head up in one batched absorb when drafting resumes. Measured against always drafting: story 39.0 tok/s (the
non-stashing rule got 40.6 and the no-head server 45.0), docs 34.0 (against 36.6), and the Zig prompt failed (no decode figure, a different output
hash), so the catch-up path has a bug and, more to the point, the head was not the main cost of a plain round on a drafting server: the verify-round
path itself is slower than the graph decode step. Reverted; drafting stays opt-in. A drafting default would need the plain round to be the graph decode
step itself, not a one-row verify window.

Long-prompt TTFT after the merge (16k Zig prompt, warm 10.40 s old, 10.63-10.66 s merged; 12.8k docs 10.35 against 10.55; cold first request 11.7
against 12.2): the difference is identical with prefill graphs off (10.40/10.65, 10.35/10.56), so it is not the graph path. An nsys comparison of rank 0's
kernel time for one 16k prefill run finds the merged build no slower in aggregate (12,466 ms against 12,574 ms): upstream's kernels replace
`fn_gdn_rows` (467 ms), `fn_attn_chunks` (343 ms), `fn_hc_dense_mix` (175 ms), `fn_qsa_scores` (99 ms) and `fn_hc_normalize` (82 ms) with
`fn_gdn_fused` (509 ms), `fn_attn_chunks_one` (314 ms), `fn_hc_dense_mix8` (108 ms), `fn_hc_normalize_cols` (154 ms), `qsa_tile::scores` (32 ms) and
`tf_route_logits_wide` (57 ms), and the all-reduce kernel (6.38 s old, 6.47 s merged) grows by 90 ms, a rank's wait on the other. With a profile of
both ranks of the merged build the two ranks' kernel totals are within 2% (12.9 s and 12.7 s), so the extra 0.2-0.3 s is not a throughput regression of
either rank's kernels but synchronisation skew (new `fn_gdn_fused` and the tile kernels have different per-layer timing on the two ranks); it was not
chased further.

The all-reduce in prefill is the largest kernel on both ranks (6.2-6.7 s of about 12.8 s for the run), but it runs on the reduce stream overlapped with
the next sub-block's compute (TF_FLASHNEXT_EXL3_PREFILL_OVERLAP), over host shared memory: `nvidia-smi topo -p2p r` reports GNS (chipset not supported)
for the pair, so no peer-to-peer. A bf16 exchange would halve those bytes but changes the rounding (the sum of two bf16-rounded partials instead of the
fp32 sum rounded once) and was not built.

Decode all-reduce, per-node graph trace of both ranks (96 tokens, short prompt, 40.5 tok/s under nsys): rank 0 recorded 9,216 all-reduce launches and
rank 1 4,978, so the two traces cannot be paired by index with confidence, and the processes' clocks are separate (an alignment check by end time
failed, 2.1-2.6 s apart). Pairing from the tail anyway, the shorter of each pair's durations had a median of 22.5 us (p90 48 us) and the longer one 51.6 us (p90
169 us): about 1.5 ms a step of pure transport and about 4.1 ms with the wait for the slower rank. If that holds, a faster transport could save under 1 ms
(3%) and the larger part of the all-reduce cost is waiting for the other rank, which only balancing the two ranks' work would remove. Which rank
arrives later was not established; that needs one trace with both ranks on the same clock (a single nsys session over both processes).

Straggler check with both ranks in one nsys session (one clock; 96 decode tokens, short prompt; 4,616 all-reduces paired by end time): the process
that started first arrives later at 2,714 of them (59%, mean wait 81 us) and the other at 1,902 (41%, mean wait 87 us); once both have arrived the
all-reduce itself takes a median of 21.9 us. So there is no fixed straggler: each layer, whichever rank had the more expensive set of picks and
misses makes the other wait, about 85 us a layer or about 4 ms of a 24 ms step, and transport is 1 ms. A static rebalance of experts or of the
shared expert would move only the 59/41 asymmetry; the rest is the per-token binomial split of ten picks across two id ranges plus each rank's own
misses. It would take dynamic balancing (a pick served by whichever rank has it resident, which needs experts held by both ranks) or removing the
per-layer dependency (a layer split) to recover it. Neither is built.

Balanced decode ownership (`TF_FLASHNEXT_EXL3_BALANCE=1`, off by default; both ranks then pin every routed expert, 42.2 GiB each instead of 21.1, load
time 47 s to 70 s). The tp=2 expert-parallel split gives rank 0 ids 0-255 and rank 1 ids 256-511 plus the shared expert, so each layer's ten picks fall
on the ranks by chance and the rank with the more expensive set makes the other wait (about 85 us a layer, 4 ms of a 24 ms step, measured with both
ranks in one nsys session). Three dealings were tried, same server settings, 72 lines, plain decode, three prompts (balance off: short 45.1, Zig 34.0,
docs 35.1 tok/s):

- by count only (`fn_exl3_tp_balance`: move the lowest-weight picks of the heavier rank): short 40.7, Zig 31.4, docs 32.7. It sends picks to a
  rank whose cache does not hold them, so the lost hits cost more than the wait saved.
- by residency and load (`fn_exl3_tp_deal`; each rank publishes its next layer's line_of flags in a tail of the layer's all-reduce, 2 x 513 floats, so
  both ranks decide on the same inputs with no extra collective; a pick resident on one rank runs there, on both on the lighter rank, on neither on its
  home rank): short 42.5, Zig 34.3, docs 35.9. The same request gives different text on two runs (the grouping of the fp32 partial sums follows the
  cache state), so it is not reproducible.
- the same with a pick resident on both staying on its home rank unless that rank is clearly heavier (current code): short 43.7, Zig 34.7 (+2.1%), docs
  36.0 (+2.5%); identical hashes across two runs of the same server. Docs prefill TTFT is about 1.1 s slower (11.9 s against 10.5 s); the Zig prompt is not.

It is neutral to slightly negative on free text and +2% on long prompts, at double the pinned memory and a slower start, and the text depends on the
cache history. Left in the tree, off by default. The wait is mostly the per-token randomness of which experts a layer needs and where they are
resident; recovering it needs either replicated hot experts on both ranks (cache capacity) or a layer split.

Decode kernel shapes against this card (36 SMs), from the both-ranks decode trace: `tf_exl3_linear_10_2_8` launches 80 blocks (96 for 12 of its calls) of
256 threads at one block an SM (2.22 and 2.67 waves, so 74% and 89% of the last waves filled), `tf_exl3_linear_10_2_4` 160 and 192 blocks at three an SM
(1.48 and 1.78 waves), the 160-block `qmmf` 2.22 waves at two an SM, `tf_exl3_grouped_2` 4.07 and 2.04 waves. Summed idle fraction: about 2.4 ms of
the 22 ms step. The host's split-K plan (`exl3.plan`, `exl3_block_dense.plan`) is linear.py's, aimed at 192 blocks. Making the block target
(`PLAN_BLOCKS` 192, 216, 320, 432) and the wide-N K split (`PLAN_WIDE_SK` 1, 2) environment-settable and sweeping them on the short prompt (plain decode,
72 lines): every block target gave the same output hash as the default and 45.2-45.6 tok/s against 45.15 (the plan never changed: min-tile and divisibility
limits cap the split), and a wide-N split of 2 changed the bits and was slower (44.6). No gain, reverted. Reaching the idle waves would need new tilings in
the kernel itself (a different tile along N or an uneven K split), not a different host plan.

Bytes a token reads (file headers): the Strata GGUF 3.9 GB of dense tensors and 0.9 GB of experts per token, ours 4.0 GB and 0.9 GB. The formats cost the
same bytes; Strata's 10-13 ms step is about 4.8 GB at the card's bandwidth, while each of our GPUs reads about 4.45 GB (the dense set replicated) in 22 ms.
By estimated bytes per kernel group our matmuls reach about 63% (hyper-connection), 74% (DeltaNet, attention, head) and 33% (expert kernels) of peak,
plus about 3-4 ms of per-layer sync wait and 3-4 ms of small kernels and gaps. A layer split would not close that gap (est. 22 to 18-19 ms).

Nsight Compute counters, read at last (cs13 allows `sudo ncu`; the account itself cannot read counters). 22 decode launches of the main kernels, rank 0, 72 lines,
plain decode, ncu's default base clocks, `gpu__dram_throughput` as a share of peak:

| kernel (grid) | time | DRAM throughput | SM throughput | issue slots active | warps active | dominant stall (per issued instruction) |
|---|---|---|---|---|---|---|
| `tf_exl3_linear_10_2_8` (96) | 68 us | 66% | 35% | 40% | 17% | long scoreboard 2.0, wait 0.5 |
| `tf_exl3_linear_10_2_4` (160) | 35 us | 63% | 33% | 40% | 20% | long scoreboard 3.0, wait 0.5 |
| `qmmf` (160) | 19 us | 77% | 12% | 9% | 16% | long scoreboard 15 |
| `qmmf` (11x8) | 25 us | 66% | 13% | 10% | 16% | long scoreboard 9.8, barrier 5.1 |
| `tf_exl3_grouped_2` (11x5x8) | 39 us | 44% | 39% | 44% | 22% | long scoreboard 2.0, wait 1.0 |
| `tf_exl3_grouped_2` (11x20) | 26 us | 33% | 29% | 41% | 22% | long scoreboard 2.1 |
| `tf_exl3_linear_10_2_4` (16 and 20 blocks) | 12 us | 12-15% | 8-9% | 24% | 8% | long scoreboard 1.7 |

The dense kernels sit at 63-66% of DRAM peak, memory-latency bound (the long-scoreboard stall dominates), with 8 or 12 resident warps an SM. Raising
residency did not help: capping the registers of every `tf_exl3_linear` instance (`__launch_bounds__(WK * 32, WK == 8 ? 2 : 4)`, so 128 registers or
fewer) left the output hash identical and decode unchanged (short 45.2, Zig 33.9-34.0, docs 35.1-35.2 tok/s, as before), so occupancy is not the limit.
What is left is each warp's own memory-level parallelism: the main loop prefetches one k step ahead in registers (8 tiles) and then decodes 8 tiles,
so the loads of a step are issued in one burst. A deeper pipeline (two or three steps in flight, through `cp.async` into shared memory because the
registers are spent) would be the next kernel change; at 85% of peak the dense kernels (about 10 ms of the 22 ms step) would save roughly 2-2.5 ms. Not built.
The 12-20 block launches (36 a step) fill a third of the GPU's SMs; the expert kernels reach only 33-44% of DRAM peak but move small amounts per launch.

Two more kernel experiments on the dense `tf_exl3_linear` instances, both reverted:

- Register cap (`__launch_bounds__(WK * 32, WK == 8 ? 2 : 4)`): the instances compiled to 128 registers with a 40-56 byte stack, and the profile of the capped
  build shows more warps resident (30% against 17%) but slower kernels in isolation: `tf_exl3_linear_10_2_8` (96 blocks) 73.7 us against 68.3, the 160-block
  `tf_exl3_linear_10_2_4` 42.3 us against 35.5, DRAM throughput 61% and 53% against 66% and 63%. End to end it changed nothing (hashes identical, decode
  45.2/33.9/35.1 tok/s).
- Interleaved k tiles (the block's warps take adjacent k tiles each step, one contiguous 10 KB stream a block instead of eight 25 KB ranges, so about 100 DRAM
  streams instead of about 770 across the card): decode 45.2/33.9/35.0 tok/s, the same as before; the sums are grouped differently so the short and docs prompts
  produced different text (Zig identical).

Neither occupancy nor the number of concurrent streams limits the dense kernels. They stay at 63-66% of DRAM peak (the hyper-connection `qmmf` reaches 77%), which may be close to what
this access pattern and GDDR7 deliver; a different tile format (not a reshuffle of the same loads) would be needed to find out.

Why the dense kernels sit at 65% of DRAM peak, found with a bandwidth test and a residency argument. A bare-loads test on one 5060 Ti (rotating 40 MB buffers so L2 cannot help):
ideal 16-byte streaming reaches 420 GB/s (94% of 448) at any grid of 96 or more blocks; the EXL3 tile access pattern alone (8 warps a block, each on its own k range, 24
scalar loads a lane a step, a trivial consumer) reads the 19.7 MB of a 96-block projection in 39.5 us (497 GB/s, the overlapping lane words partly served by L2). The real
`tf_exl3_linear_10_2_8` takes 68 us on the same matrix. So the loads are not the limit: it is residency. An eight-warp block of 147 registers fits once an SM, so the 96 blocks
run as three waves of 36; the bandwidth test had all 96 resident.

- Three rotating prefetch buffers two steps ahead for eight-warp blocks (no register copies; 0 bytes of spill): output bit-identical, the kernel 4% faster in isolation (65.3 us, DRAM 68%), decode
  +0.4% (45.3/34.0/35.2 tok/s). Kept: it is exact.
- Four-warp blocks for wide N (`TF_FLASHNEXT_EXL3_PLAN_WIDE_WK=4`, new; 8 stays the library default): three blocks fit an SM, so the 96-block launches are one resident wave. Decode, A/B/B/A
  (8, 4, 4, 8) on the three prompts: short 45.3 to 46.35 tok/s (+2.3%), Zig 33.97 to 34.5 (+1.6%), docs 35.2 to 35.7 (+1.4%), both four-warp legs agreeing. The Zig text is unchanged; the short
  and docs text differ after about 110 characters (a near-tie flips, both continuations read the same in quality) because the block's partial sums group in four, not eight. `tools/exl3_serve_tp2.sh`
  now exports it in the fast mode (`PLAN_WIDE_WK=8` restores linear.py's plan).

Narrow-N block target with the four-warp plan on (A/B/C/B/A, 192, 80, 108, 80, 192): a target of 108 gave the same plan as 192 (same hashes, 46.3/34.5/35.7 tok/s); a target of 80
(K split 2 for the 40-block projections, one resident wave) changed the bits and was slightly slower (45.9/34.4/35.6). Reverted; the narrow plan stays at linear.py's 192.

One decode layer, from the trace of both ranks in one session (a GDN layer, steady state, 455 us, rank 0): the MoE experts take 83 us (dplan, group, a 36 us gate-up
`grouped_2` and a 22 us down, the miss pass 3 us), the all-reduce 129 us (that layer's wait), the attention-side hyper-connection readout 47 us (`hc_writeback_moe`,
`hc_normalize_cols`, `qmmf` 11x8 22 us, `hc_reduce_act`, `qmmf` 160 17 us, `hc_dense_mix8`), the GDN projections 67 us (`linear_10_2_8` on one stream beside `linear_10_2_4` on another, so the
two already overlap), the GDN recurrence 14 us, the output projection 32 us, the mlp-side readout 50 us, the router 15 us. The two readouts are 21% of the layer and 29% of the bytes a token reads (1.28 GB);
their dependent chain cannot overlap the all-reduce wait because it reads the stream state the combine produces.

`qmmf` K-split cap (`TF_QMMF_SPLIT_CAP`, 8 by default; 16 leaves the cluster launch for a separate reduce kernel), A/B/B/A with the four-warp plan on: short 46.35 to 46.23 tok/s,
Zig 34.5 to 34.66, docs 35.68 to 35.96 (the docs text equals the eight-warp plan's again); within noise. Reverted.

## Toggle streamlining (2026-10-26)

The CUDA path read 118 environment names (about 80 of them `TF_FLASHNEXT_*`); 56 appear in a script, a doc or a test. Naming scheme now: user-facing settings are `TF_FLASHNEXT_<AREA>_<SETTING>`
(`DRAFT_*`, `EXPERT_*`); the EXL3 graph, prefill and KV-paging families keep their `TF_FLASHNEXT_EXL3_*` and `TF_FLASHNEXT_KVPAGE_*` prefixes, which were already coherent; names left over from upstream
(`TF_FLASHNEXT_KV_PAGES*`, `BATCH_HEADS`, `GDN_FUSED`, the `FZ_*` and `FN_*` debug set of the Mac engine) are untouched so the upstream PR does not conflict.

Renamed (same meaning): `TF_FLASHNEXT_EXL3_DRAFTS` to `TF_FLASHNEXT_DRAFT_ENABLE` (drafting with offloaded experts; off unless 1), `TF_FLASHNEXT_DRAFTS` to `TF_FLASHNEXT_DRAFT_DEPTH`,
`TF_FLASHNEXT_CONFIDENCE` to `TF_FLASHNEXT_DRAFT_CONFIDENCE`, `TF_FLASHNEXT_EXL3_LINES` to `TF_FLASHNEXT_EXPERT_LINES`, `TF_FLASHNEXT_EXL3_RESERVE_MB` to `TF_FLASHNEXT_EXPERT_RESERVE_MB`,
`TF_FLASHNEXT_EXL3_GRAPH_AFTER` to `TF_FLASHNEXT_EXL3_GRAPH_DECODE_AFTER` (parallel to `..._GRAPH_PREFILL_AFTER`).

Removed, with the value we run built in (none was used by a script, doc or test): `TF_FLASHNEXT_EXL3_FORK`, `_RECYCLE`, `_OVERLAP`, `_PREFILL_HEAD`, `_PREFILL_MIN`, `_PREFILL_PREFETCH`, `_PREFILL_KERNEL`,
`_PREFILL_TILES`, `_PREFILL_RESEED`, `_PREFILL_RING_PASS`, `TF_FLASHNEXT_KVPAGE_ROWS`, `TF_FLASHNEXT_KVPAGE_SLOTS`; and the three of this session's experiments: `TF_FLASHNEXT_EXL3_BALANCE` (code removed),
`TF_FLASHNEXT_EXL3_TP_DIRECT` (the place-free combine is always on) and `TF_FLASHNEXT_EXL3_PLAN_WIDE_WK` (four warps a block is the plan for every N; the deeper-prefetch variant of the eight-warp kernel, now unused, is removed with it).
`tools/exl3_serve_tp2.sh` drops the exports that only repeated a default (`..._GRAPH_AFTER`, `..._GRAPH_PREFILL_AFTER`).

## Verify window cost by row count (drafts on, 80 lines, depth 6)

Mean per-round verify time (ms) and draft time (ms), measured with a temporary round timer (removed):

| rows | short verify / draft | docs verify / draft |
|---|---|---|
| 2 | 39.9 / 5.5 | 52.7 / 7.6 |
| 3 | 49.0 / 6.4 | 61.4 / 9.7 |
| 4 | 56.8 / 7.5 | 54.0 / 9.5 |
| 5 | 62.2 / 8.2 | 75.6 / 8.6 |
| 7 | 58.2 / 5.2 | 94.2 / 9.8 |

A plain decode step is about 22 ms, so the first extra row costs about 18 ms and each further row 6-9 ms. There is no cliff between 2 and 3 rows; the jump is from 1 to 2 rows. The cost is the PCIe copy of experts the extra row misses in the device lines, so cache capacity and prediction (not kernel time) bound speculation. Resulting decode: short 37-40, docs 40.7-41, zig 102 tok/s (plain 48.9 / 38.2 / 36.2).
