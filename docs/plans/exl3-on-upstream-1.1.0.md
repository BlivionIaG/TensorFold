# EXL3 Qwen3.8-Flash-Next on upstream 1.1.0: what to keep, what to drop

Status: decision proposal, 2026-10-19. Branch `exl3-on-1.1.0` (tag `v1.1.0`, 76aafcf2) in the fork.
Our work: `origin/exl3-flashnext-window-wip` (672120a8, 72 commits, 113 files, +25,381 lines over 1.0.12 = 833bbfcb).
Pack: `turboderp/Qwen3.8-Flash-Next-exl3` 3.05bpw_h5_ng5 at `/data/models/Qwen3.8-Flash-Next-exl3-3.05bpw` on par1-cs13.

Every number below says whether it was **measured** (on par1-cs13 on 2026-10-18/19, one idle RTX 5060 Ti
16 GB, sm_120, nvcc 13.4, each run under `~/bin/gpurun`, so no other job was on the card) or **read** (from the code).
Numbers marked *estimate* are arithmetic from measured parts, not a run.

## Decision

**Option C.** Add EXL3 weights to upstream's Flash Next CUDA family (`zig/src/families/flashnext/cuda_*.zig`). We then
inherit its engine, lane backend, server, MTP chains, prompt cache, rank engine and graphs. Our own code stays only
where upstream has nothing: the expert offload pool and policy, the EXL3 n-gram row codec with the pread/hot-row reader,
EXL3 tensor-parallel cut rules, KV quantisation, and the float64 oracles as tests.

Why:

1. Upstream's EXL3 kernels are faster than ours on every shape of this pack, by 5 to 18 times on one row. The
   float64 errors are of the same order (table below, measured). Keeping ours (A) would keep the slower kernels.
2. Upstream already has a Flash Next CUDA family with what we lack: a server, lanes, MTP chains up to 16 rows,
   prompt-state snapshots, a NCCL rank engine, an opt-in decode graph, vision, and a sparse indexer to 262,144 tokens.
   Our engine is a test harness (`zig/tests/cuda/exl3_fwd.zig`). It has no lane backend and no server
   (`exl3_lane.zig` says "No driver, no peer access, no serve loop"). Option B would rebuild what upstream already has.
3. Upstream does not run the pack today. The refusal is one config field, and the projection and experts entry points
   are single choke points (`cuda_batch.zig` `lane()` and `moeBlock()`). So the EXL3 adapter is small and additive.
   It rebases with few conflicts.

Option C does not register a new family: `flashnext.native` is already in `native/cuda.zig`. It also does not touch
`flashnext/config.zig` (the Metal affine config). The CUDA admission is `flashnext/cuda_config.zig`.

## 1. What v1.1.0 does with Flash Next on CUDA (read, then run)

Read from the code:

- Family: `zig/src/families/flashnext/cuda_launch.zig` exports `native = cuda_lanes.Namespace(Tri)`. It is
  registered in `zig/src/native/cuda.zig` (`registry = .{ ..., flashnext.native }`), `model_type = "qwen4_exp"`,
  `formats = { "mlx-q4g32", "modelopt-mixed" }`.
- Admission: `cuda_config.zig` `mode()`. A top-level `quantization` block selects `.affine`: MLX 4-bit, group 32,
  mode affine (`mismatch`). Otherwise it selects `.modelopt`, which requires `quantization_config.quant_method ==
  "modelopt"` and `MIXED_PRECISION` (`modeloptMismatch`): NVFP4 g16 routed experts, MXFP8 g32 shared expert and
  attention projections, no KV quant. Shapes (`shapes`) and semantics (`semantics`, `rope`) are fixed to this model.
  EXL3 is not a mode, so our pack falls into `.modelopt` and is refused at `quant_method`.
- Projections: `cuda_projection.zig` `Projection` is affine (MLX) or `native: ?cuda.qmmf.Weight` (BF16, MXFP8,
  NVFP4 lane). Every prompt and decode projection goes through `cuda_batch.zig` `Pass.lane()`. Experts go through
  `Pass.moeBlock()`: NVFP4 `native_experts`, or the affine `moe.run`.
- The EXL3 operators exist in the runtime but no Flash Next code calls them: `cuda/exl3.zig` (`Linear`, `Dense`:
  every codebook and width including mul1 half steps, rows 1 to 128 a call with the same bits at any row count,
  bf16, f16 or f32 out), `cuda/exl3_experts.zig` (`Experts.run` / `runSlots`: group, rotate, grouped gate+up,
  SwiGLU epilogue, grouped down, combine, with a per-expert trellis pointer table and a width per expert),
  `cuda/exl3_prompt.zig`, `cuda/exl3_candidate.zig` (#806's tiles), `cuda/exl3_upload.zig`, `core/exl3_load.zig`
  (header scan). `qlinear` takes `.exl3`. The changelog says: "no other family serves EXL3 natively yet". The only
  EXL3 family is GLM (`families/glm/cuda_exl3.zig`), which reads 4-bit `mcg` routed experts and keeps the rest bf16.
- Tensor parallel for Flash Next on CUDA: `cuda_rank_engine.zig` (`Session`). It runs one process per rank. A TCP
  control link (`cuda_rank_wire.zig`) replays rank 0's operations on rank 1, behind an identity handshake (config,
  index, binary and checkpoint sha). It uses one NCCL communicator (`cuda_collective.zig`, nonblocking, requires NCCL
  >= 2.30). It is allowed only for ModelOpt (`cuda_lanes_load.zig`: `tp != 1 and mode != .modelopt` gives
  `UnsupportedQuantization`). Here is what it splits (`cuda_modelopt_experts.zig` `Cut`, `cuda_rank_memory.zig`):
  - routed experts: every rank holds all 512, with half the intermediate channels (640 → 320);
  - shared expert: rank 0 only;
  - attention, GDN, HC, KV: replicated on both ranks;
  - MoE output: `Collective.sum` = `ncclAllReduce` fp32 over `y [rows, 11, 2560]` (per slot, before combine);
  - head: optional vocabulary split (`cuda_head.zig`), joined with `ncclAllGather` of bf16 halves.
- The CUDA native server does not expose `--tp` in 1.1.0. `native/cuda.zig` `serves = { "--mtp-drafts",
  "--mtp-confidence" }`, and `server/cli.zig` refuses `--tp` for this backend. Only `native/hip.zig` serves `--tp`.
  The recipe (`docs/recipes/flash-next-cuda.md`) calls the two-rank ModelOpt path experimental.
- `zig/src/families/flashnext/tp.zig`, `tp_seq.zig`, `tp_host_wait.zig` and `tp_settings.zig` are the Metal
  "speed-up mode" (two Macs over MCDMA, `@import("metal")`, `@import("fabric")`). They are not CUDA.
- Graphs: `cuda_round_graph.zig` is one bounded decode-round graph, opt-in with `TF_FLASHNEXT_DECODE_GRAPH=1`.
  It is used only when a collective exists (`cuda_prompt.zig`: `e.collective != null and round_graph.enabled()`).
  The recipe says "No CUDA graphs or GPU-side rounds" for the normal path.

Run on par1-cs13 (`~/tensorfold-up110`, built with `zig build -Dnvcc=nvcc -Dsm=120`, code=0, plus `zig build native`
for the server; the pack itself was never downloaded or modified):

| Command | Result (measured) |
|---|---|
| `tensorfold models` (default HF cache) | `no checkpoints on disk that a registered Zig family can serve` (it scans only the HF cache) |
| `HF_HUB_CACHE=<cache with a snapshot symlink to the pack> tensorfold models` | `local/Qwen3.8-Flash-Next-exl3-3.05bpw: Qwen3.8 Flash Next (qwen4_exp)`. `models` matches `model_type` alone, not the format |
| `tensorfold info <pack>` | `family: Qwen3.8 Flash Next (qwen4_exp)`, **`format: full precision`** (wrong: `hub.quantization` does not know EXL3), `weights: 79.29 GiB` (counts the 30.4 GiB n-gram file and the mixer patch) |
| `tensorfold run <pack> --tokens ...` (gpurun 1) | `error: NotNemotronH` (the CUDA `run` CLI is Nemotron only) |
| `tensorfold-native serve <pack> --backend cuda --context 4096` (gpurun 1) | `error: flashnext: config.json's quantization_config.quant_method differs from the supported Flash Next CUDA layout` / `error: UnsupportedCheckpoint`, exit 1, before any weight load |
| `tensorfold-native serve <pack> ... --tp 2 --rank 0` | `tensorfold serve: error: --tp is not served by the native engine yet` (usage error, exit 2) |

## 2. Upstream tp=2 on CUDA, and two 5060 Ti without peer access

Read:

- One process per rank, not one process driving several GPUs. GLM uses `TF_GLM_RANK` / `TF_GLM_WORLD` /
  `NCCL_COMM_ID` and `cuda/tp_group.zig` (`ncclCommInitRank`). Flash Next uses `--tp/--rank/--master/--master-port`
  options (`cuda_lanes.zig` `Options`), with rank 0's id sent over its own TCP link.
- Reductions. Flash Next uses `ncclAllReduce` sum in fp32 (`cuda_collective.zig` `sum`). That is NCCL's order, not
  rank order. GLM's prefill uses `cuda/rank_scatter.zig` (#642): `ncclSend/Recv` of fp32 partials, then
  `tf_rank_sum` adds them rank 0 first, so the bits do not depend on NCCL's algorithm. There is also a ring variant
  (`TF_GLM_PREFILL_REDUCE=ring`, #902). The RoCE gather (`roce.zig`, #852) is only for two hosts with HCAs; it does
  not apply here.
- EXL3 cut rules in upstream are GLM's (`exl3_tp2_tests.zig`: rank shards vs the Python engine's, all-gathered).
  Flash Next's 640-wide expert intermediate cannot use upstream's Flash Next intermediate split. 320 is not a multiple
  of the 128-wide Hadamard block, so an EXL3 expert cannot be cut there (our `exl3_tp.zig`, from `docs/recipes/exl3.md`).
  **EXL3 Flash Next cannot ride on upstream's CUDA tp as is.** It needs expert parallel (256 routed ids a rank) or an
  N-split down / K-split gate-up, which upstream's `Cut` does not have.

Measured on the two 5060 Ti (`nvidia-smi topo`: `NODE`, peer access unsupported):

- `tf-cuda-test exl3-tp2 <rank> 127.0.0.1:29611 none`: both ranks code 0. Comm sanity passed. The linear and experts
  shard checks SKIP: they need the Python engine's RED-SNOW fixture directory, which we do not have.
  NCCL 2.27.7 (the only NCCL on the box, in `/data/src/kt-venv`) logs `P2P is disabled between connected GPUs 0 and
  1` and `Channel 00 : 0[0] -> 1[1] via SHM/direct/direct`. The transport is host shared memory.
- `tf-cuda-test flashnext-rank <rank> 29612 sum` (the Flash Next `Collective`): this needs NCCL >= 2.30, so it cannot
  start on 2.27.7. With NCCL 2.32.3 from the `nvidia-nccl-cu13` wheel, unpacked to `~/tensorfold-up110/nccl232` and
  put on `LD_LIBRARY_PATH` (nothing installed system wide), both ranks print `PASS ... two-process chained NCCL FP32
  sums exact ...` and `PASS ... BF16 all-gather preserves every bit ...`, `via SHM/direct`.
- NCCL 2.32.3 all-reduce fp32 latency over SHM (`tf-cuda-test nccl-lat`, added on this branch; min over 5 batches of 100):

| bytes | what it would carry | ms |
|---:|---|---:|
| 10,240 | one row of hidden, fp32 | 0.0199 |
| 112,640 | one row's `y [11, 2560]` fp32 (upstream's MoE sum) | 0.0457 |
| 450,560 | 4 rows' `y` | 0.1571 |
| 1,802,240 | 16 rows' `y` | 0.5934 |
| 5,242,880 | 512 rows of hidden fp32 | 1.9835 |
| 41,943,040 | 4,096 rows of hidden fp32 | 15.8281 (2.65 GB/s) |

So SHM is fine for decode, at about 20 to 46 µs a reduction. For prefill it costs a lot. *Estimate*: upstream's
per-slot sum over a 2,048-row chunk is 2048 × 11 × 2560 × 4 B = 231 MB a layer, about 87 ms a layer and about
4.2 s a chunk. Summing after combine (2048 × 2560 fp32, 21 MB) is about 8 ms a layer. bf16 would halve that, if
the bits are allowed to change.

Open upstream PRs, read-only (`gh pr list/view --repo ashhart/TensorFold`, 95 open on 2026-10-18). Note: the
maintainer lands work by cherry-pick, so several PRs still shown OPEN are already in 1.1.0. #734, #746, #797, #798,
#852 and #952 are in the 1.1.0 merge commits. #740, #741, #806 and #837 (the EXL3 operators) are in 1.1.0.

| PR | State | What | Needed? | Risk |
|---|---|---|---|---|
| #1046 | open, not draft | Flash Next CUDA prompt 1.4-1.8x faster on GB10, bit for bit with 1.1.0; 30 files in `flashnext_*` kernels and `families/flashnext/` (+2500/-134) | Yes for prefill speed. Land it **before** our hooks in `cuda_batch.zig` / `cuda_prompt*.zig` | Medium: touches the same choke points we hook; affine/NVFP4 tiles only, EXL3 path unaffected |
| #1022 | open | `native/cuda_memory`: weightBytes counts every file the weight map names | Nice to have (our pack has extra files) | Low |
| #915 | open | process-wide cap on DeviceBuffer and pinned HostBuffer bytes | Useful for the offload pool's pinned staging budget | Low-medium |
| #1031 | open | fix(cuda): preserve first-token handoff through native host | Generic serving fix; take when merged | Low |
| #635 | open | keyed graph cache + device arena (DeepSeek stack 2/14) | No (pattern reference for per-window graphs) | High (14-deep stack) |
| #636 | open | fabric: tp across CUDA processes (NCCL, mailbox, RoCE, fail-fast) (DeepSeek 3/14) | No: Flash Next has its own rank engine | High |
| #637 | open | sessions: paged-KV page pool, prefix trie, NVMe tier (DeepSeek 4/14) | No now; reference for 4-sequence cache reuse | High |
| #728 | open | MiMo-V2.6-Flash on CUDA: EXL3, DFlash, two GPUs (stacked on #703), adds bindings to `cuda/exl3.zig` | No, but it will conflict with any edit we make to `cuda/exl3.zig`: watch | Medium |
| #667 | open | DeepSeek-V4.1 host config, EXL3 packs and rank weight plans | Reference for EXL3 rank plans | n/a |
| #895, #896 (draft) | open | Living Weights on Flash Next CUDA (pocket in every pass) | No; will conflict in `cuda_batch.zig` | Medium |
| #1023 | open | core/modelopt per-module recipes | No (ModelOpt only) | Low |
| #357, #378, #379 | closed | two-rank sums through shared pinned host memory where GPUs cannot map each other; Flash Next tp 4 | No; prior art for our host bounce | n/a |

No open PR is a hard dependency. Everything Option C needs is in 1.1.0.

## 3. EXL3 kernels: upstream vs ours, on the pack's shapes (measured)

Method. Upstream: `tf-cuda-test exl3-pack <pack> 20` (new file `zig/tests/cuda/exl3_pack_tests.zig` on this branch).
It runs real tensors in stored layout through `exl3.Dense.decode` (reference kernels, and #806 `tiles`), timed as 3
warm-ups, then the min over 5 batches of 20 of the mean, with cudaEvent. For error it uses an **exact float64**
product (`x · diag(suh) H Wq H diag(svh)`, no fp16 step). Ours: `tf-cuda-test exl3-bench` from `~/tensorfold-w5kern`
(our 512-thread `project_nblock_fast`), same timing method, on the same card in the same session. The weights of
every shape but `lm_head` fit the 32 MiB L2, so both sides are "warm L2" numbers. Cold-L2 numbers for upstream are
further down.

| Projection (K × N, halves) | ours rows=1 ms | upstream Dense rows=1 | upstream tiles rows=1 | speedup (Dense) | ours rows=5 | upstream rows=5 |
|---|---:|---:|---:|---:|---:|---:|
| L0 in_proj_qkv 2560×10240, 10 | 0.2437 | 0.0473 | 0.0372 | 5.2× | 0.7751 | 0.0477 |
| L0 in_proj_z 2560×6144, 10 | 0.1355 | 0.0237 | 0.0239 | 5.7× | 0.5053 | 0.0248 |
| L0 out_proj 6144×2560, 10 | 0.2885 | 0.0243 | 0.0239 | 11.9× | 0.6128 | 0.0248 |
| L3 q_proj 2560×12288, 10 | 0.2470 | 0.0476 | 0.0431 | 5.2× | 0.9144 | 0.0480 |
| L3 k_proj / v_proj 2560×512, 10 | 0.1211 | 0.0105 | 0.0067 | 11.5× | 0.1209 | 0.0105 |
| L3 o_proj 6144×2560, 10 | 0.2884 | 0.0243 | 0.0240 | 11.9× | 0.6134 | 0.0249 |
| L3 shared gate 2560×640, 10 | 0.1210 | 0.0105 | 0.0067 | 11.5× | 0.1210 | 0.0105 |
| L3 shared down 640×2560, 10 | 0.0310 | 0.0101 | 0.0066 | 3.1× | 0.0639 | 0.0101 |
| L3 indexer qk 2560×640, 6 | 0.1232 | 0.0104 | 0.0068 | 11.8× | 0.1230 | 0.0105 |
| lm_head 2560×248320, 10 (397 MB) | 6.3606 | 1.0970 | 0.9871 | 5.8× | 31.4567 | 1.1122 |

Routed experts, one token, ten picks, layer 0 (3-bit, halves 6):

- ours: separate launches, gate 0.1361 + up 0.1360 + down 0.1010 = **0.373 ms**, plus SwiGLU and combine;
- upstream `Experts.run` (group, rotate, grouped gate+up, SwiGLU epilogue, grouped down, combine, one call): **0.0657 ms**
  warm, **0.0765 ms** cold L2 (layer 17). That is 5.7× or more.

Bandwidth (measured): upstream's `exl3` bench gives `RESULT exl3.bench.decode 385.9` GB/s (4-bit 4096×32768). `lm_head`
cold gives 361 GB/s, against 448 GB/s peak. Ours on `lm_head`: 397 MB / 6.36 ms = 62 GB/s.

Cold L2 (upstream, a 64 MiB memset between calls, its own time subtracted): in_proj_qkv 0.0560, q_proj 0.0593,
out/o_proj 0.026, small 0.6-1 MB matrices 0.008-0.010 ms, lm_head 1.101 ms. Decode reads each expert cold, so these
are the decode-realistic numbers.

Numerics (measured, max |GPU − exact f64| / max |f64| over the row):

- upstream bf16 out: rel 1.6e-3 to 3.5e-3 on every shape. This is bf16 output rounding (half an ulp is about 2e-3).
- upstream fp32 out (`y_dtype = .f32`, same row): rel **1.75e-4 to 2.4e-4** on every shape (Dense and tiles). That is
  the fp16 rotated-input contract, the same as the Python engine's.
- upstream routed experts, fp32 out: rel 2.0e-4 (`act_mode 1`, fp32 SwiGLU); rel 1.5e-3 with `act_mode 0` (GLM's
  bf16 roundings).
- ours (`exl3-real`, fp16 out, layer 0 expert 0, against the Zig f64 decoder): gate max_abs 4.47e-4 at peak 1.388
  (rel 3.2e-4), down 6.2e-5 at peak 0.127. This is the same order as upstream; the remaining difference is the
  output dtype. Ours is bit-identical to our older one-warp kernel (`exl3-bench`: `equal=yes` on every shape) and
  built with `--fmad=false`. Upstream's guarantee is different: each row has the same bits at any row count 1-128
  (`exl3.linear.rows` PASS), and it matches the Python engine's fixture bytes (that fixture SKIPs here).
- Determinism: a second upstream call gives the same bytes on every shape (`repeat=same`).

Upstream's own suites on the 5060 Ti (measured, all exit 0): `exl3` (f64 for mul1 1-8 bits, mcg, inst3, half steps,
row invariance, qlinear, prompt linear and chunks, experts mul1 4/5/6, experts rows, prompt experts; the Python
fixtures SKIP), `exl3-tiles` (3 codebooks × 65,536 states bit-equal to `exl3_format.codebookValue`; 24 width/codebook
unpacks bit-equal; FP64 within 0.1424 of bound; row windows 1-16 and prompts 2-300 bit-stable; decodeF32 and promptF32),
and `glm-exl3` (within 0.186 of tolerance). Total card time for these three: 171 s.

**Verdict: upstream's EXL3 kernels win.** We drop ours.

Expert misses over PCIe (measured, same harness). Should a miss be read zero-copy from mapped pinned host memory,
or copied first?

| experts of 10 in mapped host memory | `Experts.run` ms | pinned H2D copy of the same bytes |
|---:|---:|---:|
| 0 | 0.0657 | n/a |
| 1 | 1.2058 | 0.1454 ms (12.7 GB/s) |
| 4 | 3.6459 | 0.5311 ms (13.9 GB/s) |
| 10 | 6.8670 | 1.3572 ms (13.6 GB/s) |

Zero-copy is 5-8 times slower than the copy engine. **Misses must be staged H2D (copy engine) into a device pool line,**
as our pool does. Then the device pointer table entry is rewritten. The numbers above assume nothing is reading the
copy engine at the same time.

## Component by component: ours or upstream's

| Area | Ours | Upstream 1.1.0 | Verdict | Evidence | Action |
|---|---|---|---|---|---|
| EXL3 one-row linear | `exl3_linear.cu` 512-thread `project_nblock_fast`, strips order, fp16 out | `exl3.Dense` / `Linear` (split-K plan, 4/8 warps, stored order, bf16/f16/f32 out), #806 `tiles` | **upstream better** | measured: 5.2-11.9× faster at rows=1, flat at rows=5 where ours is linear in rows; same-order f64 error | adopt upstream, drop ours |
| EXL3 prompt linear | ours: the one-row kernel per row | `Dense.prompt` 128-row blocks, row-invariant bits; `exl3_prompt.cuh` | **upstream better** | read; rows=5 measured flat (0.048 vs 0.775 ms) | adopt upstream |
| Routed experts, SwiGLU, combine | 4 launches, ids from a pointer, `hit` lines | `Experts.run/runSlots` grouped, per-expert pointer table, width per expert, fused epilogues | **upstream better** | measured 0.0657 vs ≥0.373 ms | adopt upstream; the offload pool rewrites its pointer table |
| GDN layer | `exl3_gdn.cu` / `_tp.cu`, our state table | `flashnext_gdn.cu`, `cuda_gdn.zig`, chunked prompt chain (#1046 faster) | **upstream better** (integrated with lanes, prompt chunks, snapshots); ours not compared on time | read; ours tested vs f64 oracle | adopt upstream; keep our `exl3_gdn_cpu.zig` oracle as a test |
| Attention + sparse indexer + KV | `exl3_attn.cu` prep/pool/scores/select/decode, bf16 KV | `flashnext_attn.cu`, `cuda_attn.zig`, `cuda_batch` attnBlock (indexer to 262,144, fused chunk attention in #1046) | **upstream better** | read | adopt upstream; keep `exl3_attn_cpu.zig` as oracle |
| Hyper-connections | `exl3_hc.cu` (f32 streams, F16 weights) | `flashnext_hc.cu`, `cuda_hc.zig`, `hcWriteback/readout` | **equal in scope**; upstream's is wired | read | adopt upstream; add an F16 HC weight path (pack stores HC as F16) |
| Router | host top-10 pick (one sync a layer) | device `routerMany` + `topkMany`, no host sync | **upstream better** | read | adopt upstream (F16 `mlp.gate` must be read exactly: see risks) |
| PLE n-gram rows | EXL3 row codec (`exl3_ngram.zig`), ids (`exl3_ple_ids.zig`), mmap and pread readers with a 4096-row hot cache, byte-checked against numpy | `cuda_ple.zig` `Ngram` (mapped shards, `madvise RANDOM`, host hash, formats `mlx`/`nvfp4`) | **not comparable**: upstream cannot decode EXL3 rows; our reader adds pread and a hot cache | read + our earlier S6b runs | keep ours as `Ngram.format = .exl3` + optional pread reader (an addition) |
| MTP head + speculation | single draft, host maths (`exl3_mtp_spec.zig`), state rollback | chains up to 16 rows, confidence rule, verify windows, rollback, drafts per lane (`cuda_mtp*.zig`) | **upstream better** | read | adopt upstream; add an EXL3 head loader (fc_embedding/fc_hidden EXL3, mixer patch file, draft rows dequantised from the EXL3 lm_head) |
| tp=2 | host bounce: D2H fp16 partials, host add in rank order, H2D; head split (GDN 24 v-heads/8 k-heads a rank, attn 12 q + 1 kv a rank), expert parallel 256 ids a rank | NCCL rank engine, replayed ops, fp32 allreduce, replicated attention/GDN/KV, intermediate split, ModelOpt only, `--tp` not served on CUDA | **mixed**: upstream's transport and process model are better (stream-ordered, graph-capturable, health, identity); our cut rules are the only legal ones for EXL3 and halve KV per card | measured NCCL SHM latencies above; our bounce not timed in this task | adopt upstream's rank engine + `Collective`; bring our cut rules (expert parallel first, head split second) |
| CUDA graphs | per-layer segments A/B around the host pick, graph-safe id rewrite | one bounded decode-round graph, recaptured on owner/geometry change, tp only, opt-in | **not comparable** yet: ours exists only because of the host pick; upstream has no host pick | read | adopt upstream's round graph; keep one segment cut only where offload needs host knowledge (see plan) |
| Prefix cache / state snapshots | `exl3_cache.resumeAt` policy, `pleSnapshot/Restore` | `cuda_snap.zig` device copies of a stream's state at `at`, prompt cache store, `cache_rules` (lookahead 1, min gap 256) | **upstream better** | read | adopt upstream; extend snapshot with the EXL3 PLE tail if its shape differs (it should not) |
| KV quant | `exl3_cache.zig` int8/int4 32-point Hadamard policy (host) | none for Flash Next (bf16 KV) | **ours only** | read | keep, later, as an addition (needed for 200k × 2-4 sequences on 16 GB) |
| Expert offload | pool (`cuda/exl3_offload.zig`), policy (`exl3_offload.zig`: LRU + use count, in-flight lines, swap interval), staged H2D on a copy stream | none: all weights resident (GB10 128 GB, RTX PRO 6000 96 GB) | **ours only** | measured above: copy engine 13.7 GB/s vs zero-copy 1.5 GB/s | keep as the main addition; retarget to `Experts`' pointer table |
| Tests and oracles | real-weight tests per stage vs Zig f64 oracles (`exl3_fwd_cpu`, `_gdn_cpu`, `_attn_cpu`, `_moe`, `_ple_cpu`) | synthetic f64 + Python-engine fixtures + Triton captures + kernel contract | **complementary** | read | keep our oracles as test-only modules; propose a real-weight EXL3 fidelity test upstream |

## Ours that could go upstream (not opened; one-line pitches)

1. `exl3-pack` bench + exact-f64 check on real EXL3 tensors (`zig/tests/cuda/exl3_pack_tests.zig`): "Dense and
   Experts on a real mixed-width mul1 checkpoint, fp32-out error vs an exact float64 product, warm and cold L2."
2. `tf-cuda-test nccl-lat`: "NCCL all-reduce latency at decode and prefill sizes, to size rank reductions on
   PCIe-only rigs (SHM transport at 2.65 GB/s on two 5060 Ti)."
3. Measured guidance for anyone offloading EXL3 experts: "copy-engine staging (13.7 GB/s) beats mapped zero-copy
   reads (about 1.5 GB/s effective) by 5-8× in `Experts.run`".
4. Flash Next CUDA tp: "sum after combine, not per slot (11× less traffic: 231 MB vs 21 MB a layer per 2,048-row chunk
   on SHM)". This is an *estimate* from the measured SHM rate.
5. `tensorfold info` reports EXL3 checkpoints as `format: full precision`: `hub.quantization` should name
   `quant_method exl3` (bits 3.05, head 5). A small fix.
6. EXL3 expert-parallel / N-split cut rules for 640-wide experts (`exl3_tp.zig`, `exl3_tp_split.zig`, host-tested):
   "the legal EXL3 cuts when the intermediate is not a multiple of 128 × ranks".
7. The EXL3 n-gram row codec (`exl3_ngram.zig`, byte-checked against numpy/ExLlamaV3 `ngram_dequant`) as a third
   `Ngram.format`.
8. The stream-ordering hazard (`cuMemsetD8` / pageable `cuMemcpyHtoD` are not ordered against non-blocking streams).
   Upstream already guards most of this: `copy_source.synchronousCopy` synchronises the context when sources are
   retained. Candidate sites to audit, read only and not shown to be bugs: `cuda_prompt_step.zig:223`, `:298`,
   `:307-308` (the reference row path), and `cuda_select_main.zig:63` (`fill8(..., null)`). Pitch it only as a
   regression test.

What upstream does better, where we simply drop ours: the linear and expert kernels, the prompt path, the device
router, GDN/attention/HC glue, MTP chains, snapshots, lanes and server, and the rank process model.

## 4. Plan (Option C), in dependency order

Effort is in agent-days, one agent with par1-cs13. These are *estimates*.

0. **Prereqs (0.5 d).** Rebase base = `v1.1.0`. Take #1046 first if it merges, otherwise land on 1.1.0 and accept a
   later conflict in `cuda_batch.zig`. Runtime needs NCCL >= 2.30 for tp. On par1-cs13 that means a user-space copy
   via `LD_LIBRARY_PATH`, or adding it to `ansible/roles/par1_cs13_cuda` (an ansible run, so a user decision).
1. **Admission (0.5 d).** `cuda_config.zig`: `Mode.exl3` when `quantization_config.quant_method == "exl3"`,
   `codebook == "mul1"`, geometry + `semantics` unchanged. Widths are read per tensor from the trellis, never from
   `bits`. Add `"exl3-mul1"` to `formats` in `cuda_lanes.zig`. Host test: our pack's config passes, other codebooks
   refuse by name. `cuda_rank_memory.zig` / `deviceWeightBytes`: EXL3 sizes; routed experts are host-resident under
   offload.
2. **Store + names (1 d).** EXL3 names are `model.language_model.layers.N.*`, `lm_head`, `mtp.*`, plus the mixer in
   `mtp_hyper_connection_mixer_patch.safetensors`, which is outside the index. Upstream's MLX names are
   `language_model.model.layers.N.*`. Add a name map in `cuda_prompt_wt.zig` `Store`, and open the patch file.
   Reuse our `exl3_scan.zig` (groups, `centredOffset`) and `exl3_place.zig` rules.
3. **F16 dense tensors (1 d).** `mlp.gate`, `shared_expert_gate`, all HC tensors and `in_proj_a/b` are F16 in the pack
   (A_log, dt_bias, conv1d and linear_attn.norm are BF16). So are the PLE key/value. Upstream's loaders require bf16
   (`cuda_prompt_load.zig:228`, `:236`; `loadHc`; `projection.dense`). f16→bf16 loses 3 mantissa bits. Add an
   fp32-widened or f16 dense path. The router's top-10 is tie-sensitive, so check the picks against our
   `exl3_route` f64 oracle on real layers.
4. **Projections (1 d).** `Projection.exl3: ?cuda.exl3.Weight` (stored layout, no strips reorder). In `Pass.lane()`,
   `if (w.exl3) |e|` then `Dense.decode/prompt` (bf16 in; bf16 out, or fp32 where `sk > 1` callers read `part`).
   The Flash Next family owns one `exl3.Dense` per engine (its scratch). Test: real layers 0 and 3 against
   `exl3_fwd_cpu` (f64).
5. **Experts, resident subset (1 d).** In `moeBlock()`, an `exl3_experts` branch: `Experts.runSlots` with 513
   entries. The shared expert is id 512 in the same pointer table, since it has the same 640 shape, which matches
   upstream's 11-slot `y`. Suh/svh are stacked `[513, ·]` and resident (19.2 KB an expert, about 470 MB for 48 layers).
6. **Offload pool on the pointer table (3 d).** Routed trellis stays in pinned host RAM (about 45 GB of 3-bit
   experts). Device lines come from our `cuda/exl3_offload.zig` Pool, simplified: one region a line, since upstream
   reads the stored layout. Our policy (`exl3_offload.zig` LRU + use count, in-flight until reaped, swap every 4
   rounds, at most 96 lines) writes the `gate/up/down` pointer table and the k2 tables. A hit costs no host work.
   A miss needs the picks on the host. Keep our segment cut: graph A (block + router + `topkMany`) → D2H of
   `picks` (10 × int32 a row) → policy → staged H2D of misses on the copy stream → event → graph B (experts and
   on). Measured parts: 0.145 ms a 3-bit expert at 13.7 GB/s; `Experts.run` 0.0765 ms cold.
7. **PLE (1 d).** `Ngram.format = .exl3` with our `exl3_ngram.zig` decode in `Ngram.rows`. Compare
   `exl3_ple_ids.zig` with upstream's `Hash`: if the ids are byte-equal, drop ours. Our pread + hot-row cache is an
   optional reader behind the same `rows()` signature.
8. **MTP (1.5 d).** `cuda_mtp.zig` `Head.init` from EXL3. `fc_embedding`/`fc_hidden` are EXL3 projections. The
   mixer comes from the patch file. `cuda_mtp_load.draft` builds the draft vocabulary rows by dequantising the EXL3
   `lm_head` columns (`exl3_format.dequantize` on the 128-column blocks that hold them) to bf16, since EXL3 columns
   cannot be copied out.
9. **Single-card end-to-end (1 d).** `tensorfold-native serve <pack> --context 32768`, greedy. Compare teacher-forced
   top-1 against our f64 oracle on the 16-token prompt, then the "capital of France" continuation, then
   drafted == `"draft": false`.
10. **tp=2, expert parallel (2.5 d).** Allow `Mode.exl3` in `cuda_lanes_load.zig`'s tp gate. Rank r holds routed ids
    256r..256r+255 (plus shared on rank 0), with the attention/GDN/KV replicated as upstream does. Its expert slots
    for foreign ids are masked (`pick = -1`, which upstream's kernels skip). Sum after combine (`[rows, 2560]` fp32)
    with `Collective.sum`, or `rank_scatter` for rank-order bits. The CUDA server's `--tp` gate (`native/cuda.zig`
    `serves`) has to be opened. That is a user call: it changes the native server's flags.
11. **Graphs (2 d).** Use upstream's `cuda_round_graph` for segment B onward, and a second graph for segment A, keyed
    by the round's owners. The NCCL sum sits inside a graph (NCCL supports capture). Graph-safe: pool swaps rewrite
    the pointer table in device memory, not graph parameters.
12. **Context and memory (2 d).** KV quant from our `exl3_cache.zig` (int8 first), for 200k × 2 sequences.
    Phase 2 of tp: head split (our GDN/attention cuts). This halves KV per card, but it needs head counts as
    arguments in `flashnext_gdn.cu` / `flashnext_attn.cu`, which conflicts with upstream. Do it only if int8 KV
    plus expert parallel does not reach the target.

Total: about 17 agent-days to a served, tp=2, graphed EXL3 Flash Next. Steps 1-9 (single card, served, offload)
are about 10.5 days.

### Our 113 files under Option C

- **Move (adapt and keep, about 20 source files):** `exl3_ngram.zig`, `exl3_ple.zig` (pread/hot rows),
  `exl3_ple_ids.zig` (until compared), `exl3_offload.zig` (policy), `cuda/exl3_offload.zig` (pool, simplified),
  `exl3_step.zig` (round miss plan), `exl3_scan.zig`, `exl3_place.zig`, `exl3_admit.zig` (folded into `cuda_config`),
  `exl3_shape.zig`, `exl3_load.zig` (header walk), `exl3_tp.zig` and `exl3_tp_split.zig` (cut rules), `exl3_cache.zig`
  (KV quant, budget), `exl3_window.zig` / `exl3_graph.zig` (what replay may rewrite), `exl3.zig` barrel,
  `tools/exl3_tok.py`, `tools/exl3_ple_ids.py`, `docs/plans/exl3-flashnext-offload.md` (history).
- **Keep as test-only oracles (about 12):** `exl3_cpu.zig`, `exl3_fwd_cpu.zig`, `exl3_attn_cpu.zig`,
  `exl3_gdn_cpu.zig`, `exl3_moe.zig`, `exl3_norm.zig`, `exl3_route.zig`, `exl3_router.zig`, `exl3_ple_cpu.zig`,
  `exl3_fwd_ple.zig` (keep one of the two PLE references after cross-checking), `exl3_mtp_spec.zig`, plus the tests
  `exl3_real_tests.zig`, `exl3_offload_tests.zig`, `exl3_ple_tests.zig`, `exl3_moe_tests.zig` and `exl3_fwd_plerows.zig`,
  ported to upstream's `Dense`/`Experts` and family. The about 35 one-line `zig/tests/cuda/*.zig` symlinks follow.
- **Drop (superseded by upstream, about 25):** kernels `exl3_linear.cu`, `exl3_experts.cu`, `exl3_decode.cuh`,
  `exl3_check.cu`, `exl3_gdn.cu`, `exl3_gdn_tp.cu`, `exl3_attn.cu`, `exl3_hc.cu`. Device modules `cuda/exl3.zig`
  (ours), `cuda/exl3_attn.zig`, `cuda/exl3_gdn.zig`, `cuda/exl3_hc.zig`, `cuda/exl3_tp.zig`. Host split helpers
  that upstream does on device or through NCCL: `exl3_reduce.zig`, `exl3_head.zig`, `exl3_hc.zig`, `exl3_gdn.zig`,
  `exl3_attn.zig`, `exl3_sample.zig`, `exl3_embed.zig`, `exl3_kv.zig`, `exl3_moe` wiring, `exl3_lane.zig` and
  `exl3_strips.zig` (upstream reads the stored layout). Harness engines `exl3_fwd.zig`, `exl3_fwd_tp.zig`,
  `exl3_fwd_graph*.zig`, `exl3_fwd_wire.zig`, `exl3_mtp.zig`, plus `exl3_bench.zig` (replaced by `exl3_pack_tests.zig`),
  `exl3_tp_tests.zig`, `exl3_attn_tests.zig`, `exl3_gdn_tests.zig`, `exl3_window_tests.zig` and `exl3_fwd_tests.zig`.
  Keep these on the old branch for reference until the family passes the same checks.

### The 8 conflicts of a 3-way apply (`git merge-tree v1.1.0 origin/exl3-flashnext-window-wip`), and how to resolve them

| File | Kind | Resolution under C |
|---|---|---|
| `zig/build/cuda.zig` | content | Take upstream's kernel list. Drop our 6 entries: `exl3_linear` and `exl3_experts` now name upstream's kernels; `exl3_gdn`, `exl3_gdn_tp`, `exl3_attn` and `exl3_hc` are superseded. If an oracle test still needs one, re-add it under a distinct prefix (`fnx3_*`). |
| `zig/kernels/cuda/exl3_decode.cuh` | add/add | Take upstream's (141 lines). Ours (486) is dropped. |
| `zig/kernels/cuda/exl3_experts.cu` | add/add | Take upstream's (504). Ours (117) is dropped. |
| `zig/kernels/cuda/exl3_linear.cu` | add/add | Take upstream's (274). Ours (44) is dropped. |
| `zig/src/cuda/exl3.zig` | add/add | Take upstream's `Linear`/`Dense`. Our `Exl3` API (`linear`, `experts`, `swiglu`, `hit`, `combine`, `probeMul1`) is dropped. The mul1 probe is covered by `exl3-tiles`' 65,536-state check. |
| `zig/src/cuda/kernels.zig` | content | Take upstream's. Remove our `exl3_*_symbols` blocks. |
| `zig/src/cuda/root.zig` | content | Take upstream's. Add only `pub const exl3_pool = @import("exl3_offload.zig")` (renamed to avoid a second `exl3_*` meaning). |
| `zig/tests/cuda/main.zig` | content | Take upstream's. Re-add one line per kept command (`exl3-pack`, `nccl-lat`, `exl3-offload`, `exl3-ple`, and the family fidelity check), as new lines only. |

Under A, the same 8 would instead need every one of our files renamed (`fnx3_*`) to coexist with upstream's.

## What the targets need

| Target | Under C | Main risk |
|---|---|---|
| 200k context | upstream's sparse indexer and window reach 262,144. KV is 2,304 B a token and attention layer, so 5.5 GB a sequence at 200k with bf16 KV (`seqBytes`, read) | 16 GB cards: 2 sequences of bf16 KV is 11 GB on a card if replicated. Needs int8 KV (step 12) or head split |
| 2 concurrent sequences | lanes core, `--parallel 2`, shared rounds (upstream) | the expert pool is shared, so two sequences' picks contend for lines |
| 4 with cache reuse | `cuda_snap.zig` prompt-state copies plus the prompt cache budget (upstream) | snapshot memory competes with the pool |
| tp=2 | upstream rank engine + `Collective` over SHM (works, measured), expert parallel (step 10) | `--tp` is not served on CUDA in 1.1.0 (server flag gate); NCCL >= 2.30 must be installed; upstream rejects tp+MTP+prompt cache (`TensorRankDraftSnapshotUnavailable`) |
| graphs | upstream round graph (tp only today) + our A/B cut for offload | NCCL inside a captured graph on SHM is untested here |
| prefill | `Dense.prompt` 128-row blocks, grouped experts on all rows at once; #1046 for the affine side | SHM reductions at 2.65 GB/s cost about 8 ms a layer a 2,048-row chunk (*estimate*); per-slot sums would cost about 87 ms (*estimate*) |
| decode | kernels are cold-measured at about 0.06 ms a big projection and 0.077 ms a routed block | misses dominate: *estimate* 3.5 misses a layer × 0.145 ms × 48 ≈ 24 ms a token at a 65% hit rate, against about 15 ms of kernels |

## Risks

1. **F16 tensors.** The pack stores the router, the shared gate, HC and `in_proj_a/b` as F16, but upstream's loaders
   take bf16. A lossy conversion can flip top-10 picks. Mitigation: exact widening, plus pick checks against our oracle.
2. **Name and file layout.** The EXL3 pack's tensor names and the mixer patch file differ from the MLX checkpoint's.
   The `Store` map must cover every name, or the load fails late.
3. **Upstream churn** in `cuda_batch.zig` / `cuda_prompt*.zig` (#1046, #895/#896). Our hooks should be one-line
   branches into new files (`cuda_exl3_*.zig`) to keep rebases small.
4. **tp.** Upstream's tp path is experimental and ModelOpt only, `--tp` is not served on CUDA, and NCCL >= 2.30 is
   needed. Expert parallel leaves KV replicated. Head split means editing upstream kernels.
5. **Offload and graphs.** The host pick costs one sync a layer (48 a token) and limits graph length. Speculative
   prefetch of the next layer's experts is not designed yet.
6. **Numerics.** Upstream's kernel order is not ours, so tokens may diverge from our 0.4 s/step engine after many
   steps. Acceptance should be against the f64 oracle and teacher-forced top-1, not token equality with our engine.
7. **Draft vocabulary rows** from an EXL3 head need a dequantise step. That is new code, and the drafts become
   approximate. Acceptance rule: drafted == `"draft": false`.

## Not verified in this task

- Upstream's `exl3-tp2` linear and expert shard checks (they need the Python RED-SNOW fixtures). Only its comm
  sanity ran.
- Our tp=2 host-bounce time per step, and NCCL inside a CUDA graph on SHM.
- Any end-to-end token from upstream's family on this pack. It refuses at admission, so nothing past that ran.
- That upstream's n-gram `Hash` ids equal `exl3_ple_ids.zig`'s.
- Ours versus upstream's GDN, attention and HC kernel timings (not benchmarked side by side).
- Upstream kernels at prompt sizes (rows 16-128) on this pack. Only rows 1 and 5 were measured.

## How to reproduce

```bash
# par1-cs13, tree = v1.1.0 + this branch's exl3_pack_tests.zig
cd ~/tensorfold-up110 && zig build -Dnvcc=nvcc -Dsm=120 -j16
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-pack /data/models/Qwen3.8-Flash-Next-exl3-3.05bpw 20          # ~280 s (f64 refs)
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-pack /data/models/Qwen3.8-Flash-Next-exl3-3.05bpw 20 cold     # ~7 s
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-pack /data/models/Qwen3.8-Flash-Next-exl3-3.05bpw 20 experts  # ~21 s
# nccl-lat: one process a card inside one lock, NCCL >= 2.30 on LD_LIBRARY_PATH
~/bin/gpurun both bash -c 'export LD_LIBRARY_PATH=$HOME/tensorfold-up110/nccl232/x/nvidia/nccl/lib; rm -f /tmp/id;
  CUDA_VISIBLE_DEVICES=1 ./zig-out/bin/tf-cuda-test nccl-lat 1 /tmp/id & CUDA_VISIBLE_DEVICES=0 ./zig-out/bin/tf-cuda-test nccl-lat 0 /tmp/id; wait'
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3 ; ... exl3-tiles ; ... glm-exl3
cd ~/tensorfold-w5kern && ~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-bench /data/models/Qwen3.8-Flash-Next-exl3-3.05bpw 20
```
