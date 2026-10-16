# EXL3 Flash Next: graphs, PLE, tp=2, expert offload, cache

Branch `exl3-flashnext-offload` in
`/Users/kletorch/Projects/Infra/TensorFold-exl3-flashnext`.
Base is upstream `ashhart/TensorFold` tag `v1.0.12` at
`833bbfcb` ("TensorFold 1.0.12: Nemotron 3.5 Super VL on CUDA..."),
fast-forwarded 2026-10-16 from `c7137bb3` (1.0.11). The fork is
`BlivionIaG/TensorFold`. This branch does not track upstream.
Do not push it to `ashhart/TensorFold`.

Status on that base, host-tested (`zig test zig/src/families/flashnext/exl3.zig`,
96 passed): steps 2–10 and the host half of 11. The router
(`exl3_route.zig`) is v0.6.6 `_topk_rows`: lower-id ties, softmax
over the chosen routed logits, shared expert in the last slot. CUDA linear and
expert fatbins were checked earlier on sm_120; the expert kernel's
epilogue is still identity, and `exl3_cpu.swiglu` is the fp32
`silu(gate) * up` from v0.6.6 `act_mode` 1. VRAM residency is
`stage` / `reap` plus LRU (`schedule`): a copy is not a hit until
reap, an in-flight line is not reusable, and a miss is staged only
inside the copy budget. `planRound` still does not write the
frequency table. `observe` / `seedHottest` / `scheduleUsed` keep a
lifetime count beside that LRU: empty lines fill with the hottest
ids, and a one-off does not evict a high count. `fillOwned` then
uses any line still empty, and only with this rank's ids.
`Stay` pins the every-token GEMMs (shared expert, q/k/v/o,
GDN `in_proj` / `out_proj` / `conv1d`, router, hyper-connection,
centred norms, MTP, `lm_head`). The embedding table stays on the
host: one row is 5 KiB and the table is about 1.2 GiB. The n-gram
stays on disk, with a 32-row cache in front of `pread`. `--tp 2`
owns expert ids 0..255 on rank 0 and 256..511 on rank 1.
`expertBytes(2)` is 948,480 bytes, the half-matrix cut: a
geometry helper, not the residency line. One owned expert's
budget line is the measured full expert, 1,901,213 bytes, until
a header span is larger. `learnedBytes` is what a 16 GiB card
has left after pinned weights, KV, and a 256 MiB reserve. The
line count is that room divided by the stride at load. KV stays
bf16, one head per rank, while that room still covers the expert
floor. `residentBytes` is the device-admission figure (routed
experts, n-gram, vision, the embedding table, and other plain
tensors left out). Admission no longer invents a 16 GiB cache on
top of the weights. The CUDA family is not registered, so `--ple-on-ssd`,
`--ssd-experts`, `--kv-dtype`, and `--tp` stay refused. No serve is
running. Do not publish tok/s.
The dirty `rocm-pr` checkout at `/Users/kletorch/Projects/Infra/TensorFold`
stays where it is.

Target checkpoint: `turboderp/Qwen3.8-Flash-Next-exl3`, revision
`3.05bpw_h5_ng5` (mirror `doth4580/Qwen3.8-Flash-Next-EXL3-3.05bpw`).
Text serving only. The vision encoder is out of this plan.
`config.json` for that revision was read on 2026-10-16. The numbers
live in `zig/src/families/flashnext/exl3_shape.zig`.

No tok/s number in this plan is a claim about this branch. The 0.6.6
Spark figures in `docs/recipes/qwen3.8-flash-next.md` are history.

## What main already is

Native TensorFold 1.0 is Zig 0.17. `src/tensorfold/` is gone.
`docs/recipes/exl3.md` says no model serves an EXL3 checkpoint.
Present on main:

- Format and reference decoder: `zig/src/core/exl3_format.zig`
  (codebooks `inst3`, `mcg`, `mul1`, tile bitstream, Hadamard).
  Oracle: `zig/tests/fixtures/exl3_oracle.bin`.
- Metal projection only: `zig/src/core/exl3_projection.zig`.
- CUDA graphs as a driver wrapper: `zig/src/cuda/graph.zig`.
- Flash Next engine for MLX 4-bit / affine packs:
  `zig/src/families/flashnext/`. Two-rank code in `tp.zig` exchanges
  expert rows and adds partials in rank order. It is not an EXL3 cut.
- N-gram scale kernel `ple_scale.zig` is the MLX bf16 scale, Metal only.
- Prompt reuse already exists: `zig/src/core/prompt_cache.zig` and
  `zig/src/core/prompt_radix.zig`. Extend those. Do not add a second cache.

CUDA EXL3 projections and experts are upstream PRs #740, #741, #806,
not on this commit. The first model those PRs name is the 27B, not
Flash Next.

## What 0.6.6 did, and the gap

Tag `v0.6.6`, module `src/tensorfold/cuda/exl3/` and
`src/tensorfold/families/qwen4_exp/cuda/exl3.py` (@vcruz305 #42):

- One GPU. `load()` raises if `tp[1] > 1`.
- `--ple-on-ssd` refused. The n-gram table is memory-mapped.
- Grouped expert kernel is graph-capturable because it reads the
  routing plan from device memory.
- No host-resident expert cache, no PCIe-versus-CPU miss split.
- `--kv-dtype bf16|int8|int4` and prefix state exist in that Python
  engine. The native binary refuses `--kv-dtype`.

This branch has to run all of the following on the 3.05 pack at once:

1. CUDA graphs for verify windows and draft steps.
2. PLE, including an SSD / page-cache mode equivalent to `--ple-on-ssd`.
3. tp=2.
4. MoE expert offload in the Strata / FreeToken shape.
5. KV cache quant and prefix reuse.

## Pack facts the code must not invent past

From `3.05bpw_h5_ng5` `config.json`:

- `model_type` `qwen4_exp`. 48 layers. Pattern is three
  `linear_attention` then one `full_attention`, twelve times.
- hidden 2560, vocab 248320, heads 24, kv heads 2, head dim 256.
- 512 routed experts, top-10, expert intermediate 640, shared
  intermediate 640. Shared expert is id 512 of the same table.
- hc_count 4, hc_lowrank 320. PLE embed 2560, conv 4,
  `ple_layer_ids` is `[2]` in the published file. Read the list.
  Do not assume every layer has a PLE table.
- n-gram size 3, 8 heads per n-gram, vocab base 20_000_000,
  128 shards. Row codec (not EXL3 tiles): one fp16 scale, then
  160 tail-biting mul1 values. Branch `ng5` means 5 bits, so
  51 int16 words per row. Decoder is the Triton `_ple_rows` in
  `v0.6.6` `exl3_mm.py`, which matches ExLlamaV3 `ngram_dequant`.
- Quant: method `exl3`, version `1.4.4`, codebook `mul1`,
  average 3.05, head 5, MTP 3, vision 5. Average bits are not a
  tensor width.
- MTP: 1 hybrid full-attention layer. Final mixer lives in
  `mtp_hyper_connection_mixer_patch.safetensors`, outside the index.
- Centred RMSNorms are stored as gamma-1. `centred_offset` in
  `v0.6.6` `exl3.py` returns 1.0 or 0.0 from the per-tensor means.
  Names that take the offset: `attn_hyper_connection.hc_norm`,
  `q_norm`, `k_norm`, `indexer.q_layernorm`, `indexer.k_layernorm`,
  `norm_key`, `norm_query`, `norm_conv`,
  `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`.
- Unquantized, fp16 matmul, no cuBLAS: HC down / inject / up,
  DeltaNet `in_proj_a` / `in_proj_b`, n-gram key and value.
- Router is fp32 top-k with id tie-break. Write-back is slot order.

## Memory

A 4-bit ceiling (above the 3.05 average) for one expert matrix of
shape 2560 x 640 is 6400 tiles * 128 bytes plus fp16 suh/svh, about
825 KB. Three matrices are about 2.5 MB. 512 experts * 48 layers is
about 60 GB. The real 3-bit tensors are smaller and still tens of GB.
Two 16 GB cards cannot hold them.

The 5-bit n-gram table is about 20e6 * 16 heads * 102 bytes, about
30 GB. It stays in the file. Page cache may touch it. VRAM must not
allocate it.

Dense projections, the shared expert, routers, HC, the MTP head,
graph workspace, and the KV cache are the resident set. Context
length is the knob that has to fit in what remains.

Acceptance machines:

- par1-cs13: Ubuntu, 251 GiB RAM, two RTX 5060 Ti 16 GB, SM 12.0,
  peer access unsupported, AVX2 and no AVX-512. Offload is required.
  A CPU miss path may use AVX2 only.
- DGX Spark UMA is where the 0.6.6 numbers were measured. A resident
  path is allowed there when the budget says the experts fit. The
  same tokens must come out of the offload path.

## Offload, graphs, tp

Strata (design, not code to copy): routed experts in RAM, a hot set
in VRAM, dense and shared and router and HC and MTP head in VRAM,
n-gram table on SSD plus the OS page cache. One CUDA graph per
window replays expert ids. Each layer: hits run on the GPU; a share
of the misses is copied by the copy engine; the rest is computed on
the CPU where the weights sit; results are written back and summed.
Every four rounds, up to 96 experts the conversation asked for
replace cold cache lines. The cache is seeded from a routing profile.

FreeToken (`python/freetoken/moe/`, `offload_cache.py`,
`expert_banks.py`, `host_banks.py`, `cpu_executor.py`): same split.
There is no EXL3 fused kernel there. The EXL3 miss path is new.

Rules for this branch:

- The long-term GPU cache is not updated by copying every miss.
  A miss is either a one-round PCIe compute or a one-round CPU
  compute. Installation happens on the swap interval.
- Hits are a device-side slot lookup. The graph reads ids and slot
  pointers from buffers, so replay does not recapture.
- Copy budget is bytes per round, from measured PCIe GB/s once a
  device exists. Until then the policy takes the budget as an argument.
- Swap: every 4 rounds, at most 96 experts, hottest requested ids
  that are absent, evict the coldest residents. Shared id 512 never
  takes a routed slot.
- CPU compute uses `exl3_format.zig` as the bit oracle. AVX-512 is
  not available on par1-cs13.

tp=2 cuts, because EXL3 splits must keep 128-wide Hadamard blocks
and 16-wide tiles (`docs/recipes/exl3.md`):

- `hidden/2 = 1280` is legal. `640/2 = 320` is not.
- Gate and up are K=2560, N=640. Split K (inputs). Do not split N.
- Down is K=640, N=2560. Split N (outputs). Do not split K.
- q, k, v, o and the head split on the axis whose length divides
  by `128 * ranks`.
- `hc_lowrank` 320 is unquantized. It may split on any even
  boundary. It is not an EXL3 cut.
- Expert parallel, 256 routed experts per rank, is the VRAM split
  for 16 GB cards. Partials sum in rank order, rank 0 then rank 1,
  matching `tp.zig`.
- Two-rank parallel still rejects structured output, as main does.

KV cache (`v0.6.6` `kvcache.py`):

- `bf16`, `int8`, `int4`.
- Groups of 32. A 32-point Hadamard (`1/sqrt(32)`), fp16 absmax
  scale, midpoint grid. 8-bit stores `q - 128` as int8.
  4-bit packs two codes per byte, low nibble first.
  Dequant stays in the rotated domain:
  int8 `(code + 0.5) * scale / 128`,
  int4 `(q - 7.5) * scale / 8`.
- Byte count includes both K and V and their scales.
  `row_bytes` in that file is the per-layer formula.
- Prefix state stops one token before the prompt end, so the same
  prompt resumes. That rule already matches
  `docs/recipes/qwen3.8-flash-next.md`. Hook it. Do not fork the radix.

Graphs:

- One captured topology per window length (verify and each draft).
- Mutable replay inputs: token ids, expert ids, expert slot
  pointers, PLE staging rows, KV write index, cache-hit mask.
- Weights of dense, shared, and currently resident experts stay
  fixed for the life of that executable graph. A swap uploads new
  pointers with `Exec.setKernel` / `Exec.update`, then the next
  replay sees them. Do not recapture because ids changed.
- MTP drafts must not change the accepted token. The host-side
  stand-in is: the same ids and the same staging bytes produce the
  same planned kernel arguments eager or graphed.

## Steps

Each step names the file that owns it. Later steps do not start
until the earlier acceptance check is true. Status is what this
branch has done.

### 0. Branch

Done. Worktree above, base SHA above, upstream tracking unset.

### 1. Plan and shape

Files: this file, `zig/src/families/flashnext/exl3_shape.zig`.
Check: `zig test zig/src/families/flashnext/exl3_shape.zig`.

### 2. N-gram row codec

File: `zig/src/families/flashnext/exl3_ngram.zig`.
Import `exl3_format.zig` for `codebookValue(.mul1, state)`.

Bit order, from `_ple_rows` (v0.6.6 `exl3_mm.py`):

- Word 0 is the fp16 scale.
- Payload bits start at word 1. Value `i` in `0..159`, bit `m` in
  `0..15` (bit 0 is the least significant bit of the 16-bit state):
  `pos = i - (m / KB)`, wrap by adding 160 when negative,
  `sb = pos * KB + (m % KB)`,
  word index `1 + (sb >> 4)`, bit `sb & 15` of that little-endian
  int16, least-significant bit first.
- `KB` is 2..8. `1 + 160 * KB / 16` must equal the row width.
  5-bit rows are 51 words.
- Output element `i` is one round to fp16 of
  `f32(codebookValue) * f32(scale) + f32(bias[i])`.
  The codebook value is rounded once before the scale multiply.

Check: synthetic rows at 2, 3, 5 and 8 bits; a wrapped value-0
window; scale and bias rounding; a row that is not one scale plus
160 values is refused.

### 3. PLE placement

File: `zig/src/families/flashnext/exl3_ple.zig`.

Two modes, one gather:

- `page_cache`: the caller holds a mapped view. Lookup touches only
  the requested rows. A startup prefetch is a separate call.
- `ssd`: the caller preads exactly the requested spans. Adjacent
  rows in one shard coalesce into one read. No whole-shard read.

Shard table: file id, byte offset of row 0, row count, shared
words-per-row. Global row id maps through the prefix sums, as
`NgramTable.gather` in v0.6.6 `exl3_pack.py`.

Check: two synthetic shards, ids that cross a shard boundary,
coalesced SSD spans, a row past the end is an error, the gathered
int16 bytes equal the source.

### 4. Header scan and centred norms

File: `zig/src/families/flashnext/exl3_scan.zig`.

No weight bytes. A group is the v0.6.6 `parse_group` rules:
int16 trellis `[K/16, N/16, 16*bits]`, one of suh/su and svh/sv,
optional bias, at most one of mcg/mul1 (absent means inst3),
half-bit widths only with mul1, K and N multiples of 128.
Width is `trellis_words / 8` halves, the same `Width` as
`exl3_format.zig`.

A rank-2 int16 tensor whose name contains `ngram_embedding` and
ends in `.trellis` is an n-gram shard, not an EXL3 group.
`mtp_hyper_connection_mixer_patch.safetensors` is the MTP mixer
patch filename.

`admit` reads codebook, average bits, optional head_bits and
mtp_bits. Codebook is inst3, mcg, or mul1. Average bits in 1..8.
head_bits and mtp_bits, when set, are integers in 1..8.

`centredOffset(means)` is the v0.6.6 test:
around zero when at most 10% of means exceed 0.5 and the median
is in [-0.5, 0.25], return 1.
Around one when at least 90% exceed 0.5 and the median is in
[0.75, 1.5], return 0.
Otherwise refuse. Empty input returns 1.

Check: a legal mul1 group, a half-bit group without mul1 refused,
both markers refused, an n-gram shard not parsed as tiles, the
3.05 config admitted, a centred and an uncentred mean vector.

### 5. tp=2 cuts

File: `zig/src/families/flashnext/exl3_tp.zig`.

`Axis` is `replicate`, `split_k`, `split_n`, `expert_parallel`.
`legal(k, n, ranks)` is true for an axis only when that side
divides by `128 * ranks`.

The 3.05 role table:

| Role | K | N | Axis |
| --- | --- | --- | --- |
| gate, up | 2560 | 640 | split_k |
| down | 640 | 2560 | split_n |
| routed experts, as a set | — | — | expert_parallel, 256 each |
| shared expert | same shapes | same | replicate (one copy per rank) |
| q, o, head | from `exl3_shape` | | the legal side |
| k, v | 2560 | 512 | split_n (512/2 = 256) |
| HC low-rank linears | unquantized | | even split, not EXL3 |

Rank `r` of a split owns `[r * part, (r+1) * part)` along that
axis, tiles included. Expert parallel: rank 0 owns ids 0..255,
rank 1 owns 256..511. Reductions add rank 0's partial, then
rank 1's. No other order.

Check: gate refuses split_n, down refuses split_k, expert ranges
are 256 and disjoint, q split_n parts are 128-aligned, a 320-wide
EXL3 N refuses both splits at 2 ranks.

### 6. Expert cache policy

File: `zig/src/families/flashnext/exl3_offload.zig`.

Types: `Slot` (expert id or empty), `Decision` (`hit`, `copy`, `cpu`),
`Policy` (slot count, expert bytes, copy bytes per round, swap every
4, swap cap 96).

`planRound`:

- Shared id 512 is `hit` and is not stored in a routed slot.
- Ids in the slot table are `hit`.
- Unique misses sort by count descending, id ascending.
- The first `copy_bytes / expert_bytes` misses are `copy`.
- The rest are `cpu`.
- Copy does not install the expert into the slot table.

`swap`, called when `round % 4 == 0` and round > 0:

- Candidates are requested ids absent from the cache, same sort.
- Take at most 96.
- Evict the lowest frequency residents (tie: higher id).
- Seed: `installProfile` fills slots from a histogram before round 0.

Check: an all-hit round copies nothing; a miss list larger than
the copy budget splits; a copy does not change slots; round 2
does not swap; round 4 swaps at most 96; shared id never occupies
a slot; eviction prefers the cold id.

### 7. Graph replay inputs and KV / prefix

Files: `zig/src/families/flashnext/exl3_graph.zig`,
`zig/src/families/flashnext/exl3_cache.zig`.

Graph: a `Capture` lists buffer bindings as `fixed` or `replay`.
Expert ids, slot pointers, PLE staging, the hit mask, and the KV
index are `replay`. Dense weights are `fixed`.
`sameTopology` is true when two captures have the same binding
order and the same fixed addresses. Changing a replay address
keeps the topology. Changing a fixed address does not.
`argumentsEqual` is the eager-versus-graph stand-in.

Cache:

- `KvDtype` bf16, int8, int4.
- `layerBytes(tokens, kv_heads, head_dim, dtype)` matches
  `row_bytes * tokens` from v0.6.6, times nothing extra.
  Head dim must divide by 32 for int8 and int4.
- `quantize` / `dequant` implement the Hadamard and the midpoint
  grid above, in fp32, scale stored as f16.
- `resumeAt(prompt_len)` is `prompt_len - 1` when `prompt_len > 0`.
- `fits` takes resident bytes (dense + shared + router + HC + MTP
  + graph workspace), the KV bytes, and a device budget. It
  refuses a plan that also adds the routed-expert total or the
  n-gram table size. Those stay on the host.

Check: int8 round-trip error is bounded by half a bin times the
scale after undoing the rotation in the test's own inverse;
int4 nibble pack; a 16 GB budget refuses resident experts and
refuses a context whose KV alone exceeds the budget; resumeAt(8)
is 7; two captures with different expert-id pointers and the same
fixed weights compare equal.

### 8. CUDA EXL3 linear and grouped experts

Host half is in `exl3_strips.zig`: stored `[K/16, N/16, words]` becomes
`[N/128, K/16, 8, words]`, the v0.6.6 `strips` permute. `zig test` covers it.
`exl3_linear.cu` and `exl3_experts.cu` are written. The expert kernel
is an identity projection; SwiGLU is `exl3_cpu.swiglu` until a device
epilogue exists. This Mac has no nvcc. The sm_120 check already
passed on par1-cs13. The files:

- `zig/kernels/cuda/exl3_linear.cu` and `exl3_experts.cu`.
- Behavior from v0.6.6 `linear.cu` and `experts.cu`: row-invariant
  for 1..128 rows, widths 1..8 mixed per tensor, grouped experts
  read ids from a pointer.
- Oracle is `exl3_format.zig`, not a fresh formula.
- Launch through `zig/src/cuda/graph.zig` so a window captures once.
- Strips layout: a 128-column block's tiles in K order.
- No cuBLAS on the unquantized fp16 path.

Acceptance: one tile and one 128-row projection match the Zig
decoder on CPU-uploaded outputs. A graph replay with new ids
matches eager for those ids.

### 9. CPU miss GEMV

`exl3_cpu.zig` `gemv` matches `dequantize` then a naive matvec on a 128×128
mul1 tile (`zig test`, included in the index). An AVX2 kernel is still
unwritten, and it lands only if it matches this scalar path. par1-cs13
has no AVX-512, so that kernel must not require it.

### 10. Loader

`exl3_place.zig` names the home of a tensor (ngram host, routed host, shared
resident, centred norm, MTP patch, vision skip) from the header string alone.
Opening a safetensors shard is still next, and only on a machine that has the pack.

Walk safetensors headers with step 4. Place tensors:

- N-gram shards: step 3, mode from the flag.
- Routed experts: host banks, step 6, profile seed if a file is given.
- Shared, dense, router, HC, MTP head: resident.
- Centred norms: add the step-4 offset in fp32.
- MTP mixer patch file, if present.
- Draft head: the v0.6.6 requant of selected rows to 4-bit groups
  of 32. Verification still uses the full EXL3 head.

Do not download the ~85 GB pack until a machine with the disk and
the RAM is named. Header-only tests stay synthetic.

### 11. Engine wire-up

Native refusal sites, read on this commit:

- `zig/src/server/cli.zig` `flags`: `--ple-on-ssd`, `--ssd-experts`,
  and `--kv-dtype` have `native` left false. `parse` then returns
  `error.Usage` with the v0.6.6 install line (`cli.zig` around the
  `if (!flag.native)` check). `--tp` is native only where
  `served("--tp")` says so, which is not an EXL3 pack.
- `zig/src/families/flashnext/config.zig` `parse` returns
  `error.UnsupportedAffineMode` unless `quant_method` is `mlx` or
  `affine`. An `exl3` config dies there today.
- `zig/src/native/none.zig` and `metal.zig` tell the operator the
  Python 0.6.6 engine may serve a model type this binary has no
  backend for.

When `quantization_config.quant_method` is `exl3`, Flash Next uses
this path instead of the affine MLX loader. Flags to accept on the
native binary, which refuses them today:

- `--ple-on-ssd`
- `--kv-dtype bf16|int8|int4`
- tp=2 on an EXL3 pack
- an expert-cache size in bytes, defaulting from `fits`

Capture graphs per step 7. Swap uses pointer update, not recapture.
Rank order follows step 5. Vision tensors are ignored in this milestone.

### 12. Parity checks, then docs

When the pack is on a machine that can load one layer:

- Teacher-forced top-1 against the Zig decoder on that layer.
- Draft tokens equal serial tokens, eager and graphed.
- tp=2 tokens equal tp=1 tokens.
- Offload tokens equal resident tokens on a layer that fits both ways.
- SSD PLE rows equal the mmap rows.

Only after those checks, update
`docs/recipes/qwen3.8-flash-next.md` and `docs/recipes/exl3.md`
so they describe this branch's native path. Until then the recipes
keep saying the native engine does not serve EXL3.

Do not publish a tok/s figure without a measured run on a named machine.

## File owners

| File | Step |
| --- | --- |
| `docs/plans/exl3-flashnext-offload.md` | 1 |
| `zig/src/families/flashnext/exl3_shape.zig` | 1 |
| `zig/src/families/flashnext/exl3_ngram.zig` | 2 |
| `zig/src/families/flashnext/exl3_ple.zig` | 3 |
| `zig/src/families/flashnext/exl3_scan.zig` | 4 |
| `zig/src/families/flashnext/exl3_tp.zig` | 5 |
| `zig/src/families/flashnext/exl3_offload.zig` | 6 |
| `zig/src/families/flashnext/exl3_graph.zig` | 7 |
| `zig/src/families/flashnext/exl3_cache.zig` | 7 |
| `zig/src/families/flashnext/exl3_strips.zig` | 8, host side of the strips permutation |
| `zig/src/families/flashnext/exl3_cpu.zig` | 9 |
| `zig/src/families/flashnext/exl3_place.zig` | 10 |
| `zig/src/families/flashnext/exl3_load.zig` | 10, synthetic safetensors header |
| `zig/src/families/flashnext/exl3_admit.zig` | 11, config admission only |
| `zig/kernels/cuda/exl3_decode.cuh` | 8, device decode |
| `zig/kernels/cuda/exl3_linear.cu` | 8 |
| `zig/kernels/cuda/exl3_experts.cu` | 8, identity epilogue |
| `zig/kernels/cuda/exl3_check.cu` | 8, numerical check, not a fatbin |
| `zig/src/cuda/exl3.zig` | 8, host launch wrapper |
| `zig/src/families/flashnext/exl3.zig` | index |

Steps 2-10 and the host half of 11 pass `zig test zig/src/families/flashnext/exl3.zig`
on this worktree (Zig 0.17.0, 77 tests). The index is
`zig/src/families/flashnext/exl3.zig`, also `zig build test-exl3-flashnext`.
`exl3_format.zig` next to those files is a symlink to `zig/src/core/exl3_format.zig`,
and `safetensors.zig` is a symlink to `zig/src/core/safetensors.zig`,
because `zig test` rejects a `../` import. Array repeats in Zig 0.17 are `@splat`,
not `**`.

The CUDA kernels compile for sm_120 and the check in `exl3_check.cu` matched the
Zig decoder (max abs diff one fp16 ulp, 0.00390625) on par1-cs13. They are in the
fatbin table as `exl3_linear` and `exl3_experts`. The server still does not serve
an EXL3 checkpoint: `--ple-on-ssd`, `--ssd-experts`, `--kv-dtype`, and `--tp` stay
refused until a family open() runs these kernels. Step 12 waits on a loaded pack.

## Out of scope

- The DeepSeek-V4 SGLang server on par1-cs13. Do not relaunch it.
- Vision encoding of this pack.
- GLM-5.3 and the `rocm-pr` branch.
- A second prompt-cache implementation.
- Pushing to `ashhart/TensorFold`.
- Committing, unless the operator asks.
