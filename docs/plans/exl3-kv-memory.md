# KV memory for 200k contexts on two 16 GiB cards (K10)

Status: decision proposal, 2026-10-19. Branch `exl3-c-kv` (from `exl3-c-main` at 0d8b060e). Task: make 200k-token context
at 2 concurrent sequences (4 with heavy cache reuse) fit on the two RTX 5060 Ti with a useful expert pool, on upstream's
attention path.

Every number is tagged: **measured** (par1-cs13, one idle 5060 Ti under `~/bin/gpurun`, sm_120, nvcc 13.4, 2026-10-19),
**read** (from the code), or *estimate* (arithmetic from measured or read parts, not a run).

## Decision

Recommended combination, in this order:

1. **int8 KV, dequantised on load** (our kernels `flashnext_exl3_kv.cu`, one fp16 scale a 32-group; the 32-point
   Hadamard is an option, off by default). Halves K/V: 4.58 → 2.43 GiB a 200k stream. Quality is at the noise floor
   of the run: through all 48 layers it matches a 16× finer grid (int12) on 8,592 scored rows of two prompts (below).
2. **The indexer's raw keys in mapped pinned host memory.** They are read only when a block of 4 completes: 0.57 GiB a
   stream off the card for 2-5 µs a layer. No kernel change (upstream's kernels take the pointer).
3. **Scratch: chunk 1024 and no `head_local` when the head is whole.** 1.368 → 0.686 GiB a card for +3% prompt time.
4. **Attention head split across the two ranks** (one KV head and 12 query heads a rank, o_proj K-split, one fp32
   all-reduce a layer). Halves K/V again (1.22 GiB a stream). Upstream's attention kernels already take the head counts
   as arguments, so this needs no `.cu` change: new launchers and a loader slice in our files.
5. **Prompt-cache snapshots in host memory** (cuda_snap.zig: a mapped pinned buffer instead of a device one). That is
   how "4 sequences with heavy cache reuse" fits: two streams live on the cards, parked states live in the 251 GB of
   RAM and come back at 14 GB/s.

What this gives on rank 0 (the tighter card: it also holds the MTP head and every stream's draft cache), at 200,000
tokens, drafts on (*estimate* from the table below; formulas read, card total and kernel times measured):

| | streams | device bytes for streams | left for the expert pool | pool lines a layer |
|---|---:|---:|---:|---:|
| upstream today (bf16 KV, chunk 1536) | 1 / 2 | 5.98 / 11.97 GiB | 2.12 / does not fit | 15 / - |
| 1+2+3 (no upstream kernel or glue change) | 1 / 2 | 3.03 / 6.06 GiB | 5.76 / 2.72 GiB | 42 / 20 |
| 1+2+3+4 | 1 / 2 / 4 | 1.81 / 3.63 / 7.26 GiB | 6.97 / 5.16 / 1.53 GiB | 51 / 37 / 11 |

Not recommended as the main lever: **K/V in host memory read in place** (lever c). It works, and it is the only way to
hold many 200k streams at once, but it costs 2.5 ms a token a stream at decode, 10 ms a 4-row MTP verify window, and
50-155 ms a 256-row prompt block a layer (measured below), and it shares PCIe with the expert misses. Use host memory
for parked states (5) and for the index keys (2), not for the K/V of a stream that is decoding. **int4 KV** is rejected
on quality (below).

## 1. Upstream's attention and KV path (read)

- Per attention layer and stream (`cuda_prompt.zig` `seqBytes`, `cuda_batch_state.zig` `SeqState`): `kc`, `vc`
  `[cap][2][256]` bf16 (2,048 B a token), `ikc` raw indexer keys `[cap][128]` bf16 (256 B), `pooled` `[cap/4][128]`
  bf16 (64 B a token). 12 attention layers. Plus 36 GDN conv tails and states (0.108 GiB) and the PLE tail. At
  200,016 tokens (context + `mtp.max_rows`): **5.401 GiB** = K/V 4.578 + ikc 0.572 + pooled 0.143 + GDN 0.108.
- The sparse indexer (`fn_qsa_pool`, `fn_qsa_scores` in `flashnext_attn.cu`, `fn_qsa_select` in `flashnext_batch.cu`):
  a row is dense until it has more than 512 complete blocks of 4 (up to 2,051 keys); then it lists its 512 best blocks
  in block order plus a tail of 0-3 keys. `fn_attn_chunks` reads keys through that list (`IDS`, `NKR`, `SPR`), 512 keys
  a chunk, 5 chunks at most. So **a row never reads more than 2,051 keys**, whatever the context.
- Scores read every pooled block of the stream for every row (12.8 MB a layer at 200k): the pooled rows must stay on the
  device. `ikc` is read only by the pooling, 4 rows when a block completes.
- **Head counts are already kernel arguments**: `fn_attn_prep(..., PW, NQ, NKV, NI, IHD, HALF)`,
  `fn_attn_chunks(..., H, HK, G, ...)`, `fn_attn_merge(..., H, HK, G, ...)`, `fn_attn_gate(..., PW, NQ)`. Only the Zig
  launchers (`cuda_glue.zig`) pass the model's constants. The plan's step 12 assumed kernel edits; there are none to do.
- Snapshots (`cuda_snap.zig`): a device buffer, `cuMemcpyDtoDAsync` of each piece (conv, state, kc, vc, ikc, pooled,
  tail, and the head's cache), `bytes(at)` = 110 MiB + 27 KiB a token.
- MTP draft cache (`cuda_mtp_draft.zig` `Draft.bytes`): the head's attention layer at the window (0.441 GiB at 200k)
  plus fixed 0.142 GiB (streams for a chunk, 16 rows of logits, the saved GDN states 0.105 GiB): **0.584 GiB a stream**
  (T10 reported 0.604 at a slightly larger window). Rank 0 only.
- Pass scratch (`Scratch.bytes`): 1.368 GiB at chunk 1536, linear in the chunk. Its largest terms at 1536:
  `head_local` 0.355 (cap × vocab bytes, read only by `projectHead` for a vocabulary-split head), `y` 0.161,
  `native_part` 0.117, `part` 0.117. Block scores 0.05 GiB at 200k (`scoreBytes`).

## 2. Lever (a): int8 / int4 KV

### What was built

`zig/kernels/cuda/flashnext_exl3_kv.cu` (new file, upstream's kernels untouched):

- `fn_kv8_prep`: upstream's `fn_attn_prep` body; K and V are written as int8 codes `[pos][HK][256]` plus fp16 scales
  `[pos][HK][8]` (code `clamp(floor(x/s·128)+128, 0, 255) − 128`, value `(c+0.5)·s/128`, `s` = fp16(group max), the
  policy of our old `exl3_cache.zig`). With `ROT` the 32-groups go through a normalized Sylvester Hadamard (warp
  shuffles) and the queries are stored rotated (dots are unchanged by an orthogonal map).
- `fn_kv8_chunks` / `_ahead`: upstream's `attn_chunks` with each tile read as codes and scales (cp.async into a raw
  buffer, 1 or 2 raw buffers in flight), dequantised into the same bf16 smem tile pair, then upstream's mma and online
  softmax unchanged. 61 KiB / 78 KiB of dynamic smem.
- `fn_kv8_merge`: upstream's merge, then the inverse rotation in fp32 before the one bf16 rounding.
- `fn_kv8_pack` (bf16 rows to codes), `fn_kv_gather` (the explicit staging alternative), `fn_kv_roundtrip` (test
  support: a bf16 cache's rows replaced in place by what a 2-15 bit cache would hold).

`zig/src/families/flashnext/cuda_exl3_kv.zig`: the launchers (`Kv8`), the byte arithmetic of every lever
(`seqBytes(cap, Policy)`, host-tested equal to upstream's `seqBytes` for the default policy), and `attnBlock`: the
pass's attention block over an int8 cache (upstream's projection, pooling, scores, `fn_qsa_select`, gate and o_proj;
our prep, chunks and merge). It is not hooked into `cuda_batch.zig` yet (one branch at the top of `attnBlock`, §8).

### Kernel correctness (measured, `tf-cuda-test exl3-kv kernel`)

- `fn_kv8_pack` codes and scales equal the host's bit for bit (8,192 rows, plain and rotated).
- `fn_kv8_prep`: queries (plain), index queries and index keys bit-equal to upstream's prep; K/V codes and scales equal
  the host's packing of upstream's bf16 cache rows (plain and rotated). Rotated queries rotated back: within 5.7e-3 of
  the group's peak of upstream's (one bf16 rounding on each side).
- Attention, plain int8: **bit-identical to upstream's `fn_attn_chunks`/`fn_attn_merge` run on a bf16 cache holding
  the dequantised values**, dense (1,100 keys) and sparse (2,048 + tail), read-ahead (16 rows) and one-tile (20 rows)
  kernels. So the new kernel adds no arithmetic error of its own; the error is the cache's.

### Kernel cost (measured, `tf-cuda-test exl3-kv bench`, one attention layer, cache of 200,016 tokens on the device)

| case | bf16 (upstream) ms | int8 (ours) ms | int8 with 2 raw buffers |
|---|---:|---:|---:|
| decode, 1 row, 2,051 keys | 0.0310 | 0.0421 | same kernel |
| MTP verify window, 4 rows | 0.0317 | 0.0425 | same kernel |
| prompt, 256 sparse rows (selection overlap 0.95 / 0.80 / 0) | 1.15 / 1.20 / 1.75 | 2.57 / 2.63 / 2.80 | 2.41 / 2.43 / 2.52 |
| prompt, 256 dense rows (keys 1,795-2,050) | 0.91 | 2.18 | 2.06 |

Decode: +0.011 ms a layer, +0.13 ms a token (*estimate*, 12 layers). Prompt: about 2.1× upstream's attention, i.e.
+1.3 ms a 256-row block a layer; for a 200k prompt *estimate* 781 blocks × 12 × 1.3 ms ≈ 12 s on top of ≈ 5 min
(the dequantise loop is not overlapped with the mma yet; a register-prefetch version is the obvious next step).

### Quality

Three levels, all on real weights and real prompts.

**(i) The attention layer alone** (measured, `tf-cuda-test exl3-kv real DIR 4|8 32 256 kv_long.ids`): layers 0..3
and 0..7 through the family's own pass (`Pass.chunk`: embedding, PLE, GDN, attention with the sparse indexer, EXL3 MoE,
every routed expert resident) over a 6,543-token prompt (`doc.txt`+`long.txt`+`prefix_long_a.txt`); the last 32 tokens
fed one row at a time; at those rows the attention output from the pass's own Q, selection and K/V, against float64
of the bf16 cache (max | rms, of the peak):

| | layer 3 | layer 7 |
|---|---|---|
| upstream bf16 (the pass's output; re-run bit-equal 32/32) | 3.37e-3 \| 1.67e-3 | 2.69e-3 \| 1.69e-3 |
| int8 plain, our kernel | 8.33e-3 \| 4.77e-3 | 1.17e-2 \| 4.40e-3 |
| int8 rotated, our kernel | 5.64e-3 \| 3.28e-3 | 7.97e-3 \| 3.53e-3 |
| int8 rotated, the cache alone (float64) | 3.26e-3 \| 2.63e-3 | 5.30e-3 \| 2.89e-3 |
| int4 plain / rotated, the cache alone | 1.02e-1 / 5.30e-2 (max) | 8.63e-2 / 7.19e-2 (max) |

int8 roughly doubles the attention output's rms error against float64; int4 is 15-25× worse than bf16.

**(ii) The whole model, teacher-forced** (measured, `tf-cuda-test exl3-kv full`): all 48 layers through the family's
pass code, one layer resident at a time (every routed expert of it, then freed), then the mixer read-out and the EXL3
lm_head. This is a test harness (`exl3_kv_full.zig`), not the served engine; it exists because the single-card path
cannot hold all experts. **Its bf16 run reproduces the Python reference golden** (`exl3_ref_golden.json`, 16-token
prompt): teacher-forced top-1 equal on 16/16 positions (top-5 sets differ only at bf16 logit ties).
Variants: `int8`/`int8r`/`int4r`/`int12`/... emulate the cache (after each attention block the rows it wrote are
replaced by the quantised values, so every later row and layer reads them through upstream's kernels; the rows of the
same chunk see their own keys unquantised); `kv8`/`kv8r` run the real int8 cache through `cuda_exl3_kv.attnBlock`.
`int12` (a grid 16× finer than int8) measures how sensitive the run is to any perturbation.

| prompt, rows scored | variant | top-1 = next token | top-1 = bf16's | flips, bf16 margin > 1 / > 4 | KL(bf16‖var) mean |
|---|---|---:|---:|---:|---:|
| qbench (22 conversations, 14,496 tokens), rows 10,400-14,495 | bf16 | 3243/4095 | - | - | - |
| | int12 | 3255 | 3978/4096 | 23 / 4 | 0.0220 |
| | int8 | 3246 | 3989 | 25 / 6 | 0.0201 |
| | int8 rotated | 3243 | 4004 | 13 / 3 | 0.0156 |
| | K bf16, V int8 | 3248 | 3988 | 22 / 2 | 0.0212 |
| needle log ×3 (6,548 tokens, repetitive), rows 2,052-6,547 | bf16 | 3765/4495 | - | - | - |
| | int12 | 3761 | 4009/4496 | 308 / 135 | 0.314 |
| | int10 | 3705 | 4015 | 310 / 139 | 0.329 |
| | int8 | 3759 | 3943 | 386 / 181 | 0.361 |
| | int8 rotated | 3733 | 4023 | 289 / 129 | 0.339 |
| | int6 | 3698 | 3942 | 347 / 154 | 0.407 |
| | K int8, V int4 rotated | 3732 | 3931 | 365 / 171 | 0.406 |
| 16-token golden | int8, int8 rotated (emulated); kv8, kv8r (real) | 10/15 each (= bf16) | 16/16 | 0 | 0.002-0.011 |
| | int4 rotated | | 15/16 | | 0.007 |
| needle 2,323 tokens, last 64 rows | int8 / int8 rotated / int4 rotated | | 63 / 63 / 61 of 64 | | 0.009 / 0.019 / 0.033 |

The real int8 cache (`kv8`: our kernels in the pass through all 48 layers; every key quantised as it is written, the
chunk's own included), each beside `int12` in the same run:

| prompt, rows scored | variant | top-1 = next token | top-1 = bf16's | flips, bf16 margin > 1 / > 4 | KL mean |
|---|---|---:|---:|---:|---:|
| qbench, rows 10,400-14,495 | int12 (emulated) | 3255/4095 | 3978/4096 | 23 / 4 | 0.0220 |
| | **kv8 (plain)** | 3252 | 3986 | 20 / 2 | 0.0212 |
| | kv8r (rotated) | 3262 | 3996 | 22 / 2 | 0.0168 |
| needle log ×3, rows 2,052-6,547 | int12 (emulated) | 3761/4495 | 4009/4496 | 308 / 135 | 0.314 |
| | **kv8 (plain)** | 3743 | 4030 | 311 / 139 | 0.316 |
| | kv8r (rotated) | 3691 | 3950 | 331 / 131 | 0.350 |

Reading: on the repetitive needle log, *any* perturbation (int12 included) moves about 11% of the top-1s: the
continuation is ambiguous between near-identical earlier copies, and the run is chaotic there. On it int8 sits next to
int12 and int10; int6 and int4 V are worse. On the natural qbench conversations every 8-bit variant agrees with bf16 on
97.4-97.8% of rows, at int12's level, and next-token accuracy does not move (3243-3255 of 4095). The needle's answer
tokens (" 7391") are predicted by every variant (the real int8 runs included). **int8 KV is at the noise floor of a
16× finer grid; int4 is not.** Plain int8 matches int12 on both prompts; rotated int8 is better on qbench and on the
layer error but worse than int12 on the needle log (KL 0.350 vs 0.314, 52 fewer correct). Default: **plain** (no
shuffles either); rotation stays a kernel argument (`rot`) to re-evaluate on longer natural text.

**Not measured**: a 200k-token prompt through the whole model (one 6.5k run of the streamed harness takes ~6 min a
card); greedy generation (the harness is teacher-forced only).

## 3. Lever (b): attention head split across the two ranks

Design (our old verified cut, `git show exl3-flashnext-window-wip:zig/src/families/flashnext/exl3_tp_split.zig`): rank r
keeps query heads 12r..12r+11 and KV head r; q_proj (interleaved [query 256 | gate 256] a head) keeps 6,144 of 12,288
columns, k_proj/v_proj 256 of 512, all whole 128-column Hadamard blocks, so the N-split is exact per column; the indexer
projection and norms are replicated (each rank selects the same blocks from the same streams); o_proj keeps rows
3072r..3072r+3071 (24 Hadamard blocks), so each rank's output is a partial sum; one fp32 all-reduce of
`[rows, 2560]` a layer adds them. KV, `ikc`, `pooled` and selection per rank: K/V halved, the indexer replicated.

Diff estimate (read): **no `.cu` change** (head counts are arguments, §1). New code, ours: loader slices of q/k/v/o
(port `exl3_tp_split.slice`, ~120 lines), launchers with the rank's head counts (~60, as `Kv8` already does for the
int8 kernels: `nq`, `nkv`, `pw` are parameters), the attention block variant (~80, `cuda_exl3_kv.attnBlock` takes the
same shape), the o_proj reduce into `s.part` slice 0 (~20). One-line hooks: `cuda_batch.zig` attnBlock (the same branch
as int8), `cuda_exl3_load.zig` (slice at load), `cuda_prompt.zig` newSeq sizes, `cuda_snap.zig` row bytes. Conflict
risk: low; these files are where the int8 hook goes anyway. Not built in this task.

Cost (*estimate* from T10's measured SHM all-reduce: 0.19 ms at 1 row, 1.8 ms at 512 rows of `[rows, 2560]` fp32):
decode 12 × 0.02-0.05 ms ≈ 0.3-0.5 ms a token (summing `[1, 2560]` is 10 KB: nccl-lat measured 0.0199 ms); prompt
≈ 4 ms a layer per 1,024-row chunk, ≈ 48 ms a chunk, about +3% of a ~1.5 s chunk. Numerics: the K-split o_proj sum is
not bit-identical to tp=1 (fp32 order), within fp32 rounding.

## 4. Lever (c): K/V in pinned host memory

Measured (`exl3-kv bench`), one layer, cache of 200,016 tokens:

| | ms | effective |
|---|---:|---|
| decode row, int8 cache in mapped host memory, read in place by `fn_kv8_chunks` | 0.204 | 10.9 GB/s of 2.23 MB |
| decode row, bf16 cache in mapped host memory through upstream's `fn_attn_chunks` | 5.29-5.37 | 0.8 GB/s (upstream's cp.async pattern on sysmem; not investigated) |
| 4-row verify window, int8 in host memory | 0.84-0.86 | each row re-reads its keys (no L2 reuse of sysmem) |
| 256-row prompt block in host memory: int8 / bf16 | 50-155 / 345-389 | |
| staging alternatives for one row (4.2 MB bf16): one pinned H2D | 0.303 | 13.9 GB/s |
| 1,024 per-block `cuMemcpyHtoDAsync` | 2.39-2.50 | call overhead |
| `fn_kv_gather` from mapped host memory into device staging | 0.335 | 12.5 GB/s |
| selected ids D2H + stream synchronize | 0.006 | |

Design consequence: reading the cache in place (the kernel follows the selected ids into mapped memory) needs no
host sync, no ids round trip and stays graph-capturable; explicit staging only adds a launch and a copy, and per-block
copies are 8× slower. In place costs 12 × 0.204 ≈ **2.45 ms a token a stream** and **≈ 10 ms a 4-row verify window**
(*estimate* from the per-layer measurement), and the prompt cannot run from host memory at all (40-130× slower than
the device). It also competes for PCIe with expert misses (13.9 GB/s, both).

Selection overlap (measured, `exl3-kv real`, 6,543-token prompt): consecutive rows share 0.88 (layer 3) / 0.87
(layer 7) of their 512 blocks in the prompt, 0.84 / 0.80 for decode rows; the union of the last 16 rows covers 0.96 /
0.97 (0.92 for decode rows). So a device cache of the recent selections (a block → slot map a layer and stream, misses
read in place) would cut the PCIe bytes about 10×; that is a later step, and the overlap at 200k is not measured.

Index structures (measured): `ikc` in mapped host memory: pooling 1 row 0.0048 ms vs 0.0026 on the device, 256 rows
0.0079 vs 0.0028, pooled rows bit-equal. `pooled` (0.143 GiB a stream) stays on the device: the scores read all of it
every row.

Snapshots (measured): `cuMemcpyDtoDAsync` between the device and a mapped pinned buffer moves 256 MiB in 19.0 ms
(14.1 GB/s) out and 19.3 ms (13.9 GB/s) back. `cuda_snap.zig` already copies with `cuMemcpyDtoDAsync`, so a host
snapshot is an allocation change. A 200k int8 state (≈ 3.3 GiB with `ikc`) restores in ≈ 0.25 s (*estimate*), a
200k prompt takes ≈ 5 min (*estimate*: 48-layer prompt cost scaled from the 8-layer measurement below).

## 5. Lever (d): fixed costs

Prompt time by chunk (measured, `exl3-kv real ... 8 4 1536,1024,512,256,1536,1024,512`, layers 0..7, 6,539 tokens,
each size twice): 1536: 0.246-0.249 ms a token; 1024: 0.254-0.256 (+3%); 512: 0.285-0.293 (+17-19%); 256: 0.372
(+50%). Scratch (read): 1.368 / 0.922 / 0.477 / 0.255 GiB, and 0.355 / 0.236 / 0.118 / 0.059 of it is `head_local`,
unused unless the head is split by vocabulary. **Chunk 1024 without `head_local`: 0.686 GiB (−0.68) for +3%.**

Draft cache (read): 0.584 GiB a stream (0.441 of it the head's attention cache at 200k). Under int8 + host `ikc` it is
0.357. Allocating it only for streams that draft (and dropping it when a stream stops) saves that per idle stream;
the table below keeps drafts on for every stream (worst case) and shows drafts off.

## 6. Per-card budget

`python3 tools/exl3_kv_budget.py` (formulas read from upstream; card total 15.568 GiB measured by `cuMemGetInfo`;
weights from T10: rank 0 5.107 + MTP head 0.475, rank 1 5.245 GiB, read from the pack's headers; context + modules
0.40 GiB *estimate* (our test's modules alone: 0.134 measured); other fixed 0.115 GiB; a pool line in all 48 layers
0.136 GiB (T10); hit rates measured by E2 at tp=1, a rank's L lines taken as a tp=1 pool of 2L lines (*estimate*);
0.147 ms a miss (E2, measured)). 200,000 tokens of context, drafts on:

| lever | streams | rank 0 streams GiB | rank 0 left for pool GiB | lines a layer r0 / r1 | expert misses ms a token (*est.*) | host GiB |
|---|---:|---:|---:|---:|---:|---:|
| A upstream: bf16 KV, chunk 1536 | 1 | 5.98 | 2.12 | 15 / 22 | 15.9 | 0 |
| | 2 | 11.97 | −3.87 | does not fit | | |
| B int8 KV | 1 | 3.66 | 4.44 | 32 / 38 | 10.7 | 0 |
| | 2 | 7.32 | 0.78 | 5 / 14 | below floor | |
| C = B + index keys on host | 2 | 6.08 | 2.02 | 14 / 22 | 16.4 | 1.1 |
| D = C + chunk 1024, no head_local | 1 | 3.03 | 5.76 | 42 / 47 | 8.7 | 0.6 |
| | 2 | 6.06 | 2.72 | 20 / 27 | 14.0 | 1.1 |
| | 4 | 12.12 | −3.34 | does not fit | | |
| E = D + attention head split | 1 | 1.81 | 6.97 | 51 / 56 | 7.3 | 0.6 |
| | 2 | 3.63 | 5.16 | 37 / 45 | 9.6 | 1.1 |
| | 4 | 7.26 | 1.53 | 11 / 23 | 18.1 | 2.3 |
| F bf16 KV + head split + host index keys + D's scratch | 2 | 6.13 | 2.65 | 19 / 29 | 14.3 | 1.1 |
| G K/V in host memory (int8) + host index keys + D's scratch | 2 | 0.79 | 7.99 | 58 / 63 | 6.5 + 2.45 a stream for K/V reads; prompt not possible in place | 6.0 |
| | 4 | 1.58 | 7.20 | 52 / 59 | 7.2 + 2.45 a stream | 12.0 |

Drafts off (same script): D 2 streams 25 lines a layer, E 2 streams 43, E 4 streams 21. The pool floor is 10 lines a
layer (T10). Rank 1 has more room than rank 0 in every row (no MTP head, no draft caches).

"4 with heavy cache reuse": E holds 4 × 200k on the device but leaves rank 0 a pool of 11 lines a layer (just above the
floor). The better shape is two decoding streams on the device (E: 37 lines) and the other states as host snapshots
(lever 5), restored in a fraction of a second. Sharing one 200k prefix between streams without copying it needs paged
KV (upstream PR #637), not in 1.1.0.

## 7. Risks

1. **Quality was measured to 14.5k tokens, not 200k.** At 200k every row is sparse (as were the scored rows here) but
   the selection is over 50k blocks instead of 1-3.6k. The int8 error does not depend on the context length (a row
   reads ≤ 2,051 keys); the indexer's keys stay bf16.
2. The int8 prompt attention is 2.1× upstream's (+12 s a 200k prompt, *estimate*): the dequantise step is not
   overlapped. Decode is +0.13 ms a token.
3. The head split adds one all-reduce a layer in the pass (and into the captured graph) and is not bit-identical to
   tp=1; both are the same kind as T10's MoE sum.
4. Snapshots in host memory use `cuMemcpyDtoDAsync` on a mapped pointer (measured to work at 14 GB/s); upstream's
   `cuda_lanes.zig` budgets snapshot bytes against the device today (`snap_bytes`), which has to change with it.
5. The MTP head's draft cache keeps two KV heads on rank 0 under the head split (the head runs on rank 0 only).
6. The pool hit rates at tp=2 are an estimate from tp=1 measurements.
7. bf16 K/V read in place from host memory by upstream's kernel is 25× slower than int8 by ours for reasons not
   investigated (the cp.async pattern); irrelevant to the recommendation, but do not take upstream's kernel as a host
   KV reader.

## 8. Integration (not done here; the serve agent owns the pass wiring)

- `cuda_batch_state.zig` `SeqState`: `kv8: ?[]const exl3_kv.Kv8.Cache = null` (one line).
- `cuda_batch.zig` `attnBlock`: `if (self.seq.kv8) |c| return exl3_kv.attnBlock(self, &self.k8, lw, ai, c[ai], false);`
  (one line; `Kv8` loaded once with the pass kernels). Decode rounds over several lanes: the per-lane loop as upstream's,
  with each lane's cache (the prototype refuses lanes).
- `cuda_prompt.zig` newSeq / `seqBytes`: allocate codes + scales instead of `kc`/`vc` when `Mode.exl3` (one branch);
  `ikc` from `HostBuffer.allocMapped` (one branch).
- `cuda_snap.zig`: piece sizes from `exl3_kv` (two lines); buffer from host memory (one line).
- `cuda_mtp_draft.zig`: the head's cache the same way (two lines).
- `cuda_batch_state.zig` Scratch: skip `head_local` when the head is whole (one line); default chunk 1024 for EXL3.

## 9. bf16 at tp=2: built and measured (L1, branch `exl3-c-longctx`)

User decision (2026-10-19): **bf16 K/V is the default**; the int8 cache above stays in the tree behind a flag
(`--kv-cache fp8` or `TF_FLASHNEXT_KV=int8`, hooked per lane in `Pass.attnRows`), not validated past one 32k
single-card gate. What was built for 200k x 2 concurrent at tp=2 on bf16, all **measured** on par1-cs13 (two 5060 Ti):

- **Attention head split** (`cuda_exl3_heads.zig`, default at tp=2, `TF_FLASHNEXT_ATTN_SPLIT=0` keeps upstream's
  replicated attention): rank r loads q|k|v|indexer columns of its 12 query heads and KV head r (7,296-wide row, whole
  128-column EXL3 blocks) and o_proj input rows 3072r.. (whole 128-row blocks); the pass runs upstream's kernels with
  NQ 12, NKV 1 and one fp32 all-reduce of [rows, 2560] a layer. Sequences hold one KV head a rank.
- **Indexer raw keys in mapped pinned host memory** (`TF_FLASHNEXT_IKC_HOST=1`; on by default with int8 only).
- **Chunk 1024** for EXL3 (unless `TF_FLASHNEXT_CHUNK`), **`head_local` sized for logits passes only** (0 rows with
  a whole head, 64 with the vocabulary-split head at tp=2): scratch 0.700 GiB a card.
- **Rank 1's MTP windows without a head cache** (it never runs the head): 0.44 GiB less a drafting stream at 200k.
- **Prompt states parked in host memory** (`cuda_snap.on_host`, `Loaded.host_snaps`; `TF_FLASHNEXT_SNAP_HOST`, on by
  default for EXL3 past 32,768 tokens): the native host budgets the prompt cache in host memory.
- **Rank rooms agreed in streams** (`rankRoom`): rank 1's smaller streams no longer make the native host refuse
  `--parallel 2` by comparing rank 0's stream bytes with rank 1's byte room.
- **Deep (expert-major) prefill at tp=2** (separate commit; another agent owns tp=2 prefill): 16,384 tokens in
  15.8 s vs 57.5 s on the batched pass, same 32 greedy tokens after it.

A stream at a 200,016-token window, device bytes (startup line `flashnext: a stream at a ... window holds`):

| | K/V | index keys | pooled | DeltaNet | draft and rest | rank 0 | rank 1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| upstream (replicated bf16) | 4.578 | 0.572 | 0.143 | 0.108 | 0.594 | 5.995 | 5.995 |
| head split | 2.289 | 0.572 | 0.143 | 0.108 | 0.594 | 3.706 | 3.706 |
| + index keys on host (0.572 GiB host a stream a rank) | 2.289 | 0 | 0.143 | 0.108 | 0.594 / 0.153 | 3.134 | 2.692 |

Pool lines a layer at 200k x 2 (`TF_FLASHNEXT_EXL3_STREAMS=2 --parallel 2`, drafts on; the pool's floor is 10):

| configuration | rank 0 | rank 1 |
|---|---:|---:|
| replicated bf16 | PoolTooSmall | PoolTooSmall |
| head split | PoolTooSmall | PoolTooSmall |
| split + host index keys, no deep prefill, 1 GiB reserve | 13 | 20 (before the rank-1 window change) |
| + deep prefill 2048 rows (0.34 GiB), 1 GiB reserve | PoolTooSmall (about 9) | 26 |
| + deep prefill 2048 rows, `TENSORFOLD_MEMORY_RESERVE_GIB=0.75` | 12 | 29 |

Rank 0 binds: beside rank 1 it holds the MTP head (0.475 GiB of weights) and each drafting stream's head cache
(0.44 GiB at 200k, bf16, two KV heads) (*estimate*: `--no-drafts` frees 2 x 0.59 GiB of draft state, about 14
lines, and the head's 0.475 GiB, about 5 more).

The served configuration that fits 200k x 2 at tp=2 on bf16 (measured, startup lines): head split + host index keys
+ deep prefill 2048 rows + `TENSORFOLD_MEMORY_RESERVE_GIB=0.75`, `TF_FLASHNEXT_EXL3_STREAMS=2 --parallel 2
--context 200000`: rank 0 **12 lines a layer**, rank 1 **29**, "2 streams at once, 3.15 GiB each at a 200000-token
window", an 8 GiB prompt cache in host memory. With the default 1 GiB reserve rank 0 falls to about 9 lines, below the
pool's floor of 10. **The minimum extra lever** with drafts on is therefore one of: the 0.25 GiB reserve cut used here,
deep prefill rows 1024 instead of 2048 (0.17 GiB), chunk 512 (scratch 0.35 GiB less, prompt +17% on the batched
pass), or the MTP head's draft cache in host memory / one KV head (0.44 GiB a stream); paging K/V to host memory (the
paging agent) is the lever that changes the picture (2.29 GiB a stream a rank of K/V).

Validation (measured, par1-cs13, tp=2 unless said):

| check | result |
|---|---|
| needle at 195,938 / 195,957 prompt tokens (chat, real text, needle at 50% / 25% depth), 2 requests at once | both answers correct, drafts off and drafts on; drafted == undrafted token_sha256 on both |
| needle at 8,158 / 8,131 and 32,719 / 32,709 tokens, 2 at once (200k x 2 config, batched prefill) | all correct, drafted == undrafted |
| golden serve gate at 32k, split (default) and `TF_FLASHNEXT_ATTN_SPLIT=0` | France and Once-upon equal the golden, drafts == no drafts on all three; ids16 sha e947fa499c6e (the known near-tie, as on exl3-c-main) |
| golden serve gate, single card at 32k | France / Once-upon equal golden; ids16 071c1653e472 (as exl3-c-main) |
| greedy 32 tokens after 8,192 / 32,000 real ids, split vs replicated | different after 3-4 tokens at both lengths (split is not bit-identical: fp32 order of the o_proj sum); not shown to be a near-tie (no logprobs on this server) |
| deep prefill tp=2 vs batched tp=2, 16,384 ids | same 32 greedy tokens; at 4,096 ids they part after 3 words |
| prompt state parked in host memory, single card, 15,955-token chat prompt three times | 2nd and 3rd resume at 15,948 tokens, prefill 0.15 s vs 19.3 s, same reply sha |

At 196k x 2 the two prefills took 312 s and 563 s (interleaved, deep 2048 rows); decode of a stream while the other
prefills is 0.4-10 tok/s (the rounds share the cards with the other's deep chunks). The 196k drafts-on pass did not
reuse the drafts-off states: a state kept without the MTP head's rows cannot serve a drafting stream (`NoHeadState`,
upstream's rule), so both prompts were prefilled again.

Hook points for a KV block cache (file:line on `exl3-c-longctx`): prep writes K/V at `cuda_batch.zig:334`
(upstream path) and `cuda_exl3_heads.zig:215` (split path); attention reads K/V by key id after the selection at
`cuda_batch.zig:361-363` and `cuda_exl3_heads.zig:241-250` (`s.qids` holds each row's key ids between the two
launches); the per-layer K/V buffers are made in `cuda_exl3_kv.zig` `newSeq` (`:463`, with the mapped-host pattern
the index keys use at `:476`); snapshots copy them in `cuda_snap.zig` `move` (`:75`).

## How to reproduce

```bash
# par1-cs13, ~/tensorfold-c-kv = this branch; build outside the lock
zig build -Dnvcc=nvcc -Dsm=120 -j16
P=/data/models/Qwen3.8-Flash-Next-exl3-3.05bpw
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-kv kernel                         # ~10 s
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-kv bench                          # ~40 s
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-kv real $P 8 32 256,512,1536 ~/kv_long.ids    # ~1 min
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-kv full $P 8 256 bf16,kv8r,kv8 ~/kv_ids16.ids  # ~2 min (golden)
~/bin/gpurun 1 ./zig-out/bin/tf-cuda-test exl3-kv full $P 64 256 bf16,kv8r,kv8,int12 ~/kv_qbench.ids 10400  # ~8 min
python3 tools/exl3_kv_budget.py                                                  # the budget tables
```

Prompt files: `~/kv_long.ids` = `doc.txt,long.txt,prefix_long_a.txt` (6,543), `~/kv_needle_long.ids` =
`prefix_long_a,doc,long,needle_answer` (6,548), `~/kv_qbench.ids` = the 22 qbench `input_ids` of the pack (14,496),
`~/kv_ids16.ids` = the golden's 16 ids. Card time used by this task: about 55 minutes in total, no single run over 8
minutes (the `full` runs on 6.5k-14.5k tokens hold a card 5-7 minutes each).
