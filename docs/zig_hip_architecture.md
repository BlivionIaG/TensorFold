# Zig HIP port: architecture plan (draft)

Where the HIP backend goes next: support for more numeric formats (MLX affine today; then GPTQ/AWQ W4A16, W4A8, W8A8, MXFP4/MXFP8, plain fp16/bf16 and
EXL3) and more GPUs (RDNA2 today, RDNA3, RDNA3.5, RDNA4, maybe gfx900), with kernel choice in one place and switches
that a user, a test and a tensor-parallel group all see the same way. The code stays small: each new format or GPU
should add a small module, not another copy of the kernels.

## 0. Goal: the formats to support

| Format | Weights | Bits | Activations | Notes |
|---|---|---|---|---|
| **MLX affine** | int, scale and bias a group (32/64/128) | 2, 3, 4, 5, 6, 8 | fp16 / bf16 | today (2, 3, 4, 6, 8 checked; 5 to add) |
| **FP16 / BF16** | unquantized | 16 | fp16 / bf16 | identity decoder |
| **AWQ INT4** | int4, scale and literal zero a group (GPTQ's +1 zeros are the same decoder with a flag) | 4 | fp16 / bf16 | W4A16; the vLLM qgemm kernels as native RDNA2 entries |
| **FP8** | e4m3 (e5m2 if met), scale a tensor or a channel | 8 | fp16 / bf16; fp8 on RDNA4 | decode to f16/bf16 on RDNA2/3; native fp8 WMMA on RDNA4 |
| **MXFP4** | e2m1, e8m0 exponent a 32 block | 4 | fp16 / bf16 | OCP microscaling |
| **MXFP8** | e4m3, e8m0 exponent a 32 block | 8 | fp16 / bf16; fp8 on RDNA4 | OCP microscaling |
| **EXL3** | trellis-coded, 3INST and MUL1 codebooks, Hadamard-rotated, scale a channel | 2, 3, 4, 5, 6, 8 bpw | fp16 / bf16 | x and y take the input/output Hadamard rotations; the vLLM fork's `exl3_dot2_*` kernels as a reference |
| later: W4A8, W8A8 | int4 / int8 | 4 / 8 | int8 a token | an int8 ActEncoder and the sdot4 / wmma-iu8 Dots; not in the goal list, the design keeps room for them |

Every one of these runs on every supported GPU, through one design: the decoder of section 3.2 plus the shared
tiles, with native kernels only where they are measured faster.

## 1. What must hold (CONTRIBUTING.md, plus this backend's own rules)

- **Lanes** (CONTRIBUTING 1): every model decodes through the shared lane rounds, with drafts verified in the same
  forward. There is no serial path and no second batcher.
- **No precision traded for speed** (CONTRIBUTING 3): operands at the checkpoint's activation type (fp16 / bf16) or
  better, sums in fp32. This is what the CUDA path does (bf16 q, K, V and probabilities into fp32 MMA). Reorders are
  fine and are named in the PR. Anything below the activation type (int8 / fp8 activations, lower-precision sums) is
  an opt-in Policy mode, raised as an issue before it becomes a default.
- **Correct output.** Each format/GPU/mode is measured against an fp64 CPU forward of the same checkpoint
  (tools/truth): KL, top-1 and perplexity. Matching Python's bits is not the bar.
- **Row-exact arithmetic.** A row's bits do not depend on what it shares a launch with. drafted == serial,
  solo == together, resumed == fresh and tp runs consistent, greedy and sampled. In practice this means **one kernel
  family per op and path at any row count**. (Commit a6762a3 fixed the last violation: prefill kernels chosen by m.)
- **No speed regressions.** Prefill and decode tok/s for each model and GPU are recorded; a change that loses on any
  of them needs a reason.
- **Lean code** (CONTRIBUTING 6): one job to a module, files under about 600 lines, one-line comments that say what
  the code can't. No numbers or history in the source.
- **Every platform it touches** (CONTRIBUTING 5): HIP code lives in its own files. Shared files (server, lane core,
  engine API) change only as needed, and the PR says Metal and CUDA were not run.
- **Lean design.** A format adds a weight decoder, an activation precision adds an encoder, and a GPU instruction adds a Dot.
  Tiles and epilogues are written once, and a code object only holds the modes its GPU can run.

## 2. Today (hip-port a6762a3)

| Layer | Where | Problem |
|---|---|---|
| Runtime | `zig/src/hip/` driver, context, stream, arena, graph, rccl, link | fine as is |
| GPU identity | `rocm.Family { rdna2, rdna3 }`, build flag `TENSORFOLD_RDNA_WMMA` | two fixed families; kernels test the family, not what the GPU can do |
| Kernels | `zig/kernels/hip/{ops,prefill,decode,gdn_prefill,tp}.hip`, `rocm/affine_*.hpp/hip` | MLX unpacking is mixed into every tile; tiles exist per GPU |
| Launch | `launches.zig` (fixed struct of functions), `affine_launch.zig` (rules by m) | kernel choice is spread across ops.zig, launches.zig and affine_launch.zig |
| Weights | `weights.Affine` / `view.Affine` (MLX only) | no place for another format |
| Switches | about 13 env vars read where they are used (`TF_WMMA`, `TF_AFFINE_GEMM`, `TF_AFFINE_GEMV`, `TF_DECODE_FUSE`, `TF_FA_WIDE`, `TF_GDN_CHUNKED`, `TF_HIP_GRAPHS`, `TENSORFOLD_GRAPH`, `TF_HIP_LAUNCH`, `TF_HIP_GRAPHS_TP`, `TF_HIP_GRAPH_FAIL`, `TF_RCCL_LIB`, ...) | nothing records which were active, ranks can disagree, tests depend on the process env, a server can't report its config |

## 3. Target layers

```
 engine / lanes / server            (unchanged: windows, rounds, MTP, prefix, tp)
 model forward (families/qwen3_5)   calls ops.project / ops.attention / ...; knows nothing of formats or GPUs
 ── Policy ── resolved once at open: what the run may use
 Ops + Registry                     (op, format, path, shape) -> one kernel, chosen under Caps and Policy
 Quant formats                      mlx | gptq | awq | exl3 ...: load, slice, reference dequant, Decoder, native kernels
 Shared tiles                       stream (decode), gemm (prefill), routed, wmma: templates over Decoder x Caps
 Caps                               what this GPU has: wave, dot2 f16/bf16, sdot4/8, matrix cores, LDS
 Runtime                            (unchanged)
```

### 3.1 Caps: what a GPU can do, not its name

`hip/caps.zig`, built from the gfx id at open and compiled into each code object as macros (`TF_WAVE`, `TF_DOT2_F16`,
`TF_DOT2_BF16`, `TF_SDOT4`, `TF_SDOT8`, `TF_MATRIX`):

| GPU | wave | dot2 f16 | dot2 bf16 | sdot4/8 | matrix | activations |
|---|---|---|---|---|---|---|
| gfx900 (Vega 10) | 64 | no (packed fp16 FMA) | no | no | none | fp16 |
| gfx906 (Vega 20) | 64 | yes | no | yes | none | fp16 |
| gfx1030 (RDNA2) | 32 | yes | no | yes | none | fp16 |
| gfx1100 (RDNA3) | 32 | yes | yes | yes | wmma11 | bf16 (fp16 possible) |
| gfx1151 (RDNA3.5) | 32 | yes | yes | yes | wmma11 | bf16 |
| gfx1200/1201 (RDNA4) | 32 | yes | yes | yes | wmma12 | bf16 |

- The build reads a target table (gfx id → caps) and makes one code object per target.
  `-Dgfx=gfx1030,gfx1100,...` picks the targets.
- Kernels test caps macros, never GPU names. Warp code (shuffles, reductions, lane layouts) uses `TF_WAVE`. Wave64
  (gfx900/906) is a separate pass when hardware is available, but new code is written wave-generic from now on.
- A GPU outside the table is refused at open with its gfx id. There is no silent fallback.
- Under tp every rank must have the same caps. Ranks exchange a caps hash at join and refuse a mixed group.

### 3.2 Numeric formats: three small plug-ins, not kernels

A product `y = x · Wᵀ` in any format comes down to three choices, and the tiles take each one as a plug-in:

| Plug-in | What it does | Examples |
|---|---|---|
| **WeightDecoder** | unpacks one K chunk of one column from its stored layout into dot operands, plus the per-group terms (scale, zero/bias, shared exponent) | identity (fp16/bf16), MLX affine 2-8 bit, GPTQ/AWQ int4 (+zero), int8, mxfp4 / mxfp8 (e2m1 / e4m3 with an e8m0 exponent a 32 block), EXL3 trellis |
| **ActEncoder** | turns the activation rows into dot operands once per launch (or per row block): pass-through, or quantize with per-token / per-group scales | f16, bf16 (identity); int8 per token or per group (the A8 formats); fp8 later |
| **Dot** | the multiply-accumulate unit and its accumulator, chosen from Caps | `dot2 f16→f32`, `dot2 bf16→f32`, `sdot4 i8→i32`, `sdot8 i4→i32`, `wmma f16/bf16→f32`, `wmma iu8/iu4→i32`, `wmma fp8→f32` (RDNA4), packed-fp16 FMA (gfx900) |

A precision **mode** is a (WeightDecoder, ActEncoder, Dot) triple plus its epilogue: rescale an integer accumulator
by the weight and activation scales, then add a bias, an activation or a residual. Lean means:

- adding a weight format is one decoder (tens of lines) plus its `quant` module on the host;
- adding an activation precision is one encoder;
- a new GPU instruction is one Dot;
- the tiles (stream, gemm, matrix, routed) and their epilogues are written once.

Modes, as (decoder, encoder, dot):

| Mode | WeightDecoder | ActEncoder | Dot by GPU | Host module |
|---|---|---|---|---|
| FP16 / BF16 | identity | f16 / bf16 | dot2 f16/bf16; wmma f16/bf16 | `quant/dense.zig` |
| MLX affine 2/3/4/5/6/8 | bit unpack (packed across words for 3/5/6), scale and bias a group | f16 / bf16 | dot2; wmma bf16 | `quant/mlx.zig` |
| AWQ INT4 (and GPTQ) | int4 unpack, `(q − z) · s` a group (literal or +1 zero), optional g_idx | f16 / bf16 | dot2; wmma; native qgemm on RDNA2 | `quant/awq.zig` |
| FP8 | e4m3/e5m2 to f16/bf16 (table or bit ops), scale a tensor or channel | f16 / bf16 (fp8 on RDNA4) | dot2; wmma; wmma fp8 on RDNA4 | `quant/fp8.zig` |
| MXFP4 / MXFP8 | e2m1 / e4m3 to f16/bf16, times 2^e8m0 a 32 block | f16 / bf16 | dot2; wmma; wmma fp8 on RDNA4 (MXFP8) | `quant/mx.zig` |
| EXL3 3INST / MUL1, 2-8 bpw | trellis state stream (16-bit shift register a tile) decoded through the 3INST or MUL1 codebook, scale a channel | f16 / bf16 after the input Hadamard | dot2; wmma | `quant/exl3.zig` |
| later W4A8 / W8A8 | int4 / int8 | int8 a token or group | sdot4; wmma iu8 | `quant/awq.zig`, `quant/int8.zig` |

EXL3 needs more than a decoder:

- The input and output Hadamard rotations, and the channel scales, are pre/post steps of the projection. They become
  ActEncoder/Epilogue pieces: a rotation of x before the product and of y after it.
- The trellis decodes a 16×16 weight tile at a time, so its K chunk is a whole tile.
- If the shared tiles cannot keep up with the trellis decode cost, EXL3 keeps native kernels, registered like any
  other.

Every host module provides:

| Piece | What it is |
|---|---|
| `detect` | recognizes the checkpoint's config (`quantization` for MLX, `quantization_config.quant_method` for gptq/awq/mx, exl3's own) |
| `load` | uploads its tensors, optionally repacked into the layout its decoder reads best |
| `slice` | tp rules: which axes cut and the alignment (MLX: N rows, K groups; GPTQ: N by 8-column packs, K by groups, g_idx along; MX: K by 32 blocks) |
| `reference` | fp64 dequant (and fp64 activation quantization for A8) for tools/truth and the kernel harness |
| native kernels (optional) | a format's own kernels (vLLM qgemm for GPTQ on RDNA2, EXL3's), registered next to the shared tiles and kept only where they win |

`weights.Affine` / `view.Affine` become `quant.Projection`: a mode tag plus a handle. The model code calls
`ops.project(x, proj, m)`. Precision rules for A8 modes: activation scales are computed by the ActEncoder in one fixed
order per row, so the row-exact rule holds for them as for everything else.

**As built (step 3).** `zig/src/core/quant/` holds a module a format (`mlx.zig`, `dense.zig`) behind `quant.zig`, where
`conforms` lists at compile time what a format provides: `id`, `decoder`, `Config` and `detect`, `Host`, `matches`,
`read`, `halves`, `rows` and `bytes`, `sliceRows`, `sliceCols` and `sliceStack` (tp), `Device` and `upload`, `Matrix`
and `view`, `elements` and `reference` (fp64). `quant.Projection` is a product's shape, its split flag and a `Handle`
union of the formats' device matrices; `ops.project`, `projectGroup`, `projectRouted` and `projectRoutedAct` dispatch on
it, with the MLX launches behind them as `ops.affine*`. Nothing in `quant/` names a backend: a backend uploads through
the `Uploader` it makes (`hip/upload.zig`) and maps a decoder id to its kernels. AWQ is a module that provides the same
pieces (`detect` reads `quantization_config`, `read` a layer's qweight, scales and zeros, the slices cut by 8-column
packs and groups, `reference` dequantizes in fp64) and a decoder id for its kernels.

### 3.3 Shared tiles: written once, parameterized by the plug-ins and Caps

| Tile | Path | Rows | Notes |
|---|---|---|---|
| stream | decode | any (1-16 in a block, more blocks past 16) | one sum order per column whatever the launch |
| gemm | prefill | any (small-row and 128-row blocks give the same bits) | dot2 / sdot; byte-identical to the previous tile |
| matrix | prefill | any | wmma11 / wmma12 (f16, bf16, iu8, iu4, fp8 per GPU) |
| routed | both | items of the plan | the stream or gemm tile reading pairs through items/members |

Each tile is `template <class WeightDecoder, class ActEncoder, class Dot, class Epilogue>`, built per Caps target, and
only the modes a target's Dots support are compiled into it. That keeps code objects small: gfx1030 gets no wmma, and
gfx900 gets no sdot. Epilogues (int rescale, activation, combine, residual, rms) are template parameters, so the fused
decode tails stay fused in every mode.

**As built (step 4).** The plug-ins are `MlxDecoder<BITS>` (`kernels/hip/quant/mlx_decoder.hpp`), `IdentityAct<Dot>`
(`quant/act.hpp`), `DotF16` and `DotBF16` (`common/dot2.hpp`), `WmmaBF16` (`common/wmma.hpp`) and the epilogues `F32Out`,
`StreamRound` and `StreamSwiglu` (`tiles/epilogue.hpp`). The tiles are `gemm_tile`, `gemm_kp_tile`, `wmma_gemm_tile` and
`stream_tile`, each `template <class Dec, class Act, class T, class Epi, ...>`; `quant/mlx_tiles.hpp` instantiates them
as the kernels the launchers name, and a format adds a header like it. The previous decode tiles and the Python-parity
kernels of `tiles/dot2.hip` stay MLX-only reference code.

### 3.4 Registry: kernel choice in one place

```zig
Entry { op, format, path: .decode | .prefill, family: FamilyId, caps: CapsPredicate, policy: PolicyPredicate,
        fits: fn (Shape) bool, cost: fn (Shape, Caps) f32, launch: fn (...) }
```

- `select(op, format, path, shape)` returns the cheapest entry that fits, under Caps and Policy.
- **Family rule:** for a given (op, format, path, caps, policy), every row count must select the same `family`.
  Entries in one family must be byte-identical for every row; that is proven by the harness and declared in the
  entry. The registry checks this at open by enumerating representative shapes, and the tests check it for m = 1..256.
- `cost` comes from a **tuning table** in the repo (`zig/src/hip/tuning/<gfx>.zon`), measured by `tf-hip-test tune`
  and reviewed like code. There is no runtime autotuning: it would make choices differ between runs and between
  ranks.
- `--explain-kernels` (and a field of the server's info) lists what each (op, path) chose and why.

**As built (step 5).** `core/registry.zig` is the selection (`Entry`, `Registry`, `select`, `verify`, `explain`) and
names no backend; `launch/mlx_entries.zig` is the MLX entries over `launch/affine.zig`'s tile launchers; the costs are
`hip/tuning/gfx1030.zon` and `gfx1100.zon` (the gfx11 parts, read through `core/tuning.zig`'s rows), today's thresholds as costs and row or item limits, so a choice
is what it was. The decode path is a lane round (`shape.round`, which keeps the stream tile at any row count) and the
calls of the head and the draft, which keep the stream tile to 16 rows (15 where the matrix cores take the rest) and the
wide or GEMM tile above them; the prefill path is the K-parallel and 128-row tiles. `verify` checks the family rule for a
lane round and the prefill paths at open, and the tests check it for every GPU table; `check --explain-kernels` prints
each (op, path) choice.

### 3.5 Policy: what a run may use, chosen once

The env vars are replaced by a **Policy** value resolved at `Engine.open`, logged on one line, reported by the
server, and passed explicitly to the registry and the engine. Nothing below the engine reads the environment.

```zig
Policy {
    matrix: .auto | .on | .off,            // matrix cores (WMMA/MFMA) where the registry has entries
    activations: .auto | .f16 | .bf16,     // auto: f16 on RDNA2, bf16 where dot2/wmma bf16 exists
    attention: .auto | .f16 | .bf16 | .f32 // the prefill attention operands (int8 later)
    kernels: .auto | .shared | .native | .reference, // native: a format's own kernels when registered; reference: the slow exact ones for bisecting
    graphs: .auto | .off,                  // HIP graph replay of rounds (tp included)
    prefill_step: u32 = 1024,              // rows a prompt advances a round while others decode (a multiple of 64)
    mtp: { drafts: 0..3, confidence: f32 }, // drafts 0 = off
    prefix: { slots: u32, bytes: u64 },
    exact: .strict | .relaxed,             // strict refuses any entry not proven row-exact; relaxed may use faster non-exact ones (none exist today)
}
```

- **Precedence (lowest first):** built-in default for the caps < tuning table < model-directory config
  (`tensorfold.zon` next to the checkpoint, optional) < CLI/server flags (`--matrix off`, `--activations f16`,
  `--kernels shared`, `--mtp-drafts 2`, `--prefill-step 2048`, `--no-graphs`) < debug overrides.
- **Debug overrides** stay possible through one variable, `TF_POLICY="matrix=off,kernels=reference"`, parsed by
  the same code as the flags. When it is set, the start-up line says so. The current scattered variables are read
  once as aliases of it during the migration, then removed.
- **tp:** rank 0 resolves the Policy and sends it to the followers at join. Every rank uses that Policy. A follower
  only adds its local caps check.
- **Tests** build a Policy directly, so the test matrix never touches the environment.
- What `TF_WMMA` meant becomes `matrix=off|on|auto`. `TF_FA_WIDE=0` becomes `attention=f32`.
  `TF_AFFINE_GEMM/GEMV=old` and `TF_DECODE_FUSE=old` become `kernels=reference` (or named families once there are
  several). `TF_GDN_CHUNKED=0` becomes a reference recurrence.

### 3.6 Graphs: every round replays, whatever its streams and drafts

A round's graph used to be keyed by its streams' caches and row counts, so a new stream, a different number of accepted
drafts or a confidence cut made a new shape, and that round either captured a graph or ran eagerly. Now a launch depends
on the shape of the round alone.

**Kernels read the round from device memory, not from launch arguments.** One buffer on the device
(`forward/plan.zig`, `kernels/hip/decode/plan.hpp`), written by one copy a round, holds:

- per row: its token, its position and its slot;
- per slot: its first row and row count, and the address of its descriptor;
- per linear layer: where the round's conv and DeltaNet snapshots are.

A stream's descriptor (`state.Caches.desc`, made with its caches) holds the positions its caches take, the address of
its last kept final row (the draft head's input), and two addresses a layer (the key and value pools, or the conv window
and the state), then its page table (3.7). The window forward's rope, cache writes, attention walk, conv and DeltaNet
kernels index through it, one launch a layer whatever the number of streams, and nothing of a stream is an argument.

**Shapes come in buckets.** A round's graph is keyed by (rows, slots, keys covered):

- rows are padded up to a bucket (1, 2, 4, ... 64, the engine's limit past them); padding rows run in a scratch slot,
  the last, whose conv and DeltaNet kernels keep no snapshots, and are dropped;
- slots are the streams' plus the scratch slot, padded up to a bucket too (empty slots list nothing);
- the attention walk covers the keys the longest row sees, rounded up to a power of two;
- every kernel is row-independent (section 1), so padding never changes a real row's bits (`rows` checks it).

A graph is captured the first time its shape is met (a capture costs milliseconds) and replayed on every later round of
the shape, whatever its streams, positions and caches.

**A speculative round is two replays and two syncs.**

1. verify: the plan goes up in one copy, the graph runs the window forward, the head projection and the greedy draw of
   every row, and the tokens come back;
2. the lane core (shared with the other backends) accepts, cuts and sizes the next drafts on the host, as ever;
3. keep: one launch for every window restores each linear layer's state from the snapshot of the last kept row and
   copies the stream's last kept final row where its descriptor says;
4. draft: the head's greedy chains of every stream run as one graph keyed by (chains, drafts, confidence cut), their
   input rows gathered through a device address table, and the drafts and their probabilities come back.

The host's acceptance stays in the lane core: moving it onto the device would need the core to queue a round before
reading the last, and it also holds the stops, the thinking budget and the loop cuts that end a window early. Sampled
draws (top-k candidates, drawn with the keyed host sampler) and sampled head chains still synchronize each step, and
stay correct.

**Under tp**, rank 0's pick (replay, capture or eager) goes with each round message, every rank derives the same plan
shape from the tokens and positions the message carries, and a capture that fails on any rank turns the shape eager on
all of them. A capture of a round with collectives takes hundreds of milliseconds and a replay is no faster than the
eager round there (the ranks wait on each other), so graphs stay off under tp unless the Policy says `graphs=on`.

**Prefill steps** still run eagerly. A graph for them needs the cache pointers and positions of the attention, the conv
and the recurrence read through a descriptor too, and a tail bucket needs a device row count in the conv and DeltaNet
kernels (padding rows after the last real one would advance the states).

**Proof.**

- Graph = eager byte for byte, greedy and seeded, on short and long prompts (`check`'s lane lines), and each padded
  bucket against its rows one at a time, layer by layer (`rows`).
- `check` fails when a repeated drafted run replays under 99% of its rounds; `check --speed` reports the share of
  rounds replayed, the host time a round takes to submit its launches, and the time of each backend call (verify, keep,
  draft).

### 3.7 Prefix reuse: a radix cache over paged KV

A kept prompt used to be a whole copy of its caches at a cut point, so each copy cost a full KV prefix and two requests
that shared a system prompt each held their own. Now requests share prefix memory, and resuming copies no key or value.

**Paged KV** (`forward/pages.zig`, `forward/state.zig`).

- The full-attention layers keep K and V in one pool of pages of 64 positions, the same 64 as the chunked recurrence and
  the cut spacing, so a page edge is a chunk edge. A layer's pool is `kv_heads` runs of pages, so a head's pages sit side
  by side and a stream whose pages were taken in order reads a head's keys as one stretch, as a flat cache did. A page is
  the same id in every layer's pool.
- A stream's page table, one 32-bit page id for each 64 positions, sits right behind its descriptor (3.6), at the word
  offset the round's plan carries. The plan's decode walk (score and apply), the plan's KV write, the prefill KV write
  (`tf_page_write`) and the 64-row prefill attention tile (`tf_fa_paged_*`, a key tile is a page) all read and write
  through it. Any other prefill shape reads a gathered flat copy of the pages, so every shape runs the kernels it ran
  before, on the same bytes.
- Pages are reference-counted (`pages.Ids`). A stream takes pages as it grows, against a promise made when it starts (its
  whole window), so a round never runs out of them; a page two holders share is copied before one writes it
  (`Caches.writable`). A resumed span starts on a page edge, so a stream's first write lands on a page of its own and the
  copy is a guard, not a path a prompt takes.
- The linear layers keep a stream's conv window and DeltaNet state as before.

**The prompt cache** (`core/prompt_radix.zig`, `prompt_radix_tree.zig`, `prompt_radix_test.zig`) has the shape of upstream's
`core/prompt_cache.zig` Store, and nothing in it knows a GPU, so it can move to core: the backend gives it page reference
counts (`Pages`) and copies of its non-page state (`Snapshots`: bytes, save, restore, drop); it gives back a `Plan` (where
the pass resumes, where it keeps), `keep`, `forget`, `marks` and the counters.

- A node is a run of whole pages keyed by their tokens. A request takes references on the pages of its longest match
  and prefills the rest.
- **Marks** (where a pass keeps a state) are upstream's: the rendered history, the prefix shared with the conversation's
  last prompt, the ends of shared system blocks, each at least `min_gap` from the others and from the resume point; and
  the prompt's last whole page. All sit on page edges.
- A state is the family's snapshot at a node's end. A match resumes from the deepest node on its path with one; the
  pages below it are shared, and the stream's own duplicates of pages the tree has are swapped for the tree's.
- **Eviction** takes a state its own conversation has moved past first, oldest first, a shared cut only as the least
  recently used, and then the least valuable state: one never resumed before one that was. Pool pressure takes the
  least valuable leaf nobody holds. A node with no state and nothing below it goes with its pages.
- **Budget**: `--prompt-cache-gib` (or the Policy's `prefix.bytes`) is the tree's bytes, `--checkpoint-slots` (the
  Policy's `prefix.slots`) its snapshots. The memory plan (`memory.zig`) splits it: snapshots may take up to half of it,
  the pages the rest, and the pool holds every stream's whole window, the scratch rows and those pages. Snapshots are
  counted in the budget because they are bought from it.

Where it differs from upstream's Store: its entries are nodes of a tree, so pages are shared between them and the
limits count pages and snapshots apart (`Limits`); `begin` also hands back the pages of the resume point (`adopt`) and
`keep` takes the stream's pages and returns the tree's (`path`); `Rules.page` stands in for `planned` and the chunk
starts (a resume and a mark sit on page edges, and a row's bits do not depend on its chunk), there is no lookahead, no
spare storage and `min_prompt` is zero. Moving it to core means taking upstream's `Rules`, `Counts`, `Plan` and
`Snapshots`, and adding `Pages`, `adopt` and `path` to them.

**Linear state** is not a per-token cache: resuming needs each linear layer's conv window and DeltaNet state as of the
node's last token. A snapshot is large (35B-A3B: about 60 MB across its 30 linear layers), which is why it is kept only at
the marks, and why the tree may hold fewer snapshots than pages.

**Exactness.** Pages hold the same K and V a fresh prefill writes; resumed spans start on a page edge; snapshots are exact
copies. `check` proves resumed == fresh and the page contents:

- a radix stress: requests branching from one shared system prompt, a conversation's next turns and branches inside and
  between pages, each against its own fresh run alone, one at a time on the growing tree and then all at once;
- the pages of a resumed prompt against the pages of a fresh prefill of it, byte for byte in every attention layer;
- a shared page is copied before a write, and every page is back in the pool when the streams are gone;
- a memory line: the bytes the tree holds for requests sharing a system prompt against copies of the caches at each mark.

**Under tp**, rank 0 owns the tree and decides every match, insertion and eviction, and names pages and snapshot slots; the
other ranks apply them from the messages (`worker.Op`): the prefill message carries the stream's pages and the slot to
resume from, `pages` sets a stream's page table from an index, `snap` copies the stream's linear state into a slot, `copy`
copies a page. A message is as long as its step needs, and a step added later takes the next number.

### 3.8 Layout: folders by job

Upstream keeps a backend's runtime flat (`zig/src/cuda/`) and its kernels one file per job (`zig/kernels/cuda/`). The
HIP port follows that and adds one level where it carries more: formats, tiles and the registry. A file stays under
about 600 lines unless splitting would cut one job in two: a tile template, or a launcher table that must read as one
list. In that case the PR says so. Comments are one line, unless a longer one says something the code cannot (an
exactness rule, a hardware fact).

```
zig/src/hip/
  root.zig                     the backend's public surface
  runtime/                     driver, context, stream, memory, arena, module, graph, abi (as zig/src/cuda/)
  comm/                        rccl, link (tp transport)
  caps.zig  policy.zig         what the GPU can do; what the run may use (3.1, 3.5)
  launch/                      args, code objects, the entries of the registry (3.4), tuning/<gfx>.zon
  ops/                         the Ops facade split by job: project (dense + routed), attention, recurrence,
                               norms, moe (router, select, plan, combine), draw, tp (sums, gathers)
zig/src/core/                  quant/ (the weight formats: mlx.zig; later awq, fp8, mx, exl3, 3.2), registry.zig and
                               tuning.zig (3.4), prompt_radix*.zig (3.7), round_shape.zig (3.6)
zig/kernels/hip/
  common/                      caps macros, wave helpers, Dot plug-ins
  quant/                       one WeightDecoder header a format (mlx.hpp first), ActEncoders
  tiles/                       stream, gemm, matrix, routed: templates over decoder × encoder × dot
  attention/                   prefill tile, decode (causal_at), qk_rope
  recurrence/                  gated delta: serial (decode), chunked (prefill)
  ops/                         norms, conv, rope, moe route/select/combine, casts, the fused decode tails
  comm/                        tp kernels
  capi/                        the library's C ABI (the bring-up path)
zig/src/families/qwen3_5/      the family as 3.9 builds it: config, weights/ (the checkpoint reader), hip.zig and backend/hip/
                               (model/, forward/, engine/), beside upstream's Metal files
zig/tests/hip/                 kernels/, runtime/, bench/ (4.2)
zig/tests/qwen35/              check/ (invariants, accuracy, speed), matrix.sh
```

`zig/kernels/hip/rocm/` (the kernels copied from the Python ROCm engine) is absorbed: each file moves to the folder
of its job and is split into decoder plus tile as step 4 reaches it.

### 3.9 Converging with upstream's Zig structure

Upstream's guide (docs/recipes/adding-a-zig-family.md) asks that each speed-up be built once, in `zig/src/core/`, and
that a family supply only its config, weight map, layer graph and its own mixer kernels. Upstream's `zig-preview` line
has a native `zig/src/families/qwen3_5` family (Metal, Qwen3.5-2B) with the same exactness contract as this port.
`zig-flashnext` (this port's base) has the CUDA family registry in the native server (#443) and the core prompt cache.

This port converges with that, keeping `zig-flashnext` as its base:

- **The family is `qwen3_5`, with upstream's file roles:** qwen3_5.zig, config, weights, model, forward, state, and a
  backend per GPU API. `families/qwen35` is renamed into it; its HIP-specific engine pieces become the HIP backend.
  When `zig-preview` and `zig-flashnext` meet, the two families merge file by file instead of duplicating.
- **Generic pieces move to core** where upstream's table puts them:
  - lane kernels by format: the format plug-ins, tiles and registry of 3.2-3.4, with the device side under
    `zig/kernels/hip/` and the selection logic in core;
  - GPU-side rounds: the device round plan, graphs and speculative rounds of 3.6;
  - the prompt path: the radix cache of 3.7, folded into `core/prompt_cache.zig`'s interface;
  - serving: HIP registers through the same family registry as CUDA (#443), and hosts through `core/lane_host.zig`.
- **What stays in the family:** its config, weight map and layer graph, plus the Gated DeltaNet mixer (serial and
  chunked) and the gated attention specifics.
- **The core interfaces stay backend-neutral** (Metal, CUDA, HIP), so Flash Next, Nemotron or a new family can use the
  same pieces.

**As built (step 5d).** Upstream's `zig-flashnext` is merged and the Zig HIP port sits in its structure.

*Serving.* One `tensorfold-native` serves Linux: `native/linux.zig` hands a checkpoint to the backend whose registry lists its
`model_type`, `native/cuda.zig` (nemotron_h) or `native/hip.zig` (qwen3_5, qwen3_5_moe). A registry entry is a namespace with
`model_type`, `formats`, `default_context`, `prefill_step` and `open` (a rank above 0 also has `follow`), as upstream's CUDA
entries are; the family's own is `families/qwen3_5/backend/hip/engine/native.zig`. The host owns the card, the policy, the
tensor-parallel group (`hip.Device`, `hip.Group`), the round loop and `core/lane_host.zig`; the family loads, plans memory,
sizes and returns its lane backend, the facts the round loop reads and the window it fitted. `--backend {auto,cuda,rocm}`,
`--tp`, `--rank`, `--master`, `--p2p`, `--matrix`, `--kernels` and `--policy` ride in `engine_api.Open` next to upstream's
`--prompt-cache-gib` and `--prompt-cache-over-cap`.

*The family.* Upstream's `qwen3_5` files are untouched (`qwen3_5.zig`, `config`, `weights`, `model`, `forward`, `state`,
`backend`, `kernels`, `qualify`: Metal). The HIP family is beside them:

```
zig/src/families/qwen3_5/
  qwen3_5.zig                  Metal root (upstream)         hip.zig            the HIP family's root, as nemotron's cuda.zig
  config.zig                   Metal's admission of the 2B   weights/           the checkpoint reader, for any backend: config
                               geometry, and the Spec every                     with the quantization table, the tensor table,
                               backend reads                                    host projections, layers, experts, tp cuts
  weights.zig model.zig forward.zig state.zig backend.zig kernels.zig qualify.zig   Metal (upstream)
  backend/hip/model/           the device model: weights uploaded, view, bridge
  backend/hip/forward/         forward, window (decode round), moe, reduce, state, pages, plan
  backend/hip/engine/          engine, hip_lanes, worker, round, mtp, draw, memory, hip_prefix, native
```

The test program keeps its name, `tf-qwen35-test` (`zig/tests/qwen35/`): the scripts and this plan name it.

*Core.* `core/quant/` is the format interface and its two formats (3.2); `core/registry.zig` and `core/tuning.zig` are the kernel
choice and the rows of a GPU's table (3.4), while the entries with their kernels (`hip/launch/`) and the `.zon` tables stay
in `zig/src/hip`; `core/prompt_radix*.zig` is the paged prompt cache on `core/prompt_cache.zig`'s `Rules`, `Counts`, `Plan`
and `Snapshots`, with `Pages`, the adopt list of a resume and the `path` of a keep added (3.7); `core/round_shape.zig` is
the bucketing of a round's rows and span that keeps graphs few (3.6).

*What stays HIP, and why.*
- The plan buffer's word layout (`backend/hip/forward/plan.zig`): it is the layout the HIP kernels read.
- The prefix tree's glue (`hip_prefix.zig`): the pool's pages and the linear snapshots live in device memory, and rank 0's
  tree is mirrored to the other ranks by messages.
- The GPU tables and the entries' kernel names: they are a GPU's measurements.
- `qualify.zig` (load-time check that wide rows equal one row on the real weights) is Metal's: it compares Metal buffers. The
  HIP equivalent is `tf-qwen35-test check`, which runs at the engine's own widths.
- The radix cache is driven by the HIP backend, not by `lane_host`'s `cache` (the flat `prompt_cache.Store` with a copy of
  the whole state per entry): a resume shares pages with live streams, a keep swaps the stream's pages for the tree's and
  rank 0's decisions go to the other ranks. `lane_host` hosts the backend all the same, with the chunked prompt fill
  (`prefill_step`) and the cancel check between chunks.
- Metal's admission stays strict (its kernels are built for one geometry); widening it is a kernel job, not a loader one.

## 4. Testing

### 4.1 Before step 0c: grown one tool at a time

| Entry point | Commands | Kind |
|---|---|---|
| `zig build test` | 27 unit tests (cuts, sizes, slicing, flags) | host only |
| `tf-hip-test` | `info smoke graph cooperative library image` | runtime |
| | `affine` (Python fixtures), `gemm`, `gemv`, `decode`, `gdn` | kernels: each has its own harness, reference and output format |
| | `launches`, `overhead` | benchmarks |
| `tf-qwen35-test` | `check digest layers draw` | weights, Python-oracle layers, device draws |
| | `lanes rows logits prefill` | engine runs, invariants, logits dumps, speed |
| host scripts (outside the repo) | `tpcheck.sh replies.sh verify.sh compare.py decode.sh acc.sh prefill_ab.sh matrix.sh run_dump.sh py_prefill.py kstats.py` | wrappers that re-run `lanes`/`logits` and compare JSON |
| `zig/tests/qwen35/serve_check.py`, `serve_tp.sh` | HTTP invariants under tp | server |

The same questions get asked in several places with different outputs: is the row exact, is the output accurate,
how fast is it.

### 4.2 Target: four entry points, one question each

| Entry point | Answers | Replaces | Needs |
|---|---|---|---|
| `zig build test` | host logic is right | unchanged | nothing |
| `tf-hip-test kernels [--filter F] [--bench] [--oracle DIR]` | each registered kernel against an fp64 reference, and byte-identity within each family across row counts and block sizes | `affine gemm gemv decode gdn exact` (one harness, built: 4.3; `--bench` gives the timings the separate benches gave) | one GPU |
| `tf-hip-test runtime` | the HIP runtime works on this GPU | `info smoke graph cooperative library image` (one command, subtests by filter, built: 4.3) | one GPU |
| `tf-qwen35-test check MODEL [--tp N] [--truth T]` | **invariants** (window vs one row by layer; drafted = serial, solo = together, resumed = fresh, greedy and sampled, short and long prompts; graph = eager), **accuracy** (prefill and decode logits scored against the fp64 truth in-process, if given), **speed** (prefill lengths, decode 1/4 streams, MTP depths) | `rows lanes logits prefill draw digest check`, and the scripts tpcheck/replies/verify/compare/decode/acc/prefill_ab | one model, N GPUs |

Plus the official server tools, unchanged, for receipts (CONTRIBUTING): `tools/bench_concurrent.py --alone --serial`,
`bench_openai.py`, `prefill_cold.py` against `tensorfold-native`. `serve_check.py` folds into them: what they lack
(resumed = fresh, tp) is added to `bench_concurrent.py` as options, or stays a short check of its own.

How they compose:

- **The kernel harness is driven by the registry** (3.4). Every entry declares its family, its shapes and its
  reference, so a new format, GPU or tile is tested by being registered, with no new command.
- **`check` is driven by the Policy** (3.5). The matrix is `check` run over a list of Policies × models × tp, and
  one in-repo script (`zig/tests/qwen35/matrix.sh`, taking model directories and device lists as arguments, no hosts or
  paths in it) writes the PR's table.
- **The Python-oracle tests** (`affine` fixtures, `layers`) become an optional `--oracle DIR` of `kernels` and
  `check`. The Python engine is frozen and the fp64 truth is the bar, so they only matter as a regression check of
  the fp32 reference paths.
- **Benchmarks** (`launches`, `overhead`) live under `tf-hip-test bench`, and the old per-tile timings under
  `kernels --bench`; none is a test.
- **Every test fails before its fix and passes after**, as CONTRIBUTING asks. The regression cases found so far
  (prefill cut vs whole, follower prefix mirroring, the confidence cut, padded rows) become named cases of `check`.

| Level | Run when |
|---|---|
| `zig build test` | every build |
| `tf-hip-test kernels` (+ runtime) | every kernel change; once per GPU per merge |
| `tf-qwen35-test check` on 0.8B and 35B-A3B, tp 1 and 2 | every merge |
| matrix (every Policy × model × tp, both GPUs) and the official tools | before a PR update or a release |

### 4.3 `tf-hip-test` as built

```
tf-hip-test [--policy K=V,...] [-v] runtime [--filter F]
tf-hip-test [--policy K=V,...] [-v] kernels [--filter F] [--bench] [--oracle DIR] [--reps N]
tf-hip-test bench launches|overhead [n] [reps]
```

`--policy` (and `TF_POLICY`, and the old variables) set the run's Policy, so the matrix cores on and off, or the previous
tiles, are the same two commands with another policy. `-v` prints the steps inside each group. Every group prints one
PASS or FAIL line, and the command ends with a summary and a nonzero exit when a group failed. `--filter` takes a
substring of a group's name. Files: `runtime/` (the probes), `kernels/` (the registry-driven harness and the other
kernels), `bench/`, `args.zig` (the command line and the aliases, with host tests under `zig build test`).

**`runtime`** groups: `smoke graph cooperative library image` (and the `INFO` line of `info`).

**`kernels`** groups:

| Group | What it checks |
|---|---|
| `affine decode`, `affine prefill`, `affine sweep` | for every product (the decode shapes of three models, every width and group, ragged sizes, routed plans) each registry entry that the GPU has and the shape fits is launched on its own: its sampled outputs against dequant(W) . x in float64 within the bound of an fp32 sum (rounded outputs within the activation type's rounding too), and the entries the registry picks between by rows under the run's Policy write the same bytes |
| `rows decode` | in a lane round, each row of a launch is the row alone, at every row count up to 32, fp32 and rounded, plain products, routed plans and the gate-up activation epilogue |
| `rows prefill` | at every row count 1..200, width and group, dense and routed, every tile the registry picks between writes the same bytes, as does the engine's pick; the router's two kernels agree |
| `gdn` | the chunked DeltaNet against a float64 recurrence at four lengths |
| `decode router`, `decode tail`, `decode group`, `decode pair`, `decode chain` | the kernels outside the registry against the launches they replace (`fuse` off) and float64, or byte for byte: the router, the residual tails, products that share x, the routed activation epilogue, the MoE pick, the linear attention's conv and gated norm |
| `oracle affine` (with `--oracle DIR`) | the Python engine's affine fixtures, pinned to `kernels=reference`, bit for bit |
| `affine short`, `gdn bench` (with `--bench`) | timings only |

With `--bench` the affine groups also print, per product, each entry's microseconds and GB/s (decode) or TFLOPS (prefill),
and the decode groups the previous launches' and the merged ones' time.

The old command names stay as aliases for one release (`info smoke graph cooperative library image affine gemm gemv
decode exact gdn launches overhead`), each printing the group it now is:

| Old command or case | Now |
|---|---|
| `info` | `runtime` (its `INFO` line) |
| `smoke`, `graph`, `cooperative`, `library`, `image` | `runtime --filter NAME` |
| `affine DIR` | `kernels --filter oracle --oracle DIR` |
| `gemm [reps] [filter]` (29 products, new tile vs previous) | `kernels --filter "affine prefill"`: every prefill entry, `block` and `gemm` included, against float64 |
| `gemm sweep` (222 ragged products) | `kernels --filter "affine sweep"` (444: prefill and decode rows) |
| `gemm tiers` (1850 products and the router logits) | `kernels --filter "rows prefill"` |
| `gemm short` | `kernels --bench --filter "affine short"` |
| `gemv [reps] [filter]` (42 decode products) | `kernels --filter "affine decode"` (the same 42; each entry against float64, the stream and previous tiles included) |
| `exact` | `kernels --filter "rows decode"` |
| `decode router\|tail\|group\|pair\|chain` | `kernels --filter "decode NAME"` |
| `gdn`, `gdn bench [rows]` | `kernels --filter gdn`, `--bench --filter "gdn bench"` |
| `launches [n] [reps]`, `overhead [n] [reps]` | `bench launches`, `bench overhead` |

What changed in the comparison: the old harnesses compared one new tile with one previous tile (`TF_AFFINE_GEMV=old`,
`gemm=reference`); the registry lists every tile, so each is now compared with float64 directly, and the byte rule is the
family rule of 3.4 under the run's Policy, so `--policy gemm=reference` or `stream=reference` runs the previous rules.
The decode group launched its one-product-a-side comparison outside a lane round, which on the matrix cores refuses the
rounded output at 16 rows ("fp16 output is the decode tile"); it launches inside one now, as the engine does.

## 5. Phases

**Phase 1: the MLX path, complete.** MLX affine is the only format until all of this holds:

- **Bits and models:** every MLX width (2, 3, 4, 5, 6, 8; groups 32/64/128 at kernel level) on Qwen3.5-0.8B and
  Qwen3.5-9B. Qwen3.8-27B and Qwen3.6-35B-A3B at the widths their checkpoints come in. Widths missing on the host are
  made with tools/mlx_requant.py (MLX's own quantization rules, checked byte for byte against an official checkpoint),
  and each gets its own fp64 truth.
- **GPUs and modes:** RDNA2 fp16, and RDNA3 bf16 with the matrix cores on and off; MTP on and off; concurrent
  streams; tp 1, 2, 4 and 8; graphs at every tp.
- **Correctness:** every invariant, plus fp64 accuracy for every model.
- **Speed:** prefill ≥ 2x Python, decode ≥ 2x the port's baseline, and MTP gains with concurrent streams too.
- **Structure:** steps 1-5b below (Policy, Caps, the Quant interface, plug-in tiles, the registry, graphs everywhere, the
  radix prefix cache, and convergence with upstream's family and core structure),
  done with MLX as the only format. Each step is proven by MLX's own tests.

**Phase 2: the other formats** (steps 6-9), each added as decoder plug-ins once the structure is in place.

**Phase 3: new GPUs** (steps 10-11) as hardware becomes reachable.

## 6. Migration (each step keeps today's bits and speed)

| Step | Change | Proof |
|---|---|---|
| 0 | this plan; owners and file ownership | review |
| 0c | **Tests regrouped** (4.2; `tf-hip-test` built, 4.3): `tf-hip-test kernels` and `runtime`, `tf-qwen35-test check`, the in-repo matrix script; host scripts retired | the same cases pass; nothing loses coverage (a mapping table in the PR) |
| 0b | **CONTRIBUTING pass**, before any refactor (one owner, after the branches in flight land):<br>- the layout of 3.8 (moves first, as their own commits, then splits);<br>- `launches.zig` and `ops.hip` split by job under 600 lines;<br>- one-line comments everywhere;<br>- receipts from tools/bench_concurrent.py, bench_openai.py and prefill_cold.py against tensorfold-native;<br>- authorship under the GitHub noreply address (set) | truth scores; rows/tpcheck; speed recorded (32k prefill expected lower) |
| 1 | **Policy**: struct, resolution, flags, `TF_POLICY`, the old variables as aliases, start-up line and server info; ops/registry read Policy instead of env | all variables' behaviors unchanged (matrix of on/off runs) |
| 2 | **Caps** replace `Family`; target table; gfx1151 and gfx1200 build | fixtures, rows, tpcheck, speed unchanged |
| 3 | **Quant interface** with mlx: `quant.Projection`, `ops.project` | byte-identical logits (prefill and decode) on 0.8B/9B/35B |
| 4 | **Decoder-templated tiles**: MLX unpacking moves out of stream/gemm/matrix/routed | `tf-hip-test kernels` byte-identity old vs new; speed unchanged |
| 5 | **Registry** replaces the m-rules in affine_launch/launches/ops; tuning tables for gfx1030 and gfx1100 | family check; byte-identical logits; speed unchanged |
| 5b | **Graphs everywhere** (3.6): device round plan, shape buckets, keep and the draft head in graphs, tp (open: device-side accept, prefill steps) | graph = eager byte for byte; > 99% rounds replayed; MTP beats no-drafts with 4 streams |
| 5c | **Radix prefix cache** (3.7): paged KV with page tables in the round plan, the radix tree with state snapshots at chosen nodes, copy-on-write tails, tp mirroring | resumed = fresh; radix stress test; memory per shared prefix; time to first token on a shared system prompt |
| 5d | **Upstream sync and convergence** (3.9, done): merge upstream zig-flashnext (registry #443, prompt cache, guide); `families/qwen35` becomes `families/qwen3_5` with upstream's file roles; HIP registers through the family registry and hosts through `core/lane_host`; generic lane kernels, GPU rounds and the prompt path move to core behind backend-neutral interfaces | `check` identical digits and speed; tp2; native server through the registry; `zig build test` and the Metal build unchanged |
| 6 | **FP16 / BF16** (identity decoder) and **MLX 5-bit** | truth scores; speed |
| 7 | **AWQ INT4** (and GPTQ by flag): detect, load, slice, reference, decoder; then the vLLM qgemm ports as native RDNA2 entries | truth scores; bit-exact against Python qgemm fixtures for the native kernels; the model matrix on AWQ checkpoints |
| 8 | **FP8** (tensor/channel scales), then **MXFP4 / MXFP8** decoders | truth scores |
| 9 | **EXL3** 3INST / MUL1 at 2-8 bpw: trellis decoder, Hadamard pre/post steps, native kernels where faster | truth scores against an fp64 decode of the same trellis; speed vs the vLLM fork's kernels |
| 10 | **RDNA3.5 / RDNA4 bring-up** (when hardware is reachable): caps rows, matrix12 tile, fp8 Dot | the matrix on that GPU |
| 11 | **gfx900/906**: wave64 pass over warp code; packed-fp16 Dot where dot2 is missing | as step 10 |
| later | W4A8 / W8A8 (int8 ActEncoder, sdot4 / wmma-iu8 Dots) | as step 7 |

Steps 1-5 are refactors with no new features and sit on the files the decode work changes now (ops.zig,
launches.zig, affine_launch.zig, the affine tiles). They start once agent/decode2 merges, one owner at a time for
those files. Step 1 can start earlier because it only adds the Policy and reroutes reads.

## 7. Open questions

1. **Act-order GPTQ:** support it in step 6 (reorder K at load plus an x gather a layer), or refuse it at first?
2. **Checkpoints to test with:** AWQ, FP8, MXFP4/MXFP8 and EXL3 (3INST and MUL1, several bpw) versions of
   0.8B/9B/27B/35B-A3B. On the host now: Qwen3.6-35B-A3B-GPTQ-Int4, Qwen3.5-0.8B-exl3-3inst, Qwen3.8-27B-exl3-3.00bpw.
   Fetch or make the rest?
3. **MLX 2-bit and 5-bit:** neither is on the host. Quantize them locally with mlx-style tooling for the matrix?
4. **Policy file next to the checkpoint** (`tensorfold.zon`): wanted, or flags only?
5. **`exact=relaxed`:** keep the slot for a future faster-but-not-row-exact mode (e.g. int8 attention), or leave it
   out until a kernel needs it?
6. **Wave64 hardware:** is a gfx900/906 card available to test on, or is that design-only for now?
