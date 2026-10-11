# Qwen3.8 Flash Next on HIP: plan

Status of the `hip-flashnext` branch and the plan for serving Qwen3.8 Flash Next (`model_type: qwen4_exp`)
on TensorFold's native HIP backend: Radeon PRO V620 (gfx1030, RDNA2) and Radeon PRO W7800 (gfx1100, RDNA3).

## Verdict

Feasible, and it fits our cards at the two targets once the n-gram tables stay in host memory:

| Target | Checkpoint | Weights a rank | Longest single-stream window | Verdict |
| --- | --- | --- | --- | --- |
| 4 x V620, tp 4 | 4-bit | 19.5 GiB | about 430k positions (the model's 262,144 with room for ~1.6 such streams) | fits |
| 2 x W7800, tp 2 | 4-bit | 37.9 GiB | about 200k positions | fits, below the model's window |
| 2 x W7800, tp 2 | 3-bit | 30.4 GiB | about 660k positions | fits at the model's window |
| 4 x V620, tp 4 | 6-bit (upstream's qualified format) | 27.2 GiB | under 20k positions | not useful |
| one card or V620 tp 2 | any of 3/4/6-bit | 30-105 GiB | none | refused |

The n-gram tables (`ple_embedding.ngram_embedding`, 128 shards of 2,500,012 rows) are 29.8 GiB at 4 bits and
23.8 GiB at 3 bits; a token reads 16 rows of 160 values, about 1.6 KB, so they serve from host memory (as the
Python engine's `TF_NGRAM_HOST=1` does) and never take device memory. par1 has 503 GB of RAM.

Most of the forward maps onto kernels the Qwen3.5/3.6 HIP engine already has (MLX affine products at group 32,
the gated delta rule, q/k norm and RoPE, gated attention output, MoE routing with a shared expert, MTP drafts).
What is new is the model's own glue: hyper-connections, the n-gram (PLE) layer, the indexer's block selection
and attention over the selected keys. Those kernels are written and pass their GPU tests on gfx1030 on this
branch. The engine that strings them together (weights upload, forward, lane backend, prompt cache, MTP,
tensor parallelism) is the remaining work, roughly 4-6 weeks of review-sized PRs.

## The model

Read from the family (`zig/src/families/flashnext`: config, index, weights, pack, engine, replay, gdn, tp,
snapshot, learned), the Metal kernels it replays (`zig/kernels/metal/flashnext`) and the Python reference
(`python-0.6`, `src/tensorfold/families/qwen4_exp/model.py`, `model_layers.py`). Dimensions are the
checkpoint's (`fixtures/config.json`).

- hidden 2560; vocabulary 248,320; 48 layers; RMSNorm eps 1e-6; every norm but the DeltaNet's is centred
  (scale `1 + w` in fp32).
- **Hyper-connections**: the residual is 4 streams of 2560 (10,240 wide). Every block reads it through a
  hyper-connection: each stream RMS-normed (`hc_norm`, one scale over the 10,240), a low-rank down projection
  (10,240 -> 320) and the inject projection (10,240 -> 4) on the normed streams, both divided by 4;
  `act = silu(.)`, `inject = 2 sigmoid(.)`; the up projection (320 -> 10,240) gives `gate = sigmoid(.)`;
  the block input is the mean over streams of `gate * normed`. The block's output goes back into every
  stream times its inject gate: `h_s += branch * inject_s`. Two hyper-connections a layer (attention or
  DeltaNet, then MoE); a final mixer without inject gives the 2560-wide input of `lm_head`. There is no
  final norm beyond the mixer's.
- **Embedding**: MLX affine rows, copied into the 4 streams.
- **Layer schedule** (`layer_types`, `full_attention_interval` 4): layers 3, 7, ..., 47 are sparse attention
  (12), the other 36 Gated DeltaNet.
- **Gated DeltaNet**: 16 key heads, 48 value heads (3 value heads a key head), head dim 128; `in_proj_qkv`
  (10,240 rows), `in_proj_z` (6,144), `in_proj_b`, `in_proj_a` (48 each); conv of 4 taps then SiLU; q and k
  L2-normed (eps inside the sum), q times `dk^-0.5`; `g = exp(-exp(A_log) softplus(a + dt_bias))`,
  `beta = sigmoid(b)`; the gated delta rule with an fp32 128 x 128 state a value head; the output norm is
  `GatedRMSNorm` with a **sigmoid** gate of z (`output_gate_type`; Qwen3.5 uses SiLU); `out_proj`.
- **Sparse attention**: 24 query heads, 2 KV heads, head dim 256; `q_proj` gives [query | gate] a head;
  q and k RMS-normed, RoPE on the first 64 dims (`partial_rotary_factor` 0.25, theta 1e7, non-interleaved
  halves; the mrope sections coincide for text). Output times `sigmoid(gate)`, then `o_proj`.
  The **indexer** (`index_qk_proj`: 4 query heads + 1 key head of 128) keeps every position's raw indexer
  key; blocks of 4 (`indexer_compress_ratio`) are pooled (mean, `k_layernorm`, RoPE at the block's first
  position). A row with more than 512 complete blocks (`indexer_budget` 2048 / 4) scores each complete block
  (sum over the 4 indexer heads of `relu(q . k)`, over `sqrt(128)`), takes the 512 best (ties to the lower
  block) and attends their 2048 keys plus the keys of its incomplete block; rows with fewer blocks attend
  causally. The cache holds bf16 keys and values a position plus the raw and pooled indexer keys.
- **MoE** on every layer: router (bf16, 512 experts) and `shared_expert_gate` as a 513th logit; top 10 with
  softmax over the picked logits; routed SwiGLU experts of width 640 and a shared expert of width 640 times
  `sigmoid(shared gate)`.
- **PLE (n-gram) layer** after layer 1 (`ple_layer_ids` [2]): 16 heads (8 bigram, 8 trigram) hash the row's
  last 2 or 3 tokens of its segment (tokens before the last eos read as eos) with 64-bit wrapping products by
  three seeded multipliers, modulo 16 prime table sizes past 20M; each head reads a 160-wide row of its
  table (the 16 rows make the 2560-wide n-gram embedding). `key_proj` (-> 10,240) and `value_proj` (-> 2560);
  per stream `gate = sigmoid(signed sqrt(norm_key(k) . norm_query(h) / sqrt(2560)))`, `gated = gate * v`,
  then a depthwise causal conv of 4 taps with dilation 3 (`ngram_size`) over `norm_conv(gated)`, SiLU, and
  `h += gated + silu(conv)`. State: the conv's 9-row tail and the last 2 tokens.
- **MTP head** (1 layer, sparse attention + MoE, its own hyper-connections and mixer): input
  `fc_embedding(norm(embed(next)))` broadcast over the streams plus `fc_hidden(norm(stream))` a stream;
  its logits over the shared `lm_head` (upstream restricts the draft head to a draft vocabulary,
  `draft_vocab_default.txt`).

**Upstream's serving contract (Metal)**: one active reply per engine, greedy only (`greedy_only = true`),
prompts in chunks with the MTP head's keys for every prompt row, GPU-side rounds whose draft depth a depth
rule sets, copy drafts from the token history, a prompt cache of kept states (`--prompt-cache-gib`,
`--learn`), two-Mac speed-up mode. Exactness: drafted == serial, resumed == fresh, kept states compared
bit for bit. The HIP engine runs on the lane core like Qwen3.5, so concurrent == solo applies too.

## Checkpoints (Hugging Face, sizes from the Hub API, no weights downloaded)

| Repo | Revision | Files | Size | Format |
| --- | --- | --- | --- | --- |
| `TensorFold/Qwen3.8-Flash-Next-MLX-3bit-MTP` | `1d53b5c6729653c594121764e25e2b261c7a170a` | 18 shards | 90.77 GB (84.54 GiB) | uniform 3-bit, group 32 |
| `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` | `2b170fa6309d5d1ee380b35636075fac7945f286` | 22 shards | 113.21 GB (105.43 GiB) | uniform 4-bit, group 32 |
| `TensorFold/Qwen3.8-Flash-Next-MLX-6bit-MTP` | `45032a193a5d4126ba8534822cbaf8fb30b3e368` | 30 shards | 158.08 GB (147.22 GiB) | uniform 6-bit, group 32 (upstream's) |
| `TensorFold/Qwen3.8-Flash-Next-MLX-2bit-MTP` | `28b109043ec5576f8ad5a3192a66467915c76736` | 20 shards | 80.07 GB | uniform 2-bit, group 32 |
| `TensorFold/Qwen3.8-Flash-Next-MLX-oQ3-MTP` / `-oQ4-MTP` | `993186e7` / `069090c3` | 19 / 22 | 92.5 / 113.3 GB | mixed (746 / 232 per-tensor overrides) |

Each repo also has config.json, the tokenizer files, chat template and `model.safetensors.index.json`.
Routers, norms, conv weights, `A_log` and `dt_bias` are bf16 in every quantization; everything else is
MLX affine at the checkpoint's bits, n-gram tables included. Where the 4-bit checkpoint's bytes go:

| Part | 4-bit | 3-bit |
| --- | --- | --- |
| routed experts (incl. MTP's) | 71.8 GiB | 57.4 GiB |
| n-gram tables (host) | 29.8 GiB | 23.8 GiB |
| DeltaNet | 1.22 GiB | 0.97 GiB |
| hyper-connections | 0.37 GiB | 0.30 GiB |
| embedding, lm_head | 0.37 GiB each | 0.30 GiB each |
| attention q/k/v/o and indexer | 0.36 GiB | 0.29 GiB |
| shared experts, routers, indexers, PLE projections, MTP glue | 0.30 GiB | 0.27 GiB |
| vision tower (not loaded) | 0.84 GiB | 0.84 GiB |

A local alternative to a download: `~/models/q38n-bf16` on par1 is a bf16 `qwen4_exp` checkpoint (336 GB);
the owner's `tools/mlx_requant.py` could produce 4- and 3-bit group-32 checkpoints from it. The published
TensorFold quantizations are uniform and are what upstream's Metal engine is checked against, so they are the
better reference. par1's root filesystem has 155 GB free: the 4-bit (113 GB) or the 3-bit (91 GB) fits, not both.

## Memory

`hip/plan.zig` on this branch counts a rank's bytes from the config alone and matches the three published
checkpoints' text bytes exactly (3-bit 89,875,765,016; 4-bit 112,311,293,016; 6-bit 157,182,349,016). Split:
routed experts by rank (contiguous experts, as Qwen3.5's TP), query heads, value and key heads, `lm_head`
rows by rank; KV heads by rank up to 2 (tp 4: one KV head a rank, each held by two ranks); shared experts on
rank 0; hyper-connections, routers, embedding, indexers, PLE projections and norms on every rank.

Per position a rank holds 13 attention layers (12 and the MTP head's) of K and V (bf16) plus raw and pooled
indexer keys: 30,784 bytes at tp 1, 17,472 at tp 2 and 4. A stream's state (DeltaNet fp32 states and conv
tails, the PLE tail): 116 MB at tp 1, 58 MB at tp 2, 29 MB at tp 4.

| | V620 tp 4, 4-bit | V620 tp 4, 3-bit | W7800 tp 2, 4-bit | W7800 tp 2, 3-bit |
| --- | --- | --- | --- | --- |
| card | 29.98 GiB | 29.98 GiB | 44.98 GiB | 44.98 GiB |
| reserve (`reserveBytes`: max(1 GiB, total/32)) | 1.00 GiB | 1.00 GiB | 1.41 GiB | 1.41 GiB |
| runtime, code objects (estimate) | 0.4 GiB | 0.4 GiB | 0.4 GiB | 0.4 GiB |
| weights (rank 0) | 19.49 GiB | 15.62 GiB | 37.92 GiB | 30.36 GiB |
| scratch (placeholder until the engine counts its own) | 2.0 GiB | 2.0 GiB | 2.0 GiB | 2.0 GiB |
| left for KV and states | ~7.1 GiB | ~11.0 GiB | ~3.3 GiB | ~10.8 GiB |
| one stream's longest window | ~430k | ~670k | ~200k | ~660k |
| streams at a 32k window | ~12 | ~19 | ~5 | ~18 |
| host memory (n-gram tables, mapped once, shared by the ranks) | 29.8 GiB | 23.8 GiB | 29.8 GiB | 23.8 GiB |

With the tables on the device instead (split by shard across ranks), V620 tp 4 at 3-bit still fits (~4.9 GiB
left a rank); 4-bit does not. That mode can come later behind a flag; host tables are the default.

**Admission** (on this branch): `native.zig` reads config.json and the index (every Flash Next tensor named,
`index.admit`), refuses groups other than 32 and a `--tp` that does not split heads, experts and vocabulary
evenly, reads the card's budget through `hip.admission.budget` (free memory, reserve,
`TENSORFOLD_HIP_MEMORY_LIMIT_GB`) and refuses a `--context` that does not fit, naming the window that would.
Mixed groups (W7800 with V620) are already refused by the RCCL join (`MixedGpus`). Once the engine exists the
placeholder scratch becomes the engine's counted scratch, as Qwen3.5's `planBytes` does.

## Ops and kernels

| Op | HIP today | Change | Effort |
| --- | --- | --- | --- |
| MLX affine products at 3/4/6-bit group 32 (all projections, experts) | `tf_affine`, `tf_affine_routed`, dot2/stream/K-parallel tiles, gfx11 WMMA | none; the down projection takes the inject rows stacked under its 320 rows at upload | S |
| embedding rows | `tf_embed_rows` | written into 4 streams (4 launches or a stride argument) | S |
| hyper-connection norm and inject write-back | none | **new** `tf_fn_hc_norm` (done) | done |
| hyper-connection gates (silu, 2 sigmoid) and mix | none | **new** `tf_fn_hc_gates`, `tf_fn_hc_mix` (done) | done |
| DeltaNet conv + SiLU, gate (g, beta), gated delta rule (decode, chunked prefill, plan replay) | `tf_conv_*`, `tf_gdn_gate*`, `gated_delta_tile<128>`, `tf_gdn_*` chunked, `tf_gdn_plan/replay` | none: the kernels take key head = value head / (value heads / key heads), 3 here, as the reference groups them | none |
| DeltaNet output norm with sigmoid gate | `tf_gnorm_silu`, `tf_gnorm_out` | gate argument (done on this branch) | done |
| q/k RMS norm + partial RoPE (256 dims, 64 rotated); indexer q (128, 64) | `tf_qk_rope` | none (widths up to 512) | none |
| KV cache writes | `tf_kv_write`, `tf_kv_write_at`, pages | raw indexer keys as a third cache | S |
| indexer pooled keys, block scores, block selection | none | **new** `tf_fn_idx_pool`, `tf_fn_idx_scores`, `tf_fn_idx_select` (done) | done |
| attention over selected keys (or causal) | `score_keys`/`apply_values`, FA prefill tiles over contiguous keys | **new** `tf_fn_sparse_attention` (parts + merge, done); paged caches later | done (flat cache) |
| attention output gate | `tf_attn_gate` | none | none |
| router (bf16, 512 + shared gate) | `tf_router_*`, `moe_router_kernel` | 513 logits, as Qwen3.5's 257 | none-S |
| top-10 of 512, softmax over picks, shared gate | `tf_moe_select` (up to 1024 experts) | none | none |
| expert routing, grouped products, combine | `tf_moe_route` (seg arrays at 512 experts: 16 segs), routed affine, `moe_combine` | none: 512 experts take 16 segments, 22.5 KB of shared memory | none |
| PLE n-gram ids | host (`hip/ngram.zig`, done) or **new** `tf_fn_ngram_ids` (done) | host tables: gather 16 rows' words/scales/biases a token into a staging table, then `tf_embed_rows` | S-M |
| PLE gate, norm, dilated conv, residual | none | **new** `tf_fn_ple_gate`, `tf_fn_ple_conv` (done) | done |
| final mixer, `lm_head`, argmax / sampling | hc kernels, `tf_affine`, `tf_argmax_rows`, draw | none | none |
| MTP head (fc_embedding, fc_hidden, one layer, mixer) | Qwen3.5's MTP lane path | Flash Next's head input (broadcast + a stream projection) | M |
| tensor parallel reductions | `hip/collective.zig` (RCCL all-reduce, all-gather), `tf_moe_localize` etc. | experts by rank, heads by rank; the hc glue replicated so ranks agree | M |

Kernels follow the math of the Python reference and the rounding points the upstream Metal kernels document;
no code from MLX, vLLM, FLA, llama.cpp or the owner's vllm-rdna trees. None of the new kernels follows an
algorithm from the owner's vllm-rdna folders (the block selection is a plain radix select written here).

## PR slicing (ashhart's order)

1. **flashnext: spec in its own file; pleRef's wrapping multiply** (family, 2 commits). The first lets a
   backend module read `config.zig` without core's files; the second fixes an overflow panic in safe builds
   (the pinned multipliers in `host_test.zig` stay). Tests: the existing ones; the HIP plan's tests reach
   `pleRef` in Debug.
2. **hip: the memory budget in `admission.zig`** (shared, 1 commit). Qwen3.5's `Engine.budget` moves
   unchanged. Test: Qwen3.5's memory refusals unchanged (`tf-qwen35-test` on a card).
3. **hip: Flash Next kernels** (kernels, 1-2 commits: streams + n-gram; indexer + sparse attention). GPU tests
   against host references, each failing when its kernel is broken (mutation checked on gfx1030); rows
   independent of the rows beside them.
4. **hip: small kernel changes**: the sigmoid-gated DeltaNet output norm (on this branch); the embedding into 4
   streams; the raw indexer keys' cache write. Each with its kernel test.
5. **flashnext: HIP admission and routing** (family + `native/hip.zig` registry), plus host n-gram hashing.
   Lands with or right before 6 so the route never refuses everything.
6. **flashnext: HIP engine, one rank**: weights upload from the checkpoint (`core.checkpoint`, `core.quant`,
   names from `index.zig`), the forward (decode window and chunked prefill), the lane backend, greedy and
   sampled draws, prompt cache via the lane core's radix cache. Exactness: concurrent == solo, resumed ==
   fresh, prefill chunking independent. Needs the checkpoint.
7. **flashnext: MTP drafts on HIP**: drafted == serial at every depth.
8. **flashnext: tensor parallelism** over RCCL (tp 2 W7800, tp 4 V620): expert-parallel MoE, head-split
   attention and DeltaNet, replicated hyper-connections and indexer; tp == solo tokens.
9. Optional: device-resident n-gram tables, paged sparse attention, WMMA tiles for the hc products.

## On this branch (`hip-flashnext`, from `qwen35-rccl` merged with `qwen35-structured`)

| Commit | What | Tests |
| --- | --- | --- |
| flashnext: the affine spec in a file of its own | `spec.zig`; `affine.zig` re-exports | Mac `zig build test test-golden` |
| flashnext: pleRef wraps the multipliers' seed arithmetic | `*%` | plan tests in Debug |
| hip: the memory budget in admission.zig | shared budget | Mac tests |
| flashnext: native HIP routing for qwen4_exp behind admission | `families/flashnext/hip.zig`, `hip/plan.zig`, `hip/native.zig`, registry | exact byte totals of 3/4/6-bit; admission verdicts per target |
| hip: Flash Next's kernels | `kernels/hip/flashnext/*.hpp`, `launches/flashnext.zig` | `hip-kernel-test` on gfx1030 (V620): 65/65 |
| flashnext: a window's n-gram rows on the host | `hip/ngram.zig`; plan's table rows | pinned to the Python ids |
| hip: the gated norm's gate is SiLU or sigmoid | `tf_gnorm_out`, `tf_gnorm_silu` take the gate; Qwen3.5 passes SiLU | chain test, both gates; sigmoid against the host |

Not yet run: the kernels on gfx1100 (the W7800s were not available to this work); bf16 activations take the
same code paths with `kind` 2.

## Decisions for the owner

1. Approve a checkpoint download to par1: `TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP` at `2b170fa6`
   (113.2 GB) for the V620 tp 4 target, and/or `TensorFold/Qwen3.8-Flash-Next-MLX-3bit-MTP` at `1d53b5c6`
   (90.8 GB) for the W7800 tp 2 target; par1 has 155 GB free, so one at a time (or requantize
   `~/models/q38n-bf16` with `tools/mlx_requant.py`).
2. The HIP engine reads tensors through `core.checkpoint` (the HIP modules import core as a module), so it
   cannot import `weights.zig`/`affine.zig` (they reach core by path; a file may live in one module only).
   It reuses `config.zig`, `index.zig` and the spec and names its tensors from `index.zig`'s list. The
   alternative, making `weights.zig` generic over the checkpoint type, touches the Metal engine; ask ashhart
   which he prefers before PR 6.
3. n-gram tables in host memory by default (device-resident as an option for 3-bit tp 4).
4. Greedy only at first, as upstream's Flash Next engine; sampling comes with the lane core for free once the
   draws are exact.
