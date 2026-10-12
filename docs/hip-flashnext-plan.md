# Qwen3.8 Flash Next on HIP

Qwen3.8 Flash Next (`model_type: qwen4_exp`) on TensorFold's native HIP backend, in the `flashnext` family: the plan,
what the `hip-flashnext` branch implements, and how it measures on four Radeon PRO V620s (gfx1030) at tensor
parallelism 4 with the 4-bit MLX checkpoint.

## Status

Working end to end at tp 4 on HIP 2-5 of par1 (four V620s, RCCL), served by `tensorfold-native serve`:

- the whole forward: hyper-connections, the n-gram (PLE) layer with its tables in host memory, Gated DeltaNet
  layers, sparse attention with its indexer (block scores, top-512 selection, attention over the chosen keys),
  the 512-expert MoE with its shared expert, the final mixer and the output head;
- MTP drafts from the checkpoint's head (whole on rank 0, its own keys for every prompt row, chains of up to three
  greedy drafts over the family's draft vocabulary, cut under `--mtp-confidence`), and copy drafts from the lane
  core;
- several streams at once (`--parallel`), a round's windows in one forward;
- the prompt cache (`--prompt-cache-gib`, 16 GiB of host memory by default): stream states at 64-row cuts, every
  rank keeping its own share;
- chat, tool calls, thinking on and off, structured output (the lane core's grammar masks on the HIP draw), greedy
  and sampled replies;
- admission that sizes the streams to the memory the weights leave and refuses what does not fit.

Exactness on the real checkpoint (all on the branch head, four V620s):

| Check | Result |
| --- | --- |
| drafted == serial (`serve_check.py`, greedy and t=0.8) | pass (also through a structured-output grammar: same token hash) |
| solo == together (2 and 4 streams in shared rounds), long prompt among decoders == alone | pass, greedy and sampled |
| prompt cache: chat with cache == without, through turns, regenerate, edit and delete (`rewind.py`) | 16/16, 10,240 of ~10,280 prompt tokens resumed |
| needle in 7k, 120k and 255k-token prompts (sparse attention, the model's 262,144 window) | found all three |
| tool call round trip, thinking on/off, JSON schema output (drafts on and off) | valid |
| tp 4 against tp 1 | not comparable: one card cannot hold the weights (74.8 GiB a rank at tp 1); see below |

**Why tp 4 cannot equal tp 1 bit for bit**: a split projection's ranks each sum their own K range in fp32 and the
shares are added in rank order before rounding (`tf_fn_hc_norm` sums the gathered shares), where one rank sums the
whole K range in its tile's order; the expert-parallel MoE sums its experts' outputs per rank before the gather.
Both are deterministic and do not depend on the rows sharing a call (the shares go through an all-gather, never an
all-reduce whose order follows the message size), so each world size is self-consistent: drafted == serial and
solo == together hold at tp 4. The one-rank engine can still be checked on smaller cards' worth of weights in the
future (a 2-bit checkpoint, 75 GB, still needs tp 4 on V620s).

## Measurements (four V620s, tp 4, 4-bit, `bench.py` streamed through the server)

| Workload | Prefill | TTFT | Decode, drafts off | Decode, drafts on |
| --- | --- | --- | --- | --- |
| c=1, 4k-token prompt, 256-token reply | 1,123 tok/s | 3.60 s | 45.0 tok/s (ITL 22.2 ms) | 86.5 tok/s |
| c=1, 16k-token prompt, 128-token reply | 972 tok/s | 16.6 s | 44.6 tok/s | 68.2 tok/s |
| c=2, 4k each | 694 tok/s each (both prompts together) | 5.8 s | 29.4 each, 58.9 total | 43.8 each, 87.6 total |
| c=4, 4k each | | 10.3 s | 16.9 each, 67.5 total | 22.5 each, 89.8 total |
| c=1, 120k-token prompt | 734 tok/s | 163.5 s | 36.2 tok/s (drafts on) | |
| c=1, 255k-token prompt | 606 tok/s | 420.8 s | 41.6 tok/s (drafts on) | |

Load: 40 s (warm page cache), 21.6-22.3 GiB on rank 0 (weights, the MTP head, the scratch for 1,024-row prompt
chunks). A resumed 16k prompt answers in 272 ms.

**What was tuned** (each measured, each keeping drafted == serial and solo == together):

| Change | Effect |
| --- | --- |
| a tp block output gathered and summed inside the hyper-connection norm (one launch for gather-add-cast) | greedy decode 26.8 -> 39.5 tok/s with the next |
| the MoE route's expert prefix by a block scan (thread 0 summed 513 experts in turn: 32 us a layer) | (above) |
| the merged MoE pick on 512 threads; the stream norm on 1,024 threads a block | 39.5 -> 43.5 |
| attention parts: a kv head's query heads in one block, 32 keys staged in shared memory | decode 43.5 -> 43.7; neutral on prefill |
| a shared round's windows in one forward | c=2 drafts off 31 -> 54 tok/s total, c=4 62 |
| a window's DeltaNet conv, split, norms and gates in one launch (Qwen3.5's merged conv); grouped input products | 43.8 -> 48.2 |
| the MoE plan sized by the rows' own pairs (a routed launch walked 198 items for one row's 11) | 48.2 -> 50.0 |
| idle prompts in 1,024-row chunks (512 while others decode) | prefill 1,040 -> 1,262 tok/s on a 7k prompt |

Where a decode step's 20 ms go now (profiled): the affine products 7 ms (the output head's 62k-row slice alone
0.26 ms), RCCL all-gathers 3.5 ms (97 a token, 36 us each), routers and picks 1.5 ms, and about 1,400 launches whose
gaps add some 5 ms. Next candidates: a peer-to-peer all-gather for one-row shares, graphs for the decode step (the
step's host-side inputs already go through one pinned staging buffer), and the prefill GEMM tiles on gfx1030.

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

## Checkpoints (Hugging Face, sizes from the Hub API)

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

The 4-bit checkpoint at `2b170fa6` is on par1 (`~/models/Qwen3.8-Flash-Next-MLX-4bit-MTP`, approved by the owner) and is
what every run below used. For a 3-bit run on the W7800 pair (left out for now): `TensorFold/Qwen3.8-Flash-Next-MLX-3bit-MTP`
at `1d53b5c6` (90.8 GB, uniform 3-bit group 32) needs no conversion; the alternative is requantizing the local bf16
checkpoint (`~/models/q38n-bf16`, 336 GB) with the owner's `tools/mlx_requant.py` to MLX affine 3-bit group 32, every
tensor the 4-bit one quantizes (n-gram tables included), the routers and norms left bf16. par1 has about 48 GB free,
so either needs space first. The engine reads any of 2- to 8-bit group-32 MLX checkpoints (`hip/plan.zig` admits them
by their bytes); only the 4-bit one has been run.


## Memory (measured on the 4-bit checkpoint)

`hip/plan.zig` counts a rank's bytes from the config and matches the published 3-, 4- and 6-bit checkpoints' text
bytes exactly. At tp 4 a rank holds its 128 routed experts (rank 0 also the shared ones and the whole MTP head),
6 query heads, 4 key and 12 value heads, one kv head (each held by two ranks) and a quarter of the output head's
vocabulary; hyper-connections, routers, embedding, indexers and PLE projections are on every rank. The n-gram tables
(29.8 GiB) stay in the mapped checkpoint in host memory: a token gathers 16 rows of 100 bytes.

| V620 tp 4, 4-bit (rank 0) | |
| --- | --- |
| card | 29.98 GiB; reserve 1 GiB (`reserveBytes`) |
| weights, MTP head, scratch for 1,024-row chunks | 21.6-22.3 GiB |
| a position (13 attention layers of K and V, raw and pooled indexer keys; the head's) | 17.5 KB + 2.4 KB |
| a stream's state (DeltaNet states and conv tails, n-gram tail, window snapshots) | about 0.15 GiB |
| a stream at 16k / 32k / 262k positions | 0.42 / 0.75 / 4.65 GiB |
| admitted at once | 4 streams at 32k; 1 at the model's 262,144 |

Admission reads the card's budget (`hip.admission.budget`: free memory, the reserve,
`TENSORFOLD_HIP_MEMORY_LIMIT_GB`) before loading and refuses a `--context` that one stream cannot hold beside the
rank's weights, naming the window that would fit; after loading it counts each stream's bytes at the window and
admits as many as the memory left holds (`--parallel N` fixed and too large is refused with the number that fits).
The W7800 pair is left out by the owner's choice; by the same count the 4-bit checkpoint would hold 37.9 GiB a rank
there (a ~200k window) and the 3-bit one 30.4 GiB, but neither is run.

## Ops and kernels (as implemented)

| Op | HIP kernel |
| --- | --- |
| MLX affine products (all projections, routed and shared experts), grouped products in decode | `tf_affine`, `tf_affine_routed`, the dot2/stream/K-parallel tiles (gfx11 WMMA where present) |
| embedding into the 4 streams | `tf_embed_rows` over 4 ids a token |
| hyper-connection norm and write-back (tp shares summed inside), gates, mix | **new** `tf_fn_hc_norm`, `tf_fn_hc_gates`, `tf_fn_hc_mix` |
| MTP input norm over all 10,240 values | **new** `tf_fn_rms_wide` |
| DeltaNet conv, split, q/k norms, gates (windows) | `tf_conv_split` (Qwen3.5's merged conv) |
| DeltaNet conv, gates (prompts), delta rule | `tf_conv_prefill`, `tf_gdn_gate_prefill`, `gated_delta_tile` (windows, per-row states), chunked `tf_gdn_*` (prompts) |
| DeltaNet output norm, sigmoid gate | `tf_gnorm_out` with the new gate argument |
| q/k norms and partial RoPE, cache writes | `tf_qk_rope`, `tf_kv_write` |
| indexer pooled keys, scores, top-512 selection | **new** `tf_fn_idx_pool`, `tf_fn_idx_scores`, `tf_fn_idx_select` |
| attention over chosen keys, merge, output gate | **new** `tf_fn_sparse_parts6/12`, `tf_fn_sparse_merge`; `tf_attn_gate` |
| router, top 10 of 512 + shared gate, route, combine | `tf_router_*`, `tf_select_decode`, `tf_moe_route`, `moe_combine` (hip.moe) |
| n-gram ids, rows, gate, dilated conv | host hashing (`hip/ngram.zig`) and gather (`hip/tables.zig`), `tf_embed_rows`; **new** `tf_fn_ple_gate`, `tf_fn_ple_conv` (`tf_fn_ngram_ids` for device tables) |
| logits gather, draws (greedy, keyed sampling, grammar masks) | RCCL all-gather, `hip_serve` Drawer |

Kernels are written for TensorFold from the Python reference's math and the rounding points the upstream Metal
kernels document; nothing is taken from MLX, vLLM, FLA, llama.cpp or the owner's vllm-rdna trees, and no new kernel
follows an algorithm from those trees. A scan of every new and changed source line against the owner's
`gfx1030_optimized` folder (350,868 files) finds no shared 6-line window, exact or with identifiers and numbers
blanked.

## PR slicing (ashhart's order), as the branch's commits group

1. **Family groundwork**: the spec in its own file; pleRef's wrapping multiply (a safe-build panic).
2. **Shared moves** (each its own PR, behaviour unchanged): the memory budget into `hip/admission.zig`; expert
   stacking into `core/quant/experts.zig`; the MoE run and tp reduce into the hip module; the draw into `hip_serve`.
3. **Shared kernel changes** (each with its kernel test): the gated norm's sigmoid gate; the route's block scan;
   the merged pick on 512 threads; an exact MoE plan option.
4. **Flash Next kernels**: streams, n-gram, indexer, sparse attention, wide norm (GPU tests against host
   references, each failing when its kernel is broken).
5. **flashnext: HIP admission and routing** with the host n-gram hashing.
6. **flashnext: the HIP forward** (weights by rank, caches, tables) and **serving** (lane backend, MTP drafts,
   tp followers), then **prompt reuse**, then the batched windows and the tuning commits.

## On this branch

Branch `hip-flashnext` on the fork (`BlivionIaG/TensorFold`), from `qwen35-rccl` merged with `qwen35-structured`.
Every commit is the repository's configured author, without AI trailers; `zig fmt` clean; `lean_check.py` 534
problems as on the base. Tests: Mac `zig build test test-golden -Dcpu=apple_m1`; Linux `zig build test` from the
pushed commit; `hip-kernel-test` on gfx1030 (all kernel tests, Flash Next's included); the model checks above
(`zig/tests/flashnext_hip`: `tf-flashnext-hip-test generate`, `rewind.py`, `bench.py`; `zig/tests/qwen35/serve_check.py`
against the Flash Next server).

## Remaining work

- Not run on gfx1100 (the W7800s were not available to this work); bf16 activations take the same kernels.
- Family settings not ported: `--learn` (learned prompt states on disk) needs every rank to write and read its share
  in step (the store's peer hooks); speed-up mode is the two-Mac engine's and HIP takes `--tp` instead.
- The MTP head drafts one stream at a time (a shared round's heads could batch).
- Performance: see the candidates above; the prefill GEMM on gfx1030 runs at about a tenth of its peak.
- Device-resident n-gram tables (a flag for a 3-bit tp 4, which has room) and paged sparse caches.

## Decisions for the owner

1. The HIP loader reads tensors through `core.checkpoint` / `core.quant` and reuses the family's `config.zig`,
   `index.zig` names and spec; `weights.zig` and `affine.zig` reach core by path, which a module importing core
   cannot hold. Ask ashhart whether he prefers `weights.zig` made generic over the checkpoint type instead.
2. Prompt states live in host memory (16 GiB by default): the cards hold no room to spare at tp 4.
3. The W7800 pair: a 3-bit checkpoint (published, 90.8 GB) once disk space allows.
