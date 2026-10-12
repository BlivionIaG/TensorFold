# Qwen3.8 Flash Next on HIP

Qwen3.8 Flash Next (`model_type: qwen4_exp`) on TensorFold's native HIP backend, in the `flashnext` family: the plan,
what the `hip-flashnext` branch implements, and how it measures on four Radeon PRO V620s (gfx1030) at tensor
parallelism 4 with the 4-bit MLX checkpoint and on two Radeon PRO W7800s (gfx1100) at tp 2 with the 3-bit one.

## Status

Working end to end at tp 4 on HIP 2-5 of par1 (four V620s, 4-bit) and at tp 2 on HIP 0-1 (two W7800s, 3-bit,
`TensorFold/Qwen3.8-Flash-Next-MLX-3bit-MTP` at `1d53b5c6`), RCCL, served by `tensorfold-native serve`:

- the whole forward: hyper-connections, the n-gram (PLE) layer with its tables in host memory, Gated DeltaNet
  layers, sparse attention with its indexer (block scores, top-512 selection, attention over the chosen keys),
  the 512-expert MoE with its shared expert, the final mixer and the output head;
- MTP drafts from the checkpoint's head (whole on rank 0, its own keys for every prompt row, chains of up to three
  greedy drafts over the family's draft vocabulary, cut under `--mtp-confidence`), and copy drafts from the lane
  core;
- several streams at once (`--parallel`), a round's windows in one forward, replayed from HIP graphs by shape (below);
- the prompt cache (`--prompt-cache-gib`, 16 GiB of host memory by default): stream states at 64-row cuts, every
  rank keeping its own share;
- chat, tool calls, thinking on and off, structured output (the lane core's grammar masks on the HIP draw), greedy
  and sampled replies;
- admission that sizes the streams to the memory the weights leave and refuses what does not fit.

Exactness on the real checkpoints (all on the branch head):

| Check | V620 tp 4, 4-bit | W7800 tp 2, 3-bit |
| --- | --- | --- |
| drafted == serial (`serve_check.py`, greedy and t=0.8) | pass (also through a JSON schema: same token hash) | pass (same) |
| solo == together (2 and 4 streams in shared rounds), long prompt among decoders == alone | pass | pass |
| prompt cache: with == without through turns, regenerate, edit, delete (`rewind.py`) | 16/16 | 16/16 |
| graphs on == graphs off (`tf-flashnext-hip-test --hash`: every step's whole logits row) | identical | identical |
| needle in 7k, 120k and 255k-token prompts (the model's 262,144 window) | found all three | found all three |
| tool call round trip, thinking on/off, JSON schema output (drafts on and off) | valid | valid |
| tp 4 against tp 1 | not comparable: one card cannot hold the weights (74.8 GiB a rank at tp 1); see below | |

**Why tp 4 cannot equal tp 1 bit for bit**: a split projection's ranks each sum their own K range in fp32 and the
shares are added in rank order before rounding (`tf_fn_hc_norm` sums the gathered shares), where one rank sums the
whole K range in its tile's order; the expert-parallel MoE sums its experts' outputs per rank before the gather.
Both are deterministic and do not depend on the rows sharing a call (the shares go through an all-gather, never an
all-reduce whose order follows the message size), so each world size is self-consistent: drafted == serial and
solo == together hold at tp 4. The one-rank engine can still be checked on smaller cards' worth of weights in the
future (a 2-bit checkpoint, 75 GB, still needs tp 4 on V620s).

## Measurements (`bench.py` streamed through the server)

Four V620s, tp 4, 4-bit, before graphs (the rows below are this section's first measurements):

| Workload | Prefill | TTFT | Decode, drafts off | Decode, drafts on |
| --- | --- | --- | --- | --- |
| c=1, 4k-token prompt, 256-token reply | 1,123 tok/s | 3.60 s | 45.0 tok/s (ITL 22.2 ms) | 86.5 tok/s |
| c=1, 8k-token prompt, 128-token reply | 1,009 tok/s | 7.99 s | | 62.7 tok/s |
| c=2, 8k each, 128-token replies | 1,003 tok/s | 8.04 s | 34.0 each, 68.1 total | |
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
| a decode round replayed from graphs (below), V620 | c=1 drafts off 45.0 -> 46.5, on 86.1 -> 87.9; c=4 off 68.0 -> 68.7 |
| the head's broadcast gates sized for a prompt chunk (a bug: past 4 rows it read unrelated memory) | c=1 drafts on 86.3-88.0 -> 92.2 tok/s, 165/210 -> 167/191 drafts kept |

Where a decode step's 20 ms go now (profiled): the affine products 7 ms (the output head's 62k-row slice alone
0.26 ms), RCCL all-gathers 3.5 ms (97 a token, 36 us each), routers and picks 1.5 ms, and about 1,400 launches whose
gaps add some 5 ms. Graphs recover under 1 ms of those on gfx1030 (below): the gaps are mostly the device's own
dispatch, not the host's. Next candidates: a peer-to-peer all-gather for one-row shares (cuts 97 RCCL calls and the
graph pieces between them) and the prefill GEMM tiles on gfx1030.

Two W7800s, tp 2, 3-bit (first pass after the server started, graph captures included):

| Workload | Prefill | TTFT | Decode, drafts off | Decode, drafts on |
| --- | --- | --- | --- | --- |
| c=1, 4k-token prompt, 256-token reply | 1,085 tok/s | 3.72 s | 55.6 tok/s (ITL 18.0 ms) | 70.3 tok/s |
| c=4, 4k each | | 10.4 s | 18.2 each, 72.9 total | 20.2 each, 80.9 total |
| c=1, 16k-token prompt, 128-token reply | 921 tok/s | 17.5 s | | 78.1 tok/s |
| c=1, 7k / 120k / 255k-token prompt (needles) | 980 / 803 / 666 tok/s | 7.2 / 149.5 / 382.8 s | | 62 / 65 / 57 tok/s |

Load 30-39 s, 32.1 GiB on rank 0 (8k window) of the W7800's 48 GiB; 2 streams at the full 262,144-token window
(4.79 GiB each), `--parallel 4` there refused with the number that fits. The gfx11 matrix-core tile is used for
prompt products: `--policy matrix=off` drops a 7k prompt from 1,146 to 952 tok/s. Graphs gain more on gfx1100: the
model check decodes 58.5 tok/s with them and 48.2 without.

Both rigs on the branch head, warm (a server's second pass of the same runs: graphs recorded, 4k-token prompts,
256-token replies), decode in tok/s:

| | c=1 drafts on | c=1 drafts off (ITL) | c=4 drafts on, total | c=4 drafts off, total | prefill, TTFT (c=1) |
| --- | --- | --- | --- | --- | --- |
| V620 tp 4, 4-bit, before graphs and the head fix | 86.1 | 45.0 (22.2 ms) | 89.5 | 68.0 | 1,123 tok/s, 3.60 s |
| V620 tp 4, 4-bit, head | 93.8 | 46.7 (21.4 ms) | 93.7 | 69.8 | 1,115 tok/s, 3.62 s |
| W7800 tp 2, 3-bit, head | 77.2 | 55.7 (18.0 ms) | 78.1 | 72.9 | 1,085 tok/s, 3.72 s |

### Graphs for the decode round

A round of windows (plain one-row rounds and drafted verify windows alike, several streams together) is recorded
the second time its shape is met and replayed after (`Engine.choose` / `Engine.round`, keyed by each stream's cache
slot and rows). Every host input of the round (token ids, positions, the attention's ends and complete blocks, and the
n-gram rows hashed and gathered from the host tables) is written into the pinned staging buffer before the launch and
sent up in one copy that the graph holds (`forward.stage` / `forward.send`); no host work happens mid-forward. The
launches that took a position as a value now read it from the device (value and indexer-key writes, the pool of the
blocks a row completes, the block scores over all the blocks a stream's caches hold), so one graph serves every
position. Streams take their caches from slots kept for the server's life, so a new request reuses its slot's
graphs. RCCL calls take ~8 ms each to record inside a graph (750 ms a step), so a round is recorded as graphs cut at
its collectives (`hip/graph_split.zig`): the all-gathers run between the pieces' replays, recording takes 21-47 ms
a shape. Every rank makes the same choice in the same order (rank 0's pick goes with the round's message), a rank
that cannot record makes all of them run eagerly (`--policy graph_fail=R` tests it), and `--policy graphs=off`
turns them off. The MTP head's chain stays eager: its launches are a few dozen a draft (under 0.1 ms of gaps), and
its sparse layer still takes positions from the host.

| | graphs off | graphs on |
| --- | --- | --- |
| V620, model check, 300 greedy tokens (3 runs each) | 48.9 tok/s | 50.8 tok/s |
| W7800, model check, 64 greedy tokens | 48.2 tok/s | 58.5 tok/s |


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
by their bytes); the 4-bit one runs on the V620s and the 3-bit one on the W7800s. The 3-bit checkpoint stores the
shared expert's router row as 3-bit codes, which `core.quant`'s dequantization now reads (any width from 2 to 8).


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
On the W7800 pair the 3-bit checkpoint holds 32.1 GiB on rank 0 and leaves room for two streams at the full window.

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
| indexer pooled keys (from a block start, or the blocks rows at device positions complete), scores, top-512 | **new** `tf_fn_idx_pool`, `tf_fn_idx_scores`, `tf_fn_idx_select` |
| attention over chosen keys, merge, output gate | **new** `tf_fn_sparse_parts6/12`, `tf_fn_sparse_merge`; `tf_attn_gate` |
| router, top 10 of 512 + shared gate, route, combine | `tf_router_*`, `tf_select_decode`, `tf_moe_route`, `moe_combine` (hip.moe) |
| n-gram ids, rows, gate, dilated conv | host hashing (`hip/ngram.zig`) and gather (`hip/tables.zig`), `tf_embed_rows`; **new** `tf_fn_ple_gate`, `tf_fn_ple_conv` (`tf_fn_ngram_ids` for device tables) |
| logits gather, draws (greedy, keyed sampling, grammar masks) | RCCL all-gather, `hip_serve` Drawer |

Kernels are written for TensorFold from the Python reference's math and the rounding points the upstream Metal
kernels document; nothing is taken from MLX, vLLM, FLA, llama.cpp or the owner's vllm-rdna trees, and no new kernel
follows an algorithm from those trees. A scan of every new and changed source line against the owner's
`gfx1030_optimized` folder (356,262 files, this round's lines included) finds no shared 6-line window, exact or with identifiers and numbers
blanked.

## PR slicing (ashhart's order)

Each slice is a branch with only its own commits, stacked on the one before it:

1. **`core-quant-any-width`** (on `main`): `core/quant`'s dequantization of any 2- to 8-bit width.
2. **`flashnext-tensors`**, then **`flashnext-pack-reader`** (Metal side): the tensor description; the pack builder
   reading through `core/quant` with the family's generic checkpoint tables.
3. **`qwen35-engine-graphs`** (commits for the Qwen3.5 engine PR): the graph cache in the hip module, generic over
   the replay; graphs cut at the collectives; Qwen3.5's rounds recorded that way. **`qwen35-rccl-graphs`**: the
   RCCL commit on top, and Qwen3.5's rounds replaying graphs at any world size.
4. **`flashnext-base`**: a merge of the above with the structured-output slice, the base of the Flash Next HIP stack.
5. **`flashnext-hip-shared`**: the spec in its own file, pleRef's wrapping multiply, and the shared moves (the memory
   budget into `hip/admission.zig`, expert stacking into `core/quant/experts.zig`, the MoE run and tp reduce into the
   hip module, the draw into `hip_serve`).
6. **`flashnext-hip-kernels-shared`**: the route's block scan, the merged pick on 512 threads, the exact MoE plan.
7. **`flashnext-hip-kernels`**: Flash Next's kernels with their GPU tests, and the gated norm's sigmoid gate.
8. **`flashnext-hip-admission`**: native routing for `qwen4_exp` behind admission, the host n-gram hashing.
9. **`flashnext-hip-engine`**: the forward on one device, then serving (lane backend, MTP drafts, prompt reuse).
10. **`flashnext-hip-tp`**: tensor parallelism (weights by rank, shares summed in the stream norm, followers).
11. **`flashnext-hip-graphs`**: rounds replayed from split graphs; this document's status and measurements.

## One description of the checkpoint's tensors

The family read the checkpoint in three places with their own spellings: admission (`index.zig`), the Metal pack
builder (`pack.zig`, which turns the checkpoint into the packs the Metal engine maps) and the HIP loader; a fourth,
`weights.zig`'s reader, was unused. `tensors.zig` now describes the model's tensors once: every name (layer stems,
hyper-connections, DeltaNet, sparse attention and indexer, MoE, n-gram layer and tables, MTP head), the shape of
every projection from the config (pinned by a test to the published checkpoint's headers), the rule that decides
whether centred norms are stored as gamma or gamma - 1, and the centred scale. Admission, the pack builder and the
HIP loader all read through it; the HIP loader keeps only what is the backend's own: device upload, tensor-parallel
cuts (`core.quant`), and the checks of each projection against the description's shape. `tensors.zig` imports
nothing but the config, so both the Metal side (which reaches core by path) and the HIP module (which imports core)
can hold it. The Metal path's behaviour is unchanged: its pack, admission and host tests pass untouched, and the
three packs its full-build tests write hash the same (SHA-256) before and after. What a full unification would add:
the pack builder's own readers (`qlinear`, `denseRows`, the conv transpose) moved behind the same generic reader the
HIP loader uses (`core.quant` over a checkpoint), which needs the Metal side to import `core` as a module instead of
by path (a build change for the whole Metal engine, outside this family).

## On this branch

Branch `hip-flashnext` on the fork (`BlivionIaG/TensorFold`), from `qwen35-rccl` merged with `qwen35-structured`.
Every commit is the repository's configured author, without AI trailers; `zig fmt` clean; `lean_check.py` 534
problems as on the base. Tests: Mac `zig build test test-golden -Dcpu=apple_m1`; Linux `zig build test` from the
pushed commit; `hip-kernel-test` on gfx1030 and gfx1100 (all kernel tests, Flash Next's included); the model checks above
(`zig/tests/flashnext_hip`: `tf-flashnext-hip-test generate`, `rewind.py`, `bench.py`; `zig/tests/qwen35/serve_check.py`
against the Flash Next server).

## Remaining work

- Family settings not ported: `--learn` (learned prompt states on disk) needs every rank to write and read its share
  in step (the store's peer hooks); speed-up mode is the two-Mac engine's and HIP takes `--tp` instead.
- The MTP head drafts one stream at a time (a shared round's heads could batch).
- Performance: see the candidates above; the prefill GEMM on gfx1030 runs at about a tenth of its peak.
- Device-resident n-gram tables (a flag for a 3-bit tp 4, which has room) and paged sparse caches.

## Decisions for the owner

1. The tensor description is the smallest step to one loader (see above); moving the Metal side onto `core` as a
   module would let the pack builder share the HIP loader's reader too. Ask ashhart whether he wants that.
2. Prompt states live in host memory (16 GiB by default): the cards hold no room to spare at tp 4.
3. Graphs are on by default at any world size (`--policy graphs=off` turns them off); Qwen3.5's engine keeps them
   off under tp unless asked, because of RCCL's recording cost, which the split graphs avoid.
