# ROCm

`--backend rocm` serves Qwen3.5 / Qwen3.8 dense and Qwen3.6-35B-A3B MoE MLX affine checkpoints on AMD Radeon GPUs,
behind the same torch server as CUDA. `auto` picks it where `/dev/kfd` exists and the family has a ROCm engine
(`rocm_engine`). Text only.

## Setup

Install a ROCm build of PyTorch, `hipcc` (ROCm) and `ninja`, then TensorFold:

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP --backend rocm --name local-model --host 0.0.0.0 --port 8080
```

The first start compiles the HIP kernels for the visible GPU only (one gfx target, wave32); later starts reuse them.
An extension rebuilds when any of its sources, the headers beside them or its flags change. Pin a card with
`HIP_VISIBLE_DEVICES`.

| | |
| --- | --- |
| GPUs measured | Radeon PRO V620 (gfx1030, RDNA2, FP16 activations), Radeon PRO W7800 (gfx1100, RDNA3, BF16 activations) |
| GPUs built for | gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200-1201; RDNA1 is refused |
| Weights | MLX affine 2, 3, 4, 5, 6 and 8 bits, groups 32, 64 and 128, mixed widths per tensor as the config names them; scales and biases fp32, bf16 or fp16 as stored |
| Tested stacks | V620: ROCm 7.14, torch 2.12.0+rocm7.14.0. W7800: ROCm 7.2, torch 2.12.0+rocm7.2 |

Checkpoints measured below: `mlx-community/Qwen3.5-0.8B-MLX-8bit`, `mlx-community/Qwen3.5-4B-3bit`,
`mlx-community/Qwen3.5-9B-MLX-4bit` with the head of `mlx-community/Qwen3.5-9B-MTP-4bit` beside it as
`mtp.safetensors`, `mlx-community/Qwen3.5-9B-6bit`, `mlx-community/Qwen3.5-9B-MLX-8bit`,
`TensorFold/Qwen3.8-27B-MLX-4bit`, `leonsarmiento/Qwen3.8-27B-3bit-mtp-mlx` (3 bits, embeddings at 4, an
unquantized MTP `fc`) and `TensorFold/Qwen3.6-35B-A3B-MLX-4bit-MTP`.

The server takes one request at a time on ROCm. Chat completions, completions, the Responses API, `response_format`
grammars, tool calls, keyed sampling and the prefix cache work as on CUDA.

## Kernels

The weights stay packed on the GPU; each kernel unpacks the codes it reads. Every output is the group sum of
`x * code` in fp32, scaled and biased once per group, with no atomics, and a row's bits do not depend on how many
rows share the launch.

- Projections run the dot2 tiles (`affine_tiles.hip`, `affine_dot2.hip`) at every width: FP16 activations with
  `v_dot2_f32_f16` on RDNA2, BF16 with `v_dot2_f32_bf16` on gfx11 / gfx12. One row takes a tile whose lanes read a
  weight row contiguously, 2-8 rows a column tile, longer prompts a 128x128 GEMM tile. All of them keep one pair chain
  and one group fold, so a row's bits are the same alone, in a batch or in a prefill.
- MoE experts run the same tiles over a routing plan: one launch per expert projection for every routed pair.
- `TENSORFOLD_ROCM_SCHEDULE=wmma` runs gfx11's WMMA tiles instead, at every row count. They are slower than the dot2
  tiles today and are kept for later work.
- Attention: a split-over-keys decode walk, and one prefill kernel for every prefill row whatever its span (a Triton
  FA2 tile on gfx1030 when Triton is installed, the HIP FA2 tile otherwise), so a resumed prompt has a fresh one's
  bits. The KV cache is the activation dtype.
- The Gated DeltaNet recurrence of the linear-attention layers, bit-exact against its PyTorch reference.

## Decode

A request's one-token decode step is captured as a HIP graph after its first step and replayed, on every rank
(`TENSORFOLD_GRAPH=0` keeps it eager). Decode attention and RoPE read the position on the device, so a replay gives
the eager step's bits.

## MTP drafting

A checkpoint's MTP head (its `mtp.*` tensors, with or without the `language_model.` prefix, or a side
`mtp*.safetensors`) drafts 4 tokens a round; `--no-drafts` turns it off. Each draft is verified with the serial
step's own sampling key, so a drafted reply is the serial reply token for token, greedy or sampled, at any tp.
Verification is one main forward a draft, so drafting does not speed decode up yet.

## Tensor parallel

One process a rank, RCCL between them. `--tp 2`, `4` or `8` are ROCm only. Heads and MLP columns are split per rank;
o, out and down are split by input groups and their fp32 shares are summed before each residual add; the vocabulary is
split for the head. With fewer KV heads than ranks each KV head is kept by the ranks whose query heads read it. A MoE
layer splits by expert: each rank holds `E / tp` routed experts (the shared one on rank 0), every rank routes every
token, and the ranks' fp32 shares are summed like a down projection.

On one host, give each rank its own card. With every card visible, rank r takes card r:

```bash
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 1 --master 127.0.0.1 &
tensorfold serve TensorFold/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 0 --master 127.0.0.1 --name local-model
```

Rank 0 serves HTTP; the other ranks follow it. Two ranks on one card are refused. `--p2p` / `--no-p2p` force RCCL
peer-to-peer on or off; unset, RCCL decides.

## W4A16 (work in progress)

GPTQ / AWQ W4A16 experts run through grouped int4 kernels on gfx1030, for exports in the MLX layout (stacked
`switch_mlp` experts, an affine embedding), on one rank. Hugging Face GPTQ / AWQ exports (`quantization_config`,
per-expert tensors, unquantized dense layers) are refused at load.

## Measurements

`python -m tensorfold.rocm.bench MODEL_DIR 1024 256 1 --served [--mtp N] [--tp N --rank R --master ADDR]` times the
engine `tensorfold serve` runs, one request at a time: prefill is the prompt over the time to the first token,
decode the other tokens over the time after it, median of two runs. `... 1024 256 8` (without `--served`) is eight
requests in one batched generate.

**V620 (gfx1030), up to 8 cards**, served engine, 1,024-token prompt, 256 tokens, prefill / decode tok/s (MTP: the checkpoint's head drafting 2 a round; batched: 8 requests in one generate)

| Model | tp=1 | tp=2 | tp=4 | tp=8 | tp=1 MTP 2 | tp=2 MTP 2 | tp=4 MTP 2 | tp=8 MTP 2 | tp=1 batched x8 | tp=2 batched x8 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.8B 8-bit | 3,933 / 127 | 4,735 / 77.0 | 4,744 / 101 | 3,902 / 67.0 | - | - | - | - | 6,326 / 316 | 9,915 / 160 |
| 4B 3-bit | 1,185 / 51.6 | 1,724 / 51.1 | 1,779 / 52.4 | 1,392 / 40.6 | - | - | - | - | 1,497 / 212 | 2,535 / 219 |
| 9B 4-bit + MTP head | 697 / 28.7 | 1,079 / 39.2 | 1,208 / 44.7 | 1,200 / 37.3 | 685 / 25.5 | 1,089 / 35.2 | 1,190 / 38.7 | 1,202 / 34.5 | 822 / 113 | 1,434 / 171 |
| 9B 6-bit | 694 / 28.3 | 1,092 / 37.7 | 1,193 / 42.5 | 1,195 / 36.2 | - | - | - | - | 796 / 113 | 1,417 / 177 |
| 9B 8-bit | 634 / 19.5 | 1,000 / 31.7 | 1,139 / 38.3 | 1,193 / 35.7 | - | - | - | - | 754 / 78.7 | 1,323 / 124 |
| 27B 4-bit | 243 / 12.4 | 411 / 15.8 | 511 / 18.2 | 401 / 16.5 | - | - | - | - | 250 / 43.6 | 454 / 74.4 |
| 27B 3-bit + MTP head | 256 / 12.4 | 469 / 16.0 | 590 / 18.5 | 630 / 16.4 | re-measuring | re-measuring | re-measuring | re-measuring | 255 / 44.8 | 447 / 73.3 |
| 35B-A3B 4-bit + MTP head | 819 / 58.7 | 964 / 36.7 | 1,296 / 36.0 | 1,243 / 28.5 | 846 / 50.9 | 1,207 / 31.4 | 1,312 / 31.5 | 1,329 / 25.8 | 1,409 / 130 | 2,009 / 101 |

Mode gates (graphs off/on x serial/drafted, greedy and t=0.7, identical replies): 24/25 pass
- The failing cell is `Qwen3.6-35B-A3B-GPTQ-Int4`: it does not load (a Hugging Face GPTQ export; the loader refuses
  it with `ValueError: Hugging Face GPTQ / AWQ exports (quantization_config) are not served on ROCm`).

**W7800 (gfx1100), up to 2 cards**, served engine, 1,024-token prompt, 256 tokens, prefill / decode tok/s (MTP: the checkpoint's head drafting 2 a round; batched: 8 requests in one generate)

| Model | tp=1 | tp=2 | tp=1 MTP 2 | tp=2 MTP 2 | tp=1 batched x8 | tp=2 batched x8 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.8B 8-bit | re-measuring | re-measuring | re-measuring | re-measuring | re-measuring | re-measuring |
| 4B 3-bit | 1,642 / 38.0 | 2,321 / 42.0 | - | - | 1,627 / 224 | 2,536 / 237 |
| 9B 4-bit + MTP head | 895 / 35.6 | 1,438 / 46.5 | 903 / 31.9 | 1,434 / 42.3 | 841 / 135 | 1,471 / 205 |
| 9B 6-bit | 918 / 33.0 | 1,465 / 42.8 | - | - | 929 / 136 | 1,495 / 210 |
| 9B 8-bit | re-measuring | re-measuring | re-measuring | re-measuring | re-measuring | re-measuring |
| 27B 4-bit | 263 / 12.9 | 452 / 17.0 | - | - | 269 / 47.8 | 453 / 79.3 |
| 27B 3-bit + MTP head | 273 / 10.9 | 470 / 13.9 | re-measuring | re-measuring | 278 / 48.4 | 473 / 81.1 |
| 35B-A3B 4-bit + MTP head | 1,238 / 60.1 | 1,759 / 38.3 | 1,235 / 53.0 | 1,752 / 35.0 | 1,612 / 143 | 2,059 / 146 |

Mode gates (graphs off/on x serial/drafted, greedy and t=0.7, identical replies): 16/16 pass

Being re-measured: the 8-bit W7800 rows (taken on the WMMA tiles) and the 27B 3-bit MTP cells (taken before its
unquantized `fc` loaded).
