# ROCm implementation

`--backend rocm` serves Qwen3.5 / Qwen3.8 dense MLX affine checkpoints on AMD Radeon GPUs, behind the same torch
server as CUDA. `auto` picks it where `/dev/kfd` exists and the family has a ROCm engine (`rocm_engine`). Text only.

## Setup

Install a ROCm build of PyTorch, `hipcc` (ROCm) and `ninja`, then TensorFold:

```bash
python -m pip install git+https://github.com/ashhart/TensorFold.git
tensorfold serve mlx-community/Qwen3.5-9B-MLX-8bit --backend rocm --name local-model --host 0.0.0.0 --port 8080
```

The first start compiles the HIP kernels for the visible GPU only (one gfx target, wave32); later starts reuse them.
An extension rebuilds when any of its sources, the headers beside them or its flags change. Pin a card with
`HIP_VISIBLE_DEVICES`.

| | |
| --- | --- |
| GPUs measured | Radeon PRO W7800 (gfx1100, RDNA3, BF16 activations), Radeon PRO V620 (gfx1030, RDNA2, FP16 activations) |
| GPUs built for | gfx1030-1036, gfx1100-1103, gfx1150-1153, gfx1200-1201; RDNA1 is refused |
| Weights | MLX affine 2, 3, 4, 5, 6 and 8 bits, groups 32, 64 and 128; scales and biases fp32, bf16 or fp16 as stored |
| Checkpoints served | `mlx-community/Qwen3.5-0.8B-MLX-8bit`, `mlx-community/Qwen3.5-9B-MLX-8bit`, `mlx-community/Qwen3.5-9B-6bit`, `Vontra/Qwen3.8-27B-MLX-4bit` |
| Tested stacks | W7800: ROCm 7.2, torch 2.12.0+rocm7.2. V620: ROCm 7.14, torch 2.12.0+rocm7.14.0 |

The server takes one request at a time on ROCm. Chat completions, completions, the Responses API, `response_format`
grammars, tool calls, keyed sampling and the prefix cache work as on CUDA.

## Kernels

The weights stay packed on the GPU; each kernel unpacks the codes it reads. Every output is the group sum of
`x * code` in fp32, scaled and biased once per group, with no atomics, and a row's bits do not depend on how many
rows share the launch.

- RDNA2 (`affine_dot2.hip`), FP16 activations and `v_dot2_f32_f16`, every width through three tiles: 1-8 rows,
  32 columns a block whose 8 waves take one group each a round and fold in group order; 9-63 rows, a 128x32 tile;
  64 rows and more, a 128x128 tile with 8x8 outputs a thread.
- RDNA3 / RDNA 3.5 / RDNA4 (`affine_wmma.hip`), BF16 16x16x16 WMMA through rocWMMA, with paired and grouped launches
  for 8-bit projections that share an input. Widths other than 8 use the generic WMMA tile and are much slower.
- Attention: a Flash-Attention-2 HIP prefill tile, a split-over-keys decode walk, and a Triton FA2 tile (BF16 WMMA
  on gfx1100, FP16 on gfx1030) at the lengths where it measured faster. The KV cache is the activation dtype.
- The Gated DeltaNet recurrence of the linear-attention layers, bit-exact against its PyTorch reference.

## Tensor parallel

One process a rank, RCCL between them. `--tp 2`, `4` or `8` are ROCm only. Heads, KV heads and MLP columns are
split per rank; o, out and down are split by input groups and their fp32 shares are summed before each residual
add; the vocabulary is split for the head. Heads, KV heads and the vocabulary must split evenly: the 0.8B runs at
tp=2 at most, the 9B and 27B at tp=4.

On one host, give each rank its own card. With every card visible, rank r takes card r:

```bash
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 1 --master 127.0.0.1 &
tensorfold serve Vontra/Qwen3.8-27B-MLX-4bit --backend rocm --tp 2 --rank 0 --master 127.0.0.1 --name local-model
```

Rank 0 serves HTTP; the other ranks follow it. `--p2p` opts in to RCCL peer-to-peer. tp has run on one machine
(eight V620s on PCIe); tp=8 is untested.

MTP drafting is not supported on ROCm yet: a checkpoint's MTP head loads, but pass `--no-drafts`.

## Measurements

`python -m tensorfold.rocm.bench MODEL_DIR [PROMPT GENERATED CONCURRENCY] [--runs N] [--tp N --rank R --master
ADDR]`. One run a cell, one shared wall for the concurrent requests, end tokens ignored. Prefill tok/s / decode
tok/s.

V620, `Vontra/Qwen3.8-27B-MLX-4bit`, 1,024-token prompt, 512 tokens generated:

| | 1 request | 8 requests |
| --- | --- | --- |
| tp=1 (`ed4463e`) | 252 / 9.88 | 242 / 41.8 |
| tp=2 (`6bb7ed8`) | 288 / 8.62 | 286 / 40.2 |
| tp=4 (`6bb7ed8`) | 591 / 13.1 | 593 / 92.3 |

V620, `mlx-community/Qwen3.5-9B-MLX-8bit`, same load (`6bb7ed8`): tp=2 793 / 15.3 and 1,387 / 125; tp=4 1,552 / 25.1
and 1,641 / 173.

W7800, tp=1 (`0b04605`), 1,024 / 512 and 16,384 / 1,024:

| | 1k, 1 request | 1k, 8 requests | 16k, 1 request | 16k, 8 requests |
| --- | --- | --- | --- | --- |
| 0.8B 8-bit | 3,592 / 49.2 | 3,479 / 312 | 3,228 / 47.4 | 3,445 / 209 |
| 9B 8-bit | 259 / 9.0 | 340 / 60.4 | 256 / 8.49 | 328 / 46.7 |
| 9B 6-bit | 116 / 3.58 | 112 / 25.7 | 110 / 3.52 | 109 / 23.0 |
| 27B 4-bit | 32 / 1.28 | 32 / 9.0 | 31 / 1.26 | 31 / 8.06 |

On the V620 the 27B at 16,384 tokens with 8 requests needs more than its 30 GiB at tp=1 and runs at tp=2.
tp=2 does not speed up 27B decode at one request yet: the two smallest linear-attention projections are
latency-bound at any width, and the per-layer all-reduces wait on rank skew.
