# GLM-5.3-Flash on the M5 Ultra, before and after #871

Apple M5 Ultra (64-core GPU), 256 GB, macOS 27.0. Zig 0.17.0, `-Dcpu=apple_m1 -Dmetal-std=metal4.0`. #871 changes `core/affine_mm`'s code-tile loads on GPUs with tensor units (8-byte loads, `frag_get_tg`); GLM's prompt matmuls use `affine_mm`.

| Directory | What |
| --- | --- |
| `matmul/` | `tf-mm-bench M K N old new`: `affine_mm.metal` at 87c3851a (before #871) and at 2ceb6403 (#871), random 4-bit matrices, the dense entry at 64 x 64 tiles, 128 threads; `*_bf16.txt` bf16 outputs, `*_f321.txt` fp32 outputs. Every file: "differ from first 0/N". |
| `served/` | `Vontra/GLM-5.3-Flash-MLX-4bit-MTP` (snapshot 76add2a3) served by the native engine built at 2a5f091b (#868 head, `g868`) and f364a81b (#871 head, `g871`), alternated: g868, g871, g871, g868. Default load limit (the MTP head stays on disk; copy drafts on), `--context 32768 --prompt-cache-gib 0`, port 8091. Each session: `bench_openai.py` (256 tokens, 2 reps), `bench_concurrent.py --levels 1,4 --alone --serial`, `two_turn.py` (an 8k chat), `vmmap` footprint. `glmbench.sh` ran it. |
