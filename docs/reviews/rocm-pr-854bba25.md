# Review: ROCm MLX affine path at 854bba25

Review of rocm-pr @ 854bba25 (ashhart/TensorFold#389). Remarks only. Not a code fix.

1. Wrong result, decode graph. `_graph_forward` runs the decode step while capturing, then calls `graph.replay()` on that same token before sampling. Gated-delta conv and recurrence update state in place, so the capture token applies that state twice and later replays continue from the bad state. KV `index_copy_` at the same slot hides it. `tests/rocm/test_graph.py` skips unless `TENSORFOLD_GOLDEN_MODEL` is set, so a green CPU run does not cover this.

2. Wrong result, MTP residual. In `src/tensorfold/rocm/mtp.py`, the head norms `e_proj + h_proj` and then adds attention onto that norm. The residual is not the pre-norm sum. Drafts can be the wrong function while "drafted = serial" still passes, because verify emits the serial token.

3. Crash risk at tp 2/4/8. The captured decode all-reduces on RCCL, then sampling broadcasts on the same communicator outside the graph. Capture plus the extra replay runs that all-reduce twice on the capture token. Same-stream order might save it. This is the usual graph desync. P2P is left to RCCL on a discrete GPU (`_resolve_p2p` forces it only for an integrated multi-die part). The PR's own table has tp=2 decode slower than tp=1.

4. Wrong result on the MLX W4A16 MoE path. Activations are fp16, `qgemm.moe` demands bf16, then the kernel converts back to fp16 before `v_dot2_f32_f16`. That drops mantissa the dense path never drops. The dot itself is legal on gfx1030. The kernel dequants and then calls `__builtin_amdgcn_fdot2`. It never issues `sdot4`, so it is not a packed-int path. Hugging Face GPTQ exports still do not load, as the PR says. `qgemm.matmul` requires fp16 and `qgemm.moe` requires bf16.

5. Perf trap. Decode attention with a device position scores the full KV allocation, not `pos+1`. Unused keys become `-inf`, and every replay walks the whole window. That is dense causal, not a sparse/QSA score.

6. Silent fallback. On gfx1030 (`TENSORFOLD_RDNA_WMMA=0`), `launch_affine` on a non-fp16 activation falls through to `affine_gemv_kernel`: scalar fma, one thread per output, no `fdot2`. The Python check only rejects fp16 on a WMMA part. It never rejects bf16 here, so a missed cast is the reference kernel, not the decode tile.

7. Graph warmup. The fast FP16 tile fills a constant LUT with `hipStreamSynchronize` on the launch stream. If that first fill happens inside decode graph capture, the capture is illegal and the engine drops to eager. Warm the LUT before capture.

Also still true, and already in the PR body: the server runs one request at a time, MTP verifies one draft with a full target forward so drafting does not speed decode up, and there are no checks on the branch.

HIP:

- `ensure_byte_lut` calls `hipStreamSynchronize` on the launch stream, so a capture that is the first fast-tile launch is illegal. Warm that LUT before capture.
- The gfx1030 W4A16 kernel dequants and then uses `fdot2`, not `sdot4`.

Arch:

- Scoring the full KV allocation is dense causal, not QSA, and QSA keeps its own heap.
- The MTP path that norms `e_proj + h_proj` and then adds attention is not the gated residual or mHC contract. Mixers stay leftover dense.

On gfx1030, a non-fp16 activation falls through to `affine_gemv_kernel`: scalar fma, one thread per output, not `fdot2`.

Pin note: ROCm 7.14.0. This is not a dest transplant.
