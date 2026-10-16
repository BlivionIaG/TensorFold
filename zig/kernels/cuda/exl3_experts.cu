// Routed experts. Gate/up and down stay one identity projection each, not a fused gate+up pair.
// The check compares that identity path to the Zig dequantize. expert_slot is line * span for the one
// matrix this launch passes. span is tiles_k * tiles_n * tile_words, not the 1,901,213-byte residency line.
// ids are line indices, token-major int32[tokens * topk], read every launch so a graph replay can swap the buffer.
// Trellis is [line, strips...], suh [line, K], svh [line, N]. y is [tokens * topk, N] in that slot order.
// SwiGLU is its own kernel between those projections. It does not replace project_nblock.
// combine has no mask. The host stores a finite fp16 zero in a CPU slot's y, and a 0 weight. slot_ptrs and hit_mask are not read.
// Launch gate/up and down: grid (N/128, tokens * topk), block 32. K and N are multiples of 128.
// Launch SwiGLU: grid ((n + 127) / 128, rows), block 128. rows is tokens * topk, n is the gate/up N.
// combine: out[token, n] = sum_s fp32(weights[token, s]) * fp32(y[token, s, n]), fma from 0, then fp16.
// Launch combine: grid ((N + 127) / 128, tokens), block 128.

#include <cuda_fp16.h>
#include <stdint.h>

#include "exl3_decode.cuh"

namespace {

__device__ __forceinline__ void expert_slot(const __half* __restrict__ x, const int* __restrict__ ids,
                                            const int16_t* __restrict__ trellis, const __half* __restrict__ suh,
                                            const __half* __restrict__ svh, __half* __restrict__ y, int tokens, int K, int N,
                                            int n_experts_in_batch, int topk, int halves, int codebook, float* xhat, float* acc,
                                            float* tile) {
    const int nblock = static_cast<int>(blockIdx.x);
    const int slot = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    const int nslots = tokens * topk;
    if (slot >= nslots || lane >= 32 || nblock >= (N >> 7)) return;
    const int expert = ids[slot];
    if (expert < 0 || expert >= n_experts_in_batch) return;
    const int token = slot / topk;
    const int tiles_k = K >> 4;
    const int tiles_n = N >> 4;
    const size_t span = static_cast<size_t>(tiles_k) * static_cast<size_t>(tiles_n) * static_cast<size_t>(tf_exl3::tile_words(halves));
    tf_exl3::project_nblock(x + static_cast<size_t>(token) * static_cast<size_t>(K), trellis + static_cast<size_t>(expert) * span,
                            suh + static_cast<size_t>(expert) * static_cast<size_t>(K),
                            svh + static_cast<size_t>(expert) * static_cast<size_t>(N),
                            y + static_cast<size_t>(slot) * static_cast<size_t>(N), K, N, halves, codebook, nblock, lane, xhat, acc,
                            tile);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(32) exl3_experts_gate_up_f16(const __half* __restrict__ x, const int* __restrict__ ids,
                                                                          const int16_t* __restrict__ trellis,
                                                                          const __half* __restrict__ suh, const __half* __restrict__ svh,
                                                                          __half* __restrict__ y, int tokens, int K, int N,
                                                                          int n_experts_in_batch, int topk, int halves, int codebook) {
    __shared__ float xhat[128];
    __shared__ float acc[128];
    __shared__ float tile[256];
    expert_slot(x, ids, trellis, suh, svh, y, tokens, K, N, n_experts_in_batch, topk, halves, codebook, xhat, acc, tile);
}

// fp32 silu(gate) * up, then fp16 for down. Unclamped: this pack's limit is +inf.
// The identity projections stay in the kernels above and below. No slot_ptrs, no hit_mask.
extern "C" __global__ void exl3_experts_swiglu_f16(const __half* __restrict__ gate, const __half* __restrict__ up,
                                                   __half* __restrict__ y, int rows, int n) {
    const int row = static_cast<int>(blockIdx.y);
    const int col = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (row >= rows || col >= n) return;
    const size_t i = static_cast<size_t>(row) * static_cast<size_t>(n) + static_cast<size_t>(col);
    const float g = __half2float(gate[i]);
    const float u = __half2float(up[i]);
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, expf(-g)));
    y[i] = __float2half_rn(__fmul_rn(s, u));
}

extern "C" __global__ void __launch_bounds__(32) exl3_experts_down_f16(const __half* __restrict__ x, const int* __restrict__ ids,
                                                                       const int16_t* __restrict__ trellis, const __half* __restrict__ suh,
                                                                       const __half* __restrict__ svh, __half* __restrict__ y, int tokens,
                                                                       int K, int N, int n_experts_in_batch, int topk, int halves,
                                                                       int codebook) {
    __shared__ float xhat[128];
    __shared__ float acc[128];
    __shared__ float tile[256];
    expert_slot(x, ids, trellis, suh, svh, y, tokens, K, N, n_experts_in_batch, topk, halves, codebook, xhat, acc, tile);
}

extern "C" __global__ void exl3_experts_combine_f16(const __half* __restrict__ y, const float* __restrict__ weights,
                                                    __half* __restrict__ out, int tokens, int N, int topk) {
    const int token = static_cast<int>(blockIdx.y);
    const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (token >= tokens || n >= N) return;
    float acc = 0.f;
    for (int s = 0; s < topk; ++s) {
        const float yv = __half2float(y[(static_cast<size_t>(token * topk + s) * static_cast<size_t>(N)) + static_cast<size_t>(n)]);
        acc = fmaf(weights[token * topk + s], yv, acc);
    }
    out[static_cast<size_t>(token) * static_cast<size_t>(N) + static_cast<size_t>(n)] = __float2half_rn(acc);
}
