// Routed experts, identity epilogue: no SiLU and no clamp. Gate/up and down are the same projection;
// one launch is one matrix, not a fused gate+up pair. The check compares that identity path to the Zig dequantize.
// ids are token-major int32[tokens * topk] and are read on every launch, so a graph replay can swap the buffer.
// Trellis is packed [expert, strips...], suh [expert, K], svh [expert, N]. y is [tokens * topk, N] in that slot order.
// Launch gate/up and down: grid (N/128, tokens * topk), block 32. K and N are multiples of 128.
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
