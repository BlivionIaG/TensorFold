// y[rows, N] = x[rows, K] @ W, W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh).
// Trellis is strips order [N/128, K/16, 8, words], int16. H is the 128 Sylvester Hadamard / sqrt(128).
// exl3_linear_f16_ref: the original one-warp kernel (grid (N/128, rows), block 32), kept for the bench and equality checks.
// exl3_linear_f16: same numerics, bit for bit, block kFastThreads (project_nblock_fast).
// Launch: grid (N/128, rows), block 32 (ref) or 512 (fast). K and N are multiples of 128. rows is 1..128 in the check, any positive count works.

#include <cuda_fp16.h>
#include <stdint.h>

#include "exl3_decode.cuh"

extern "C" __global__ void __launch_bounds__(32) exl3_linear_f16_ref(const __half* __restrict__ x, const int16_t* __restrict__ trellis,
                                                                 const __half* __restrict__ suh, const __half* __restrict__ svh,
                                                                 __half* __restrict__ y, int rows, int K, int N, int halves,
                                                                 int codebook) {
    const int nblock = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    __shared__ float xhat[128];
    __shared__ float acc[128];
    __shared__ float tile[256];
    if (row >= rows || lane >= 32 || nblock >= (N >> 7)) return;
    tf_exl3::project_nblock(x + static_cast<size_t>(row) * static_cast<size_t>(K), trellis, suh, svh,
                            y + static_cast<size_t>(row) * static_cast<size_t>(N), K, N, halves, codebook, nblock, lane, xhat,
                            acc, tile);
}

extern "C" __global__ void __launch_bounds__(tf_exl3::kFastThreads) exl3_linear_f16(
    const __half* __restrict__ x, const int16_t* __restrict__ trellis, const __half* __restrict__ suh,
    const __half* __restrict__ svh, __half* __restrict__ y, int rows, int K, int N, int halves, int codebook) {
    const int nblock = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    __shared__ tf_exl3::FastShared sm;
    if (row >= rows || nblock >= (N >> 7)) return;
    tf_exl3::project_nblock_fast(x + static_cast<size_t>(row) * static_cast<size_t>(K), trellis, suh, svh,
                                 y + static_cast<size_t>(row) * static_cast<size_t>(N), K, N, halves, codebook, nblock, sm);
}

// Exhaustive check of the fp32 mul1 codebook against the binary64 reference: out[s] = 1 when equal.
extern "C" __global__ void exl3_mul1_probe(uint32_t* bad) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= 65536u) return;
    if (tf_exl3::mul1_bits_fast(s) != tf_exl3::codebook_bits(tf_exl3::kMul1, s)) atomicAdd(bad, 1u);
}
