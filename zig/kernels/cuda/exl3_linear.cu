// y[rows, N] = x[rows, K] @ W, W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh).
// Trellis is strips order [N/128, K/16, 8, words], int16. H is the 128 Sylvester Hadamard / sqrt(128).
// Launch: grid (N/128, rows), block 32. K and N are multiples of 128. rows is 1..128 in the check, any positive count works.

#include <cuda_fp16.h>
#include <stdint.h>

#include "exl3_decode.cuh"

extern "C" __global__ void __launch_bounds__(32) exl3_linear_f16(const __half* __restrict__ x, const int16_t* __restrict__ trellis,
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
