// W4A16 GPTQ decode kernel for gfx1030, ported from vLLM-rdna (opengfx1030/vllm-rdna) q_gemm_rdna2.cu.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "qgemm.hpp"
#include "qgemm_common.hpp"

#include <hip/hip_fp16.h>

#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)

namespace tf {
namespace rocm {

// A thread stages one K element a row and computes 4 columns, 256 K a block.
constexpr int kBlockKn = 256;
constexpr int kThreads = 256;

template <int M_COUNT>
__global__ void __launch_bounds__(kThreads) gptq_decode_kernel(
    const __half* __restrict__ a, const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros, const __half* __restrict__ scales,
    __half* __restrict__ c, const int size_m, const int size_n, const int size_k, const int groups,
    const int zero_offset, const int* __restrict__ g_idx) {
    const int t = threadIdx.x;
    const int offset_n = blockIdx.x * kBlockKn * 4;
    const int offset_m = blockIdx.y * M_COUNT;
    const int offset_k = blockIdx.z * kBlockKn;
    const int end_k = min(offset_k + kBlockKn, size_k);
    const int n = offset_n + t * 4;

    // PAD keeps rows of block_a off one LDS bank.
    constexpr int kLdsPad = 8;
    __shared__ __half block_a[M_COUNT][kBlockKn + kLdsPad];

    if (offset_k + t < end_k) {
#pragma unroll
        for (int m = 0; m < M_COUNT; ++m) {
            __half av;
            if (offset_m + m < size_m) {
                const __half* a_row = a + static_cast<long long>(offset_m + m) * size_k;
                av = g_idx ? a_row[g_idx[offset_k + t]] : a_row[offset_k + t];
            } else {
                av = __float2half_rn(0.0f);  // zero-pad invalid M rows; their write is skipped below
            }
            block_a[m][t] = av;
        }
    }
    __syncthreads();

    if (n >= size_n) return;

    const int groupsize = size_k / groups;
    int group = offset_k / groupsize;
    int nextgroup = (group + 1) * groupsize;

    // qweight is [K / 8, N] packed at dim 0.
    const uint32_t* b_ptr = qweight + static_cast<long long>(offset_k / 8) * size_n + n;

    __half2 z[4][2], y[4][2];
    refresh_group<4>(group, n, qzeros, scales, size_n, zero_offset, z, y);

    float block_c[M_COUNT][4];
#pragma unroll
    for (int m = 0; m < M_COUNT; ++m) {
#pragma unroll
        for (int j = 0; j < 4; ++j) block_c[m][j] = 0.0f;
    }

    // Group transitions are checked every 32 K (one outer step), so group_size >= 32.
    int k = offset_k;
    while (k < end_k) {
        if (k == nextgroup) {
            group++;
            nextgroup += groupsize;
            refresh_group<4>(group, n, qzeros, scales, size_n, zero_offset, z, y);
        }

        // The four weight words are issued up front to overlap the earlier dequant.
        int4 b_w[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) b_w[j] = *reinterpret_cast<const int4*>(b_ptr + j * size_n);
        b_ptr += 4 * size_n;

#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int a_off = (k - offset_k) + 8 * j;
            __half2 dq[4][4];
            dequant4x8(static_cast<uint32_t>(b_w[j].x), dq[0], z[0], y[0]);
            dequant4x8(static_cast<uint32_t>(b_w[j].y), dq[1], z[1], y[1]);
            dequant4x8(static_cast<uint32_t>(b_w[j].z), dq[2], z[2], y[2]);
            dequant4x8(static_cast<uint32_t>(b_w[j].w), dq[3], z[3], y[3]);
#pragma unroll
            for (int m = 0; m < M_COUNT; ++m) {
                const __half* a_ptr = reinterpret_cast<const __half*>(&block_a[m][a_off]);
                block_c[m][0] += dot8(dq[0], a_ptr);
                block_c[m][1] += dot8(dq[1], a_ptr);
                block_c[m][2] += dot8(dq[2], a_ptr);
                block_c[m][3] += dot8(dq[3], a_ptr);
            }
        }
        k += 32;  // 4 weight words * 8 nibbles
    }

    epilogue<M_COUNT>(block_c, offset_m, size_m, size_n, n, c);
}

template <int M_COUNT>
void launch_for_mcount(const Gptq& g, hipStream_t stream) {
    dim3 block(kThreads);
    dim3 grid((g.n + kBlockKn * 4 - 1) / (kBlockKn * 4), (g.m + M_COUNT - 1) / M_COUNT,
              (g.k + kBlockKn - 1) / kBlockKn);
    gptq_decode_kernel<M_COUNT><<<grid, block, 0, stream>>>(
        static_cast<const __half*>(g.a), g.qweight, g.qzeros, static_cast<const __half*>(g.scales),
        static_cast<__half*>(g.out), g.m, g.n, g.k, g.groups, g.zero_offset, g.g_idx);
}

hipError_t launch_gptq_decode(const Gptq& g, hipStream_t stream) {
    if (g.m < 1 || g.n < 1 || g.k < 1 || g.k % 32 != 0 || g.n % 8 != 0 || g.k % g.groups != 0) {
        return hipErrorInvalidValue;
    }
    if (g.k / g.groups < 32) return hipErrorInvalidValue;
    if (g.m == 1) {
        launch_for_mcount<1>(g, stream);
    } else if (g.m <= 3) {
        launch_for_mcount<2>(g, stream);
    } else if (g.m <= 7) {
        launch_for_mcount<4>(g, stream);
    } else {
        launch_for_mcount<8>(g, stream);
    }
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf

#else  // No gfx1030 device pass: keep symbol parity without a schedule.

namespace tf {
namespace rocm {

hipError_t launch_gptq_decode(const Gptq&, hipStream_t) { return hipErrorInvalidValue; }

}  // namespace rocm
}  // namespace tf

#endif
