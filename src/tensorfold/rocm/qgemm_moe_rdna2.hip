// Grouped-expert W4A16 GPTQ for RDNA2 (gfx1030), on TensorFold's MoE plan.
// Ported from vLLM-rdna (opengfx1030/vllm-rdna, branch rdna_extras)
// csrc/rocm/moe_q_gemm_rdna2.cu, re-shaped to TensorFold's contract.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// TensorFold's plan gives one item per expert run: (expert, first, count) over a
// members list of pair rows. Each item owns distinct output rows, so unlike the
// vLLM kernel this writes them directly: no atomics, no K-split, and the pair's
// activation folds into the epilogue. Activations are bf16 (TensorFold's MoE
// dtype) and widen exactly to fp16 for the v_dot2 dot.

#include "qgemm.hpp"
#include "qgemm_common.hpp"

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>

#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)

namespace tf {
namespace rocm {

constexpr int kThreads = 256;
constexpr int kBlockKn = 256;
constexpr int kLdsPad = 8;

__device__ __forceinline__ float bf16_rn(float v) { return static_cast<float>(hip_bfloat16(v)); }

__device__ __forceinline__ float relu2_bf(float a) {
    const float u = fmaxf(bf16_rn(a), 0.0f);
    return u * u;
}

// bf16(bf16(silu(g)) * u) on bf16 g and u, clipped first when limit > 0.
__device__ __forceinline__ float swiglu_bf(float g, float u, float limit) {
    float gv = bf16_rn(g);
    float uv = bf16_rn(u);
    if (limit > 0.0f) {
        gv = fminf(gv, limit);
        uv = fminf(fmaxf(uv, -limit), limit);
    }
    return bf16_rn(gv / (1.0f + expf(-gv))) * uv;
}

// mats is 2 for a fused gate+up (SwiGLU) and 1 otherwise. EPI 0 writes fp32 rows
// (down), 1 bf16 relu^2, 2 bf16 SwiGLU.
template <int MATS, int EPI, int BLOCK_M>
__global__ __launch_bounds__(kThreads) void moe_gptq_kernel(
    const hip_bfloat16* __restrict__ a, void* __restrict__ out, const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros, const __half* __restrict__ scales,
    const int32_t* __restrict__ items, const int32_t* __restrict__ members, const int size_n,
    const int size_k, const int groups, const int slots, const int experts, const int zero_offset,
    const float limit) {
    const int item = blockIdx.x;
    const int expert = items[item * 3 + 0];
    const int first = items[item * 3 + 1];
    const int count = items[item * 3 + 2];
    if (count <= 0 || expert < 0) return;

    const int t = threadIdx.x;
    const int n = blockIdx.y * kBlockKn * 4 + t * 4;
    const bool active = (n < size_n);

    const int groupsize = size_k / groups;
    const int words_row = size_k / 8;
    const long long mat_stride = static_cast<long long>(experts) * words_row * size_n;
    const long long scale_stride = static_cast<long long>(experts) * groups * size_n;
    const long long zero_stride = static_cast<long long>(experts) * groups * (size_n / 8);

    for (int base = 0; base < count; base += BLOCK_M) {
        const int chunk = min(BLOCK_M, count - base);
        float block_c[MATS][BLOCK_M][4];
#pragma unroll
        for (int mm = 0; mm < MATS; ++mm) {
#pragma unroll
            for (int m = 0; m < BLOCK_M; ++m) {
#pragma unroll
                for (int j = 0; j < 4; ++j) block_c[mm][m][j] = 0.0f;
            }
        }

        __shared__ __half block_a[BLOCK_M][kBlockKn + kLdsPad];

        for (int k0 = 0; k0 < size_k; k0 += kBlockKn) {
            const int end_k = min(k0 + kBlockKn, size_k);
#pragma unroll 1
            for (int idx = t; idx < BLOCK_M * kBlockKn; idx += kThreads) {
                const int m = idx / kBlockKn;
                const int kk = idx % kBlockKn;
                if (m < chunk && k0 + kk < end_k) {
                    // members holds a pair id; its activation row is the pair over the slots.
                    const int32_t pair = members[first + base + m];
                    const long long row = pair / slots;
                    block_a[m][kk] = __float2half_rn(static_cast<float>(a[row * size_k + k0 + kk]));
                }
            }
            __syncthreads();

            // An inactive thread keeps taking part in the barriers the K loop runs, or the
            // survivors would sync against a partial block; it just skips the work.
            if (!active) {
                __syncthreads();
                continue;
            }
            const int group0 = k0 / groupsize;
            __half2 z[MATS][4][2], y[MATS][4][2];
#pragma unroll
            for (int mm = 0; mm < MATS; ++mm) {
                const long long we = mm * mat_stride + static_cast<long long>(expert) * words_row * size_n;
                const long long se = mm * scale_stride + static_cast<long long>(expert) * groups * size_n;
                const long long ze = mm * zero_stride + static_cast<long long>(expert) * groups * (size_n / 8);
                refresh_group<4>(group0, n, qzeros + ze, scales + se, size_n, zero_offset, z[mm], y[mm]);
            }
            int group = group0;
            int nextgroup = (group + 1) * groupsize;

            const uint32_t* b_ptr[MATS];
#pragma unroll
            for (int mm = 0; mm < MATS; ++mm) {
                b_ptr[mm] = qweight + (mm * mat_stride + static_cast<long long>(expert) * words_row * size_n) +
                            static_cast<long long>(k0 / 8) * size_n + n;
            }

            for (int k = k0; k < end_k; k += 32) {
                if (k == nextgroup) {
                    group++;
                    nextgroup += groupsize;
#pragma unroll
                    for (int mm = 0; mm < MATS; ++mm) {
                        const long long se = mm * scale_stride + static_cast<long long>(expert) * groups * size_n;
                        const long long ze = mm * zero_stride + static_cast<long long>(expert) * groups * (size_n / 8);
                        refresh_group<4>(group, n, qzeros + ze, scales + se, size_n, zero_offset, z[mm], y[mm]);
                    }
                }
#pragma unroll
                for (int mm = 0; mm < MATS; ++mm) {
                    int4 b_w[4];
#pragma unroll
                    for (int j = 0; j < 4; ++j) b_w[j] = *reinterpret_cast<const int4*>(b_ptr[mm] + j * size_n);
                    b_ptr[mm] += 4 * size_n;

#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int a_off = (k - k0) + 8 * j;
                        __half2 dq[4][4];
                        dequant4x8(static_cast<uint32_t>(b_w[j].x), dq[0], z[mm][0], y[mm][0]);
                        dequant4x8(static_cast<uint32_t>(b_w[j].y), dq[1], z[mm][1], y[mm][1]);
                        dequant4x8(static_cast<uint32_t>(b_w[j].z), dq[2], z[mm][2], y[mm][2]);
                        dequant4x8(static_cast<uint32_t>(b_w[j].w), dq[3], z[mm][3], y[mm][3]);
#pragma unroll
                        for (int m = 0; m < BLOCK_M; ++m) {
                            if (m >= chunk) continue;
                            const __half* a_ptr = reinterpret_cast<const __half*>(&block_a[m][a_off]);
                            block_c[mm][m][0] += dot8(dq[0], a_ptr);
                            block_c[mm][m][1] += dot8(dq[1], a_ptr);
                            block_c[mm][m][2] += dot8(dq[2], a_ptr);
                            block_c[mm][m][3] += dot8(dq[3], a_ptr);
                        }
                    }
                }
            }
            __syncthreads();
        }

#pragma unroll
        for (int m = 0; m < BLOCK_M; ++m) {
            if (m >= chunk) continue;
            if (!active) continue;
            const long long row = members[first + base + m];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const long long col = static_cast<long long>(row) * size_n + n + j;
                if (EPI == 0) {
                    static_cast<float*>(out)[col] = block_c[0][m][j];
                } else {
                    float v;
                    if (EPI == 1) {
                        v = relu2_bf(block_c[0][m][j]);
                    } else {
                        v = swiglu_bf(block_c[0][m][j], block_c[1][m][j], limit);
                    }
                    static_cast<hip_bfloat16*>(out)[col] = hip_bfloat16(v);
                }
            }
        }
    }
}

#else  // No gfx1030 device pass: keep symbol parity.

template <int MATS, int EPI, int BLOCK_M>
__global__ __launch_bounds__(kThreads) void moe_gptq_kernel(
    const hip_bfloat16*, void*, const uint32_t*, const uint32_t*, const __half*, const int32_t*,
    const int32_t*, const int, const int, const int, const int, const int, const int, const float) {}

#endif

template <int MATS, int EPI, int BLOCK_M>
void launch_moe_for(const MoeGptq& g, int experts, hipStream_t stream) {
    dim3 block(kThreads);
    dim3 grid(g.items_count, (g.n + kBlockKn * 4 - 1) / (kBlockKn * 4));
    moe_gptq_kernel<MATS, EPI, BLOCK_M><<<grid, block, 0, stream>>>(
        static_cast<const hip_bfloat16*>(g.a), g.out, g.qweight, g.qzeros,
        static_cast<const __half*>(g.scales), g.items, g.members, g.n, g.k, g.groups, g.slots, experts,
        g.zero_offset, g.limit);
}

template <int MATS, int EPI>
void launch_moe_blocks(const MoeGptq& g, int block_m, int experts, hipStream_t stream) {
    switch (block_m) {
        case 1: launch_moe_for<MATS, EPI, 1>(g, experts, stream); break;
        case 2: launch_moe_for<MATS, EPI, 2>(g, experts, stream); break;
        case 4: launch_moe_for<MATS, EPI, 4>(g, experts, stream); break;
        case 8: launch_moe_for<MATS, EPI, 8>(g, experts, stream); break;
    }
}

hipError_t launch_moe_gptq(const MoeGptq& g, int experts, int epi, int block_m, hipStream_t stream) {
    if (g.n < 1 || g.k < 1 || g.k % 32 != 0 || g.n % 8 != 0 || g.k % g.groups != 0) return hipErrorInvalidValue;
    if (g.k / g.groups < 32 || g.items_count < 1) return hipErrorInvalidValue;
    if (block_m != 1 && block_m != 2 && block_m != 4 && block_m != 8) return hipErrorInvalidValue;
    switch (epi) {
        case 0: launch_moe_blocks<1, 0>(g, block_m, experts, stream); break;
        case 1: launch_moe_blocks<1, 1>(g, block_m, experts, stream); break;
        case 2: launch_moe_blocks<2, 2>(g, block_m, experts, stream); break;
        default: return hipErrorInvalidValue;
    }
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf
