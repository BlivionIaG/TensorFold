// W4A16 GPTQ prefill kernel for gfx1030, ported from vLLM-rdna (opengfx1030/vllm-rdna) q_gemm_rdna2_prefill.cu.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "qgemm.hpp"
#include "qgemm_common.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include <hip/hip_fp16.h>

#if defined(__gfx1030__) || !defined(__HIP_DEVICE_COMPILE__)

namespace tf {
namespace rocm {

// N_PER_THREAD 4 shares the pk4 epilogue; LDS_PAD 0 keeps the wide-M A tile inside 64 KiB of LDS.
template <int Threads_, int NPerThread_, int KStep_, int MTile_, int LdsPad_>
struct Config {
    static constexpr int kThreads = Threads_;
    static constexpr int kNPerThread = NPerThread_;
    static constexpr int kNTile = Threads_ * NPerThread_;
    static constexpr int kKStep = KStep_;
    static constexpr int kMTile = MTile_;
    static constexpr int kLdsPad = LdsPad_;
};

using ConfigV1 = Config<512, 4, 32, 8, 8>;   // small M
using ConfigA = Config<256, 4, 32, 16, 0>;   // general prefill, large N
using ConfigC = Config<128, 4, 32, 16, 0>;   // small N (N_TILE = 512)

template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::kThreads) void gemm_static_kernel(
    const __half* __restrict__ a, const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros, const __half* __restrict__ scales, __half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ g_idx, const int split_k) {
    constexpr int kPerSplit = K_PER_SPLIT;
    constexpr int kThreads = Config::kThreads;
    constexpr int kNPerThread = Config::kNPerThread;
    constexpr int kNTile = Config::kNTile;
    constexpr int kKStep = Config::kKStep;
    constexpr int kMTile = Config::kMTile;
    constexpr int kLdsPad = Config::kLdsPad;

    const int t = threadIdx.x;
    const int n = blockIdx.x * kNTile + t * kNPerThread;
    const int m_tile = blockIdx.y * kMTile;
    const bool active = (n < size_n);

    const int k_split = blockIdx.z;
    const int k_start = k_split * kPerSplit;
    const int k_end = (k_split == split_k - 1) ? size_k : (k_start + kPerSplit);

    const int groupsize = size_k / groups;
    int group = k_start / groupsize;
    int nextgroup = (group + 1) * groupsize;

    __half2 z[kNPerThread][2], y[kNPerThread][2];
    if (active) {
        refresh_group<kNPerThread>(group, n, qzeros, scales, size_n, zero_offset, z, y);
    }

    float block_c[kMTile][kNPerThread];
#pragma unroll
    for (int m = 0; m < kMTile; ++m) {
#pragma unroll
        for (int j = 0; j < kNPerThread; ++j) block_c[m][j] = 0.0f;
    }

    __shared__ __half block_a[kMTile][kPerSplit + kLdsPad];
    if (g_idx) {
#pragma unroll 1
        for (int idx = t; idx < kMTile * kPerSplit; idx += kThreads) {
            const int m = idx / kPerSplit;
            const int kk = idx % kPerSplit;
            const int m_row = m_tile + m;
            if (m_row < size_m) block_a[m][kk] = (a + static_cast<long long>(m_row) * size_k)[g_idx[k_start + kk]];
        }
    } else {
#pragma unroll 1
        for (int idx = t; idx < (kMTile * kPerSplit) / 4; idx += kThreads) {
            const int quad = idx % (kPerSplit / 4);
            const int m = idx / (kPerSplit / 4);
            const int m_row = m_tile + m;
            if (m_row < size_m) {
                const float2 a4 =
                    *reinterpret_cast<const float2*>(a + static_cast<long long>(m_row) * size_k + k_start + quad * 4);
                *reinterpret_cast<float2*>(&block_a[m][quad * 4]) = a4;
            }
        }
    }
    __syncthreads();

    const uint32_t* b_ptr = qweight + static_cast<long long>(k_start / 8) * size_n + n;

    int k = k_start;
    if (active) {
        while (k < k_end) {
            int4 b_prefetch[kKStep / 8];
#pragma unroll
            for (int j = 0; j < kKStep / 8; ++j) b_prefetch[j] = *reinterpret_cast<const int4*>(b_ptr + j * size_n);
            b_ptr += (kKStep / 8) * size_n;

#pragma unroll
            for (int j = 0; j < kKStep / 8; ++j) {
                // A K_STEP may cross groups, so the (block-uniform) refresh fires mid-block.
                if (k + 8 * j == nextgroup) {
                    group++;
                    nextgroup += groupsize;
                    refresh_group<kNPerThread>(group, n, qzeros, scales, size_n, zero_offset, z, y);
                }
                const int a_off = 8 * j;
                __half2 dq[kNPerThread][4];
                uint32_t w[kNPerThread];
                w[0] = static_cast<uint32_t>(b_prefetch[j].x);
                w[1] = static_cast<uint32_t>(b_prefetch[j].y);
                w[2] = static_cast<uint32_t>(b_prefetch[j].z);
                w[3] = static_cast<uint32_t>(b_prefetch[j].w);
#pragma unroll
                for (int col = 0; col < kNPerThread; ++col) dequant4x8(w[col], dq[col], z[col], y[col]);

#pragma unroll
                for (int m = 0; m < kMTile; ++m) {
                    const int m_row = m_tile + m;
                    if (m_row >= size_m) continue;
                    const __half* a_window = &block_a[m][(k - k_start) + a_off];
#pragma unroll
                    for (int col = 0; col < kNPerThread; ++col) block_c[m][col] += dot8(dq[col], a_window);
                }
            }
            k += kKStep;
        }
        epilogue<kMTile>(block_c, m_tile, size_m, size_n, n, c);
    }
}

// Runtime LDS stride. Always works, at a small address-compute cost.
template <typename Config>
__global__ __launch_bounds__(Config::kThreads) void gemm_dynamic_kernel(
    const __half* __restrict__ a, const uint32_t* __restrict__ qweight,
    const uint32_t* __restrict__ qzeros, const __half* __restrict__ scales, __half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups, const int zero_offset,
    const int* __restrict__ g_idx, const int split_k) {
    constexpr int kThreads = Config::kThreads;
    constexpr int kNPerThread = Config::kNPerThread;
    constexpr int kNTile = Config::kNTile;
    constexpr int kKStep = Config::kKStep;
    constexpr int kMTile = Config::kMTile;
    constexpr int kLdsPad = Config::kLdsPad;

    const int t = threadIdx.x;
    const int n = blockIdx.x * kNTile + t * kNPerThread;
    const int m_tile = blockIdx.y * kMTile;
    const bool active = (n < size_n);

    const int k_split = blockIdx.z;
    const int k_per_split = size_k / split_k;
    const int k_start = k_split * k_per_split;
    const int k_end = (k_split == split_k - 1) ? size_k : (k_start + k_per_split);
    // Round the stride to 8 halfs so the 128-bit LDS loads stay aligned.
    const int row_stride = ((k_per_split + kLdsPad + 7) / 8) * 8;

    const int groupsize = size_k / groups;
    int group = k_start / groupsize;
    int nextgroup = (group + 1) * groupsize;

    __half2 z[kNPerThread][2], y[kNPerThread][2];
    if (active) {
        refresh_group<kNPerThread>(group, n, qzeros, scales, size_n, zero_offset, z, y);
    }

    float block_c[kMTile][kNPerThread];
#pragma unroll
    for (int m = 0; m < kMTile; ++m) {
#pragma unroll
        for (int j = 0; j < kNPerThread; ++j) block_c[m][j] = 0.0f;
    }

    extern __shared__ __half block_a[];
    if (g_idx) {
#pragma unroll 1
        for (int idx = t; idx < kMTile * k_per_split; idx += kThreads) {
            const int m = idx / k_per_split;
            const int kk = idx % k_per_split;
            const int m_row = m_tile + m;
            if (m_row < size_m) {
                block_a[m * row_stride + kk] = (a + static_cast<long long>(m_row) * size_k)[g_idx[k_start + kk]];
            }
        }
    } else if (k_per_split % 4 == 0) {
        const int quads = (kMTile * k_per_split) / 4;
#pragma unroll 1
        for (int idx = t; idx < quads; idx += kThreads) {
            const int quad = idx % (k_per_split / 4);
            const int m = idx / (k_per_split / 4);
            const int m_row = m_tile + m;
            if (m_row < size_m) {
                const float2 a4 =
                    *reinterpret_cast<const float2*>(a + static_cast<long long>(m_row) * size_k + k_start + quad * 4);
                *reinterpret_cast<float2*>(block_a + m * row_stride + quad * 4) = a4;
            }
        }
    } else {
        const int pairs = (kMTile * k_per_split) / 2;
#pragma unroll 1
        for (int idx = t; idx < pairs; idx += kThreads) {
            const int pair = idx % (k_per_split / 2);
            const int m = idx / (k_per_split / 2);
            const int m_row = m_tile + m;
            if (m_row < size_m) {
                const __half2 a2 =
                    *reinterpret_cast<const __half2*>(a + static_cast<long long>(m_row) * size_k + k_start + pair * 2);
                *reinterpret_cast<__half2*>(block_a + m * row_stride + pair * 2) = a2;
            }
        }
    }
    __syncthreads();

    const uint32_t* b_ptr = qweight + static_cast<long long>(k_start / 8) * size_n + n;

    int k = k_start;
    if (active) {
        while (k < k_end) {
            int4 b_prefetch[kKStep / 8];
#pragma unroll
            for (int j = 0; j < kKStep / 8; ++j) b_prefetch[j] = *reinterpret_cast<const int4*>(b_ptr + j * size_n);
            b_ptr += (kKStep / 8) * size_n;

#pragma unroll
            for (int j = 0; j < kKStep / 8; ++j) {
                if (k + 8 * j == nextgroup) {
                    group++;
                    nextgroup += groupsize;
                    refresh_group<kNPerThread>(group, n, qzeros, scales, size_n, zero_offset, z, y);
                }
                const int a_off = 8 * j;
                __half2 dq[kNPerThread][4];
                uint32_t w[kNPerThread];
                w[0] = static_cast<uint32_t>(b_prefetch[j].x);
                w[1] = static_cast<uint32_t>(b_prefetch[j].y);
                w[2] = static_cast<uint32_t>(b_prefetch[j].z);
                w[3] = static_cast<uint32_t>(b_prefetch[j].w);
#pragma unroll
                for (int col = 0; col < kNPerThread; ++col) dequant4x8(w[col], dq[col], z[col], y[col]);

#pragma unroll
                for (int m = 0; m < kMTile; ++m) {
                    const int m_row = m_tile + m;
                    if (m_row >= size_m) continue;
                    // The dynamic stride is a multiple of 8 halfs, so this is 16-byte aligned.
                    const float4 a8 =
                        *reinterpret_cast<const float4*>(block_a + m * row_stride + (k - k_start) + a_off);
                    const __half* a_ptr = reinterpret_cast<const __half*>(&a8);
#pragma unroll
                    for (int col = 0; col < kNPerThread; ++col) block_c[m][col] += dot8(dq[col], a_ptr);
                }
            }
            k += kKStep;
        }
        epilogue<kMTile>(block_c, m_tile, size_m, size_n, n, c);
    }
}

#else  // No gfx1030 device pass: keep the symbols without a schedule.

template <typename Config, int K_PER_SPLIT>
__global__ __launch_bounds__(Config::kThreads) void gemm_static_kernel(
    const __half*, const uint32_t*, const uint32_t*, const __half*, __half*, const int, const int,
    const int, const int, const int, const int*, const int) {}

template <typename Config>
__global__ __launch_bounds__(Config::kThreads) void gemm_dynamic_kernel(
    const __half*, const uint32_t*, const uint32_t*, const __half*, __half*, const int, const int,
    const int, const int, const int, const int*, const int) {}

#endif

enum ConfigId : int { kConfigV1 = 0, kConfigA = 1, kConfigC = 3 };

template <typename Config>
void launch_for_config(const Gptq& g, int split_k, hipStream_t stream) {
    constexpr int kNTile = Config::kNTile;
    constexpr int kMTile = Config::kMTile;
    constexpr int kLdsPad = Config::kLdsPad;

    dim3 block(Config::kThreads);
    dim3 grid((g.n + kNTile - 1) / kNTile, (g.m + kMTile - 1) / kMTile, split_k);
    const int k_per_split = g.k / split_k;
    const int lds_k_stride = ((k_per_split + kLdsPad + 7) / 8) * 8;
    const size_t shmem = kMTile * lds_k_stride * sizeof(__half);
    const __half* a = static_cast<const __half*>(g.a);
    const __half* scales = static_cast<const __half*>(g.scales);
    __half* c = static_cast<__half*>(g.out);

    switch (k_per_split) {
        case 256:
            gemm_static_kernel<Config, 256><<<grid, block, shmem, stream>>>(
                a, g.qweight, g.qzeros, scales, c, g.m, g.n, g.k, g.groups, g.zero_offset, g.g_idx, split_k);
            break;
        case 512:
            gemm_static_kernel<Config, 512><<<grid, block, shmem, stream>>>(
                a, g.qweight, g.qzeros, scales, c, g.m, g.n, g.k, g.groups, g.zero_offset, g.g_idx, split_k);
            break;
        case 1024:
            gemm_static_kernel<Config, 1024><<<grid, block, shmem, stream>>>(
                a, g.qweight, g.qzeros, scales, c, g.m, g.n, g.k, g.groups, g.zero_offset, g.g_idx, split_k);
            break;
        default:
            gemm_dynamic_kernel<Config><<<grid, block, shmem, stream>>>(
                a, g.qweight, g.qzeros, scales, c, g.m, g.n, g.k, g.groups, g.zero_offset, g.g_idx, split_k);
            break;
    }
}

// Split-K within 64 KiB of LDS; a split must be a K_STEP multiple, since the last chunk is not clamped.
template <typename Config>
int compute_split_k(int size_m, int size_n, int size_k) {
    constexpr int kNTile = Config::kNTile;
    constexpr int kMTile = Config::kMTile;
    constexpr int kLdsPad = Config::kLdsPad;
    constexpr int kKStep = Config::kKStep;

    const int blocks_per_k_split = ((size_m + kMTile - 1) / kMTile) * ((size_n + kNTile - 1) / kNTile);
    auto lds_bytes = [&](int split) {
        const int k_per_split = size_k / split;
        const int row_stride = ((k_per_split + kLdsPad + 7) / 8) * 8;
        return kMTile * row_stride * sizeof(__half);
    };
    const size_t lds_budget = (blocks_per_k_split > 1024) ? (16 * 1024)
                             : (blocks_per_k_split > 256) ? (64 * 1024)
                                                          : (32 * 1024);

    const int max_split_k = size_k / kKStep;
    int legacy = 1;
    while (legacy < max_split_k && lds_bytes(legacy) > lds_budget) legacy *= 2;
    while (legacy < 16 && max_split_k >= legacy * 2 &&
           (blocks_per_k_split * legacy < 2048 || (size_k / legacy) > 2048)) {
        const int candidate = legacy * 2;
        if (lds_bytes(candidate) > lds_budget) break;
        legacy = candidate;
    }
    int result;
    if (size_k % legacy == 0 && (size_k / legacy) % kKStep == 0) {
        result = legacy;
    } else {
        int splits[16];
        int count = 0;
        for (int s = 1; s <= 16; ++s) {
            if (size_k % s == 0 && (size_k / s) % kKStep == 0) splits[count++] = s;
        }
        int i = 0;
        while (i + 1 < count && lds_bytes(splits[i]) > lds_budget) ++i;
        while (i + 1 < count && (blocks_per_k_split * splits[i] < 2048 || (size_k / splits[i]) > 2048)) {
            if (lds_bytes(splits[i + 1]) > lds_budget) break;
            ++i;
        }
        result = splits[i];
    }

    static const bool debug_split = std::getenv("TENSORFOLD_RDNA2_PREFILL_DEBUG") != nullptr;
    if (debug_split) printf("[rdna2_prefill_split] m=%d n=%d k=%d split=%d\n", size_m, size_n, size_k, result);
    return result;
}

// Tile choice by M.
int select_config(int size_m, int size_n, int size_k) {
    static const int force = []() {
        const char* e = std::getenv("TENSORFOLD_RDNA2_PREFILL_FORCE_CONFIG");
        return e ? std::atoi(e) : -1;
    }();
    if (force >= 0) return force;

    if (size_m > 256) return size_n >= 4096 ? kConfigA : kConfigC;
    if (size_m < 4 && size_n > 4096) return kConfigA;
    if (size_m >= 4 && size_m <= 8 && size_n >= 2048 && size_k >= 1024) return kConfigC;
    if (size_m == 12 && size_n >= 2560 && size_n <= 8192 && size_k >= 512) return kConfigC;
    if (size_m == 32 && size_n >= 5120 && size_n <= 6144 && size_k >= 1536) return kConfigC;
    return kConfigV1;
}

int maybe_force_split_k(int computed) {
    static const int force = []() {
        const char* e = std::getenv("TENSORFOLD_RDNA2_PREFILL_FORCE_SPLIT_K");
        return e ? std::atoi(e) : -1;
    }();
    return force > 0 ? force : computed;
}

hipError_t launch_gptq_prefill(const Gptq& g, hipStream_t stream) {
    if (g.m < 1 || g.n < 1 || g.k < 1 || g.k % 32 != 0 || g.n % 8 != 0 || g.k % g.groups != 0) {
        return hipErrorInvalidValue;
    }
    if (g.k / g.groups < 32) return hipErrorInvalidValue;
    switch (select_config(g.m, g.n, g.k)) {
        case kConfigV1:
            launch_for_config<ConfigV1>(g, maybe_force_split_k(compute_split_k<ConfigV1>(g.m, g.n, g.k)), stream);
            break;
        case kConfigC:
            launch_for_config<ConfigC>(g, maybe_force_split_k(compute_split_k<ConfigC>(g.m, g.n, g.k)), stream);
            break;
        case kConfigA:
        default:
            launch_for_config<ConfigA>(g, maybe_force_split_k(compute_split_k<ConfigA>(g.m, g.n, g.k)), stream);
            break;
    }
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf
