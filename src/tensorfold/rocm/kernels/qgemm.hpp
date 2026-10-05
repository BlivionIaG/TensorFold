#pragma once

// W4A16 GPTQ for gfx1030, ported from vLLM-rdna (opengfx1030/vllm-rdna) q_gemm_rdna2.cu and _prefill.cu.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

// GPTQv1 packs the zeros with a +1 offset; GPTQv2 does not.
enum ZeroOffset : int { kZeroV2 = 0, kZeroV1 = 1 };

// a fp16; out fp16 and zero before the launch: both kernels accumulate into it with atomics.
struct Gptq {
    const void* a;            // (m, k) __half, row major, contiguous
    const uint32_t* qweight;  // (k / 8, n) int4, GPTQ-shuffled
    const uint32_t* qzeros;   // (groups, n / 8) packed 4-bit zeros
    const void* scales;       // (groups, n) __half
    const int* g_idx;         // (k) act-order permutation, or null for identity
    void* out;                // (m, n) __half, zeroed
    int m, n, k, groups, zero_offset;
};

// M_COUNT tiles 1/2/4/8 cover M up to 15; the prefill kernel is the large-M path.
hipError_t launch_gptq_decode(const Gptq& g, hipStream_t stream);
hipError_t launch_gptq_prefill(const Gptq& g, hipStream_t stream);

// Grouped-expert W4A16 over TensorFold's plan; epi 0 fp32 rows (down), 1 bf16 relu^2, 2 bf16 SwiGLU (mats = 2).
struct MoeGptq {
    const void* a;            // (rows, k) __nv_bfloat16
    void* out;                // epi 0: (pairs, n) float; else (pairs, n) __nv_bfloat16
    const uint32_t* qweight;  // (mats, experts, k / 8, n)
    const uint32_t* qzeros;   // (mats, experts, groups, n / 8)
    const void* scales;       // (mats, experts, groups, n) __half
    const int32_t* items;     // (items, 3) expert, first, count
    const int32_t* members;   // (pairs)
    int items_count, n, k, groups, slots, zero_offset;
    float limit;              // SwiGLU clip (0 = none)
};
hipError_t launch_moe_gptq(const MoeGptq& g, int experts, int epi, int block_m, hipStream_t stream);

}  // namespace rocm
}  // namespace tf
