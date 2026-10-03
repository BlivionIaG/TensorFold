#pragma once

// Shared W4A16 GPTQ primitives for qgemm_rdna2.hip and qgemm_rdna2_prefill.hip.
// Ported from vLLM-rdna csrc/rocm/q_gemm_rdna2_common.cuh and qdq_4_rdna2.cuh.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// fp16 only. The dequant is the exllamav2 bit-trick: the mantissa of half is
// wide enough to hold a nibble shifted by 4 bits, so the upper-nibble pairs are
// read as `q * 16 + 1024` and divided by 16 inside the FMA.

#include <hip/hip_fp16.h>

#include <cstdint>

namespace tf {
namespace rocm {

// 4 V_DOT2_F32_F16 calls covering 8 consecutive K positions. hipcc does not
// lower an hfma2 chain to v_dot2 on gfx1030, so the builtin is written out and
// the accumulator stays fp32 (fp16 accumulation loses ~3 bits).
__device__ __forceinline__ float dot8(const __half2 (&dq)[4], const __half* a) {
    float r = 0.0f;
    const __half2* a2 = reinterpret_cast<const __half2*>(a);
#pragma unroll
    for (int i = 0; i < 4; ++i) r = __builtin_amdgcn_fdot2(dq[i], a2[i], r, false);
    return r;
}

// Scale-baked constants for one (zero, scale) pair:
//   z[0] = scale * (-1024 - zero)  for the low pairs  (q + 1024)
//   z[1] = scale * (-64   - zero)  for the high pairs (q * 16 + 1024)
//   y[0] = scale                   y[1] = scale / 16
__device__ __forceinline__ void prep_zero_scale(uint32_t zero, __half scale, __half2 (&z)[2],
                                                __half2 (&y)[2]) {
    // half bits 0xE400 are -1024.0; ORing the zero into the mantissa subtracts it.
    union {
        uint16_t u;
        __half h;
    } z1;
    z1.u = static_cast<uint16_t>(0xE400 | zero);
    const __half z16 = __hsub(__int2half_rn(-64), __int2half_rn(static_cast<int>(zero)));
    const __half2 s2 = __half2half2(scale);
    z[0] = __hmul2(s2, __half2half2(z1.h));
    z[1] = __hmul2(s2, __half2half2(z16));
    y[0] = __hmul2(s2, __half2half2(__float2half_rn(1.0f)));
    y[1] = __hmul2(s2, __half2half2(__float2half_rn(1.0f / 16.0f)));
}

// One int32 (8 shuffled nibbles) -> four half2 of (q - zero) * scale.
__device__ __forceinline__ void dequant4x8(uint32_t qa, __half2 (&dq)[4], const __half2 (&z)[2],
                                           const __half2 (&y)[2]) {
    const uint32_t c0 = 0x64006400u;  // half2(1024, 1024)
    union {
        uint32_t u;
        __half2 h2;
    } q0, q1, q2, q3;
    q0.u = (qa & 0x000F000Fu) | c0;  // (q0 + 1024, q1 + 1024)
    q1.u = (qa & 0x00F000F0u) | c0;  // (q2 * 16 + 1024, q3 * 16 + 1024)
    const uint32_t hi = qa >> 8;
    q2.u = (hi & 0x000F000Fu) | c0;
    q3.u = (hi & 0x00F000F0u) | c0;
    dq[0] = __hfma2(q0.h2, y[0], z[0]);
    dq[1] = __hfma2(q1.h2, y[1], z[1]);
    dq[2] = __hfma2(q2.h2, y[0], z[0]);
    dq[3] = __hfma2(q3.h2, y[1], z[1]);
}

// gfx1030 has no v_global_atomic_pk_add_f16 and HIP exposes no atomicAdd(__half*),
// so both are CAS loops. The 4-column form needs n % 4 == 0 and an 8-byte-aligned
// target (n a multiple of 4, N a multiple of 8).
__device__ __forceinline__ void atomic_add_pk4(__half* addr, __half2 v01, __half2 v23) {
    auto* p = reinterpret_cast<unsigned long long*>(addr);
    unsigned long long old = *p;
    for (;;) {
        union {
            unsigned long long u;
            __half2 h2[2];
        } cur, sum;
        cur.u = old;
        sum.h2[0] = __hadd2(cur.h2[0], v01);
        sum.h2[1] = __hadd2(cur.h2[1], v23);
        const unsigned long long prev = atomicCAS(p, old, sum.u);
        if (prev == old) break;
        old = prev;
    }
}

// Slower scalar form, only for an N tail short of the 4-column aligned bulk.
__device__ __forceinline__ void atomic_add_h(__half* addr, __half v) {
    auto* p = reinterpret_cast<unsigned short*>(addr);
    unsigned short old = *p;
    for (;;) {
        const __half cur = *reinterpret_cast<__half*>(&old);
        const __half sum = __hadd(cur, v);
        const unsigned short sum_u = *reinterpret_cast<const unsigned short*>(&sum);
        const unsigned short prev = atomicCAS(p, old, sum_u);
        if (prev == old) break;
        old = prev;
    }
}

// Precondition: n % 4 == 0, so the four nibbles for columns n..n+3 are one word of
// the (groups, N / 8) packed-zeros tensor.
__device__ __forceinline__ void load4_zeros(const uint32_t* qzeros_row, int n, int (&zeros)[4]) {
    const int qcol = n / 8;
    const int shift = (n & 0x07) * 4;
    const uint32_t d = qzeros_row[qcol] >> shift;
    zeros[0] = static_cast<int>(d & 0xF);
    zeros[1] = static_cast<int>((d >> 4) & 0xF);
    zeros[2] = static_cast<int>((d >> 8) & 0xF);
    zeros[3] = static_cast<int>((d >> 12) & 0xF);
}

__device__ __forceinline__ void load4_scales(const __half* scales_row, int n, __half (&scales)[4]) {
    scales[0] = scales_row[n + 0];
    scales[1] = scales_row[n + 1];
    scales[2] = scales_row[n + 2];
    scales[3] = scales_row[n + 3];
}

// Refresh the (z, y) constants for group g and N_COLS consecutive columns from n.
template <int N_COLS>
__device__ __forceinline__ void refresh_group(int g, int n, const uint32_t* qzeros, const __half* scales,
                                              int size_n, int zero_offset, __half2 (&z)[N_COLS][2],
                                              __half2 (&y)[N_COLS][2]) {
    const uint32_t* qz_row = qzeros + g * (size_n / 8);
    const __half* sc_row = scales + g * size_n;
    int zeros[N_COLS];
    __half scale[N_COLS];
    load4_zeros(qz_row, n, zeros);
    load4_scales(sc_row, n, scale);
#pragma unroll
    for (int i = 0; i < N_COLS; ++i) {
        prep_zero_scale(static_cast<uint32_t>(zeros[i] + zero_offset), scale[i], z[i], y[i]);
    }
}

// Write M_TILE rows of 4 consecutive N columns through the packed f16 CAS.
template <int M_TILE>
__device__ __forceinline__ void epilogue(const float block_c[M_TILE][4], int m_tile, int size_m,
                                         int size_n, int n, __half* c) {
#pragma unroll
    for (int m = 0; m < M_TILE; ++m) {
        const int m_row = m_tile + m;
        if (m_row >= size_m) continue;
        __half* c_row = c + static_cast<long long>(m_row) * size_n + n;
        const __half2 r01 = __halves2half2(__float2half_rn(block_c[m][0]), __float2half_rn(block_c[m][1]));
        const __half2 r23 = __halves2half2(__float2half_rn(block_c[m][2]), __float2half_rn(block_c[m][3]));
        atomic_add_pk4(c_row, r01, r23);
    }
}

}  // namespace rocm
}  // namespace tf
