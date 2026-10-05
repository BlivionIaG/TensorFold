#pragma once

// WMMA device helpers shared by the single, column, pair and group tiles.

#include "affine.hpp"
#include "arch.hpp"

#if TENSORFOLD_RDNA_WMMA
// rocWMMA rejects gfx1103 / gfx1152 / gfx1153, the same gfx11 WMMA: name them gfx1150 for its builtin choice.
#if defined(__gfx1103__) || defined(__gfx1152__) || defined(__gfx1153__)
#define __gfx1150__ 1
#endif
#include <rocwmma/rocwmma.hpp>
#endif

namespace tf {
namespace rocm {

struct GroupSide {
    const uint32_t* words;
    GroupTable scale;
    GroupTable bias;
    float* out;
    int n;
    int tiles;
};

hipError_t launch_affine_wmma_group(const Affine& a, GroupSide s0, GroupSide s1, GroupSide s2, GroupSide s3, int total,
                                    hipStream_t stream);

#if TENSORFOLD_RDNA_WMMA

constexpr int kTile = 16;

// 8-bit codes are whole bytes. A column's 16 codes are two dword pairs at a 16-byte step.
__device__ inline uint2 load_b8(const uint32_t* words, int words_per_row, int n0, int k0, int n, int lane) {
    int col = lane & 15;
    if (n0 + col >= n) return make_uint2(0u, 0u);
    const uint32_t* ptr = words + static_cast<long long>(n0 + col) * words_per_row + (k0 >> 2) + (lane >> 4) * 2;
    uint2 packed;
    ::memcpy(&packed, ptr, sizeof(packed));
    return packed;
}

__device__ inline void store_b8(hip_bfloat16* tile, uint2 packed, int lane) {
    hip_bfloat16* dst = tile + (lane & 15) * kTile + (lane >> 4) * 8;
    unsigned int halves[2] = {packed.x, packed.y};
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        unsigned int word = halves[half];
#pragma unroll
        for (int i = 0; i < 4; ++i) dst[half * 4 + i] = hip_bfloat16(code_bf16((word >> (i * 8)) & 0xffu));
    }
}

__device__ inline hip_bfloat16 bf16_bits(unsigned short bits) {
    hip_bfloat16 out;
    ::memcpy(&out, &bits, sizeof(out));
    return out;
}

__device__ inline void store_bf8(hip_bfloat16* dst, uint4 packed) {
    unsigned int words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        dst[i * 2] = bf16_bits(static_cast<unsigned short>(words[i]));
        dst[i * 2 + 1] = bf16_bits(static_cast<unsigned short>(words[i] >> 16));
    }
}

// One lane owns one activation row. Rows past M are zeros, and odd rows park at r ^ 1.
__device__ inline void fill_a8(hip_bfloat16* As, hip_bfloat16* AsOdd, const hip_bfloat16* x, int m0, int k0, int m,
                               int k, int lane, bool replay) {
    if (lane >= kTile) return;
    hip_bfloat16* row = As + lane * kTile;
    if (m0 + lane < m) {
        const hip_bfloat16* src = x + static_cast<long long>(m0 + lane) * k + k0;
        uint4 lo, hi;
        ::memcpy(&lo, src, sizeof(lo));
        ::memcpy(&hi, src + 8, sizeof(hi));
        store_bf8(row, lo);
        store_bf8(row + 8, hi);
    } else {
#pragma unroll
        for (int i = 0; i < kTile; ++i) row[i] = hip_bfloat16(0.f);
    }
    if (!replay) return;
    hip_bfloat16* odd = AsOdd + (lane ^ 1) * kTile;
#pragma unroll
    for (int i = 0; i < kTile; ++i) odd[i] = row[i];
}

__device__ inline void write_xsum(const hip_bfloat16* x, GroupTable scale, GroupTable bias, float* xsum, float* scol,
                           float* bcol, int lane, int m0, int n0, int base, int g, int group, int groups, int m,
                           int k, int n) {
    if (lane >= kTile) return;
    float sum = 0.f;
    if (m0 + lane < m) {
        const hip_bfloat16* xrow = x + static_cast<long long>(m0 + lane) * k + base;
        for (int t = 0; t < group; ++t) sum += static_cast<float>(xrow[t]);
    }
    xsum[lane] = sum;
    int col = n0 + lane;
    long long at = static_cast<long long>(col) * groups + g;
    scol[lane] = col < n ? scale[at] : 0.f;
    bcol[lane] = col < n ? bias[at] : 0.f;
}

__device__ inline void fma_group(float* acc, const float* dot_even, const float* dot_odd, const float* xsum,
                          const float* scol, const float* bcol, bool replay, int lane) {
    for (int e = lane; e < kTile * kTile; e += 32) {
        int r = e / kTile;
        int c = e % kTile;
        float dot = (replay && (r & 1)) ? dot_odd[(r ^ 1) * kTile + c] : dot_even[e];
        acc[e] = fmaf(dot, scol[c], acc[e]);
        acc[e] = fmaf(xsum[r], bcol[c], acc[e]);
    }
}

template <typename FragA, typename FragB, typename FragC>
__device__ inline void mma_tile(FragC& even, FragC& odd, const hip_bfloat16* As, const hip_bfloat16* AsOdd,
                         const hip_bfloat16* Bs, bool replay) {
    using namespace rocwmma;
    FragA fa;
    FragB fb;
    load_matrix_sync(fa, As, kTile);
    load_matrix_sync(fb, Bs, kTile);
    mma_sync(even, fa, fb, even);
    if (replay) {
        FragA fa_odd;
        load_matrix_sync(fa_odd, AsOdd, kTile);
        mma_sync(odd, fa_odd, fb, odd);
    }
}

// Every thread takes the same `live` path, so the barriers are uniform.
template <typename FragA, typename FragB, typename FragC>
__device__ inline void consume8(FragC& even, FragC& odd, hip_bfloat16* As, hip_bfloat16* AsOdd, hip_bfloat16* Bs,
                         const hip_bfloat16* x, uint2 packed, int m0, int k0, int m, int k, int lane, bool replay,
                         bool live) {
    if (!live) return;
    store_b8(Bs, packed, lane);
    fill_a8(As, AsOdd, x, m0, k0, m, k, lane, replay);
    __syncthreads();
    mma_tile<FragA, FragB, FragC>(even, odd, As, AsOdd, Bs, replay);
    __syncthreads();
}

// One activation tile feeds both sides. `live` is uniform across the wave.
template <typename FragA, typename FragB, typename FragC>
__device__ inline void consume_pair(FragC* even, FragC* odd, hip_bfloat16* As, hip_bfloat16* AsOdd, hip_bfloat16* Bs,
                             const hip_bfloat16* x, uint2 first, uint2 second, int m0, int k0, int m, int k,
                             int lane, bool replay, bool live) {
    if (!live) return;
    fill_a8(As, AsOdd, x, m0, k0, m, k, lane, replay);
    store_b8(Bs, first, lane);
    __syncthreads();
    mma_tile<FragA, FragB, FragC>(even[0], odd[0], As, AsOdd, Bs, replay);
    __syncthreads();
    store_b8(Bs, second, lane);
    __syncthreads();
    mma_tile<FragA, FragB, FragC>(even[1], odd[1], As, AsOdd, Bs, replay);
    __syncthreads();
}

#endif

}  // namespace rocm
}  // namespace tf
