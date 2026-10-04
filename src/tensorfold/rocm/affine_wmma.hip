// RDNA3, RDNA 3.5 and RDNA4 tile. One wave owns a 16x16 output. Each K step unpacks
// a 16x16 patch of codes into LDS for the WMMA.
// RDNA 3.5 uses the same gfx11 WMMA as RDNA3. rocWMMA hides that layout against gfx12.
//
// A row computed on its own always sits in row 0 of the tile. On gfx1100 the even
// rows of this 16x16x16 BF16 WMMA share row 0's rounding and the odd rows differ
// by one ulp, which would change an odd row's bits when a window grows. Those
// rows are replayed in the even rows of a second WMMA. RDNA4 uses the same
// replay; a part whose pairing differs fails the row-invariance test.
//
// Compiled without TENSORFOLD_RDNA_WMMA (RDNA2), the launch refuses and
// the GEMV is the only schedule.

#include "affine.hpp"
#include "affine_api.hpp"
#include "arch.hpp"

#include <cstring>
#include <stdexcept>
#include <string>

#if TENSORFOLD_RDNA_WMMA
// rocWMMA 7.2 names gfx1100–1102, gfx1150 and gfx1151. gfx1103, gfx1152 and
// gfx1153 are that same gfx11 WMMA, and the header rejects any other name.
// Calling them gfx1150 selects the gfx11 builtin; the compiler target is unchanged.
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

__device__ void write_xsum(const hip_bfloat16* x, GroupTable scale, GroupTable bias, float* xsum, float* scol,
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

__device__ void fma_group(float* acc, const float* dot_even, const float* dot_odd, const float* xsum,
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
__device__ void mma_tile(FragC& even, FragC& odd, const hip_bfloat16* As, const hip_bfloat16* AsOdd,
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
__device__ void consume8(FragC& even, FragC& odd, hip_bfloat16* As, hip_bfloat16* AsOdd, hip_bfloat16* Bs,
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
__device__ void consume_pair(FragC* even, FragC* odd, hip_bfloat16* As, hip_bfloat16* AsOdd, hip_bfloat16* Bs,
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

template <bool kEight>
__global__ void __launch_bounds__(32) affine_wmma_kernel(Affine a) {
    using namespace rocwmma;
    int n0 = blockIdx.x * kTile;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];  // logical row r parked at physical row r ^ 1
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kTile];
    __shared__ float bcol[kTile];

    for (int e = lane; e < kTile * kTile; e += 32) acc[e] = 0.f;
    // Odd rows of this tile are past M: the single WMMA already uses row 0's rounding.
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        // The whole group is in flight before the scale sum, so the K steps do not each wait on HBM.
        uint2 b0, b1, b2, b3, b4, b5, b6, b7;
        if constexpr (kEight) {
            b0 = load_b8(a.words, words, n0, base, a.n, lane);
            b1 = load_b8(a.words, words, n0, base + 16, a.n, lane);
            b2 = a.group > 32 ? load_b8(a.words, words, n0, base + 32, a.n, lane) : make_uint2(0u, 0u);
            b3 = a.group > 48 ? load_b8(a.words, words, n0, base + 48, a.n, lane) : make_uint2(0u, 0u);
            b4 = a.group > 64 ? load_b8(a.words, words, n0, base + 64, a.n, lane) : make_uint2(0u, 0u);
            b5 = a.group > 80 ? load_b8(a.words, words, n0, base + 80, a.n, lane) : make_uint2(0u, 0u);
            b6 = a.group > 96 ? load_b8(a.words, words, n0, base + 96, a.n, lane) : make_uint2(0u, 0u);
            b7 = a.group > 112 ? load_b8(a.words, words, n0, base + 112, a.n, lane) : make_uint2(0u, 0u);
        }
        write_xsum(x, a.scale, a.bias, xsum, scol, bcol, lane, m0, n0, base, group, a.group, groups, a.m, a.k,
                   a.n);
        __syncthreads();

        FragC part_even, part_odd;
        fill_fragment(part_even, 0.f);
        fill_fragment(part_odd, 0.f);
        if constexpr (kEight) {
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b0, m0, base, a.m, a.k, lane,
                                          replay, true);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b1, m0, base + 16, a.m, a.k, lane,
                                          replay, true);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b2, m0, base + 32, a.m, a.k, lane,
                                          replay, a.group > 32);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b3, m0, base + 48, a.m, a.k, lane,
                                          replay, a.group > 48);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b4, m0, base + 64, a.m, a.k, lane,
                                          replay, a.group > 64);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b5, m0, base + 80, a.m, a.k, lane,
                                          replay, a.group > 80);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b6, m0, base + 96, a.m, a.k, lane,
                                          replay, a.group > 96);
            consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b7, m0, base + 112, a.m, a.k, lane,
                                          replay, a.group > 112);
        } else {
            for (int step = 0; step < a.group; step += kTile) {
                int k0 = base + step;
                for (int e = lane; e < kTile * kTile; e += 32) {
                    int r = e / kTile;
                    int c = e % kTile;
                    int grow = m0 + r;
                    int gcol = k0 + c;
                    hip_bfloat16 xv = hip_bfloat16(0.f);
                    if (grow < a.m && gcol < a.k) xv = x[static_cast<long long>(grow) * a.k + gcol];
                    As[r * kTile + c] = xv;
                    if (replay) AsOdd[(r ^ 1) * kTile + c] = xv;

                    int ncol = n0 + r;
                    int nk = k0 + c;
                    hip_bfloat16 qv = hip_bfloat16(0.f);
                    if (ncol < a.n && nk < a.k) {
                        const uint32_t* wrow = a.words + static_cast<long long>(ncol) * words;
                        qv = hip_bfloat16(code_bf16(affine_code(wrow, nk, a.bits, words)));
                    }
                    Bs[r * kTile + c] = qv;  // col-major (k, n): index n * 16 + k
                }
                __syncthreads();
                mma_tile<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, replay);
                __syncthreads();
            }
        }

        store_matrix_sync(dot_even, part_even, kTile, mem_row_major);
        if (replay) store_matrix_sync(dot_odd, part_odd, kTile, mem_row_major);
        __syncthreads();
        // The replay's even physical row r ^ 1 holds logical row r.
        fma_group(acc, dot_even, dot_odd, xsum, scol, bcol, replay, lane);
        __syncthreads();
    }

    for (int e = lane; e < kTile * kTile; e += 32) {
        int grow = m0 + e / kTile;
        int gcol = n0 + e % kTile;
        if (grow < a.m && gcol < a.n) a.out[static_cast<long long>(grow) * a.n + gcol] = acc[e];
    }
}

// One wave, two outputs. A is loaded once per K step and both sides use it.
__global__ void __launch_bounds__(32) affine_wmma_pair_kernel(Affine a, AffineSide first, AffineSide second) {
    using namespace rocwmma;
    int n0 = blockIdx.x * kTile;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    AffineSide sides[2] = {first, second};

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[2][kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[2][kTile];
    __shared__ float bcol[2][kTile];

    for (int side = 0; side < 2; ++side) {
        for (int e = lane; e < kTile * kTile; e += 32) acc[side][e] = 0.f;
    }
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        if (lane < kTile) {
            float sum = 0.f;
            int row = m0 + lane;
            if (row < a.m) {
                const hip_bfloat16* xrow = x + static_cast<long long>(row) * a.k + base;
                for (int t = 0; t < a.group; ++t) sum += static_cast<float>(xrow[t]);
            }
            xsum[lane] = sum;
            int col = n0 + lane;
            for (int side = 0; side < 2; ++side) {
                long long at = static_cast<long long>(col) * groups + group;
                scol[side][lane] = col < sides[side].n ? sides[side].scale[at] : 0.f;
                bcol[side][lane] = col < sides[side].n ? sides[side].bias[at] : 0.f;
            }
        }
        __syncthreads();

        FragC part_even[2], part_odd[2];
        for (int side = 0; side < 2; ++side) {
            fill_fragment(part_even[side], 0.f);
            fill_fragment(part_odd[side], 0.f);
        }
        // Both sides of this group are read before either WMMA, one chunk of 64 codes at a time.
        for (int chunk = 0; chunk < a.group; chunk += 64) {
            int kbase = base + chunk;
            int left = a.group - chunk;
            uint2 p0 = load_b8(sides[0].words, words, n0, kbase, sides[0].n, lane);
            uint2 q0 = load_b8(sides[1].words, words, n0, kbase, sides[1].n, lane);
            uint2 p1 = make_uint2(0u, 0u), p2 = make_uint2(0u, 0u), p3 = make_uint2(0u, 0u);
            uint2 q1 = make_uint2(0u, 0u), q2 = make_uint2(0u, 0u), q3 = make_uint2(0u, 0u);
            if (left > 16) {
                p1 = load_b8(sides[0].words, words, n0, kbase + 16, sides[0].n, lane);
                q1 = load_b8(sides[1].words, words, n0, kbase + 16, sides[1].n, lane);
            }
            if (left > 32) {
                p2 = load_b8(sides[0].words, words, n0, kbase + 32, sides[0].n, lane);
                q2 = load_b8(sides[1].words, words, n0, kbase + 32, sides[1].n, lane);
            }
            if (left > 48) {
                p3 = load_b8(sides[0].words, words, n0, kbase + 48, sides[0].n, lane);
                q3 = load_b8(sides[1].words, words, n0, kbase + 48, sides[1].n, lane);
            }
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p0, q0, m0, kbase, a.m, a.k,
                                              lane, replay, true);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p1, q1, m0, kbase + 16, a.m,
                                              a.k, lane, replay, left > 16);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p2, q2, m0, kbase + 32, a.m,
                                              a.k, lane, replay, left > 32);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p3, q3, m0, kbase + 48, a.m,
                                              a.k, lane, replay, left > 48);
        }
        for (int side = 0; side < 2; ++side) {
            store_matrix_sync(dot_even, part_even[side], kTile, mem_row_major);
            if (replay) store_matrix_sync(dot_odd, part_odd[side], kTile, mem_row_major);
            __syncthreads();
            for (int e = lane; e < kTile * kTile; e += 32) {
                int r = e / kTile;
                int c = e % kTile;
                float dot = (replay && (r & 1)) ? dot_odd[(r ^ 1) * kTile + c] : dot_even[e];
                acc[side][e] = fmaf(dot, scol[side][c], acc[side][e]);
                acc[side][e] = fmaf(xsum[r], bcol[side][c], acc[side][e]);
            }
            __syncthreads();
        }
    }

    for (int side = 0; side < 2; ++side) {
        for (int e = lane; e < kTile * kTile; e += 32) {
            int grow = m0 + e / kTile;
            int gcol = n0 + e % kTile;
            if (grow < a.m && gcol < sides[side].n) {
                sides[side].out[static_cast<long long>(grow) * sides[side].n + gcol] = acc[side][e];
            }
        }
    }
}

// One block owns one 16-column tile of one side. The side is uniform inside the block.
__device__ void pick_group(int tile, GroupSide s0, GroupSide s1, GroupSide s2, GroupSide s3, const uint32_t*& words,
                           GroupTable& scale, GroupTable& bias, float*& out, int& n, int& n0) {
    int b1 = s0.tiles;
    int b2 = b1 + s1.tiles;
    int b3 = b2 + s2.tiles;
    GroupSide side = s0;
    int local = tile;
    if (tile >= b1 && tile < b2) {
        side = s1;
        local = tile - b1;
    } else if (tile >= b2 && tile < b3) {
        side = s2;
        local = tile - b2;
    } else if (tile >= b3) {
        side = s3;
        local = tile - b3;
    }
    words = side.words;
    scale = side.scale;
    bias = side.bias;
    out = side.out;
    n = side.n;
    n0 = local * kTile;
}

__global__ void __launch_bounds__(32) affine_wmma_group_kernel(Affine a, GroupSide s0, GroupSide s1, GroupSide s2,
                                                              GroupSide s3) {
    using namespace rocwmma;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words_per_row = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    const uint32_t* side_words = nullptr;
    GroupTable side_scale{};
    GroupTable side_bias{};
    float* side_out = nullptr;
    int side_n = 0;
    int n0 = 0;
    pick_group(blockIdx.x, s0, s1, s2, s3, side_words, side_scale, side_bias, side_out, side_n, n0);

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kTile];
    __shared__ float bcol[kTile];

    for (int e = lane; e < kTile * kTile; e += 32) acc[e] = 0.f;
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        uint2 b0 = load_b8(side_words, words_per_row, n0, base, side_n, lane);
        uint2 b1 = load_b8(side_words, words_per_row, n0, base + 16, side_n, lane);
        uint2 b2 = a.group > 32 ? load_b8(side_words, words_per_row, n0, base + 32, side_n, lane) : make_uint2(0u, 0u);
        uint2 b3 = a.group > 48 ? load_b8(side_words, words_per_row, n0, base + 48, side_n, lane) : make_uint2(0u, 0u);
        uint2 b4 = a.group > 64 ? load_b8(side_words, words_per_row, n0, base + 64, side_n, lane) : make_uint2(0u, 0u);
        uint2 b5 = a.group > 80 ? load_b8(side_words, words_per_row, n0, base + 80, side_n, lane) : make_uint2(0u, 0u);
        uint2 b6 = a.group > 96 ? load_b8(side_words, words_per_row, n0, base + 96, side_n, lane) : make_uint2(0u, 0u);
        uint2 b7 = a.group > 112 ? load_b8(side_words, words_per_row, n0, base + 112, side_n, lane)
                                 : make_uint2(0u, 0u);
        write_xsum(x, side_scale, side_bias, xsum, scol, bcol, lane, m0, n0, base, group, a.group, groups, a.m, a.k,
                   side_n);
        __syncthreads();

        FragC part_even, part_odd;
        fill_fragment(part_even, 0.f);
        fill_fragment(part_odd, 0.f);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b0, m0, base, a.m, a.k, lane, replay,
                                      true);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b1, m0, base + 16, a.m, a.k, lane,
                                      replay, true);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b2, m0, base + 32, a.m, a.k, lane,
                                      replay, a.group > 32);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b3, m0, base + 48, a.m, a.k, lane,
                                      replay, a.group > 48);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b4, m0, base + 64, a.m, a.k, lane,
                                      replay, a.group > 64);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b5, m0, base + 80, a.m, a.k, lane,
                                      replay, a.group > 80);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b6, m0, base + 96, a.m, a.k, lane,
                                      replay, a.group > 96);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b7, m0, base + 112, a.m, a.k, lane,
                                      replay, a.group > 112);

        store_matrix_sync(dot_even, part_even, kTile, mem_row_major);
        if (replay) store_matrix_sync(dot_odd, part_odd, kTile, mem_row_major);
        __syncthreads();
        fma_group(acc, dot_even, dot_odd, xsum, scol, bcol, replay, lane);
        __syncthreads();
    }

    for (int e = lane; e < kTile * kTile; e += 32) {
        int grow = m0 + e / kTile;
        int gcol = n0 + e % kTile;
        if (grow < a.m && gcol < side_n) side_out[static_cast<long long>(grow) * side_n + gcol] = acc[e];
    }
}

hipError_t launch_affine_wmma_group(const Affine& a, GroupSide s0, GroupSide s1, GroupSide s2, GroupSide s3,
                                    int total_tiles, hipStream_t stream) {
    if (a.m < 1 || a.bits != 8 || a.group % kTile || a.k % a.group || total_tiles < 1) return hipErrorInvalidValue;
    dim3 block(32);
    dim3 grid(total_tiles, (a.m + kTile - 1) / kTile);
    affine_wmma_group_kernel<<<grid, block, 0, stream>>>(a, s0, s1, s2, s3);
    return hipGetLastError();
}

hipError_t launch_affine_wmma_pair(const Affine& a, AffineSide first, AffineSide second, hipStream_t stream) {
    if (a.m < 1 || first.n < 1 || first.n != second.n || a.group % kTile || a.k % a.group) {
        return hipErrorInvalidValue;
    }
    dim3 block(32);
    dim3 grid((first.n + kTile - 1) / kTile, (a.m + kTile - 1) / kTile);
    affine_wmma_pair_kernel<<<grid, block, 0, stream>>>(a, first, second);
    return hipGetLastError();
}

// One wave owns several 16-column tiles and one row tile. The activation tile is loaded once per
// K step and every column uses it. Each column's WMMA and replay match the one-column kernel.
constexpr int kCols = 4;

template <typename FragA, typename FragB, typename FragC>
__device__ void cols_step(FragC* even, FragC* odd, hip_bfloat16* As, hip_bfloat16* AsOdd,
                          hip_bfloat16 Bs[][kTile * kTile], const hip_bfloat16* x, const uint2* packed, int m0,
                          int k0, int m, int k, int lane, bool replay) {
    fill_a8(As, AsOdd, x, m0, k0, m, k, lane, replay);
#pragma unroll
    for (int col = 0; col < kCols; ++col) store_b8(Bs[col], packed[col], lane);
    __syncthreads();
#pragma unroll
    for (int col = 0; col < kCols; ++col) {
        mma_tile<FragA, FragB, FragC>(even[col], odd[col], As, AsOdd, Bs[col], replay);
    }
    __syncthreads();
}

template <int Group>
__global__ void __launch_bounds__(32) affine_wmma_cols(Affine a) {
    using namespace rocwmma;
    constexpr int kSteps = Group / kTile;
    const int lane = threadIdx.x;
    const int nbase = blockIdx.x * (kCols * kTile);
    const int m0 = blockIdx.y * kTile;
    const int words = a.k * a.bits / 32;
    const int groups = a.k / Group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    const bool replay = m0 + 1 < a.m;

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kCols][kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kCols][kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kCols][kTile];
    __shared__ float bcol[kCols][kTile];

    for (int col = 0; col < kCols; ++col) {
        for (int e = lane; e < kTile * kTile; e += 32) acc[col][e] = 0.f;
    }

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        const int base = group * Group;
        // Every code load for this group is issued before the first WMMA, so the misses overlap.
        uint2 stage[kSteps][kCols];
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                stage[step][col] = load_b8(a.words, words, nbase + col * kTile, base + step * kTile, a.n, lane);
            }
        }
        if (lane < kTile) {
            float sum = 0.f;
            if (m0 + lane < a.m) {
                const hip_bfloat16* xrow = x + static_cast<long long>(m0 + lane) * a.k + base;
                for (int t = 0; t < Group; ++t) sum += static_cast<float>(xrow[t]);
            }
            xsum[lane] = sum;
#pragma unroll
            for (int col = 0; col < kCols; ++col) {
                const int gcol = nbase + col * kTile + lane;
                const long long at = static_cast<long long>(gcol) * groups + group;
                scol[col][lane] = gcol < a.n ? a.scale[at] : 0.f;
                bcol[col][lane] = gcol < a.n ? a.bias[at] : 0.f;
            }
        }
        __syncthreads();

        FragC part_even[kCols], part_odd[kCols];
#pragma unroll
        for (int col = 0; col < kCols; ++col) {
            fill_fragment(part_even[col], 0.f);
            fill_fragment(part_odd[col], 0.f);
        }
#pragma unroll
        for (int step = 0; step < kSteps; ++step) {
            cols_step<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, stage[step], m0, base + step * kTile,
                                           a.m, a.k, lane, replay);
        }
        for (int col = 0; col < kCols; ++col) {
            store_matrix_sync(dot_even, part_even[col], kTile, mem_row_major);
            if (replay) store_matrix_sync(dot_odd, part_odd[col], kTile, mem_row_major);
            __syncthreads();
            fma_group(acc[col], dot_even, dot_odd, xsum, scol[col], bcol[col], replay, lane);
            __syncthreads();
        }
    }

    for (int col = 0; col < kCols; ++col) {
        for (int e = lane; e < kTile * kTile; e += 32) {
            const int grow = m0 + e / kTile;
            const int gcol = nbase + col * kTile + e % kTile;
            if (grow < a.m && gcol < a.n) a.out[static_cast<long long>(grow) * a.n + gcol] = acc[col][e];
        }
    }
}

hipError_t launch_affine_wmma(const Affine& a, hipStream_t stream) {
    // Tiles past M or N are filled with zeros, so a single row uses the same WMMA as a full tile.
    if (a.m < 1 || a.n < 1 || a.group % kTile || a.k % a.group) return hipErrorInvalidValue;
    if (a.bits == 8 && a.m > kTile && a.n > kTile && (a.group == 32 || a.group == 64 || a.group == 128)) {
        dim3 block(32);
        dim3 grid((a.n + kCols * kTile - 1) / (kCols * kTile), (a.m + kTile - 1) / kTile);
        if (a.group == 32) {
            affine_wmma_cols<32><<<grid, block, 0, stream>>>(a);
        } else if (a.group == 64) {
            affine_wmma_cols<64><<<grid, block, 0, stream>>>(a);
        } else {
            affine_wmma_cols<128><<<grid, block, 0, stream>>>(a);
        }
        return hipGetLastError();
    }
    dim3 block(32);
    dim3 grid((a.n + kTile - 1) / kTile, (a.m + kTile - 1) / kTile);
    if (a.bits == 8) {
        affine_wmma_kernel<true><<<grid, block, 0, stream>>>(a);
    } else {
        affine_wmma_kernel<false><<<grid, block, 0, stream>>>(a);
    }
    return hipGetLastError();
}

#else

hipError_t launch_affine_wmma(const Affine&, hipStream_t) { return hipErrorNotSupported; }

hipError_t launch_affine_wmma_pair(const Affine&, AffineSide, AffineSide, hipStream_t) {
    return hipErrorNotSupported;
}

hipError_t launch_affine_wmma_group(const Affine&, GroupSide, GroupSide, GroupSide, GroupSide, int, hipStream_t) {
    return hipErrorNotSupported;
}

#endif

hipError_t launch_affine(const Affine& a, int schedule, hipStream_t stream) {
    if (a.fp16) {
        if (schedule == 2) return hipErrorNotSupported;
#if TENSORFOLD_RDNA_WMMA
        return hipErrorNotSupported;
#else
        return launch_affine_dot2(a, schedule == 1, stream);
#endif
    }
#if TENSORFOLD_RDNA_WMMA
    // Widths other than 8 run the BF16 dot2 tiles: the generic WMMA tile re-unpacks every code per K step.
    if (schedule == 0 && a.bits != 8 && (a.group == 32 || a.group == 64 || a.group == 128) && a.k % a.group == 0 &&
        a.k % 16 == 0) {
        return launch_affine_dot2_bf16(a, stream);
    }
    bool tile = schedule != 1 && a.group % 16 == 0 && a.k % a.group == 0;
#else
    bool tile = false;
#endif
    if (schedule == 3) return launch_affine_gemv_cols(a, stream);
    if (schedule == 2 && !tile) return hipErrorNotSupported;
    if (tile) return launch_affine_wmma(a, stream);
    return launch_affine_gemv(a, stream);
}

}  // namespace rocm
}  // namespace tf

void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half) {
    tf::rocm::Affine a{x, static_cast<const uint32_t*>(words), {scale, scale_kind}, {bias, scale_kind},
                       static_cast<float*>(out), m, n, k, bits, group, fp16};
    if (out_half) {
        if (!fp16 || schedule != 0 || splits > 1) throw std::runtime_error("affine kernel: fp16 output is the decode tile");
        a.out = nullptr;
        a.out16 = static_cast<__half*>(out);
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = (partial != nullptr && splits > 1)
                         ? tf::rocm::launch_affine_dot2_split(a, partial, splits, hip_stream)
                         : tf::rocm::launch_affine(a, schedule, hip_stream);
    if (err != hipSuccess) throw std::runtime_error(std::string("affine kernel: ") + hipGetErrorString(err));
}

void affine_group_launch(const void* x, const void* const* words, const void* const* scale, const void* const* bias,
                         int scale_kind, void* const* out, const int* ns, int nsides, int m, int k, int bits,
                         int group, void* stream) {
    if (nsides < 1 || nsides > 4) throw std::runtime_error("affine group kernel: 1 to 4 sides");
    tf::rocm::GroupSide sides[4] = {};
    int total = 0;
    for (int i = 0; i < nsides; ++i) {
        sides[i].words = static_cast<const uint32_t*>(words[i]);
        sides[i].scale = {scale[i], scale_kind};
        sides[i].bias = {bias[i], scale_kind};
        sides[i].out = static_cast<float*>(out[i]);
        sides[i].n = ns[i];
        sides[i].tiles = (ns[i] + 15) / 16;
        total += sides[i].tiles;
    }
    tf::rocm::Affine a{x, nullptr, {}, {}, nullptr, m, 0, k, bits, group, 0};
    hipError_t err = tf::rocm::launch_affine_wmma_group(a, sides[0], sides[1], sides[2], sides[3], total,
                                                        static_cast<hipStream_t>(stream));
    if (err != hipSuccess) throw std::runtime_error(std::string("affine group kernel: ") + hipGetErrorString(err));
}

void affine_pair_launch(const void* x, const void* words0, const void* scale0, const void* bias0, void* out0,
                        const void* words1, const void* scale1, const void* bias1, void* out1, int scale_kind, int m,
                        int n, int k, int bits, int group, void* stream) {
    tf::rocm::Affine a{x, nullptr, {}, {}, nullptr, m, n, k, bits, group, 0};
    tf::rocm::AffineSide first{static_cast<const uint32_t*>(words0), {scale0, scale_kind}, {bias0, scale_kind},
                               static_cast<float*>(out0), n};
    tf::rocm::AffineSide second{static_cast<const uint32_t*>(words1), {scale1, scale_kind}, {bias1, scale_kind},
                                static_cast<float*>(out1), n};
    hipError_t err = tf::rocm::launch_affine_wmma_pair(a, first, second, static_cast<hipStream_t>(stream));
    if (err != hipSuccess) throw std::runtime_error(std::string("affine pair kernel: ") + hipGetErrorString(err));
}
