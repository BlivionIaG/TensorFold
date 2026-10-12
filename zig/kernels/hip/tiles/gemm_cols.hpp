#pragma once

// Short-prompt GEMM for a small GPU, a lane a column through all of K: affine_gemm_block's chains and fold, same bits.

#include <type_traits>
#include <utility>

#include "common/dot2.hpp"
#include "common/vec.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

constexpr int kColsWaves = 4;
constexpr int kColsChunks = 4;  // 32-code chunks of x a step: every group of 128 codes or fewer ends inside one

// Codes 2I and 2I + 1 of a chunk as T's pair, the same values as Dec::pair: 4- and 8-bit FP16 by a byte permute.
template <class Dec, typename T, int I>
__device__ inline typename T::pair cols_pair(const uint32_t (&w)[Dec::kBits]) {
    if constexpr (std::is_same_v<typename T::elem, __half> && Dec::kBits == 4) {
        // byte I into bytes 0 and 2: the low code is 1024 + c in FP16, the high one at bits 4-7 of 64 + c
        constexpr uint32_t b = I & 3;
        const uint32_t t = __builtin_amdgcn_perm(0u, w[I >> 2], 0x0c000c00u | (b << 16) | b);
        const uint32_t bits = (t & 0x00f0000fu) | 0x54006400u;
        return __hsub2(__builtin_bit_cast(__half2, bits), __builtin_bit_cast(__half2, 0x54006400u));
    } else if constexpr (std::is_same_v<typename T::elem, __half> && Dec::kBits == 8) {
        constexpr uint32_t b = (I & 1) * 2;
        const uint32_t t = __builtin_amdgcn_perm(0u, w[I >> 1], 0x0c000c00u | ((b + 1) << 16) | b);
        return __hsub2(__builtin_bit_cast(__half2, t | 0x64006400u), __builtin_bit_cast(__half2, 0x64006400u));
    } else {
        return Dec::template pair<T>(w, 2 * I);
    }
}

// A barrier over the block's LDS alone: __syncthreads would also wait for every global load still in flight.
__device__ inline void cols_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// A chunk's pairs against RG rows of x, laid [pair][quad lane][row / 4]: dot2 takes row 4k + J from quad lane J by DPP.
template <class Dec, typename T, int RG, int... I>
__device__ inline void cols_dots(const uint32_t* xq, const uint32_t (&w)[Dec::kBits], float (&dot)[RG],
                                 std::integer_sequence<int, I...>) {
    constexpr int kAhead = 2;  // pairs of x read ahead of their dots: the dots' inline asm keeps the compiler from it
    uint32_t x[kAhead + 1][RG / 4];
    auto read = [&](int pair, uint32_t (&dst)[RG / 4]) {
#pragma unroll
        for (int k = 0; k < RG / 4; ++k) dst[k] = xq[pair * RG + k];
    };
#pragma unroll
    for (int i = 0; i < kAhead; ++i) read(i, x[i]);
    (([&] {
         if constexpr (I + kAhead < sizeof...(I)) read(I + kAhead, x[(I + kAhead) % (kAhead + 1)]);
         const uint32_t (&xc)[RG / 4] = x[I % (kAhead + 1)];
         const uint32_t q = __builtin_bit_cast(uint32_t, cols_pair<Dec, T, I>(w));
#pragma unroll
         for (int k = 0; k < RG / 4; ++k) {
             dot[4 * k] = T::template quad<0>(xc[k], q, dot[4 * k]);
             dot[4 * k + 1] = T::template quad<1>(xc[k], q, dot[4 * k + 1]);
             dot[4 * k + 2] = T::template quad<2>(xc[k], q, dot[4 * k + 2]);
             dot[4 * k + 3] = T::template quad<3>(xc[k], q, dot[4 * k + 3]);
         }
     }()),
     ...);
}

// RG rows a block, a lane a column; a wave loads its 32 columns' words SC chunks at a time on long runs, turned in LDS.
template <class Dec, class Act, class T, class Epi, int RG, int SC>
__device__ __forceinline__ void gemm_cols_tile(typename Dec::Args& a) {
    static_assert(RG % 4 == 0 && RG <= 32, "rows come in fours");
    constexpr int KC = kColsChunks;
    constexpr int kRun = SC * Dec::kBits;  // words of a column's run a weight step
    constexpr int kLanesCol = kRun / 4;
    static_assert(kRun % 4 == 0 && kLanesCol <= 32 && 32 % kLanesCol == 0, "a run is whole 16-byte pieces");
    constexpr int kLoads = kLanesCol;  // 16-byte loads a lane a step: 32 columns, 32 / kLanesCol a load
    constexpr int kLd = kRun + 4;      // LDS words a column: a lane's chunk reads hit distinct banks
    constexpr int kSub = SC / KC;      // x steps a weight step
    static_assert(SC % KC == 0 && kSub % 2 == 0, "x steps pair up in a weight step");
    constexpr int kThreads = 32 * kColsWaves;
    constexpr int kXs = KC * 16 * RG;  // words of x a step
    if (!Dec::take_item(a, blockIdx.z)) return;
    using pair = typename T::pair;
    const int rgs = gridDim.x / ((a.n + kThreads - 1) / kThreads);
    const int row0 = (blockIdx.x % rgs) * RG;
    if (row0 >= a.m) return;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int colw = (blockIdx.x / rgs) * kThreads + wave * 32;  // the wave's first column
    const int col = colw + lane;
    const int nch = a.k >> 5;
    const int per = a.group >> 5;
    const int gstep = KC / per > 0 ? KC / per : 1;  // groups an x step ends at most
    const long long groups = a.k / a.group;
    const long long words_row = Dec::template row_words<long long>(a);
    const long long tb = static_cast<long long>(col < a.n ? col : a.n - 1) * groups;
    const typename Dec::StreamTerms terms(a);

    __shared__ __attribute__((aligned(16))) uint32_t wl[kColsWaves][32 * kLd];
    __shared__ __attribute__((aligned(16))) uint32_t xs[3][kXs];
    __shared__ __attribute__((aligned(16))) float sxs[2][KC][RG];
    uint32_t* wv = wl[wave];

    // a weight step: load i takes column i * (32 / kLanesCol) + lane / kLanesCol, piece lane % kLanesCol of its run
    const int piece = lane % kLanesCol;
    const uint32_t* wsrc[kLoads];
#pragma unroll
    for (int i = 0; i < kLoads; ++i) {
        const int c = colw + i * (32 / kLanesCol) + lane / kLanesCol;
        wsrc[i] = Dec::words(a) + static_cast<long long>(c < a.n ? c : a.n - 1) * words_row;
    }
    u32x4 pre[kLoads];
    auto wload = [&](int step) {
        // a piece past the row's end reads its last piece, never used
        const long long at = static_cast<long long>(step) * kRun + piece * 4;
        const long long word = at + 4 <= words_row ? at : words_row - 4;
#pragma unroll
        for (int i = 0; i < kLoads; ++i) pre[i] = *reinterpret_cast<const u32x4*>(wsrc[i] + word);
    };
    auto wstore = [&] {
#pragma unroll
        for (int i = 0; i < kLoads; ++i) {
            uint32_t* d = wv + (i * (32 / kLanesCol) + lane / kLanesCol) * kLd + piece * 4;
            d[0] = pre[i].x;
            d[1] = pre[i].y;
            d[2] = pre[i].z;
            d[3] = pre[i].w;
        }
    };
    // x of KC chunks as cols_dots reads it; e % RG is the row, e / RG its 4 pairs; loaded two steps ahead
    constexpr int kVec = KC * 4 * RG;
    constexpr int kVecT = (kVec + kThreads - 1) / kThreads;
    u32x4 xv[kVecT];
    const typename Act::elem* xrow[kVecT];
#pragma unroll
    for (int i = 0; i < kVecT; ++i) {
        const int r = (threadIdx.x + i * kThreads) % RG;
        xrow[i] = Act::row(a, row0 + r < a.m ? row0 + r : 0);
    }
    auto stage_load = [&](int xstep) {
#pragma unroll
        for (int i = 0; i < kVecT; ++i) {
            const int e = threadIdx.x + i * kThreads;
            if (kVec % kThreads != 0 && e >= kVec) break;
            const int q = e / RG;
            const int chunk = xstep * KC + (q >> 2) < nch ? xstep * KC + (q >> 2) : nch - 1;
            xv[i] = reinterpret_cast<const u32x4*>(xrow[i] + chunk * 32LL)[q & 3];
        }
    };
    auto stage_store = [&](int xstep) {
        uint32_t* dst = xs[xstep % 3];
#pragma unroll
        for (int i = 0; i < kVecT; ++i) {
            const int e = threadIdx.x + i * kThreads;
            if (kVec % kThreads != 0 && e >= kVec) break;
            const int r = e % RG;
            const int q = e / RG;
            const int at = (r & 3) * (RG / 4) + (r >> 2);
            dst[(4 * q) * RG + at] = xv[i].x;
            dst[(4 * q + 1) * RG + at] = xv[i].y;
            dst[(4 * q + 2) * RG + at] = xv[i].z;
            dst[(4 * q + 3) * RG + at] = xv[i].w;
        }
    };
    // sums of x of each (row, group) ending in x step `xstep`, a lane each, lo then hi of each pair as the GEMM tile's
    auto sums = [&](int xstep) {
        const uint32_t* src = xs[xstep % 3];
        const int pairs = per * 16;
        for (int t = lane; t < RG * gstep; t += 32) {
            const int r = t % RG;
            const int gi = t / RG;
            const int at = (r & 3) * (RG / 4) + (r >> 2);
            float run = 0.f;
            for (int i = gi * pairs; i < (gi + 1) * pairs; ++i) {
                const pair p = __builtin_bit_cast(pair, src[i * RG + at]);
                run += T::lo(p);
                run += T::hi(p);
            }
            sxs[xstep & 1][gi][r] = run;
        }
    };
    // the group terms of a weight step's chunks that end a group, loaded with its words
    unsigned short sc[SC], bc[SC], sn[SC], bn[SC];
    auto tload = [&](int step) {
#pragma unroll
        for (int q = 0; q < SC; ++q) {
            const int chunk = step * SC + q < nch ? step * SC + q : nch - 1;
            const long long at = tb + chunk / per;
            sn[q] = terms.scale_raw(at);
            bn[q] = terms.bias_raw(at);
        }
    };

    float acc[RG];
    float dot[RG];
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        acc[r] = 0.f;
        dot[r] = 0.f;
    }
    const int xsteps = (nch + KC - 1) / KC;
    const int wsteps = (xsteps + kSub - 1) / kSub;
    wload(0);
    tload(0);
    stage_load(0);
    stage_store(0);
    if (xsteps > 1) {
        stage_load(1);
        stage_store(1);
    }
    wstore();
    cols_barrier();
    if (wave == 0) sums(0);
#pragma unroll
    for (int q = 0; q < SC; ++q) {
        sc[q] = sn[q];
        bc[q] = bn[q];
    }
    if (wsteps > 1) {
        wload(1);
        tload(1);
    }
    cols_barrier();
    // x step s: its x and step s + 1's in LDS with step s's sums; step s + 2's x comes in through registers
    for (int ws = 0; ws < wsteps; ++ws) {
#pragma unroll
        for (int sub = 0; sub < kSub; ++sub) {
            const int s = ws * kSub + sub;
            if (s >= xsteps) break;
            if (s + 2 < xsteps) stage_load(s + 2);
            if (wave == (s & (kColsWaves - 1)) && s + 1 < xsteps) sums(s + 1);
            const uint32_t* xb0 = xs[s % 3] + (lane & 3) * (RG / 4);
#pragma unroll
            for (int q = 0; q < KC; ++q) {
                const int chunk = s * KC + q;
                if (chunk >= nch) break;
                const uint32_t* wc = wv + lane * kLd + (sub * KC + q) * Dec::kBits;
                uint32_t w[Dec::kBits];
#pragma unroll
                for (int i = 0; i < Dec::kBits; ++i) w[i] = wc[i];
                cols_dots<Dec, T, RG>(xb0 + q * 16 * RG, w, dot, std::make_integer_sequence<int, 16>{});
                if (chunk % per == per - 1) {
                    const float scale = terms.widen(sc[sub * KC + q]);
                    const float bias = terms.widen(bc[sub * KC + q]);
                    const float* sx = sxs[s & 1][q / per];
#pragma unroll
                    for (int k = 0; k < RG / 4; ++k) {
#pragma unroll
                        for (int j = 0; j < 4; ++j) {
                            const int r = 4 * k + j;
                            acc[r] = Dec::fold_scale(acc[r], dot[r], scale);
                            acc[r] = Dec::fold_bias(acc[r], sx[r], bias);
                            dot[r] = 0.f;
                        }
                    }
                }
            }
            if (s + 2 < xsteps) stage_store(s + 2);
            // the wave's next weight step goes into LDS once its chunks are read; the one after it starts loading
            if (sub == kSub - 1 && ws + 1 < wsteps) {
                wstore();
#pragma unroll
                for (int q = 0; q < SC; ++q) {
                    sc[q] = sn[q];
                    bc[q] = bn[q];
                }
                if (ws + 2 < wsteps) {
                    wload(ws + 2);
                    tload(ws + 2);
                }
            }
            cols_barrier();
        }
    }
#pragma unroll
    for (int r = 0; r < RG; ++r) {
        if (row0 + r >= a.m) break;
        if (col < a.n) Epi::store(a, row0 + r, col, acc[r]);
    }
}

}  // namespace rocm
}  // namespace tf
