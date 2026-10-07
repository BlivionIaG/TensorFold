#pragma once

// Prefill GEMM tile of the dot2 schedule (m >= 64): 128 x 128 outputs a block, a wave 16 rows x 128 columns.
// An output is what affine_dot2_block computes, bit for bit: per group a dot2 chain in ascending k from zero, the
// group's sum of x (lo then hi of each pair, in order), then acc = fma(dot, scale, acc); acc = fma(sumx, bias, acc).
// x is shared inside a quad of lanes (DPP) instead of read from LDS; the row sums of x come from their own threads, and a
// group's bias term runs a stage after its scale term; the codes become pairs by bit operations (0x6400 | code is 1024 + code).

#include <type_traits>
#include <utility>

#include "common/dot2.hpp"
#include "common/vec.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

constexpr int kGemmM = 128;
constexpr int kGemmN = 128;
constexpr int kGemmK = 32;
constexpr int kGemmLd = kGemmK + 8;  // 20 words a row: 16-byte aligned, eight 16-byte readers hit eight bank groups

// Rows of x a lane keeps (it owns 64 / RT columns): 16 where the dot2 takes x from another lane of the quad inside the
// instruction (RDNA2 FP16), 8 where that costs a v_mov a row (the BF16 dot2 of gfx11 has no DPP form).
template <typename T>
struct GemmShape {
    static constexpr int rt = 16;
};

template <>
struct GemmShape<DotBF16> {
    static constexpr int rt = 8;
};

// One 128 x 128 block of the product y = x . W^T: the decoder Dec reads the weight words and group terms, Act the rows of x,
// T multiplies and Epi stores; RT rows of x a lane keep (GemmShape).
template <class Dec, class Act, class T, class Epi, int RT>
__device__ __forceinline__ void gemm_tile(typename Dec::Args& a) {
    if (!Dec::take_item(a, blockIdx.z)) return;
    const int bx = blockIdx.x, by = blockIdx.y;
    if (by * kGemmM >= a.m) return;
    using pair = typename T::pair;
    constexpr int kLdw = kGemmLd / 2;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][kGemmM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][kGemmN * kLdw];
    __shared__ float sx[2][kGemmM];
    // Scales and biases of the last three groups: a group's bias term runs a stage after its scale term, so with a
    // group a stage a waiting wave still reads the bias of the group before the last that the others have staged.
    __shared__ float sc[3][kGemmN];
    __shared__ float bi[3][kGemmN];

    const int tid = threadIdx.x;
    const int m0 = by * kGemmM;
    const int n0 = bx * kGemmN;
    const int words_row = Dec::template row_words<int>(a);
    const int groups = a.k / a.group;
    const int stages = a.k / kGemmK;
    const int per = a.group / kGemmK;
    const int line = tid & (kGemmN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);  // codes 16 * side ..; side 1 also stages x

    // This thread's column of codes, and half of a row of x (two threads share a row's 64 bytes of the stage). Past the
    // edge a thread reads the last column or the first row and stores nothing that is used.
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * words_row;
    const int xrow = tid >> 1;
    const int xhalf = tid & 1;
    // Rows of x past the end (a routed item's few rows) are neither loaded, stored nor summed: a wave of stagers has
    // 16 rows, a wave of summers 32, so a whole wave skips.
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const bool xlive = wave * 16 < a.m - m0;
    const bool sxlive = (line & ~31) < a.m - m0;
    const int row_s = m0 + xrow < a.m ? m0 + xrow : 0;
    const typename Act::elem* xrow_ptr = Act::row(a, row_s);
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(xrow_ptr);

    // Two stages a fetch: the 64 bytes of x a row has in each (lane h of a row's two takes stage h, so a row is one
    // 128-byte request) and the two code pieces of a column, which are contiguous.
    uint32_t wreg[2][Dec::kWords];
    u32x4 xreg[4];
    typename Dec::Term term[2];
    float run = 0.f;

    // Every thread issues the same loads, so none waits on a branch. A stage past the last repeats it.
    auto fetch = [&](int st0) {
        const int st1 = st0 + 1 < stages ? st0 + 1 : stages - 1;
        const int st = st0 < stages ? st0 : stages - 1;
        Dec::template load<true>(wsrc + st * Dec::kWords, wreg[0]);
        Dec::template load<true>(wsrc + st1 * Dec::kWords, wreg[1]);
        const int sx_ = xhalf ? st1 : st;
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[sx_ * 4 + v];
        }
        const long long base = static_cast<long long>(col_s) * groups;
        term[0] = Dec::term(a, base + st / per);
        term[1] = Dec::term(a, base + st1 / per);
    };
    // Stage st (parity J) from the buffer `buf`: this thread's 16 codes of its column, its stage's row of x if it holds
    // that stage, and the stage's scale and bias at a group's last stage.
    auto stash = [&]<int J>(std::integral_constant<int, J>, int st, int buf) {
        u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + line * kLdw + side * 8);
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int t = S * 16 + v * 8 + h * 2;
                    d[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wreg[J], t));
                }
                dst[v] = u32x4{d[0], d[1], d[2], d[3]};
            }
        };
        if (side == 0) decode(std::integral_constant<int, 0>{}); else decode(std::integral_constant<int, 1>{});
        if (xhalf == J && xlive) {
            u32x4* xdst = reinterpret_cast<u32x4*>(xs[buf] + xrow * kLdw);
#pragma unroll
            for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
        }
        if (side == 0 && st % per == per - 1) {
            sc[(st / per) % 3][line] = Dec::scale_value(a, term[J]);
            bi[(st / per) % 3][line] = Dec::bias_value(a, term[J]);
        }
    };
    // Side 1 sums a row's x of the stage in LDS, lo then hi of each pair in order, on through the group.
    auto sumx = [&](int st, int buf) {
        if (st % per == 0) run = 0.f;
        const u32x4* src = reinterpret_cast<const u32x4*>(xs[buf] + line * kLdw);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 u = src[v];
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const pair p = __builtin_bit_cast(pair, q[h]);
                run += T::lo(p);
                run += T::hi(p);
            }
        }
        if (st % per == per - 1) sx[(st / per) & 1][line] = run;
    };

    // A wave owns 16 rows of x and 128 columns: with RT = 16 every lane takes the same rows and columns lane + 32 j; with 8
    // the lanes 0-15 take rows 0-7 and 16-31 rows 8-15, and columns lane % 16 + 16 j. Lane p of a quad loads x rows 4 k + p
    // and every lane takes row 4 k + i from lane i of the quad (DPP).
    constexpr int CT = 64 / RT;
    const int lane = tid & 31;
    // A wave whose 16 rows are all past the end skips its dots; with 3 groups or fewer live, every other block turns the
    // groups by 2 waves so the two blocks of a WGP keep all four SIMDs busy (a wave runs on SIMD wave % 4).
    const int rg = (wave + ((a.m - m0 <= 48 && (bx & 1)) ? 6 : 0)) & 7;
    const bool live = rg * 16 < a.m - m0;
    const int hrow = RT == 8 ? (lane >> 4) * 8 : 0;
    constexpr int kCStride = RT == 8 ? 16 : 32;
    const int cbase = RT == 8 ? lane & 15 : lane;
    float acc[RT][CT];
    float dot[RT][CT];
#pragma unroll
    for (int r = 0; r < RT; ++r) {
#pragma unroll
        for (int j = 0; j < CT; ++j) {
            acc[r][j] = 0.f;
            dot[r][j] = 0.f;
        }
    }
    // The bias terms of a group come a stage after its scale terms: its row sums are published at the end of the last
    // stage, and acc is untouched until the next group's scale term, so the order of the two fma holds.
    auto apply_bias = [&](int slot, int g) {
        float s_x[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r) s_x[r] = sx[g][rg * 16 + hrow + r];
#pragma unroll
        for (int j = 0; j < CT; ++j) {
            const float bias = bi[slot][cbase + kCStride * j];
#pragma unroll
            for (int r = 0; r < RT; ++r) acc[r][j] = Dec::fold_bias(acc[r][j], s_x[r], bias);
        }
    };
    fetch(0);
    stash(std::integral_constant<int, 0>{}, 0, 0);
    __syncthreads();
    // Two stages a trip so a stage's buffer and parity are known when the code is built.
    auto step = [&]<int P>(std::integral_constant<int, P>, int s) {
        constexpr int cur = P;
        if (live && s > 0 && (s - 1) % per == per - 1) apply_bias(((s - 1) / per) % 3, ((s - 1) / per) & 1);
        if constexpr (P == 1) fetch(s + 1);
        if (live) {
            constexpr int kXRegs = RT / 4;  // x words (4 pairs each) a lane loads a chunk
            const uint32_t* xb = xs[cur] + (rg * 16 + hrow + (lane & 3)) * kLdw;
            const uint32_t* wb = ws[cur] + cbase * kLdw;
            u32x4 xr[2][kXRegs];
            u32x4 wr[2][CT];
            auto load = [&](int c, u32x4 (&xo)[kXRegs], u32x4 (&wo)[CT]) {
#pragma unroll
                for (int k = 0; k < kXRegs; ++k) xo[k] = *reinterpret_cast<const u32x4*>(xb + 4 * k * kLdw + c * 4);
#pragma unroll
                for (int j = 0; j < CT; ++j) wo[j] = *reinterpret_cast<const u32x4*>(wb + kCStride * j * kLdw + c * 4);
            };
            load(0, xr[0], wr[0]);
#pragma unroll
            for (int c = 0; c < kGemmK / 8; ++c) {
                if (c + 1 < kGemmK / 8) load(c + 1, xr[(c + 1) & 1], wr[(c + 1) & 1]);
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    uint32_t wq[CT];
#pragma unroll
                    for (int j = 0; j < CT; ++j) {
                        const u32x4 u = wr[c & 1][j];
                        wq[j] = q == 0 ? u.x : q == 1 ? u.y : q == 2 ? u.z : u.w;
                    }
#pragma unroll
                    for (int k = 0; k < kXRegs; ++k) {
                        const u32x4 u = xr[c & 1][k];
                        const uint32_t xq = q == 0 ? u.x : q == 1 ? u.y : q == 2 ? u.z : u.w;
                        [&]<int... I>(std::integer_sequence<int, I...>) {
                            (([&] {
#pragma unroll
                                 for (int j = 0; j < CT; ++j) {
                                     dot[4 * k + I][j] = T::template quad<I>(xq, wq[j], dot[4 * k + I][j]);
                                 }
                             }()),
                             ...);
                        }(std::make_integer_sequence<int, 4>{});
                    }
                }
            }
            if (s % per == per - 1) {
                const int g = (s / per) % 3;
#pragma unroll
                for (int j = 0; j < CT; ++j) {
                    const float scale = sc[g][cbase + kCStride * j];
#pragma unroll
                    for (int r = 0; r < RT; ++r) {
                        acc[r][j] = Dec::fold_scale(acc[r][j], dot[r][j], scale);
                        dot[r][j] = 0.f;
                    }
                }
            }
        }
        if (side == 1 && sxlive) sumx(s, cur);
        stash(std::integral_constant<int, P ^ 1>{}, s + 1, cur ^ 1);
        __syncthreads();
    };
    for (int s = 0; s < stages; s += 2) {
        step(std::integral_constant<int, 0>{}, s);
        if (s + 1 < stages) step(std::integral_constant<int, 1>{}, s + 1);
    }
    if (live && (stages - 1) % per == per - 1) apply_bias(((stages - 1) / per) % 3, ((stages - 1) / per) & 1);
#pragma unroll
    for (int r = 0; r < RT; ++r) {
        const int row = m0 + rg * 16 + hrow + r;
        if (row >= a.m) continue;
#pragma unroll
        for (int j = 0; j < CT; ++j) {
            const int col = n0 + cbase + kCStride * j;
            if (col < a.n) Epi::store(a, row, col, acc[r][j]);
        }
    }
}

}  // namespace rocm
}  // namespace tf
