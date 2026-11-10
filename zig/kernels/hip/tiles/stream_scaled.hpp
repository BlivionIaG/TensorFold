#pragma once

// The stream tile of gfx900 fp16 with TF_G9_MIX=1 (tiles/stream.hpp includes it there instead of its own text, so that every other
// code object stays byte for byte what it was): x staged scaled by 2^10 or widened to fp32, see kScaledX.

// Streaming decode tile for 1 to 16 rows: a lane owns a 32-code chunk of a column; lanes fold by shuffle.

#include <type_traits>
#include <utility>

#include "common/dot2.hpp"
#include "common/vec.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

constexpr int kStreamWaves = 4;

// Tile shapes that exist: CB columns a lane times R rows at most 16 accumulators.
constexpr bool stream_shape(int rows, int cb) { return cb * rows <= 16; }
constexpr int kStreamRows = 16;  // the most rows a launch takes

// The pair (half `parity` of a, half `parity` of b).
__device__ inline uint32_t stream_pick(uint32_t a, uint32_t b, int parity) {
    return __builtin_amdgcn_perm(b, a, parity ? 0x07060302u : 0x05040100u);
}

// gfx900 fp16 (a Dot with kStreamScaledX): v_mad_mix_f32 flushes a denormal fp16 operand, so a stream tile that mixes on fp16 x must not
// see one. Staged x is scaled by 2^10 (every nonzero fp16 x, 2^-24 and up, becomes normal; exact) when the block's x are all below 32 in
// magnitude (so the scaled x stay finite), and the lanes undo the scale in the group's scale and bias (a power of two: exact). A block with a
// larger x (or a NaN) takes the unstaged path, where x is widened to fp32 (an fp32 operand is not flushed): same bits as the fused pair.
struct StreamWideX {
    float lo, hi;
};
template <typename T>
inline constexpr bool kScaledX = requires { requires T::kStreamScaledX; };

template <typename T>
__device__ __forceinline__ StreamWideX stream_wide(typename T::pair p) {
    return {T::lo(p), T::hi(p)};
}

// The stream tile's dot: a Dot may have its own form of it (gfx900 fp16: v_mad_mix_f32), else its dot.
template <typename T>
__device__ __forceinline__ float stream_dot(typename T::pair x, typename T::pair q, float acc) {
    if constexpr (requires { T::dot_stream(x, q, acc); }) return T::dot_stream(x, q, acc);
    else return T::dot(x, q, acc);
}
template <typename T>
__device__ __forceinline__ float stream_dot(StreamWideX x, typename T::pair q, float acc) {
    return T::dot_stream_wide(x.lo, x.hi, q, acc);
}

template <class Dec, typename T, int... I>
__device__ inline void stream_x_pairs(const uint32_t (&xn)[16], typename T::pair (&xp)[16],
                                      std::integer_sequence<int, I...>) {
    (([&] {
         constexpr int lo = Dec::template lo<I>();
         constexpr int hi = lo + Dec::gap();
         if constexpr (Dec::kGapPairs) {
             xp[I] = __builtin_bit_cast(typename T::pair, stream_pick(xn[lo >> 1], xn[hi >> 1], lo & 1));
         } else {
             xp[I] = __builtin_bit_cast(typename T::pair, xn[I]);
         }
     }()),
     ...);
}

// One column's chunk against RG rows: each code pair is made once and dotted with every row's x.
template <class Dec, typename T, bool WIDE, int RG, int... I>
__device__ inline void stream_dots(const typename T::pair (&xp)[RG][16], const uint32_t (&w)[Dec::kWords], float (&d)[RG],
                                   std::integer_sequence<int, I...>) {
    (([&] {
         const typename T::pair q = Dec::template stream_pair<T, I>(w);
#pragma unroll
         for (int r = 0; r < RG; ++r) {
             if constexpr (WIDE) d[r] = stream_dot<T>(stream_wide<T>(xp[r][I]), q, d[r]);  // widened where used: no 32 registers of x
             else d[r] = stream_dot<T>(xp[r][I], q, d[r]);
         }
     }()),
     ...);
}

// The block's R rows of x into LDS (R * (2k + k / 4) bytes): chunk q of row r as pairs at [r][v][q][4], then the sums of x, then (kScaledX)
// the chunk's unscale factor. A chunk whose x are all below 32 in magnitude is staged scaled by 2^10 (every nonzero fp16 x, 2^-24 and up,
// becomes normal, so the mix does not flush it) and its sum of x and its unscale factor (2^-10, else 1) are stored for the lane to undo it.
template <class Dec, class Act, class T, int R>
__device__ inline void stream_stage(const typename Dec::Args& a, int row0, uint32_t* pairs, float* sums) {
    using pair = typename T::pair;
    const int nch = a.k >> 5;
    for (int at = threadIdx.x; at < R * nch; at += 32 * kStreamWaves) {
        const int r = at / nch;
        const int q = at - r * nch;
        const u32x4* src = reinterpret_cast<const u32x4*>(Act::row(a, row0 + r < a.m ? row0 + r : 0) + q * 32LL);
        uint32_t xn[16];
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 t = src[v];
            xn[4 * v] = t.x;
            xn[4 * v + 1] = t.y;
            xn[4 * v + 2] = t.z;
            xn[4 * v + 3] = t.w;
        }
        float unscale = 1.f;
        if constexpr (kScaledX<T>) {
            uint32_t top = 0;
#pragma unroll
            for (int i = 0; i < 16; ++i) top = max(top, max(xn[i] & 0x7fffu, (xn[i] >> 16) & 0x7fffu));
            // below 32 (0x5000): a NaN or an infinity is not
            if (top < 0x5000u) {
                unscale = 0x1p-10f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    // widened (the conversion keeps a denormal half), scaled and narrowed again: exact
                    const pair p = __builtin_bit_cast(pair, xn[i]);
                    xn[i] = __builtin_bit_cast(uint32_t, __floats2half2_rn(T::lo(p) * 1024.f, T::hi(p) * 1024.f));
                }
            }
        }
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) s = stream_dot<T>(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, T::ones), s);
        sums[at] = s * unscale;
        if constexpr (kScaledX<T>) sums[R * nch + at] = unscale;
        pair xp[16];
        stream_x_pairs<Dec, T>(xn, xp, std::make_integer_sequence<int, 16>{});
        // word by word: a u32x4 built from the pairs' bit casts lost a lane on gfx1100 (hipcc 7.14)
#pragma unroll
        for (int i = 0; i < 16; ++i)
            pairs[((r * 4 + (i >> 2)) * nch + q) * 4 + (i & 3)] = __builtin_bit_cast(uint32_t, xp[i]);
    }
    __syncthreads();
}

// Epi::kPair: a lane's CB columns are CB / 2 gate columns and the same up columns; out is silu(gate) * up.
template <class Dec, class Act, class T, class Epi, int R, int CB, bool WIDE, bool SHARED>
__device__ inline void stream_body(const typename Dec::Args& a, int lpc_log2, int gshift, int bx, int pair_cols, float limit) {
    using pair = typename T::pair;
    constexpr bool PAIR = Epi::kPair;
    constexpr int RG = R < 4 ? R : 4;  // rows a pass keeps in registers
    constexpr int NG = R / RG;
    constexpr uint32_t kOne = T::ones;
    const int row0 = blockIdx.y * R;  // a block's rows: more than 8 rows are 8-row blocks on the grid's y
    const int lpc = 1 << lpc_log2;
    const int lane = threadIdx.x & 31;
    const int j = lane & (lpc - 1);
    const int slot = lane >> lpc_log2;
    constexpr int CH = PAIR ? CB / 2 : CB;  // output columns a lane
    const int cols = PAIR ? pair_cols : a.n;
    const int per_wave = (32 >> lpc_log2) * CH;
    const int col0 = (bx * kStreamWaves + (threadIdx.x >> 5)) * per_wave + slot * CH;
    const int nch = a.k >> 5;
    const long long words_row = Dec::template row_words<long long>(a);
    const long long groups = a.k / a.group;
    const uint32_t* wp[CB];
    long long sb[CB];
#pragma unroll
    for (int c = 0; c < CB; ++c) {
        const int cj = PAIR ? c % CH : c;
        const long long col = (col0 + cj < cols ? col0 + cj : cols - 1) + (PAIR && c >= CH ? pair_cols : 0);
        wp[c] = Dec::words(a) + col * words_row;
        sb[c] = col * groups;
    }
    const typename Act::elem* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) xr[r] = Act::row(a, row0 + r < a.m ? row0 + r : 0);
    float acc[R][CB];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < CB; ++c) acc[r][c] = 0.f;
    }

    // SHARED: the lanes read x pairs and sums of x from the block's stage instead of loading x (same values, same bits). On gfx900 at
    // one row the stage is written after the first round's weights are requested, so the weights' latency runs under the stage's.
#if defined(__gfx900__)
    constexpr bool kStageLate = R == 1;
#else
    constexpr bool kStageLate = false;
#endif
    extern __shared__ uint32_t stage[];
    if constexpr (SHARED && !kStageLate) stream_stage<Dec, Act, T, R>(a, row0, stage, reinterpret_cast<float*>(stage + R * nch * 16));
    const uint32_t* xs = stage;
    const float* sums = reinterpret_cast<const float*>(stage + R * nch * 16);
    const typename Dec::StreamTerms terms(a);

    // The loads of one round: every column's chunk and its scale and bias, none waited for before the dots.
    auto fetch = [&](int base, uint32_t (&w)[CB][Dec::kWords], unsigned short (&sr)[CB], unsigned short (&br)[CB]) {
        const int q = base + j < nch ? base + j : nch - 1;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            Dec::template load<WIDE>(wp[c] + static_cast<long long>(q) * Dec::kWords, w[c]);
            const long long at = sb[c] + (q >> gshift);
            sr[c] = terms.scale_raw(at);
            br[c] = terms.bias_raw(at);
        }
    };
    // The dots of one round. A lane past the row's last chunk read the last one: its scale and bias are zero.
    auto work = [&](auto staged, int base, const uint32_t (&w)[CB][Dec::kWords], const unsigned short (&sr)[CB],
                    const unsigned short (&br)[CB]) {
        constexpr bool ST = decltype(staged)::value;
        // kScaledX: the staged x are scaled by 2^10 (pairs), the others are widened to fp32
        const bool live = base + j < nch;
        const int q = live ? base + j : nch - 1;
        float sc[CB];
        float bi[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            sc[c] = live ? terms.widen(sr[c]) : 0.f;
            bi[c] = live ? terms.widen(br[c]) : 0.f;
            if constexpr (Dec::kDeferSub) bi[c] = fmaf(-Dec::kStreamMark, sc[c], bi[c]);
        }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            if (row0 + g * RG >= a.m) break;
            pair xp[RG][16];
            float sx[RG];
            float un[RG];
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                if constexpr (ST) {
                    const int row = g * RG + r;
#pragma unroll
                    for (int i = 0; i < 16; ++i)
                        xp[r][i] = __builtin_bit_cast(pair, xs[((row * 4 + (i >> 2)) * nch + q) * 4 + (i & 3)]);
                    sx[r] = sums[row * nch + q];
                    if constexpr (kScaledX<T>) un[r] = sums[R * nch + row * nch + q];
                } else {
                uint32_t xn[16];
                const u32x4* src =
                    reinterpret_cast<const u32x4*>(xr[g * RG + r] + static_cast<long long>(q) * 32);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const u32x4 t = src[v];
                    xn[4 * v] = t.x;
                    xn[4 * v + 1] = t.y;
                    xn[4 * v + 2] = t.z;
                    xn[4 * v + 3] = t.w;
                }
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    if constexpr (kScaledX<T>)
                        s = stream_dot<T>(stream_wide<T>(__builtin_bit_cast(pair, xn[i])), __builtin_bit_cast(pair, kOne), s);
                    else
                        s = stream_dot<T>(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, kOne), s);
                }
                sx[r] = s;
                un[r] = 1.f;
                stream_x_pairs<Dec, T>(xn, xp[r], std::make_integer_sequence<int, 16>{});
                }
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                float d[RG];
#pragma unroll
                for (int r = 0; r < RG; ++r) d[r] = 0.f;
                stream_dots<Dec, T, kScaledX<T> && !ST, RG>(xp, w[c], d, std::make_integer_sequence<int, 16>{});
#pragma unroll
                for (int r = 0; r < RG; ++r) {
                    acc[g * RG + r][c] = Dec::fold_scale(acc[g * RG + r][c], d[r], kScaledX<T> && ST ? sc[c] * un[r] : sc[c]);
                    acc[g * RG + r][c] = Dec::fold_bias(acc[g * RG + r][c], sx[r], bi[c]);
                }
            }
        }
    };

    // Columns stagger their start round by group of 8 to spread memory channels; a column's sum order ignores shape.
    const int rounds = (nch + lpc - 1) >> lpc_log2;
    const int rot = rounds > 1 ? (col0 >> 3) % rounds : 0;
    auto at = [&](int r) {
        const int ri = r + rot >= rounds ? r + rot - rounds : r + rot;
        return ri << lpc_log2;
    };
    uint32_t wa[CB][Dec::kWords];
    unsigned short sa[CB], ba[CB];
    // The rounds, with the stage (staged) or without it.
    uint32_t wb[CB][Dec::kWords];
    unsigned short sbb[CB], bbb[CB];
    auto run = [&](auto staged) {
        if constexpr (R == 1) {
            for (int r = 0; r < rounds; r += 2) {
                if (r + 1 < rounds) fetch(at(r + 1), wb, sbb, bbb);
                work(staged, at(r), wa, sa, ba);
                if (r + 1 >= rounds) break;
                if (r + 2 < rounds) fetch(at(r + 2), wa, sa, ba);
                work(staged, at(r + 1), wb, sbb, bbb);
            }
        } else {
            // several rows: the dots cover the loads' latency, and registers go to the rows' x
            for (int r = 0; r < rounds; ++r) {
                fetch(at(r), wa, sa, ba);
                work(staged, at(r), wa, sa, ba);
            }
        }
    };
    if constexpr (R == 1) fetch(at(0), wa, sa, ba);
    if constexpr (SHARED && kStageLate) stream_stage<Dec, Act, T, R>(a, row0, stage, reinterpret_cast<float*>(stage + R * nch * 16));
    run(std::bool_constant<SHARED>{});

#pragma unroll
    for (int r = 0; r < R; ++r) {
        float v[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            v[c] = acc[r][c];
            for (int d = lpc >> 1; d > 0; d >>= 1) v[c] += __shfl_xor(v[c], d, 32);
        }
        if (j != 0 || row0 + r >= a.m) continue;
#pragma unroll
        for (int c = 0; c < CH; ++c) {
            const int col = col0 + c;
            if (col >= cols) continue;
            Epi::put(a, out_row(a, row0 + r), col, v, c, pair_cols, limit);
        }
    }
}

// Item z of a plan (or product, or group side), CB columns a lane, R rows, chunk group q >> gshift; `shared` stages x.
template <class Dec, class Act, class T, class Epi, int R, int CB, class Sides>
__device__ inline void stream_tile(typename Dec::Args& a, const Sides& sides, int lpc_log2, int gshift, int shared) {
    int bx = blockIdx.x;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        bx -= sides.first[s];
        Dec::bind(a, sides, s);
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    if (!Dec::take_item(a, blockIdx.z) || static_cast<int>(blockIdx.y) * R >= a.m) return;
    // every chunk is wide-aligned when the words and the row stride are
    constexpr bool kWide = Dec::kWideBytes != 4;
    const bool wide = kWide && Dec::wide_aligned(a);
    const int pc = sides.pair_cols;
    if constexpr (R <= 2) {
        if constexpr (kWide) {
            if (shared && wide)
                return stream_body<Dec, Act, T, Epi, R, CB, true, true>(a, lpc_log2, gshift, bx, pc, sides.limit);
        }
        if (shared) return stream_body<Dec, Act, T, Epi, R, CB, false, true>(a, lpc_log2, gshift, bx, pc, sides.limit);
    }
    if constexpr (kWide) {
        if (wide) return stream_body<Dec, Act, T, Epi, R, CB, true, false>(a, lpc_log2, gshift, bx, pc, sides.limit);
    }
    stream_body<Dec, Act, T, Epi, R, CB, false, false>(a, lpc_log2, gshift, bx, pc, sides.limit);
}

}  // namespace rocm
}  // namespace tf
