#pragma once

// Decode tile for 1 to 8 rows that streams the weights: a lane owns a 32-code chunk of a column's row, so a wave's
// load is a contiguous run of one row, and the rows of x it needs stay in registers. y = sum over chunks of
// scale * dot(x, code) + bias * sum(x), so a chunk needs no group alignment: its lane loads its own scale and bias.
// Lanes a column (lpc, a power of two up to 32) walk the row in rounds; each lane carries CB columns, all of whose
// loads are issued before the first dot. No LDS, no barrier: the column's lanes fold with shuffles at the end.
// The codes become pairs with a few bit operations (0x6400 | code is 1024 + code in FP16, the subtraction is exact).

#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>

#include "tiles/dot2.hpp"

namespace tf {
namespace rocm {

using stream_u32x4 = unsigned __attribute__((ext_vector_type(4)));

constexpr int kStreamWaves = 4;

// Tile shapes that exist: CB columns a lane times R rows at most 16 accumulators.
constexpr bool stream_shape(int rows, int cb) { return cb * rows <= 16; }
constexpr int kStreamRows = 16;  // the most rows a launch takes

// Codes of pair i of a 32-code chunk: codes 16 bits apart in one word when the width divides 16, else neighbours.
template <int BITS>
constexpr int stream_lo(int i) {
    if constexpr (BITS == 4) return (i >> 2) * 8 + (i & 3);
    else if constexpr (BITS == 8) return (i >> 1) * 4 + (i & 1);
    else if constexpr (BITS == 2) return (i >> 3) * 16 + (i & 7);
    else return 2 * i;
}

template <int BITS>
constexpr int stream_gap() {
    if constexpr (BITS == 4) return 4;
    else if constexpr (BITS == 8) return 2;
    else if constexpr (BITS == 2) return 8;
    else return 1;
}

// Pair i of the chunk's codes as the activation type's pair (the codes are below 256, exact in both types).
template <typename T, int BITS, int I>
__device__ inline typename T::pair stream_codes(const uint32_t (&w)[BITS]) {
    constexpr int lo = stream_lo<BITS>(I);
    constexpr int hi = lo + stream_gap<BITS>();
    uint32_t bits;
    if constexpr (BITS == 2 || BITS == 4 || BITS == 8) {
        constexpr int per = 32 / BITS;
        constexpr uint32_t mask = ((1u << BITS) - 1u) * 0x00010001u;
        bits = (w[lo / per] >> (BITS * (lo % per))) & mask;
    } else {
        bits = piece_bits<BITS>(w, lo) | (piece_bits<BITS>(w, hi) << 16);
    }
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        bits |= 0x64006400u;
        return __hsub2(__builtin_bit_cast(__half2, bits), __builtin_bit_cast(__half2, 0x64006400u));
    } else {
        // The float of an integer below 256 has its low 16 bits clear, so its top half is the BF16.
        const uint32_t a = __builtin_bit_cast(uint32_t, static_cast<float>(bits & 0xffffu));
        const uint32_t b = __builtin_bit_cast(uint32_t, static_cast<float>(bits >> 16));
        return __builtin_bit_cast(bf16x2, (a >> 16) | (b & 0xffff0000u));
    }
}

// A 32-code chunk is BITS words: 16-byte loads when BITS % 4 == 0 and WIDE, 8-byte when even, else by word.
template <int BITS, bool WIDE>
__device__ inline void stream_load(const uint32_t* src, uint32_t (&w)[BITS]) {
    if constexpr (WIDE && BITS % 4 == 0) {
#pragma unroll
        for (int i = 0; i < BITS / 4; ++i) {
            const stream_u32x4 v = reinterpret_cast<const stream_u32x4*>(src)[i];
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    } else if constexpr (WIDE && BITS % 2 == 0) {
#pragma unroll
        for (int i = 0; i < BITS / 2; ++i) {
            const uint2 v = reinterpret_cast<const uint2*>(src)[i];
            w[2 * i] = v.x;
            w[2 * i + 1] = v.y;
        }
    } else {
#pragma unroll
        for (int i = 0; i < BITS; ++i) w[i] = src[i];
    }
}

// The pair (half `parity` of a, half `parity` of b).
__device__ inline uint32_t stream_pick(uint32_t a, uint32_t b, int parity) {
    return __builtin_amdgcn_perm(b, a, parity ? 0x07060302u : 0x05040100u);
}

// x of one row in the order of the code pairs: pair I takes the halves of the codes it is multiplied with.
template <typename T, int BITS, int... I>
__device__ inline void stream_x_pairs(const uint32_t (&xn)[16], typename T::pair (&xp)[16],
                                      std::integer_sequence<int, I...>) {
    (([&] {
         constexpr int lo = stream_lo<BITS>(I);
         constexpr int hi = lo + stream_gap<BITS>();
         if constexpr (BITS == 2 || BITS == 4 || BITS == 8) {
             xp[I] = __builtin_bit_cast(typename T::pair, stream_pick(xn[lo >> 1], xn[hi >> 1], lo & 1));
         } else {
             xp[I] = __builtin_bit_cast(typename T::pair, xn[I]);
         }
     }()),
     ...);
}

// One column's chunk against RG rows: each code pair is made once and dotted with every row's x.
template <typename T, int BITS, int RG, int... I>
__device__ inline void stream_dots(const typename T::pair (&xp)[RG][16], const uint32_t (&w)[BITS], float (&d)[RG],
                                   std::integer_sequence<int, I...>) {
    (([&] {
         const typename T::pair q = stream_codes<T, BITS, I>(w);
#pragma unroll
         for (int r = 0; r < RG; ++r) d[r] = T::dot(xp[r][I], q, d[r]);
     }()),
     ...);
}

// A float as the activation type, rounded to nearest even.
template <typename T>
__device__ inline typename T::elem stream_narrow(float v) {
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        return __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        const uint32_t bits = v != v ? 0x7fc0u : (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
        return __builtin_bit_cast(typename T::elem, static_cast<unsigned short>(bits));
    }
}

// A 16-bit table entry (BF16 or FP16) as a float.
__device__ inline float stream_table(unsigned short raw, bool half_tab) {
    return half_tab ? __half2float(__ushort_as_half(raw)) : __uint_as_float(static_cast<uint32_t>(raw) << 16);
}

// Up to four products that share x, K, width and group in one launch: block bx of the launch belongs to the side whose
// blocks [first[s], first[s + 1]) hold it. out is each side's (rows, n) output: fp32, or the activation type when out_half.
constexpr int kStreamSides = 4;

struct StreamSides {
    const uint32_t* words[kStreamSides];
    const void* scale[kStreamSides];
    const void* bias[kStreamSides];
    void* out[kStreamSides];
    int n[kStreamSides];
    int first[kStreamSides + 1];
    int count;  // 0 is the plain product of the Affine
    int out_half;
    int pair_cols;  // > 0: the (gate | up) rows of a stacked product with this many columns each, out is its activation
    float limit;    // the activation's clamp when above 0
};

// silu(gate) * up, both clamped by limit first when it is above 0 (moe_act_kernel's arithmetic).
__device__ inline float stream_act(float g, float u, float limit) {
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return g / (1.f + expf(-g)) * u;
}

// PAIR: a lane's CB columns are CB / 2 gate columns and the same columns of the up half (rows pair_cols on), and the
// output is silu(gate) * up in the activation type, (rows, pair_cols).
template <typename T, int BITS, int R, int CB, bool WIDE, bool PAIR>
__device__ inline void stream_body(const Affine& a, int lpc_log2, int gshift, int bx, int pair_cols, float limit) {
    using pair = typename T::pair;
    constexpr int RG = R < 4 ? R : 4;  // rows a pass keeps in registers
    constexpr int NG = R / RG;
    constexpr uint32_t kOne = std::is_same_v<typename T::elem, __half> ? 0x3c003c00u : 0x3f803f80u;
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
    const long long words_row = static_cast<long long>(a.k) * BITS / 32;
    const long long groups = a.k / a.group;
    const uint32_t* wp[CB];
    long long sb[CB];
#pragma unroll
    for (int c = 0; c < CB; ++c) {
        const int cj = PAIR ? c % CH : c;
        const long long col = (col0 + cj < cols ? col0 + cj : cols - 1) + (PAIR && c >= CH ? pair_cols : 0);
        wp[c] = a.words + col * words_row;
        sb[c] = col * groups;
    }
    const typename T::elem* xr[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        xr[r] = static_cast<const typename T::elem*>(a.x) + x_row(a, row0 + r < a.m ? row0 + r : 0) * a.k;
    }
    float acc[R][CB];
#pragma unroll
    for (int r = 0; r < R; ++r) {
#pragma unroll
        for (int c = 0; c < CB; ++c) acc[r][c] = 0.f;
    }

    const unsigned short* stab = static_cast<const unsigned short*>(a.scale.p);
    const unsigned short* btab = static_cast<const unsigned short*>(a.bias.p);
    const bool half_tab = a.scale.kind == kScaleF16;

    // The loads of one round: every column's chunk and its scale and bias, none waited for before the dots.
    auto fetch = [&](int base, uint32_t (&w)[CB][BITS], unsigned short (&sr)[CB], unsigned short (&br)[CB]) {
        const int q = base + j < nch ? base + j : nch - 1;
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            stream_load<BITS, WIDE>(wp[c] + static_cast<long long>(q) * BITS, w[c]);
            const long long at = sb[c] + (q >> gshift);
            sr[c] = stab[at];
            br[c] = btab[at];
        }
    };
    // The dots of one round. A lane past the row's last chunk read the last one: its scale and bias are zero.
    auto work = [&](int base, const uint32_t (&w)[CB][BITS], const unsigned short (&sr)[CB],
                    const unsigned short (&br)[CB]) {
        const bool live = base + j < nch;
        const int q = live ? base + j : nch - 1;
        float sc[CB];
        float bi[CB];
#pragma unroll
        for (int c = 0; c < CB; ++c) {
            sc[c] = live ? stream_table(sr[c], half_tab) : 0.f;
            bi[c] = live ? stream_table(br[c], half_tab) : 0.f;
        }
#pragma unroll
        for (int g = 0; g < NG; ++g) {
            if (row0 + g * RG >= a.m) break;
            pair xp[RG][16];
            float sx[RG];
#pragma unroll
            for (int r = 0; r < RG; ++r) {
                uint32_t xn[16];
                const stream_u32x4* src =
                    reinterpret_cast<const stream_u32x4*>(xr[g * RG + r] + static_cast<long long>(q) * 32);
#pragma unroll
                for (int v = 0; v < 4; ++v) {
                    const stream_u32x4 t = src[v];
                    xn[4 * v] = t.x;
                    xn[4 * v + 1] = t.y;
                    xn[4 * v + 2] = t.z;
                    xn[4 * v + 3] = t.w;
                }
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    s = T::dot(__builtin_bit_cast(pair, xn[i]), __builtin_bit_cast(pair, kOne), s);
                }
                sx[r] = s;
                stream_x_pairs<T, BITS>(xn, xp[r], std::make_integer_sequence<int, 16>{});
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                float d[RG];
#pragma unroll
                for (int r = 0; r < RG; ++r) d[r] = 0.f;
                stream_dots<T, BITS, RG>(xp, w[c], d, std::make_integer_sequence<int, 16>{});
#pragma unroll
                for (int r = 0; r < RG; ++r) {
                    acc[g * RG + r][c] = fmaf(d[r], sc[c], acc[g * RG + r][c]);
                    acc[g * RG + r][c] = fmaf(sx[r], bi[c], acc[g * RG + r][c]);
                }
            }
        }
    };

    // One row keeps two rounds of loads in flight: the next is fetched before this one's dots. Columns start on different
    // rounds, by their group of 8 (so a column's sum has one order whatever the launch's shape): rows a power of two bytes
    // apart would otherwise send every wave to the same memory channels at once.
    const int rounds = (nch + lpc - 1) >> lpc_log2;
    const int rot = rounds > 1 ? (col0 >> 3) % rounds : 0;
    auto at = [&](int r) {
        const int ri = r + rot >= rounds ? r + rot - rounds : r + rot;
        return ri << lpc_log2;
    };
    uint32_t wa[CB][BITS];
    unsigned short sa[CB], ba[CB];
    if constexpr (R == 1) {
        uint32_t wb[CB][BITS];
        unsigned short sbb[CB], bbb[CB];
        fetch(at(0), wa, sa, ba);
        for (int r = 0; r < rounds; r += 2) {
            if (r + 1 < rounds) fetch(at(r + 1), wb, sbb, bbb);
            work(at(r), wa, sa, ba);
            if (r + 1 >= rounds) break;
            if (r + 2 < rounds) fetch(at(r + 2), wa, sa, ba);
            work(at(r + 1), wb, sbb, bbb);
        }
    } else {
        // several rows: the dots cover the loads' latency, and registers go to the rows' x
        for (int r = 0; r < rounds; ++r) {
            fetch(at(r), wa, sa, ba);
            work(at(r), wa, sa, ba);
        }
    }

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
            if constexpr (PAIR) {
                static_cast<typename T::elem*>(static_cast<void*>(a.out16))[out_row(a, row0 + r) * pair_cols + col] =
                    stream_narrow<T>(stream_act(v[c], v[c + CH], limit));
            } else if (a.out16 != nullptr) {
                static_cast<typename T::elem*>(static_cast<void*>(a.out16))[out_row(a, row0 + r) * a.n + col] =
                    stream_narrow<T>(v[c]);
            } else {
                a.out[out_row(a, row0 + r) * a.n + col] = v[c];
            }
        }
    }
}

// Item z of the plan (or the plain product, or one side of a group) at CB columns a lane and R rows; the chunks'
// group is q >> gshift.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream(Affine a, StreamSides sides, int lpc_log2,
                                                                       int gshift) {
    int bx = blockIdx.x;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        bx -= sides.first[s];
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
        a.bias.p = sides.bias[s];
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    if (!take_item(a, blockIdx.z) || static_cast<int>(blockIdx.y) * R >= a.m) return;
    constexpr int kWide = BITS % 4 == 0 ? 16 : BITS % 2 == 0 ? 8 : 4;
    if constexpr (kWide == 4) {
        stream_body<T, BITS, R, CB, false, PAIR>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
    } else {
        // every chunk is wide-aligned when the words and the row stride are
        const long long words_row = static_cast<long long>(a.k) * BITS / 32;
        const bool wide = ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(words_row * 4)) & (kWide - 1)) == 0;
        if (wide) {
            stream_body<T, BITS, R, CB, true, PAIR>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
        } else {
            stream_body<T, BITS, R, CB, false, PAIR>(a, lpc_log2, gshift, bx, sides.pair_cols, sides.limit);
        }
    }
}

// Waves a launch should have to hide the weight loads' latency, and the most code words a lane keeps in flight: a lane
// with more (8 columns of 8-bit codes) leaves the SIMD too few waves, and its loads run at 80% of what 2 or 4 columns reach.
constexpr int kStreamWavesWanted = 3000;
constexpr int kStreamCodeWords = 32;

template <typename T, int BITS, int R, int CB>
hipError_t stream_launch_as(const Affine& a, int lpc_log2, int items, hipStream_t stream) {
    const int per_block = kStreamWaves * (32 >> lpc_log2) * CB;
    const dim3 grid((a.n + per_block - 1) / per_block, (a.m + R - 1) / R, items);
    const int gshift = a.group == 32 ? 0 : a.group == 64 ? 1 : 2;
    StreamSides plain = {};
    affine_dot2_stream<T, BITS, R, CB, false><<<grid, 32 * kStreamWaves, 0, stream>>>(a, plain, lpc_log2, gshift);
    return hipGetLastError();
}

// The widest column count a lane carries that still gives the card enough waves.
template <int R>
int stream_cb(const Affine& a, int lpc_log2, int items) {
    const int cbs[3] = {8, 4, 2};
    int pick = 2;
    for (int cb : cbs) {
        if (!stream_shape(R, cb) || cb * a.bits > kStreamCodeWords) continue;
        pick = cb;
        const int per_block = kStreamWaves * (32 >> lpc_log2) * cb;
        if (((a.n + per_block - 1) / per_block) * kStreamWaves * items >= kStreamWavesWanted) break;
    }
    return pick;
}

template <typename T, int BITS, int R>
hipError_t stream_launch_rows(const Affine& a, int lpc_log2, int items, hipStream_t stream) {
    const int cb = stream_cb<R>(a, lpc_log2, items * ((a.m + R - 1) / R));
    if constexpr (stream_shape(R, 8)) {
        if (cb == 8) return stream_launch_as<T, BITS, R, 8>(a, lpc_log2, items, stream);
    }
    if constexpr (stream_shape(R, 4)) {
        if (cb == 4) return stream_launch_as<T, BITS, R, 4>(a, lpc_log2, items, stream);
    }
    return stream_launch_as<T, BITS, R, 2>(a, lpc_log2, items, stream);
}

template <typename T, int BITS>
hipError_t stream_launch_bits(const Affine& a, int lpc_log2, int items, hipStream_t stream) {
    if (a.m == 1) return stream_launch_rows<T, BITS, 1>(a, lpc_log2, items, stream);
    if (a.m == 2) return stream_launch_rows<T, BITS, 2>(a, lpc_log2, items, stream);
    return stream_launch_rows<T, BITS, 4>(a, lpc_log2, items, stream);
}

template <int BITS>
hipError_t stream_launch_type(const Affine& a, int lpc_log2, int items, hipStream_t stream) {
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) return stream_launch_bits<DotBF16, BITS>(a, lpc_log2, items, stream);
#endif
    return stream_launch_bits<DotF16, BITS>(a, lpc_log2, items, stream);
}

// The activation of a stacked (gate | up) product: out (rows, pair_cols) = silu(gate) * up in the activation type, the
// kernel's `a.n` the stacked width. Same columns a lane, half of them gate and half up.
template <typename T, int BITS, int R, int CB>
hipError_t stream_pair_as(const Affine& a, int pair_cols, float limit, int lpc_log2, int items, hipStream_t stream) {
    const int per_block = kStreamWaves * (32 >> lpc_log2) * (CB / 2);
    const dim3 grid((pair_cols + per_block - 1) / per_block, (a.m + R - 1) / R, items);
    StreamSides sides = {};
    sides.pair_cols = pair_cols;
    sides.limit = limit;
    const int gshift = a.group == 32 ? 0 : a.group == 64 ? 1 : 2;
    affine_dot2_stream<T, BITS, R, CB, true><<<grid, 32 * kStreamWaves, 0, stream>>>(a, sides, lpc_log2, gshift);
    return hipGetLastError();
}

template <typename T, int BITS, int R>
hipError_t stream_pair_rows(const Affine& a, int pair_cols, float limit, int lpc_log2, int items, hipStream_t stream) {
    // the widest columns a lane carries (a gate and an up for each pair) that still give the card enough waves
    const int cbs[3] = {8, 4, 2};
    int cb = 2;
    for (int c : cbs) {
        if (!stream_shape(R, c) || c * BITS > kStreamCodeWords) continue;
        cb = c;
        const int per_block = kStreamWaves * (32 >> lpc_log2) * (c / 2);
        if (((pair_cols + per_block - 1) / per_block) * kStreamWaves * items * ((a.m + R - 1) / R) >= kStreamWavesWanted) {
            break;
        }
    }
    if constexpr (stream_shape(R, 8)) {
        if (cb == 8) return stream_pair_as<T, BITS, R, 8>(a, pair_cols, limit, lpc_log2, items, stream);
    }
    if constexpr (stream_shape(R, 4)) {
        if (cb == 4) return stream_pair_as<T, BITS, R, 4>(a, pair_cols, limit, lpc_log2, items, stream);
    }
    return stream_pair_as<T, BITS, R, 2>(a, pair_cols, limit, lpc_log2, items, stream);
}

template <typename T, int BITS>
hipError_t stream_pair_bits(const Affine& a, int pair_cols, float limit, int lpc_log2, int items, hipStream_t stream) {
    if (a.m == 1) return stream_pair_rows<T, BITS, 1>(a, pair_cols, limit, lpc_log2, items, stream);
    if (a.m == 2) return stream_pair_rows<T, BITS, 2>(a, pair_cols, limit, lpc_log2, items, stream);
    return stream_pair_rows<T, BITS, 4>(a, pair_cols, limit, lpc_log2, items, stream);
}

template <int BITS>
hipError_t stream_pair_type(const Affine& a, int pair_cols, float limit, int lpc_log2, int items, hipStream_t stream) {
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) return stream_pair_bits<DotBF16, BITS>(a, pair_cols, limit, lpc_log2, items, stream);
#endif
    return stream_pair_bits<DotF16, BITS>(a, pair_cols, limit, lpc_log2, items, stream);
}

// TF_AFFINE_GEMV=old keeps the previous decode tiles.
inline bool stream_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("TF_AFFINE_GEMV");
        return v == nullptr || std::strcmp(v, "old") != 0;
    }();
    return on;
}

// The stream tile of 1 to 8 rows; false when the shape keeps the previous tiles.
inline bool launch_affine_dot2_stream(const Affine& a, hipStream_t stream, int items, hipError_t* err) {
    if (!stream_enabled() || a.m < 1 || a.m > kStreamRows || a.n < 1 || a.group % 32 || a.group > kLaneGroupMax ||
        a.k % a.group || (reinterpret_cast<uintptr_t>(a.x) & 15) != 0 || a.scale.kind == kScaleF32 ||
        a.bias.kind != a.scale.kind) {
        return false;
    }
    int lpc_log2 = 0;
    while ((1 << lpc_log2) < (a.k >> 5) && lpc_log2 < 5) ++lpc_log2;
    switch (a.bits) {
        case 2: *err = stream_launch_type<2>(a, lpc_log2, items, stream); break;
        case 3: *err = stream_launch_type<3>(a, lpc_log2, items, stream); break;
        case 4: *err = stream_launch_type<4>(a, lpc_log2, items, stream); break;
        case 5: *err = stream_launch_type<5>(a, lpc_log2, items, stream); break;
        case 6: *err = stream_launch_type<6>(a, lpc_log2, items, stream); break;
        case 8: *err = stream_launch_type<8>(a, lpc_log2, items, stream); break;
        default: return false;
    }
    return true;
}

// out (m, n / 2) = silu(gate) * up of the stacked (gate | up) product of `a`, in the activation type; false when the
// shape keeps the separate products. Plain or routed (items).
inline bool launch_affine_dot2_stream_pair(const Affine& a, float limit, hipStream_t stream, int items, hipError_t* err) {
    if (a.n % 2 != 0 || a.out16 == nullptr || a.n < 2 || !stream_enabled() || a.m < 1 || a.m > kStreamRows ||
        a.group % 32 || a.group > kLaneGroupMax || a.k % a.group || (reinterpret_cast<uintptr_t>(a.x) & 15) != 0 ||
        a.scale.kind == kScaleF32 || a.bias.kind != a.scale.kind) {
        return false;
    }
    int lpc_log2 = 0;
    while ((1 << lpc_log2) < (a.k >> 5) && lpc_log2 < 5) ++lpc_log2;
    const int cols = a.n / 2;
    switch (a.bits) {
        case 2: *err = stream_pair_type<2>(a, cols, limit, lpc_log2, items, stream); break;
        case 3: *err = stream_pair_type<3>(a, cols, limit, lpc_log2, items, stream); break;
        case 4: *err = stream_pair_type<4>(a, cols, limit, lpc_log2, items, stream); break;
        case 5: *err = stream_pair_type<5>(a, cols, limit, lpc_log2, items, stream); break;
        case 6: *err = stream_pair_type<6>(a, cols, limit, lpc_log2, items, stream); break;
        case 8: *err = stream_pair_type<8>(a, cols, limit, lpc_log2, items, stream); break;
        default: return false;
    }
    return true;
}

}  // namespace rocm
}  // namespace tf
