#pragma once

// The MLX affine instantiations of the shared tiles: each kernel is a tile (tiles/*.hpp) over the MLX decoder, the identity
// encoder and the activation type's Dot, under the symbol the launchers look up, plus the C library's host launch helpers.

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/dot2.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/matrix_gemm.hpp"
#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

// ---- the prefill GEMM tile ----

template <typename T, int BITS, int RT>
__global__ void __launch_bounds__(256) affine_gemm_block(Affine a) {
    gemm_tile<MlxDecoder<BITS>, IdentityAct<T>, T, F32Out, RT>(a);
}

template <int BITS>
hipError_t launch_gemm_bits(const Affine& a, hipStream_t stream, int items) {
    const dim3 block(256);
    const dim3 grid((a.n + kGemmN - 1) / kGemmN, (a.m + kGemmM - 1) / kGemmM, items);
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) {
        affine_gemm_block<DotBF16, BITS, GemmShape<DotBF16>::rt><<<grid, block, 0, stream>>>(a);
        return hipGetLastError();
    }
#endif
    affine_gemm_block<DotF16, BITS, GemmShape<DotF16>::rt><<<grid, block, 0, stream>>>(a);
    return hipGetLastError();
}

// The shapes and pointers the GEMM tile takes: its piece loads need the words aligned to their width.
inline bool affine_gemm_supported(const Affine& a) {
    const bool alike = a.scale.kind == a.bias.kind;
    int align = 4;
    switch (a.bits) {
        case 2: case 6: align = 8; break;
        case 4: case 8: align = 16; break;
        case 3: case 5: break;
        default: return false;
    }
    return alike && a.m >= 1 && a.n >= 1 && a.group % kGemmK == 0 && a.k % a.group == 0 &&
           reinterpret_cast<uintptr_t>(a.words) % align == 0;
}

inline hipError_t launch_affine_gemm(const Affine& a, hipStream_t stream, int items = 1) {
    switch (a.bits) {
        case 2: return launch_gemm_bits<2>(a, stream, items);
        case 3: return launch_gemm_bits<3>(a, stream, items);
        case 4: return launch_gemm_bits<4>(a, stream, items);
        case 5: return launch_gemm_bits<5>(a, stream, items);
        case 6: return launch_gemm_bits<6>(a, stream, items);
        case 8: return launch_gemm_bits<8>(a, stream, items);
        default: return hipErrorInvalidValue;
    }
}

// ---- the K-parallel tiles of a short prompt ----

template <typename T, int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
__global__ void __launch_bounds__(32 * WAVES) affine_gemm_kp(Affine a) {
    gemm_kp_tile<MlxDecoder<BITS>, IdentityAct<T>, T, F32Out, CB, R, WAVES, RB, LOOP>(a);
}

template <int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
hipError_t launch_kp_tier(const Affine& a, hipStream_t stream, int items) {
    const int blocks = (a.m + R - 1) / R < RB ? (a.m + R - 1) / R : RB;
    const int cols = WAVES * CB * (32 / kp_lanes(a.k / a.group));
    const dim3 grid(((a.n + cols - 1) / cols) * blocks, 1, items);
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) {
        affine_gemm_kp<DotBF16, BITS, CB, R, WAVES, RB, LOOP><<<grid, dim3(32 * WAVES), 0, stream>>>(a);
        return hipGetLastError();
    }
#endif
    affine_gemm_kp<DotF16, BITS, CB, R, WAVES, RB, LOOP><<<grid, dim3(32 * WAVES), 0, stream>>>(a);
    return hipGetLastError();
}

// The shapes of launch/affine.zig's kp_tiles, by rows.
template <int BITS>
hipError_t launch_kp_bits(const Affine& a, hipStream_t stream, int items) {
    if (a.m <= 2) return launch_kp_tier<BITS, 8, 1, 2, 2, false>(a, stream, items);
    if (a.m <= 4) return launch_kp_tier<BITS, 4, 4, 2, 4, true>(a, stream, items);
    if (a.m <= 8) return launch_kp_tier<BITS, 4, 2, 2, 4, true>(a, stream, items);
    if (a.m <= 32) return launch_kp_tier<BITS, 4, 8, 2, 4, false>(a, stream, items);
    return hipErrorInvalidValue;
}

// ---- the prefill GEMM tile on the matrix cores of gfx11 ----

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_gemm(Affine a) {
    wmma_gemm_tile<MlxDecoder<BITS>, BF16Act, WmmaBF16, F32Out>(a);
}

template <int BITS>
hipError_t launch_wmma_gemm_bits(const Affine& a, hipStream_t stream, int items) {
    const dim3 grid((a.n + kGemmN - 1) / kGemmN, (a.m + kGemmM - 1) / kGemmM, items);
    affine_wmma_gemm<BITS><<<grid, dim3(256), 0, stream>>>(a);
    return hipGetLastError();
}

inline hipError_t launch_affine_wmma_gemm(const Affine& a, hipStream_t stream, int items = 1) {
    switch (a.bits) {
        case 2: return launch_wmma_gemm_bits<2>(a, stream, items);
        case 3: return launch_wmma_gemm_bits<3>(a, stream, items);
        case 4: return launch_wmma_gemm_bits<4>(a, stream, items);
        case 5: return launch_wmma_gemm_bits<5>(a, stream, items);
        case 6: return launch_wmma_gemm_bits<6>(a, stream, items);
        case 8: return launch_wmma_gemm_bits<8>(a, stream, items);
        default: return hipErrorInvalidValue;
    }
}

// ---- the decode stream tile ----

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

// Item z of the plan (or the plain product, or one side of a group) at CB columns a lane and R rows; PAIR is the routed gate
// and up with the activation as its epilogue.
template <typename T, int BITS, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) affine_dot2_stream(Affine a, StreamSides sides, int lpc_log2,
                                                                       int gshift) {
    using Epi = std::conditional_t<PAIR, StreamSwiglu<T>, StreamRound<T>>;
    stream_tile<MlxDecoder<BITS>, IdentityAct<T>, T, Epi, R, CB>(a, sides, lpc_log2, gshift);
}

// Waves a launch needs to hide the weight loads' latency, and the most code words a lane keeps in flight (more leaves the SIMD too few waves).
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

// TF_AFFINE_GEMV=old keeps the reference decode tiles.
inline bool stream_enabled() {
    static const bool on = [] {
        const char* v = std::getenv("TF_AFFINE_GEMV");
        return v == nullptr || std::strcmp(v, "old") != 0;
    }();
    return on;
}

// The stream tile of 1 to 8 rows; false when the shape keeps the reference tiles.
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
