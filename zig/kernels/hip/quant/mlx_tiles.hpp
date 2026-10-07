#pragma once

// The MLX affine instantiations of the shared tiles: each kernel here is a tile (tiles/*.hpp) over the MLX decoder, the
// identity encoder and the Dot of the activation type, under the symbol name the launchers look up, plus the host
// launch helpers of the C library. A format adds a header like this one beside its decoder.

#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/gemm_kp.hpp"
#include "tiles/matrix_gemm.hpp"

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

}  // namespace rocm
}  // namespace tf
