#pragma once

// The dot types of the dot2 tiles; a code comes from the quant header.

#include <hip/hip_fp16.h>

#include "common/arch.hpp"
#include "quant/mlx_pieces.hpp"

namespace tf {
namespace rocm {

// Activation types of the tiles: FP16 v_dot2_f32_f16 (RDNA2) and BF16 v_dot2_f32_bf16 (gfx11 / gfx12).
struct DotF16 {
    using elem = __half;
    using pair = __half2;
    __device__ static elem zero() { return __float2half(0.f); }
    __device__ static pair two(elem a, elem b) { return __halves2half2(a, b); }
    __device__ static float lo(pair p) { return __low2float(p); }
    __device__ static float hi(pair p) { return __high2float(p); }
    template <int BITS>
    __device__ static elem code(const uint32_t (&w)[BITS], int t) { return piece_code<BITS>(w, t); }
    __device__ static float dot(pair x, pair q, float acc) { return __builtin_amdgcn_fdot2(x, q, acc, false); }
};

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

struct DotBF16 {
    using elem = __bf16;
    using pair = bf16x2;
    __device__ static elem zero() { return static_cast<__bf16>(0.f); }
    __device__ static pair two(elem a, elem b) { return pair{a, b}; }
    __device__ static float lo(pair p) { return static_cast<float>(p.x); }
    __device__ static float hi(pair p) { return static_cast<float>(p.y); }
    template <int BITS>
    __device__ static elem code(const uint32_t (&w)[BITS], int t) {
        return static_cast<__bf16>(static_cast<float>(piece_bits<BITS>(w, t)));
    }
    __device__ static float dot(pair x, pair q, float acc) {
#if TF_DEVICE_BF16_DOT2
        return __builtin_amdgcn_fdot2_f32_bf16(x, q, acc, false);
#else
        __builtin_trap();
        return acc;
#endif
    }
};

}  // namespace rocm
}  // namespace tf
