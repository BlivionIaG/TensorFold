#pragma once

// The Dot plug-ins of the tiles: the multiply-accumulate unit of an activation type and its operands.

#include <hip/hip_fp16.h>

#include "common/arch.hpp"
#include "common/fdot2.hpp"

namespace tf {
namespace rocm {

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));

// One dot2 whose x pair is lane I of the lane's quad (DPP quad_perm), as a pair of 32-bit words.
template <typename T, int I>
__device__ inline float dot_quad_shared(uint32_t x, uint32_t w, float acc) {
    const uint32_t shared = __builtin_amdgcn_mov_dpp(static_cast<int>(x), I * 0x55, 0xf, 0xf, false);
    return T::dot(__builtin_bit_cast(typename T::pair, shared), __builtin_bit_cast(typename T::pair, w), acc);
}

// FP16 v_dot2_f32_f16 (RDNA2 and gfx11 / gfx12).
struct DotF16 {
    using elem = __half;
    using pair = __half2;
    // A pair of 1.0: dotted with x it gives the sum of x.
    static constexpr uint32_t ones = 0x3c003c00u;
    __device__ static elem zero() { return __float2half(0.f); }
    __device__ static pair two(elem a, elem b) { return __halves2half2(a, b); }
    __device__ static float lo(pair p) { return __low2float(p); }
    __device__ static float hi(pair p) { return __high2float(p); }
    __device__ static float dot(pair x, pair q, float acc) { return TF_FDOT2(x, q, acc); }
#if TF_STREAM_FLUSH_MIX
    // gfx900, TF_G9_MIX=2: two v_mad_mix_f32 on the fp16 pair of x (its denormal halves read as zero)
    __device__ static float dot_stream(pair x, pair q, float acc) {
        const uint32_t xb = __builtin_bit_cast(uint32_t, x);
        const uint32_t qb = __builtin_bit_cast(uint32_t, q);
        float t, r;
        asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,1,0]" : "=v"(t) : "v"(xb), "v"(qb), "v"(acc));
        asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[1,1,0] op_sel_hi:[1,1,0]" : "=v"(r) : "v"(xb), "v"(qb), "v"(t));
        return r;
    }
#endif
#if TF_STREAM_WIDE_X
    // gfx900: the stream tile's x is two fp32 (a denormal fp16 half would be flushed by the mix), q the pair of codes: two v_mad_mix_f32,
    // each product exact in fp32, so the bits are those of the convert-and-fuse pair.
    static constexpr bool kStreamWideX = true;
    __device__ static float dot_stream_wide(float xl, float xh, pair q, float acc) {
        const uint32_t qb = __builtin_bit_cast(uint32_t, q);
        float t, r;
        asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel_hi:[0,1,0]" : "=v"(t) : "v"(xl), "v"(qb), "v"(acc));
        asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[0,1,0] op_sel_hi:[0,1,0]" : "=v"(r) : "v"(xh), "v"(qb), "v"(t));
        return r;
    }
#endif
    // On RDNA2 the DPP form of v_dot2c_f32_f16 does it in one instruction at full rate (row_share would halve it).
    // Elsewhere the x word is moved across the quad first (a GPU without the instruction, gfx900, takes fdot2's fused pair).
    template <int I>
    __device__ static float quad(uint32_t x, uint32_t w, float acc) {
#if TF_DOT2_F16 && !TF_DOT2_BF16
        asm("v_dot2c_f32_f16_dpp %0, %1, %2 quad_perm:[%3,%3,%3,%3] row_mask:0xf bank_mask:0xf"
            : "+v"(acc)
            : "v"(x), "v"(w), "n"(I));
        return acc;
#else
        return dot_quad_shared<DotF16, I>(x, w, acc);
#endif
    }
};

// BF16 v_dot2_f32_bf16 (gfx11 / gfx12).
struct DotBF16 {
    using elem = __bf16;
    using pair = bf16x2;
    static constexpr uint32_t ones = 0x3f803f80u;
    __device__ static elem zero() { return static_cast<__bf16>(0.f); }
    __device__ static pair two(elem a, elem b) { return pair{a, b}; }
    __device__ static float lo(pair p) { return static_cast<float>(p.x); }
    __device__ static float hi(pair p) { return static_cast<float>(p.y); }
    __device__ static float dot(pair x, pair q, float acc) {
#if TF_DEVICE_BF16_DOT2
        return __builtin_amdgcn_fdot2_f32_bf16(x, q, acc, false);
#else
        __builtin_trap();
        return acc;
#endif
    }
    // The BF16 dot2 has no DPP form: the x word is moved across the quad first.
    template <int I>
    __device__ static float quad(uint32_t x, uint32_t w, float acc) {
        return dot_quad_shared<DotBF16, I>(x, w, acc);
    }
};

}  // namespace rocm
}  // namespace tf
