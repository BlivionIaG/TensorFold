#pragma once

// v_dot2_f32_f16: TF_FDOT2(x, q, acc) is x.x * q.x + x.y * q.y + acc in fp32, x and q a pair of fp16.
//
// gfx906 and every RDNA GPU have the instruction: there it is __builtin_amdgcn_fdot2 and nothing else (a macro, so that the RDNA
// code objects stay byte for byte what they were). gfx900 (Vega 10) does not: the two products (each exact in fp32, 11-bit
// significands) are added to the accumulator one after the other (v_mad_mix_f32, which equals a fused multiply-add on an exact product).

// Variant switches of gfx900 (-Dhip-cflags=-DTF_G9_MIX=0 ... builds the others, to compare on the card; RDNA ignores them):
//   TF_G9_MIX  1: the dot2 is two v_mad_mix_f32 (below)      0: converts and fuses
//   TF_G9_FOLD  1: the stream tile leaves the 1024 of its code pairs to the group's bias (not the same bits)    0: subtracts it per pair
//   TF_G9_ANDOR 1: the stream tile's code pairs by v_and_or_b32 (quant/mlx_decoder.hpp)     0: the compiler's and + or
#ifndef TF_G9_MIX
#define TF_G9_MIX 1
#endif
#ifndef TF_G9_FOLD
#define TF_G9_FOLD 0
#endif
#ifndef TF_G9_ANDOR
#define TF_G9_ANDOR 1
#endif

#if defined(__gfx900__)
#define TF_DEVICE_FDOT2 0
#else
#define TF_DEVICE_FDOT2 1
#endif

#if TF_DEVICE_FDOT2

#define TF_FDOT2(x, q, acc) __builtin_amdgcn_fdot2((x), (q), (acc), false)

#else

namespace tf {
namespace rocm {

// TF_G9_MIX=1 (default): two v_mad_mix_f32 (gfx9: the fp16 halves of the operands widened inside the instruction, denormal halves
// included). Each product is exact in fp32, so the mad rounds once like a fused multiply-add and the bits are those of the fma pair of
// TF_G9_MIX=0, which converts the four halves and fuses twice (six instructions where the mix form takes two).
template <class Pair>
__device__ __forceinline__ float fdot2_fused(Pair x, Pair q, float acc) {
#if TF_G9_MIX
    const uint32_t xb = __builtin_bit_cast(uint32_t, x);
    const uint32_t qb = __builtin_bit_cast(uint32_t, q);
    float r;
    float t;
    asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel_hi:[1,1,0]" : "=v"(t) : "v"(xb), "v"(qb), "v"(acc));
    asm("v_mad_mix_f32 %0, %1, %2, %3 op_sel:[1,1,0] op_sel_hi:[1,1,0]" : "=v"(r) : "v"(xb), "v"(qb), "v"(t));
    return r;
#else
    return fmaf(static_cast<float>(x.y), static_cast<float>(q.y), fmaf(static_cast<float>(x.x), static_cast<float>(q.x), acc));
#endif
}

}  // namespace rocm
}  // namespace tf

#define TF_FDOT2(x, q, acc) ::tf::rocm::fdot2_fused((x), (q), (acc))

#endif
