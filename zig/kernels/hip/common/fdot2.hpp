#pragma once

// v_dot2_f32_f16: TF_FDOT2(x, q, acc) is x.x * q.x + x.y * q.y + acc in fp32, x and q a pair of fp16.
//
// gfx906 and every RDNA GPU have the instruction: there it is __builtin_amdgcn_fdot2 and nothing else (a macro, so that the RDNA
// code objects stay byte for byte what they were). gfx900 (Vega 10) does not: the two products (each exact in fp32, 11-bit
// significands) are added to the accumulator one after the other (two fused multiply-adds; the stream tile's v_mad_mix_f32 form is below).

// Variant switches of gfx900 (-Dhip-cflags=-DTF_G9_MIX=0 ... builds the others, to compare on the card; RDNA ignores them):
//   TF_G9_MIX  1: the stream tile's dot2 is two v_mad_mix_f32, the fused pair's bits (below)   2: the same on fp16 x as they are (a denormal
//              x flushed)   0: converts and fuses
//   TF_G9_FOLD  1: the stream tile leaves the 1024 of its code pairs to the group's bias (not the same bits)    0: subtracts it per pair
//   TF_G9_ANDOR 1: the stream tile's code pairs by v_and_or_b32 (quant/mlx_decoder.hpp)     0: the compiler's and + or
#ifndef TF_G9_MIX
#define TF_G9_MIX 2
#endif
#ifndef TF_G9_FOLD
#define TF_G9_FOLD 0
#endif
#ifndef TF_G9_ANDOR
#define TF_G9_ANDOR 1
#endif

// gfx900: v_mad_mix_f32 flushes a denormal fp16 operand whatever MODE.FP_DENORM says (measured on a Vega 56: 56 of 4096 random 96-term dots
// differ from the fused pair, by up to 6e-6 of the terms' magnitude); an fp32 operand it does not flush.
//   TF_G9_MIX=1: the stream tile mixes on x scaled by 2^10 (staged, normal fp16) or widened to fp32 (tiles/stream.hpp): the fused pair's bits.
//   TF_G9_MIX=2: the stream tile mixes on the fp16 x as they are: a denormal x (|x| < 6.1e-5) counts as zero.
#if defined(__gfx900__) && TF_G9_MIX == 1
#define TF_STREAM_SCALED_X 1
#else
#define TF_STREAM_SCALED_X 0
#endif
#if defined(__gfx900__) && (TF_G9_MIX == 1 || TF_G9_MIX == 2)
#define TF_STREAM_MIX_F16 1
#else
#define TF_STREAM_MIX_F16 0
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

// Two fused steps of converted halves (denormal halves kept; each product is exact in fp32, so each step rounds once).
template <class Pair>
__device__ __forceinline__ float fdot2_fused(Pair x, Pair q, float acc) {
    return fmaf(static_cast<float>(x.y), static_cast<float>(q.y), fmaf(static_cast<float>(x.x), static_cast<float>(q.x), acc));
}

}  // namespace rocm
}  // namespace tf

#define TF_FDOT2(x, q, acc) ::tf::rocm::fdot2_fused((x), (q), (acc))

#endif
