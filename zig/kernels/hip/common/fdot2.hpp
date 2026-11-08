#pragma once

// v_dot2_f32_f16: TF_FDOT2(x, q, acc) is x.x * q.x + x.y * q.y + acc in fp32, x and q a pair of fp16.
//
// gfx906 and every RDNA GPU have the instruction: there it is __builtin_amdgcn_fdot2 and nothing else (a macro, so that the RDNA
// code objects stay byte for byte what they were). gfx900 (Vega 10) does not: the two products (each exact in fp32, 11-bit
// significands) are fused into the accumulator one after the other.

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

template <class Pair>
__device__ __forceinline__ float fdot2_fused(Pair x, Pair q, float acc) {
    return fmaf(static_cast<float>(x.y), static_cast<float>(q.y), fmaf(static_cast<float>(x.x), static_cast<float>(q.x), acc));
}

}  // namespace rocm
}  // namespace tf

#define TF_FDOT2(x, q, acc) ::tf::rocm::fdot2_fused((x), (q), (acc))

#endif
