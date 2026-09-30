#pragma once

// One affine formula, two schedules.
//
// RDNA2 (gfx103x) has no WMMA. BF16 x uses the wave32 GEMV. FP16 x uses
// v_dot2_f32_f16, one thread per output. RDNA1 (gfx101x) is not a target:
// gfx1010 has no v_dot2, and gfx1011/gfx1012 only have v_dot2_f32_f16.
// RDNA3 (gfx1100–1103) and RDNA 3.5 (gfx1150–1153) use the gfx11 WMMA 16x16x16 BF16.
// RDNA4 (gfx1200/gfx1201) uses the gfx12 builtin. Every one of those builds takes WMMA
// for every row count, including one row padded with zeros. A GEMV beside it would
// change a row's bits when a window grows. gfx11 and gfx12 do not share a register
// layout (bhalf16 versus bhalf8); rocWMMA is the only place that lives.

namespace tf {
namespace rocm {

enum class Gen { rdna2, rdna3, rdna4 };

// Device code only. A host translation unit does not see the offload-arch
// macro, so host dispatch uses TENSORFOLD_RDNA_WMMA from the build.
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
inline constexpr Gen kGen = Gen::rdna4;
inline constexpr bool kWmma = true;
#elif defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx11_generic__)
inline constexpr Gen kGen = Gen::rdna3;
inline constexpr bool kWmma = true;
#elif defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) \
    || defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
inline constexpr Gen kGen = Gen::rdna2;
inline constexpr bool kWmma = false;
#elif defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
static_assert(false, "RDNA1 (gfx101x) is not a TensorFold target");
#else
inline constexpr Gen kGen = Gen::rdna2;
inline constexpr bool kWmma = false;
#endif

}  // namespace rocm
}  // namespace tf
