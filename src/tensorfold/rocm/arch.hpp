#pragma once

// Device-code architecture: gfx11 and gfx12 have v_dot2_f32_bf16 (BF16 tiles), gfx103x the FP16 one; RDNA1 refused.

#if defined(__gfx1010__) || defined(__gfx1011__) || defined(__gfx1012__)
static_assert(false, "RDNA1 (gfx101x) is not a TensorFold target");
#endif

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) \
    || defined(__gfx1150__) || defined(__gfx1151__) || defined(__gfx1152__) || defined(__gfx1153__) \
    || defined(__gfx11_generic__) || defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define TF_DEVICE_BF16_DOT2 1
#else
#define TF_DEVICE_BF16_DOT2 0
#endif
