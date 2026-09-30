#pragma once

// x (m, k) is bf16, or fp16 on RDNA2. words (n, k * bits / 32) int32.
// scale and bias (n, k / group) fp32, out (m, n) fp32.
// schedule 0 picks WMMA on a build that has it and the GEMV otherwise. 1 is the GEMV. 2 is WMMA.
// fp16 selects v_dot2_f32_f16. A WMMA build refuses it.
void affine_launch(const void* x, const void* words, const void* scale, const void* bias, void* out, int m, int n,
                   int k, int bits, int group, int schedule, int fp16);
