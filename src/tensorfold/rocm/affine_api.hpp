#pragma once

// x (m, k) is bf16, or fp16 on RDNA2. words (n, k * bits / 32) int32.
// scale and bias (n, k / group) fp32, out (m, n) fp32.
// schedule 0 picks WMMA on a build that has it and the GEMV otherwise. 1 is the GEMV. 2 is WMMA.
// fp16 selects v_dot2_f32_f16. A WMMA build refuses it.
void affine_launch(const void* x, const void* words, const void* scale, const void* bias, void* out, int m, int n,
                   int k, int bits, int group, int schedule, int fp16, void* stream, float* partial, int splits);
// 0 splits a short FP16 column grid on group boundaries. 1 is one launch. 2 forces the split.
int affine_dot2_splits(int m, int n, int k, int group, int mode);
void affine_pair_launch(const void* x, const void* words0, const void* scale0, const void* bias0, void* out0,
                        const void* words1, const void* scale1, const void* bias1, void* out1, int m, int n, int k,
                        int bits, int group, void* stream);
// Up to four packed products that share x and K. Each column tile matches a solo WMMA launch.
void affine_group_launch(const void* x, const void* const* words, const void* const* scale, const void* const* bias,
                         void* const* out, const int* ns, int nsides, int m, int k, int bits, int group,
                         void* stream);
