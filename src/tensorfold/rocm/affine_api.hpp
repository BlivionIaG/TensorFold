#pragma once

// x (m, k) is bf16, or fp16 on RDNA2. words (n, k * bits / 32) int32.
// scale and bias (n, k / group), fp32, bf16 or fp16 (scale_kind 0, 1, 2, both the same type). out (m, n) fp32.
// schedule 0 picks WMMA on a build that has it and the GEMV otherwise. 1 is the ordered GEMV.
// 2 is WMMA. 3 is the short-batch column stream.
// fp16 selects v_dot2_f32_f16. A WMMA build refuses it.
void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half = 0);
// out_half: out is (m, n) fp16, written by the RDNA2 decode tile (fp16 x, m <= 8, schedule 0, one launch).
// 0 splits a short FP16 column grid on group boundaries. 1 is one launch. 2 forces the split.
int affine_dot2_splits(int m, int n, int k, int group, int mode);
void affine_pair_launch(const void* x, const void* words0, const void* scale0, const void* bias0, void* out0,
                        const void* words1, const void* scale1, const void* bias1, void* out1, int scale_kind, int m,
                        int n, int k, int bits, int group, void* stream);
// Up to four packed products that share x and K. Each column tile matches a solo WMMA launch.
void affine_group_launch(const void* x, const void* const* words, const void* const* scale, const void* const* bias,
                         int scale_kind, void* const* out, const int* ns, int nsides, int m, int k, int bits,
                         int group, void* stream);
// Every item of a routed plan in one FP16 launch: items (count, 3) int32 (expert, first, rows), members the pair
// ids sorted by expert, a pair's x row its id over x_div. words (E, n, k * bits / 32), scale and bias (E, n, k /
// group), out (pairs, n) fp32 by pair id. rows is the most rows an item holds.
void affine_routed_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind,
                          void* out, const int* items, int count, const int* members, int x_div, int rows, int n,
                          int k, int bits, int group, void* stream);
