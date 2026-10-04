#pragma once

// x (m, k) bf16, or fp16 on RDNA2; words (n, k * bits / 32); scale and bias (n, k / group) of one type; out (m, n).
void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half = 0);
// out_half: out is fp16, rounded by the RDNA2 decode tile (fp16 x, m <= 8, schedule 0, one launch).
int affine_dot2_splits(int m, int n, int k, int group, int mode);
// Every item of a routed plan in one launch over stacked (E, n, ...) weights; out (pairs, n) fp32 by pair id.
void affine_routed_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind,
                          void* out, const int* items, int count, const int* members, int x_div, int rows, int n,
                          int k, int bits, int group, int fp16, void* stream);
