#pragma once

#include <hip/hip_runtime.h>

// One row of RMSNorm, a length-1 causal conv, and a length-1 RoPE. Wider prefill stays on the host.

// kind: 0 fp32, 1 fp16, 2 bf16 for x and y. weight is fp32. The arithmetic is fp32.
void rms_launch(const void* x, const float* weight, void* y, int kind, int rows, int width, float eps,
                hipStream_t stream);
void conv_decode_launch(const float* x, const float* weight, float* state, float* y, int batch, int channels,
                        int kernel, hipStream_t stream);
// pos_dev, when set, is the position on the device (read when the kernel runs) and pos is ignored.
void rope_decode_launch(const float* x, float* y, int rows, int width, int rotary, int pos, float theta,
                        hipStream_t stream, const int* pos_dev = nullptr);

// MoE glue, one launch each. x kind: 1 fp16, 2 bf16. router: logits[r, e] = sum_d x[r, d] * rows[e, d] in fp32,
// lane l of a wave adding d = l, l + 32, ... in order then the warp tree, so a row's bits do not depend on R.
void moe_router_launch(const void* x, int kind, const float* rows, float* logits, int r, int d, int e,
                       hipStream_t stream);
// Each row's top_k of the first `experts` logits (largest first, the lower id among equal ones), weights
// exp(l_k - l_0) summed in pick order and rounded through bf16, then slot top_k: id `experts`, weight
// bf16(sigmoid(bf16(logit[experts]))). With items set (one row), item k is (pick_k, k, 1) for k < slots,
// members[k] = k, and items slots .. capacity get count 0.
void moe_select_launch(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int r,
                       int experts, int top_k, hipStream_t stream);
// out[p, i] = silu(min(g, limit)) * clamp(u, -limit, limit), g = both[p, i], u = both[p, width + i]; limit 0 is
// none. out kind 1 fp16, 2 bf16.
void moe_act_launch(const float* both, void* out, int kind, int pairs, int width, float limit, hipStream_t stream);
// out[r, d] = sum over slots s in order of y[r, s, d] * wts[r, s], fp32, rounded once to out kind (0 fp32,
// 1 fp16, 2 bf16).
void moe_combine_launch(const float* y, const float* wts, void* out, int kind, int r, int slots, int d,
                        hipStream_t stream);
// Gated DeltaNet gate and beta for `count` (row, head) values, heads the period of a_log / dt_bias:
// beta = 1 / (1 + exp(-b)), gate = exp(-exp(a_log) * softplus(a + dt_bias)), softplus linear past 20. fp32 out.
void gdn_gate_launch(const void* a, const void* b, int kind, const float* a_log, const float* dt_bias, float* gate,
                     float* beta, int count, int heads, hipStream_t stream);
