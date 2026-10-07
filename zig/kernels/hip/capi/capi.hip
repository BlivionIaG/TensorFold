// The ROCm launchers behind a C ABI: raw device pointers and the caller's stream; a call returns 0, or 1 with its message kept.

#include <hip/hip_runtime.h>

#include <cstring>
#include <exception>

#include "ops/act.hpp"
#include "rocm/affine_api.hpp"
#include "attention/attention.hpp"
#include "recurrence/gated_delta.hpp"

namespace {

thread_local char last_error[512] = "";

template <typename F>
int guarded(F&& f) {
    try {
        f();
        hipError_t err = hipGetLastError();
        if (err != hipSuccess) {
            std::strncpy(last_error, hipGetErrorString(err), sizeof(last_error) - 1);
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::strncpy(last_error, e.what(), sizeof(last_error) - 1);
        return 1;
    }
}

}  // namespace

extern "C" {

const char* tf_last_error() { return last_error; }

// 0 rdna2 (dot2, fp16 activations), 1 a WMMA build (gfx11 / gfx12, bf16 activations): host dispatch differs.
int tf_wmma_build() { return TENSORFOLD_RDNA_WMMA; }

int tf_rms(const void* x, const float* weight, void* y, int kind, int rows, int width, float eps, hipStream_t s) {
    return guarded([&] { rms_launch(x, weight, y, kind, rows, width, eps, s); });
}

int tf_conv_decode(const float* x, const float* weight, float* state, float* y, int batch, int channels, int kernel,
                   hipStream_t s) {
    return guarded([&] { conv_decode_launch(x, weight, state, y, batch, channels, kernel, s); });
}

int tf_conv_rows(const float* x, const float* weight, float* state, float* y, float* states, int rows, int channels,
                 int kernel, hipStream_t s) {
    return guarded([&] { conv_rows_launch(x, weight, state, y, states, rows, channels, kernel, s); });
}

int tf_rope_decode(const float* x, float* y, int rows, int width, int rotary, int pos, float theta, hipStream_t s,
                   const int* pos_dev, int per) {
    return guarded([&] { rope_decode_launch(x, y, rows, width, rotary, pos, theta, s, pos_dev, per); });
}

int tf_moe_router(const void* x, int kind, const float* rows, float* logits, int r, int d, int e, hipStream_t s) {
    return guarded([&] { moe_router_launch(x, kind, rows, logits, r, d, e, s); });
}

int tf_moe_select(const float* logits, int* pick, float* wts, int* items, int* members, int capacity, int r,
                  int experts, int top_k, hipStream_t s) {
    return guarded([&] { moe_select_launch(logits, pick, wts, items, members, capacity, r, experts, top_k, s); });
}

int tf_moe_act(const float* both, void* out, int kind, int pairs, int width, float limit, hipStream_t s) {
    return guarded([&] { moe_act_launch(both, out, kind, pairs, width, limit, s); });
}

int tf_moe_combine(const float* y, const float* wts, void* out, int kind, int r, int slots, int d, hipStream_t s) {
    return guarded([&] { moe_combine_launch(y, wts, out, kind, r, slots, d, s); });
}

int tf_gdn_gate(const void* a, const void* b, int kind, const float* a_log, const float* dt_bias, float* gate,
                float* beta, int count, int heads, hipStream_t s) {
    return guarded([&] { gdn_gate_launch(a, b, kind, a_log, dt_bias, gate, beta, count, heads, s); });
}

int tf_causal(const float* q, const void* k, const void* v, float* out, int batch, int qlen, int span, int heads,
              int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
              long long v_sb, long long v_sh, long long v_ss, int cache_kind, float* scores, float* stats,
              float* partials, hipStream_t s, const int* pos) {
    return guarded([&] {
        causal_launch(q, k, v, out, batch, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb,
                      v_sh, v_ss, cache_kind, scores, stats, partials, s, pos);
    });
}

int tf_gated_delta(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* state,
                   float* y, int batch, int length, int key_heads, int value_heads, int dk, int dv, hipStream_t s,
                   float* states) {
    return guarded([&] {
        gated_delta_launch(q, k, v, gate, beta, state, y, batch, length, key_heads, value_heads, dk, dv, s, states);
    });
}

int tf_affine_dot2_splits(int m, int n, int k, int group, int mode) {
    return affine_dot2_splits(m, n, k, group, mode);
}

int tf_affine(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out, int m,
              int n, int k, int bits, int group, int schedule, int fp16, hipStream_t s, float* partial, int splits,
              int out_half) {
    return guarded([&] {
        affine_launch(x, words, scale, bias, scale_kind, out, m, n, k, bits, group, schedule, fp16, s, partial, splits,
                      out_half);
    });
}

int tf_affine_routed(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                     const int* items, int count, const int* members, int x_div, int rows, int n, int k, int bits,
                     int group, int fp16, hipStream_t s) {
    return guarded([&] {
        affine_routed_launch(x, words, scale, bias, scale_kind, out, items, count, members, x_div, rows, n, k, bits,
                             group, fp16, s);
    });
}

}  // extern "C"
