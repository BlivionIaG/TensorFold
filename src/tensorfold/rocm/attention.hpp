#pragma once

// q and out are fp32 (batch, heads, qlen, d). k and v are the cache dtype, with the
// given element strides, and only the first ``span`` keys are valid. d is at most 256.
// cache_kind: 0 fp16, 1 bf16, 2 fp32.
// scores (batch, heads, span), stats (batch, heads, 2), partials (batch, heads, tiles, d). Null unless qlen is 1.
void causal_launch(const float* q, const void* k, const void* v, float* out, int batch, int qlen, int span,
                   int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                   long long k_ss, long long v_sb, long long v_sh, long long v_ss, int cache_kind, float* scores,
                   float* stats, float* partials, void* stream);
