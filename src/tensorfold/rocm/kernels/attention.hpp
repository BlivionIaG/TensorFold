#pragma once

// q, out fp32 (batch, heads, qlen, d); k, v the cache dtype (kind 0 fp16, 1 bf16, 2 fp32), d <= 256.
void causal_launch(const float* q, const void* k, const void* v, float* out, int batch, int qlen, int span,
                   int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                   long long k_ss, long long v_sb, long long v_sh, long long v_ss, int cache_kind, float* scores,
                   float* stats, float* partials, void* stream, const int* pos = nullptr);
// pos: a device position a query (partials hold (span + 127) / 128 tiles); null scratch runs the prefill tile.
