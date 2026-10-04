#pragma once

// q, k: (batch, length, key_heads, dk). v, y: (batch, length, value_heads, dv).
// gate, beta: (batch, length, value_heads). state: (batch, value_heads, dv, dk), updated in place.
// dk is 16 or 128. One wave owns one value row and walks length in order.
void gated_delta_launch(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* state, float* y, int batch, int length, int key_heads, int value_heads, int dk,
                        int dv, void* stream);
