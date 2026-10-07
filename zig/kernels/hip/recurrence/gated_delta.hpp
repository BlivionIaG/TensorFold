#pragma once

// q, k (B, L, Hk, dk); v, y (B, L, Hv, dv); gate, beta (B, L, Hv); state (B, Hv, dv, dk) in place; optional states keep every step's.
void gated_delta_launch(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* state, float* y, int batch, int length, int key_heads, int value_heads, int dk,
                        int dv, void* stream, float* states = nullptr);
