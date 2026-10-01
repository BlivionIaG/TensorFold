#pragma once

#include <hip/hip_runtime.h>

// One row of RMSNorm, a length-1 causal conv, and a length-1 RoPE. Wider prefill stays on the host.

void rms_launch(const float* x, const float* weight, float* y, int rows, int width, float eps, hipStream_t stream);
void conv_decode_launch(const float* x, const float* weight, float* state, float* y, int batch, int channels,
                        int kernel, hipStream_t stream);
void rope_decode_launch(const float* x, float* y, int rows, int width, int rotary, int pos, float theta,
                        hipStream_t stream);
