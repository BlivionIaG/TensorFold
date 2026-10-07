// MLX affine 4-bit product, bf16 x, scales and biases: y[m, n] = sum_g scale * (x . codes) + bias * sum(x), fp32 sums.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

// One thread an output: per group an ordered fma of x and code and a plain sum of x, then two fmas into the row.
extern "C" __global__ void __launch_bounds__(64)
    tf_affine4_bf16(const hip_bfloat16* x, const uint32_t* words, const hip_bfloat16* scale, const hip_bfloat16* bias,
                    float* out, uint32_t m, uint32_t n, uint32_t k, uint32_t group) {
    const uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = blockIdx.y;
    if (row >= m || col >= n) return;
    const uint32_t groups = k / group;
    const uint32_t* w = words + static_cast<size_t>(col) * (k / 8);
    const hip_bfloat16* xr = x + static_cast<size_t>(row) * k;
    float acc = 0.f;
    for (uint32_t g = 0; g < groups; ++g) {
        float dot = 0.f;
        float sum = 0.f;
        for (uint32_t t = g * group; t < (g + 1) * group; ++t) {
            const float xv = static_cast<float>(xr[t]);
            const float q = static_cast<float>((w[t >> 3] >> (4 * (t & 7))) & 15u);
            dot = fmaf(xv, q, dot);
            sum += xv;
        }
        const size_t at = static_cast<size_t>(col) * groups + g;
        acc = fmaf(dot, static_cast<float>(scale[at]), acc);
        acc = fmaf(sum, static_cast<float>(bias[at]), acc);
    }
    out[static_cast<size_t>(row) * n + col] = acc;
}
