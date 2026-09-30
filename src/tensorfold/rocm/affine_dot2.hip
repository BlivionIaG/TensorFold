// RDNA2 FP16 activations. One thread owns one output and walks the whole K in pairs,
// so a row's bits do not depend on any other row. Codes are rounded through BF16
// and widened to FP16. The pair product is v_dot2_f32_f16; hipcc does not lower an
// hfma2 chain to that instruction on gfx1030. Scale and bias stay fp32, outside
// the group sum. The packed words stay the only copy of the weight.

#include "affine.hpp"
#include "arch.hpp"

#include <hip/hip_fp16.h>

namespace tf {
namespace rocm {

__device__ __half code_f16(uint32_t code) {
    return __float2half_rn(code_bf16(code));
}

__global__ void __launch_bounds__(32) affine_dot2_kernel(Affine a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= a.m || col >= a.n) return;

    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const uint32_t* wrow = a.words + static_cast<long long>(col) * words;
    const __half* xrow = static_cast<const __half*>(a.x) + static_cast<long long>(row) * a.k;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        float dot = 0.f;
        float sum = 0.f;
        int base = group * a.group;
        for (int t = 0; t < a.group; t += 2) {
            __half x0 = xrow[base + t];
            __half x1 = xrow[base + t + 1];
            __half q0 = code_f16(affine_code(wrow, base + t, a.bits, words));
            __half q1 = code_f16(affine_code(wrow, base + t + 1, a.bits, words));
            dot = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), __halves2half2(q0, q1), dot, false);
            sum += static_cast<float>(x0);
            sum += static_cast<float>(x1);
        }
        acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + group], acc);
    }
    a.out[static_cast<long long>(row) * a.n + col] = acc;
}

hipError_t launch_affine_dot2(const Affine& a) {
    if (a.m < 1 || a.n < 1 || a.k < 1 || a.group % 2 != 0 || a.k % a.group != 0) return hipErrorInvalidValue;
    dim3 block(32);
    dim3 grid((a.n + 31) / 32, a.m);
    affine_dot2_kernel<<<grid, block>>>(a);
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf
