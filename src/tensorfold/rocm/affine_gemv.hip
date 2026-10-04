// Wave32 GEMV: the scalar formula, one thread an output, an ordered fp32 fma of BF16 factors (the reference).

#include "affine.hpp"
#include "arch.hpp"

namespace tf {
namespace rocm {

__global__ void __launch_bounds__(32) affine_gemv_kernel(Affine a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= a.m || col >= a.n) return;

    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const uint32_t* wrow = a.words + static_cast<long long>(col) * words;
    const hip_bfloat16* xrow = static_cast<const hip_bfloat16*>(a.x) + static_cast<long long>(row) * a.k;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        float dot = 0.f;
        float sum = 0.f;
        int base = group * a.group;
        for (int t = 0; t < a.group; ++t) {
            float x = static_cast<float>(xrow[base + t]);
            float q = code_bf16(affine_code(wrow, base + t, a.bits, words));
            dot = fmaf(x, q, dot);
            sum += x;
        }
        acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + group], acc);
    }
    a.out[static_cast<long long>(row) * a.n + col] = acc;
}

hipError_t launch_affine_gemv(const Affine& a, hipStream_t stream) {
    if (a.m < 1 || a.n < 1 || a.k < 1 || a.k % a.group != 0) return hipErrorInvalidValue;
    dim3 block(32);
    dim3 grid((a.n + 31) / 32, a.m);
    affine_gemv_kernel<<<grid, block, 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf
