// Wave32 GEMV. This is the BF16 schedule on RDNA2, which has no WMMA,
// and it is the explicit scalar formula on every part. FP16 activations on
// RDNA2 use the v_dot2 kernel instead.
// One thread owns one output and dequantizes the packed word in registers.
// The group dot is an ordered f32 fma of BF16 factors, so a row's bits do not
// depend on any other row. RDNA3, RDNA 3.5 and RDNA4 do not take this path for
// "auto": those builds always use the WMMA, including a single row padded with zeros.

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

// One warp streams one output column, 8-bit only. Up to 16 rows share that read. The warp
// sums the products in its own order, so a row's bits differ from the same row on WMMA.
__global__ void __launch_bounds__(32) affine_gemv_cols_kernel(Affine a) {
    const int col = blockIdx.x;
    const int lane = threadIdx.x;
    if (col >= a.n) return;
    const int words = a.k * a.bits / 32;
    const int groups = a.k / a.group;
    const int words_per_group = a.group / 4;
    const uint32_t* wrow = a.words + static_cast<long long>(col) * words;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.f;
    for (int group = 0; group < groups; ++group) {
        float dot[16], sum[16];
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            dot[r] = 0.f;
            sum[r] = 0.f;
        }
        const int word0 = group * words_per_group;
        for (int w = lane; w < words_per_group; w += 32) {
            const uint32_t packed = wrow[word0 + w];
            const int k0 = group * a.group + w * 4;
#pragma unroll
            for (int b = 0; b < 4; ++b) {
                const float q = code_bf16((packed >> (8 * b)) & 255u);
                const int k = k0 + b;
                for (int r = 0; r < a.m; ++r) {
                    const float xv = static_cast<float>(x[static_cast<long long>(r) * a.k + k]);
                    dot[r] = fmaf(xv, q, dot[r]);
                    sum[r] += xv;
                }
            }
        }
        for (int r = 0; r < a.m; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                dot[r] += __shfl_xor(dot[r], off);
                sum[r] += __shfl_xor(sum[r], off);
            }
        }
        if (lane == 0) {
            const long long g = static_cast<long long>(col) * groups + group;
            for (int r = 0; r < a.m; ++r) {
                acc[r] = fmaf(dot[r], a.scale[g], acc[r]);
                acc[r] = fmaf(sum[r], a.bias[g], acc[r]);
            }
        }
    }
    if (lane == 0) {
        for (int r = 0; r < a.m; ++r) a.out[static_cast<long long>(r) * a.n + col] = acc[r];
    }
}

hipError_t launch_affine_gemv_cols(const Affine& a, hipStream_t stream) {
    if (a.m < 1 || a.m > 16 || a.n < 1 || a.bits != 8 || a.group % 4 != 0 || a.k % a.group != 0) {
        return hipErrorInvalidValue;
    }
    affine_gemv_cols_kernel<<<a.n, 32, 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf
