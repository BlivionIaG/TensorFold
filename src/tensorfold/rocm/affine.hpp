#pragma once

// MLX affine words, little-endian, matching tensorfold/cuda/kernels/affine_kernels.py.
// y[m, n] = sum_groups (scale[n, g] * sum_k x[m, k] * bf16(code[n, k])
//                       + bias[n, g] * sum_k x[m, k])
// with k ranging over the group and code rounded through BF16 before the product.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace tf {
namespace rocm {

struct Affine {
    const void* x;           // (m, k) row major, bf16 or fp16
    const uint32_t* words;   // (n, k * bits / 32) row major
    const float* scale;      // (n, k / group) row major
    const float* bias;
    float* out;  // (m, n)
    int m, n, k, bits, group;
    int fp16;  // 0 is bf16, 1 is fp16
};

__host__ __device__ inline uint32_t affine_code(const uint32_t* row, int k, int bits, int words) {
    int bit = k * bits;
    int word = bit >> 5;
    int shift = bit & 31;
    uint32_t low = row[word];
    uint32_t high = (shift + bits > 32 && word + 1 < words) ? row[word + 1] : 0u;
    uint32_t value = (low >> shift);
    if (shift + bits > 32) value |= high << ((32 - shift) & 31);
    return value & ((1u << bits) - 1u);
}

// Integer code as the BF16 value the product sees. 129 is not exact in BF16.
__host__ __device__ inline float code_bf16(uint32_t code) {
    return static_cast<float>(hip_bfloat16(static_cast<float>(code)));
}

hipError_t launch_affine_gemv(const Affine& a, hipStream_t stream);
hipError_t launch_affine_wmma(const Affine& a, hipStream_t stream);

struct AffineSide {
    const uint32_t* words;
    const float* scale;
    const float* bias;
    float* out;
    int n;
};

// Two same-shape WMMA products that share x. Each side matches a solo launch.
hipError_t launch_affine_wmma_pair(const Affine& a, AffineSide first, AffineSide second, hipStream_t stream);
hipError_t launch_affine_dot2(const Affine& a, hipStream_t stream);
// schedule 1 is the GEMV. Anything else is the WMMA on a build that has it.
// fp16 is the RDNA2 v_dot2 schedule, and a WMMA build refuses it.
hipError_t launch_affine(const Affine& a, int schedule, hipStream_t stream);

}  // namespace rocm
}  // namespace tf
