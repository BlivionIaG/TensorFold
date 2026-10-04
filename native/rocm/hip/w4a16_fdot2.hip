// gfx1030 fp16 W4A16 product. 3inst states become fp16, then an explicit v_dot2_f32_f16.
// hipcc does not form that dot from the 3inst half2 chain, so the builtin is written out.
// Produce: hipcc --offload-arch=gfx1030 -O3 -cb 3inst -mno-wavefrontsize64 --genco, through gfx1030.hip.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cstdint>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1030__)
#error "TensorFold ROCm device code is gfx1030 only"
#endif

namespace tf {
namespace rocm {

__device__ __forceinline__ uint32_t mix3inst(uint32_t s) {
    uint32_t x = s * 89226354u + 64248484u;
    return (x & 0x8FFF8FFFu) ^ 0x3B603B60u;
}

__device__ __forceinline__ uint32_t packBytes(uint32_t b0, uint32_t b1, uint32_t b2, uint32_t b3) {
    return (b0 & 0xffu) | ((b1 & 0xffu) << 8) | ((b2 & 0xffu) << 16) | ((b3 & 0xffu) << 24);
}

// Two 3inst states as one half2: the low and high byte pairs, added.
__device__ __forceinline__ __half2 cb3inst(uint32_t s0, uint32_t s1) {
    const uint32_t x0 = mix3inst(s0);
    const uint32_t x1 = mix3inst(s1);
    const uint32_t lo = packBytes(x0, x0 >> 8, x1, x1 >> 8);
    const uint32_t hi = packBytes(x0 >> 16, x0 >> 24, x1 >> 16, x1 >> 24);
    const __half2 a = *reinterpret_cast<const __half2*>(&lo);
    const __half2 b = *reinterpret_cast<const __half2*>(&hi);
    return __hadd2(a, b);
}

__global__ void __launch_bounds__(32) w4a16_fdot2_kernel(const __half* __restrict__ act,
                                                        const uint16_t* __restrict__ states,
                                                        __half* __restrict__ out, int m, int n, int k) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (row >= m || col >= n) return;
    const __half* a = act + static_cast<long long>(row) * k;
    const uint16_t* w = states + static_cast<long long>(col) * k;
    float acc = 0.f;
    for (int t = 0; t < k; t += 2) {
        const __half2 x = __halves2half2(a[t], a[t + 1]);
        const __half2 q = cb3inst(w[t], w[t + 1]);
        acc = __builtin_amdgcn_fdot2(x, q, acc, false);
    }
    out[static_cast<long long>(row) * n + col] = __float2half_rn(acc);
}

}  // namespace rocm
}  // namespace tf

// act_kind 1 is fp16. Any other kind returns before a launch.
extern "C" hipError_t tf_gfx1030_w4a16_fdot2(const void* act, int act_kind, const uint16_t* states, void* out,
                                            int m, int n, int k, hipStream_t stream) {
    if (act_kind != 1 || act == nullptr || states == nullptr || out == nullptr || m < 1 || n < 1 || k < 2 || (k & 1))
        return hipErrorInvalidValue;
    const dim3 block(32);
    const dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>(m));
    tf::rocm::w4a16_fdot2_kernel<<<grid, block, 0, stream>>>(static_cast<const __half*>(act), states,
                                                            static_cast<__half*>(out), m, n, k);
    return hipGetLastError();
}
