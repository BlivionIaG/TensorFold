// gfx1030 packed-int product. Four signed bytes per word, v_dot4_i32_i8 via sdot4.
// Produce: hipcc --offload-arch=gfx1030 -O3 -mno-wavefrontsize64 --genco, through gfx1030.hip.
#include <hip/hip_runtime.h>

#if defined(__HIP_DEVICE_COMPILE__) && !defined(__gfx1030__)
#error "TensorFold ROCm device code is gfx1030 only"
#endif

__global__ void __launch_bounds__(32) sdot4_kernel(const int* __restrict__ a, const int* __restrict__ b,
                                                   int* __restrict__ out, int m, int n, int k_packs) {
    const int col = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    if (row >= m || col >= n) return;
    const int* ar = a + static_cast<long long>(row) * k_packs;
    const int* br = b + static_cast<long long>(col) * k_packs;
    int acc = 0;
    for (int p = 0; p < k_packs; ++p) acc = __builtin_amdgcn_sdot4(ar[p], br[p], acc, false);
    out[static_cast<long long>(row) * n + col] = acc;
}

extern "C" hipError_t tf_gfx1030_sdot4(const int* a, const int* b, int* out, int m, int n, int k_packs,
                                      hipStream_t stream) {
    if (a == nullptr || b == nullptr || out == nullptr || m < 1 || n < 1 || k_packs < 1) return hipErrorInvalidValue;
    const dim3 block(32);
    const dim3 grid(static_cast<unsigned>((n + 31) / 32), static_cast<unsigned>(m));
    sdot4_kernel<<<grid, block, 0, stream>>>(a, b, out, m, n, k_packs);
    return hipGetLastError();
}
