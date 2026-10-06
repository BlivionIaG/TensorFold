// The tensor-parallel forward's own ops: a residual add of an fp32 sum (rounded once to the activation dtype, as torch's
// widened add into the residual) and the expert share's id remapping and masks. kind: 0 fp32, 1 fp16, 2 bf16.

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstring>

namespace {

__device__ __forceinline__ float bf16_to_float(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

// c10::BFloat16's round to nearest even, NaN to 0x7fc0.
__device__ __forceinline__ uint16_t float_to_bf16(float f) {
    if (f != f) return 0x7fc0;
    uint32_t u = __float_as_uint(f);
    uint32_t bias = ((u >> 16) & 1u) + 0x7fffu;
    return static_cast<uint16_t>((u + bias) >> 16);
}

__device__ __forceinline__ long long gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

// out = round(float(x) + y): the residual of a rank-summed fp32 share.
__global__ void add_wide_kernel(const void* x, const float* y, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    if (kind == 1) {
        float v = __half2float(static_cast<const __half*>(x)[i]) + y[i];
        static_cast<__half*>(out)[i] = __float2half_rn(v);
    } else {
        float v = bf16_to_float(static_cast<const uint16_t*>(x)[i]) + y[i];
        static_cast<uint16_t*>(out)[i] = float_to_bf16(v);
    }
}

// out[i] = the rank's local id of expert pick[i], or `skip` (an id past the rank's experts) when another rank holds it.
__global__ void localize_kernel(const int* pick, const int* remap, int* out, int n, int skip) {
    int i = static_cast<int>(gid());
    if (i >= n) return;
    int local = remap[pick[i]];
    out[i] = local >= 0 ? local : skip;
}

// An item of the skip expert does no work.
__global__ void foreign_items_kernel(int* items, int count, int skip) {
    int i = static_cast<int>(gid());
    if (i < count && items[i * 3] == skip) items[i * 3 + 2] = 0;
}

// A pair another rank's expert answers has zero rows (selected, not multiplied: the buffer holds anything).
__global__ void zero_foreign_kernel(float* y, const int* picks, int skip, long long pairs, int d) {
    long long i = gid();
    if (i >= pairs * d) return;
    if (picks[i / d] == skip) y[i] = 0.f;
}

thread_local char tp_error[256] = "";

int finish() {
    hipError_t err = hipGetLastError();
    if (err == hipSuccess) return 0;
    std::strncpy(tp_error, hipGetErrorString(err), sizeof(tp_error) - 1);
    return 1;
}

unsigned blocks(long long n, int threads) { return static_cast<unsigned>((n + threads - 1) / threads); }

}  // namespace

extern "C" {

const char* tf_tp_error() { return tp_error; }

int tf_add_wide(const void* x, const float* y, void* out, int kind, long long n, hipStream_t s) {
    if (n == 0) return 0;
    add_wide_kernel<<<blocks(n, 256), 256, 0, s>>>(x, y, out, kind, n);
    return finish();
}

int tf_moe_localize(const int* pick, const int* remap, int* out, int n, int skip, hipStream_t s) {
    if (n == 0) return 0;
    localize_kernel<<<blocks(n, 256), 256, 0, s>>>(pick, remap, out, n, skip);
    return finish();
}

int tf_moe_foreign_items(int* items, int count, int skip, hipStream_t s) {
    if (count == 0) return 0;
    foreign_items_kernel<<<blocks(count, 256), 256, 0, s>>>(items, count, skip);
    return finish();
}

int tf_moe_zero_foreign(float* y, const int* picks, int skip, long long pairs, int d, hipStream_t s) {
    if (pairs == 0) return 0;
    zero_foreign_kernel<<<blocks(pairs * d, 256), 256, 0, s>>>(y, picks, skip, pairs, d);
    return finish();
}

}  // extern "C"
