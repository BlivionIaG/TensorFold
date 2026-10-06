// Decode-step kernels of a few rows (a round's windows), launched from Zig: the MoE router's logits a wave an expert,
// and the residual tails that merge several small launches. No torch bits: sums in the order that is fastest.
// kind: 1 fp16, 2 bf16 (act.hpp's numbering).

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

namespace {

__device__ __forceinline__ float bf16_up(uint32_t bits) { return __uint_as_float(bits << 16); }

// Four consecutive activations (8 bytes) widened.
template <int KIND>
__device__ __forceinline__ float4 widen4(uint2 v) {
    if constexpr (KIND == 1) {
        const __half2 a = __builtin_bit_cast(__half2, v.x);
        const __half2 b = __builtin_bit_cast(__half2, v.y);
        return make_float4(__low2float(a), __high2float(a), __low2float(b), __high2float(b));
    } else {
        return make_float4(bf16_up(v.x & 0xffffu), bf16_up(v.x >> 16), bf16_up(v.y & 0xffffu), bf16_up(v.y >> 16));
    }
}

constexpr int kRouterWaves = 4;
constexpr int kRouterSteps = 16;  // float4 loads a lane keeps in flight
constexpr int kRouterRows = 8;

// logits[r, e] = x[r] . rows[e] for up to 8 rows: a wave an expert, a lane four consecutive weights a step, every load
// of the expert's row in flight at once. d is a multiple of 4 and the weights' rows are 16-byte aligned.
template <int KIND>
__device__ void router_decode(const void* x, const float* rows, float* logits, int r, int d, int e) {
    const int lane = threadIdx.x & 31;
    const int expert = blockIdx.x * kRouterWaves + (threadIdx.x >> 5);
    if (expert >= e) return;
    const float* w = rows + static_cast<long long>(expert) * d;
    float acc[kRouterRows];
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) acc[k] = 0.f;
    for (int base = 0; base < d; base += kRouterSteps * 128) {
        float4 wv[kRouterSteps];
#pragma unroll
        for (int s = 0; s < kRouterSteps; ++s) {
            const int at = base + (lane + 32 * s) * 4;
            wv[s] = at < d ? *reinterpret_cast<const float4*>(w + at) : make_float4(0.f, 0.f, 0.f, 0.f);
        }
#pragma unroll
        for (int k = 0; k < kRouterRows; ++k) {
            if (k >= r) continue;
            const uint16_t* xr = static_cast<const uint16_t*>(x) + static_cast<long long>(k) * d;
#pragma unroll
            for (int s = 0; s < kRouterSteps; ++s) {
                const int at = base + (lane + 32 * s) * 4;
                if (at >= d) continue;
                const float4 xv = widen4<KIND>(*reinterpret_cast<const uint2*>(xr + at));
                acc[k] = fmaf(xv.x, wv[s].x, acc[k]);
                acc[k] = fmaf(xv.y, wv[s].y, acc[k]);
                acc[k] = fmaf(xv.z, wv[s].z, acc[k]);
                acc[k] = fmaf(xv.w, wv[s].w, acc[k]);
            }
        }
    }
#pragma unroll
    for (int k = 0; k < kRouterRows; ++k) {
        float v = acc[k];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) v += __shfl_xor(v, mask, 32);
        if (lane == 0 && k < r) logits[static_cast<long long>(k) * e + expert] = v;
    }
}


__device__ __forceinline__ float load_kind(const void* p, int kind, long long i) {
    return kind == 1 ? __half2float(static_cast<const __half*>(p)[i]) : bf16_up(static_cast<const uint16_t*>(p)[i]);
}

// A float rounded to the activation kind (fp16 or bf16, to nearest even) and widened again.
__device__ __forceinline__ float round_kind(float v, int kind) {
    if (kind == 1) return __half2float(__float2half_rn(v));
    if (v != v) return v;
    const uint32_t u = __float_as_uint(v);
    return __uint_as_float(((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16) << 16);
}

__device__ __forceinline__ void store_kind(void* p, int kind, long long i, float v) {
    if (kind == 1) {
        static_cast<__half*>(p)[i] = __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        static_cast<uint16_t*>(p)[i] = v != v ? 0x7fc0 : static_cast<uint16_t>((u + (((u >> 16) & 1u) + 0x7fffu)) >> 16);
    }
}

constexpr int kTailThreads = 512;
constexpr int kTailWidth = 8192;

// One row a block: x = round(x + t) where t is y (the activation kind) or, with `slots` > 0, the sum over slots of
// y[row * slots + s] * wts[row * slots + s] (fp32, in slot order) rounded to the kind; then normed = x * rsqrt(mean(x^2)
// + eps) * weight, rounded to the kind. What the add, the MoE combine and the next RMS norm each did, in one launch.
__device__ void tail_body(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind,
                          int slots, int width, float eps) {
    __shared__ float row_v[kTailWidth];
    __shared__ float part[kTailThreads / 32];
    const int row = blockIdx.x;
    const long long at = static_cast<long long>(row) * width;
    float sum = 0.f;
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float t;
        if (slots > 0) {
            float acc = 0.f;
            for (int s = 0; s < slots; ++s) {
                acc += static_cast<const float*>(y)[(static_cast<long long>(row) * slots + s) * width + i] *
                       wts[row * slots + s];
            }
            t = round_kind(acc, kind);
        } else {
            t = load_kind(y, kind, at + i);
        }
        const float v = round_kind(load_kind(x, kind, at + i) + t, kind);
        store_kind(x, kind, at + i, v);
        row_v[i] = v;
        sum += v * v;
    }
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < kTailThreads / 32; ++w) total += part[w];
    const float inv = rsqrtf(total / static_cast<float>(width) + eps);
    for (int i = threadIdx.x; i < width; i += kTailThreads) {
        float value = row_v[i] * inv;
        if (weight != nullptr) value *= weight[i];
        store_kind(normed, kind, at + i, value);
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_f16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<1>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(32 * kRouterWaves)
    tf_router_decode_bf16(const void* x, const float* rows, float* logits, int r, int d, int e) {
    router_decode<2>(x, rows, logits, r, d, e);
}

extern "C" __global__ void __launch_bounds__(kTailThreads)
    tf_tail(void* x, const void* y, const float* wts, const float* weight, void* normed, int kind, int slots, int width,
            float eps) {
    tail_body(x, y, wts, weight, normed, kind, slots, width, eps);
}
