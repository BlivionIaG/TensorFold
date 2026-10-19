// bitid.hip: the NaN fixes' old and new expressions on every finite input: their bits must be equal
#include "ops/common.hpp"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

__device__ __forceinline__ uint16_t bf16_bits(float v) {  // tiles/epilogue.hpp's narrow for bf16
    const uint32_t u = __float_as_uint(v);
    return v != v ? 0x7fc0u : (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
}

// tf_mamba_dt and the window's dt: every 16-bit pattern of the kind, each bias, each clamp
__global__ void dt_k(int kind, const float* bias, int nb, float lo, float hi, unsigned long long* bad, unsigned long long* done) {
    const unsigned x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= 65536) return;
    uint16_t bits = x;
    const float in = load(&bits, kind, 0);
    for (int b = 0; b < nb; ++b) {
        const float v = in + bias[b];
        const float sp = fmaxf(v, 0.0f) + log1pf(expf(-fabsf(v)));
        const float old = fminf(fmaxf(sp, lo), hi);
        const float now = sp != sp ? sp : fminf(fmaxf(sp, lo), hi);
        if (v != v) continue;  // only finite (and infinite) inputs must keep their bits
        atomicAdd(done, 1ull);
        if (__float_as_uint(old) != __float_as_uint(now)) atomicAdd(bad, 1ull);
    }
}

// tf_nemo_relu2 and the stream tile's relu^2 epilogue: every fp32 pattern
__global__ void relu_k(int kind, unsigned long long base, unsigned long long* bad, unsigned long long* done) {
    const unsigned long long x = base + blockIdx.x * static_cast<unsigned long long>(blockDim.x) + threadIdx.x;
    if (x > 0xffffffffull) return;
    const float up = __uint_as_float(static_cast<uint32_t>(x));
    if (up != up) return;
    // tf_nemo_relu2
    const float r = rounded(up, kind);
    const float u0 = fmaxf(r, 0.f);
    const float u1 = r != r ? r : fmaxf(r, 0.f);
    uint16_t o0, o1;
    store(&o0, kind, 0, u0 * u0);
    store(&o1, kind, 0, u1 * u1);
    // the epilogue: the sum narrowed, widened, relu, squared, narrowed
    float w = kind == 1 ? __half2float(__float2half_rn(up)) : __uint_as_float(static_cast<uint32_t>(bf16_bits(up)) << 16);
    float e0 = fmaxf(w, 0.f), e1 = w != w ? w : fmaxf(w, 0.f);
    float s0 = e0 * e0, s1 = e1 * e1;
    asm("" : "+v"(s0));
    asm("" : "+v"(s1));
    const uint16_t p0 = kind == 1 ? __half_as_ushort(__float2half_rn(s0)) : bf16_bits(s0);
    const uint16_t p1 = kind == 1 ? __half_as_ushort(__float2half_rn(s1)) : bf16_bits(s1);
    atomicAdd(done, 1ull);
    if (o0 != o1 || p0 != p1) atomicAdd(bad, 1ull);
}

int main() {
    unsigned long long *bad, *done;
    hipMalloc(&bad, 8); hipMalloc(&done, 8);
    const float hb[] = {0.f, -0.f, 0.5f, -0.5f, 1e-3f, -3.f, 7.25f, -88.f, 88.f, 1e30f, -1e30f, 1e-40f};
    float* bias; hipMalloc(&bias, sizeof hb); hipMemcpy(bias, hb, sizeof hb, hipMemcpyHostToDevice);
    int fails = 0;
    for (int kind = 1; kind <= 2; ++kind) {
        for (int c = 0; c < 2; ++c) {
            const float lo = c ? 0.001f : 0.f, hi = c ? 0.1f : INFINITY;
            hipMemset(bad, 0, 8); hipMemset(done, 0, 8);
            dt_k<<<256, 256>>>(kind, bias, 12, lo, hi, bad, done);
            unsigned long long b, d; hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost); hipMemcpy(&d, done, 8, hipMemcpyDeviceToHost);
            printf("dt kind %d clamp [%g, %g]: %llu inputs, %llu differ\n", kind, lo, hi, d, b);
            fails += b != 0 || d == 0;
        }
        hipMemset(bad, 0, 8); hipMemset(done, 0, 8);
        for (unsigned long long base = 0; base <= 0xffffffffull; base += 1ull << 28) relu_k<<<(1u << 28) / 256, 256>>>(kind, base, bad, done);
        unsigned long long b, d; hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost); hipMemcpy(&d, done, 8, hipMemcpyDeviceToHost);
        printf("relu2 and its epilogue kind %d: %llu non-NaN fp32 inputs, %llu differ\n", kind, d, b);
        fails += b != 0 || d == 0;
    }
    printf(fails ? "FAIL\n" : "PASS every finite input keeps its bits\n");
    return fails;
}
