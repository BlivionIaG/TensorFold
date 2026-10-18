// Gated DeltaNet kernels for tensor-parallel ranks (tp=2 on Qwen3.8-Flash-Next EXL3). Same arithmetic, same order of operations
// as exl3_gdn.cu, with the head counts as arguments so a rank can own a contiguous range of key heads (nkl) and value heads (nvl).
// A rank's conv channels are laid out q (nkl*128) | k (nkl*128) | v (nvl*128); its A_log / dt_bias / in_proj_a / in_proj_b pointers are
// already offset to its first value head. With nkl = 16 and nvl = 48 these are exl3_gdn.cu's kernels bit for bit.
// Compiled with --fmad=false, IEEE division and square root (the glue flags).

#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int DK = 128;
constexpr int DV = 128;
constexpr int TAPS = 4;

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

// out[t][0..nvl) = in_proj_a x, out[t][nvl..2 nvl) = in_proj_b x. wa / wb are fp16 [nvl][K] (this rank's rows).
// Grid (ceil(2 nvl / 8), tokens), block 256: a warp is one output row.
extern "C" __global__ void __launch_bounds__(256) exl3_gdn_tp_ab_f16(const __half* __restrict__ x, const __half* __restrict__ wa,
                                                                     const __half* __restrict__ wb, float* __restrict__ out,
                                                                     int tokens, int K, int nvl) {
    const int t = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * 8 + warp;
    if (t >= tokens || row >= 2 * nvl) return;
    const __half* w = (row < nvl ? wa + static_cast<size_t>(row) * K : wb + static_cast<size_t>(row - nvl) * K);
    const __half* xr = x + static_cast<size_t>(t) * K;
    float acc = 0.0f;
    for (int i = lane; i < K; i += 32) acc += __half2float(xr[i]) * __half2float(w[i]);
    acc = warp_sum(acc);
    if (lane == 0) out[static_cast<size_t>(t) * (2 * nvl) + row] = acc;
}

// Causal depthwise conv (taps 4, no bias) + SiLU, C channels (this rank's), one thread a channel. cw is bf16 [C][4], gathered.
extern "C" __global__ void __launch_bounds__(128) exl3_gdn_tp_conv_f16(const __half* __restrict__ qkv, const uint16_t* __restrict__ cw,
                                                                       __half* __restrict__ state, float* __restrict__ out, int tokens,
                                                                       int C) {
    const int ch = static_cast<int>(blockIdx.x) * 128 + static_cast<int>(threadIdx.x);
    if (ch >= C) return;
    const float w0 = bf16_to_f32(cw[ch * TAPS + 0]);
    const float w1 = bf16_to_f32(cw[ch * TAPS + 1]);
    const float w2 = bf16_to_f32(cw[ch * TAPS + 2]);
    const float w3 = bf16_to_f32(cw[ch * TAPS + 3]);
    float s0 = __half2float(state[ch]);
    float s1 = __half2float(state[C + ch]);
    float s2 = __half2float(state[2 * C + ch]);
    for (int t = 0; t < tokens; ++t) {
        const float x = __half2float(qkv[static_cast<size_t>(t) * C + ch]);
        float acc = w0 * s0;
        acc = acc + w1 * s1;
        acc = acc + w2 * s2;
        acc = acc + w3 * x;
        out[static_cast<size_t>(t) * C + ch] = acc / (1.0f + expf(-acc));
        s0 = s1;
        s1 = s2;
        s2 = x;
    }
    state[ch] = __float2half(s0);
    state[C + ch] = __float2half(s1);
    state[2 * C + ch] = __float2half(s2);
}

// The delta rule for `tokens` tokens in order over this rank's nvl value heads (nkl key heads, nvl / nkl value heads a key head).
extern "C" __global__ void __launch_bounds__(128) exl3_gdn_tp_delta_f32(const float* __restrict__ act, const float* __restrict__ ab,
                                                                        const uint16_t* __restrict__ a_log, const uint16_t* __restrict__ dt_bias,
                                                                        float* __restrict__ state, float* __restrict__ y, int tokens,
                                                                        int nkl, int nvl) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * 4 + warp;  // hv * 128 + v (rank-local hv)
    if (row >= nvl * DV) return;
    const int C = 2 * nkl * DK + nvl * DV;
    const int ZW = nvl * DV;
    const int AB = 2 * nvl;
    const int hv = row / DV;
    const int hk = hv / (nvl / nkl);
    const float alog = bf16_to_f32(a_log[hv]);
    const float dtb = bf16_to_f32(dt_bias[hv]);
    float4* srow = reinterpret_cast<float4*>(state + static_cast<size_t>(row) * DK);
    float4 s = srow[lane];
    const float inv_sqrt_dk = 1.0f / sqrtf(static_cast<float>(DK));
    for (int t = 0; t < tokens; ++t) {
        const float* at = act + static_cast<size_t>(t) * C;
        const float4 q = *reinterpret_cast<const float4*>(at + hk * DK + lane * 4);
        const float4 k = *reinterpret_cast<const float4*>(at + nkl * DK + hk * DK + lane * 4);
        const float v = at[2 * nkl * DK + row];
        float qq = q.x * q.x;
        qq = qq + q.y * q.y;
        qq = qq + q.z * q.z;
        qq = qq + q.w * q.w;
        float kk = k.x * k.x;
        kk = kk + k.y * k.y;
        kk = kk + k.z * k.z;
        kk = kk + k.w * k.w;
        qq = warp_sum(qq);
        kk = warp_sum(kk);
        const float iq = 1.0f / sqrtf(qq + 1e-6f) * inv_sqrt_dk;
        const float ik = 1.0f / sqrtf(kk + 1e-6f);
        const float a = ab[static_cast<size_t>(t) * AB + hv];
        const float b = ab[static_cast<size_t>(t) * AB + nvl + hv];
        const float xa = a + dtb;
        const float sp = xa > 20.0f ? xa : log1pf(expf(xa));
        const float decay = expf(-expf(alog) * sp);
        const float beta = sigmoid_f(b);
        const float4 kn = make_float4(k.x * ik, k.y * ik, k.z * ik, k.w * ik);
        const float4 qn = make_float4(q.x * iq, q.y * iq, q.z * iq, q.w * iq);
        s.x *= decay;
        s.y *= decay;
        s.z *= decay;
        s.w *= decay;
        float kv = s.x * kn.x;
        kv = kv + s.y * kn.y;
        kv = kv + s.z * kn.z;
        kv = kv + s.w * kn.w;
        kv = warp_sum(kv);
        const float delta = (v - kv) * beta;
        s.x += kn.x * delta;
        s.y += kn.y * delta;
        s.z += kn.z * delta;
        s.w += kn.w * delta;
        float o = s.x * qn.x;
        o = o + s.y * qn.y;
        o = o + s.z * qn.z;
        o = o + s.w * qn.w;
        o = warp_sum(o);
        if (lane == 0) y[static_cast<size_t>(t) * ZW + row] = o;
    }
    srow[lane] = s;
}

// Gated RMS norm per (token, rank-local value head). Grid (nvl, tokens), block 32.
extern "C" __global__ void __launch_bounds__(32) exl3_gdn_tp_norm_f16(const float* __restrict__ y, const __half* __restrict__ z,
                                                                      const uint16_t* __restrict__ w, __half* __restrict__ out,
                                                                      int tokens, float eps, int nvl) {
    const int hv = static_cast<int>(blockIdx.x);
    const int t = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (t >= tokens) return;
    const size_t base = static_cast<size_t>(t) * (nvl * DV) + static_cast<size_t>(hv) * DV;
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[i] = y[base + lane * 4 + i];
        ss += v[i] * v[i];
    }
    ss = warp_sum(ss);
    const float inv = 1.0f / sqrtf(ss / static_cast<float>(DV) + eps);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int d = lane * 4 + i;
        const float g = sigmoid_f(__half2float(z[base + d]));
        out[base + d] = __float2half(bf16_to_f32(w[d]) * (v[i] * inv) * g);
    }
}
