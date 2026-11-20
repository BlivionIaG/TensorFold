// GLPv3 paired steering; Matt Suiche/weightless and Arditi et al.; see docs/recipes/deepseek-v41-glp.md.
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "common/reduce.cuh"
#include "common/vec.cuh"

namespace dsv41_glp {

constexpr int NT = 320;                 // threads a CTA: a row's groups of eight columns, two a thread at width 5120
constexpr int GPT = 4;                  // groups of eight columns a thread holds
constexpr int MAXWIDTH = NT * 8 * GPT;  // widest row
constexpr int MAXW = 8;                 // ranks a writer reduces
constexpr int CB = 128;                 // columns a block of the final's square chain (the mHC final's NB x 128)
constexpr int ADD = 0, PROJECT = 1;

// Planted faults for the contract's controls; the production image instantiates NONE only.
enum Control : int { NONE = 0, SUM = 1, HALF = 2, DROP_TAIL = 3, DROP_NONFINITE = 4, ATOMIC = 5, TRUNCATE = 6 };

struct StreamArgs {
    uint16_t* x; long long xs;        // streams [R, 4 width] bf16, row stride xs: edited in place
    uint16_t* tap; long long ts;      // the DSpark tap [R, width] bf16, row stride ts (null: none)
    const float* dir;                 // [width]
    float alpha; int mode; int R; int width;
};
struct WriterArgs {
    uint16_t* g; long long gr; long long gs;   // gathered partials [world, R, width] bf16: rank stride, row stride
    const float* dir;
    float alpha; int mode; int R; int width; int world;
};
struct FinalArgs {
    const uint16_t* x; long long xs;  // streams [R, 4 width] bf16
    const float* pre;                 // [R, 4] the carried pre-mix
    const void* nw; uint16_t* out;    // norm weight [width] (bf16 or fp32), the normed input [R, width] bf16
    float eps; int nwbf16; int R; int width;
};

__device__ __forceinline__ uint4 ld16(const void* p) { return *reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ void widen8(const uint4 v, float (&f)[8]) {     // 8 bf16, column ascending -> fp32
    f[0] = __uint_as_float(v.x << 16); f[1] = __uint_as_float(v.x & 0xffff0000u);
    f[2] = __uint_as_float(v.y << 16); f[3] = __uint_as_float(v.y & 0xffff0000u);
    f[4] = __uint_as_float(v.z << 16); f[5] = __uint_as_float(v.z & 0xffff0000u);
    f[6] = __uint_as_float(v.w << 16); f[7] = __uint_as_float(v.w & 0xffff0000u);
}
template <int CTL> __device__ __forceinline__ uint32_t bits(float f) {       // the one bf16 rounding: RNE
    if (CTL == TRUNCATE) return __float_as_uint(f) >> 16;
    unsigned short h;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}
template <int CTL> __device__ __forceinline__ float rnd(float f) { return __uint_as_float(bits<CTL>(f) << 16); }
template <int CTL> __device__ __forceinline__ uint4 pack_controlled8(const float (&f)[8]) {
    if constexpr (CTL == TRUNCATE) {
        return make_uint4(bits<TRUNCATE>(f[0]) | (bits<TRUNCATE>(f[1]) << 16), bits<TRUNCATE>(f[2]) | (bits<TRUNCATE>(f[3]) << 16),
                          bits<TRUNCATE>(f[4]) | (bits<TRUNCATE>(f[5]) << 16), bits<TRUNCATE>(f[6]) | (bits<TRUNCATE>(f[7]) << 16));
    } else {
        return tf_common::pack8(f);
    }
}
__device__ __forceinline__ void load8(const float* p, float (&d)[8]) {
    const float4 a = __ldg(reinterpret_cast<const float4*>(p)), b = __ldg(reinterpret_cast<const float4*>(p) + 1);
    d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w; d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}

// Scale each BF16 copy by 1/4 before FP32 addition to avoid overflow; SUM deliberately omits that scale.
template <int CTL> __device__ __forceinline__ float mean4(float a, float b, float c, float d) {
    if (CTL == SUM) return __fadd_rn(__fadd_rn(__fadd_rn(a, b), c), d);
    return __fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(a, 0.25f), __fmul_rn(b, 0.25f)), __fmul_rn(c, 0.25f)),
                     __fmul_rn(d, 0.25f));
}

// one term of the dot chain: an fp32 fma (HALF: the planted fp16 accumulator; DROP_NONFINITE: a fmax-like drop)
template <int CTL> __device__ __forceinline__ float step(float acc, float m, float d) {
    if (CTL == HALF) return __half2float(__hadd(__float2half(acc), __float2half(__fmul_rn(m, d))));
    if (CTL == DROP_NONFINITE) {
        const float p = __fmul_rn(m, d);
        return isfinite(p) ? __fadd_rn(acc, p) : acc;
    }
    return __fmaf_rn(m, d, acc);
}

// the row's dot: every thread's chain, then a fixed order over the CTA (ATOMIC: the planted unordered fault)
template <int CTL> __device__ __forceinline__ float row_dot(float acc, float* scratch) {
    if (CTL == ATOMIC) {
        __shared__ float tot;
        if (threadIdx.x == 0) tot = 0.0f;
        __syncthreads();
        atomicAdd(&tot, acc);
        __syncthreads();
        return tot;
    }
    return tf_common::block_sum_serial(acc, scratch);
}

// the edit of one element: project subtracts s * d (s = alpha * dot), add adds alpha * d; fp32 ops, RNE by the caller
__device__ __forceinline__ float edit(int mode, float x, float s, float alpha, float d) {
    return mode == PROJECT ? __fsub_rn(x, __fmul_rn(s, d)) : __fadd_rn(x, __fmul_rn(alpha, d));
}

// ---- fold: the four copies in place, the tap of the steered copies ----
template <int CTL> __device__ __forceinline__ void stream_body(const StreamArgs& a) {
    if (a.alpha == 0.0f) return;
    __shared__ float scratch[NT / 32];
    const int r = blockIdx.x, t = threadIdx.x, ng = a.width >> 3, w = a.width;
    uint16_t* xr = a.x + static_cast<long long>(r) * a.xs;
    const bool proj = a.mode == PROJECT;
    uint4 v[GPT][4];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < GPT; ++i) {
        const int g = t + i * NT;
        if (g < ng) {
#pragma unroll
            for (int c = 0; c < 4; ++c) v[i][c] = ld16(xr + c * w + g * 8);
            if (proj) {
                float f[4][8], d[8];
#pragma unroll
                for (int c = 0; c < 4; ++c) widen8(v[i][c], f[c]);
                load8(a.dir + g * 8, d);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (CTL == DROP_TAIL && g == ng - 1 && k == 7) continue;
                    acc = step<CTL>(acc, mean4<CTL>(f[0][k], f[1][k], f[2][k], f[3][k]), d[k]);
                }
            }
        }
    }
    float s = 0.0f;
    if (proj) s = __fmul_rn(a.alpha, row_dot<CTL>(acc, scratch));
#pragma unroll
    for (int i = 0; i < GPT; ++i) {
        const int g = t + i * NT;
        if (g >= ng) continue;
        float f[4][8], d[8], y[4][8];
#pragma unroll
        for (int c = 0; c < 4; ++c) widen8(v[i][c], f[c]);
        load8(a.dir + g * 8, d);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
#pragma unroll
            for (int k = 0; k < 8; ++k) y[c][k] = rnd<CTL>(edit(a.mode, f[c][k], s, a.alpha, d[k]));
            *reinterpret_cast<uint4*>(xr + c * w + g * 8) = pack_controlled8<CTL>(y[c]);
        }
        if (a.tap != nullptr) {
            float tp[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) tp[k] = mean4<CTL>(y[0][k], y[1][k], y[2][k], y[3][k]);
            *reinterpret_cast<uint4*>(a.tap + static_cast<long long>(r) * a.ts + g * 8) = pack_controlled8<CTL>(tp);
        }
    }
}

// Store the edited writer in rank 0 and -0 elsewhere so hc_post's rank-ordered sum preserves its bits.
template <int CTL> __device__ __forceinline__ void writer_body(const WriterArgs& a) {
    if (a.alpha == 0.0f) return;
    __shared__ float scratch[NT / 32];
    const int r = blockIdx.x, t = threadIdx.x, ng = a.width >> 3;
    uint16_t* row = a.g + static_cast<long long>(r) * a.gs;
    const bool proj = a.mode == PROJECT;
    float br[GPT][8];
    float acc = 0.0f;
#pragma unroll
    for (int i = 0; i < GPT; ++i) {
        const int g = t + i * NT;
        if (g < ng) {
            widen8(ld16(row + g * 8), br[i]);
            for (int rk = 1; rk < a.world; ++rk) {
                float p[8];
                widen8(ld16(row + rk * a.gr + g * 8), p);
#pragma unroll
                for (int k = 0; k < 8; ++k) br[i][k] = __fadd_rn(br[i][k], p[k]);
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) br[i][k] = rnd<NONE>(br[i][k]);    // the post reads the rounded partial sum
            if (proj) {
                float d[8];
                load8(a.dir + g * 8, d);
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    if (CTL == DROP_TAIL && g == ng - 1 && k == 7) continue;
                    acc = step<CTL>(acc, br[i][k], d[k]);
                }
            }
        }
    }
    float s = 0.0f;
    if (proj) s = __fmul_rn(a.alpha, row_dot<CTL>(acc, scratch));
    const uint4 negzero = make_uint4(0x80008000u, 0x80008000u, 0x80008000u, 0x80008000u);
#pragma unroll
    for (int i = 0; i < GPT; ++i) {
        const int g = t + i * NT;
        if (g >= ng) continue;
        float d[8], y[8];
        load8(a.dir + g * 8, d);
#pragma unroll
        for (int k = 0; k < 8; ++k) y[k] = edit(a.mode, br[i][k], s, a.alpha, d[k]);
        *reinterpret_cast<uint4*>(row + g * 8) = pack_controlled8<CTL>(y);
        for (int rk = 1; rk < a.world; ++rk) *reinterpret_cast<uint4*>(row + rk * a.gr + g * 8) = negzero;
    }
}

// Match mhc_cuda's final collapse/RMS order on edited streams; the recipe documents rounding and error bounds.
template <int CTL> __device__ __forceinline__ void final_body(const FinalArgs& a) {
    extern __shared__ __align__(16) unsigned char dyn[];
    uint16_t* cs = reinterpret_cast<uint16_t*>(dyn);
    __shared__ float part[MAXWIDTH / CB];
    __shared__ float rn;
    const int r = blockIdx.x, t = threadIdx.x, ng = a.width >> 3, w = a.width;
    const uint16_t* xr = a.x + static_cast<long long>(r) * a.xs;
    const float q0 = __ldg(a.pre + r * 4), q1 = __ldg(a.pre + r * 4 + 1), q2 = __ldg(a.pre + r * 4 + 2), q3 = __ldg(a.pre + r * 4 + 3);
    for (int g = t; g < ng; g += NT) {
        float x[4][8], c[8];
#pragma unroll
        for (int s = 0; s < 4; ++s) widen8(ld16(xr + s * w + g * 8), x[s]);
#pragma unroll
        for (int k = 0; k < 8; ++k)
            c[k] = rnd<CTL>(__fadd_rn(__fadd_rn(__fadd_rn(__fmul_rn(q0, x[0][k]), __fmul_rn(q1, x[1][k])),
                                                __fmul_rn(q2, x[2][k])), __fmul_rn(q3, x[3][k])));
        *reinterpret_cast<uint4*>(cs + g * 8) = pack_controlled8<CTL>(c);
    }
    __syncthreads();
    const int nb = w / CB;
    if (t < nb) {
        float acc = 0.0f;
        for (int q = 0; q < CB / 8; ++q) {
            float v[8];
            widen8(*reinterpret_cast<const uint4*>(cs + t * CB + q * 8), v);
#pragma unroll
            for (int k = 0; k < 8; ++k) acc = __fmaf_rn(v[k], v[k], acc);
        }
        part[t] = acc;
    }
    __syncthreads();
    if (t == 0) {
        float total = 0.0f;
        for (int e = 0; e < nb; ++e) total = __fadd_rn(total, part[e]);
        rn = tf_common::rms_scale(total, w, a.eps);
    }
    __syncthreads();
    for (int g = t; g < ng; g += NT) {
        float c[8], wv[8], o[8];
        widen8(*reinterpret_cast<const uint4*>(cs + g * 8), c);
        if (a.nwbf16) {
            widen8(ld16(static_cast<const char*>(a.nw) + g * 16), wv);
        } else {
            load8(static_cast<const float*>(a.nw) + g * 8, wv);
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) o[k] = __fmul_rn(__fmul_rn(c[k], rn), wv[k]);
        *reinterpret_cast<uint4*>(a.out + static_cast<long long>(r) * w + g * 8) = pack_controlled8<CTL>(o);
    }
}

}  // namespace dsv41_glp
