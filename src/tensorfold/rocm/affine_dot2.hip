// RDNA2 FP16 activations. One thread owns one output and walks the whole K in pairs,
// so a row's bits do not depend on any other row. Codes are rounded through BF16
// and widened to FP16. The pair product is v_dot2_f32_f16; hipcc does not lower an
// hfma2 chain to that instruction on gfx1030. Scale and bias stay fp32, outside
// the group sum. The packed words stay the only copy of the weight.

#include "affine.hpp"
#include "arch.hpp"

#include <hip/hip_fp16.h>

namespace tf {
namespace rocm {

__device__ __half code_f16(uint32_t code) {
    return __float2half_rn(code_bf16(code));
}

__global__ void __launch_bounds__(32) affine_dot2_kernel(Affine a) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (row >= a.m || col >= a.n) return;

    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const uint32_t* wrow = a.words + static_cast<long long>(col) * words;
    const __half* xrow = static_cast<const __half*>(a.x) + static_cast<long long>(row) * a.k;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        float dot = 0.f;
        float sum = 0.f;
        int base = group * a.group;
        for (int t = 0; t < a.group; t += 2) {
            __half x0 = xrow[base + t];
            __half x1 = xrow[base + t + 1];
            __half q0 = code_f16(affine_code(wrow, base + t, a.bits, words));
            __half q1 = code_f16(affine_code(wrow, base + t + 1, a.bits, words));
            dot = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), __halves2half2(q0, q1), dot, false);
            sum += static_cast<float>(x0);
            sum += static_cast<float>(x1);
        }
        acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + group], acc);
    }
    a.out[static_cast<long long>(row) * a.n + col] = acc;
}

// Wide tile, including one row. 256 threads each own one output column. A K chunk is
// loaded once into shared memory (activations reused across the columns), then each
// thread walks that chunk with v_dot2_f32_f16 in pair order and applies the group
// scale only after the whole group. No atomics. A short column grid runs the same
// walk per group in affine_dot2_span and folds those groups in this order. The
// one-thread kernel remains for a group this tile cannot hold.
constexpr int kFastM = 8;
constexpr int kFastN = 256;
constexpr int kFastK = 64;
constexpr int kFastLd = 68;

__global__ void __launch_bounds__(kFastN, 1) affine_dot2_fast(Affine a) {
    __shared__ __half xs[kFastM * kFastLd];
    __shared__ __half ws[kFastN * kFastLd];
    int tid = threadIdx.x;
    int col = blockIdx.x * kFastN + tid;
    int row0 = blockIdx.y * kFastM;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    int chunk = a.group < kFastK ? a.group : kFastK;
    const __half* x = static_cast<const __half*>(a.x);
    float acc[kFastM];
#pragma unroll
    for (int m = 0; m < kFastM; ++m) acc[m] = 0.f;

    for (int group = 0; group < groups; ++group) {
        int gbase = group * a.group;
        float dot[kFastM];
        float sumx[kFastM];
#pragma unroll
        for (int m = 0; m < kFastM; ++m) {
            dot[m] = 0.f;
            sumx[m] = 0.f;
        }
        for (int k0 = 0; k0 < a.group; k0 += chunk) {
            if (a.bits == 8) {
                int words_per = chunk >> 2;
                int nwords = kFastN * words_per;
                for (int e = tid; e < nwords; e += kFastN) {
                    int c = e / words_per;
                    int w = e - c * words_per;
                    int gc = blockIdx.x * kFastN + c;
                    int gk = gbase + k0 + (w << 2);
                    uint32_t packed = 0u;
                    if (gc < a.n && gk < a.k) {
                        packed = a.words[static_cast<long long>(gc) * words + (gk >> 2)];
                    }
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        ws[c * kFastLd + (w << 2) + b] = code_f16((packed >> (8 * b)) & 0xffu);
                    }
                }
            } else {
                int span = kFastN * chunk;
                for (int e = tid; e < span; e += kFastN) {
                    int c = e / chunk;
                    int kk = e - c * chunk;
                    int gc = blockIdx.x * kFastN + c;
                    int gk = gbase + k0 + kk;
                    __half qv = __float2half(0.f);
                    if (gc < a.n && gk < a.k) {
                        const uint32_t* wrow = a.words + static_cast<long long>(gc) * words;
                        qv = code_f16(affine_code(wrow, gk, a.bits, words));
                    }
                    ws[c * kFastLd + kk] = qv;
                }
            }
            int xspan = kFastM * chunk;
            for (int e = tid; e < xspan; e += kFastN) {
                int r = e / chunk;
                int kk = e - r * chunk;
                int gr = row0 + r;
                int gk = gbase + k0 + kk;
                __half xv = __float2half(0.f);
                if (gr < a.m && gk < a.k) xv = x[static_cast<long long>(gr) * a.k + gk];
                xs[r * kFastLd + kk] = xv;
            }
            __syncthreads();
            if (col < a.n) {
                for (int t = 0; t < chunk; t += 2) {
                    __half2 q = __halves2half2(ws[tid * kFastLd + t], ws[tid * kFastLd + t + 1]);
#pragma unroll
                    for (int m = 0; m < kFastM; ++m) {
                        __half x0 = xs[m * kFastLd + t];
                        __half x1 = xs[m * kFastLd + t + 1];
                        dot[m] = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), q, dot[m], false);
                        sumx[m] += static_cast<float>(x0);
                        sumx[m] += static_cast<float>(x1);
                    }
                }
            }
            __syncthreads();
        }
        if (col < a.n) {
            float scale = a.scale[static_cast<long long>(col) * groups + group];
            float biasv = a.bias[static_cast<long long>(col) * groups + group];
#pragma unroll
            for (int m = 0; m < kFastM; ++m) {
                acc[m] = fmaf(dot[m], scale, acc[m]);
                acc[m] = fmaf(sumx[m], biasv, acc[m]);
            }
        }
    }
    if (col < a.n) {
#pragma unroll
        for (int m = 0; m < kFastM; ++m) {
            int row = row0 + m;
            if (row < a.m) a.out[static_cast<long long>(row) * a.n + col] = acc[m];
        }
    }
}

// Prefill tile. Each output is still one thread's fdot2 chain in pair order, so the
// bits match the one-thread kernel.
constexpr int kDotBM = 16;
constexpr int kDotBN = 16;
constexpr int kDotGroup = 128;

__global__ void affine_dot2_tiled(Affine a) {
    __shared__ __half xs[kDotBM * kDotGroup];
    __shared__ __half ws[kDotBN * kDotGroup];
    int local_row = threadIdx.x / kDotBN;
    int local_col = threadIdx.x % kDotBN;
    int row = blockIdx.y * kDotBM + local_row;
    int col = blockIdx.x * kDotBN + local_col;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    bool live = row < a.m && col < a.n;
    float acc = 0.f;
    const __half* x = static_cast<const __half*>(a.x);

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        int span = kDotBM * a.group;
        for (int e = threadIdx.x; e < span; e += blockDim.x) {
            int r = e / a.group;
            int c = e - r * a.group;
            int gr = blockIdx.y * kDotBM + r;
            int gk = base + c;
            __half xv = __float2half(0.f);
            if (gr < a.m && gk < a.k) xv = x[static_cast<long long>(gr) * a.k + gk];
            xs[r * a.group + c] = xv;
        }
        for (int e = threadIdx.x; e < span; e += blockDim.x) {
            int r = e / a.group;
            int c = e - r * a.group;
            int gc = blockIdx.x * kDotBN + r;
            int gk = base + c;
            __half qv = __float2half(0.f);
            if (gc < a.n && gk < a.k) {
                const uint32_t* wrow = a.words + static_cast<long long>(gc) * words;
                qv = code_f16(affine_code(wrow, gk, a.bits, words));
            }
            ws[r * a.group + c] = qv;
        }
        __syncthreads();
        if (live) {
            float dot = 0.f;
            float sum = 0.f;
            const __half* xrow = xs + local_row * a.group;
            const __half* wrow = ws + local_col * a.group;
            for (int t = 0; t < a.group; t += 2) {
                __half x0 = xrow[t];
                __half x1 = xrow[t + 1];
                __half q0 = wrow[t];
                __half q1 = wrow[t + 1];
                dot = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), __halves2half2(q0, q1), dot, false);
                sum += static_cast<float>(x0);
                sum += static_cast<float>(x1);
            }
            int g = group;
            acc = fmaf(dot, a.scale[static_cast<long long>(col) * groups + g], acc);
            acc = fmaf(sum, a.bias[static_cast<long long>(col) * groups + g], acc);
        }
        __syncthreads();
    }
    if (live) a.out[static_cast<long long>(row) * a.n + col] = acc;
}

// Same pair walk as affine_dot2_fast for groups [z * per, z * per + per). Writes each
// group's dot and sum of x. The fold kernel applies scale and bias in group order.
__global__ void __launch_bounds__(kFastN, 1) affine_dot2_span(Affine a, float* partial, int per) {
    __shared__ __half xs[kFastM * kFastLd];
    __shared__ __half ws[kFastN * kFastLd];
    int tid = threadIdx.x;
    int col = blockIdx.x * kFastN + tid;
    int row0 = blockIdx.y * kFastM;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    int chunk = a.group < kFastK ? a.group : kFastK;
    int g0 = blockIdx.z * per;
    int g1 = g0 + per;
    const __half* x = static_cast<const __half*>(a.x);

    for (int group = g0; group < g1; ++group) {
        int gbase = group * a.group;
        float dot[kFastM];
        float sumx[kFastM];
#pragma unroll
        for (int m = 0; m < kFastM; ++m) {
            dot[m] = 0.f;
            sumx[m] = 0.f;
        }
        for (int k0 = 0; k0 < a.group; k0 += chunk) {
            if (a.bits == 8) {
                int words_per = chunk >> 2;
                int nwords = kFastN * words_per;
                for (int e = tid; e < nwords; e += kFastN) {
                    int c = e / words_per;
                    int w = e - c * words_per;
                    int gc = blockIdx.x * kFastN + c;
                    int gk = gbase + k0 + (w << 2);
                    uint32_t packed = 0u;
                    if (gc < a.n && gk < a.k) {
                        packed = a.words[static_cast<long long>(gc) * words + (gk >> 2)];
                    }
#pragma unroll
                    for (int b = 0; b < 4; ++b) {
                        ws[c * kFastLd + (w << 2) + b] = code_f16((packed >> (8 * b)) & 0xffu);
                    }
                }
            } else {
                int span = kFastN * chunk;
                for (int e = tid; e < span; e += kFastN) {
                    int c = e / chunk;
                    int kk = e - c * chunk;
                    int gc = blockIdx.x * kFastN + c;
                    int gk = gbase + k0 + kk;
                    __half qv = __float2half(0.f);
                    if (gc < a.n && gk < a.k) {
                        const uint32_t* wrow = a.words + static_cast<long long>(gc) * words;
                        qv = code_f16(affine_code(wrow, gk, a.bits, words));
                    }
                    ws[c * kFastLd + kk] = qv;
                }
            }
            int xspan = kFastM * chunk;
            for (int e = tid; e < xspan; e += kFastN) {
                int r = e / chunk;
                int kk = e - r * chunk;
                int gr = row0 + r;
                int gk = gbase + k0 + kk;
                __half xv = __float2half(0.f);
                if (gr < a.m && gk < a.k) xv = x[static_cast<long long>(gr) * a.k + gk];
                xs[r * kFastLd + kk] = xv;
            }
            __syncthreads();
            if (col < a.n) {
                for (int t = 0; t < chunk; t += 2) {
                    __half2 q = __halves2half2(ws[tid * kFastLd + t], ws[tid * kFastLd + t + 1]);
#pragma unroll
                    for (int m = 0; m < kFastM; ++m) {
                        __half x0 = xs[m * kFastLd + t];
                        __half x1 = xs[m * kFastLd + t + 1];
                        dot[m] = __builtin_amdgcn_fdot2(__halves2half2(x0, x1), q, dot[m], false);
                        sumx[m] += static_cast<float>(x0);
                        sumx[m] += static_cast<float>(x1);
                    }
                }
            }
            __syncthreads();
        }
        if (col < a.n) {
#pragma unroll
            for (int m = 0; m < kFastM; ++m) {
                int row = row0 + m;
                if (row < a.m) {
                    float* slot = partial + ((static_cast<long long>(row) * a.n + col) * groups + group) * 2;
                    slot[0] = dot[m];
                    slot[1] = sumx[m];
                }
            }
        }
    }
}

__global__ void __launch_bounds__(kFastN) affine_dot2_fold(const float* partial, const float* scale, const float* bias,
                                                          float* out, int m, int n, int groups) {
    int col = blockIdx.x * kFastN + threadIdx.x;
    int row = blockIdx.y;
    if (row >= m || col >= n) return;
    const float* base = partial + (static_cast<long long>(row) * n + col) * groups * 2;
    float acc = 0.f;
    for (int group = 0; group < groups; ++group) {
        acc = fmaf(base[group * 2], scale[static_cast<long long>(col) * groups + group], acc);
        acc = fmaf(base[group * 2 + 1], bias[static_cast<long long>(col) * groups + group], acc);
    }
    out[static_cast<long long>(row) * n + col] = acc;
}

// mode 1 keeps one launch. mode 2 splits even a wide grid. 0 splits only a short grid.
int affine_dot2_splits(int m, int n, int k, int group, int mode) {
    if (mode == 1 || m < 1 || n < 512) return 1;
    if (group != 32 && group != 64 && group != 128) return 1;
    if (k < group || k % group != 0) return 1;
    int groups = k / group;
    if (groups < 2) return 1;
    // A full row tile (M >= 8) and a wide column grid already fill the card. Gate is 48
    // column blocks and does not get faster. Down, the square projections, and qkv do.
    int blocks = ((n + kFastN - 1) / kFastN) * ((m + kFastM - 1) / kFastM);
    if (blocks < 1 || (mode != 2 && (m >= kFastM || blocks >= 40))) return 1;
    int want = 256 / blocks;
    if (mode == 2 && want < 2) want = 2;
    if (want < 2) return 1;
    if (want > groups) want = groups;
    for (int splits = want; splits >= 2; --splits) {
        if (groups % splits == 0) return splits;
    }
    return 1;
}

hipError_t launch_affine_dot2_split(const Affine& a, float* partial, int splits, hipStream_t stream) {
    if (!a.fp16 || partial == nullptr || splits < 2) return hipErrorInvalidValue;
    int groups = a.k / a.group;
    if (groups % splits != 0) return hipErrorInvalidValue;
    dim3 block(kFastN);
    dim3 grid((a.n + kFastN - 1) / kFastN, (a.m + kFastM - 1) / kFastM, splits);
    affine_dot2_span<<<grid, block, 0, stream>>>(a, partial, groups / splits);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) return err;
    dim3 fold((a.n + kFastN - 1) / kFastN, a.m);
    affine_dot2_fold<<<fold, block, 0, stream>>>(partial, a.scale, a.bias, a.out, a.m, a.n, groups);
    return hipGetLastError();
}

hipError_t launch_affine_dot2(const Affine& a, hipStream_t stream) {
    if (a.m < 1 || a.n < 1 || a.k < 1 || a.group % 2 != 0 || a.k % a.group != 0) return hipErrorInvalidValue;
    if (a.group == 32 || a.group == 64 || a.group == 128) {
        dim3 block(kFastN);
        dim3 grid((a.n + kFastN - 1) / kFastN, (a.m + kFastM - 1) / kFastM);
        affine_dot2_fast<<<grid, block, 0, stream>>>(a);
        return hipGetLastError();
    }
    if (a.m > 1 && a.group <= kDotGroup) {
        dim3 block(kDotBM * kDotBN);
        dim3 grid((a.n + kDotBN - 1) / kDotBN, (a.m + kDotBM - 1) / kDotBM);
        affine_dot2_tiled<<<grid, block, 0, stream>>>(a);
        return hipGetLastError();
    }
    dim3 block(32);
    dim3 grid((a.n + 31) / 32, a.m);
    affine_dot2_kernel<<<grid, block, 0, stream>>>(a);
    return hipGetLastError();
}

}  // namespace rocm
}  // namespace tf

int affine_dot2_splits(int m, int n, int k, int group, int mode) {
    return tf::rocm::affine_dot2_splits(m, n, k, group, mode);
}
