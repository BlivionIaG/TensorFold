#pragma once

#include "../ops/common.hpp"

// Attention over a row's listed keys (the indexer's), or its first n keys, in parts merged in part order.

namespace {

constexpr int kSparseWaves = 8;
constexpr int kSparseMostDims = 256;
constexpr int kSparsePer = kSparseMostDims / 32;

// Block (h, r, p): head h of row r over part p of its keys; wave w takes every 8th key from the part's start.
extern "C" __global__ void __launch_bounds__(kSparseWaves * 32)
tf_fn_sparse_parts(const float* q, const void* k, const void* v, int cache_kind, const int* ids, long long id_stride,
                   const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                   int dims, long long cap, float scale, int parts) {
    __shared__ float ms[kSparseWaves][2];
    __shared__ float tile[kSparseWaves][kSparseMostDims];
    const int h = blockIdx.x, r = blockIdx.y, part = blockIdx.z;
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / parts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / parts);
    const long long head_base = static_cast<long long>(h / (heads / kv_heads)) * cap * dims;
    const float* qh = q + (static_cast<long long>(r) * heads + h) * dims;
    float qr[kSparsePer], o[kSparsePer];
#pragma unroll
    for (int i = 0; i < kSparsePer; ++i) {
        const int d = lane + 32 * i;
        qr[i] = d < dims ? qh[d] * scale : 0.f;
        o[i] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int j = lo + wave; j < hi; j += kSparseWaves) {
        const long long key = listed ? ids[r * id_stride + j] : j;
        const long long at = head_base + key * dims;
        float s = 0.f;
#pragma unroll
        for (int i = 0; i < kSparsePer; ++i) {
            const int d = lane + 32 * i;
            if (d < dims) s += qr[i] * load(k, cache_kind, at + d);
        }
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) s += __shfl_xor(s, mask, 32);
        const float top = fmaxf(m, s);
        const float f = expf(m - top), e = expf(s - top);
        l = l * f + e;
#pragma unroll
        for (int i = 0; i < kSparsePer; ++i) {
            const int d = lane + 32 * i;
            if (d < dims) o[i] = o[i] * f + e * load(v, cache_kind, at + d);
        }
        m = top;
    }
    if (lane == 0) {
        ms[wave][0] = m;
        ms[wave][1] = l;
    }
    __syncthreads();
    float top = -INFINITY;
    for (int w = 0; w < kSparseWaves; ++w) top = fmaxf(top, ms[w][0]);
    const float mine = m == -INFINITY ? 0.f : expf(m - top);
#pragma unroll
    for (int i = 0; i < kSparsePer; ++i) tile[wave][lane + 32 * i] = o[i] * mine;
    __syncthreads();
    const long long at = (static_cast<long long>(r) * heads + h) * parts + part;
    for (int d = threadIdx.x; d < dims; d += kSparseWaves * 32) {
        float acc = 0.f;
        for (int w = 0; w < kSparseWaves; ++w) acc += tile[w][d];
        parts_o[at * dims + d] = acc;
    }
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int w = 0; w < kSparseWaves; ++w) total += ms[w][0] == -INFINITY ? 0.f : ms[w][1] * expf(ms[w][0] - top);
        parts_ms[at * 2] = top;
        parts_ms[at * 2 + 1] = total;
    }
}

// Block (h, r), a thread a dim: the parts of head h of row r combined in part order, fp32 rows-major out.
extern "C" __global__ void tf_fn_sparse_merge(const float* parts_o, const float* parts_ms, float* out, int heads,
                                              int dims, int parts) {
    const int h = blockIdx.x, r = blockIdx.y;
    const long long at = (static_cast<long long>(r) * heads + h) * parts;
    float top = -INFINITY;
    for (int p = 0; p < parts; ++p) top = fmaxf(top, parts_ms[(at + p) * 2]);
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        float total = 0.f, acc = 0.f;
        for (int p = 0; p < parts; ++p) {
            const float mp = parts_ms[(at + p) * 2];
            const float w = mp == -INFINITY ? 0.f : expf(mp - top);
            total += parts_ms[(at + p) * 2 + 1] * w;
            acc += parts_o[(at + p) * dims + d] * w;
        }
        out[(static_cast<long long>(r) * heads + h) * dims + d] = total > 0.f ? acc / total : 0.f;
    }
}

}  // namespace
