// Prefill attention with fa_prefill's bits (rocm/attention_fa.hip) for long prompts: 64 query rows a block share each
// 64-key tile. A score keeps fa_pair's four accumulators over the head dimension (dim mod 4, ascending), the tile's
// online softmax stays one row a thread, and every output keeps its key-ordered sum, so a row's bits are the same.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace {

constexpr int kRows = 64;     // query rows a block
constexpr int kKeys = 64;     // keys a tile (fa_prefill's Bc for a 16-bit cache: the softmax steps)
constexpr int kChunk = 64;    // head dimensions a score pass stages
constexpr int kThreads = 256;

__device__ __forceinline__ float wide(__half v) { return __half2float(v); }
__device__ __forceinline__ float wide(hip_bfloat16 v) { return static_cast<float>(v); }

template <typename Cache>
__device__ void fa_wide(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale,
                        int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
                        long long v_ss) {
    // the score pass's q and k chunks, or the value tile (64 keys x 256 dims): one buffer
    constexpr int kQLd = kChunk + 1;
    constexpr int kKLd = kChunk + 2;
    constexpr int kVLd = 256 + 8;
    constexpr int kScoreBytes = kRows * kQLd * 4 + kKeys * kKLd * static_cast<int>(sizeof(Cache));
    constexpr int kValueBytes = kKeys * kVLd * static_cast<int>(sizeof(Cache));
    __shared__ alignas(16) unsigned char stage[kScoreBytes > kValueBytes ? kScoreBytes : kValueBytes];
    constexpr int kPLd = kRows + 4;   // probabilities key-major: a thread's four rows of a key in one read
    __shared__ alignas(16) float sP[kKeys * kPLd];
    __shared__ float sM[kRows];
    __shared__ float sL[kRows];
    __shared__ float sA[kRows];
    float* sQ = reinterpret_cast<float*>(stage);
    Cache* sK = reinterpret_cast<Cache*>(stage + kRows * kQLd * 4);
    Cache* sV = reinterpret_cast<Cache*>(stage);

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const Cache* kbase = k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const Cache* vbase = v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;

    // score pass: rows sr .. sr + 3 against keys sk .. sk + 3; PV pass: rows pr .. pr + 3 at dims pd .. pd + 15
    const int sr = (tid / 16) * 4, sk = (tid % 16) * 4;
    const int pr = (tid / 16) * 4, pd = (tid % 16) * 16;
    float o[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int m = 0; m < 16; ++m) o[i][m] = 0.f;
    if (tid < kRows) {
        sM[tid] = -INFINITY;
        sL[tid] = 0.f;
        sA[tid] = 1.f;
    }

    for (int k0 = 0; k0 < max_vis; k0 += kKeys) {
        float acc[4][4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j)
#pragma unroll
                for (int m = 0; m < 4; ++m) acc[i][j][m] = 0.f;
        for (int c0 = 0; c0 < d; c0 += kChunk) {
            __syncthreads();
            for (int e = tid; e < kRows * kChunk; e += kThreads) {
                const int row = e / kChunk, dim = e % kChunk;
                sQ[row * kQLd + dim] = row < q_hi ? qrows[static_cast<long long>(row) * d + c0 + dim] : 0.f;
            }
            for (int e = tid; e < kKeys * kChunk; e += kThreads) {
                const int key = e / kChunk, dim = e % kChunk;
                sK[key * kKLd + dim] = k0 + key < span ? kbase[static_cast<long long>(k0 + key) * k_ss + c0 + dim]
                                                       : Cache(0.f);
            }
            __syncthreads();
#pragma unroll 4
            for (int dd = 0; dd < kChunk; dd += 4) {
                float qv[4][4], kf[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int m = 0; m < 4; ++m) qv[i][m] = sQ[(sr + i) * kQLd + dd + m];
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int m = 0; m < 4; ++m) kf[j][m] = wide(sK[(sk + j) * kKLd + dd + m]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int m = 0; m < 4; ++m) acc[i][j][m] = fmaf(qv[i][m], kf[j][m], acc[i][j][m]);
            }
        }
        // fa_pair's sum of its four accumulators, the scale, the causal mask (fa_prefill: a row past q_hi scores 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + i;
            int vis = q_pos0 + q0 + row + 1;
            if (vis > span) vis = span;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float s = (acc[i][j][0] + acc[i][j][1]) + (acc[i][j][2] + acc[i][j][3]);
                s *= scale;
                if (k0 + sk + j >= vis) s = -INFINITY;
                sP[(sk + j) * kPLd + row] = row < q_hi ? s : 0.f;
            }
        }
        __syncthreads();
        // fa_prefill's online softmax: the tile max a row a thread (fmaxf is exact in any order), every probability
        // at once (an exp of its own score), then each row's sum a thread over its keys in order
        __shared__ float sN[kRows];   // the tile's next max, -inf when the row has no visible key in it
        if (tid < kRows) {
            const int row = tid;
            const float* p = sP + row;
            float tile_max = -INFINITY;
            bool any = false;
            if (row < q_hi) {
                for (int key = 0; key < kKeys; ++key) {
                    const float score = p[key * kPLd];
                    if (score != -INFINITY) {
                        tile_max = fmaxf(tile_max, score);
                        any = true;
                    }
                }
            }
            sN[row] = any ? fmaxf(sM[row], tile_max) : -INFINITY;
        }
        __syncthreads();
        for (int e = tid; e < kRows * kKeys; e += kThreads) {
            const int row = e / kKeys, key = e % kKeys;
            float* slot = sP + key * kPLd + row;
            const float next_m = sN[row];
            *slot = (next_m == -INFINITY || *slot == -INFINITY) ? 0.f : expf(*slot - next_m);
        }
        __syncthreads();
        if (tid < kRows) {
            const int row = tid;
            const float next_m = sN[row];
            if (next_m == -INFINITY) {
                sA[row] = 1.f;
            } else {
                const float* p = sP + row;
                const float alpha = sM[row] == -INFINITY ? 0.f : expf(sM[row] - next_m);
                float sum = 0.f;
                for (int key = 0; key < kKeys; ++key) sum += p[key * kPLd];
                sL[row] = fmaf(alpha, sL[row], sum);
                sM[row] = next_m;
                sA[row] = alpha;
            }
        }
        for (int e = tid; e < kKeys * d; e += kThreads) {
            const int key = e / d, dim = e % d;
            sV[key * kVLd + dim] = k0 + key < span ? vbase[static_cast<long long>(k0 + key) * v_ss + dim] : Cache(0.f);
        }
        __syncthreads();
        // fa_prefill's PV: an output's products over the tile's keys in order, then the rescale of what came before
        if (pd < d) {
            float pv[4][16];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 16; ++m) pv[i][m] = 0.f;
            for (int key = 0; key < kKeys; ++key) {
                const float4 p4 = *reinterpret_cast<const float4*>(sP + key * kPLd + pr);
                const float prob[4] = {p4.x, p4.y, p4.z, p4.w};
                float vv[16];
#pragma unroll
                for (int m = 0; m < 16; ++m) vv[m] = wide(sV[key * kVLd + pd + m]);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int m = 0; m < 16; ++m) pv[i][m] = fmaf(prob[i], vv[m], pv[i][m]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 16; ++m) o[i][m] = fmaf(sA[pr + i], o[i][m], pv[i][m]);
        }
    }
    __syncthreads();
    if (pd < d) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = pr + i;
            if (row >= q_hi) continue;
            const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
            for (int m = 0; m < 16; ++m) orows[static_cast<long long>(row) * d + pd + m] = o[i][m] * inv;
        }
    }
}


using half2_t = _Float16 __attribute__((ext_vector_type(2)));

// A 16-byte LDS read as four fp16 pairs.
__device__ __forceinline__ void pairs4(const void* at, half2_t out[4]) {
    const uint4 w = *reinterpret_cast<const uint4*>(at);
    out[0] = __builtin_bit_cast(half2_t, w.x);
    out[1] = __builtin_bit_cast(half2_t, w.y);
    out[2] = __builtin_bit_cast(half2_t, w.z);
    out[3] = __builtin_bit_cast(half2_t, w.w);
}

// fa_wide on an fp16 cache with dot2 (fp16 pairs into fp32) for both products. q is rounded to fp16 once and stays in
// LDS, a tile's scores stay in registers (16 lanes share a row: max and sum by shuffles) and only the probabilities
// go through LDS, as fp16 key pairs. Shared-memory rows are XOR-swizzled by 16-byte chunk against bank conflicts.
__device__ void fa_dot2(const float* __restrict__ q, const __half* __restrict__ k, const __half* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale,
                        int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
                        long long v_ss) {
    constexpr int kQChunks = 256 / 8;      // 16-byte chunks a q row
    constexpr int kKChunks = kChunk / 8;   // a staged key row's
    constexpr int kPLd = kRows + 4;        // a key pair's probabilities, row-indexed
    __shared__ uint4 sQ[kRows * kQChunks];
    __shared__ uint4 sKV[kKeys * kKChunks];   // a key chunk (key, 8 dims) or a value chunk (key pair, 4 dims)
    __shared__ alignas(16) half2_t sH[kKeys / 2 * kPLd];
    __shared__ float sA[kRows], sL[kRows];

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const __half* kbase = k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const __half* vbase = v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;
    const int chunks = d / 8;

    for (int e = tid; e < kRows * chunks; e += kThreads) {
        const int row = e / chunks, c = e % chunks;
        _Float16 h[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            h[m] = static_cast<_Float16>(row < q_hi ? qrows[static_cast<long long>(row) * d + 8 * c + m] : 0.f);
        sQ[row * kQChunks + (c ^ (row & 7))] = __builtin_bit_cast(uint4, h);
    }

    // scores: rows sr + 16i against keys 4sk .. 4sk + 3; PV: rows pr .. pr + 3 at dims 4pc .. 4pc + 3 of each chunk
    const int sr = tid / 16, sk = tid % 16;
    const int pr = (tid / 16) * 4, pc = tid % 16;
    float m_run[4], l_run[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = -INFINITY, l_run[i] = 0.f;
    float o[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int m = 0; m < 16; ++m) o[i][m] = 0.f;

    for (int k0 = 0; k0 < max_vis; k0 += kKeys) {
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int c0 = 0; c0 < chunks; c0 += kKChunks) {
            uint4 staged[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                staged[t] = k0 + key < span
                                ? *reinterpret_cast<const uint4*>(kbase + static_cast<long long>(k0 + key) * k_ss + 8 * (c0 + c))
                                : uint4{0, 0, 0, 0};
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                sKV[key * kKChunks + (c ^ ((key >> 2) & 7))] = staged[t];
            }
            __syncthreads();
#pragma unroll 2
            for (int c = 0; c < kKChunks; ++c) {
                half2_t qv[4][4], kf[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = sr + 16 * i;
                    pairs4(&sQ[row * kQChunks + ((c0 + c) ^ (row & 7))], qv[i]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int key = 4 * sk + j;
                    pairs4(&sKV[key * kKChunks + (c ^ ((key >> 2) & 7))], kf[j]);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int m = 0; m < 4; ++m) acc[i][j] = __builtin_amdgcn_fdot2(qv[i][m], kf[j][m], acc[i][j], false);
            }
        }
        // the online softmax in registers: a row's 64 keys sit on its 16 lanes
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            int vis = q_pos0 + q0 + row + 1;
            if (vis > span) vis = span;
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = (row < q_hi && k0 + 4 * sk + j < vis) ? acc[i][j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[i][j]);
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, lane, 16));
            const float next_m = fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : __expf(m_run[i] - next_m);
            float p[4], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = next_m == -INFINITY ? 0.f : __expf(acc[i][j] - next_m);
                sum += p[j];
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) sum += __shfl_xor(sum, lane, 16);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
            sH[(2 * sk) * kPLd + row] = half2_t{static_cast<_Float16>(p[0]), static_cast<_Float16>(p[1])};
            sH[(2 * sk + 1) * kPLd + row] = half2_t{static_cast<_Float16>(p[2]), static_cast<_Float16>(p[3])};
            if (sk == 0) sA[row] = alpha;
        }
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
            const int c0 = at * kKChunks;
            // a value chunk: key pair `pair`, dims 8g .. 8g + 7 interleaved as (key, key + 1) fp16 pairs
            const int pair = tid / kKChunks, g = tid % kKChunks;
            const int key = k0 + 2 * pair;
            const uint4 a = key < span ? *reinterpret_cast<const uint4*>(vbase + static_cast<long long>(key) * v_ss + 8 * (c0 + g))
                                       : uint4{0, 0, 0, 0};
            const uint4 b = key + 1 < span
                                ? *reinterpret_cast<const uint4*>(vbase + static_cast<long long>(key + 1) * v_ss + 8 * (c0 + g))
                                : uint4{0, 0, 0, 0};
            __syncthreads();
            const _Float16* ha = reinterpret_cast<const _Float16*>(&a);
            const _Float16* hb = reinterpret_cast<const _Float16*>(&b);
            half2_t lo[4], hi[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) lo[m] = half2_t{ha[m], hb[m]}, hi[m] = half2_t{ha[4 + m], hb[4 + m]};
            sKV[pair * 16 + 2 * g] = __builtin_bit_cast(uint4, lo);
            sKV[pair * 16 + 2 * g + 1] = __builtin_bit_cast(uint4, hi);
            __syncthreads();
            float pv[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) pv[i][m] = 0.f;
#pragma unroll 4
            for (int pp = 0; pp < kKeys / 2; ++pp) {
                half2_t pr4[4], vv[4];
                pairs4(sH + pp * kPLd + pr, pr4);
                pairs4(&sKV[pp * 16 + pc], vv);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int m = 0; m < 4; ++m) pv[i][m] = __builtin_amdgcn_fdot2(pr4[i], vv[m], pv[i][m], false);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float& acc_o = o[i][4 * at + m];
                    acc_o = fmaf(sA[pr + i], acc_o, pv[i][m]);
                }
        }
    }
    if (sk == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) sL[sr + 16 * i] = l_run[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = pr + i;
        if (row >= q_hi) continue;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
#pragma unroll
            for (int m = 0; m < 4; ++m)
                orows[static_cast<long long>(row) * d + 64 * at + 4 * pc + m] = o[i][4 * at + m] * inv;
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
               int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
               long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_wide<hip_bfloat16>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh,
                          v_ss);
}
