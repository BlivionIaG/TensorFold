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
    constexpr int kVLd = 256 + 2;
    constexpr int kScoreBytes = kRows * kQLd * 4 + kKeys * kKLd * static_cast<int>(sizeof(Cache));
    constexpr int kValueBytes = kKeys * kVLd * static_cast<int>(sizeof(Cache));
    __shared__ alignas(16) unsigned char stage[kScoreBytes > kValueBytes ? kScoreBytes : kValueBytes];
    __shared__ float sP[kRows * (kKeys + 1)];
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

    // score pass: rows sr .. sr + 3 against keys sk .. sk + 3; PV pass: rows pr .. pr + 15 at dims pd .. pd + 3
    const int sr = (tid / 16) * 4, sk = (tid % 16) * 4;
    const int pr = (tid / 64) * 16, pd = (tid % 64) * 4;
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; ++i)
#pragma unroll
        for (int m = 0; m < 4; ++m) o[i][m] = 0.f;
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
                sP[row * (kKeys + 1) + sk + j] = row < q_hi ? s : 0.f;
            }
        }
        __syncthreads();
        // fa_prefill's online softmax: the tile max a row a thread (fmaxf is exact in any order), every probability
        // at once (an exp of its own score), then each row's sum a thread over its keys in order
        __shared__ float sN[kRows];   // the tile's next max, -inf when the row has no visible key in it
        if (tid < kRows) {
            const int row = tid;
            const float* p = sP + row * (kKeys + 1);
            float tile_max = -INFINITY;
            bool any = false;
            if (row < q_hi) {
                for (int key = 0; key < kKeys; ++key) {
                    const float score = p[key];
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
            float* slot = sP + row * (kKeys + 1) + key;
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
                const float* p = sP + row * (kKeys + 1);
                const float alpha = sM[row] == -INFINITY ? 0.f : expf(sM[row] - next_m);
                float sum = 0.f;
                for (int key = 0; key < kKeys; ++key) sum += p[key];
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
            float pv[16][4];
#pragma unroll
            for (int i = 0; i < 16; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) pv[i][m] = 0.f;
            for (int key = 0; key < kKeys; ++key) {
                float vv[4];
#pragma unroll
                for (int m = 0; m < 4; ++m) vv[m] = wide(sV[key * kVLd + pd + m]);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    const float prob = sP[(pr + i) * (kKeys + 1) + key];
#pragma unroll
                    for (int m = 0; m < 4; ++m) pv[i][m] = fmaf(prob, vv[m], pv[i][m]);
                }
            }
#pragma unroll
            for (int i = 0; i < 16; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) o[i][m] = fmaf(sA[pr + i], o[i][m], pv[i][m]);
        }
    }
    __syncthreads();
    if (pd < d) {
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const int row = pr + i;
            if (row >= q_hi) continue;
            const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
            for (int m = 0; m < 4; ++m) orows[static_cast<long long>(row) * d + pd + m] = o[i][m] * inv;
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
               int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
               long long v_sb, long long v_sh, long long v_ss) {
    fa_wide<__half>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_wide<hip_bfloat16>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh,
                          v_ss);
}
