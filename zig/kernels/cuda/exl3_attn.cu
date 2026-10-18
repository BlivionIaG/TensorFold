// Qwen3.8-Flash-Next full-attention layer, EXL3 pack: everything between the EXL3 projections (q/k/v/index_qk, then o_proj).
// Model definition: qwen4_exp model_layers.SparseAttention + Indexer, and the Python family's glue.attn_prep / attention._pool/_scores/_select.
//
//   prep    q/k/indexer-q: fp32 RMSNorm with the (centred) scale, round to bf16, rotate-half RoPE on the first 64 dims
//           (positions are the 1D text position; mrope with equal axes reduces to it), round to bf16.
//           K and V go into the device KV table at `pos0[0] + row`; the raw (unnormed) index key goes to `ik`.
//   pool    every 4-key block the batch completes: fp32 mean of the raw index keys -> bf16 -> RMSNorm -> RoPE at the block's
//           first position (4*b) -> bf16.
//   scores  per row past 512 complete blocks: sum over the 4 index heads of relu(q . pooled) / sqrt(128), fp32.
//   select  per sparse row: the 512 best blocks (ties to the lower block id), in block order, as key indices, then the
//           unfinished tail (0..3 keys).
//   decode  per (row, q head): softmax(q . K / 16) V over the visible keys (dense up to 2051 keys, else the selected list),
//           then bf16(o) * sigmoid(gate) -> fp16 for the o_proj EXL3 linear.
//
// Every position comes from the device pointer `pos0`, so a CUDA graph replays these with a one-int update.
// KV table layout (identical to exl3_kv.Table.locate for one layer): [token][kv head][K 256 bf16 | V 256 bf16].

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace {
constexpr int HD = 256;       // head_dim
constexpr int NI = 4;         // indexer heads
constexpr int IHD = 128;      // indexer head dim
constexpr int HALF = 32;      // rotary_dim / 2 (rotary_dim = 0.25 * 256)
constexpr int RATIO = 4;      // indexer_compress_ratio
constexpr int TOP = 512;      // indexer_budget / ratio
constexpr int IDW = 2056;     // ids stride: TOP * RATIO + RATIO + 4
constexpr int MAXK = 2056;    // most keys a row ever reads (dense: end <= 2051, sparse: 2048 + tail)

__device__ __forceinline__ float bf16r(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

__device__ __forceinline__ float block_sum(float v, float* red, int tid, int nthreads) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int warps = nthreads >> 5;
    __syncthreads();
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    float t = 0.f;
    for (int w = 0; w < warps; ++w) t += red[w];
    return t;
}

__device__ __forceinline__ float block_max(float v, float* red, int tid, int nthreads) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    const int warps = nthreads >> 5;
    __syncthreads();
    if ((tid & 31) == 0) red[tid >> 5] = v;
    __syncthreads();
    float t = red[0];
    for (int w = 1; w < warps; ++w) t = fmaxf(t, red[w]);
    return t;
}

// RMSNorm (fp32, scale already centred), bf16, rotate-half RoPE over the first 2*HALF dims, bf16. `x` is this thread's input
// element (0 past `width`). `sm` holds `width` floats.
__device__ __forceinline__ float norm_rope(float x, int width, const float* w, const float* inv, float eps, float pos, float* sm,
                                           float* red, int tid, int nthreads) {
    const float ss = block_sum(tid < width ? x * x : 0.f, red, tid, nthreads);
    const float rinv = 1.0f / sqrtf(ss / (float)width + eps);
    float xn = 0.f;
    if (tid < width) {
        xn = bf16r(x * rinv * w[tid]);
        sm[tid] = xn;
    }
    __syncthreads();
    float out = xn;
    if (tid < HALF) {
        const float ang = pos * inv[tid];
        out = xn * cosf(ang) - sm[tid + HALF] * sinf(ang);
    } else if (tid < 2 * HALF) {
        const float ang = pos * inv[tid - HALF];
        out = sm[tid - HALF] * sinf(ang) + xn * cosf(ang);
    }
    return bf16r(out);
}
}  // namespace

// grid (rows, q_heads + 2 * kv_heads + NI + 1), block 256.
extern "C" __global__ void __launch_bounds__(256)
    exl3_attn_prep(const __half* __restrict__ qp, const __half* __restrict__ kp, const __half* __restrict__ vp,
                   const __half* __restrict__ ip, const float* __restrict__ qw, const float* __restrict__ kw,
                   const float* __restrict__ iw, const float* __restrict__ inv, const int* __restrict__ pos0,
                   __nv_bfloat16* __restrict__ q_out, __nv_bfloat16* __restrict__ kv, __nv_bfloat16* __restrict__ iq_out,
                   __nv_bfloat16* __restrict__ ik, int rows, int tokens, int q_heads, int kv_heads, float eps) {
    __shared__ float sm[HD];
    __shared__ float red[8];
    const int r = blockIdx.x;
    int role = blockIdx.y;
    const int tid = threadIdx.x;
    const int pos = pos0[0] + r;
    if (r >= rows || pos >= tokens) return;
    const float posf = (float)pos;

    if (role < q_heads) {
        const float x = __half2float(qp[((size_t)r * q_heads + role) * (2 * HD) + tid]);
        const float o = norm_rope(x, HD, qw, inv, eps, posf, sm, red, tid, 256);
        q_out[((size_t)r * q_heads + role) * HD + tid] = __float2bfloat16_rn(o);
        return;
    }
    role -= q_heads;
    if (role < kv_heads) {
        const float x = __half2float(kp[((size_t)r * kv_heads + role) * HD + tid]);
        const float o = norm_rope(x, HD, kw, inv, eps, posf, sm, red, tid, 256);
        kv[((size_t)pos * kv_heads + role) * (2 * HD) + tid] = __float2bfloat16_rn(o);
        return;
    }
    role -= kv_heads;
    if (role < kv_heads) {
        kv[((size_t)pos * kv_heads + role) * (2 * HD) + HD + tid] =
            __float2bfloat16_rn(__half2float(vp[((size_t)r * kv_heads + role) * HD + tid]));
        return;
    }
    role -= kv_heads;
    if (role < NI) {
        const float x = tid < IHD ? __half2float(ip[(size_t)r * (NI + 1) * IHD + role * IHD + tid]) : 0.f;
        const float o = norm_rope(x, IHD, iw, inv, eps, posf, sm, red, tid, 256);
        if (tid < IHD) iq_out[((size_t)r * NI + role) * IHD + tid] = __float2bfloat16_rn(o);
        return;
    }
    if (tid < IHD) ik[(size_t)pos * IHD + tid] = __float2bfloat16_rn(__half2float(ip[(size_t)r * (NI + 1) * IHD + NI * IHD + tid]));
}

// grid (rows / RATIO + 2), block 128. Pools block b = pos0 / RATIO + blockIdx.x when the batch completes it.
extern "C" __global__ void __launch_bounds__(128)
    exl3_attn_pool(const __nv_bfloat16* __restrict__ ik, __nv_bfloat16* __restrict__ pooled, const float* __restrict__ kw,
                   const float* __restrict__ inv, const int* __restrict__ pos0, int rows, int tokens, float eps) {
    __shared__ float sm[IHD];
    __shared__ float red[4];
    const int p0 = pos0[0];
    const int b = p0 / RATIO + (int)blockIdx.x;
    const int d = threadIdx.x;
    if (RATIO * b + RATIO > p0 + rows || RATIO * b + RATIO > tokens) return;
    float acc = __bfloat162float(ik[(size_t)(RATIO * b) * IHD + d]);
    for (int k = 1; k < RATIO; ++k) acc = acc + __bfloat162float(ik[(size_t)(RATIO * b + k) * IHD + d]);
    const float x = bf16r(acc / (float)RATIO);
    const float o = norm_rope(x, IHD, kw, inv, eps, (float)(RATIO * b), sm, red, d, 128);
    pooled[(size_t)b * IHD + d] = __float2bfloat16_rn(o);
}

// grid (rows, ceil(blocks / 128)), block 128. Rows with at most TOP complete blocks are dense and write nothing.
extern "C" __global__ void __launch_bounds__(128)
    exl3_attn_scores(const __nv_bfloat16* __restrict__ iq, const __nv_bfloat16* __restrict__ pooled, float* __restrict__ scores,
                     const int* __restrict__ pos0, int nb) {
    __shared__ float q[NI * IHD];
    const int r = blockIdx.x;
    const int complete = (pos0[0] + r + 1) / RATIO;
    if (complete <= TOP || (int)blockIdx.y * 128 >= complete) return;
    for (int i = threadIdx.x; i < NI * IHD; i += 128) q[i] = __bfloat162float(iq[(size_t)r * NI * IHD + i]);
    __syncthreads();
    const int b = (int)blockIdx.y * 128 + threadIdx.x;
    if (b >= complete) return;
    float total = 0.f;
    for (int h = 0; h < NI; ++h) {
        float dot = 0.f;
        for (int d = 0; d < IHD; ++d) dot = dot + q[h * IHD + d] * __bfloat162float(pooled[(size_t)b * IHD + d]);
        total = total + fmaxf(dot, 0.f);
    }
    scores[(size_t)r * nb + b] = total / sqrtf((float)IHD);
}

__device__ __forceinline__ uint32_t order_key(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// grid (rows), block 1024. For each sparse row: ids[r][0 .. TOP*RATIO) the keys of the TOP best blocks in block order
// (ties to the lower block id), then the unfinished tail. nk[r] is the key count (dense rows: the row's end).
extern "C" __global__ void __launch_bounds__(1024)
    exl3_attn_select(const float* __restrict__ scores, int* __restrict__ ids, int* __restrict__ nk, const int* __restrict__ pos0,
                     int nb) {
    __shared__ unsigned hist[256];
    __shared__ unsigned s_prefix, s_k;
    __shared__ int scan_a[1024];
    __shared__ int s_base_eq, s_base_ch;
    const int r = blockIdx.x;
    const int tid = threadIdx.x;
    const int end = pos0[0] + r + 1;
    const int complete = end / RATIO;
    if (complete <= TOP) {
        if (tid == 0) nk[r] = end;
        return;
    }
    const float* row = scores + (size_t)r * nb;
    if (tid == 0) {
        s_prefix = 0;
        s_k = TOP;
    }
    __syncthreads();
    // radix select of the TOP-th largest order key: four passes of one byte from the top
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        if (tid < 256) hist[tid] = 0;
        __syncthreads();
        const unsigned prefix = s_prefix;
        for (int b = tid; b < complete; b += 1024) {
            const uint32_t key = order_key(row[b]);
            if (pass == 0 || (key >> (shift + 8)) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
            unsigned cum = 0, k = s_k;
            int sel = 0;
            for (int dgt = 255; dgt >= 0; --dgt) {
                if (cum + hist[dgt] >= k) {
                    sel = dgt;
                    break;
                }
                cum += hist[dgt];
            }
            s_k = k - cum;
            s_prefix = (prefix << 8) | (unsigned)sel;
        }
        __syncthreads();
    }
    const uint32_t cut = s_prefix;
    const int need = (int)s_k;  // equal-to-cut blocks to take, lowest ids first
    if (tid == 0) {
        s_base_eq = 0;
        s_base_ch = 0;
    }
    __syncthreads();
    int* out = ids + (size_t)r * IDW;
    for (int c0 = 0; c0 < complete; c0 += 1024) {
        const int b = c0 + tid;
        const bool valid = b < complete;
        const uint32_t key = valid ? order_key(row[b]) : 0u;
        const int gt = valid && key > cut;
        const int eq = valid && key == cut;
        // inclusive scan of eq
        scan_a[tid] = eq;
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) {
            const int v = tid >= o ? scan_a[tid - o] : 0;
            __syncthreads();
            scan_a[tid] += v;
            __syncthreads();
        }
        const int eq_excl = scan_a[tid] - eq;
        const int eq_total = scan_a[1023];
        const int chosen = gt || (eq && (s_base_eq + eq_excl) < need);
        __syncthreads();
        scan_a[tid] = chosen;
        __syncthreads();
        for (int o = 1; o < 1024; o <<= 1) {
            const int v = tid >= o ? scan_a[tid - o] : 0;
            __syncthreads();
            scan_a[tid] += v;
            __syncthreads();
        }
        const int place = s_base_ch + scan_a[tid] - chosen;
        const int ch_total = scan_a[1023];
        if (chosen) {
            for (int k = 0; k < RATIO; ++k) out[place * RATIO + k] = b * RATIO + k;
        }
        __syncthreads();
        if (tid == 0) {
            s_base_eq += eq_total;
            s_base_ch += ch_total;
        }
        __syncthreads();
    }
    if (tid < RATIO) {
        const int idx = RATIO * complete + tid;
        if (idx < end) out[TOP * RATIO + tid] = idx;
    }
    if (tid == 0) nk[r] = TOP * RATIO + end - RATIO * complete;
}

// grid (rows, q_heads), block 256 (one thread per head dim). gate = the second half of this head's q_proj output.
extern "C" __global__ void __launch_bounds__(256)
    exl3_attn_decode(const __nv_bfloat16* __restrict__ q, const __half* __restrict__ qp, const __nv_bfloat16* __restrict__ kv,
                     const int* __restrict__ ids, __half* __restrict__ out, const int* __restrict__ pos0, int q_heads,
                     int kv_heads) {
    __shared__ float sq[HD];
    __shared__ float sc[MAXK];
    __shared__ int sk[MAXK];
    __shared__ float red[8];
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int end = pos0[0] + r + 1;
    const int complete = end / RATIO;
    const bool sparse = complete > TOP;
    const int nk = sparse ? TOP * RATIO + end - RATIO * complete : end;
    const int kvh = h / (q_heads / kv_heads);

    sq[tid] = __bfloat162float(q[((size_t)r * q_heads + h) * HD + tid]);
    for (int j = tid; j < nk; j += 256) sk[j] = sparse ? ids[(size_t)r * IDW + j] : j;
    __syncthreads();

    for (int j = warp; j < nk; j += 8) {
        const uint4 raw = *reinterpret_cast<const uint4*>(kv + ((size_t)sk[j] * kv_heads + kvh) * (2 * HD) + lane * 8);
        const __nv_bfloat16* e = reinterpret_cast<const __nv_bfloat16*>(&raw);
        float dot = 0.f;
        for (int i = 0; i < 8; ++i) dot = dot + sq[lane * 8 + i] * __bfloat162float(e[i]);
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) sc[j] = dot * 0.0625f;  // head_dim ** -0.5
    }
    __syncthreads();
    float m = -INFINITY;
    for (int j = tid; j < nk; j += 256) m = fmaxf(m, sc[j]);
    m = block_max(m, red, tid, 256);
    float l = 0.f;
    for (int j = tid; j < nk; j += 256) {
        const float p = expf(sc[j] - m);
        sc[j] = p;
        l += p;
    }
    l = block_sum(l, red, tid, 256);
    __syncthreads();
    float acc = 0.f;
    for (int j = 0; j < nk; ++j) {
        const float v = __bfloat162float(kv[((size_t)sk[j] * kv_heads + kvh) * (2 * HD) + HD + tid]);
        acc = acc + sc[j] * v;
    }
    const float o = bf16r(acc / l);
    const float g = __half2float(qp[((size_t)r * q_heads + h) * (2 * HD) + HD + tid]);
    const float res = bf16r(o / (1.0f + expf(-g)));
    out[((size_t)r * q_heads + h) * HD + tid] = __float2half(res);
}
