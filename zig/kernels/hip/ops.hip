// The PyTorch ops between the ROCm forward's own kernels, with their bits: one rounding where torch rounds, the same
// libm calls, no contraction (-ffp-contract=off). kind: 0 fp32, 1 fp16, 2 bf16 (act.hpp's numbering).

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

__device__ __forceinline__ float load(const void* p, int kind, long long i) {
    if (kind == 0) return static_cast<const float*>(p)[i];
    if (kind == 1) return __half2float(static_cast<const __half*>(p)[i]);
    return bf16_to_float(static_cast<const uint16_t*>(p)[i]);
}

__device__ __forceinline__ void store(void* p, int kind, long long i, float v) {
    if (kind == 0) static_cast<float*>(p)[i] = v;
    else if (kind == 1) static_cast<__half*>(p)[i] = __float2half_rn(v);
    else static_cast<uint16_t*>(p)[i] = float_to_bf16(v);
}

// A value rounded to `kind` and widened again: what torch holds after an op that returns that dtype.
__device__ __forceinline__ float rounded(float v, int kind) {
    if (kind == 1) return __half2float(__float2half_rn(v));
    if (kind == 2) return bf16_to_float(float_to_bf16(v));
    return v;
}

// torch's silu and sigmoid on an fp32 operand: x / (1 + exp(-x)) and 1 / (1 + exp(-x)).
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__device__ __forceinline__ long long gid() { return static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x; }

// An MLX code at index i of a packed row, including one that crosses two words (qwen_math._codes).
__device__ __forceinline__ uint32_t code_at(const uint32_t* row, int words, int bits, long long i) {
    long long bit = i * bits;
    int word = static_cast<int>(bit / 32), shift = static_cast<int>(bit % 32);
    uint64_t low = row[word];
    uint64_t high = (shift + bits > 32 && word + 1 < words) ? row[word + 1] : 0;
    uint64_t value = (low >> shift) | (high << ((32 - shift) & 31));
    return static_cast<uint32_t>(value & ((1u << bits) - 1u));
}

__global__ void embed_rows_kernel(const uint32_t* words, const void* scale, const void* bias, int scale_kind,
                                  const int* ids, int n, int bits, int group, int k, void* out, int out_kind) {
    long long i = gid();
    if (i >= static_cast<long long>(n) * k) return;
    int r = static_cast<int>(i / k), c = static_cast<int>(i % k);
    long long id = ids[r];
    int words_row = k * bits / 32, groups = k / group;
    float code = static_cast<float>(code_at(words + id * words_row, words_row, bits, c));
    float s = load(scale, scale_kind, id * groups + c / group);
    float b = load(bias, scale_kind, id * groups + c / group);
    float y = code * s;
    y = y + b;
    store(out, out_kind, i, y);
}

__global__ void cast_kernel(const void* src, int skind, void* dst, int dkind, long long n) {
    long long i = gid();
    if (i < n) store(dst, dkind, i, load(src, skind, i));
}

__global__ void silu_mul_kernel(const void* gate, const void* up, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    float s = rounded(silu(load(gate, kind, i)), kind);
    store(out, kind, i, s * load(up, kind, i));
}

__global__ void add_kernel(const void* x, const void* y, void* out, int kind, long long n) {
    long long i = gid();
    if (i < n) store(out, kind, i, load(x, kind, i) + load(y, kind, i));
}

// out[r, h * d + j] = att * sigmoid(gate[r, h * 2d + d + j]) rounded to kind (gated attention's o input); att is
// (heads, len, d) from the prefill tile, or (len, heads, d) with rows_major from a window's queries.
__global__ void attn_gate_kernel(const float* att, const void* qg, void* out, int kind, int len, int heads, int d,
                                 int rows_major) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * heads));
    float a = rows_major ? att[i] : att[(static_cast<long long>(h) * len + r) * d + j];
    float g = load(qg, kind, (static_cast<long long>(r) * heads + h) * 2 * d + d + j);
    store(out, kind, i, a * sigmoid(g));
}

// out = y * silu(z) widened, rounded to kind: the linear-attention output before its projection.
__global__ void gnorm_silu_kernel(const float* y, const void* z, void* out, int kind, long long n) {
    long long i = gid();
    if (i >= n) return;
    float s = rounded(silu(load(z, kind, i)), kind);
    store(out, kind, i, y[i] * s);
}

// rows of `cols` values from a source with row stride `stride` at column `offset`, kept in kind.
__global__ void copy_cols_kernel(const void* src, long long stride, int offset, void* dst, int kind, int rows,
                                 int cols) {
    long long i = gid();
    if (i >= static_cast<long long>(rows) * cols) return;
    long long r = i / cols;
    int c = static_cast<int>(i % cols);
    long long from = r * stride + offset + c;
    if (kind == 0) static_cast<float*>(dst)[i] = static_cast<const float*>(src)[from];
    else static_cast<uint16_t*>(dst)[i] = static_cast<const uint16_t*>(src)[from];
}

// apply_rope's prefill formula on x (len, heads, d) of kind: freq = 1 / theta^(i / half) as torch makes it, then the
// two products and their sum per half; rounded to kind and stored as out_kind at out[h * s_head + r * s_row + j].
__global__ void rope_prefill_kernel(const void* x, int kind, void* out, int out_kind, long long s_head, long long s_row,
                                    int len, int heads, int d, int rotary, int pos0, float theta) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * heads));
    long long base = (static_cast<long long>(r) * heads + h) * d;
    int half = rotary / 2;
    float y;
    if (j >= rotary) {
        y = load(x, kind, base + j);
    } else {
        int f = j < half ? j : j - half;
        float expo = static_cast<float>(f) / static_cast<float>(half);
        float freq = 1.0f / powf(theta, expo);
        float ang = static_cast<float>(pos0 + r) * freq;
        float c = cosf(ang), s = sinf(ang);
        float x1 = load(x, kind, base + f), x2 = load(x, kind, base + f + half);
        y = j < half ? x1 * c - x2 * s : x1 * s + x2 * c;
    }
    store(out, out_kind, h * s_head + r * s_row + j, rounded(y, kind));
}

// keys or values (len, kv_heads, d) of kind into the cache (kv_heads, total, d) at slot pos0.
__global__ void kv_write_kernel(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total,
                                int pos0) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % kv_heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    long long to = (static_cast<long long>(h) * total + pos0 + r) * d + j;
    if (kind == 0) static_cast<float*>(cache)[to] = static_cast<const float*>(src)[i];
    else static_cast<uint16_t*>(cache)[to] = static_cast<const uint16_t*>(src)[i];
}

// kv_write with the first slot read from the device (`pos`, one int32), so a captured graph replays at any position.
__global__ void kv_write_at_kernel(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total,
                                   const int* pos) {
    long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    int j = static_cast<int>(i % d);
    int h = static_cast<int>((i / d) % kv_heads);
    int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    long long to = (static_cast<long long>(h) * total + *pos + r) * d + j;
    if (kind == 0) static_cast<float*>(cache)[to] = static_cast<const float*>(src)[i];
    else static_cast<uint16_t*>(cache)[to] = static_cast<const uint16_t*>(src)[i];
}

// causal_conv's prefill loop: window = [state | x], out = sum over taps in order of window * weight from a zero,
// then silu; new_state = the window's last kernel - 1 rows. state null: zeros.
__global__ void conv_prefill_kernel(const void* x, int kind, const float* weight, const float* state, float* out,
                                    float* new_state, int len, int channels, int kernel) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    int kept = kernel - 1;
    auto window = [&](int t) -> float {
        if (t < kept) return state ? state[static_cast<long long>(t) * channels + c] : 0.0f;
        return load(x, kind, static_cast<long long>(t - kept) * channels + c);
    };
    for (int r = 0; r < len; ++r) {
        float acc = 0.0f;
        for (int tap = 0; tap < kernel; ++tap) {
            float p = window(r + tap) * weight[static_cast<long long>(c) * kernel + tap];
            acc = acc + p;
        }
        out[static_cast<long long>(r) * channels + c] = silu(acc);
    }
    for (int t = 0; t < kept; ++t) new_state[static_cast<long long>(t) * channels + c] = window(len + t);
}

// _gate_beta: beta = sigmoid(b), gate = exp(-exp(a_log) * softplus(a + dt_bias)) with torch's softplus (beta 1,
// threshold 20) over rows of `heads`.
__global__ void gdn_gate_prefill_kernel(const void* a, const void* b, int kind, const float* a_log,
                                        const float* dt_bias, float* gate, float* beta, int count, int heads) {
    long long i = gid();
    if (i >= static_cast<long long>(count) * heads) return;
    int h = static_cast<int>(i % heads);
    beta[i] = sigmoid(load(b, kind, i));
    float x = load(a, kind, i) + dt_bias[h];
    float xb = x * 1.0f;
    float sp = xb > 20.0f ? x : log1pf(expf(xb)) / 1.0f;
    float e = -expf(a_log[h]);
    gate[i] = expf(e * sp);
}

// experts.route: members = the pairs sorted by expert (stable), items = (expert, first, count) of at most `tile`
// pairs, one slot of `capacity` each; one block, the counts in shared memory.
__global__ void moe_route_kernel(const int* picks, int pairs, int experts, int tile, int* members, int* items,
                                 int capacity) {
    extern __shared__ int shared[];
    int* counts = shared;
    int* starts = shared + experts;
    int* ends = shared + 2 * experts;
    for (int e = threadIdx.x; e < experts; e += blockDim.x) counts[e] = 0;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int p = 0; p < pairs; ++p) counts[picks[p]] += 1;
        int start = 0, end = 0;
        for (int e = 0; e < experts; ++e) {
            starts[e] = start;
            start += counts[e];
            end += (counts[e] + tile - 1) / tile;
            ends[e] = end;
        }
    }
    __syncthreads();
    // Stable order: a pair's place is its expert's start plus the earlier pairs of that expert.
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        int e = picks[p], before = 0;
        for (int q = 0; q < p; ++q) before += picks[q] == e;
        members[starts[e] + before] = p;
    }
    for (int slot = threadIdx.x; slot < capacity; slot += blockDim.x) {
        int lo = 0, hi = experts;
        while (lo < hi) {                       // searchsorted(ends, slot, right=True)
            int mid = (lo + hi) / 2;
            if (ends[mid] <= slot) lo = mid + 1;
            else hi = mid;
        }
        int e = lo < experts - 1 ? lo : experts - 1;
        int reps = (counts[e] + tile - 1) / tile;
        int within = slot - (ends[e] - reps);
        int count = counts[e] - within * tile;
        items[slot * 3] = e;
        items[slot * 3 + 1] = starts[e] + within * tile;
        items[slot * 3 + 2] = count < 0 ? 0 : (count > tile ? tile : count);
    }
}

thread_local char op_error[256] = "";

int finish() {
    hipError_t err = hipGetLastError();
    if (err == hipSuccess) return 0;
    std::strncpy(op_error, hipGetErrorString(err), sizeof(op_error) - 1);
    return 1;
}

unsigned blocks(long long n, int threads) { return static_cast<unsigned>((n + threads - 1) / threads); }

}  // namespace

extern "C" {

const char* tf_op_error() { return op_error; }

int tf_embed_rows(const void* words, const void* scale, const void* bias, int scale_kind, const int* ids, int n,
                  int bits, int group, int k, void* out, int out_kind, hipStream_t s) {
    long long total = static_cast<long long>(n) * k;
    if (total == 0) return 0;
    embed_rows_kernel<<<blocks(total, 256), 256, 0, s>>>(static_cast<const uint32_t*>(words), scale, bias,
                                                         scale_kind, ids, n, bits, group, k, out, out_kind);
    return finish();
}

int tf_cast(const void* src, int skind, void* dst, int dkind, long long n, hipStream_t s) {
    if (n == 0) return 0;
    cast_kernel<<<blocks(n, 256), 256, 0, s>>>(src, skind, dst, dkind, n);
    return finish();
}

int tf_silu_mul(const void* gate, const void* up, void* out, int kind, long long n, hipStream_t s) {
    if (n == 0) return 0;
    silu_mul_kernel<<<blocks(n, 256), 256, 0, s>>>(gate, up, out, kind, n);
    return finish();
}

int tf_add(const void* x, const void* y, void* out, int kind, long long n, hipStream_t s) {
    if (n == 0) return 0;
    add_kernel<<<blocks(n, 256), 256, 0, s>>>(x, y, out, kind, n);
    return finish();
}

int tf_attn_gate(const float* att, const void* qg, void* out, int kind, int len, int heads, int d, int rows_major,
                 hipStream_t s) {
    long long n = static_cast<long long>(len) * heads * d;
    if (n == 0) return 0;
    attn_gate_kernel<<<blocks(n, 256), 256, 0, s>>>(att, qg, out, kind, len, heads, d, rows_major);
    return finish();
}

int tf_gnorm_silu(const float* y, const void* z, void* out, int kind, long long n, hipStream_t s) {
    if (n == 0) return 0;
    gnorm_silu_kernel<<<blocks(n, 256), 256, 0, s>>>(y, z, out, kind, n);
    return finish();
}

int tf_copy_cols(const void* src, long long stride, int offset, void* dst, int kind, int rows, int cols,
                 hipStream_t s) {
    long long n = static_cast<long long>(rows) * cols;
    if (n == 0) return 0;
    copy_cols_kernel<<<blocks(n, 256), 256, 0, s>>>(src, stride, offset, dst, kind, rows, cols);
    return finish();
}

int tf_rope_prefill(const void* x, int kind, void* out, int out_kind, long long s_head, long long s_row, int len,
                    int heads, int d, int rotary, int pos0, float theta, hipStream_t s) {
    long long n = static_cast<long long>(len) * heads * d;
    if (n == 0) return 0;
    rope_prefill_kernel<<<blocks(n, 256), 256, 0, s>>>(x, kind, out, out_kind, s_head, s_row, len, heads, d, rotary,
                                                       pos0, theta);
    return finish();
}

int tf_kv_write(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total, int pos0,
                hipStream_t s) {
    long long n = static_cast<long long>(len) * kv_heads * d;
    if (n == 0) return 0;
    kv_write_kernel<<<blocks(n, 256), 256, 0, s>>>(src, cache, kind, len, kv_heads, d, total, pos0);
    return finish();
}

int tf_kv_write_at(const void* src, void* cache, int kind, int len, int kv_heads, int d, int total, const int* pos,
                   hipStream_t s) {
    long long n = static_cast<long long>(len) * kv_heads * d;
    if (n == 0) return 0;
    kv_write_at_kernel<<<blocks(n, 256), 256, 0, s>>>(src, cache, kind, len, kv_heads, d, total, pos);
    return finish();
}

int tf_conv_prefill(const void* x, int kind, const float* weight, const float* state, float* out, float* new_state,
                    int len, int channels, int kernel, hipStream_t s) {
    conv_prefill_kernel<<<blocks(channels, 128), 128, 0, s>>>(x, kind, weight, state, out, new_state, len, channels,
                                                              kernel);
    return finish();
}

int tf_gdn_gate_prefill(const void* a, const void* b, int kind, const float* a_log, const float* dt_bias, float* gate,
                        float* beta, int count, int heads, hipStream_t s) {
    long long n = static_cast<long long>(count) * heads;
    if (n == 0) return 0;
    gdn_gate_prefill_kernel<<<blocks(n, 256), 256, 0, s>>>(a, b, kind, a_log, dt_bias, gate, beta, count, heads);
    return finish();
}

int tf_moe_route(const int* picks, int pairs, int experts, int tile, int* members, int* items, int capacity,
                 hipStream_t s) {
    moe_route_kernel<<<1, 256, 3 * experts * sizeof(int), s>>>(picks, pairs, experts, tile, members, items, capacity);
    return finish();
}

}  // extern "C"
