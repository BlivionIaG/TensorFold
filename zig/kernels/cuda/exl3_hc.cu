// Hyper-connection steps and the router dots for Qwen3.8-Flash-Next on the EXL3 path (hc_count 4, hidden 2560, lowrank 320).
// Maths: families/flashnext/exl3_hc.zig (host f32 `pre` / `post` / `expand`), itself from qwen4_exp model_layers.py
// HyperConnection. h is stream-major f32 [token][4][2560]; the three hyper-connection matrices and the router gates are the
// checkpoint's F16 tensors (out-major, torch nn.Linear layout); norm weights are F16 and the centring offset is added in f32.
//
// pre = exl3_hc_norm -> exl3_hc_low (down matrix + the 4 block-inject rows) -> exl3_hc_mix -> exl3_hc_out
// post = exl3_hc_post, expand = exl3_hc_expand_bf16. The final collapse is `pre` with the mixer weights and no inject rows.
// Reductions are single-thread, index order: the same f32 sums as the host (see the note above exl3_hc_norm).
//
// Compiled with --fmad=false, IEEE division and square root (the glue flags).

#include <cuda_fp16.h>
#include <stdint.h>

namespace {

constexpr int S = 4;               // streams
constexpr int D = 2560;            // hidden
constexpr int W = S * D;           // 10240
constexpr int LOW = 320;           // hc_lowrank
constexpr int INJ = 4;             // block_inject rows
constexpr int GATE_ROWS = 513;     // 512 routed gate rows + the shared-expert gate row

__device__ __forceinline__ float bf16_to_f32(uint16_t u) { return __uint_as_float(static_cast<uint32_t>(u) << 16); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int m = 16; m > 0; m >>= 1) v += __shfl_xor_sync(0xffffffffu, v, m);
    return v;
}

__device__ __forceinline__ float sigmoid_f(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

// h[t][s][i] = embedding row t (bf16) for every stream s. Grid (W / 256, tokens), block 256.
extern "C" __global__ void __launch_bounds__(256) exl3_hc_expand_bf16(const uint16_t* __restrict__ emb, float* __restrict__ h,
                                                                      int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int c = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (t >= tokens || c >= W) return;
    h[static_cast<size_t>(t) * W + c] = bf16_to_f32(emb[static_cast<size_t>(t) * D + (c % D)]);
}

// Every reduction below is ONE thread walking its terms in index order with separate multiply and add (--fmad=false), which is
// the host's order in exl3_hc.zig `pre` and exl3_router.zig `dot`: the f32 sums are then the same numbers, the router sees the
// same logits and picks the same experts as the host path (a different summation order flips near-tied top-10 picks and moves
// the logits by percent). Parallelism comes from the number of rows, not from splitting a row: 324 + 10240 + 513 rows a token.

// normed[t][s][i] = h / rms * (w + offset), rms = sqrt(sum_i h^2 / 2560 + eps) with the sum in index order.
// Grid (S, tokens), block 256: thread 0 sums the row, all threads scale it.
extern "C" __global__ void __launch_bounds__(256) exl3_hc_norm(const float* __restrict__ h, const __half* __restrict__ w,
                                                               float offset, float eps, float* __restrict__ normed, int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int s = static_cast<int>(blockIdx.x);
    if (t >= tokens) return;
    const float* row = h + static_cast<size_t>(t) * W + s * D;
    __shared__ float rms;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < D; ++i) {
            const float v = row[i];
            ss += v * v;
        }
        rms = sqrtf(ss / static_cast<float>(D) + eps);
    }
    __syncthreads();
    float* out = normed + static_cast<size_t>(t) * W + s * D;
    for (int i = threadIdx.x; i < D; i += 256) out[i] = row[i] / rms * (__half2float(w[s * D + i]) + offset);
}

// Rows 0..319: low[t][r] = silu((down[r] . normed) / 4). Rows 320..323 (when `inj` is given): inject[t][s] = 2 sigmoid((inj[s] . normed) / 4).
// Grid ((LOW + INJ + 127) / 128, tokens), block 128: a thread is one row.
extern "C" __global__ void __launch_bounds__(128) exl3_hc_low(const float* __restrict__ normed, const __half* __restrict__ down,
                                                              const __half* __restrict__ inj, float* __restrict__ low,
                                                              float* __restrict__ inject, int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.x) * 128 + static_cast<int>(threadIdx.x);
    if (t >= tokens || row >= LOW + INJ) return;
    const bool is_inject = row >= LOW;
    if (is_inject && inj == nullptr) return;
    const __half* w = is_inject ? inj + static_cast<size_t>(row - LOW) * W : down + static_cast<size_t>(row) * W;
    const float* x = normed + static_cast<size_t>(t) * W;
    float acc = 0.0f;
    for (int i = 0; i < W; ++i) acc += __half2float(w[i]) * x[i];
    const float z = acc * 0.25f;
    if (is_inject) {
        inject[static_cast<size_t>(t) * INJ + (row - LOW)] = 2.0f * sigmoid_f(z);
    } else {
        low[static_cast<size_t>(t) * LOW + row] = z / (1.0f + expf(-z));
    }
}

// mix[t][j] = sigmoid(up[j] . low). Grid (W / 128, tokens), block 128: a thread is one output.
extern "C" __global__ void __launch_bounds__(128) exl3_hc_mix(const float* __restrict__ low, const __half* __restrict__ up,
                                                              float* __restrict__ mix, int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int j = static_cast<int>(blockIdx.x) * 128 + static_cast<int>(threadIdx.x);
    if (t >= tokens || j >= W) return;
    const __half* w = up + static_cast<size_t>(j) * LOW;
    const float* x = low + static_cast<size_t>(t) * LOW;
    float acc = 0.0f;
    for (int i = 0; i < LOW; ++i) acc += __half2float(w[i]) * x[i];
    mix[static_cast<size_t>(t) * W + j] = sigmoid_f(acc);
}

// mixed[t][i] = (sum_s mix[s][i] * normed[s][i]) / 4, rounded to fp16: the block input every projection reads.
// Grid (D / 256, tokens), block 256.
extern "C" __global__ void __launch_bounds__(256) exl3_hc_out(const float* __restrict__ mix, const float* __restrict__ normed,
                                                              __half* __restrict__ x, int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int i = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (t >= tokens || i >= D) return;
    const size_t base = static_cast<size_t>(t) * W;
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < S; ++s) acc += mix[base + s * D + i] * normed[base + s * D + i];
    x[static_cast<size_t>(t) * D + i] = __float2half_rn(acc * 0.25f);
}

// h[t][s][i] += branch[t][i] * inject[t][s]; branch is the fp16 block output. Grid (W / 256, tokens), block 256.
extern "C" __global__ void __launch_bounds__(256) exl3_hc_post(float* __restrict__ h, const __half* __restrict__ branch,
                                                               const float* __restrict__ inject, int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int c = static_cast<int>(blockIdx.x) * 256 + static_cast<int>(threadIdx.x);
    if (t >= tokens || c >= W) return;
    const int s = c / D;
    const int i = c - s * D;
    const float b = __half2float(branch[static_cast<size_t>(t) * D + i]);
    h[static_cast<size_t>(t) * W + c] += b * inject[static_cast<size_t>(t) * INJ + s];
}

// logits[t][r] = gate[r] . x[t] for the 512 routed rows, r = 512: shared_expert_gate . x[t]. fp32 sum in index order over the F16 weights.
// Grid ((513 + 127) / 128, tokens), block 128: a thread is one row.
extern "C" __global__ void __launch_bounds__(128) exl3_hc_gate(const __half* __restrict__ x, const __half* __restrict__ gate,
                                                               const __half* __restrict__ sgate, float* __restrict__ logits,
                                                               int tokens) {
    const int t = static_cast<int>(blockIdx.y);
    const int row = static_cast<int>(blockIdx.x) * 128 + static_cast<int>(threadIdx.x);
    if (t >= tokens || row >= GATE_ROWS) return;
    const __half* w = row < 512 ? gate + static_cast<size_t>(row) * D : sgate;
    const __half* xr = x + static_cast<size_t>(t) * D;
    float acc = 0.0f;
    for (int i = 0; i < D; ++i) acc += __half2float(w[i]) * __half2float(xr[i]);
    logits[static_cast<size_t>(t) * GATE_ROWS + row] = acc;
}
