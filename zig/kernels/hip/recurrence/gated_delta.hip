// Gated DeltaNet: a wave per (batch, value head, value row) walks the sequence, lane sums as the Python reference.

#include "decode/plan.hpp"

#include <hip/hip_runtime.h>

namespace tf {
namespace rocm {

__device__ float lane0_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int DK>
__global__ void __launch_bounds__(32)
gated_delta_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                   const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                   float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                   float* __restrict__ states) {
    constexpr int SPAN = (DK + 31) / 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (row >= dv) return;
    const int key_head = head / (value_heads / key_heads);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row =
        ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t = 0; t < length; ++t) {
        const size_t step = (static_cast<size_t>(batch) * length + t) * value_heads + head;
        const float decay = gate[step];
        const float step_beta = beta[step];
        const float value = v[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
                              static_cast<size_t>(head) * dv + row];
        const size_t key =
            ((static_cast<size_t>(batch) * length + t) * key_heads + key_head) * DK;

        float remembered = 0.f;
        float ks[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) {
                ks[i] = k[key + idx];
                s[i] = s[i] * decay;
                remembered = remembered + s[i] * ks[i];
            }
        }
        const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) s[i] = s[i] + ks[i] * delta;
        }
        if (states != nullptr) {
            const size_t snap = (((static_cast<size_t>(batch) * length + t) * value_heads + head) * dv + row) * DK;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) states[snap + idx] = s[i];
            }
        }

        float readout = 0.f;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) readout = readout + s[i] * q[key + idx];
        }
        // Every lane has to enter the shuffle. Lane 0 owns the stored sum.
        readout = lane0_sum(readout);
        if (lane == 0) {
            y[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
              static_cast<size_t>(head) * dv + row] = readout;
        }
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

// Value rows share a staged tile of k and q, in the wave kernel's order (-ffp-contract=off keeps it).
template <int DK, int ROWS, int STEPS>
__device__ __forceinline__ void
gated_delta_tile_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                      const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                      float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                      float* __restrict__ states, int batch, bool write_back = true) {
    constexpr int SPAN = (DK + 31) / 32;
    constexpr int WIDTH = 32 * ROWS;
    const int lane = threadIdx.x;
    const int local = threadIdx.y;
    const int row = blockIdx.x * ROWS + local;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const int tid = local * 32 + lane;

    __shared__ float ks[STEPS][DK];
    __shared__ float qs[STEPS][DK];
    __shared__ float gs[STEPS];
    __shared__ float bs[STEPS];
    __shared__ float vs[STEPS][ROWS];

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row = ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t0 = 0; t0 < length; t0 += STEPS) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
        for (int i = tid; i < nvec; i += WIDTH) {
            const int elem = i * 4;
            const int step = elem / DK;
            const int col = elem - step * DK;
            const size_t at =
                ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + col;
            const float4 key = *reinterpret_cast<const float4*>(k + at);
            ks[step][col] = key.x;
            ks[step][col + 1] = key.y;
            ks[step][col + 2] = key.z;
            ks[step][col + 3] = key.w;
            // a replay (no q) advances the state alone
            if (q == nullptr) continue;
            const float4 query = *reinterpret_cast<const float4*>(q + at);
            qs[step][col] = query.x;
            qs[step][col + 1] = query.y;
            qs[step][col + 2] = query.z;
            qs[step][col + 3] = query.w;
        }
        for (int i = tid; i < n; i += WIDTH) {
            const size_t step = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[step];
            bs[i] = beta[step];
        }
        for (int i = tid; i < n * ROWS; i += WIDTH) {
            const int step = i / ROWS;
            const int which = i - step * ROWS;
            vs[step][which] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * dv +
                                blockIdx.x * ROWS + which];
        }
        __syncthreads();

        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt];
            const float step_beta = bs[tt];
            const float value = vs[tt][local];
            float remembered = 0.f;
            float held[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) {
                    held[i] = ks[tt][idx];
                    s[i] = s[i] * decay;
                    remembered = remembered + s[i] * held[i];
                }
            }
            const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) s[i] = s[i] + held[i] * delta;
            }
            if (states != nullptr) {
                const size_t snap =
                    (((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row) * DK;
#pragma unroll
                for (int i = 0; i < SPAN; ++i) {
                    const int idx = lane + i * 32;
                    if (idx < DK) states[snap + idx] = s[i];
                }
            }
            if (q == nullptr) continue;
            float readout = 0.f;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) readout = readout + s[i] * qs[tt][idx];
            }
            readout = lane0_sum(readout);
            if (lane == 0) {
                y[((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * dv + row] = readout;
            }
        }
        __syncthreads();
    }

    if (!write_back) return;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

template <int DK, int ROWS, int STEPS>
__global__ void __launch_bounds__(32 * ROWS)
gated_delta_tile(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                 const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                 float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
                 float* __restrict__ states) {
    gated_delta_tile_body<DK, ROWS, STEPS>(q, k, v, gate, beta, state, y, length, key_heads, value_heads, dv, states,
                                           blockIdx.z);
}

// The plan kernel's window of steps staged in LDS. A decode round is one row: 16 steps stage 16.6 KB, three blocks a CU on gfx900's 64 KiB
// LDS, so the grid's 512 live blocks ran in three rounds; four steps stage 4.2 KB and one round holds them. The scan is the same.
#if defined(__gfx900__)
constexpr int kPlanSteps = 4;
#else
constexpr int kPlanSteps = 16;
#endif

#if defined(__gfx900__)
// gfx900's plan scan: gated_delta_tile_body<DK, 8, kPlanSteps> with its arithmetic and its LDS tile unchanged, and the loads reordered. The tile
// body's staging loops and its conditional state loads each wait for their load before the next one goes out (a decode step is about a dozen
// dependent round trips to memory); here the slot's count, first row and descriptor address go out together, then the state pointer and the window's
// k, q, gate, beta and v, then the state, and only the stores to LDS wait.
template <int DK>
__device__ __forceinline__ void
gated_delta_plan_scan(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                      int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    constexpr int ROWS = 8;
    constexpr int STEPS = kPlanSteps;
    constexpr int SPAN = (DK + 31) / 32;
    constexpr int WIDTH = 32 * ROWS;
    constexpr int ITER = (STEPS * (DK / 4) + WIDTH - 1) / WIDTH;
    const int slot = blockIdx.z;
    const int length = plan.count[slot];
    const long long row0 = plan.first[slot];
    const unsigned long long desc_at = plan.desc[slot];
    // the loads above stay above the return (the padding slot's descriptor is only an address: not read)
    asm volatile("" ::: "memory");
    if (length == 0) return;
    const unsigned long long* desc = reinterpret_cast<const unsigned long long*>(desc_at);
    float* state = reinterpret_cast<float*>(desc[plan_second(layer)]);
    const bool replayed = length > 1 && slot + 1 < static_cast<int>(gridDim.z);
    const int lane = threadIdx.x & 31;
    const int local = threadIdx.y;
    const int row = blockIdx.x * ROWS + local;
    const int head = blockIdx.y;
    const int key_head = head / (value_heads / key_heads);
    const int tid = local * 32 + lane;
    const float* qr = q + row0 * key_heads * DK;
    const float* kr = k + row0 * key_heads * DK;
    const float* vr = v + row0 * value_heads * dv;
    const float* gr = gate + row0 * value_heads;
    const float* br = beta + row0 * value_heads;
    float* yr = y + row0 * value_heads * dv;

    __shared__ float ks[STEPS][DK];
    __shared__ float qs[STEPS][DK];
    __shared__ float gs[STEPS];
    __shared__ float bs[STEPS];
    __shared__ float vs[STEPS][ROWS];

    // a window's inputs in registers: the indices clamp to the window's last, the stores below are the ones that are guarded
    float4 kw[ITER], qw[ITER];
    float gw, bw, vw;
    auto fetch = [&](int t0) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
#pragma unroll
        for (int j = 0; j < ITER; ++j) {
            const int i = min(tid + j * WIDTH, nvec - 1);
            const int elem = i * 4;
            const int step = elem / DK;
            const int col = elem - step * DK;
            const size_t at = (static_cast<size_t>(t0 + step) * key_heads + key_head) * DK + col;
            kw[j] = *reinterpret_cast<const float4*>(kr + at);
            qw[j] = *reinterpret_cast<const float4*>(qr + at);
        }
        const int g = min(tid, n - 1);
        gw = gr[static_cast<size_t>(t0 + g) * value_heads + head];
        bw = br[static_cast<size_t>(t0 + g) * value_heads + head];
        const int vi = min(tid, n * ROWS - 1);
        const int vstep = vi / ROWS;
        vw = vr[(static_cast<size_t>(t0 + vstep) * value_heads + head) * dv + blockIdx.x * ROWS + (vi - vstep * ROWS)];
    };
    fetch(0);
    // the window's loads go out before the state loads that wait for the descriptor's
    asm volatile("" ::: "memory");

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row = (static_cast<size_t>(head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t0 = 0; t0 < length; t0 += STEPS) {
        const int n = min(STEPS, length - t0);
        const int nvec = n * (DK / 4);
        if (t0 > 0) fetch(t0);
#pragma unroll
        for (int j = 0; j < ITER; ++j) {
            const int i = tid + j * WIDTH;
            if (i < nvec) {
                const int elem = i * 4;
                const int step = elem / DK;
                const int col = elem - step * DK;
                ks[step][col] = kw[j].x;
                ks[step][col + 1] = kw[j].y;
                ks[step][col + 2] = kw[j].z;
                ks[step][col + 3] = kw[j].w;
                qs[step][col] = qw[j].x;
                qs[step][col + 1] = qw[j].y;
                qs[step][col + 2] = qw[j].z;
                qs[step][col + 3] = qw[j].w;
            }
        }
        if (tid < n) {
            gs[tid] = gw;
            bs[tid] = bw;
        }
        if (tid < n * ROWS) vs[tid / ROWS][tid % ROWS] = vw;
        __syncthreads();

        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt];
            const float step_beta = bs[tt];
            const float value = vs[tt][local];
            float remembered = 0.f;
            float held[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) {
                    held[i] = ks[tt][idx];
                    s[i] = s[i] * decay;
                    remembered = remembered + s[i] * held[i];
                }
            }
            const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) s[i] = s[i] + held[i] * delta;
            }
            float readout = 0.f;
#pragma unroll
            for (int i = 0; i < SPAN; ++i) {
                const int idx = lane + i * 32;
                if (idx < DK) readout = readout + s[i] * qs[tt][idx];
            }
            readout = lane0_sum(readout);
            if (lane == 0) yr[static_cast<size_t>(t0 + tt) * value_heads * dv + static_cast<size_t>(head) * dv + row] = readout;
        }
        __syncthreads();
    }

    if (replayed) return;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}
#endif

// A round's windows in one launch, a grid plane a slot; rows, state and snapshots come from the plan, same scan.
template <int DK>
__device__ __forceinline__ void
gated_delta_plan_body(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                      int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
#if defined(__gfx900__)
    gated_delta_plan_scan<DK>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
#else
    const int slot = blockIdx.z;
    const int length = plan.count[slot];
    if (length == 0) return;
    const long long row0 = plan.first[slot];
    const unsigned long long* desc = plan_desc(plan, slot);
    float* state = reinterpret_cast<float*>(desc[plan_second(layer)]);
    // a window of rows leaves the state as the round found it, for its keep to replay (padding rows keep nothing)
    const bool replayed = length > 1 && slot + 1 < static_cast<int>(gridDim.z);
    gated_delta_tile_body<DK, 8, kPlanSteps>(q + row0 * key_heads * DK, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                             gate + row0 * value_heads, beta + row0 * value_heads, state,
                                             y + row0 * value_heads * dv, length, key_heads, value_heads, dv, nullptr, 0,
                                             !replayed);
#endif
}

// A keep: each slot's state run from the round's start through its kept rows, by the same scan on the inputs kept.
template <int DK>
__device__ __forceinline__ void gated_delta_replay_body(const int* keep, int rows, int key_heads, int value_heads, int dv,
                                                        PlanArgs plan, int slots) {
    const int slot = blockIdx.z % slots;
    const int layer = blockIdx.z / slots;
    const int n = keep[slot];
    if (n <= 0 || plan.count[slot] <= 1 || plan.snaps[2 * layer + 1] == 0) return;
    const long long row0 = plan.first[slot];
    const float* k = reinterpret_cast<const float*>(plan.snaps[2 * layer + 1]);
    const float* v = k + static_cast<long long>(rows) * key_heads * DK;
    const float* gate = v + static_cast<long long>(rows) * value_heads * dv;
    const float* beta = gate + static_cast<long long>(rows) * value_heads;
    float* state = reinterpret_cast<float*>(plan_desc(plan, slot)[plan_second(layer)]);
    gated_delta_tile_body<DK, 8, 16>(nullptr, k + row0 * key_heads * DK, v + row0 * value_heads * dv,
                                     gate + row0 * value_heads, beta + row0 * value_heads, state, nullptr, n, key_heads,
                                     value_heads, dv, nullptr, 0);
}

// A prompt's recurrence for DK = DV = 128, 4 lanes a row and 2 rows a lane; lane0_sum's sums in its order, same bits.
constexpr int kPromptSteps = 16;

__device__ __forceinline__ float quad_sum(const float (&part)[8]) {
    // xor 16, 8, 4 of lane0_sum: virtual lanes j ^ 4, j ^ 2, j ^ 1 in registers (each pair's add commutes)
    const float a0 = part[0] + part[4], a1 = part[1] + part[5], a2 = part[2] + part[6], a3 = part[3] + part[7];
    const float b0 = a0 + a2, b1 = a1 + a3;
    float c = b0 + b1;
    // xor 2 and 1: the other lanes of the row
    c = c + __shfl_xor(c, 2, 32);
    c = c + __shfl_xor(c, 1, 32);
    return c;
}

template <int WAVES>
__device__ __forceinline__ void
gated_delta_prompt_body(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                        const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                        float* __restrict__ y, int length, int key_heads, int value_heads) {
    constexpr int DK = 128, DV = 128, T = kPromptSteps, THREADS = 32 * WAVES, ROWS = 16 * WAVES;
    const int lane = threadIdx.x, wave = threadIdx.y;
    const int tid = wave * 32 + lane;
    const int p = lane & 3;
    const int first = blockIdx.x * ROWS;  // the block's first Dv row
    const int row0 = first + wave * 16 + 2 * (lane >> 2);  // this lane's rows: row0, row0 + 1
    const int head = blockIdx.y, batch = blockIdx.z;
    const int key_head = head / (value_heads / key_heads);

    __shared__ __attribute__((aligned(16))) float ks[T][DK];  // transposed: [t][p * 32 + m] = k[t][p + 4 m]
    __shared__ __attribute__((aligned(16))) float qs[T][DK];
    __shared__ float vs[T][ROWS];
    __shared__ float gs[T];
    __shared__ float bs[T];

    float s[2][32];
    for (int r = 0; r < 2; ++r) {
        const float* at = state + ((static_cast<size_t>(batch) * value_heads + head) * DV + row0 + r) * DK;
#pragma unroll
        for (int m = 0; m < 32; ++m) s[r][m] = at[p + 4 * m];
    }

    for (int t0 = 0; t0 < length; t0 += T) {
        const int n = min(T, length - t0);
        for (int i = tid; i < n * DK; i += THREADS) {
            const int step = i / DK, e = i - step * DK;
            const size_t at = ((static_cast<size_t>(batch) * length + (t0 + step)) * key_heads + key_head) * DK + e;
            const int to = (e & 3) * 32 + (e >> 2);
            ks[step][to] = k[at];
            qs[step][to] = q[at];
        }
        for (int i = tid; i < n * ROWS; i += THREADS) {
            const int step = i / ROWS, e = i - step * ROWS;
            vs[step][e] = v[((static_cast<size_t>(batch) * length + (t0 + step)) * value_heads + head) * DV + first + e];
        }
        for (int i = tid; i < n; i += THREADS) {
            const size_t at = (static_cast<size_t>(batch) * length + (t0 + i)) * value_heads + head;
            gs[i] = gate[at];
            bs[i] = beta[at];
        }
        __syncthreads();
        for (int tt = 0; tt < n; ++tt) {
            const float decay = gs[tt], step_beta = bs[tt];
            float kv[32], qv[32];
            const float4* k4 = reinterpret_cast<const float4*>(&ks[tt][p * 32]);
            const float4* q4 = reinterpret_cast<const float4*>(&qs[tt][p * 32]);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                const float4 a = k4[c], b = q4[c];
                kv[4 * c] = a.x, kv[4 * c + 1] = a.y, kv[4 * c + 2] = a.z, kv[4 * c + 3] = a.w;
                qv[4 * c] = b.x, qv[4 * c + 1] = b.y, qv[4 * c + 2] = b.z, qv[4 * c + 3] = b.w;
            }
            float out[2];
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float part[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) part[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = j + 8 * i;
                        s[r][m] = s[r][m] * decay;
                        part[j] = part[j] + s[r][m] * kv[m];
                    }
                }
                const float delta = (vs[tt][row0 - first + r] - quad_sum(part)) * step_beta;
#pragma unroll
                for (int j = 0; j < 8; ++j) part[j] = 0.f;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const int m = j + 8 * i;
                        s[r][m] = s[r][m] + kv[m] * delta;
                        part[j] = part[j] + s[r][m] * qv[m];
                    }
                }
                out[r] = quad_sum(part);
            }
            if (p == 0) {
                float* yo = y + ((static_cast<size_t>(batch) * length + (t0 + tt)) * value_heads + head) * DV + row0;
                yo[0] = out[0];
                yo[1] = out[1];
            }
        }
        __syncthreads();
    }
    for (int r = 0; r < 2; ++r) {
        float* at = state + ((static_cast<size_t>(batch) * value_heads + head) * DV + row0 + r) * DK;
#pragma unroll
        for (int m = 0; m < 32; ++m) at[p + 4 * m] = s[r][m];
    }
}

}  // namespace rocm
}  // namespace tf

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_128(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
                int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<128>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_128(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<128>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_replay_16(const int* keep, int rows, int key_heads, int value_heads, int dv, PlanArgs plan, int slots) {
    tf::rocm::gated_delta_replay_body<16>(keep, rows, key_heads, value_heads, dv, plan, slots);
}

extern "C" __global__ void __launch_bounds__(256)
tf_gdn_plan_16(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* y,
               int key_heads, int value_heads, int dv, PlanArgs plan, int layer) {
    tf::rocm::gated_delta_plan_body<16>(q, k, v, gate, beta, y, key_heads, value_heads, dv, plan, layer);
}

// The template kernels launches.zig resolves.
template __global__ void tf::rocm::gated_delta_tile<128, 8, 16>(const float* __restrict__ q,
    const float* __restrict__ k, const float* __restrict__ v, const float* __restrict__ gate,
    const float* __restrict__ beta, float* __restrict__ state, float* __restrict__ y, int length, int key_heads,
    int value_heads, int dv, float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_tile<16, 8, 16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<128>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);
template __global__ void tf::rocm::gated_delta_kernel<16>(const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ gate, const float* __restrict__ beta,
    float* __restrict__ state, float* __restrict__ y, int length, int key_heads, int value_heads, int dv,
    float* __restrict__ states);

// Half a value head a block (4 waves, 64 rows; grid 2 x value heads), the fastest of the three block sizes on gfx1151.
extern "C" __global__ void __launch_bounds__(128)
tf_gdn_prompt_128(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* state,
                  float* y, int length, int key_heads, int value_heads) {
    tf::rocm::gated_delta_prompt_body<4>(q, k, v, gate, beta, state, y, length, key_heads, value_heads);
}
