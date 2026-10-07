// A lane round's attention walk, cache writes and keep, read from the round's plan on the device (plan.hpp).

#include "ops/common.hpp"
#include "decode/plan.hpp"

namespace {

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

// Row r's keys (already rotated, fp32) and values (the activation kind) into its slot's pages at its position.
__device__ void plan_kv_write_body(const float* keys, const void* values, int kind, int rows, int kv_heads, int d,
                                     PlanArgs plan, int layer) {
    const long long i = gid();
    if (i >= static_cast<long long>(rows) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    const unsigned long long* desc = plan_desc(plan, plan.slot[r]);
    const long long to = page_at(plan_pages(desc), kv_heads, h, plan.pos[r], d) + j;
    store(reinterpret_cast<void*>(desc[plan_first(layer)]), kind, to, keys[i]);
    static_cast<uint16_t*>(reinterpret_cast<void*>(desc[plan_second(layer)]))[to] = static_cast<const uint16_t*>(values)[i];
}

// attention.hip's score_keys over a plan's rows: a warp scores a key of the row's slot, at most its position visible.
template <int KIND>
__device__ void plan_score_body(const float* __restrict__ q, float* __restrict__ scores, int span, int heads,
                                  int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    const int batch = blockIdx.z;
    const int visible = min(plan.pos[batch] + 1, span);
    const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
    const void* keys = reinterpret_cast<const void*>(desc[plan_first(layer)]);
    const unsigned* table = plan_pages(desc);
    const int lane = threadIdx.x & 31;
    const int key = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int head = blockIdx.y;
    const float* qrow = q + (static_cast<long long>(batch) * heads + head) * d;
    float qreg[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int idx = lane + i * 32;
        qreg[i] = idx < d ? qrow[idx] : 0.f;
    }
    float partial = 0.f;
    if (key < visible) {
        const int kv = head / (heads / kv_heads);
        const long long base = page_at(table, kv_heads, kv, key, d);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int idx = lane + i * 32;
            if (idx < d) partial += qreg[i] * load(keys, KIND, base + idx);
        }
    }
    const float dot = wave_sum(partial);
    if (lane == 0 && key < span) {
        const float score = key < visible ? dot * scale : -INFINITY;
        scores[(static_cast<long long>(batch) * heads + head) * span + key] = score;
    }
}

// attention.hip's apply_values over a plan's rows: a block a 128-key tile, the tile's weighted values of the row's slot.
template <int KIND>
__device__ void plan_apply_body(const float* __restrict__ scores, const float* __restrict__ stats,
                                  float* __restrict__ partials, int span, int heads, int kv_heads, int d,
                                  PlanArgs plan, int layer) {
    const int batch = blockIdx.z;
    const int visible = min(plan.pos[batch] + 1, span);
    const int tile = blockIdx.x;
    const int head = blockIdx.y;
    const int lane = threadIdx.x;
    const int key0 = tile * 128;
    if (lane >= d) return;
    float* dest = partials + (((static_cast<long long>(batch) * heads + head) * gridDim.x) + tile) * d + lane;
    if (key0 >= visible) {  // a tile past the keys in use adds zero
        *dest = 0.f;
        return;
    }
    const unsigned long long* desc = plan_desc(plan, plan.slot[batch]);
    const void* values = reinterpret_cast<const void*>(desc[plan_second(layer)]);
    const unsigned* table = plan_pages(desc);
    const int kv = head / (heads / kv_heads);
    const float* row = scores + (static_cast<long long>(batch) * heads + head) * span;
    const float max_score = stats[(static_cast<long long>(batch) * heads + head) * 2];
    const float denom = stats[(static_cast<long long>(batch) * heads + head) * 2 + 1];
    float acc = 0.f;
    const int stop = min(key0 + 128, visible);
    for (int key = key0; key < stop; ++key) {
        const float weight = expf(row[key] - max_score) / denom;
        acc += weight * load(values, KIND, page_at(table, kv_heads, kv, key, d) + lane);
    }
    *dest = acc;
}

// The words of a row copied by a block row of threads, four at a time.
__device__ __forceinline__ void copy_words(unsigned int* to, const unsigned int* from, int words) {
    const int stride = gridDim.x * blockDim.x;
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4; i < words; i += stride * 4) {
        *reinterpret_cast<uint4*>(to + i) = *reinterpret_cast<const uint4*>(from + i);
    }
}

}  // namespace

extern "C" {

__global__ void __launch_bounds__(256)
tf_plan_kv_write(const float* keys, const void* values, int kind, int rows, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_kv_write_body(keys, values, kind, rows, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_f16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<1>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(128)
tf_plan_score_bf16(const float* q, float* scores, int span, int heads, int kv_heads, int d, float scale, PlanArgs plan, int layer) {
    plan_score_body<2>(q, scores, span, heads, kv_heads, d, scale, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_f16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<1>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

__global__ void __launch_bounds__(256)
tf_plan_apply_bf16(const float* scores, const float* stats, float* partials, int span, int heads, int kv_heads, int d, PlanArgs plan, int layer) {
    plan_apply_body<2>(scores, stats, partials, span, heads, kv_heads, d, plan, layer);
}

// Positions pos0 .. pos0 + len of a stream's pages from `src` (16-bit values at h * s_head + r * s_row + j): a prompt
// pass's keys (already rotated and rounded) and values.
__global__ void __launch_bounds__(256)
tf_page_write(const uint16_t* src, uint16_t* pool, const unsigned* table, int len, int kv_heads, int d, long long s_head,
              long long s_row, int pos0) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int h = static_cast<int>((i / d) % kv_heads);
    const int r = static_cast<int>(i / (static_cast<long long>(d) * kv_heads));
    pool[page_at(table, kv_heads, h, pos0 + r, d) + j] = src[h * s_head + r * s_row + j];
}

// The first `len` positions of a stream's pages as (kv_heads, len, d): the flat layout the other attention kernels read.
__global__ void __launch_bounds__(256)
tf_page_gather(const uint16_t* pool, const unsigned* table, uint16_t* dst, int len, int kv_heads, int d) {
    const long long i = gid();
    if (i >= static_cast<long long>(len) * kv_heads * d) return;
    const int j = static_cast<int>(i % d);
    const int r = static_cast<int>((i / d) % len);
    const int h = static_cast<int>(i / (static_cast<long long>(d) * len));
    dst[i] = pool[page_at(table, kv_heads, h, r, d) + j];
}

// Row r of `dst` is `words` words from the address srcs[r]: a launch whose rows come from anywhere keeps its arguments.
__global__ void __launch_bounds__(256) tf_plan_gather(const unsigned long long* srcs, unsigned int* dst, int words) {
    const unsigned int* from = reinterpret_cast<const unsigned int*>(srcs[blockIdx.y]);
    unsigned int* to = dst + static_cast<long long>(blockIdx.y) * words;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < words; i += gridDim.x * blockDim.x) to[i] = from[i];
}

// Keeps each listed slot's first `keep[slot]` rows (a negative count lists none): a linear layer's conv window and state
// as they stood after the last kept row, and the slot's last kept final row. A grid plane a layer, then one for the rows.
__global__ void __launch_bounds__(256)
tf_plan_keep(PlanArgs plan, const int* keep, const unsigned int* hidden, int hidden_words, int conv_words, int delta_words, int layers) {
    const int slot = blockIdx.z;
    const int n = keep[slot];
    if (n < 0) return;
    const int row = plan.first[slot] + n - 1;
    const unsigned long long* desc = plan_desc(plan, slot);
    if (blockIdx.y == layers) {
        if (desc[1] != 0) copy_words(reinterpret_cast<unsigned int*>(desc[1]), hidden + static_cast<long long>(row) * hidden_words, hidden_words);
        return;
    }
    const int layer = blockIdx.y;
    if (n >= plan.count[slot] || plan.snaps[2 * layer] == 0) return;
    copy_words(reinterpret_cast<unsigned int*>(desc[plan_first(layer)]),
               reinterpret_cast<const unsigned int*>(plan.snaps[2 * layer]) + static_cast<long long>(row) * conv_words, conv_words);
    copy_words(reinterpret_cast<unsigned int*>(desc[plan_second(layer)]),
               reinterpret_cast<const unsigned int*>(plan.snaps[2 * layer + 1]) + static_cast<long long>(row) * delta_words, delta_words);
}

}  // extern "C"
