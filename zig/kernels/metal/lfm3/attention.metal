// Fixed contiguous key partitions keep one query's arithmetic independent of neighboring rows and lane width.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint L3CHUNKS = 64;
constant constexpr uint L3PER = 4096;
constant constexpr uint L3PART = 1024;
struct lfm3_attn_args { uint len, cap; float scale; uint window, chunk_base; };

#ifndef L3_ATTN_ACC
#define L3_ATTN_ACC float
#endif

inline float lfm3_attention_dot(const thread float* q, const device bfloat* k) {
  L3_ATTN_ACC dot = 0.0f;
  for (uint j = 0; j < 4; ++j) dot += q[j] * float(k[32 * j]);
  return simd_sum(float(dot));
}

inline void lfm3_attention_partial(const device bfloat* queries, const device bfloat* keys,
    const device bfloat* values, constant lfm3_attn_args& p, device float* pm, device float* pl, device float* po,
    uint head, uint row, uint partition, uint sg, uint lane,
    threadgroup float* maxima, threadgroup float* totals, threadgroup float* outputs) {
  const uint chunk = partition + p.chunk_base, count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = max(chunk * L3PER + sg * L3PART, cutoff);
  const uint end = min(chunk * L3PER + (sg + 1) * L3PART, count);
  const uint kh = head / 4;
  float q[4];
  L3_ATTN_ACC result[4] = {0, 0, 0, 0};
  for (uint j = 0; j < 4; ++j) q[j] = float(queries[(size_t(row) * 16 + head) * 128 + lane + 32 * j]);
  float maximum = -INFINITY;
  L3_ATTN_ACC total = 0.0f;
  for (uint key = first; key < end; ++key) {
    const size_t base = (size_t(kh) * p.cap + key) * 128 + lane;
    const float score = lfm3_attention_dot(q, keys + base) * p.scale;
    const float next = max(maximum, score);
    const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
    total = total * alpha + probability;
    for (uint j = 0; j < 4; ++j) result[j] = result[j] * alpha + probability * float(values[base + 32 * j]);
    maximum = next;
  }
  if (lane == 0) { maxima[sg] = maximum; totals[sg] = total; }
  for (uint j = 0; j < 4; ++j) outputs[sg * 128 + lane + 32 * j] = result[j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg != 0) return;
  const size_t target = (size_t(row) * 16 + head) * L3CHUNKS + chunk;
  float merged_max = -INFINITY, merged_total = 0.0f, merged[4] = {0, 0, 0, 0};
  for (uint part = 0; part < 4; ++part) {
    if (totals[part] == 0.0f) continue;
    const float next = max(merged_max, maxima[part]);
    const float a = merged_max == -INFINITY ? 0.0f : metal::precise::exp(merged_max - next);
    const float b = metal::precise::exp(maxima[part] - next);
    merged_total = merged_total * a + totals[part] * b;
    for (uint j = 0; j < 4; ++j) merged[j] = merged[j] * a + outputs[part * 128 + lane + 32 * j] * b;
    merged_max = next;
  }
  if (lane == 0) { pm[target] = merged_max; pl[target] = merged_total; }
  for (uint j = 0; j < 4; ++j) po[target * 128 + lane + 32 * j] = merged[j];
}

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_partial(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                 const device bfloat* values [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float maxima[4], totals[4], outputs[4 * 128];
  lfm3_attention_partial(queries, keys, values, p, pm, pl, po, tg.x, tg.y, tg.z, sg, lane, maxima, totals, outputs);
}

inline float lfm3_attention_merge(const device float* pm, const device float* pl, const device float* po,
                                  constant lfm3_attn_args& p, uint head, uint row, uint lane,
                                  thread float* result) {
  const uint count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = cutoff / L3PER, last = (count + L3PER - 1) / L3PER;
  const size_t base = (size_t(row) * 16 + head) * L3CHUNKS;
  float maximum = -INFINITY;
  L3_ATTN_ACC total = 0.0f;
  for (uint chunk = first; chunk < last; ++chunk) {
    const size_t i = base + chunk;
    if (pl[i] == 0.0f) continue;
    const float next = max(maximum, pm[i]);
    const float a = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float b = metal::precise::exp(pm[i] - next);
    total = total * a + pl[i] * b;
    for (uint j = 0; j < 4; ++j) result[j] = float(L3_ATTN_ACC(result[j] * a + po[i * 128 + lane + 32 * j] * b));
    maximum = next;
  }
  return float(total);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                               const device float* po [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                               device bfloat* y [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  float result[4] = {0, 0, 0, 0};
  const float total = lfm3_attention_merge(pm, pl, po, p, tg.x, tg.y, lane, result);
  for (uint j = 0; j < 4; ++j) y[(size_t(tg.y) * 16 + tg.x) * 128 + lane + 32 * j] = bfloat(result[j] / total);
}

// The merge tap invokes the production chunk loop before division and BF16 output rounding.
[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attention_merge_sums(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    constant lfm3_attn_args& p [[buffer(3)]], device float* out [[buffer(4)]],
    uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float result[4] = {0, 0, 0, 0};
  const float total = lfm3_attention_merge(pm, pl, po, p, tg.x, tg.y, lane, result);
  const size_t base = (size_t(tg.y) * 16 + tg.x) * 129;
  for (uint j = 0; j < 4; ++j) out[base + lane + 32 * j] = result[j];
  if (lane == 0) out[base + 128] = total;
}

// The contract observes the production dot before softmax can mask its accumulation error.
[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attention_dot_sums(
    const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], device float* out [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float values[4];
  for (uint j = 0; j < 4; ++j) values[j] = float(q[size_t(row) * 128 + lane + 32 * j]);
  const float total = lfm3_attention_dot(values, k + size_t(row) * 128 + lane);
  if (lane == 0) out[row] = total;
}

struct lfm3_attn_segment { ulong keys, values; uint base, rows; lfm3_attn_args args; uint pad; };

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_segments(
    const device bfloat* queries [[buffer(0)]], constant lfm3_attn_segment* segments [[buffer(1)]],
    const device uint* row_segments [[buffer(2)]], device float* pm [[buffer(4)]], device float* pl [[buffer(5)]],
    device float* po [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const uint row = tg.y - seg.base;
  const uint last = (seg.args.len + row + 1 + L3PER - 1) / L3PER;
  if (tg.z + seg.args.chunk_base >= last) return;
  threadgroup float maxima[4], totals[4], outputs[4 * 128];
  const size_t offset = size_t(seg.base) * 16 * L3CHUNKS;
  lfm3_attention_partial(queries + size_t(seg.base) * 16 * 128,
    reinterpret_cast<const device bfloat*>(seg.keys), reinterpret_cast<const device bfloat*>(seg.values),
    seg.args, pm + offset, pl + offset, po + offset * 128, tg.x, row, tg.z, sg, lane, maxima, totals, outputs);
}

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_segments_merge(
    const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
    constant lfm3_attn_segment* segments [[buffer(3)]], device bfloat* y [[buffer(4)]],
    const device uint* row_segments [[buffer(5)]], uint2 tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  constant lfm3_attn_segment& seg = segments[row_segments[tg.y]];
  const size_t offset = size_t(seg.base) * 16 * L3CHUNKS;
  float result[4] = {0, 0, 0, 0};
  const float total = lfm3_attention_merge(pm + offset, pl + offset, po + offset * 128,
    seg.args, tg.x, tg.y - seg.base, lane, result);
#ifdef L3_SEGMENT_BAD_TAIL
  if (tg.y + 1 == seg.base + seg.rows) return;
#endif
  for (uint j = 0; j < 4; ++j) y[(size_t(tg.y) * 16 + tg.x) * 128 + lane + 32 * j] = bfloat(result[j] / total);
}
