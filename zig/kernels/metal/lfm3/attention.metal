// Fixed contiguous key partitions keep one query's arithmetic independent of neighboring rows and lane width.
#include <metal_stdlib>
using namespace metal;

constant constexpr uint L3CHUNKS = 64;
constant constexpr uint L3PER = 4096;
constant constexpr uint L3PART = 1024;
struct lfm3_attn_args { uint len, cap; float scale; uint window, chunk_base; };

[[kernel, max_total_threads_per_threadgroup(128)]] void lfm3_attn_partial(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                 const device bfloat* values [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                                 device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]],
                                 uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y, chunk = tg.z + p.chunk_base, count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = max(chunk * L3PER + sg * L3PART, cutoff);
  const uint end = min(chunk * L3PER + (sg + 1) * L3PART, count);
  const uint kh = head / 4;
  float q[4], result[4] = {0, 0, 0, 0};
  for (uint j = 0; j < 4; ++j) q[j] = float(queries[(size_t(row) * 16 + head) * 128 + lane + 32 * j]);
  float maximum = -INFINITY, total = 0.0f;
  for (uint key = first; key < end; ++key) {
    const size_t base = (size_t(kh) * p.cap + key) * 128 + lane;
    float dot = 0.0f;
    for (uint j = 0; j < 4; ++j) dot += q[j] * float(keys[base + 32 * j]);
    const float score = simd_sum(dot) * p.scale;
    const float next = max(maximum, score);
    const float alpha = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float probability = score == -INFINITY ? 0.0f : metal::precise::exp(score - next);
    total = total * alpha + probability;
    for (uint j = 0; j < 4; ++j) result[j] = result[j] * alpha + probability * float(values[base + 32 * j]);
    maximum = next;
  }
  threadgroup float maxima[4], totals[4], outputs[4 * 128];
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

[[kernel, max_total_threads_per_threadgroup(32)]] void lfm3_attn_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]],
                               const device float* po [[buffer(2)]], constant lfm3_attn_args& p [[buffer(3)]],
                               device bfloat* y [[buffer(4)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, row = tg.y, count = p.len + row + 1;
  const uint cutoff = p.window == 0 ? 0 : count > p.window ? count - p.window : 0;
  const uint first = cutoff / L3PER, last = (count + L3PER - 1) / L3PER;
  const size_t base = (size_t(row) * 16 + head) * L3CHUNKS;
  float maximum = -INFINITY, total = 0.0f, result[4] = {0, 0, 0, 0};
  for (uint chunk = first; chunk < last; ++chunk) {
    const size_t i = base + chunk;
    if (pl[i] == 0.0f) continue;
    const float next = max(maximum, pm[i]);
    const float a = maximum == -INFINITY ? 0.0f : metal::precise::exp(maximum - next);
    const float b = metal::precise::exp(pm[i] - next);
    total = total * a + pl[i] * b;
    for (uint j = 0; j < 4; ++j) result[j] = result[j] * a + po[i * 128 + lane + 32 * j] * b;
    maximum = next;
  }
  for (uint j = 0; j < 4; ++j) y[(size_t(row) * 16 + head) * 128 + lane + 32 * j] = bfloat(result[j] / total);
}
