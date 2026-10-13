// LFM2-MoE's kernels beside the core row projections: every row's arithmetic is its own, whatever the rows beside it.
#include <metal_stdlib>
using namespace metal;

// The hidden width, heads, head width, experts and picks (the config's pinned geometry).
constant constexpr uint D = 2048;
constant constexpr uint QH = 32;
constant constexpr uint KVH = 8;
constant constexpr uint HD = 64;
constant constexpr uint E = 64;
constant constexpr uint TOPK = 4;
// Attention: key chunks a row splits into at most.
constant constexpr uint CHUNKS = 32;

// Table row ids[r] as bf16 scale * q + bias (MLX's dequantize), one packed unit (1 byte, 3 at 6 bits) a thread.
template <int BITS>
inline void lfm2_unit(const device uint8_t* table, const device bfloat* scales, const device bfloat* biases, long id, int unit,
                      device bfloat* out) {
  constexpr int PER = BITS == 6 ? 4 : 8 / BITS;
  const int first = unit * PER;
  const bfloat scale = scales[id * (D / 64) + first / 64];
  const bfloat bias = biases[id * (D / 64) + first / 64];
  const device uint8_t* w = table + id * (long(D) * BITS / 8) + long(unit) * (BITS == 6 ? 3 : 1);
  if (BITS == 6) {
    out[0] = (w[0] & 0x3f) * scale + bias;
    out[1] = (((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * scale + bias;
    out[2] = (((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * scale + bias;
    out[3] = ((w[2] >> 2) & 0x3f) * scale + bias;
  } else {
    const uint byte = w[0];
    for (int i = 0; i < PER; ++i) {
      const uint8_t q = BITS == 8 ? uint8_t(byte) : uint8_t((byte >> (BITS * i)) & ((1u << BITS) - 1u));
      out[i] = scale * q + bias;
    }
  }
}

template <int BITS>
[[kernel]] void lfm2_embed(const device uint* ids [[buffer(0)]], const device uint8_t* table [[buffer(1)]],
                           const device bfloat* scales [[buffer(2)]], const device bfloat* biases [[buffer(3)]],
                           device bfloat* out [[buffer(4)]], uint2 at [[thread_position_in_grid]]) {
  constexpr int PER = BITS == 6 ? 4 : 8 / BITS;
  if (at.x < D / PER) lfm2_unit<BITS>(table, scales, biases, long(ids[at.y]), int(at.x), out + size_t(at.y) * D + PER * at.x);
}
template [[host_name("lfm2_embed_b4")]] [[kernel]] decltype(lfm2_embed<4>) lfm2_embed<4>;
template [[host_name("lfm2_embed_b6")]] [[kernel]] decltype(lfm2_embed<6>) lfm2_embed<6>;
template [[host_name("lfm2_embed_b8")]] [[kernel]] decltype(lfm2_embed<8>) lfm2_embed<8>;

// h += r (bf16) when `add`, then x = w * bf16(h * rsqrt(mean(h^2) + eps)): MLX's one-row RMS order, 4 values a thread.
[[kernel]] void lfm2_norm(device bfloat* h [[buffer(0)]], const device bfloat* r [[buffer(1)]], const device bfloat* w [[buffer(2)]],
                          constant float& eps [[buffer(3)]], constant uint& add [[buffer(4)]], device bfloat* x [[buffer(5)]],
                          uint row [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float sums[32];
  threadgroup float inv[1];
  const size_t base = size_t(row) * D + 4 * t;
  float v[4];
  float sq = 0.0f;
  for (int j = 0; j < 4; ++j) {
    bfloat hv = h[base + j];
    if (add != 0) {
      hv = hv + r[base + j];
      h[base + j] = hv;
    }
    v[j] = hv;
    sq += v[j] * v[j];
  }
  sq = simd_sum(sq);
  if (sg == 0) sums[lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) sums[sg] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    const float all = simd_sum(sums[lane]);
    if (lane == 0) inv[0] = metal::precise::rsqrt(all / float(D) + eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int j = 0; j < 4; ++j) x[base + j] = w[4 * t + j] * static_cast<bfloat>(v[j] * inv[0]);
}

// A segment's short conv: Bx = B * x, 3 causal taps over (state [2, D], Bx), y = C * conv; SNAP: each row's state.
[[kernel]] void lfm2_conv(const device bfloat* bcx [[buffer(0)]], const device bfloat* state [[buffer(1)]],
                          const device bfloat* w [[buffer(2)]], device bfloat* y [[buffer(3)]], device bfloat* snap [[buffer(4)]],
                          uint2 at [[thread_position_in_grid]]) {
  const uint ch = at.x, r = at.y;
  if (ch >= D) return;
  bfloat bx[3];
  for (int tap = 0; tap < 3; ++tap) {
    const int src = int(r) + tap - 2;
    if (src >= 0) {
      const device bfloat* row = bcx + size_t(src) * 3 * D;
      bx[tap] = row[ch] * row[2 * D + ch];
    } else {
      bx[tap] = state[size_t(src + 2) * D + ch];
    }
  }
  float acc = 0.0f;
  for (int tap = 0; tap < 3; ++tap) acc += float(bx[tap]) * float(w[ch * 3 + tap]);
  y[size_t(r) * D + ch] = bcx[size_t(r) * 3 * D + D + ch] * bfloat(acc);
  snap[(size_t(r) * 2) * D + ch] = bx[1];
  snap[(size_t(r) * 2 + 1) * D + ch] = bx[2];
}

// One head of one row: RMS norm over its 64 values (lane l: l and l + 32), then RoPE on the pair at position `pos`.
inline void lfm2_head(const device bfloat* src, const device bfloat* gain, float eps, float log2_theta, uint pos, uint lane,
                      thread bfloat& a, thread bfloat& b) {
  const float x0 = src[lane], x1 = src[lane + 32];
  const float inv = metal::precise::rsqrt(simd_sum(x0 * x0 + x1 * x1) / float(HD) + eps);
  const float n0 = float(gain[lane] * static_cast<bfloat>(x0 * inv));
  const float n1 = float(gain[lane + 32] * static_cast<bfloat>(x1 * inv));
  const float theta = float(pos) * metal::exp2(-(float(lane) / float(HD / 2)) * log2_theta);
  const float c = metal::precise::cos(theta), s = metal::precise::sin(theta);
  a = static_cast<bfloat>(n0 * c - n1 * s);
  b = static_cast<bfloat>(n0 * s + n1 * c);
}

// A simdgroup a (head, row): queries normed and rotated, keys and values into the [KVH, cap, HD] cache at len + r.
struct lfm2_qk_args { uint len, cap; float eps, log2_theta; };
[[kernel]] void lfm2_qk(const device bfloat* q [[buffer(0)]], const device bfloat* k [[buffer(1)]], const device bfloat* v [[buffer(2)]],
                        const device bfloat* qn [[buffer(3)]], const device bfloat* kn [[buffer(4)]], constant lfm2_qk_args& p [[buffer(5)]],
                        device bfloat* queries [[buffer(6)]], device bfloat* keys [[buffer(7)]], device bfloat* values [[buffer(8)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, r = tg.y, pos = p.len + r;
  bfloat a, b;
  if (head < QH) {
    lfm2_head(q + (size_t(r) * QH + head) * HD, qn, p.eps, p.log2_theta, pos, lane, a, b);
    device bfloat* out = queries + (size_t(r) * QH + head) * HD;
    out[lane] = a;
    out[lane + 32] = b;
    return;
  }
  const uint kh = head - QH;
  lfm2_head(k + (size_t(r) * KVH + kh) * HD, kn, p.eps, p.log2_theta, pos, lane, a, b);
  const size_t at = (size_t(kh) * p.cap + pos) * HD;
  keys[at + lane] = a;
  keys[at + lane + 32] = b;
  values[at + lane] = v[(size_t(r) * KVH + kh) * HD + lane];
  values[at + lane + 32] = v[(size_t(r) * KVH + kh) * HD + lane + 32];
}

// A row's key chunk from its own key count alone (64-key multiples, at most CHUNKS): sums independent of the window.
inline uint lfm2_chunk(uint keys) {
  const uint per = (keys + CHUNKS - 1) / CHUNKS;
  return (per + 63) / 64 * 64;
}

// One key chunk's online fp32 max, sum and value sums in key order for (chunk, head, row) into PM, PL and PO.
struct lfm2_attn_args { uint len, cap; float scale; };
[[kernel]] void lfm2_attn_partial(const device bfloat* queries [[buffer(0)]], const device bfloat* keys [[buffer(1)]],
                                  const device bfloat* values [[buffer(2)]], constant lfm2_attn_args& p [[buffer(3)]],
                                  device float* pm [[buffer(4)]], device float* pl [[buffer(5)]], device float* po [[buffer(6)]],
                                  uint3 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint chunk = tg.x, head = tg.y, r = tg.z;
  const uint count = p.len + r + 1;
  const uint per = lfm2_chunk(count);
  const uint first = chunk * per;
  if (first >= count) return;
  const uint last = min(count, first + per);
  const uint kh = head / (QH / KVH);
  const device bfloat* qr = queries + (size_t(r) * QH + head) * HD;
  const float q0 = float(qr[lane]) * p.scale, q1 = float(qr[lane + 32]) * p.scale;
  const device bfloat* kb = keys + size_t(kh) * p.cap * HD;
  const device bfloat* vb = values + size_t(kh) * p.cap * HD;
  float m = -INFINITY, l = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (uint key = first; key < last; ++key) {
    const device bfloat* kr = kb + size_t(key) * HD;
    const float s = simd_sum(q0 * float(kr[lane]) + q1 * float(kr[lane + 32]));
    const float m2 = max(m, s);
    const float old = metal::fast::exp(m - m2), now = metal::fast::exp(s - m2);
    const device bfloat* vr = vb + size_t(key) * HD;
    l = l * old + now;
    o0 = o0 * old + now * float(vr[lane]);
    o1 = o1 * old + now * float(vr[lane + 32]);
    m = m2;
  }
  const size_t at = (size_t(r) * QH + head) * CHUNKS + chunk;
  if (lane == 0) {
    pm[at] = m;
    pl[at] = l;
  }
  po[at * HD + lane] = o0;
  po[at * HD + lane + 32] = o1;
}

// The chunks merged in chunk order for (head, row): Y [rows, QH*HD] bf16.
[[kernel]] void lfm2_attn_merge(const device float* pm [[buffer(0)]], const device float* pl [[buffer(1)]], const device float* po [[buffer(2)]],
                                constant lfm2_attn_args& p [[buffer(3)]], device bfloat* y [[buffer(4)]],
                                uint2 tg [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  const uint head = tg.x, r = tg.y;
  const uint count = p.len + r + 1;
  const uint per = lfm2_chunk(count);
  const uint chunks = (count + per - 1) / per;
  const size_t at = (size_t(r) * QH + head) * CHUNKS;
  float m = -INFINITY;
  for (uint c = 0; c < chunks; ++c) m = max(m, pm[at + c]);
  float l = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (uint c = 0; c < chunks; ++c) {
    const float f = metal::fast::exp(pm[at + c] - m);
    l += pl[at + c] * f;
    o0 += po[(at + c) * HD + lane] * f;
    o1 += po[(at + c) * HD + lane + 32] * f;
  }
  device bfloat* out = y + (size_t(r) * QH + head) * HD;
  out[lane] = static_cast<bfloat>(o0 / l);
  out[lane + 32] = static_cast<bfloat>(o1 / l);
}

// Router logits [rows, E] in fp32: a simdgroup an expert, lane l over inputs 4l..4l+3 of each 128, then simd_sum.
[[kernel]] void lfm2_route_logits(const device bfloat* x [[buffer(0)]], const device float* router [[buffer(1)]],
                                  device float* logits [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const uint e = tg.x * 8 + sg, row = tg.y;
  const device bfloat4* xr = (const device bfloat4*)(x + size_t(row) * D) + lane;
  const device float4* wr = (const device float4*)(router + size_t(e) * D) + lane;
  float acc = 0.0f;
  for (uint i = 0; i < D / 128; ++i) {
    const float4 xv = float4(xr[i * 32]), wv = wr[i * 32];
    acc += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
  }
  acc = simd_sum(acc);
  if (lane == 0) logits[size_t(row) * E + e] = acc;
}

// A row's picks, a simdgroup a row: top-k of sigmoid + bias (lower id on ties), sigmoids / (sum + 1e-6) * scale.
[[kernel]] void lfm2_route_select(const device float* logits [[buffer(0)]], const device float* bias [[buffer(1)]],
                                  constant float& scale [[buffer(2)]], device uint* ids [[buffer(3)]], device bfloat* wts [[buffer(4)]],
                                  uint row [[threadgroup_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
  float gate[2], score[2];
  for (uint h = 0; h < 2; ++h) {
    const uint e = lane + 32 * h;
    gate[h] = 1.0f / (1.0f + metal::precise::exp(-logits[size_t(row) * E + e]));
    score[h] = gate[h] + bias[e];
  }
  uint pick[TOPK];
  float picked[TOPK];
  float total = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const uint mine = score[1] > score[0] ? 1 : 0;
    float best = score[mine];
    uint at = lane + 32 * mine;
    for (uint off = 16; off > 0; off >>= 1) {
      const float b = simd_shuffle_xor(best, off);
      const uint i = simd_shuffle_xor(at, off);
      if (b > best || (b == best && i < at)) {
        best = b;
        at = i;
      }
    }
    pick[k] = at;
    picked[k] = simd_shuffle(gate[at / 32], at % 32);
    total += picked[k];
    if (at % 32 == lane) score[at / 32] = -INFINITY;
  }
  if (lane < TOPK) {
    ids[row * TOPK + lane] = pick[lane];
    wts[row * TOPK + lane] = static_cast<bfloat>(picked[lane] / (total + 1e-6f) * scale);
  }
}

// act = silu(gate) * up over [rows, n] (bf16 at each step, as mlx-lm's compiled swiglu and core/tile_qmm's twin).
[[kernel]] void lfm2_swiglu(const device bfloat* gate [[buffer(0)]], const device bfloat* up [[buffer(1)]],
                            device bfloat* act [[buffer(2)]], constant uint& count [[buffer(3)]], uint i [[thread_position_in_grid]]) {
  if (i >= count) return;
  const float g = gate[i];
  const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-g)));
  const bfloat silu = static_cast<bfloat>(g * float(s));
  act[i] = static_cast<bfloat>(float(silu) * float(up[i]));
}

// R [rows, D] = sum over a row's picks of bf16(y * weight), in pick order with an fp32 sum.
[[kernel]] void lfm2_combine(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]], device bfloat* r [[buffer(2)]],
                             uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  float acc = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const size_t slot = size_t(at.y) * TOPK + k;
    acc += float(static_cast<bfloat>(float(y[slot * D + at.x]) * float(wts[slot])));
  }
  r[size_t(at.y) * D + at.x] = static_cast<bfloat>(acc);
}

// Picks sorted by expert, in slot order within one: STARTS [E + 1], ORDER (sorted -> slot), POS (slot -> sorted).
[[kernel]] void lfm2_sort(const device uint* ids [[buffer(0)]], constant uint& slots [[buffer(1)]], device uint* starts [[buffer(2)]],
                          device uint* order [[buffer(3)]], device uint* pos [[buffer(4)]], uint e [[thread_position_in_threadgroup]]) {
  threadgroup uint counts[E];
  uint n = 0;
  for (uint j = 0; j < slots; ++j) n += ids[j] == e ? 1 : 0;
  counts[e] = n;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint at = 0;
  for (uint i = 0; i < e; ++i) at += counts[i];
  starts[e] = at;
  if (e == E - 1) starts[E] = at + n;
  for (uint j = 0; j < slots; ++j)
    if (ids[j] == e) {
      order[at] = j;
      pos[j] = at;
      ++at;
    }
}

// XS [slots, D]: sorted row j is X's row order[j] / TOPK (a token's input for each of its picks).
[[kernel]] void lfm2_gather(const device bfloat* x [[buffer(0)]], const device uint* order [[buffer(1)]], device bfloat* xs [[buffer(2)]],
                            uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  xs[size_t(at.y) * D + at.x] = x[size_t(order[at.y] / TOPK) * D + at.x];
}

// lfm2_combine with Y's rows in sorted order (slot s at POS[s]).
[[kernel]] void lfm2_combine_sorted(const device bfloat* y [[buffer(0)]], const device bfloat* wts [[buffer(1)]], device bfloat* r [[buffer(2)]],
                                    const device uint* pos [[buffer(3)]], uint2 at [[thread_position_in_grid]]) {
  if (at.x >= D) return;
  float acc = 0.0f;
  for (uint k = 0; k < TOPK; ++k) {
    const size_t slot = size_t(at.y) * TOPK + k;
    acc += float(static_cast<bfloat>(float(y[size_t(pos[slot]) * D + at.x]) * float(wts[slot])));
  }
  r[size_t(at.y) * D + at.x] = static_cast<bfloat>(acc);
}
