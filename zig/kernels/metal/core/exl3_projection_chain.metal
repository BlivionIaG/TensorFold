// Chain kernels: the epilogue of one EXL3 projection fused with the block-local work that follows it.
// Appended to the core projection library, so the helpers (tf_exl3_hadamard, tf_exl3_scale, tf_exl3_round_bf16)
// are the very ones the unfused rotate_in / rotate_out kernels use. Every expression below is copied from the
// kernel it replaces, statement for statement: the compiler contracts a multiply-add only inside one expression, so
// the same statements give the same bits.

// A bf16 value as a float: the load tf_exl3_rotate_in does of a stored activation.
inline float tf_exl3_chain_bf(ushort bits) { return as_type<float>(uint(bits) << 16); }

// The next projection's rotate_in applied to a block whose bf16 values the lanes hold (lane l: l, l + 32, ...).
// `at` is the first element of the lane's strip in the next projection's flat input; its blocks are 128 aligned.
inline void tf_exl3_chain_rotate_in(thread float (&v)[4], uint lane, const device ushort* su, uint su_signs,
                                    uint first, device half* rotated, ulong dest) {
  for (uint j = 0; j < 4; ++j) v[j] = v[j] * tf_exl3_scale(su, first + 32 * j, su_signs);
  tf_exl3_hadamard(v, lane);
  for (uint j = 0; j < 4; ++j) rotated[dest + 32 * j] = half(v[j] * 0.08838834764831845f);
}

struct TfExl3Next {
  uint k, su_signs, reserved0, reserved1;
};

// rotate_out of a (dense or indexed) projection, then the next projection's rotate_in on the very bf16 values the
// rotate_out stores: one dispatch where there were two. `y` still receives the bf16 output; `rotated` is the next
// projection's prerotated input, row-major with `next.k` halfs a row. The output row stride a.y_stride must be a
// multiple of 128 so no block straddles two rows of the next projection.
kernel void tf_exl3_rotate_out_in(const device float* sums [[buffer(0)]],
                                  const device ushort* sv [[buffer(1)]],
                                  const device half* bias [[buffer(2)]],
                                  device uchar* y [[buffer(3)]],
                                  constant TfExl3Args& a [[buffer(4)]],
                                  const device uint* output_ids [[buffer(5)]],
                                  const device uint* routes [[buffer(6)]],
                                  const device TfExl3Expert* descriptors [[buffer(7)]],
                                  const device ushort* su_next [[buffer(8)]],
                                  constant TfExl3Next& next [[buffer(9)]],
                                  device half* rotated [[buffer(10)]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint3 tptg [[threads_per_threadgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = (tg.x * (tptg.x / 32u) + sg) * 128;
  const bool active = tf_exl3_active(routes, a, row);
  const uint id = a.indexed ? routes[row] : 0;
  const uint width = (a.indexed && active && descriptors[id].reserved != 0) ? descriptors[id].reserved : a.n;
  if (base >= width || !active) return;
  const device ushort* scales = a.indexed ? descriptors[id].sv : sv;
  const device half* biases = a.indexed ? descriptors[id].bias : bias;
  const uint signs = a.indexed ? descriptors[id].sv_signs : a.sv_signs;
  const uint has_bias = a.indexed ? descriptors[id].has_bias : a.has_bias;
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = sums[ulong(row) * a.n + base + lane + 32 * j];
  tf_exl3_hadamard(v, lane);
  const uint dest = a.output_ids ? output_ids[row] : row;
  const ulong first = ulong(dest) * a.y_stride + base;
  for (uint j = 0; j < 4; ++j) {
    const uint n = base + lane + 32 * j;
    float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
    if (has_bias) out += float(biases[n]);
    const bfloat stored = bfloat(out);
    reinterpret_cast<device bfloat*>(y)[ulong(dest) * a.y_stride + n] = stored;
    v[j] = tf_exl3_chain_bf(as_type<ushort>(stored));
  }
  tf_exl3_chain_rotate_in(v, lane, su_next, next.su_signs, uint(first % next.k) + lane, rotated, first + lane);
}

struct TfExl3RmsRot {
  uint rows, width, x_stride, y_stride;
  float eps;
  uint weight_f32, su_signs, reserved;
};

inline float tf_exl3_chain_weight(const device uchar* w, ulong i, uint f32) {
  if (f32) return reinterpret_cast<const device float*>(w)[i];
  return tf_exl3_chain_bf(reinterpret_cast<const device ushort*>(w)[i]);
}

// dsv41_rms over a row, then the next projection's rotate_in over the rounded bf16 result: one threadgroup a row, one
// SIMDgroup a 128-element block (a block a SIMDgroup, the groups striding when there are more blocks than groups).
// The sum of squares runs on the first 256 threads in dsv41_rms's own order; `y` receives the normalized bf16 row
// (in place over `x` is allowed) and `rotated` the next projection's prerotated input (width halfs a row).
// Contraction is off for the whole kernel, as for dsv41_rms (its sum of squares must not fuse).
#pragma clang fp contract(off)
kernel void tf_exl3_rms_rotate_in(const device ushort* x [[buffer(0)]],
                                  const device uchar* w [[buffer(1)]],
                                  device ushort* y [[buffer(2)]],
                                  constant TfExl3RmsRot& d [[buffer(3)]],
                                  const device ushort* su [[buffer(4)]],
                                  device half* rotated [[buffer(5)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint t [[thread_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint threads [[threads_per_threadgroup]]) {
  threadgroup float sums[8];
  if (row >= d.rows) return;
  if (t < 256) {
    float s = 0;
    for (uint i = t; i < d.width; i += 256) { float a = tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]); s += a * a; }
    s = simd_sum(s);
    if (lane == 0) sums[sg] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    float total = simd_sum(lane < 8 ? sums[lane] : 0.0f);
    const float factor = 1.0f / float(d.width);
    if (lane == 0) sums[0] = rsqrt(total * factor + d.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float scale = sums[0];
  for (uint block = sg; block < d.width / 128u; block += threads / 32u) {
    const uint first = block * 128u + lane;
    float v[4];
    for (uint j = 0; j < 4; ++j) {
      const uint i = first + 32 * j;
      const ushort bits = tf_exl3_round_bf16((tf_exl3_chain_bf(x[ulong(row) * d.x_stride + i]) * scale) * tf_exl3_chain_weight(w, i, d.weight_f32));
      y[ulong(row) * d.y_stride + i] = bits;
      v[j] = tf_exl3_chain_bf(bits);
    }
    tf_exl3_chain_rotate_in(v, lane, su, d.su_signs, first, rotated, ulong(row) * d.width + first);
  }
}
#pragma clang fp contract(on)

struct TfExl3Rope {
  uint width, rotary, position, step, inverse, reserved0, reserved1, reserved2;
};

// rotate_out of a projection whose output rows are heads of `width` columns, then dsv41_rope over each head's last
// `rotary` columns, on the bf16 values the rotate_out stores: one dispatch where there were two. The rotary columns
// are a multiple of 32 and fill the end of a head's last 128-column block, so the adjacent pair a rope rotates sits in
// two lanes of one SIMDgroup. The row is rotated at position + row * step, as dsv41_rope's absolute-position grid.
kernel void tf_exl3_rotate_out_rope(const device float* sums [[buffer(0)]],
                                    const device ushort* sv [[buffer(1)]],
                                    const device half* bias [[buffer(2)]],
                                    device uchar* y [[buffer(3)]],
                                    constant TfExl3Args& a [[buffer(4)]],
                                    const device uint* output_ids [[buffer(5)]],
                                    const device uint* routes [[buffer(6)]],
                                    const device TfExl3Expert* descriptors [[buffer(7)]],
                                    const device float2* factors [[buffer(8)]],
                                    constant TfExl3Rope& rp [[buffer(9)]],
                                    uint lane [[thread_index_in_simdgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]],
                                    uint3 tptg [[threads_per_threadgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  const uint row = tg.y, base = (tg.x * (tptg.x / 32u) + sg) * 128;
  const bool active = tf_exl3_active(routes, a, row);
  const uint id = a.indexed ? routes[row] : 0;
  const uint width = (a.indexed && active && descriptors[id].reserved != 0) ? descriptors[id].reserved : a.n;
  if (base >= width || !active) return;
  const device ushort* scales = a.indexed ? descriptors[id].sv : sv;
  const device half* biases = a.indexed ? descriptors[id].bias : bias;
  const uint signs = a.indexed ? descriptors[id].sv_signs : a.sv_signs;
  const uint has_bias = a.indexed ? descriptors[id].has_bias : a.has_bias;
  float v[4];
  for (uint j = 0; j < 4; ++j) v[j] = sums[ulong(row) * a.n + base + lane + 32 * j];
  tf_exl3_hadamard(v, lane);
  const uint dest = a.output_ids ? output_ids[row] : row;
  for (uint j = 0; j < 4; ++j) {
    const uint n = base + lane + 32 * j;
    float out = (v[j] * 0.08838834764831845f) * tf_exl3_scale(scales, n, signs);
    if (has_bias) out += float(biases[n]);
    ushort bits = as_type<ushort>(bfloat(out));
    // The pair's other element, fetched by every lane (the rotary columns are SIMDgroup uniform).
    const float other = tf_exl3_chain_bf(ushort(simd_shuffle_xor(uint(bits), 1)));
    const uint d = n % rp.width;
    if (d >= rp.width - rp.rotary) {
      const float2 rotation = factors[(ulong(rp.position) + ulong(dest) * rp.step) * (rp.rotary / 2) + (d - (rp.width - rp.rotary)) / 2];
      const float c = rotation.x, s = rp.inverse ? -rotation.y : rotation.y;
      const float own = tf_exl3_chain_bf(bits);
      uint u;
      if ((n & 1u) == 0u) u = as_type<uint>(own * c - other * s);
      else u = as_type<uint>(other * s + own * c);
      bits = ushort((u + 0x7fff + ((u >> 16) & 1)) >> 16);
    }
    reinterpret_cast<device ushort*>(y)[ulong(dest) * a.y_stride + n] = bits;
  }
}
