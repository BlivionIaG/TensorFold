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
