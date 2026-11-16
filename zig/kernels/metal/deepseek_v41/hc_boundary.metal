// Hyper-Connection boundary: the release's hc_mixes/hc_pre/hc_post contract in two launches, mix partials then split.

// Written from the release's Block.hc_mixes/hc_pre/hc_post and hc_split_sinkhorn_kernel (revision 314113f8).
#include <metal_stdlib>
using namespace metal;

constant constexpr int kCopies = 4;                             // hc_mult
constant constexpr int kWidth = 5120;                          // one copy's width
constant constexpr int kStream = kCopies * kWidth;             // the flattened hc stream
constant constexpr int kMixes = (2 + kCopies) * kCopies;       // pre, post and the comb cells
constant constexpr int kSlices = kStream / 1024;               // the mix projection's column slices
constant constexpr int kUnits = kStream / 128;                 // 128-column units, one simdgroup's share of a slice
constant constexpr int kUnitsPerCopy = kWidth / 128;           // the units of one stream copy
constant constexpr int kParts = kMixes + 1;                    // the mixes and the stream's squares

struct HcArgs { float eps; float hc_eps; int iters; };
static_assert(sizeof(HcArgs) == 12, "HcArgs must mirror mhc.zig's BoundaryArgs");

inline float hc_round(float v) { return float(bfloat(v)); }

// One thread's four adjacent columns of a slice: the stream's squares and its products against the 24 hc_fn rows.
inline void hc_mix_column(const float* v, const device float* fn, int col, float* out) {
  out[kMixes] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int q = 0; q < kMixes; q++) {
    const device float* f = fn + size_t(q) * kStream + col;
    out[q] = f[0] * v[0] + f[1] * v[1] + f[2] * v[2] + f[3] * v[3];
  }
}

// A slice's reduction: the warp sums, then the first kParts threads add the eight warps in index order and publish.
inline void hc_publish_slice(threadgroup float* warp, device float* part, const float* m,
                             uint lane, uint sg, uint t, uint r, uint j) {
  for (int q = 0; q <= kMixes; q++) {
    const float s = simd_sum(m[q]);
    if (lane == 0) warp[sg * kParts + q] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) <= kMixes) {
    float s = 0.0f;
    for (int k = 0; k < 8; k++) s += warp[k * kParts + int(t)];
    part[(size_t(r) * kSlices + j) * kParts + t] = s;
  }
}

// The mix projection's partials: a threadgroup a (1024-column slice, row), 256 threads, four columns a thread.
[[kernel]] void tf_hc_mix_part(const device bfloat* X [[buffer(0)]], const device float* FN [[buffer(1)]],
                               device float* PART [[buffer(2)]], uint2 tg [[threadgroup_position_in_grid]],
                               uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp[8 * kParts];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const device bfloat* x = X + size_t(r) * kStream + col;
  float v[4];
  for (int i = 0; i < 4; i++) v[i] = float(x[i]);
  float parts[kParts];
  hc_mix_column(v, FN, col, parts);
  hc_publish_slice(&warp[0], PART, parts, lane, sg, t, uint(r), uint(j));
}

// The pending branch into XO (the release's hc_post) and the same slice partials from the bf16 written.
[[kernel]] void tf_hc_expand_mix(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                                 const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                                 const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                                 device bfloat* XO [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                 uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float warp[8 * kParts];
  const int j = int(tg.x), r = int(tg.y);
  const uint t = tpos.x;
  const int col = 1024 * j + 4 * int(t);
  const int k = col / kWidth, d0 = col % kWidth;
  const device bfloat* x = X + size_t(r) * kStream;
  const device float* comb = COMB + r * kCopies * kCopies;
  float v[4];
  for (int i = 0; i < 4; i++) {
    const int d = d0 + i;
    const float branch = float(BR[size_t(r) * kWidth + d]);
    float stream[kCopies];
    for (int jj = 0; jj < kCopies; jj++) stream[jj] = float(x[jj * kWidth + d]);
    float mixed = 0.0f;
    for (int jj = 0; jj < kCopies; jj++) mixed += comb[jj * kCopies + k] * stream[jj];
    const bfloat out = bfloat(POST[r * kCopies + k] * branch + mixed);
    XO[size_t(r) * kStream + col + i] = out;
    v[i] = float(out);
  }
  float parts[kParts];
  hc_mix_column(v, FN, col, parts);
  hc_publish_slice(&warp[0], PART, parts, lane, sg, t, uint(r), uint(j));
}

// The release's split (kernel.py hc_split_sinkhorn_kernel): lanes 0-3 carry pre and post.
// EARLY stops the Sinkhorn once its state repeats: every pass past the first is the same function of the last one's
// result, so a result equal to the one two passes back (a fixed point, or a two-cycle) fixes all the later ones.
template <bool EARLY>
inline void hc_split_row(threadgroup const float* tot, constant const HcArgs& a,
                         const device float* base, const device float* scale, uint lane, uint r,
                         device float* pre_out, device float* post_out, device float* comb_out) {
  const float rstd = precise::rsqrt(tot[kMixes] / float(kStream) + a.eps);
  if (lane < uint(kCopies)) {
    pre_out[r * kCopies + lane] =
        1.0f / (1.0f + precise::exp(-(tot[lane] * rstd * scale[0] + base[lane]))) + a.hc_eps;
    post_out[r * kCopies + lane] =
        2.0f / (1.0f + precise::exp(-(tot[kCopies + lane] * rstd * scale[1] + base[kCopies + lane])));
  }
  const bool live = lane < uint(kCopies * kCopies);
  const uint cell = live ? lane : 0;
  const float z = tot[2 * kCopies + cell] * rstd * scale[2] + base[2 * kCopies + cell];
  float top = max(z, simd_shuffle_xor(z, ushort(1)));
  top = max(top, simd_shuffle_xor(top, ushort(2)));
  const float e = precise::exp(z - top);
  // Match the unfused split's FP32 butterfly: row XOR 2,1; column XOR 8,4.
  float row = e + simd_shuffle_xor(e, ushort(2));
  row = row + simd_shuffle_xor(row, ushort(1));
  float comb = live ? e / row + a.hc_eps : 0.0f;
  float older = 0.0f, last = 0.0f;  // the results one and two passes back
  for (uint it = 0; it < uint(a.iters); it++) {
    if (it > 0) {
      float row_sum = comb + simd_shuffle_xor(comb, ushort(2));
      row_sum = row_sum + simd_shuffle_xor(row_sum, ushort(1));
      comb = live ? comb / (row_sum + a.hc_eps) : 0.0f;
    }
    float col_sum = comb + simd_shuffle_xor(comb, ushort(8));
    col_sum = col_sum + simd_shuffle_xor(col_sum, ushort(4));
    comb = live ? comb / (col_sum + a.hc_eps) : 0.0f;
    if (EARLY) {
      if (it >= 2 && (it & 1u) == 0u && simd_all(!live || comb == older)) {
        if (((uint(a.iters) - 1u - it) & 1u) != 0u) comb = last;
        break;
      }
      older = last;
      last = comb;
    }
  }
  if (live) comb_out[r * kCopies * kCopies + lane] = comb;
}

// The decode-sized boundary, in two launches. Same arithmetic as tf_hc_expand_mix + tf_hc_boundary_mix: a unit's
// 32-lane sum is the old warp sum, and the boundary adds the eight units of a slice in order and then the slices.
//
// A threadgroup a (row, 128-column range) with a simdgroup for each of the four stream copies, so the collapse of the
// four copies needs no second pass over the streams: Y holds the rounded collapse for the norm.
[[kernel, max_total_threads_per_threadgroup(128)]]
void tf_hc_expand_collapse(const device bfloat* X [[buffer(0)]], const device bfloat* BR [[buffer(1)]],
                           const device float* POST [[buffer(2)]], const device float* COMB [[buffer(3)]],
                           const device float* FN [[buffer(4)]], device float* PART [[buffer(5)]],
                           device bfloat* XO [[buffer(6)]], const device float* PRE_IN [[buffer(7)]],
                           device bfloat* Y [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                           uint2 tpos [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float stream[kCopies * 128];
  const int range = int(tg.x), r = int(tg.y);
  const int k = int(sg);
  const int d0 = 128 * range + 4 * int(lane);
  const int col = k * kWidth + d0;
  // The hc_fn columns first: their read overlaps the expand's. Kept tensors pack at 256 bytes, so four floats are aligned.
  float4 f[kMixes];
  for (int q = 0; q < kMixes; q++) f[q] = *reinterpret_cast<const device float4*>(FN + size_t(q) * kStream + col);
  // Every read of the expand before any write of it: a store may alias the next read, so it would end the overlap.
  const device bfloat* x = X + size_t(r) * kStream;
  const device float* comb = COMB + r * kCopies * kCopies;
  float branch[4], old[4][kCopies];
  for (int i = 0; i < 4; i++) {
    branch[i] = float(BR[size_t(r) * kWidth + d0 + i]);
    for (int jj = 0; jj < kCopies; jj++) old[i][jj] = float(x[jj * kWidth + d0 + i]);
  }
  float mix_col[kCopies];
  for (int jj = 0; jj < kCopies; jj++) mix_col[jj] = comb[jj * kCopies + k];
  const float post = POST[r * kCopies + k];
  float v[4];
  for (int i = 0; i < 4; i++) {
    float mixed = 0.0f;
    for (int jj = 0; jj < kCopies; jj++) mixed += mix_col[jj] * old[i][jj];
    const bfloat out = bfloat(post * branch[i] + mixed);
    XO[size_t(r) * kStream + col + i] = out;
    v[i] = float(out);
    stream[k * 128 + 4 * int(lane) + i] = v[i];
  }
  float parts[kParts];
  parts[kMixes] = v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3];
  for (int q = 0; q < kMixes; q++) parts[q] = f[q].x * v[0] + f[q].y * v[1] + f[q].z * v[2] + f[q].w * v[3];
  float mine = 0.0f;
  for (int q = 0; q <= kMixes; q++) {
    const float s = simd_sum(parts[q]);
    if (int(lane) == q) mine = s;
  }
  if (int(lane) <= kMixes) PART[(size_t(r) * kUnits + k * kUnitsPerCopy + range) * kParts + lane] = mine;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = int(tpos.x);
  float collapsed = 0.0f;
  for (int s = 0; s < kCopies; s++) collapsed += PRE_IN[r * kCopies + s] * stream[s * 128 + t];
  Y[size_t(r) * kWidth + 128 * range + t] = bfloat(hc_round(collapsed));
}

// Two threadgroups a row, 1024 threads: 2r the norm of the collapse Y (written in place as the normed row), 2r+1 the
// split of the unit partials. They run side by side, so the Sinkhorn is off the norm's path.
[[kernel, max_total_threads_per_threadgroup(1024)]]
void tf_hc_boundary_rows(device bfloat* Y [[buffer(0)]], const device float* PART [[buffer(1)]],
                         const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                         const device bfloat* NORMW [[buffer(4)]], constant HcArgs& a [[buffer(5)]],
                         device float* PRE_OUT [[buffer(6)]], device float* POST_OUT [[buffer(7)]],
                         device float* COMB_OUT [[buffer(8)]], constant uint& rows [[buffer(9)]],
                         uint g [[threadgroup_position_in_grid]], uint t [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float units[kUnits * kParts];  // the partials, a mix's units together
  threadgroup float slices[kSlices * kParts];
  threadgroup float totals[kParts];
  threadgroup float sums[32];
  threadgroup float inv_rms[1];
  const uint r = g >> 1;
  if (g & 1) {
    // Each slice's eight units added in order from zero by a thread of its own, then the slices in order.
    for (uint i = t; i < uint(kUnits * kParts); i += 1024) units[(i % kParts) * kUnits + i / kParts] = PART[size_t(r) * kUnits * kParts + i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t < uint(kSlices * kParts)) {
      const uint j = t / kParts, q = t % kParts;
      float w = 0.0f;
      for (uint k = 0; k < 8; k++) w += units[q * kUnits + 8 * j + k];
      slices[t] = w;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(t) <= kMixes) {
      float s = 0.0f;
      for (int j = 0; j < kSlices; j++) s += slices[j * kParts + t];
      totals[t] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) hc_split_row<true>(&totals[0], a, BASE, SCALE, lane, r, PRE_OUT, POST_OUT, COMB_OUT);
    return;
  }
  float y[5], weight[5];
  for (int i = 0; i < 5; i++) {
    y[i] = float(Y[size_t(r) * kWidth + int(t) + 1024 * i]);
    weight[i] = float(NORMW[int(t) + 1024 * i]);
  }
  float squares = 0.0f;
  for (int i = 0; i < 5; i++) squares += y[i] * y[i];
  const float lane_squares = simd_sum(squares);
  if (lane == 0) sums[sg] = lane_squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t == 0) {
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += sums[k];
    inv_rms[0] = precise::rsqrt(s / float(kWidth) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = inv_rms[0];
  for (int i = 0; i < 5; i++) Y[size_t(r) * kWidth + int(t) + 1024 * i] = bfloat(weight[i] * (y[i] * inv));
}

// A threadgroup a row, 1024 threads: the split, collapse and norm; MIXES=false consumes a pre made elsewhere.
template <bool MIXES>
[[kernel]] void tf_hc_boundary(const device bfloat* X [[buffer(0)]], const device float* PART [[buffer(1)]],
                               const device float* BASE [[buffer(2)]], const device float* SCALE [[buffer(3)]],
                               const device float* PRE_IN [[buffer(4)]], const device bfloat* NORMW [[buffer(5)]],
                               constant HcArgs& a [[buffer(6)]], device float* PRE_OUT [[buffer(7)]],
                               device float* POST_OUT [[buffer(8)]], device float* COMB_OUT [[buffer(9)]],
                               device bfloat* NORMED [[buffer(10)]], uint r [[threadgroup_position_in_grid]],
                               uint t [[thread_position_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                               uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float sums[32];
  threadgroup float slice[kParts];
  threadgroup float pre[kCopies];
  const device bfloat* x = X + size_t(r) * kStream;
  // The slices' totals beside the lagged collapse, which overlaps the norm's reduction.
  if (MIXES && int(t) <= kMixes) {
    float s = 0.0f;
    for (int j = 0; j < kSlices; j++) s += PART[(size_t(r) * kSlices + j) * kParts + t];
    slice[t] = s;
  }
  if (t < uint(kCopies)) pre[t] = PRE_IN[r * kCopies + t];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float y[5];
  float squares = 0.0f;
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    float v = 0.0f;
    for (int s = 0; s < kCopies; s++) v += pre[s] * float(x[s * kWidth + d]);
    y[i] = hc_round(v);
    squares += y[i] * y[i];
  }
  const float lane_squares = simd_sum(squares);
  if (lane == 0) sums[sg] = lane_squares;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (MIXES && sg == 0) hc_split_row<false>(&slice[0], a, BASE, SCALE, lane, r, PRE_OUT, POST_OUT, COMB_OUT);
  if (t == 32) {  // simdgroup 1, while simdgroup 0 runs the split
    float s = 0.0f;
    for (int k = 0; k < 32; k++) s += sums[k];
    pre[0] = precise::rsqrt(s / float(kWidth) + a.eps);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float inv = pre[0];
  for (int i = 0; i < 5; i++) {
    const int d = int(t) + 1024 * i;
    NORMED[size_t(r) * kWidth + d] = bfloat(float(NORMW[d]) * (y[i] * inv));
  }
}

template [[host_name("tf_hc_boundary_mix")]] [[kernel]] decltype(tf_hc_boundary<true>) tf_hc_boundary<true>;
template [[host_name("tf_hc_boundary_plain")]] [[kernel]] decltype(tf_hc_boundary<false>) tf_hc_boundary<false>;

// The launch-overhead probe: a work-free dispatch that reads the boundary's own partial sums and writes them back.
kernel void tf_hc_probe(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                        uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const uint at = g * (kSlices * kParts) + tid;
  if (tid <= uint(kMixes)) sink[at] = part[at];
}

// The independent form of the probe: each dispatch has its own sink, so the chain carries no memory dependency.
kernel void tf_hc_probe_indep(const device float *part [[buffer(0)]], device float *sink [[buffer(1)]],
                              uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  if (tid <= uint(kMixes)) sink[tid] = part[tid];
}
