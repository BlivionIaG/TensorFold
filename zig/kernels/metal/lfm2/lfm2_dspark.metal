// DSpark's Markov step, compiled after lfm2.metal's text (its includes and constants).

// DSpark's Markov scan: a threadgroup's best of 512 tokens of a row, bf16(base + bf16(W2 . W1[prev])), W2 8-bit codes.
struct lfm2_markov_args { uint stride, rank, vocab, prev_stride; };
constant constexpr uint MARKOV_SPAN = 512;
[[kernel]] void lfm2_markov_scan(const device bfloat* logits [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                 const device uint8_t* w2 [[buffer(2)]], const device uint* prev [[buffer(3)]],
                                 constant lfm2_markov_args& p [[buffer(4)]], device float* best [[buffer(5)]],
                                 device uint* at [[buffer(6)]], const device bfloat* w2s [[buffer(7)]],
                                 const device bfloat* w2b [[buffer(8)]], uint2 tg [[threadgroup_position_in_grid]],
                                 uint2 grid [[threadgroups_per_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float e[256];
  threadgroup float sv[8];
  threadgroup uint si[8];
  const uint s = tg.y, t = sg * 32 + lane, id = prev[size_t(s) * p.prev_stride];
  for (uint i = t; i < p.rank; i += 256) e[i] = id < p.vocab ? float(w1[size_t(id) * p.rank + i]) : 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const device bfloat* row = logits + size_t(s) * p.stride;
  float mv = -INFINITY;
  uint mi = 0xffffffffu;
  bool nan = false;
  const uint first = tg.x * MARKOV_SPAN, last = min(first + MARKOV_SPAN, p.vocab);
  for (uint v = first + sg; v < last; v += 8) {
    float d = 0.0f;
    for (uint i = 8 * lane; i < p.rank; i += 256) {
      const size_t g = (size_t(v) * p.rank + i) / 64;
      const float sc = w2s[g], bi = w2b[g];
      const uchar4 lo = *(const device uchar4*)(w2 + size_t(v) * p.rank + i);
      const uchar4 hi = *(const device uchar4*)(w2 + size_t(v) * p.rank + i + 4);
      for (int j = 0; j < 4; ++j) d = fma(fma(sc, float(lo[j]), bi), e[i + j], d);
      for (int j = 0; j < 4; ++j) d = fma(fma(sc, float(hi[j]), bi), e[i + 4 + j], d);
    }
    d = simd_sum(d);
    const float sc = float(static_cast<bfloat>(float(row[v]) + float(static_cast<bfloat>(d))));
    nan = nan || isnan(sc);
    if (sc > mv || (mi == 0xffffffffu && sc == mv)) {
      mv = sc;
      mi = v;
    }
  }
  if (lane == 0) {
    sv[sg] = nan ? NAN : mv;
    si[sg] = mi;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float bv = sv[0];
  uint bi = si[0];
  for (uint g = 1; g < 8; ++g) {
    if (isnan(bv) || isnan(sv[g])) bv = NAN;
    else if (sv[g] > bv || (sv[g] == bv && si[g] < bi)) {
      bv = sv[g];
      bi = si[g];
    }
  }
  best[size_t(s) * grid.x + tg.x] = bv;
  at[size_t(s) * grid.x + tg.x] = bi;
}

// DSpark's Markov step pick: the scan's best over a stream's threadgroups (ties to the lowest token), as a Pick.
[[kernel]] void lfm2_markov_pick(const device float* best [[buffer(0)]], const device uint* at [[buffer(1)]],
                                 constant uint& groups [[buffer(2)]], device uint* picks [[buffer(3)]],
                                 uint s [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup float tv[256];
  threadgroup uint ti[256];
  threadgroup uint tn[256];
  float bv = -INFINITY;
  uint bi = 0xffffffffu, nan = 0;
  for (uint g = t; g < groups; g += 256) {
    const float v = best[size_t(s) * groups + g];
    const uint i = at[size_t(s) * groups + g];
    if (isnan(v)) nan = 1;
    else if (v > bv || (v == bv && i < bi)) {
      bv = v;
      bi = i;
    }
  }
  tv[t] = bv;
  ti[t] = bi;
  tn[t] = nan;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint w = 128; w > 0; w >>= 1) {
    if (t < w) {
      const float v = tv[t + w];
      const uint i = ti[t + w];
      if (v > tv[t] || (v == tv[t] && i < ti[t])) {
        tv[t] = v;
        ti[t] = i;
      }
      tn[t] |= tn[t + w];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (t == 0) {
    picks[2 * s] = ti[0];
    picks[2 * s + 1] = tn[0];
  }
}

// A verify window's held drafts from the drafter's picks: picks[i * stride] into ids[i], a token past the vocabulary 0.
[[kernel]] void lfm2_draft_ids(const device uint* picks [[buffer(0)]], constant uint4& p [[buffer(1)]],
                               device uint* ids [[buffer(2)]], uint i [[thread_position_in_grid]]) {
  if (i >= p.y) return;
  const uint token = picks[size_t(i) * p.x];
  ids[i] = token < p.z ? token : 0u;
}

// DSpark's confidence logit of draft i of stream s: proj . [x row s * n + i, W1[previous token]] in fp32, no bias.
struct lfm2_confidence_args { uint n, pass, vocab, rank, hidden; };
[[kernel]] void lfm2_dspark_confidence(const device bfloat* x [[buffer(0)]], const device bfloat* w1 [[buffer(1)]],
                                       const device uint* ids [[buffer(2)]], const device uint* picks [[buffer(3)]],
                                       const device bfloat* proj [[buffer(4)]],
                                       constant lfm2_confidence_args& p [[buffer(5)]], device float* out [[buffer(6)]],
                                       uint2 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  threadgroup float part[8];
  const uint i = tg.x, s = tg.y;
  // the previous token: the block's pending id for the first draft, else the Markov step's pick before it
  const uint id = i == 0 ? ids[size_t(s) * p.n] : picks[2 * (size_t(i - 1) * p.pass + s)];
  const device bfloat* row = x + (size_t(s) * p.n + i) * p.hidden;
  float d = 0.0f;
  for (uint k = t; k < p.hidden; k += 256) d = fma(float(row[k]), float(proj[k]), d);
  if (id < p.vocab)
    for (uint k = t; k < p.rank; k += 256) d = fma(float(w1[size_t(id) * p.rank + k]), float(proj[p.hidden + k]), d);
  d = simd_sum(d);
  if (lane == 0) part[sg] = d;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t != 0) return;
  float sum = 0.0f;
  for (uint g = 0; g < 8; ++g) sum += part[g];
  out[size_t(i) * p.pass + s] = sum;
}
