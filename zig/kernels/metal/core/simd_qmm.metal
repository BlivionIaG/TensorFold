// Row-exact 4-bit projections before the M5: 8-row simdgroup-matrix tiles, and a scalar twin for 1-2 rows.
#include <metal_stdlib>
using namespace metal;

// K inputs, N outputs, R rows; one is 1.0 read at run time so the compiler can't fold or reorder the fma sums.
struct TfSqArgs { int K, N, R; float one; };

// The bf16 at index e (0..7) of 8 packed bf16, as fp32.
inline float tf_sq_bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}

// A row's 8 inputs summed left to right.
inline float tf_sq_sum8(uint4 v, float one) {
  float t = tf_sq_bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(tf_sq_bf8(v, e), one, t);
  return t;
}

// 2^-4s: inputs pre-scaled so a nibble masked in place (code * 2^4s) multiplies to the exact product.
inline float tf_sq_pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }

// R rows in 8-row tiles, NT 8-column tiles; simdgroups take K chunks c, c + SGS, ..., summed by a fixed tree.
template <int S, int NT, int RT>
inline void tf_sq_tiles(const device uint4* X, const device uint* W, const device bfloat* SC, const device bfloat* BI,
                        int K, int N, int R, float one, device bfloat* OUT, uint sgi, uint sgs, uint lane, uint3 tg,
                        threadgroup float* red) {
  const int G = K / 64, SGS = int(sgs);
  const int sg = int(sgi);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int nb = int(tg.x) * (8 * NT);
  const int rb = int(tg.y) * (8 * RT);
  const device uint2* W2 = (const device uint2*)W;
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) {
    xr0[rt] = min(rb + 8 * rt + fn, R - 1);
    xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1);
  }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      uint2 wv[NT];
      for (int t = 0; t < NT; t++) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = X[size_t(xr0[rt]) * (K / 8) + 8 * g + fm];
        xb[rt] = X[size_t(xr1[rt]) * (K / 8) + 8 * g + fm];
        float v = tf_sq_sum8(xa[rt], one), u = tf_sq_sum8(xb[rt], one);
        v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = v; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int s = 0; s < 8; s++) {
        const float ps = tf_sq_pre(s);
        const uint mask = 0xFu << (4 * s);
        simdgroup_matrix<float, 8, 8> bm[RT];
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = tf_sq_bf8(xa[rt], s) * ps;
          bm[rt].thread_elements()[1] = tf_sq_bf8(xb[rt], s) * ps;
        }
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = float(wv[t].x & mask);
          am.thread_elements()[1] = float(wv[t].y & mask);
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++)
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
}

template <int S, int NT, int RT>
[[kernel]] void tf_sq_mma(const device uint4* X [[buffer(0)]],
                          const device uint* W [[buffer(1)]],
                          const device bfloat* SC [[buffer(2)]],
                          const device bfloat* BI [[buffer(3)]],
                          constant TfSqArgs& a [[buffer(4)]],
                          device bfloat* OUT [[buffer(5)]],
                          uint sgi [[simdgroup_index_in_threadgroup]],
                          uint sgs [[simdgroups_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  tf_sq_tiles<S, NT, RT>(X, W, SC, BI, a.K, a.N, a.R, a.one, OUT, sgi, sgs, lane, tg, red);
}

// The tiles over rows sorted by expert: work item tg.y = (expert << 16) | row tile of WORK[1..WORK[0]], as tf_sq_mma.
template <int S, int NT, int RT>
[[kernel]] void tf_sq_mma_grouped(const device uint4* X [[buffer(0)]],
                                  const device uint* W [[buffer(1)]],
                                  const device bfloat* SC [[buffer(2)]],
                                  const device bfloat* BI [[buffer(3)]],
                                  constant TfSqArgs& a [[buffer(4)]],
                                  device bfloat* OUT [[buffer(5)]],
                                  const device uint* starts [[buffer(6)]],
                                  const device uint* work [[buffer(7)]],
                                  uint sgi [[simdgroup_index_in_threadgroup]],
                                  uint sgs [[simdgroups_per_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  if (tg.y >= work[0]) return;
  const uint item = work[1 + tg.y], e = item >> 16;
  const uint first = starts[e], rows = starts[e + 1] - first;
  const size_t K = size_t(a.K), N = size_t(a.N);
  tf_sq_tiles<S, NT, RT>(X + first * (K / 8), W + e * N * (K / 8), SC + e * N * (K / 64), BI + e * N * (K / 64), a.K, a.N,
                         int(rows), a.one, OUT + first * N, sgi, sgs, lane, uint3(tg.x, item & 0xFFFFu, 0), red);
}

// The grouped tiles' work list from STARTS: count, then (expert << 16) | tile for each expert's ceil(rows / (8 RT)) tiles.
[[kernel]] void tf_sq_group_work(const device uint* starts [[buffer(0)]], constant uint2& er [[buffer(1)]], device uint* work [[buffer(2)]],
                                 uint t [[thread_position_in_grid]]) {
  if (t != 0) return;
  uint n = 0;
  for (uint e = 0; e < er.x; ++e) {
    const uint tiles = (starts[e + 1] - starts[e] + 8 * er.y - 1) / (8 * er.y);
    for (uint i = 0; i < tiles; ++i) work[1 + n++] = (e << 16) | i;
  }
  work[0] = n;
}

// 1-2 rows over the same weight loads: each lane runs one chunk of NR outputs with the MMA kernel's chains.
constant constexpr int TF_SQ_XP = 76;  // floats a staged group: 64 inputs, 8 word sums, pad (bank spread)

template <int S, int NR, int XB, int RS>
inline void tf_sq_twin(const device uint4* X, const device uint* W, const device bfloat* SC, const device bfloat* BI,
                       int K, int N, float one, device bfloat* OUT, uint sgi, uint sgs, uint lane, uint3 tg,
                       threadgroup float* xs) {
  constexpr int XP = TF_SQ_XP;
  constexpr int SLOTS = 32 / S;
  const int G = K / 64, SGS = int(sgs);
  const int tid = int(sgi) * 32 + int(lane);
  const int c = int(lane) % S;
  const int n0 = (int(tg.x) * SGS + int(sgi)) * (SLOTS * NR) + int(lane) / S;
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint4 nw[NR][2];
  for (int u = 0; u < NR; u++)
    for (int h = 0; h < 2; h++) nw[u][h] = c < G ? wr[u][2 * c + h] : uint4(0);
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * 8; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * 8);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * 8)) / 8, j = idx % 8;
      const uint4 v = X[size_t(r) * (K / 8) + 8 * (b0 + gl) + j];
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      for (int e = 0; e < 8; e++) xr[8 * e + j] = tf_sq_bf8(v, e) * tf_sq_pre(e);
      xr[64 + j] = tf_sq_sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wv[NR][2];
      for (int u = 0; u < NR; u++)
        for (int h = 0; h < 2; h++) wv[u][h] = nw[u][h];
      if (g + S < G)
        for (int u = 0; u < NR; u++)
          for (int h = 0; h < 2; h++) nw[u][h] = wr[u][2 * (g + S) + h];
      float xsum[RS];
      float P[NR][RS];
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + 64);
        const float4 p1 = *(const threadgroup float4*)(xg + 68);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      for (int s = 0; s < 8; s++) {
        float xq[RS][8];
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        for (int u = 0; u < NR; u++)
          for (int i = 0; i < 8; i++) {
            const float q = float(wv[u][i / 4][i % 4] & (0xFu << (4 * s)));
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][i], q, P[u][r]);
          }
      }
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  for (int u = 0; u < NR; u++)
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
}

template <int S, int NR, int XB, int RS>
[[kernel]] void tf_sq_scalar(const device uint4* X [[buffer(0)]],
                             const device uint* W [[buffer(1)]],
                             const device bfloat* SC [[buffer(2)]],
                             const device bfloat* BI [[buffer(3)]],
                             constant TfSqArgs& a [[buffer(4)]],
                             device bfloat* OUT [[buffer(5)]],
                             uint sgi [[simdgroup_index_in_threadgroup]],
                             uint sgs [[simdgroups_per_threadgroup]],
                             uint lane [[thread_index_in_simdgroup]],
                             uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float xs[RS * XB * TF_SQ_XP];
  tf_sq_twin<S, NR, XB, RS>(X, W, SC, BI, a.K, a.N, a.one, OUT, sgi, sgs, lane, tg, xs);
}

// The twin on one row a slot (tg.y): expert ids[slot]'s matrix, x row slot / a.R, out row slot.
template <int S, int NR, int XB>
[[kernel]] void tf_sq_scalar_indexed(const device uint4* X [[buffer(0)]],
                                     const device uint* W [[buffer(1)]],
                                     const device bfloat* SC [[buffer(2)]],
                                     const device bfloat* BI [[buffer(3)]],
                                     constant TfSqArgs& a [[buffer(4)]],
                                     device bfloat* OUT [[buffer(5)]],
                                     const device uint* ids [[buffer(6)]],
                                     uint sgi [[simdgroup_index_in_threadgroup]],
                                     uint sgs [[simdgroups_per_threadgroup]],
                                     uint lane [[thread_index_in_simdgroup]],
                                     uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup float xs[XB * TF_SQ_XP];
  const size_t K = size_t(a.K), N = size_t(a.N), e = ids[tg.y];
  tf_sq_twin<S, NR, XB, 1>(X + size_t(tg.y / uint(a.R)) * (K / 8), W + e * N * (K / 8), SC + e * N * (K / 64), BI + e * N * (K / 64),
                           a.K, a.N, a.one, OUT + size_t(tg.y) * N, sgi, sgs, lane, uint3(tg.x, 0, 0), xs);
}

#define TF_SQ_MMA(S, NT, RT) template [[host_name("tf_sq_mma_" #S "_" #NT "_" #RT)]] [[kernel]] decltype(tf_sq_mma<S, NT, RT>) tf_sq_mma<S, NT, RT>;
TF_SQ_MMA(8, 1, 1) TF_SQ_MMA(8, 2, 1) TF_SQ_MMA(8, 4, 1) TF_SQ_MMA(8, 1, 2) TF_SQ_MMA(8, 2, 2) TF_SQ_MMA(8, 4, 2)
TF_SQ_MMA(16, 1, 1) TF_SQ_MMA(16, 2, 1) TF_SQ_MMA(16, 4, 1) TF_SQ_MMA(16, 1, 2) TF_SQ_MMA(16, 2, 2)
TF_SQ_MMA(32, 1, 1) TF_SQ_MMA(32, 2, 1) TF_SQ_MMA(32, 1, 2)
#define TF_SQ_GROUPED(S, NT, RT) template [[host_name("tf_sq_grouped_" #S "_" #NT "_" #RT)]] [[kernel]] decltype(tf_sq_mma_grouped<S, NT, RT>) tf_sq_mma_grouped<S, NT, RT>;
TF_SQ_GROUPED(8, 1, 1) TF_SQ_GROUPED(8, 2, 1) TF_SQ_GROUPED(8, 4, 1) TF_SQ_GROUPED(8, 1, 2) TF_SQ_GROUPED(8, 2, 2) TF_SQ_GROUPED(8, 4, 2)
TF_SQ_GROUPED(16, 1, 1) TF_SQ_GROUPED(16, 2, 1) TF_SQ_GROUPED(16, 4, 1) TF_SQ_GROUPED(16, 1, 2) TF_SQ_GROUPED(16, 2, 2)
TF_SQ_GROUPED(32, 1, 1) TF_SQ_GROUPED(32, 2, 1) TF_SQ_GROUPED(32, 1, 2)
#define TF_SQ_SCALAR(S, NR, XB, RS) template [[host_name("tf_sq_scalar_" #S "_" #NR "_" #RS)]] [[kernel]] decltype(tf_sq_scalar<S, NR, XB, RS>) tf_sq_scalar<S, NR, XB, RS>;
TF_SQ_SCALAR(8, 1, 32, 1) TF_SQ_SCALAR(8, 2, 32, 1) TF_SQ_SCALAR(8, 1, 16, 2) TF_SQ_SCALAR(8, 2, 16, 2)
TF_SQ_SCALAR(16, 1, 32, 1) TF_SQ_SCALAR(16, 2, 32, 1) TF_SQ_SCALAR(16, 1, 16, 2) TF_SQ_SCALAR(16, 2, 16, 2)
TF_SQ_SCALAR(32, 1, 32, 1) TF_SQ_SCALAR(32, 2, 32, 1) TF_SQ_SCALAR(32, 1, 32, 2) TF_SQ_SCALAR(32, 2, 32, 2)
#define TF_SQ_INDEXED(S, NR, XB) template [[host_name("tf_sq_indexed_" #S "_" #NR)]] [[kernel]] decltype(tf_sq_scalar_indexed<S, NR, XB>) tf_sq_scalar_indexed<S, NR, XB>;
TF_SQ_INDEXED(8, 1, 32) TF_SQ_INDEXED(8, 2, 32) TF_SQ_INDEXED(16, 1, 32) TF_SQ_INDEXED(16, 2, 32) TF_SQ_INDEXED(32, 1, 32) TF_SQ_INDEXED(32, 2, 32)
