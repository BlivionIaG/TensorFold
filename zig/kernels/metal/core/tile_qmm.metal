// Prompt matmuls before the M5: 32x32 output tiles, exact codes and inputs in simdgroup MMAs, fp32 group and row sums.
#include <metal_stdlib>
using namespace metal;
#ifndef TQ_BM
#define TQ_BM 32
#define TQ_BN 32
#endif

// K inputs, N outputs, R rows; one is 1.0 read at run time so the compiler can't fold or reorder the sums.
struct TfTqArgs { int K, N, R; float one; };

constant constexpr int TQ = TQ_BM;  // rows a tile (the work list's row tiles)
constant constexpr int TQN = TQ_BN;  // columns a tile
constant constexpr int TP = 72;  // bf16 a staged row: 64 values, pad (bank spread)

// bf16 bits of a small exact integer (its float's top half: no rounding below 256).
inline uint tf_tq_bf(uint v) { return as_type<uint>(float(v)) >> 16; }

// 16 codes from BITS / 2 words of a little-endian bit stream as 8 words of bf16 pairs.
template <int BITS>
inline void tf_tq_codes(const thread uint* w, threadgroup uint* out) {
  #pragma unroll
  for (int i = 0; i < 8; ++i) {
    uint q[2];
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int bit = (2 * i + h) * BITS, at = bit / 32, sh = bit % 32;
      uint v = w[at] >> sh;
      if (sh + BITS > 32) v |= w[at + 1] << (32 - sh);
      q[h] = v & ((1u << BITS) - 1u);
    }
    out[i] = tf_tq_bf(q[0]) | (tf_tq_bf(q[1]) << 16);
  }
}

// One 32x32 tile of Y = X W^T: four simdgroups of 16x16, each group of 64 inputs summed by bf16 MMAs into fp32 in k order,
// then acc = fma(bias, xsum, fma(scale, group, acc)) per output; XS [R, K / 64] from tf_tile_xsum.
template <int BITS, int BM, int BN>
inline void tf_tq_tile(const device bfloat* X, const device float* XS, const device uint8_t* W, const device bfloat* SC,
                       const device bfloat* BI, int K, int N, int R, device bfloat* OUT, int rb, int nb, uint tid, uint sg,
                       uint lane, threadgroup bfloat* xs, threadgroup bfloat* ws, threadgroup float2* sb,
                       threadgroup float* xsum) {
  constexpr int TI = BM / 16, TJ = BN / 16;  // 8x8 blocks a simdgroup (2x2 simdgroups)
  const int G = K / 64;
  const int sr = int(sg) / 2 * (BM / 2), sc = int(sg) % 2 * (BN / 2);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  float acc[TI][TJ][2];
  for (int i = 0; i < TI; ++i)
    for (int j = 0; j < TJ; ++j) acc[i][j][0] = acc[i][j][1] = 0.0f;
  for (int g = 0; g < G; ++g) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = int(tid); idx < BM * 8; idx += 128) {
      const int r = idx / 8, part = idx % 8;
      *(threadgroup uint4*)(xs + r * TP + part * 8) = *((const device uint4*)(X + size_t(min(rb + r, R - 1)) * K + g * 64) + part);
    }
    for (int idx = int(tid); idx < BN * 4; idx += 128) {
      const int n = idx / 4, part = idx % 4;
      const device uint* src = (const device uint*)(W + size_t(min(nb + n, N - 1)) * (K * BITS / 8)) + (g * 4 + part) * (BITS / 2);
      uint rw[BITS / 2];
      for (int i = 0; i < BITS / 2; ++i) rw[i] = src[i];
      tf_tq_codes<BITS>(rw, (threadgroup uint*)(ws + n * TP + part * 16));
    }
    for (int idx = int(tid); idx < BN; idx += 128) {
      const size_t n = size_t(min(nb + idx, N - 1)) * G + g;
      sb[idx] = float2(float(SC[n]), float(BI[n]));
    }
    for (int idx = int(tid); idx < BM; idx += 128) xsum[idx] = XS[size_t(min(rb + idx, R - 1)) * G + g];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> P[TI][TJ];
    for (int i = 0; i < TI; ++i)
      for (int j = 0; j < TJ; ++j) P[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kk = 0; kk < 64; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> a[TI], b[TJ];
      // Each lane's two elements (row fm, columns fn and fn + 1) read directly: A along k, B down its two columns.
      for (int i = 0; i < TI; ++i) {
        const threadgroup bfloat* ap = xs + (sr + 8 * i + fm) * TP + kk + fn;
        a[i].thread_elements()[0] = ap[0];
        a[i].thread_elements()[1] = ap[1];
      }
      for (int j = 0; j < TJ; ++j) {
        const threadgroup bfloat* bp = ws + (sc + 8 * j + fn) * TP + kk + fm;
        b[j].thread_elements()[0] = bp[0];
        b[j].thread_elements()[1] = bp[TP];
      }
      for (int i = 0; i < TI; ++i)
        for (int j = 0; j < TJ; ++j) simdgroup_multiply_accumulate(P[i][j], a[i], b[j], P[i][j]);
    }
    float xsr[TI];
    for (int i = 0; i < TI; ++i) xsr[i] = xsum[sr + 8 * i + fm];
    for (int j = 0; j < TJ; ++j)
      for (int e = 0; e < 2; ++e) {
        const float2 v = sb[sc + 8 * j + fn + e];
        for (int i = 0; i < TI; ++i) acc[i][j][e] = fma(v.y, xsr[i], fma(v.x, P[i][j].thread_elements()[e], acc[i][j][e]));
      }
  }
  for (int i = 0; i < TI; ++i)
    for (int j = 0; j < TJ; ++j)
      for (int e = 0; e < 2; ++e) {
        const int row = rb + sr + 8 * i + fm, n = nb + sc + 8 * j + fn + e;
        if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[i][j][e]);
      }
}

// Y [R, N] = X [R, K] W^T, a 32x32 tile a threadgroup of 128 (tg.x columns, tg.y rows); XS at buffer 6.
template <int BITS>
[[kernel]] void tf_tile_qmm(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]], const device float* XS [[buffer(6)]],
                            uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[TQ * TP], ws[TQN * TP];
  threadgroup float2 sb[TQN];
  threadgroup float xsum[TQ];
  tf_tq_tile<BITS, TQ, TQN>(X, XS, W, SC, BI, a.K, a.N, a.R, OUT, int(tg.y) * TQ, int(tg.x) * TQN, tid, sg, lane, xs, ws, sb, xsum);
}

// The tiles over rows sorted by expert: item tg.y of WORK[1..WORK[0]] = (expert << 16) | row tile, rows from STARTS.
template <int BITS>
[[kernel]] void tf_tile_qmm_grouped(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* starts [[buffer(6)]], const device uint* work [[buffer(7)]],
                                    const device float* XS [[buffer(8)]], uint tid [[thread_index_in_threadgroup]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  threadgroup bfloat xs[TQ * TP], ws[TQN * TP];
  threadgroup float2 sb[TQN];
  threadgroup float xsum[TQ];
  if (tg.y >= work[0]) return;
  const uint item = work[1 + tg.y], e = item >> 16;
  const uint first = starts[e], rows = starts[e + 1] - first;
  const size_t K = size_t(a.K), N = size_t(a.N);
  tf_tq_tile<BITS, TQ, TQN>(X + first * K, XS + first * (K / 64), W + e * N * (K * BITS / 8), SC + e * N * (K / 64), BI + e * N * (K / 64),
                   a.K, a.N, int(rows), OUT + first * N, int(item & 0xFFFFu) * TQ, int(tg.x) * TQN, tid, sg, lane, xs, ws, sb, xsum);
}

// The grouped tiles' work list from STARTS [E + 1]: count, then (expert << 16) | tile for each expert's 32-row tiles.
[[kernel]] void tf_tile_qmm_work(const device uint* starts [[buffer(0)]], constant uint& experts [[buffer(1)]],
                                 device uint* work [[buffer(2)]]) {
  uint n = 0;
  for (uint e = 0; e < experts; ++e) {
    const uint tiles = (starts[e + 1] - starts[e] + TQ - 1) / TQ;
    for (uint i = 0; i < tiles; ++i) work[1 + n++] = (e << 16) | i;
  }
  work[0] = n;
}

// Each row's group sums for the twin [rows, K / 64]: x0 then + x1 ... + x63 in order, as the tiles' xsum.
[[kernel]] void tf_tile_xsum(const device bfloat* X [[buffer(0)]], constant TfTqArgs& a [[buffer(1)]], device float* XS [[buffer(2)]],
                             uint2 at [[thread_position_in_grid]]) {
  const int G = a.K / 64;
  if (int(at.x) >= G) return;
  const device bfloat* x = X + size_t(at.y) * a.K + at.x * 64;
  float t = float(x[0]);
  for (int k = 1; k < 64; ++k) t = fma(float(x[k]), a.one, t);
  XS[size_t(at.y) * G + at.x] = t;
}

// Code k of a group's little-endian bit stream (words w) as a float; k is a constant after unrolling.
template <int BITS>
inline float tf_tq_code(const thread uint* w, int k) {
  const int bit = k * BITS, at = bit / 32, sh = bit % 32;
  uint v = w[at] >> sh;
  if (sh + BITS > 32) v |= w[at + 1] << (32 - sh);
  return float(v & ((1u << BITS) - 1u));
}

// Columns a simdgroup and simdgroups a threadgroup for the twin.
// Columns a simdgroup (fewer at 6 and 8 bits, whose words take more registers) and simdgroups a threadgroup.
template <int BITS> struct TfTqCols { static constant constexpr int value = 4; };
constant constexpr int TQ_SG = 2;

// Group g's 64-step fma chains (the tiles' MMA order) of x's 64 inputs against NC columns' codes, `stride` bytes apart,
// and the group's input sum. At 4 bits a code stays in place (code * 16^j) against x * 16^-j: the same exact product.
template <int BITS, int NC>
inline float tf_tq_group(const device bfloat* x, const device uint8_t* w, size_t stride, int g, thread float* p, float one) {
  constexpr int GW = 2 * BITS;  // words a group of 64 codes
  float t = 0.0f;
  for (int c = 0; c < NC; ++c) p[c] = 0.0f;
  uint wq[BITS == 4 ? NC : 1][8];  // 4 bits: each column's group, two 16-byte loads
  if (BITS == 4)
    #pragma unroll
    for (int c = 0; c < NC; ++c) {
      const device uint4* src = (const device uint4*)(w + c * stride + size_t(g) * (GW * 4));
      const uint4 a = src[0], b = src[1];
      wq[c][0] = a.x; wq[c][1] = a.y; wq[c][2] = a.z; wq[c][3] = a.w;
      wq[c][4] = b.x; wq[c][5] = b.y; wq[c][6] = b.z; wq[c][7] = b.w;
    }
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 xv = ((const device uint4*)(x + size_t(g) * 64))[kb];
    const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
    float xf[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
      t = kb == 0 && j == 0 ? v : fma(v, one, t);
      xf[j] = BITS == 4 ? v * as_type<float>(uint(127 - 4 * j) << 23) : v;
    }
    if (BITS == 4) {
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const uint word = wq[c][kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[c] = fma(xf[j], float(word & (0xFu << (4 * j))), p[c]);
      }
    } else {
      // Codes 8 kb.. of this group sit in the two words from bit 8 kb BITS (48 or 64 bits).
      const int bit = kb * 8 * BITS, at = bit / 32, sh = bit % 32;
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const device uint* wc = (const device uint*)(w + c * stride + size_t(g) * (GW * 4)) + at;
        const ulong v = ((sh + 8 * BITS > 64 ? ulong(wc[2]) << 32 : 0ul) | 0ul);
        const ulong lo = (ulong(wc[1]) << 32) | wc[0];
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
          const int o = sh + BITS * j;
          const uint q = o + BITS <= 64 ? uint(lo >> o) : uint((lo >> o) | (v << (64 - o)));
          p[c] = fma(xf[j], float(q & ((1u << BITS) - 1u)), p[c]);
        }
      }
    }
  }
  return t;
}

// NC columns of one row at the tiles' bits: lanes take groups (staging each group's partials and input sum),
// then lanes 0..NC-1 chain a column's groups in order with its scales and biases.
template <int BITS, int NC>
inline float tf_tq_columns(const device bfloat* x, float one, const device uint8_t* w, const device bfloat* sc,
                           const device bfloat* bi, int K, uint lane, threadgroup float* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  threadgroup float* sums = part + NC * G;
  for (int g = int(lane); g < G; g += 32) {
    float p[NC];
    sums[g] = tf_tq_group<BITS, NC>(x, w, stride, g, p, one);
    for (int c = 0; c < NC; ++c) part[c * G + g] = p[c];
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int c = min(int(lane), NC - 1);
  const device bfloat* s = sc + c * G;
  const device bfloat* b = bi + c * G;
  const threadgroup float* pc = part + c * G;
  float acc = 0.0f;
  #pragma unroll(8)
  for (int g = 0; g < G; ++g) acc = fma(float(b[g]), sums[g], fma(float(s[g]), pc[g], acc));
  simdgroup_barrier(mem_flags::mem_threadgroup);
  return acc;
}

// mlx-lm's compiled swiglu on bf16 inputs: silu(g) * u, rounded to bf16 at each step.
inline bfloat tf_tq_swiglu(bfloat gate, bfloat up) {
  const float g = gate;
  const bfloat s = static_cast<bfloat>(1.0f / (1.0f + metal::precise::exp(-g)));
  const bfloat silu = static_cast<bfloat>(g * float(s));
  return static_cast<bfloat>(float(silu) * float(up));
}

// Y [rows, N] = X W^T at the tiles' bits (decode and windows): NC columns a simdgroup, TQ_SG a threadgroup; tg.y the row.
template <int BITS>
[[kernel]] void tf_tile_qmv(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                            threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, r = int(tg.y), K = a.K, G = K / 64;
  if (n >= a.N) return;
  const float y = tf_tq_columns<BITS, NC>(X + size_t(r) * K, a.one, W + size_t(n) * (K * BITS / 8), SC + size_t(n) * G,
                                         BI + size_t(n) * G, K, lane, part + sg * (NC + 1) * G);
  if (int(lane) < NC) OUT[size_t(r) * a.N + n + lane] = bfloat(y);
}

// The twin by slot (tg.y): expert ids[slot]'s matrix on x row slot / a.R, out row slot.
template <int BITS>
[[kernel]] void tf_tile_qmv_indexed(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* ids [[buffer(6)]], threadgroup float* part [[threadgroup(0)]],
                                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                                    uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  const float y = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane,
                                         part + sg * (NC + 1) * G);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = bfloat(y);
}

// Gate and up by slot at the tiles' bits (W/SC/BI and W2/SC2/BI2, the same shape), out = swiglu of their bf16 rows.
template <int BITS>
[[kernel]] void tf_tile_qmv_glu(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                const device uint* ids [[buffer(6)]], const device uint8_t* W2 [[buffer(7)]],
                                const device bfloat* SC2 [[buffer(8)]], const device bfloat* BI2 [[buffer(9)]],
                                threadgroup float* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  threadgroup float* own = part + sg * (NC + 1) * G;
  const float g = tf_tq_columns<BITS, NC>(X + row * K, a.one, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K, lane, own);
  const float u = tf_tq_columns<BITS, NC>(X + row * K, a.one, W2 + m * (K * BITS / 8), SC2 + m * G, BI2 + m * G, K, lane, own);
  if (int(lane) < NC) OUT[size_t(tg.y) * a.N + n + lane] = tf_tq_swiglu(bfloat(g), bfloat(u));
}

template [[host_name("tf_tile_qmm_q4")]] [[kernel]] decltype(tf_tile_qmm<4>) tf_tile_qmm<4>;
template [[host_name("tf_tile_qmm_q6")]] [[kernel]] decltype(tf_tile_qmm<6>) tf_tile_qmm<6>;
template [[host_name("tf_tile_qmm_q8")]] [[kernel]] decltype(tf_tile_qmm<8>) tf_tile_qmm<8>;
template [[host_name("tf_tile_qmm_grouped_q4")]] [[kernel]] decltype(tf_tile_qmm_grouped<4>) tf_tile_qmm_grouped<4>;
template [[host_name("tf_tile_qmm_grouped_q6")]] [[kernel]] decltype(tf_tile_qmm_grouped<6>) tf_tile_qmm_grouped<6>;
template [[host_name("tf_tile_qmm_grouped_q8")]] [[kernel]] decltype(tf_tile_qmm_grouped<8>) tf_tile_qmm_grouped<8>;
template [[host_name("tf_tile_qmv_q4")]] [[kernel]] decltype(tf_tile_qmv<4>) tf_tile_qmv<4>;
template [[host_name("tf_tile_qmv_q6")]] [[kernel]] decltype(tf_tile_qmv<6>) tf_tile_qmv<6>;
template [[host_name("tf_tile_qmv_q8")]] [[kernel]] decltype(tf_tile_qmv<8>) tf_tile_qmv<8>;
template [[host_name("tf_tile_qmv_indexed_q4")]] [[kernel]] decltype(tf_tile_qmv_indexed<4>) tf_tile_qmv_indexed<4>;
template [[host_name("tf_tile_qmv_indexed_q6")]] [[kernel]] decltype(tf_tile_qmv_indexed<6>) tf_tile_qmv_indexed<6>;
template [[host_name("tf_tile_qmv_indexed_q8")]] [[kernel]] decltype(tf_tile_qmv_indexed<8>) tf_tile_qmv_indexed<8>;
template [[host_name("tf_tile_qmv_glu_q4")]] [[kernel]] decltype(tf_tile_qmv_glu<4>) tf_tile_qmv_glu<4>;
template [[host_name("tf_tile_qmv_glu_q6")]] [[kernel]] decltype(tf_tile_qmv_glu<6>) tf_tile_qmv_glu<6>;
template [[host_name("tf_tile_qmv_glu_q8")]] [[kernel]] decltype(tf_tile_qmv_glu<8>) tf_tile_qmv_glu<8>;
