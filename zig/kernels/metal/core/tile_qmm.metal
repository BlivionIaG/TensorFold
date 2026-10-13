// Prompt matmuls before the M5: 32x32 output tiles, exact codes and inputs in simdgroup MMAs, fp32 group and row sums.
#include <metal_stdlib>
using namespace metal;

// K inputs, N outputs, R rows; one is 1.0 read at run time so the compiler can't fold or reorder the sums.
struct TfTqArgs { int K, N, R; float one; };

constant constexpr int TQ = 32;  // rows and columns a tile
constant constexpr int TP = 72;  // bf16 a staged row: 64 values, pad (bank spread)

// bf16 bits of a small exact integer (its float's top half: no rounding below 256).
inline uint tf_tq_bf(uint v) { return as_type<uint>(float(v)) >> 16; }

// 32 codes from `BITS` words of a little-endian bit stream as 16 words of bf16 pairs.
template <int BITS>
inline void tf_tq_codes(const thread uint* w, threadgroup uint* out) {
  #pragma unroll
  for (int i = 0; i < 16; ++i) {
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
template <int BITS>
inline void tf_tq_tile(const device bfloat* X, const device float* XS, const device uint8_t* W, const device bfloat* SC,
                       const device bfloat* BI, int K, int N, int R, device bfloat* OUT, int rb, int nb, uint tid, uint sg,
                       uint lane, threadgroup bfloat* xs, threadgroup bfloat* ws, threadgroup float2* sb) {
  const int G = K / 64;
  const int sr = int(sg) / 2 * 16, sc = int(sg) % 2 * 16;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  // Staging: thread t moves x row t / 4's 16 inputs (t % 4) twice over, and threads < 64 one column half's 32 codes.
  const int xr = int(tid) / 8, xp = int(tid) % 8;
  const device bfloat* xa = X + size_t(min(rb + xr, R - 1)) * K + xp * 8;
  const device bfloat* xb = X + size_t(min(rb + xr + 16, R - 1)) * K + xp * 8;
  const int wn = int(tid) / 2, wp = int(tid) % 2;
  const device uint* wsrc = (const device uint*)(W + size_t(min(nb + wn, N - 1)) * (K * BITS / 8)) + wp * BITS;
  const size_t sbn = size_t(min(nb + int(tid), N - 1)) * G;
  uint4 ra, rb2;
  uint rw[BITS];
  float2 rs;
  auto fetch = [&](int g) {
    ra = *(const device uint4*)(xa + g * 64);
    rb2 = *(const device uint4*)(xb + g * 64);
    if (tid < 64)
      for (int i = 0; i < BITS; ++i) rw[i] = wsrc[g * 2 * BITS + i];
    if (tid < TQ) rs = float2(float(SC[sbn + g]), float(BI[sbn + g]));
  };
  auto store = [&](int buf) {
    threadgroup bfloat* x = xs + buf * TQ * TP;
    *(threadgroup uint4*)(x + xr * TP + xp * 8) = ra;
    *(threadgroup uint4*)(x + (xr + 16) * TP + xp * 8) = rb2;
    if (tid < 64) tf_tq_codes<BITS>(rw, (threadgroup uint*)(ws + buf * TQ * TP + wn * TP + wp * 32));
    if (tid < TQ) sb[buf * TQ + tid] = rs;
  };
  float acc[2][2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j][0] = acc[i][j][1] = 0.0f;
  for (int g = 0; g < G; ++g) {
    const int buf = 0;
    fetch(g);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    store(0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const threadgroup bfloat* x = xs + buf * TQ * TP;
    const threadgroup bfloat* w = ws + buf * TQ * TP;
    simdgroup_matrix<float, 8, 8> P[2][2];
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) P[i][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int kk = 0; kk < 64; kk += 8) {
      simdgroup_matrix<bfloat, 8, 8> a[2], b[2];
      for (int i = 0; i < 2; ++i) simdgroup_load(a[i], x + (sr + 8 * i) * TP + kk, TP);
      for (int j = 0; j < 2; ++j) simdgroup_load(b[j], w + (sc + 8 * j) * TP + kk, TP, ulong2(0, 0), true);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(P[i][j], a[i], b[j], P[i][j]);
    }
    float xsr[2];
    for (int i = 0; i < 2; ++i) xsr[i] = XS[size_t(min(rb + sr + 8 * i + fm, R - 1)) * G + g];
    for (int j = 0; j < 2; ++j)
      for (int e = 0; e < 2; ++e) {
        const float2 v = sb[buf * TQ + sc + 8 * j + fn + e];
        for (int i = 0; i < 2; ++i) acc[i][j][e] = fma(v.y, xsr[i], fma(v.x, P[i][j].thread_elements()[e], acc[i][j][e]));
      }
  }
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
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
  threadgroup bfloat xs[TQ * TP], ws[TQ * TP];
  threadgroup float2 sb[TQ];
  tf_tq_tile<BITS>(X, XS, W, SC, BI, a.K, a.N, a.R, OUT, int(tg.y) * TQ, int(tg.x) * TQ, tid, sg, lane, xs, ws, sb);
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
  threadgroup bfloat xs[TQ * TP], ws[TQ * TP];
  threadgroup float2 sb[TQ];
  if (tg.y >= work[0]) return;
  const uint item = work[1 + tg.y], e = item >> 16;
  const uint first = starts[e], rows = starts[e + 1] - first;
  const size_t K = size_t(a.K), N = size_t(a.N);
  tf_tq_tile<BITS>(X + first * K, XS + first * (K / 64), W + e * N * (K * BITS / 8), SC + e * N * (K / 64), BI + e * N * (K / 64),
                   a.K, a.N, int(rows), OUT + first * N, int(item & 0xFFFFu) * TQ, int(tg.x) * TQ, tid, sg, lane, xs, ws, sb);
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
template <int BITS> struct TfTqCols { static constant constexpr int value = BITS == 4 ? 8 : 4; };
constant constexpr int TQ_SG = 2;

// Group g's 64-step fma chains (the tiles' MMA order) of x's 64 inputs against TQ_NC columns' codes, `stride` bytes apart.
// At 4 bits a code stays in place (code * 16^s) against x * 16^-s: the same exact product, two operations fewer.
template <int BITS, int NC>
inline void tf_tq_group(const device bfloat* x, const device uint8_t* w, size_t stride, int g, thread float* p) {
  constexpr int GW = 2 * BITS;  // words a group of 64 codes
  for (int c = 0; c < NC; ++c) p[c] = 0.0f;
  #pragma unroll
  for (int kb = 0; kb < 8; ++kb) {
    const uint4 xv = ((const device uint4*)(x + size_t(g) * 64))[kb];
    const uint xw[4] = {xv.x, xv.y, xv.z, xv.w};
    float xf[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      const float v = as_type<float>((j % 2) ? (xw[j / 2] & 0xFFFF0000u) : (xw[j / 2] << 16));
      xf[j] = BITS == 4 ? v * as_type<float>(uint(127 - 4 * j) << 23) : v;
    }
    if (BITS == 4) {
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const uint word = ((const device uint*)(w + c * stride + size_t(g) * (GW * 4)))[kb];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[c] = fma(xf[j], float(word & (0xFu << (4 * j))), p[c]);
      }
    } else {
      #pragma unroll
      for (int c = 0; c < NC; ++c) {
        const device uint* wc = (const device uint*)(w + c * stride + size_t(g) * (GW * 4));
        uint wv[GW];
        #pragma unroll
        for (int i = 0; i < GW; ++i) wv[i] = wc[i];
        #pragma unroll
        for (int j = 0; j < 8; ++j) p[c] = fma(xf[j], tf_tq_code<BITS>(wv, kb * 8 + j), p[c]);
      }
    }
  }
}

// NC columns of one row at the tiles' bits: lanes take groups (staging each group's partial, scale, bias and input sum),
// then lanes 0..NC-1 chain a column's groups in order.
template <int BITS, int NC>
inline void tf_tq_columns(const device bfloat* x, const device float* xs, const device uint8_t* w, const device bfloat* sc,
                          const device bfloat* bi, int K, device bfloat* out, uint lane, threadgroup float4* part) {
  const int G = K / 64;
  const size_t stride = size_t(K) * BITS / 8;
  for (int g = int(lane); g < G; g += 32) {
    float p[NC];
    tf_tq_group<BITS, NC>(x, w, stride, g, p);
    const float t = xs[g];
    for (int c = 0; c < NC; ++c) part[c * G + g] = float4(p[c], float(sc[c * G + g]), float(bi[c * G + g]), t);
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if (int(lane) >= NC) return;
  const int c = int(lane);
  float acc = 0.0f;
  for (int g = 0; g < G; ++g) {
    const float4 v = part[c * G + g];
    acc = fma(v.z, v.w, fma(v.y, v.x, acc));
  }
  out[c] = bfloat(acc);
}

// Y [rows, N] = X W^T at the tiles' bits (decode and windows): TQ_NC columns a simdgroup, TQ_SG a threadgroup; tg.y the row.
template <int BITS>
[[kernel]] void tf_tile_qmv(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                            const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                            constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]], const device float* XS [[buffer(6)]],
                            threadgroup float4* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, r = int(tg.y), K = a.K, G = K / 64;
  if (n >= a.N) return;
  tf_tq_columns<BITS, NC>(X + size_t(r) * K, XS + size_t(r) * G, W + size_t(n) * (K * BITS / 8), SC + size_t(n) * G, BI + size_t(n) * G,
                          K, OUT + size_t(r) * a.N + n, lane, part + sg * NC * G);
}

// The twin by slot (tg.y): expert ids[slot]'s matrix on x row slot / a.R, out row slot.
template <int BITS>
[[kernel]] void tf_tile_qmv_indexed(const device bfloat* X [[buffer(0)]], const device uint8_t* W [[buffer(1)]],
                                    const device bfloat* SC [[buffer(2)]], const device bfloat* BI [[buffer(3)]],
                                    constant TfTqArgs& a [[buffer(4)]], device bfloat* OUT [[buffer(5)]],
                                    const device uint* ids [[buffer(6)]], const device float* XS [[buffer(7)]],
                                    threadgroup float4* part [[threadgroup(0)]], uint sg [[simdgroup_index_in_threadgroup]],
                                    uint lane [[thread_index_in_simdgroup]], uint3 tg [[threadgroup_position_in_grid]]) {
  constexpr int NC = TfTqCols<BITS>::value;
  const int n = (int(tg.x) * TQ_SG + int(sg)) * NC, K = a.K, G = K / 64;
  if (n >= a.N) return;
  const size_t e = ids[tg.y], row = tg.y / uint(a.R), m = e * size_t(a.N) + n;
  tf_tq_columns<BITS, NC>(X + row * K, XS + row * G, W + m * (K * BITS / 8), SC + m * G, BI + m * G, K,
                          OUT + size_t(tg.y) * a.N + n, lane, part + sg * NC * G);
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
