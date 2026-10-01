
  // qmv_rows' per-row arithmetic for several rows on the matrix units. Threadgroup (i, t): outputs 8 i .. 8 i + 7, rows
  // 8 t .. 8 t + 7 (rows past R read row R - 1, dropped). qmv_rows' lane l at step s reads block b = 32 s + l
  // (values VPT b ..); simdgroup j of SG keeps lanes l = j (32 / SG) .. : per block an MMA from zero gives each
  // (output, row) that lane's dot, fma(scale, dot, bias * sum) joins the lane's partial in step order, and the 32
  // partials of each (output, row) meet in one simd_sum, as in qmv_rows.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int VPT = lane_values(BITS), NB = K / VPT, STEPS = NB / 32, KG = K / GS, WPR = K * BITS / 32;
  constexpr int L = 32 / SG;
  constexpr bool ONE_GROUP = L * VPT <= GS;             // a simdgroup's blocks of a step share one scale and bias
  static_assert(L == 4, "the sums load a step's 4 blocks as one float4");
  threadgroup float red[64 * 33];
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int o = min(nb + fm, N - 1);
  const device uint* wrow = W + size_t(o) * WPR;
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  const device bfloat* xa = X + size_t(ra) * K;
  const device bfloat* xc = X + size_t(rc) * K;
  const device float* sa = SUMS + size_t(ra) * NB;
  const device float* scs = SUMS + size_t(rc) * NB;
  float acc0[L], acc1[L];
  for (int j = 0; j < L; j++) { acc0[j] = 0.0f; acc1[j] = 0.0f; }
  constexpr uint MASK = (1u << BITS) - 1u;
  for (int t = 0; t < STEPS; t++) {
    const int b0 = 32 * t + sg * L;
    const float4 sua = *(const device float4*)(sa + b0), suc = *(const device float4*)(scs + b0);
    const float sums_a[4] = {sua.x, sua.y, sua.z, sua.w}, sums_c[4] = {suc.x, suc.y, suc.z, suc.w};
    float sc = 0.0f, bi = 0.0f;
    if (ONE_GROUP) { const size_t at = size_t(o) * KG + b0 * VPT / GS; sc = float(S[at]); bi = float(B[at]); }
    PRAGMA_UNROLL
    for (int j = 0; j < L; j++) {
      const int b = b0 + j;
      const int v0 = b * VPT;
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int h = 0; h < VPT / 8; h++) {
        simdgroup_matrix<float, 8, 8> am, bm;
        const int bit = (v0 + 8 * h + fn) * BITS, word = bit >> 5, shift = bit & 31;
        const uint hi = shift + 2 * BITS > 32 ? wrow[word + 1] : 0u;
        const ulong pair = ((ulong(hi) << 32) | ulong(wrow[word])) >> shift;
        am.thread_elements()[0] = float(uint(pair) & MASK);
        am.thread_elements()[1] = float(uint(pair >> BITS) & MASK);
        bm.thread_elements()[0] = float(xa[v0 + 8 * h + fm]);
        bm.thread_elements()[1] = float(xc[v0 + 8 * h + fm]);
        simdgroup_multiply_accumulate(P, am, bm, P);
      }
      if (!ONE_GROUP) { const size_t at = size_t(o) * KG + v0 / GS; sc = float(S[at]); bi = float(B[at]); }
      acc0[j] += fma(sc, P.thread_elements()[0], bi * sums_a[j]);
      acc1[j] += fma(sc, P.thread_elements()[1], bi * sums_c[j]);
    }
  }
  for (int j = 0; j < L; j++) {
    red[(fm * 8 + fn) * 33 + sg * L + j] = acc0[j];
    red[(fm * 8 + fn + 1) * 33 + sg * L + j] = acc1[j];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int e = sg; e < 64; e += SG) {
    const float v = simd_sum(red[e * 33 + int(lane)]);
    const int n = nb + e / 8, row = rb + e % 8;
    if (lane == 0 && n < N && row < R) OUT[size_t(row) * N + n] = bfloat(v);
  }
