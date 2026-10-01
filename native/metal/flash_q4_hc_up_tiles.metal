
  // _HC_UP2 for 8 rows a tile on the matrix units, the same bits: the prologue sums the split partials in order into
  // the SiLU inputs; an up row's dot adds its groups' qgroup_dot values (one FMA chain over 32 inputs in order, then
  // fma(scale, dot, bias * sum)) in group order. Threadgroup (i, t): dims DT i .. of every stream, rows 8 t ..;
  // simdgroup s: stream s.
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GPR = LOW / 32, NT = 32 * S, LP = LOW + 4;
  const int d0 = int(threadgroup_position_in_grid.x) * DT;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int nr = min(8, R - rb);
  threadgroup float act[8 * LP];
  threadgroup float sums[8][GPR];
  threadgroup float rinv[8 * S];
  threadgroup float prod[S][DT][8];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  for (int i = int(t); i < 8 * ND; i += NT) {
    const int r = i / ND, cc = i % ND;
    const int row = min(rb + r, R - 1);
    float v = 0.0f;
    for (int kk = 0; kk < KS; kk++) v += PART[(size_t(kk) * R + row) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[r * LP + cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0 && r < nr) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < 8 * GPR; i += NT) {
    const int r = i / GPR, q = i % GPR;
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) dx += act[r * LP + 32 * q + v];
    sums[r][q] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int tt = 0; tt < DT / 8; tt++) {
    const int orow = s * D + d0 + 8 * tt + fm;
    const device uint* wq = QW + size_t(orow) * GPR * 4;
    float u0 = 0.0f, u1 = 0.0f;
    for (int q = 0; q < GPR; q++) {
      simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int st = 0; st < 4; st++) {
        const uint word = wq[4 * q + st];
        const int e = 32 * q + 8 * st + fm;
        simdgroup_matrix<float, 8, 8> am, bm;
        am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
        am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
        bm.thread_elements()[0] = act[fn * LP + e];
        bm.thread_elements()[1] = act[(fn + 1) * LP + e];
        simdgroup_multiply_accumulate(P, am, bm, P);
      }
      const float sc = float(QS[orow * GPR + q]), bi = float(QB[orow * GPR + q]);
      u0 += fma(sc, P.thread_elements()[0], bi * sums[fn][q]);
      u1 += fma(sc, P.thread_elements()[1], bi * sums[fn + 1][q]);
    }
    for (int e = 0; e < 2; e++) {
      const int r = fn + e, row = min(rb + r, R - 1);
      const float normed = float(bfloat((float(HN[size_t(row) * W + orow]) * rinv[r * S + s]) * NW[orow]));
      prod[s][8 * tt + fm][r] = float(bfloat(bsig(float(bfloat(e ? u1 : u0))) * normed));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < DT * 8; i += NT) {
    const int d = i / 8, r = i % 8;
    float total = 0.0f;
    for (int ss = 0; ss < S; ss++) total += prod[ss][d][r];
    if (r < nr) MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
  }
