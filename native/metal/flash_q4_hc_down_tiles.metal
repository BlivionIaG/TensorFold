
  // _HC_DOWN_SPLIT for 8 rows a tile on the matrix units, the same bits: a group's dot is one FMA chain over its 32
  // inputs in order (4 chained 8-step MMAs), then fma(scale, dot, bias * sum), and a split's 32 groups meet in one
  // simd_sum with lane = group. Threadgroup (i, k, t): outputs 8 i .., split k, rows 8 t ..; simdgroup c: groups 4 c ..
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int fm = tile_fm(int(lane)), fn = tile_fn(int(lane));
  const int R = rows[0];
  constexpr int W = S * D, GROUPS = W / 32;
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int k = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * 8;
  threadgroup float rinv[8 * S];
  threadgroup float sums[8][32];
  threadgroup float red[64][33];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  {
    const int r = int(lane) % 8, j = 4 * c + int(lane) / 8;
    const int row = min(rb + r, R - 1);
    const int e0 = 32 * (32 * k + j);
    float dx = 0.0f;
    for (int v = 0; v < 32; v++) {
      const int e = e0 + v;
      dx += float(bfloat((float(HN[size_t(row) * W + e]) * rinv[r * S + e / D]) * NW[e]));
    }
    sums[r][j] = dx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = min(nb + fm, ND - 1);
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  for (int jj = 0; jj < 4; jj++) {
    const int j = 4 * c + jj, g = 32 * k + j;
    const device uint* wq = QW + (size_t(o) * GROUPS + g) * 4;
    simdgroup_matrix<float, 8, 8> P = simdgroup_matrix<float, 8, 8>(0.0f);
    for (int st = 0; st < 4; st++) {
      const uint word = wq[st];
      const int e = 32 * g + 8 * st + fm;
      simdgroup_matrix<float, 8, 8> am, bm;
      am.thread_elements()[0] = nib((word >> (4 * fn)) & 0xFu);
      am.thread_elements()[1] = nib((word >> (4 * fn + 4)) & 0xFu);
      bm.thread_elements()[0] = float(bfloat((float(HN[size_t(ra) * W + e]) * rinv[(ra - rb) * S + e / D]) * NW[e]));
      bm.thread_elements()[1] = float(bfloat((float(HN[size_t(rc) * W + e]) * rinv[(rc - rb) * S + e / D]) * NW[e]));
      simdgroup_multiply_accumulate(P, am, bm, P);
    }
    const float sc = float(QS[o * GROUPS + g]), bi = float(QB[o * GROUPS + g]);
    red[fm * 8 + fn][j] = fma(sc, P.thread_elements()[0], bi * sums[ra - rb][j]);
    red[fm * 8 + fn + 1][j] = fma(sc, P.thread_elements()[1], bi * sums[rc - rb][j]);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = 8 * c; p < 8 * c + 8; p++) {
    const float v = simd_sum(red[p][lane]);
    const int out = nb + p / 8, row = rb + p % 8;
    if (lane == 0 && out < ND && row < R) PART[(size_t(k) * R + row) * ND + out] = v;
  }
