
  // Split-K down projection of the normed streams on the matrix units, every row reading the weights once:
  // threadgroup (i, k, t) takes outputs 8 i .. 8 i + 7 over input groups 32 k .. 32 k + 31 for rows 8 t .. 8 t + 7
  // (rows past R read row R - 1, dropped); simdgroup c of 8 takes 4 of the groups, and the 8 add in order into
  // PART[k][r][o], which the up projection sums in k order.
  const uint lane = thread_index_in_simdgroup;
  const int c = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = rows[0];
  constexpr int W = S * D, G = W / 32;
  const int nb = int(threadgroup_position_in_grid.x) * 8;
  const int k = int(threadgroup_position_in_grid.y);
  const int rb = int(threadgroup_position_in_grid.z) * 8;
  threadgroup float rinv[8 * S];
  threadgroup float red[8][64];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int o = min(nb + fm, ND - 1);
  const int ra = min(rb + fn, R - 1), rc = min(rb + fn + 1, R - 1);
  const device uint* wq = QW + size_t(o) * (W / 8) + fn / 2;
  float acc0 = 0.0f, acc1 = 0.0f;
  for (int j = 0; j < 4; j++) {
    const int g = 32 * k + 4 * c + j;
    const int e0 = 32 * g + 8 * (fm / 2);            // the lane's 8 inputs (one stream: D % 8 == 0)
    const float ia = rinv[(ra - rb) * S + e0 / D], ic = rinv[(rc - rb) * S + e0 / D];
    const uint4 ha = ((const device uint4*)HN)[(size_t(ra) * W + e0) / 8];
    const uint4 hc = ((const device uint4*)HN)[(size_t(rc) * W + e0) / 8];
    float xa[8], xc[8];
    for (int i = 0; i < 8; i++) {
      const float w = NW[e0 + i];
      xa[i] = float(bfloat((bfv(ha, i) * ia) * w));
      xc[i] = float(bfloat((bfv(hc, i) * ic) * w));
    }
    mma_group(wq[4 * g], xa, xc, fm, float(QS[size_t(o) * G + g]), float(QB[size_t(o) * G + g]), acc0, acc1);
  }
  red[c][2 * lane] = acc0;
  red[c][2 * lane + 1] = acc1;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 64) {
    float v = 0.0f;
    for (int cc = 0; cc < 8; cc++) v += red[cc][t];
    const int l = int(t) / 2, lq = l / 4;
    const int out = nb + (lq & 4) + ((l / 2) % 4), row = rb + (lq & 2) * 2 + (l % 2) * 2 + int(t) % 2;
    if (out < ND && row < R) PART[(size_t(k) * R + row) * ND + out] = v;
  }
