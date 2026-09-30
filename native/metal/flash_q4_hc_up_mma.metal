
  // The up projection for dims d0 .. d0 + DT - 1 of every stream x 8 rows, every row reading the weights once.
  // Prologue: the down projection's outputs from the split-K partials (summed in chunk order) -> bf16 -> / S ->
  // bf16 -> SiLU (bf16) for the rows here; threadgroup 0 of a row tile also writes their inject gates 2 sigmoid.
  // Simdgroup s takes up rows s D + d0 .. (DT / 8 tiles of 8); then sigmoid of each (bf16) times the normed stream
  // (bf16), summed over the streams in order, / S, into MIXED.
  const uint lane = thread_index_in_simdgroup;
  const int s = int(simdgroup_index_in_threadgroup);
  const uint t = thread_position_in_threadgroup.x;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = rows[0];
  constexpr int W = S * D, GL = LOW / 32, TT = DT / 8;
  const int d0 = int(threadgroup_position_in_grid.x) * DT;
  const int rb = int(threadgroup_position_in_grid.y) * 8;
  const int nr = min(8, R - rb);                       // rows of this tile
  threadgroup float rinv[8 * S];
  threadgroup float act[8][LOW];
  threadgroup float prod[S][DT][8];
  if (t < 8 * S) rinv[t] = stream_rinv(SSP, min(rb + int(t) / S, R - 1), int(t) % S, D / 256, S, D, eps[0]);
  for (int i = int(t); i < nr * ND; i += 32 * S) {
    const int r = i / ND, cc = i % ND;
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + rb + r) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[r][cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[(rb + r) * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int ra = min(fn, nr - 1), rc = min(fn + 1, nr - 1);
  for (int tt = 0; tt < TT; tt++) {
    const int o = s * D + d0 + 8 * tt + fm;
    const device uint* wq = QW + size_t(o) * (LOW / 8) + fn / 2;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int g = 0; g < GL; g++) {
      const int e0 = 32 * g + 8 * (fm / 2);
      float xa[8], xc[8];
      for (int i = 0; i < 8; i++) { xa[i] = act[ra][e0 + i]; xc[i] = act[rc][e0 + i]; }
      mma_group(wq[4 * g], xa, xc, fm, float(QS[size_t(o) * GL + g]), float(QB[size_t(o) * GL + g]), acc0, acc1);
    }
    for (int e = 0; e < 2; e++) {
      const int row = rb + min(fn + e, nr - 1);
      const float normed = float(bfloat((float(HN[size_t(row) * W + o]) * rinv[(row - rb) * S + s]) * NW[o]));
      prod[s][8 * tt + fm][fn + e] = float(bfloat(bsig(float(bfloat(e ? acc1 : acc0))) * normed));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = int(t); i < DT * 8; i += 32 * S) {
    const int d = i / 8, r = i % 8;
    float total = 0.0f;
    for (int k = 0; k < S; k++) total += prod[k][d][r];
    if (r < nr) MIXED[size_t(rb + r) * D + d0 + d] = bfloat(total / float(S));
  }
