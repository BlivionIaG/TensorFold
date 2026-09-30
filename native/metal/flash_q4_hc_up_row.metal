
  // The up projection of one row, bit for bit the MMA path's: threadgroup (i, r) takes dims 8 i .. 8 i + 7 of the
  // S streams. The prologue is the MMA path's; thread (g, s, d) takes one group's product sum, then each output's
  // chain over the groups in order, the sigmoid times the normed stream, summed over the streams in order.
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  constexpr int W = S * D, GL = LOW / 32, NO = 8 * S;
  const int d0 = int(threadgroup_position_in_grid.x) * 8;
  const int r = int(threadgroup_position_in_grid.z);
  threadgroup float act[LOW];
  threadgroup float vs[GL];
  threadgroup float ps[GL][NO];
  threadgroup float prod[S][8];
  for (int cc = int(t); cc < ND; cc += GL * NO) {
    float v = 0.0f;
    for (int k = 0; k < KS; k++) v += PART[(size_t(k) * R + r) * ND + cc];
    const float v4 = float(bfloat(float(bfloat(v)) / float(S)));
    if (cc < LOW) act[cc] = bsilu(v4);
    else if (threadgroup_position_in_grid.x == 0) INJOUT[r * S + (cc - LOW)] = bfloat(2.0f * bsig(v4));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < GL) vs[t] = scalar_sum(act + 32 * t);
  const int g = int(t) / NO, n = int(t) % NO;
  const int s = n / 8, o = s * D + d0 + n % 8;
  ps[g][n] = scalar_group(QW + size_t(o) * (LOW / 8) + 4 * g, act + 32 * g);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < NO) {
    float acc = 0.0f;
    for (int gg = 0; gg < GL; gg++)
      acc = fma(float(QB[size_t(o) * GL + gg]), vs[gg], fma(float(QS[size_t(o) * GL + gg]), ps[gg][n], acc));
    const float normed = float(bfloat((float(HN[size_t(r) * W + o]) * stream_rinv(SSP, r, s, D / 256, S, D, eps[0]))
                                      * NW[o]));
    prod[s][n % 8] = float(bfloat(bsig(float(bfloat(acc))) * normed));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(t) < 8) {
    float total = 0.0f;
    for (int k = 0; k < S; k++) total += prod[k][t];
    MIXED[size_t(r) * D + d0 + int(t)] = bfloat(total / float(S));
  }
