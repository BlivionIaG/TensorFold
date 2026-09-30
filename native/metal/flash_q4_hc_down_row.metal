
  // The down projection of one row, bit for bit the MMA path's: threadgroup (i, k, r) takes outputs 32 i ..
  // 32 i + 31 over input groups 32 k .. 32 k + 31 of row r. Thread (g, o) takes one group's product sum; then,
  // as the MMA path, 8 chains of 4 groups each (scale and bias folded in order) add in order into PART[k][r][o].
  const uint t = thread_position_in_threadgroup.x;
  const int R = rows[0];
  constexpr int W = S * D, G = W / 32;
  const int ob = int(threadgroup_position_in_grid.x) * 32;
  const int k = int(threadgroup_position_in_grid.y);
  const int r = int(threadgroup_position_in_grid.z);
  threadgroup float xs[1024];
  threadgroup float vs[32];
  threadgroup float ps[32][32];
  threadgroup float red[8][32];
  {
    const int e = 1024 * k + int(t);
    const float ri = stream_rinv(SSP, r, e / D, D / 256, S, D, eps[0]);
    xs[t] = float(bfloat((float(HN[size_t(r) * W + e]) * ri) * NW[e]));
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32) vs[t] = scalar_sum(xs + 32 * t);
  const int gl = int(t) / 32, ol = int(t) % 32;
  const int o = min(ob + ol, ND - 1);
  ps[gl][ol] = scalar_group(QW + size_t(o) * (W / 8) + 4 * (32 * k + gl), xs + 32 * gl);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 256) {
    const int c = int(t) / 32;
    float acc = 0.0f;
    for (int j = 0; j < 4; j++) {
      const int g = 32 * k + 4 * c + j;
      acc = fma(float(QB[size_t(o) * G + g]), vs[4 * c + j],
                fma(float(QS[size_t(o) * G + g]), ps[4 * c + j][ol], acc));
    }
    red[c][ol] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (t < 32 && ob + ol < ND) {
    float v = 0.0f;
    for (int c = 0; c < 8; c++) v += red[c][ol];
    PART[(size_t(k) * R + r) * ND + ob + ol] = v;
  }
