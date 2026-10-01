
  // Two experts a threadgroup, four simdgroups an expert: simdgroup q of an expert takes inputs q D / 4 .. (lane l:
  // 4 consecutive inputs every 128), fp32 in order, simd_sum; the quarters add in order. Rows in order.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int e = int(threadgroup_position_in_grid.x) * 2 + sg / 4, q = sg % 4;
  const int R = rows[0];
  constexpr int QD = D / 4;
  threadgroup float part[MAXR][8];
  const int ee = min(e, NE - 1);
  const device bfloat* w = GW + size_t(ee) * D + q * QD + 4 * int(lane);
  float wv[QD / 32];
  for (int i = 0; i < QD / 128; i++)
    for (int j = 0; j < 4; j++) wv[4 * i + j] = float(w[128 * i + j]);
  for (int r = 0; r < R; r++) {
    const device bfloat* xr = X + r * D + q * QD + 4 * int(lane);
    float a = 0.0f;
    for (int i = 0; i < QD / 128; i++)
      for (int j = 0; j < 4; j++) a = fma(float(xr[128 * i + j]), wv[4 * i + j], a);
    a = simd_sum(a);
    if (lane == 0) part[r][sg] = a;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int t = int(thread_position_in_threadgroup.x);
  if (t < 2 * R) {
    const int r = t / 2, k = t % 2, e2 = int(threadgroup_position_in_grid.x) * 2 + k;
    const float sum = ((part[r][4 * k] + part[r][4 * k + 1]) + part[r][4 * k + 2]) + part[r][4 * k + 3];
    if (e2 < NE) OUT[r * NE + e2] = float(sum);
  }
