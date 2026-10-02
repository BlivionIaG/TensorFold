
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];

  // fc2 over each member pair's activation (bf16 out)
  #pragma clang loop unroll(disable)
  for (int m = first; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                          X + size_t(p) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) Y[size_t(p) * N + row0 + j] = bfloat(acc[j]);
  }
