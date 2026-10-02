
  const uint lane = thread_index_in_simdgroup;
  const int u = int(threadgroup_position_in_grid.z);
  if (u >= UCOUNT[0]) return;
  const size_t e = size_t(UIDS[u]);
  const int row0 = (int(threadgroup_position_in_grid.y) * SG + int(simdgroup_index_in_threadgroup)) * RPS;
  const size_t at = e * N + size_t(row0);
  const int first = START[u], last = START[u] + COUNT[u];

  // fc1 and mlx_lm's relu2 on its bf16 output: bf16(max(bf16(sum), 0)^2)
  #pragma clang loop unroll(disable)
  for (int m = first; m < last; m++) {
    const int p = MEMBERS[m];
    float acc[RPS];
    tf_rowdot<K, GS, RPS>((const device uint8_t*)W + at * (K / 2), S + at * (K / GS), B + at * (K / GS),
                          X + size_t(p / TOPK) * K, lane, acc);
    if (lane == 0)
      for (int j = 0; j < RPS; j++) {
        const float h = metal::max(float(bfloat(acc[j])), 0.0f);
        ACT[size_t(p) * N + row0 + j] = bfloat(h * h);
      }
  }
