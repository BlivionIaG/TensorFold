
  // one threadgroup of T threads per row; thread t owns elements t, t + T, t + 2T, ...
  const uint t = thread_position_in_threadgroup.x;
  const uint r = threadgroup_position_in_grid.x;
  threadgroup bfloat xb[D];
  if (int(r) >= dims[0]) {
    for (int g = int(t); g < D / 64; g += T) XS[g * dims[1] + int(r)] = 0.0f;
    return;
  }
  constexpr int PER = D / T;
  threadgroup float partial[T / 32];
  float hv[PER];
  float ss = 0.0f;
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const int at = int(r) * D + c;
    float delta;
    {
      float routed = 0.0f;
      for (int e = 0; e < E; e++) routed = fma(float(Y[(int(r) * E + e) * D + c]), WE[int(r) * E + e], routed);
      delta = float(bfloat(float(bfloat(routed)) + float(SH[at])));
    }
    const bfloat hn = bfloat(float(H[at]) + delta);
    HN[at] = hn;
    hv[i] = float(hn);
    ss = fma(hv[i], hv[i], ss);
  }
  ss = simd_sum(ss);
  if (thread_index_in_simdgroup == 0) partial[simdgroup_index_in_threadgroup] = ss;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float total = 0.0f;
  for (int s = 0; s < T / 32; s++) total += partial[s];
  const float scale = metal::rsqrt(total / float(D) + eps[0]);
  for (int i = 0; i < PER; i++) {
    const int c = int(t) + i * T;
    const bfloat x = bfloat(float(W[c]) * (hv[i] * scale));
    OUT[int(r) * D + c] = x;
    xb[c] = x;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int g = int(t); g < D / 64; g += T) {
    float acc = 0.0f;
    for (int i = 0; i < 64; i++) acc += float(xb[g * 64 + i]);
    XS[g * dims[1] + int(r)] = acc;
  }
