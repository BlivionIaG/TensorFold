
  // qmv_rows' sum of each (row, block): block b's VPT inputs added in order from zero
  const int b = int(thread_position_in_grid.x), r = int(thread_position_in_grid.y);
  if (b >= K / VPT) return;
  const device bfloat* xp = X + size_t(r) * K + b * VPT;
  float sum = 0.0f;
  for (int i = 0; i < VPT; i++) sum += float(xp[i]);
  SUMS[size_t(r) * (K / VPT) + b] = sum;
