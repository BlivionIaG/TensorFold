
  // grid (CD, R): channel ch of row rr. Rows come in segments, one per stream (SEG: a row's segment, START: a
  // segment's first row); a segment's taps before its first row come from its conv state, row SLOT[s] of CS_IN.
  // Writes the conv output bf16(silu(bf16(conv))) as mlx_lm rounds it, and the row's conv state (its segment's
  // last KC-1 inputs).
  constexpr int CD = XD + 2 * NG * DS;
  const int ch = int(thread_position_in_grid.x);
  const int rr = int(thread_position_in_grid.y);
  const int s = SEG[rr];
  const int b = START[s];
  const int loc = rr - b;
  const int slot = SLOT[s];
  #define TAP(lp) ((lp) < 0 ? CS_IN[(slot * (KC - 1) + (lp) + KC - 1) * CD + ch] : P[(b + (lp)) * PROJ + XOFF + ch])
  float a = float(CB[ch]);
  for (int k = 0; k < KC; k++) a = fma(CW[k * CD + ch], float(TAP(loc - (KC - 1) + k)), a);
  const float cv = float(bfloat(a));
  XBC[rr * CD + ch] = bfloat(cv / (1.0f + metal::exp(-cv)));
  for (int k = 0; k < KC - 1; k++) CS_OUT[(rr * (KC - 1) + k) * CD + ch] = TAP(loc - (KC - 2) + k);
  #undef TAP
