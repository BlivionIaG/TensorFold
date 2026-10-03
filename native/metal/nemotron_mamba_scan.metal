
  // grid (32, DH, H): lane = NS state elements of channel d of head h. Rows in order; segment s starts from its
  // stream's SSM state, row SLOT[s] of S_IN. A row's arithmetic depends only on its own inputs and the state
  // before it.
  const uint lane = thread_position_in_threadgroup.x;
  const uint d = thread_position_in_grid.y;
  const uint h = thread_position_in_grid.z;
  const uint g = h / (H / NG);
  const int R = dims[0];
  constexpr int NS = DS / 32;
  constexpr int CD = XD + 2 * NG * DS;
  const int cx = int(h) * DH + int(d);
  const int cb = XD + int(g) * DS + int(lane) * NS;
  const int cc = XD + NG * DS + int(g) * DS + int(lane) * NS;
  const int sbase = cx * DS + int(lane) * NS;
  const float A = -metal::exp(float(A_LOG[h]));
  const float dskip = float(bfloat(float(DSKIP[h])));
  const float dtb = float(DT_BIAS[h]);
  float st[NS];
  int cur = -1;
  for (int rr = 0; rr < R; rr++) {
    const int s = SEG[rr];
    if (s != cur) {
      cur = s;
      for (int i = 0; i < NS; i++) st[i] = float(S_IN[size_t(SLOT[s]) * SSZ + sbase + i]);
    }
    const float xv = float(XBC[rr * CD + cx]);
    float dt = float(P[rr * PROJ + DTOFF + int(h)]) + dtb;
    dt = metal::max(dt, 0.0f) + metal::log(1.0f + metal::exp(-metal::abs(dt)));   // softplus (logaddexp(x, 0))
    dt = metal::clamp(dt, limits[0], limits[1]);
    const float dA = metal::exp(A * dt);
    const float xdt = xv * dt;
    float acc = 0.0f;
    for (int i = 0; i < NS; i++) {
      const float sv = dA * st[i] + xdt * float(XBC[rr * CD + cb + i]);
      st[i] = sv;
      acc += sv * float(XBC[rr * CD + cc + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
      const float y = float(bfloat(acc + xv * dskip));
      const float z = float(P[rr * PROJ + cx]);
      const float sz = float(bfloat(z / (1.0f + metal::exp(-z))));
      Y[rr * XD + cx] = bfloat(sz * y);
    }
    // the SSM state after this row, in its slot of S_OUT (a verify window keeps the state of its last accepted
    // row; STORE[rr] < 0: a row whose state is not kept)
    const int so = STORE[rr];
    if (so >= 0)
      for (int i = 0; i < NS; i++) S_OUT[size_t(so) * SSZ + sbase + i] = st[i];
  }
