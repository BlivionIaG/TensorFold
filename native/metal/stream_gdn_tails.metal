
  const uint c = thread_position_in_grid.x;
  const uint t = thread_position_in_grid.y;
  const uint st = thread_position_in_grid.z;
  const device bfloat16_t* Cb = CS0;
  device bfloat16_t* Tb = T0;
  switch (st) {
    case 1: Cb = CS1; Tb = T1; break;
    case 2: Cb = CS2; Tb = T2; break;
    case 3: Cb = CS3; Tb = T3; break;
    case 4: Cb = CS4; Tb = T4; break;
    case 5: Cb = CS5; Tb = T5; break;
    case 6: Cb = CS6; Tb = T6; break;
    case 7: Cb = CS7; Tb = T7; break;
  }
  const int src = tails[st * NKEEP + t];              // < NKEEP: a conv state row; else NKEEP + a row of all rows
  if (int(c) < C) Tb[t * C + c] = src < NKEEP ? Cb[src * C + c] : QKV[(src - NKEEP) * C + c];
