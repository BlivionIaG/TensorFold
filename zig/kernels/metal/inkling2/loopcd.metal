// LoopCD: h' = h_R + w (h_R - h_1) a plane at a time, BF16 in and out, one rounding (metal_loop.zig).

struct LpcdArgs {
  uint rows, hidden, planes, stride;
  float w, pad0, pad1, pad2;
};

// Element i of a row: the final pass's value, the first pass's, then the guided value written in place.
[[kernel, max_total_threads_per_threadgroup(256)]] void inkling2_loop_loopcd_combine(
    const device bfloat* hr [[buffer(0)]], const device bfloat* h1 [[buffer(1)]],
    device bfloat* out [[buffer(2)]], constant LpcdArgs& a [[buffer(3)]],
    uint3 grid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
  const uint i = grid.x * 256u + tid.x;
  if (i >= a.hidden) return;
  const ulong o = ulong(grid.y) * a.stride + ulong(grid.z) * a.hidden + i;
  const float r = float(hr[o]);
  const float p = float(h1[o]);
  out[o] = bfloat(fma(a.w, r - p, r));
}
