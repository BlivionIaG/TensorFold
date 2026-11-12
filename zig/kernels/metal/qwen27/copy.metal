// Copies preserve BF16 bits while projection stacks become independent row-major views.
#include <metal_stdlib>
using namespace metal;

struct Slice { uint rows, width, stride, offset; };

kernel void q27_slice(device const ushort* X [[buffer(0)]], constant Slice& p [[buffer(1)]],
                      device ushort* Y [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) Y[t.y * p.width + t.x] = X[t.y * p.stride + p.offset + t.x];
}

struct Add { uint rows, width; };

kernel void q27_add(device const bfloat* X [[buffer(0)]], device const bfloat* R [[buffer(1)]],
                    constant Add& p [[buffer(2)]], device bfloat* Y [[buffer(3)]],
                    uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) {
    const uint i = t.y * p.width + t.x;
    Y[i] = bfloat(float(X[i]) + float(R[i]));
  }
}

struct Gather { uint rows, width, capacity, planes; };

kernel void q27_tap_gather(device const ushort* X [[buffer(0)]], device const uint* kept [[buffer(1)]],
                           constant Gather& p [[buffer(2)]], device ushort* Y [[buffer(3)]],
                           uint2 t [[thread_position_in_grid]]) {
  if (t.x >= p.width * p.planes || t.y >= p.rows || kept[t.y] >= p.capacity) return;
  const uint plane = t.x / p.width, column = t.x % p.width;
  Y[t.y * (p.width * p.planes) + t.x] = X[(plane * p.capacity + kept[t.y]) * p.width + column];
}

struct Take { uint rows, width; };

// An image row's bits over its placeholder's embedding (take[r]: its row in R, or -1 to keep the embedding).
[[max_total_threads_per_threadgroup(256)]] kernel void q27_take(device ushort* X [[buffer(0)]], device const ushort* R [[buffer(1)]],
                     device const int* take [[buffer(2)]], constant Take& p [[buffer(3)]],
                     uint2 t [[thread_position_in_grid]]) {
  if (t.x >= p.width || t.y >= p.rows || take[t.y] < 0) return;
  X[t.y * p.width + t.x] = R[ulong(take[t.y]) * p.width + t.x];
}

struct Row { uint rows, width, stamp; };

// A layer range's hand-off row: bf16 H + R rounded to bf16, widened to f32; stamp 0 is plain, else stamped groups.
[[max_total_threads_per_threadgroup(256)]] kernel void q27_row_store(device const bfloat* H [[buffer(0)]], device const bfloat* R [[buffer(1)]],
                     constant Row& p [[buffer(2)]], device uchar* out [[buffer(3)]],
                     uint2 t [[thread_position_in_grid]]) {
  if (t.y >= p.rows) return;
  if (p.stamp == 0) {
    if (t.x < p.width) {
      const uint i = t.y * p.width + t.x;
      ((device float*)out)[i] = float(bfloat(float(H[i]) + float(R[i])));
    }
    return;
  }
  const uint groups = (p.width + 2) / 3;
  if (t.x >= groups) return;
  uint4 g = uint4(0, 0, 0, p.stamp);
  for (uint k = 0; k < 3; ++k) {
    const uint c = 3 * t.x + k;
    if (c < p.width) {
      const uint i = t.y * p.width + c;
      g[k] = as_type<uint>(float(bfloat(float(H[i]) + float(R[i]))));
    }
  }
  ((device uint4*)out)[t.y * groups + t.x] = g;
}

// The inverse for plain rows: bf16 bits from f32 words, exact for words that were bf16; stamped rows read elsewhere.
[[max_total_threads_per_threadgroup(256)]] kernel void q27_row_load(device const float* in [[buffer(0)]], constant Take& p [[buffer(1)]],
                     device bfloat* X [[buffer(2)]], uint2 t [[thread_position_in_grid]]) {
  if (t.x < p.width && t.y < p.rows) X[t.y * p.width + t.x] = bfloat(in[t.y * p.width + t.x]);
}
