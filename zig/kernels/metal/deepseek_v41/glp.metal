// In-place GLPv3 paired fold/writer edits with FP32 math and BF16 RNE; contract and credits are in the GLP recipe.
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)

#ifndef TF_CTL
#define TF_CTL 0
#endif

#define TF_GLP_TG 256u

// Control 1: a non-finite input is read as 0, so the output class changes.
inline float tf_glp_in(float v) { return (TF_CTL == 1 && !isfinite(v)) ? 0.0f : v; }

// Control 3: a running total held in fp16.
inline float tf_glp_run(float v) { return TF_CTL == 3 ? float(half(v)) : v; }

struct TfGlpArgs {
  uint dim, rows;
  float alpha;
  uint project;
};
static_assert(sizeof(TfGlpArgs) == 16, "TfGlpArgs must mirror metal/glp.zig's Args");

// The butterfly of a simdgroup: every lane holds the lanes' sum, added in the same pairs in every launch.
inline float tf_glp_tree(float v) {
  for (ushort m = 1; m < 32; m *= 2) v += simd_shuffle_xor(v, m);
  return v;
}

// Scale each BF16 copy by 1/4 before FP32 addition to avoid overflow; control 2 deliberately uses the plain sum.
template <uint COPIES>
inline float tf_glp_mean(const device bfloat* x, uint dim, uint i) {
  const float k = (COPIES == 1 || TF_CTL == 2) ? 1.0f : 0.25f;
  float s = tf_glp_in(float(x[i])) * k;
  for (uint c = 1; c < COPIES; ++c) s = s + tf_glp_in(float(x[c * dim + i])) * k;
  return s;
}

// X [rows][COPIES][dim] bf16 edited in place by the row's direction D [dim] fp32.
template <uint COPIES>
[[kernel, max_total_threads_per_threadgroup(TF_GLP_TG)]] void tf_dsv41_glp_edit(
    device bfloat* X [[buffer(0)]], const device float* D [[buffer(1)]], constant TfGlpArgs& a [[buffer(2)]],
    uint r [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]], uint sg [[simdgroup_index_in_threadgroup]]) {
  threadgroup float part[TF_GLP_TG / 32];
  // alpha and rows are the launch's: every thread of every group takes the same exit
  if (a.alpha == 0.0f || r >= a.rows) return;
  device bfloat* x = X + ulong(r) * COPIES * a.dim;
  // add: the same step on every copy; project: alpha times the mean's component along d, removed from every copy
  float step = a.alpha;
  if (a.project != 0 && TF_CTL != 4) {
    float s = 0.0f;
    for (uint i = t; i < a.dim; i += TF_GLP_TG) s = tf_glp_run(s + tf_glp_mean<COPIES>(x, a.dim, i) * D[i]);
    s = tf_glp_tree(s);
    if (lane == 0) part[sg] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = part[0];
    for (uint k = 1; k < TF_GLP_TG / 32; ++k) total = tf_glp_run(total + part[k]);
    step = a.alpha * total;
    // every thread's reads of the row (any element) precede every thread's stores of it
    threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  }
  const bool minus = a.project != 0 && TF_CTL != 4;
  for (uint i = t; i < a.dim; i += TF_GLP_TG) {
    const float d = D[i];
    for (uint k = 0; k < COPIES; ++k) {
      if (TF_CTL == 5 && k > 0) continue;
      const float v = tf_glp_in(float(x[k * a.dim + i]));
      x[k * a.dim + i] = bfloat(minus ? v - step * d : v + step * d);
    }
  }
  if (TF_CTL == 6 && t == 0) x[ulong(COPIES) * a.dim * (a.rows - r)] = bfloat(1.0f);
}

template [[host_name("tf_dsv41_glp_fold")]] [[kernel]] decltype(tf_dsv41_glp_edit<4>) tf_dsv41_glp_edit<4>;
template [[host_name("tf_dsv41_glp_writer")]] [[kernel]] decltype(tf_dsv41_glp_edit<1>) tf_dsv41_glp_edit<1>;
