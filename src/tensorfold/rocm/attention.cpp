#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "attention.hpp"

// q is made contiguous by the caller. k and v keep the cache strides, including a short prefix of a longer buffer.

void causal(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, at::Tensor& out, double scale,
            int64_t q_pos0) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kFloat && q.dim() == 4,
                "q: (batch, heads, qlen, d) fp32");
    const int64_t batch = q.size(0), heads = q.size(1), qlen = q.size(2), d = q.size(3);
    TORCH_CHECK(k.is_cuda() && v.is_cuda() && k.scalar_type() == v.scalar_type() && k.sizes() == v.sizes() &&
                    k.dim() == 4 && k.size(0) == batch && k.size(3) == d,
                "k and v: (batch, kv heads, span, d), same dtype");
    int kind = k.scalar_type() == at::kHalf ? 0 : k.scalar_type() == at::kBFloat16 ? 1 : 2;
    TORCH_CHECK(kind != 2 || k.scalar_type() == at::kFloat, "k and v are fp16, bf16, or fp32");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.sizes() == q.sizes(),
                "out matches q");
    TORCH_CHECK(d <= 256 && heads % k.size(1) == 0, "d <= 256 and heads are a multiple of kv heads");
    c10::cuda::CUDAGuard guard(q.device());
    float* scores = nullptr;
    float* stats = nullptr;
    float* partials = nullptr;
    at::Tensor score_buf;
    at::Tensor stat_buf;
    at::Tensor partial_buf;
    if (qlen == 1) {
        int64_t visible = q_pos0 + 1;
        if (visible > k.size(2)) visible = k.size(2);
        int64_t tiles = (visible + 127) / 128;
        score_buf = at::empty({batch, heads, k.size(2)}, q.options());
        stat_buf = at::empty({batch, heads, 2}, q.options());
        partial_buf = at::zeros({batch, heads, tiles, d}, q.options());
        scores = score_buf.data_ptr<float>();
        stats = stat_buf.data_ptr<float>();
        partials = partial_buf.data_ptr<float>();
    }
    causal_launch(q.data_ptr<float>(), k.data_ptr(), v.data_ptr(), out.data_ptr<float>(), static_cast<int>(batch),
                  static_cast<int>(qlen), static_cast<int>(k.size(2)), static_cast<int>(heads),
                  static_cast<int>(k.size(1)), static_cast<int>(d), static_cast<float>(scale),
                  static_cast<int>(q_pos0), k.stride(0), k.stride(1), k.stride(2), v.stride(0), v.stride(1),
                  v.stride(2), kind, scores, stats, partials);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("causal", &causal); }
