#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "gated_delta.hpp"

// The kernel updates state in place. q and k stay packed by key head; the wave maps a value head itself.

void gdn(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, const at::Tensor& gate,
         const at::Tensor& beta, at::Tensor& state, at::Tensor& y) {
    TORCH_CHECK(q.is_cuda() && q.is_contiguous() && q.scalar_type() == at::kFloat && q.dim() == 4,
                "q: (batch, length, key heads, dk) fp32");
    const int64_t batch = q.size(0), length = q.size(1), key_heads = q.size(2), dk = q.size(3);
    TORCH_CHECK(k.sizes() == q.sizes() && k.is_cuda() && k.is_contiguous() && k.scalar_type() == at::kFloat,
                "k matches q");
    TORCH_CHECK(v.is_cuda() && v.is_contiguous() && v.scalar_type() == at::kFloat && v.dim() == 4 &&
                    v.size(0) == batch && v.size(1) == length,
                "v: (batch, length, value heads, dv) fp32");
    const int64_t value_heads = v.size(2), dv = v.size(3);
    TORCH_CHECK(gate.sizes() == beta.sizes() && gate.is_cuda() && beta.is_cuda() && gate.is_contiguous() &&
                    beta.is_contiguous() && gate.scalar_type() == at::kFloat && beta.scalar_type() == at::kFloat &&
                    gate.dim() == 3 && gate.size(0) == batch && gate.size(1) == length && gate.size(2) == value_heads,
                "gate and beta: (batch, length, value heads) fp32");
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat && state.dim() == 4 &&
                    state.size(0) == batch && state.size(1) == value_heads && state.size(2) == dv &&
                    state.size(3) == dk,
                "state: (batch, value heads, dv, dk) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kFloat && y.sizes() == v.sizes(),
                "y matches v");
    TORCH_CHECK(dk == 16 || dk == 128, "dk is 16 or 128");
    TORCH_CHECK(value_heads % key_heads == 0, "value heads are a multiple of key heads");
    c10::cuda::CUDAGuard guard(q.device());
    gated_delta_launch(q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(), gate.data_ptr<float>(),
                       beta.data_ptr<float>(), state.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(batch),
                       static_cast<int>(length), static_cast<int>(key_heads), static_cast<int>(value_heads),
                       static_cast<int>(dk), static_cast<int>(dv));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gdn", &gdn); }
