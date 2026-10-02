#include <torch/extension.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "act.hpp"

namespace {

void keep(const at::Tensor& tensor, c10::cuda::CUDAStream stream) {
    c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
}

}  // namespace

void rms(const at::Tensor& x, const at::Tensor& weight, at::Tensor& y, double eps) {
    const auto type = x.scalar_type();
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 &&
                    (type == at::kFloat || type == at::kHalf || type == at::kBFloat16),
                "x: (rows, d) fp32, fp16 or bf16");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes() && y.scalar_type() == type, "y matches x");
    const int64_t rows = x.size(0), width = x.size(1);
    TORCH_CHECK(rows >= 1 && width >= 1 && width <= 8192, "rms width");
    const float* wptr = nullptr;
    if (weight.defined() && weight.numel() > 0) {
        TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat &&
                        weight.numel() == width,
                    "rms weight");
        wptr = weight.data_ptr<float>();
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    keep(x, stream);
    keep(y, stream);
    if (wptr != nullptr) keep(weight, stream);
    const int kind = type == at::kHalf ? 1 : type == at::kBFloat16 ? 2 : 0;
    rms_launch(x.data_ptr(), wptr, y.data_ptr(), kind, static_cast<int>(rows), static_cast<int>(width),
               static_cast<float>(eps), stream.stream());
}

void conv_decode(const at::Tensor& x, const at::Tensor& weight, at::Tensor& state, at::Tensor& y) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 3 && x.size(1) == 1,
                "x: (batch, 1, channels) fp32");
    const int64_t batch = x.size(0), channels = x.size(2);
    TORCH_CHECK(weight.is_cuda() && weight.is_contiguous() && weight.scalar_type() == at::kFloat && weight.dim() == 2 &&
                    weight.size(0) == channels && weight.size(1) >= 1 && weight.size(1) <= 8,
                "weight: (channels, kernel) fp32, kernel 1..8");
    const int64_t kernel = weight.size(1);
    TORCH_CHECK(state.is_cuda() && state.is_contiguous() && state.scalar_type() == at::kFloat &&
                    state.sizes() == at::IntArrayRef({batch, kernel - 1, channels}),
                "state: (batch, kernel - 1, channels) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.scalar_type() == at::kFloat && y.sizes() == x.sizes(), "y");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {x, weight, state, y}) keep(tensor, stream);
    conv_decode_launch(x.data_ptr<float>(), weight.data_ptr<float>(), state.data_ptr<float>(), y.data_ptr<float>(),
                       static_cast<int>(batch), static_cast<int>(channels), static_cast<int>(kernel), stream.stream());
}

void rope_decode(const at::Tensor& x, at::Tensor& y, int64_t pos, int64_t rotary, double theta) {
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kFloat && x.dim() == 2, "x: (rows, d) fp32");
    TORCH_CHECK(y.is_cuda() && y.is_contiguous() && y.sizes() == x.sizes(), "y");
    const int64_t rows = x.size(0), width = x.size(1);
    TORCH_CHECK(rotary > 0 && rotary <= width && rotary % 2 == 0 && width <= 8192, "rotary dim");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    keep(x, stream);
    keep(y, stream);
    rope_decode_launch(x.data_ptr<float>(), y.data_ptr<float>(), static_cast<int>(rows), static_cast<int>(width),
                       static_cast<int>(rotary), static_cast<int>(pos), static_cast<float>(theta), stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rms", &rms);
    m.def("conv_decode", &conv_decode);
    m.def("rope_decode", &rope_decode);
}
