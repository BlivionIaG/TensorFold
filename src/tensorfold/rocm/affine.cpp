#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "affine_api.hpp"

// Packed MLX affine words. The kernel reads those words; this function does not decode them into a BF16 weight.

void affine(const at::Tensor& x, const at::Tensor& words, const at::Tensor& scale, const at::Tensor& bias,
            at::Tensor& out, int64_t bits, int64_t group, int64_t schedule) {
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8, "bits 2/3/4/5/6/8");
    TORCH_CHECK(group == 32 || group == 64 || group == 128, "groups of 32, 64 or 128");
    TORCH_CHECK(schedule == 0 || schedule == 1 || schedule == 2, "schedule 0, 1 or 2");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.dim() == 2 && x.size(0) >= 1 &&
                    (x.scalar_type() == at::kBFloat16 || x.scalar_type() == at::kHalf),
                "x: (M, K) bf16, or fp16 on RDNA2, contiguous");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % group == 0 && (k * bits) % 32 == 0, "K is whole groups and whole packed words");
    const int64_t n = out.size(1), groups = k / group, words_row = k * bits / 32;
    TORCH_CHECK(words.is_cuda() && words.is_contiguous() && words.scalar_type() == at::kInt && words.dim() == 2 &&
                    words.size(0) == n && words.size(1) == words_row,
                "words: (N, K * bits / 32) int32, the packed weight");
    TORCH_CHECK(scale.is_cuda() && scale.is_contiguous() && bias.is_cuda() && bias.is_contiguous() &&
                    scale.scalar_type() == at::kFloat && bias.scalar_type() == at::kFloat && scale.sizes() == bias.sizes() &&
                    scale.size(0) == n && scale.size(1) == groups,
                "scale and bias: (N, K / group) fp32");
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kFloat && out.size(0) == m &&
                    out.size(1) == n,
                "out: (M, N) fp32");
    c10::cuda::CUDAGuard guard(x.device());
    affine_launch(x.data_ptr(), words.data_ptr(), scale.data_ptr(), bias.data_ptr(), out.data_ptr(),
                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(bits),
                  static_cast<int>(group), static_cast<int>(schedule), x.scalar_type() == at::kHalf ? 1 : 0,
                  c10::cuda::getCurrentCUDAStream().stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("affine", &affine); }
