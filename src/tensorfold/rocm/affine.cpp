#include <torch/extension.h>

#include <vector>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "affine_api.hpp"

void affine(const at::Tensor& x, const at::Tensor& words, const at::Tensor& scale, const at::Tensor& bias,
            at::Tensor& out, int64_t bits, int64_t group, int64_t schedule, int64_t split_mode) {
    TORCH_CHECK(bits == 2 || bits == 3 || bits == 4 || bits == 5 || bits == 6 || bits == 8, "bits 2/3/4/5/6/8");
    TORCH_CHECK(group == 32 || group == 64 || group == 128, "groups of 32, 64 or 128");
    TORCH_CHECK(schedule >= 0 && schedule <= 3, "schedule 0, 1, 2, or 3");
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
    auto stream = c10::cuda::getCurrentCUDAStream();
    int splits = 1;
    at::Tensor scratch;
    float* partial = nullptr;
    // The vocabulary grid is already wide. A short N with a long K splits whole groups.
    if (x.scalar_type() == at::kHalf && schedule != 2) {
        splits = affine_dot2_splits(static_cast<int>(m), static_cast<int>(n), static_cast<int>(k),
                                    static_cast<int>(group), static_cast<int>(split_mode));
    }
    if (splits > 1) {
        scratch = at::empty({m * n * groups * 2}, out.options());
        partial = scratch.data_ptr<float>();
    }
    for (const at::Tensor& tensor : {x, words, scale, bias, out, scratch}) {
        if (tensor.defined()) {
            c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
        }
    }
    affine_launch(x.data_ptr(), words.data_ptr(), scale.data_ptr(), bias.data_ptr(), out.data_ptr(),
                  static_cast<int>(m), static_cast<int>(n), static_cast<int>(k), static_cast<int>(bits),
                  static_cast<int>(group), static_cast<int>(schedule), x.scalar_type() == at::kHalf ? 1 : 0,
                  stream.stream(), partial, splits);
}

void affine_pair(const at::Tensor& x, const at::Tensor& words0, const at::Tensor& scale0, const at::Tensor& bias0,
                 at::Tensor& out0, const at::Tensor& words1, const at::Tensor& scale1, const at::Tensor& bias1,
                 at::Tensor& out1, int64_t bits, int64_t group) {
    TORCH_CHECK(bits == 8 && (group == 32 || group == 64 || group == 128), "pair is 8-bit");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x bf16");
    const int64_t m = x.size(0), k = x.size(1);
    auto check = [&](const at::Tensor& words, const at::Tensor& scale, const at::Tensor& bias, const at::Tensor& out) {
        TORCH_CHECK(words.is_cuda() && words.is_contiguous() && words.scalar_type() == at::kInt, "words");
        TORCH_CHECK(scale.is_cuda() && bias.is_cuda() && out.is_cuda() && out.is_contiguous(), "pair tensors");
        TORCH_CHECK(words.size(1) == k * bits / 32 && scale.size(0) == words.size(0) && out.size(0) == m &&
                        out.size(1) == words.size(0) && scale.size(1) == k / group,
                    "pair shapes");
    };
    check(words0, scale0, bias0, out0);
    check(words1, scale1, bias1, out1);
    TORCH_CHECK(words0.size(0) == words1.size(0), "pair outputs share N");
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : {x, words0, scale0, bias0, out0, words1, scale1, bias1, out1}) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    affine_pair_launch(x.data_ptr(), words0.data_ptr(), scale0.data_ptr(), bias0.data_ptr(), out0.data_ptr(),
                       words1.data_ptr(), scale1.data_ptr(), bias1.data_ptr(), out1.data_ptr(), static_cast<int>(m),
                       static_cast<int>(words0.size(0)), static_cast<int>(k), static_cast<int>(bits),
                       static_cast<int>(group), stream.stream());
}

void affine_group(const at::Tensor& x, std::vector<at::Tensor> words, std::vector<at::Tensor> scale,
                  std::vector<at::Tensor> bias, std::vector<at::Tensor> out, int64_t bits, int64_t group) {
    TORCH_CHECK(bits == 8 && (group == 32 || group == 64 || group == 128), "group matmul is 8-bit");
    TORCH_CHECK(x.is_cuda() && x.is_contiguous() && x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x bf16");
    const int64_t nsides = static_cast<int64_t>(words.size());
    TORCH_CHECK(nsides >= 1 && nsides <= 4 && scale.size() == static_cast<size_t>(nsides) &&
                    bias.size() == static_cast<size_t>(nsides) && out.size() == static_cast<size_t>(nsides),
                "group matmul takes 1 to 4 sides");
    const int64_t m = x.size(0), k = x.size(1);
    TORCH_CHECK(k % group == 0, "K is whole groups");
    const void* wptr[4] = {};
    const void* sptr[4] = {};
    const void* bptr[4] = {};
    void* optr[4] = {};
    int ns[4] = {};
    std::vector<at::Tensor> kept;
    kept.reserve(static_cast<size_t>(nsides) * 4 + 1);
    kept.push_back(x);
    for (int64_t i = 0; i < nsides; ++i) {
        const int64_t n = words[i].size(0);
        TORCH_CHECK(words[i].is_cuda() && words[i].is_contiguous() && words[i].scalar_type() == at::kInt &&
                        words[i].size(1) == k * bits / 32,
                    "words");
        TORCH_CHECK(scale[i].is_cuda() && bias[i].is_cuda() && out[i].is_cuda() && scale[i].is_contiguous() &&
                        bias[i].is_contiguous() && out[i].is_contiguous(),
                    "group tensors");
        TORCH_CHECK(scale[i].size(0) == n && scale[i].size(1) == k / group && out[i].size(0) == m &&
                        out[i].size(1) == n,
                    "group shapes");
        wptr[i] = words[i].data_ptr();
        sptr[i] = scale[i].data_ptr();
        bptr[i] = bias[i].data_ptr();
        optr[i] = out[i].data_ptr();
        ns[i] = static_cast<int>(n);
        kept.push_back(words[i]);
        kept.push_back(scale[i]);
        kept.push_back(bias[i]);
        kept.push_back(out[i]);
    }
    c10::cuda::CUDAGuard guard(x.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    for (const at::Tensor& tensor : kept) {
        c10::cuda::CUDACachingAllocator::recordStream(tensor.storage().data_ptr(), stream);
    }
    affine_group_launch(x.data_ptr(), wptr, sptr, bptr, optr, ns, static_cast<int>(nsides), static_cast<int>(m),
                        static_cast<int>(k), static_cast<int>(bits), static_cast<int>(group), stream.stream());
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("affine", &affine);
    m.def("affine_pair", &affine_pair);
    m.def("affine_group", &affine_group);
}
