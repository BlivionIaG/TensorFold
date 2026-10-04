#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

#include "qgemm.hpp"

// W4A16 GPTQ on RDNA2, ported from vLLM-rdna csrc/rocm/q_gemm_rdna2.cu; the fp16 output is zeroed for atomics.
// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

at::Tensor gptq_matmul(const at::Tensor& a, const at::Tensor& qweight, const at::Tensor& qzeros,
                       const at::Tensor& scales, const at::Tensor& g_idx, bool use_v2_format,
                       bool prefill) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kHalf,
                "a: (M, K) fp16, contiguous");
    TORCH_CHECK(qweight.is_cuda() && qweight.is_contiguous() && qweight.dim() == 2 &&
                    qweight.scalar_type() == at::kInt,
                "qweight: (K / 8, N) int32, contiguous");
    TORCH_CHECK(qzeros.is_cuda() && qzeros.is_contiguous() && qzeros.dim() == 2 &&
                    qzeros.scalar_type() == at::kInt,
                "qzeros: (groups, N / 8) int32, contiguous");
    TORCH_CHECK(scales.is_cuda() && scales.is_contiguous() && scales.dim() == 2 &&
                    scales.scalar_type() == at::kHalf,
                "scales: (groups, N) fp16, contiguous");

    const int64_t m = a.size(0), k = a.size(1), n = qweight.size(1);
    TORCH_CHECK(qweight.size(0) * 8 == k, "qweight first dim must be K / 8");
    TORCH_CHECK(k % 32 == 0, "K must be a multiple of 32");
    TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
    const int64_t groups = qzeros.size(0);
    TORCH_CHECK(k % groups == 0 && k / groups >= 32, "K must be whole groups of at least 32");
    TORCH_CHECK(scales.size(0) == groups && scales.size(1) == n, "scales must be (groups, N)");
    TORCH_CHECK(qzeros.size(1) == n / 8, "qzeros must be (groups, N / 8)");

    const int* g_idx_ptr = nullptr;
    if (g_idx.numel() > 0) {
        TORCH_CHECK(g_idx.is_cuda() && g_idx.scalar_type() == at::kInt && g_idx.numel() == k,
                    "g_idx: (K) int32 act-order, or empty");
        g_idx_ptr = g_idx.data_ptr<int>();
    }

    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    at::Tensor out = at::zeros({m, n}, a.options());

    const tf::rocm::Gptq g{a.data_ptr(), reinterpret_cast<const uint32_t*>(qweight.data_ptr()),
                           reinterpret_cast<const uint32_t*>(qzeros.data_ptr()), scales.data_ptr(),
                           g_idx_ptr, out.data_ptr(), static_cast<int>(m), static_cast<int>(n),
                           static_cast<int>(k), static_cast<int>(groups),
                           use_v2_format ? tf::rocm::kZeroV2 : tf::rocm::kZeroV1};
    const hipError_t err = prefill ? tf::rocm::launch_gptq_prefill(g, stream)
                                   : tf::rocm::launch_gptq_decode(g, stream);
    TORCH_CHECK(err == hipSuccess, "RDNA W4A16 launch failed: ", hipGetErrorString(err));
    return out;
}

at::Tensor moe_gptq(const at::Tensor& a, const at::Tensor& qweight, const at::Tensor& qzeros,
                    const at::Tensor& scales, const at::Tensor& items, const at::Tensor& members,
                    int64_t rows, int64_t slots, int64_t epi, int64_t block_m, bool use_v2_format,
                    double limit) {
    TORCH_CHECK(a.is_cuda() && a.is_contiguous() && a.dim() == 2 && a.scalar_type() == at::kBFloat16,
                "a: (rows, K) bf16, contiguous");
    TORCH_CHECK(qweight.is_cuda() && qweight.is_contiguous() && qweight.dim() == 4 &&
                    qweight.scalar_type() == at::kInt,
                "qweight: (mats, experts, K / 8, N) int32, contiguous");
    TORCH_CHECK(qzeros.is_cuda() && qzeros.dim() == 4 && qzeros.scalar_type() == at::kInt,
                "qzeros: (mats, experts, groups, N / 8) int32");
    TORCH_CHECK(scales.is_cuda() && scales.dim() == 4 && scales.scalar_type() == at::kHalf,
                "scales: (mats, experts, groups, N) fp16");
    TORCH_CHECK(items.is_cuda() && items.dim() == 2 && items.size(1) == 3 && items.scalar_type() == at::kInt,
                "items: (items, 3) int32");
    TORCH_CHECK(members.is_cuda() && members.dim() == 1 && members.scalar_type() == at::kInt,
                "members: (pairs) int32");
    TORCH_CHECK(epi >= 0 && epi <= 2, "epi is 0 (fp32), 1 (relu^2) or 2 (SwiGLU)");

    const int64_t experts = qweight.size(1), k = a.size(1), n = qweight.size(3), groups = qzeros.size(2);
    TORCH_CHECK(qweight.size(0) == (epi == 2 ? 2 : 1), "SwiGLU wants a fused gate+up pair");
    TORCH_CHECK(k % 32 == 0 && n % 8 == 0 && k % groups == 0 && k / groups >= 32,
                "K whole groups of 32+, K a multiple of 32, N a multiple of 8");
    TORCH_CHECK(scales.size(0) == qweight.size(0) && scales.size(1) == experts && scales.size(2) == groups &&
                    scales.size(3) == n && qzeros.size(0) == qweight.size(0) &&
                    qzeros.size(1) == experts && qzeros.size(3) == n / 8,
                "scales and qzeros must match qweight");
    TORCH_CHECK(members.numel() >= rows * slots, "members must cover every pair");

    c10::cuda::CUDAGuard guard(a.device());
    auto stream = c10::cuda::getCurrentCUDAStream();
    at::Tensor out = at::empty({rows * slots, n},
                               a.options().dtype(epi == 0 ? at::kFloat : at::kBFloat16));

    const tf::rocm::MoeGptq g{a.data_ptr(), out.data_ptr(),
                              reinterpret_cast<const uint32_t*>(qweight.data_ptr()),
                              reinterpret_cast<const uint32_t*>(qzeros.data_ptr()), scales.data_ptr(),
                              items.data_ptr<int>(), members.data_ptr<int>(), static_cast<int>(items.size(0)),
                              static_cast<int>(n), static_cast<int>(k), static_cast<int>(groups),
                              static_cast<int>(slots),
                              use_v2_format ? tf::rocm::kZeroV2 : tf::rocm::kZeroV1,
                              static_cast<float>(limit)};
    const hipError_t err = tf::rocm::launch_moe_gptq(g, static_cast<int>(experts), static_cast<int>(epi),
                                                     static_cast<int>(block_m), stream);
    TORCH_CHECK(err == hipSuccess, "RDNA W4A16 MoE launch failed: ", hipGetErrorString(err));
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gptq_matmul", &gptq_matmul);
    m.def("moe_gptq", &moe_gptq);
}
