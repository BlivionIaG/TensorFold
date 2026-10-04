// Schedule dispatch: auto is the dot2 family (FP16 on RDNA2, BF16 on gfx11 / gfx12), gemv the scalar reference.

#include <stdexcept>
#include <string>

#include "affine.hpp"
#include "affine_api.hpp"

namespace tf {
namespace rocm {

hipError_t launch_affine(const Affine& a, int schedule, hipStream_t stream) {
    if (a.fp16) {
#if TENSORFOLD_BF16_DOT2
        return hipErrorNotSupported;
#else
        return launch_affine_dot2(a, schedule == 1, stream);
#endif
    }
#if TENSORFOLD_BF16_DOT2
    if (schedule == 0 && (a.group == 32 || a.group == 64 || a.group == 128) && a.k % a.group == 0 && a.k % 16 == 0) {
        return launch_affine_dot2_bf16(a, stream);
    }
#endif
    return launch_affine_gemv(a, stream);
}

}  // namespace rocm
}  // namespace tf

void affine_launch(const void* x, const void* words, const void* scale, const void* bias, int scale_kind, void* out,
                   int m, int n, int k, int bits, int group, int schedule, int fp16, void* stream, float* partial,
                   int splits, int out_half) {
    tf::rocm::Affine a{x, static_cast<const uint32_t*>(words), {scale, scale_kind}, {bias, scale_kind},
                       static_cast<float*>(out), m, n, k, bits, group, fp16};
    if (out_half) {
        if (!fp16 || schedule != 0 || splits > 1) {
            throw std::runtime_error("affine kernel: fp16 output is the decode tile");
        }
        a.out = nullptr;
        a.out16 = static_cast<__half*>(out);
    }
    hipStream_t hip_stream = static_cast<hipStream_t>(stream);
    hipError_t err = (partial != nullptr && splits > 1)
                         ? tf::rocm::launch_affine_dot2_split(a, partial, splits, hip_stream)
                         : tf::rocm::launch_affine(a, schedule, hip_stream);
    if (err != hipSuccess) throw std::runtime_error(std::string("affine kernel: ") + hipGetErrorString(err));
}
