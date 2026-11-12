#pragma once

// The tiles split a block into groups of 32 threads: a wavefront on RDNA, half of one on gfx900 (wave64).

#if !defined(TF_WAVE)
#error "TF_WAVE (the caps table's wavefront width) comes from the build"
#endif

namespace tf {
namespace rocm {

// `tid >> 5`, scalar where a group is a wavefront (wave32); on wave64 a wavefront holds two groups, so no readfirstlane.
__device__ __forceinline__ int group32(int tid) {
#if TF_WAVE == 32
    return __builtin_amdgcn_readfirstlane(tid >> 5);
#else
    return tid >> 5;
#endif
}

}  // namespace rocm
}  // namespace tf
