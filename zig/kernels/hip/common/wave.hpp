#pragma once

// The tiles split a block into groups of 32 threads (a "wave" in their comments): one wavefront on RDNA, half of one on gfx900,
// whose wavefronts have 64 lanes. Shuffles take width 32, so they stay inside a group; this is the group's index.

#if !defined(TF_WAVE)
#error "TF_WAVE (the caps table's wavefront width) comes from the build"
#endif

namespace tf {
namespace rocm {

// `tid >> 5` of a thread, kept in a scalar register where the group is the wavefront (RDNA); on wave64 the two groups of a wavefront
// differ, so readfirstlane would hand the upper one the lower one's index.
__device__ __forceinline__ int group32(int tid) {
#if TF_WAVE == 32
    return __builtin_amdgcn_readfirstlane(tid >> 5);
#else
    return tid >> 5;
#endif
}

}  // namespace rocm
}  // namespace tf
