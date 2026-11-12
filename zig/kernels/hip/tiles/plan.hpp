#pragma once

// Routed plan, any weight format: item z = (expert, first, count) takes pairs members[first..first + count).

// Expert offload (caps `expert_ptrs`): only the tiles built with TF_EXPERT_PTRS read a pointer table and take `Affine::ptrs`.

namespace tf {
namespace rocm {

#if defined(TF_EXPERT_PTRS)
// Offloaded experts: `ptab` set, expert e's blob is at ptab[2e] + ptab_off; ptab[2e + 1] names a line the tile also fills.
struct ExpertPtrs {
    const unsigned long long* ptab = nullptr;
    long long ptab_off = 0;
    long long dst_delta = 0;  // set by the kernel from its item
};
#endif

struct Routing {
    const int* items = nullptr;  // (count, 3) int32; nullptr is one plain (m, n) product
    const int* members = nullptr;
    int x_div = 1;
    int first = 0;  // set by the kernel from its item
};

// Row r of this block's x and out: the plain matrix's r, or the routed item's r-th pair.
template <class Args>
__device__ inline long long x_row(const Args& a, int r) {
    return a.route.items ? a.route.members[a.route.first + r] / a.route.x_div : r;
}

template <class Args>
__device__ inline long long out_row(const Args& a, int r) {
    return a.route.items ? a.route.members[a.route.first + r] : r;
}

}  // namespace rocm
}  // namespace tf
