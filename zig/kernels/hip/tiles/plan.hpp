#pragma once

// Routed plan, any weight format: item z = (expert, first, count) takes pairs members[first..first + count).

namespace tf {
namespace rocm {

struct Routing {
    const int* items = nullptr;  // (count, 3) int32; nullptr is one plain (m, n) product
    const int* members = nullptr;
    int x_div = 1;
    int first = 0;  // set by the kernel from its item
#if defined(TF_EXPERT_PTRS)
    // Offloaded experts (caps `expert_ptrs`): nonzero, item expert e's weights are the blob at ptab[2e] + ptab_off (words, scales, biases in that
    // order, each of the projection's own size), not the stack's e-th slice: a VRAM cache slot's address or a host-tier address. ptab[2e + 1] is
    // 0, or the address of a VRAM cache line the stream tile writes the blob it reads into (read-through admission: the miss's one trip over the link
    // is also its admission); the kernel keeps the byte distance between the two blobs in `dst_delta`.
    const unsigned long long* ptab = nullptr;
    long long ptab_off = 0;
    long long dst_delta = 0;  // set by the kernel from its item
#endif
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
