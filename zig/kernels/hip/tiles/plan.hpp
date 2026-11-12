#pragma once

// Routed plan, any weight format: item z = (expert, first, count) takes pairs members[first..first + count).
//
// Expert offload (caps `expert_ptrs`, gfx900): TF_EXPERT_PTRS builds the code that reads a pointer table (mlx.hpp take_item, stream.hpp read-through
// admission) and adds `Affine::ptrs`, the last member of the arguments, for it. Only the code object for offloaded stacks (tiles/dot2_tiles_ptr.hip)
// is built with it: the code costs the other tiles VGPRs and occupancy (dense Qwen3.5-4B decode on the Vega: 81.4 -> 77.0 tok/s when every tile
// carried it), and the bigger arguments cost every launch, so a dense stack's launches never go through it and its tiles keep the arguments they had.

namespace tf {
namespace rocm {

#if defined(TF_EXPERT_PTRS)
// Offloaded experts: nonzero `ptab`, item expert e's weights are the blob at ptab[2e] + ptab_off (words, scales, biases in that order, each of the
// projection's own size), not the stack's e-th slice: a VRAM cache slot's address or a host-tier address. ptab[2e + 1] is 0, or the address of a VRAM
// cache line the stream tile writes the blob it reads into (read-through admission: the miss's one trip over the link is also its admission); the
// kernel keeps the byte distance between the two blobs in `dst_delta`.
//
// The 64 bytes before `ptab` are the table's header (four u64): [0] the address of the window's slot flags (one u32 a window slot, nonzero once the slot's
// blob has been copied in; 0: no flags, the host has copied every blob before it answered), [1] the address of the window's first slot, [2] the bytes of
// all its slots, [3] the bytes of one slot. An entry whose blob is in a window slot waits for that slot's flag before any of its bytes is read: the host
// answers a layer's request as soon as it has decided where the experts sit and fills the slots after (hip/expert_store.zig, LATENCY.md M3). Entries
// that name a cache line or the prompt's scratch are outside the window and wait for nothing.
struct ExpertPtrs {
    const unsigned long long* ptab = nullptr;
    long long ptab_off = 0;
    long long dst_delta = 0;  // set by the kernel from its item
};

// Waits until the host has copied the blob of the window slot `src` points at, when the table's header has flags and `src` is a slot of the window.
__device__ inline void expert_blob_ready(const unsigned long long* ptab, unsigned long long src) {
    const unsigned long long* head = ptab - 8;
    const unsigned long long flags = head[0];
    if (flags == 0) return;
    const unsigned long long at = src - head[1];
    if (at >= head[2]) return;
    const unsigned* flag = reinterpret_cast<const unsigned*>(flags) + at / head[3];
    while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) == 0) __builtin_amdgcn_s_sleep(2);
}
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
