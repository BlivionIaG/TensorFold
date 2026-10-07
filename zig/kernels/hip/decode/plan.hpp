#pragma once

#include <hip/hip_runtime.h>

// A lane round's plan as the kernels read it from device memory: each row's position and slot, each slot's rows and
// its caches. Nothing of a stream is a launch argument, so one launch (or graph) serves any streams, positions and caches.
struct PlanArgs {
    const int* pos;                   // per row: its position
    const int* slot;                  // per row: its slot
    const int* first;                 // per slot: its first row
    const int* count;                 // per slot: its rows
    const unsigned long long* desc;   // per slot: the address of its descriptor
    const unsigned long long* snaps;  // per layer: the round's conv and DeltaNet snapshots, a row each
};

// A slot's descriptor: [0] positions its caches hold, [1] its last kept final row, then per layer two addresses
// (keys, values of an attention layer; conv window, DeltaNet state of a linear one).
__device__ __forceinline__ const unsigned long long* plan_desc(const PlanArgs& p, int slot) {
    return reinterpret_cast<const unsigned long long*>(p.desc[slot]);
}

__device__ __forceinline__ int plan_first(int layer) { return 2 + 2 * layer; }

__device__ __forceinline__ int plan_second(int layer) { return 3 + 2 * layer; }
