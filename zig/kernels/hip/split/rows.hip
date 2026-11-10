// The split offload's boundary rows (docs/proposals/metal-tb4-split-moe.md): a hidden row crosses a host-mapped window as
// fp32 whatever either side's activation type is. Import widens nothing and rounds to nearest even into the activation
// kind; export widens exactly; consume does the import from a level-B stamped row, whose 16-byte groups carry three fp32
// values and the sequence number (zig/src/split/wire.zig), re-reading a group until its stamp is current.
//
// Window memory is written by another device and by the host, so every read of it must reach memory: reads here are
// system-scope atomics (scalars) or one volatile 16-byte vector load (a group), which the backend emits with the
// coherence bits of the chip (glc on gfx9, glc dlc on gfx10/11, scope:SCOPE_SYS on gfx12) and never merges or reorders.
// A group's stamp and data come from the one load, so no torn group is accepted as long as the memory system returns a
// 16-byte aligned access as one unit; that is [INFERENCE] until a card run.

#include "ops/common.hpp"

namespace {

typedef unsigned v4u __attribute__((ext_vector_type(4)));

constexpr int kValues = 3;  // fp32 payload values a 16-byte group
constexpr int kSleep = 8;   // s_sleep between two reads of a stale group: 8 * 64 clocks

__device__ __forceinline__ float window_load(const float* p) {
    return __uint_as_float(
        __hip_atomic_load(reinterpret_cast<const unsigned*>(p), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM));
}

__device__ __forceinline__ void window_store(unsigned* p, unsigned v) {
    __hip_atomic_store(p, v, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
}

}  // namespace

extern "C" {

// dst[i] = src[i] rounded to `kind` (1 fp16, 2 bf16, 0 a copy), i < n; src is an fp32 row in a window. Grid (ceil(n/256)).
__global__ void __launch_bounds__(256) tf_split_import_row(const float* src, void* dst, int n, int kind) {
    const long long i = gid();
    if (i < n) store(dst, kind, i, window_load(src + i));
}

// dst[i] = src[i] widened to fp32, i < n; dst is an fp32 row in a window. Grid (ceil(n/256)).
__global__ void __launch_bounds__(256) tf_split_export_row(const void* src, float* dst, int n, int kind) {
    const long long i = gid();
    if (i < n) window_store(reinterpret_cast<unsigned*>(dst + i), __float_as_uint(load(src, kind, i)));
}

// dst[i] = value i of the stamped row `src` rounded to `kind`, i < n: ceil(n/3) groups of three fp32 values and a u32
// stamp. A group whose stamp is not `seq` is read again after a short sleep, at most `retries` more times; a group still
// stale after that is converted as last read and makes status 1 (else 0). One workgroup of 1024 lanes, a group a lane a
// pass, so a row of hidden up to 3072 is one pass. status is written once, by lane 0, after every lane is done.
__global__ void __launch_bounds__(1024)
tf_split_consume_stamped(const v4u* src, unsigned seq, void* dst, int n, int kind, unsigned* status, unsigned retries) {
    __shared__ int stale;
    if (threadIdx.x == 0) stale = 0;
    __syncthreads();
    const int groups = (n + kValues - 1) / kValues;
    for (int g = threadIdx.x; g < groups; g += blockDim.x) {
        const volatile v4u* at = src + g;
        v4u v = *at;
        for (unsigned tries = 0; v.w != seq && tries < retries; ++tries) {
            __builtin_amdgcn_s_sleep(kSleep);
            v = *at;
        }
        if (v.w != seq) atomicOr(&stale, 1);
        const int first = g * kValues;
        store(dst, kind, first, __uint_as_float(v.x));
        if (first + 1 < n) store(dst, kind, first + 1, __uint_as_float(v.y));
        if (first + 2 < n) store(dst, kind, first + 2, __uint_as_float(v.z));
    }
    __syncthreads();
    if (threadIdx.x == 0) window_store(status, stale ? 1u : 0u);
}

}  // extern "C"
