#pragma once

// Epilogues: what a tile does with the finished sums of its outputs. The GEMM tiles hand over one fp32 value at a time; the
// stream tile a lane's columns, rounded to the activation type or, for the routed gate and up, turned into the activation.

#include <hip/hip_fp16.h>

#include <type_traits>

#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// The product in fp32 at its (row, col) of out, the plan's pair row for a routed item.
struct F32Out {
    template <class Args>
    __device__ static void store(const Args& a, int row, int col, float v) {
        a.out[out_row(a, row) * a.n + col] = v;
    }
};

// A float as the activation type, rounded to nearest even.
template <typename T>
__device__ inline typename T::elem narrow(float v) {
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        return __float2half_rn(v);
    } else {
        const uint32_t u = __float_as_uint(v);
        const uint32_t bits = v != v ? 0x7fc0u : (u + (((u >> 16) & 1u) + 0x7fffu)) >> 16;
        return __builtin_bit_cast(typename T::elem, static_cast<unsigned short>(bits));
    }
}

// A float product rounded to the activation type; FP16 is one rounding of the exact product (a mixed-precision multiply-add
// with a zero addend), spelled out so that no code layout makes it a rounding to fp32 and another to FP16.
template <typename T>
__device__ inline typename T::elem narrow_product(float a, float b) {
    if constexpr (std::is_same_v<typename T::elem, __half>) {
        uint32_t bits;
        asm("v_fma_mixlo_f16 %0, %1, %2, 0" : "=v"(bits) : "v"(a), "v"(b));
        return __builtin_bit_cast(typename T::elem, static_cast<unsigned short>(bits));
    } else {
        return narrow<T>(a * b);
    }
}

// silu(gate) * up in the activation type, both clamped by limit first when it is above 0 (moe_act_kernel's arithmetic).
template <typename T>
__device__ inline typename T::elem swiglu(float g, float u, float limit) {
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fminf(fmaxf(u, -limit), limit);
    }
    return narrow_product<T>(g / (1.f + expf(-g)), u);
}

// The stream tile's epilogues take the CB sums of a lane's columns for one row (`v`), the column `c` of them and the width
// of the paired output.

// The sum rounded to the activation type where `out16` is set (the decode tails' own rounding), else stored in fp32.
template <class T>
struct StreamRound {
    static constexpr bool kPair = false;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int, float) {
        if (a.out16 != nullptr) {
            static_cast<typename T::elem*>(static_cast<void*>(a.out16))[row * a.n + col] = narrow<T>(v[c]);
        } else {
            a.out[row * a.n + col] = v[c];
        }
    }
};

// A lane's columns are CB / 2 gate columns and the same columns of the up half: the output is silu(gate) * up in the
// activation type, `pair_cols` wide.
template <class T>
struct StreamSwiglu {
    static constexpr bool kPair = true;

    template <int CB, class Args>
    __device__ static void put(const Args& a, long long row, int col, const float (&v)[CB], int c, int pair_cols, float limit) {
        constexpr int CH = CB / 2;
        static_cast<typename T::elem*>(static_cast<void*>(a.out16))[row * pair_cols + col] = swiglu<T>(v[c], v[c + CH], limit);
    }
};

}  // namespace rocm
}  // namespace tf
