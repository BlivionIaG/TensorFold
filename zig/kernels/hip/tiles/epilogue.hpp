#pragma once

// Epilogues: what a tile does with the finished sums of its outputs. The GEMM tiles hand over one fp32 value at a time.
// An epilogue that rounds to the activation type, adds a bias, a residual or a norm is another class here.

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

}  // namespace rocm
}  // namespace tf
