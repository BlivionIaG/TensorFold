#pragma once

// The tile constants and launchers shared by the dot2 files.

#include <hip/hip_fp16.h>

#include "common/arch.hpp"
#include "common/dot2.hpp"
#include "quant/mlx.hpp"
#include "quant/mlx_pieces.hpp"

namespace tf {
namespace rocm {

constexpr int kLaneCols = 32;
constexpr int kLaneWaves = 8;
constexpr int kLaneRows = 8;
constexpr int kLaneGroupMax = 128;
constexpr int kBlockRows = 64;  // from here the GEMM tile beats the 128-row column tile

hipError_t ensure_byte_lut(hipStream_t stream);
// The stream tile for 9 to 16 rows (tiles/stream.hpp); false when the shape keeps the other tiles.
bool launch_affine_dot2_stream_wide(const Affine& a, hipStream_t stream, hipError_t* err);
// out16 (m, n / 2) = silu(gate) * up, clamped when limit > 0; hipErrorInvalidValue when the shape keeps two products.
hipError_t launch_affine_dot2_pair(const Affine& a, float limit, hipStream_t stream, int items = 1);
hipError_t launch_affine_dot2_lanes(const Affine& a, hipStream_t stream, int items = 1);
hipError_t launch_affine_dot2_block(const Affine& a, hipStream_t stream, int items = 1);
hipError_t launch_affine_dot2_block_old(const Affine& a, hipStream_t stream, int items = 1);

}  // namespace rocm
}  // namespace tf
