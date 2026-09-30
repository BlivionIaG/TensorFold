// RDNA3, RDNA 3.5 and RDNA4 tile. One wave owns a 16x16 output. Each K step unpacks
// a 16x16 patch of codes into LDS for the WMMA; the packed words stay the only copy
// of the weight. That patch is staging for one K step, not a stored matrix.
// RDNA 3.5 uses the same gfx11 WMMA as RDNA3. rocWMMA hides that layout against gfx12.
//
// A row computed on its own always sits in row 0 of the tile. On gfx1100 the even
// rows of this 16x16x16 BF16 WMMA share row 0's rounding and the odd rows differ
// by one ulp, which would change an odd row's bits when a window grows. Those
// rows are replayed in the even rows of a second WMMA. RDNA4 uses the same
// replay; a part whose pairing differs fails the row-invariance test.
//
// Compiled without TENSORFOLD_RDNA_WMMA (RDNA2), the launch refuses and
// the GEMV is the only schedule.

#include "affine.hpp"
#include "affine_api.hpp"
#include "arch.hpp"

#include <stdexcept>
#include <string>

#if TENSORFOLD_RDNA_WMMA
// rocWMMA 7.2 names gfx1100–1102, gfx1150 and gfx1151. gfx1103, gfx1152 and
// gfx1153 are that same gfx11 WMMA, and the header rejects any other name.
// Calling them gfx1150 selects the gfx11 builtin; the compiler target is unchanged.
#if defined(__gfx1103__) || defined(__gfx1152__) || defined(__gfx1153__)
#define __gfx1150__ 1
#endif
#include <rocwmma/rocwmma.hpp>
#endif

namespace tf {
namespace rocm {

#if TENSORFOLD_RDNA_WMMA

constexpr int kTile = 16;

__global__ void __launch_bounds__(32) affine_wmma_kernel(Affine a) {
    using namespace rocwmma;
    int n0 = blockIdx.x * kTile;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];  // logical row r parked at physical row r ^ 1
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kTile];
    __shared__ float bcol[kTile];

    for (int e = lane; e < kTile * kTile; e += 32) acc[e] = 0.f;
    // Odd rows of this tile are past M: the single WMMA already uses row 0's rounding.
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        if (lane < kTile) {
            float sum = 0.f;
            int row = m0 + lane;
            if (row < a.m) {
                const hip_bfloat16* xrow = x + static_cast<long long>(row) * a.k + base;
                for (int t = 0; t < a.group; ++t) sum += static_cast<float>(xrow[t]);
            }
            xsum[lane] = sum;
            int col = n0 + lane;
            scol[lane] = col < a.n ? a.scale[static_cast<long long>(col) * groups + group] : 0.f;
            bcol[lane] = col < a.n ? a.bias[static_cast<long long>(col) * groups + group] : 0.f;
        }
        __syncthreads();

        FragC part_even, part_odd;
        fill_fragment(part_even, 0.f);
        fill_fragment(part_odd, 0.f);
        for (int step = 0; step < a.group; step += kTile) {
            int k0 = base + step;
            for (int e = lane; e < kTile * kTile; e += 32) {
                int r = e / kTile;
                int c = e % kTile;
                int grow = m0 + r;
                int gcol = k0 + c;
                hip_bfloat16 xv = hip_bfloat16(0.f);
                if (grow < a.m && gcol < a.k) xv = x[static_cast<long long>(grow) * a.k + gcol];
                As[r * kTile + c] = xv;
                if (replay) AsOdd[(r ^ 1) * kTile + c] = xv;

                int ncol = n0 + r;
                int nk = k0 + c;
                hip_bfloat16 qv = hip_bfloat16(0.f);
                if (ncol < a.n && nk < a.k) {
                    const uint32_t* wrow = a.words + static_cast<long long>(ncol) * words;
                    qv = hip_bfloat16(code_bf16(affine_code(wrow, nk, a.bits, words)));
                }
                Bs[r * kTile + c] = qv;  // col-major (k, n): index n * 16 + k
            }
            __syncthreads();
            FragA fa;
            FragB fb;
            load_matrix_sync(fa, As, kTile);
            load_matrix_sync(fb, Bs, kTile);
            mma_sync(part_even, fa, fb, part_even);
            if (replay) {
                FragA fa_odd;
                load_matrix_sync(fa_odd, AsOdd, kTile);
                mma_sync(part_odd, fa_odd, fb, part_odd);
            }
            __syncthreads();
        }

        store_matrix_sync(dot_even, part_even, kTile, mem_row_major);
        if (replay) store_matrix_sync(dot_odd, part_odd, kTile, mem_row_major);
        __syncthreads();
        for (int e = lane; e < kTile * kTile; e += 32) {
            int r = e / kTile;
            int c = e % kTile;
            // The replay's even physical row r ^ 1 holds logical row r.
            float dot = (replay && (r & 1)) ? dot_odd[(r ^ 1) * kTile + c] : dot_even[e];
            acc[e] = fmaf(dot, scol[c], acc[e]);
            acc[e] = fmaf(xsum[r], bcol[c], acc[e]);
        }
        __syncthreads();
    }

    for (int e = lane; e < kTile * kTile; e += 32) {
        int grow = m0 + e / kTile;
        int gcol = n0 + e % kTile;
        if (grow < a.m && gcol < a.n) a.out[static_cast<long long>(grow) * a.n + gcol] = acc[e];
    }
}

hipError_t launch_affine_wmma(const Affine& a) {
    // Tiles past M or N are filled with zeros, so a single row uses the same WMMA as a full tile.
    if (a.m < 1 || a.n < 1 || a.group % kTile || a.k % a.group) return hipErrorInvalidValue;
    dim3 block(32);
    dim3 grid((a.n + kTile - 1) / kTile, (a.m + kTile - 1) / kTile);
    affine_wmma_kernel<<<grid, block>>>(a);
    return hipGetLastError();
}

#else

hipError_t launch_affine_wmma(const Affine&) { return hipErrorNotSupported; }

#endif

hipError_t launch_affine(const Affine& a, int schedule) {
    if (a.fp16) {
        if (schedule == 2) return hipErrorNotSupported;
#if TENSORFOLD_RDNA_WMMA
        return hipErrorNotSupported;
#else
        return launch_affine_dot2(a);
#endif
    }
#if TENSORFOLD_RDNA_WMMA
    bool tile = schedule != 1 && a.group % 16 == 0 && a.k % a.group == 0;
#else
    bool tile = false;
#endif
    if (schedule == 2 && !tile) return hipErrorNotSupported;
    if (tile) return launch_affine_wmma(a);
    return launch_affine_gemv(a);
}

}  // namespace rocm
}  // namespace tf

void affine_launch(const void* x, const void* words, const void* scale, const void* bias, void* out, int m, int n,
                   int k, int bits, int group, int schedule, int fp16) {
    tf::rocm::Affine a{x, static_cast<const uint32_t*>(words), static_cast<const float*>(scale),
                       static_cast<const float*>(bias), static_cast<float*>(out), m, n, k, bits, group, fp16};
    hipError_t err = tf::rocm::launch_affine(a, schedule);
    if (err != hipSuccess) throw std::runtime_error(std::string("affine kernel: ") + hipGetErrorString(err));
}
