// WMMA tiles for two or four 8-bit projections that share an input: one activation load a K step.

#include <stdexcept>
#include <string>

#include "affine_api.hpp"
#include "affine_wmma.hpp"

namespace tf {
namespace rocm {

#if TENSORFOLD_RDNA_WMMA

// One wave, two outputs. A is loaded once per K step and both sides use it.
__global__ void __launch_bounds__(32) affine_wmma_pair_kernel(Affine a, AffineSide first, AffineSide second) {
    using namespace rocwmma;
    int n0 = blockIdx.x * kTile;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    AffineSide sides[2] = {first, second};

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[2][kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[2][kTile];
    __shared__ float bcol[2][kTile];

    for (int side = 0; side < 2; ++side) {
        for (int e = lane; e < kTile * kTile; e += 32) acc[side][e] = 0.f;
    }
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
            for (int side = 0; side < 2; ++side) {
                long long at = static_cast<long long>(col) * groups + group;
                scol[side][lane] = col < sides[side].n ? sides[side].scale[at] : 0.f;
                bcol[side][lane] = col < sides[side].n ? sides[side].bias[at] : 0.f;
            }
        }
        __syncthreads();

        FragC part_even[2], part_odd[2];
        for (int side = 0; side < 2; ++side) {
            fill_fragment(part_even[side], 0.f);
            fill_fragment(part_odd[side], 0.f);
        }
        // Both sides of this group are read before either WMMA, one chunk of 64 codes at a time.
        for (int chunk = 0; chunk < a.group; chunk += 64) {
            int kbase = base + chunk;
            int left = a.group - chunk;
            uint2 p0 = load_b8(sides[0].words, words, n0, kbase, sides[0].n, lane);
            uint2 q0 = load_b8(sides[1].words, words, n0, kbase, sides[1].n, lane);
            uint2 p1 = make_uint2(0u, 0u), p2 = make_uint2(0u, 0u), p3 = make_uint2(0u, 0u);
            uint2 q1 = make_uint2(0u, 0u), q2 = make_uint2(0u, 0u), q3 = make_uint2(0u, 0u);
            if (left > 16) {
                p1 = load_b8(sides[0].words, words, n0, kbase + 16, sides[0].n, lane);
                q1 = load_b8(sides[1].words, words, n0, kbase + 16, sides[1].n, lane);
            }
            if (left > 32) {
                p2 = load_b8(sides[0].words, words, n0, kbase + 32, sides[0].n, lane);
                q2 = load_b8(sides[1].words, words, n0, kbase + 32, sides[1].n, lane);
            }
            if (left > 48) {
                p3 = load_b8(sides[0].words, words, n0, kbase + 48, sides[0].n, lane);
                q3 = load_b8(sides[1].words, words, n0, kbase + 48, sides[1].n, lane);
            }
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p0, q0, m0, kbase, a.m, a.k,
                                              lane, replay, true);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p1, q1, m0, kbase + 16, a.m,
                                              a.k, lane, replay, left > 16);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p2, q2, m0, kbase + 32, a.m,
                                              a.k, lane, replay, left > 32);
            consume_pair<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, p3, q3, m0, kbase + 48, a.m,
                                              a.k, lane, replay, left > 48);
        }
        for (int side = 0; side < 2; ++side) {
            store_matrix_sync(dot_even, part_even[side], kTile, mem_row_major);
            if (replay) store_matrix_sync(dot_odd, part_odd[side], kTile, mem_row_major);
            __syncthreads();
            for (int e = lane; e < kTile * kTile; e += 32) {
                int r = e / kTile;
                int c = e % kTile;
                float dot = (replay && (r & 1)) ? dot_odd[(r ^ 1) * kTile + c] : dot_even[e];
                acc[side][e] = fmaf(dot, scol[side][c], acc[side][e]);
                acc[side][e] = fmaf(xsum[r], bcol[side][c], acc[side][e]);
            }
            __syncthreads();
        }
    }

    for (int side = 0; side < 2; ++side) {
        for (int e = lane; e < kTile * kTile; e += 32) {
            int grow = m0 + e / kTile;
            int gcol = n0 + e % kTile;
            if (grow < a.m && gcol < sides[side].n) {
                sides[side].out[static_cast<long long>(grow) * sides[side].n + gcol] = acc[side][e];
            }
        }
    }
}

// One block owns one 16-column tile of one side. The side is uniform inside the block.
__device__ void pick_group(int tile, GroupSide s0, GroupSide s1, GroupSide s2, GroupSide s3, const uint32_t*& words,
                           GroupTable& scale, GroupTable& bias, float*& out, int& n, int& n0) {
    int b1 = s0.tiles;
    int b2 = b1 + s1.tiles;
    int b3 = b2 + s2.tiles;
    GroupSide side = s0;
    int local = tile;
    if (tile >= b1 && tile < b2) {
        side = s1;
        local = tile - b1;
    } else if (tile >= b2 && tile < b3) {
        side = s2;
        local = tile - b2;
    } else if (tile >= b3) {
        side = s3;
        local = tile - b3;
    }
    words = side.words;
    scale = side.scale;
    bias = side.bias;
    out = side.out;
    n = side.n;
    n0 = local * kTile;
}

__global__ void __launch_bounds__(32) affine_wmma_group_kernel(Affine a, GroupSide s0, GroupSide s1, GroupSide s2,
                                                              GroupSide s3) {
    using namespace rocwmma;
    int m0 = blockIdx.y * kTile;
    int lane = threadIdx.x;
    int words_per_row = a.k * a.bits / 32;
    int groups = a.k / a.group;
    const hip_bfloat16* x = static_cast<const hip_bfloat16*>(a.x);
    const uint32_t* side_words = nullptr;
    GroupTable side_scale{};
    GroupTable side_bias{};
    float* side_out = nullptr;
    int side_n = 0;
    int n0 = 0;
    pick_group(blockIdx.x, s0, s1, s2, s3, side_words, side_scale, side_bias, side_out, side_n, n0);

    __shared__ hip_bfloat16 As[kTile * kTile];
    __shared__ hip_bfloat16 AsOdd[kTile * kTile];
    __shared__ hip_bfloat16 Bs[kTile * kTile];
    __shared__ float dot_even[kTile * kTile];
    __shared__ float dot_odd[kTile * kTile];
    __shared__ float acc[kTile * kTile];
    __shared__ float xsum[kTile];
    __shared__ float scol[kTile];
    __shared__ float bcol[kTile];

    for (int e = lane; e < kTile * kTile; e += 32) acc[e] = 0.f;
    const bool replay = m0 + 1 < a.m;

    using FragA = fragment<matrix_a, kTile, kTile, kTile, bfloat16_t, row_major>;
    using FragB = fragment<matrix_b, kTile, kTile, kTile, bfloat16_t, col_major>;
    using FragC = fragment<accumulator, kTile, kTile, kTile, float>;

    for (int group = 0; group < groups; ++group) {
        int base = group * a.group;
        uint2 b0 = load_b8(side_words, words_per_row, n0, base, side_n, lane);
        uint2 b1 = load_b8(side_words, words_per_row, n0, base + 16, side_n, lane);
        uint2 b2 = a.group > 32 ? load_b8(side_words, words_per_row, n0, base + 32, side_n, lane) : make_uint2(0u, 0u);
        uint2 b3 = a.group > 48 ? load_b8(side_words, words_per_row, n0, base + 48, side_n, lane) : make_uint2(0u, 0u);
        uint2 b4 = a.group > 64 ? load_b8(side_words, words_per_row, n0, base + 64, side_n, lane) : make_uint2(0u, 0u);
        uint2 b5 = a.group > 80 ? load_b8(side_words, words_per_row, n0, base + 80, side_n, lane) : make_uint2(0u, 0u);
        uint2 b6 = a.group > 96 ? load_b8(side_words, words_per_row, n0, base + 96, side_n, lane) : make_uint2(0u, 0u);
        uint2 b7 = a.group > 112 ? load_b8(side_words, words_per_row, n0, base + 112, side_n, lane)
                                 : make_uint2(0u, 0u);
        write_xsum(x, side_scale, side_bias, xsum, scol, bcol, lane, m0, n0, base, group, a.group, groups, a.m, a.k,
                   side_n);
        __syncthreads();

        FragC part_even, part_odd;
        fill_fragment(part_even, 0.f);
        fill_fragment(part_odd, 0.f);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b0, m0, base, a.m, a.k, lane, replay,
                                      true);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b1, m0, base + 16, a.m, a.k, lane,
                                      replay, true);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b2, m0, base + 32, a.m, a.k, lane,
                                      replay, a.group > 32);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b3, m0, base + 48, a.m, a.k, lane,
                                      replay, a.group > 48);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b4, m0, base + 64, a.m, a.k, lane,
                                      replay, a.group > 64);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b5, m0, base + 80, a.m, a.k, lane,
                                      replay, a.group > 80);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b6, m0, base + 96, a.m, a.k, lane,
                                      replay, a.group > 96);
        consume8<FragA, FragB, FragC>(part_even, part_odd, As, AsOdd, Bs, x, b7, m0, base + 112, a.m, a.k, lane,
                                      replay, a.group > 112);

        store_matrix_sync(dot_even, part_even, kTile, mem_row_major);
        if (replay) store_matrix_sync(dot_odd, part_odd, kTile, mem_row_major);
        __syncthreads();
        fma_group(acc, dot_even, dot_odd, xsum, scol, bcol, replay, lane);
        __syncthreads();
    }

    for (int e = lane; e < kTile * kTile; e += 32) {
        int grow = m0 + e / kTile;
        int gcol = n0 + e % kTile;
        if (grow < a.m && gcol < side_n) side_out[static_cast<long long>(grow) * side_n + gcol] = acc[e];
    }
}

hipError_t launch_affine_wmma_group(const Affine& a, GroupSide s0, GroupSide s1, GroupSide s2, GroupSide s3,
                                    int total_tiles, hipStream_t stream) {
    if (a.m < 1 || a.bits != 8 || a.group % kTile || a.k % a.group || total_tiles < 1) return hipErrorInvalidValue;
    dim3 block(32);
    dim3 grid(total_tiles, (a.m + kTile - 1) / kTile);
    affine_wmma_group_kernel<<<grid, block, 0, stream>>>(a, s0, s1, s2, s3);
    return hipGetLastError();
}

hipError_t launch_affine_wmma_pair(const Affine& a, AffineSide first, AffineSide second, hipStream_t stream) {
    if (a.m < 1 || first.n < 1 || first.n != second.n || a.group % kTile || a.k % a.group) {
        return hipErrorInvalidValue;
    }
    dim3 block(32);
    dim3 grid((first.n + kTile - 1) / kTile, (a.m + kTile - 1) / kTile);
    affine_wmma_pair_kernel<<<grid, block, 0, stream>>>(a, first, second);
    return hipGetLastError();
}

#else

hipError_t launch_affine_wmma_pair(const Affine&, AffineSide, AffineSide, hipStream_t) {
    return hipErrorNotSupported;
}

hipError_t launch_affine_wmma_group(const Affine&, GroupSide, GroupSide, GroupSide, GroupSide, int, hipStream_t) {
    return hipErrorNotSupported;
}

#endif

}  // namespace rocm
}  // namespace tf

void affine_group_launch(const void* x, const void* const* words, const void* const* scale, const void* const* bias,
                         int scale_kind, void* const* out, const int* ns, int nsides, int m, int k, int bits,
                         int group, void* stream) {
    if (nsides < 1 || nsides > 4) throw std::runtime_error("affine group kernel: 1 to 4 sides");
    tf::rocm::GroupSide sides[4] = {};
    int total = 0;
    for (int i = 0; i < nsides; ++i) {
        sides[i].words = static_cast<const uint32_t*>(words[i]);
        sides[i].scale = {scale[i], scale_kind};
        sides[i].bias = {bias[i], scale_kind};
        sides[i].out = static_cast<float*>(out[i]);
        sides[i].n = ns[i];
        sides[i].tiles = (ns[i] + 15) / 16;
        total += sides[i].tiles;
    }
    tf::rocm::Affine a{x, nullptr, {}, {}, nullptr, m, 0, k, bits, group, 0};
    hipError_t err = tf::rocm::launch_affine_wmma_group(a, sides[0], sides[1], sides[2], sides[3], total,
                                                        static_cast<hipStream_t>(stream));
    if (err != hipSuccess) throw std::runtime_error(std::string("affine group kernel: ") + hipGetErrorString(err));
}

void affine_pair_launch(const void* x, const void* words0, const void* scale0, const void* bias0, void* out0,
                        const void* words1, const void* scale1, const void* bias1, void* out1, int scale_kind, int m,
                        int n, int k, int bits, int group, void* stream) {
    tf::rocm::Affine a{x, nullptr, {}, {}, nullptr, m, n, k, bits, group, 0};
    tf::rocm::AffineSide first{static_cast<const uint32_t*>(words0), {scale0, scale_kind}, {bias0, scale_kind},
                               static_cast<float*>(out0), n};
    tf::rocm::AffineSide second{static_cast<const uint32_t*>(words1), {scale1, scale_kind}, {bias1, scale_kind},
                                static_cast<float*>(out1), n};
    hipError_t err = tf::rocm::launch_affine_wmma_pair(a, first, second, static_cast<hipStream_t>(stream));
    if (err != hipSuccess) throw std::runtime_error(std::string("affine pair kernel: ") + hipGetErrorString(err));
}
