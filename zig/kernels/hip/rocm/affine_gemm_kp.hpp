#pragma once

// Prefill GEMM tile for a few rows: a lane owns a group of a column, so a wave reads a column's row in one contiguous run
// and the weights stream from many memory channels at once (columns whose rows are a power of two bytes apart camp on a
// few channels when every block walks them in step). An output has affine_gemm_block's bits: a group's dot2 chain and sum
// of x in ascending k, then the groups of a round, exchanged through LDS, folded in order with the same two fma.

#include "affine_gemm.hpp"

namespace tf {
namespace rocm {

// The lanes a column takes, a group each: 32, or 16 or 8 when it has few groups (a wave then takes more columns).
__host__ __device__ inline int kp_lanes(int groups) { return groups > 16 ? 32 : groups > 8 ? 16 : 8; }

// CB columns a lane set, R rows a pass, WAVES waves a block; the rows are passes over at most RB row blocks, which have
// consecutive block ids so the cache shares the weights between them.
template <typename T, int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
__global__ void __launch_bounds__(32 * WAVES) affine_gemm_kp(Affine a) {
    const int blocks = (a.m + R - 1) / R < RB ? (a.m + R - 1) / R : RB;  // of the launch: a routed item has its own rows
    if (!take_item(a, blockIdx.z)) return;
    constexpr int NO = R * CB;  // outputs of a lane set
    constexpr int PS = NO + 1;
    constexpr int OPL = (4 * NO + 31) / 32;  // outputs a lane folds, at most
    const int groups = a.k / a.group;
    const int gp = kp_lanes(groups);  // the lanes of a set; a wave has 32 / gp sets
    const int sets = 32 / gp;
    const int colw = ((blockIdx.x / blocks) * WAVES + (threadIdx.x >> 5)) * CB * sets;
    if (colw >= a.n) return;
    __shared__ float part[WAVES][32][PS];
    __shared__ float spart[WAVES][32][R + 1];
    __shared__ float scl[WAVES][32][CB + 1];
    __shared__ float bil[WAVES][32][CB + 1];

    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int per = a.group / 32;
    const int rounds = (groups + gp - 1) / gp;
    const int gi = lane & (gp - 1);
    const int col0 = colw + (lane / gp) * CB;  // this lane's columns
    const long long words_row = static_cast<long long>(a.k) * BITS / 32;
    const uint32_t* wsrc[CB];
    long long tb[CB];
#pragma unroll
    for (int c = 0; c < CB; ++c) {
        const int col = col0 + c < a.n ? col0 + c : a.n - 1;
        wsrc[c] = a.words + col * words_row;
        tb[c] = static_cast<long long>(col) * groups;
    }
    auto pass = [&](int row0) {
        const typename T::elem* xr[R];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            xr[r] = static_cast<const typename T::elem*>(a.x) + x_row(a, row0 + r < a.m ? row0 + r : 0) * a.k;
        }
        float acc[OPL];
#pragma unroll
        for (int t = 0; t < OPL; ++t) acc[t] = 0.f;

        // Stage s of group g's code words of every column, and the group's scale and bias (the groups past the last repeat it).
        uint32_t w[CB][BITS];
        uint32_t wn[CB][BITS];
        TableBits sr[CB];
        TableBits br[CB];
        auto fetch = [&](uint32_t (&dst)[CB][BITS], int g, int s) {
            const int gl = g < groups ? g : groups - 1;
#pragma unroll
            for (int c = 0; c < CB; ++c) load_aligned<BITS>(wsrc[c] + (static_cast<long long>(gl) * per + s) * BITS, dst[c]);
        };
        auto tables = [&](int g) {
            const int gl = g < groups ? g : groups - 1;
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                sr[c] = table_bits(a.scale, tb[c] + gl);
                br[c] = table_bits(a.bias, tb[c] + gl);
            }
        };
        fetch(w, gi, 0);
        tables(gi);
        for (int round = 0; round < rounds; ++round) {
            const int g = round * gp + gi;
            const int gl = g < groups ? g : groups - 1;
            float d[R][CB];
            float sx[R];
#pragma unroll
            for (int r = 0; r < R; ++r) {
                sx[r] = 0.f;
#pragma unroll
                for (int c = 0; c < CB; ++c) d[r][c] = 0.f;
            }
            for (int s = 0; s < per; ++s) {
                // the next stage's loads (the next round's first) go out before this stage's dots
                if (s + 1 < per) fetch(wn, g, s + 1);
                else if (round + 1 < rounds) fetch(wn, g + gp, 0);
                typename T::pair wd[CB][16];
#pragma unroll
                for (int c = 0; c < CB; ++c) {
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        wd[c][i] = code_pair<T>(piece_bits<BITS>(w[c], 2 * i), piece_bits<BITS>(w[c], 2 * i + 1));
                    }
                }
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const u32x4* src = reinterpret_cast<const u32x4*>(xr[r] + static_cast<long long>(gl) * a.group + s * 32);
                    uint32_t xn[16];
#pragma unroll
                    for (int v = 0; v < 4; ++v) {
                        const u32x4 u = src[v];
                        xn[4 * v] = u.x;
                        xn[4 * v + 1] = u.y;
                        xn[4 * v + 2] = u.z;
                        xn[4 * v + 3] = u.w;
                    }
                    float run = sx[r];
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const typename T::pair xp = __builtin_bit_cast(typename T::pair, xn[i]);
                        run += T::lo(xp);
                        run += T::hi(xp);
                    }
                    sx[r] = run;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const typename T::pair xp = __builtin_bit_cast(typename T::pair, xn[i]);
#pragma unroll
                        for (int c = 0; c < CB; ++c) d[r][c] = T::dot(xp, wd[c][i], d[r][c]);
                    }
                }
#pragma unroll
                for (int c = 0; c < CB; ++c) {
#pragma unroll
                    for (int i = 0; i < BITS; ++i) w[c][i] = wn[c][i];
                }
            }
            // this group's results to LDS, then the round's groups in order
#pragma unroll
            for (int r = 0; r < R; ++r) {
                spart[wave][lane][r] = sx[r];
#pragma unroll
                for (int c = 0; c < CB; ++c) part[wave][lane][r * CB + c] = d[r][c];
            }
#pragma unroll
            for (int c = 0; c < CB; ++c) {
                scl[wave][lane][c] = table_float(a.scale, sr[c]);
                bil[wave][lane][c] = table_float(a.bias, br[c]);
            }
            if (round + 1 < rounds) tables(g + gp);
            __syncwarp();
            const int count = groups - round * gp < gp ? groups - round * gp : gp;
#pragma unroll
            for (int t = 0; t < OPL; ++t) {
                const int os = lane + 32 * t;
                if (os >= NO * sets) continue;
                const int base = (os / NO) * gp;  // the first lane of the output's set
                const int o = os % NO;
                const int r = o / CB;
                const int c = o % CB;
                float v = acc[t];
                for (int gg = 0; gg < count; ++gg) {
                    v = fmaf(part[wave][base + gg][o], scl[wave][base + gg][c], v);
                    v = fmaf(spart[wave][base + gg][r], bil[wave][base + gg][c], v);
                }
                acc[t] = v;
            }
            __syncwarp();
        }
#pragma unroll
        for (int t = 0; t < OPL; ++t) {
            const int os = lane + 32 * t;
            if (os >= NO * sets) continue;
            const int o = os % NO;
            const int row = row0 + o / CB;
            const int col = colw + (os / NO) * CB + o % CB;
            if (row < a.m && col < a.n) a.out[out_row(a, row) * a.n + col] = acc[t];
        }
    };
    if constexpr (LOOP) {
        for (int row0 = (blockIdx.x % blocks) * R; row0 < a.m; row0 += blocks * R) pass(row0);
    } else {
        pass((blockIdx.x % blocks) * R);
    }
}

template <int BITS, int CB, int R, int WAVES, int RB, bool LOOP>
hipError_t launch_kp_tier(const Affine& a, hipStream_t stream, int items) {
    const int blocks = (a.m + R - 1) / R < RB ? (a.m + R - 1) / R : RB;
    const int cols = WAVES * CB * (32 / kp_lanes(a.k / a.group));
    const dim3 grid(((a.n + cols - 1) / cols) * blocks, 1, items);
#if TENSORFOLD_RDNA_WMMA
    if (!a.fp16) {
        affine_gemm_kp<DotBF16, BITS, CB, R, WAVES, RB, LOOP><<<grid, dim3(32 * WAVES), 0, stream>>>(a);
        return hipGetLastError();
    }
#endif
    affine_gemm_kp<DotF16, BITS, CB, R, WAVES, RB, LOOP><<<grid, dim3(32 * WAVES), 0, stream>>>(a);
    return hipGetLastError();
}

// The shapes of affine_launch.zig's kp_tiles, by rows.
template <int BITS>
hipError_t launch_kp_bits(const Affine& a, hipStream_t stream, int items) {
    if (a.m <= 2) return launch_kp_tier<BITS, 8, 1, 2, 2, false>(a, stream, items);
    if (a.m <= 4) return launch_kp_tier<BITS, 4, 4, 2, 4, true>(a, stream, items);
    if (a.m <= 8) return launch_kp_tier<BITS, 4, 2, 2, 4, true>(a, stream, items);
    if (a.m <= 32) return launch_kp_tier<BITS, 4, 8, 2, 4, false>(a, stream, items);
    return hipErrorInvalidValue;
}

}  // namespace rocm
}  // namespace tf
