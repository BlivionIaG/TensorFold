#pragma once

// Prefill GEMM tile on the matrix cores of gfx11 (BF16 activations, m >= 16): 128 x 128 outputs a block, 8 waves of
// 32 rows x 64 columns. The codes are BF16 exactly, so a group's dot is v_wmma_f32_16x16x16_bf16 over the staged tiles
// (x row-major, codes column-major) from a zero accumulator; the sum of x, the scale and bias folds and the stages are
// affine_gemm_block's. The matrix core adds the 16 products of a step as a chain of dot2 pairs would: every product
// compared so far has the dot2 tile's bits, which nothing but the measurement promises.

#include "affine_gemm.hpp"

namespace tf {
namespace rocm {

typedef __bf16 bf16x16 __attribute__((ext_vector_type(16)));
typedef unsigned u32x8 __attribute__((ext_vector_type(8)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

// The 16 BF16 of a row's 16 k as one fragment: two 16-byte words of the staged tile.
__device__ inline bf16x16 frag16(const uint32_t* p) {
    const u32x4 lo = *reinterpret_cast<const u32x4*>(p);
    const u32x4 hi = *reinterpret_cast<const u32x4*>(p + 4);
    return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
}

template <int BITS>
__global__ void __launch_bounds__(256) affine_wmma_gemm(Affine a) {
#if TF_DEVICE_WMMA_GFX11
    if (!take_item(a, blockIdx.z)) return;
    const int bx = blockIdx.x;
    const int m0 = blockIdx.y * kGemmM;
    if (m0 >= a.m) return;
    constexpr int kLdw = kGemmLd / 2;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][kGemmM * kLdw];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][kGemmN * kLdw];
    __shared__ float sx[2][kGemmM];
    __shared__ float sc[3][kGemmN];
    __shared__ float bi[3][kGemmN];

    const int tid = threadIdx.x;
    const int n0 = bx * kGemmN;
    const int words_row = a.k * BITS / 32;
    const int groups = a.k / a.group;
    const int stages = a.k / kGemmK;
    const int per = a.group / kGemmK;
    const int rows_left = a.m - m0;
    const int line = tid & (kGemmN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;

    // Staging, as in gemm_tile: a thread's codes of a column (half of its 32 with the other side), half a row of x, two
    // stages a fetch, the row sums by the second side, scales and biases a stage ahead.
    const int col_s = n0 + line < a.n ? n0 + line : a.n - 1;
    const uint32_t* wsrc = a.words + static_cast<long long>(col_s) * words_row;
    const int xrow = tid >> 1;
    const int xhalf = tid & 1;
    const bool xlive = wave * 16 < rows_left;
    const bool sums = side == 1 && (line & ~31) < rows_left;
    const int row_s = m0 + xrow < a.m ? m0 + xrow : 0;
    const typename DotBF16::elem* xrow_ptr = static_cast<const typename DotBF16::elem*>(a.x) + x_row(a, row_s) * a.k;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(xrow_ptr);
    uint32_t wreg[2][BITS];
    u32x4 xreg[4];
    TableBits screg[2];
    TableBits bireg[2];
    float run = 0.f;
    auto fetch = [&](int st0) {
        const int st1 = st0 + 1 < stages ? st0 + 1 : stages - 1;
        const int st = st0 < stages ? st0 : stages - 1;
        load_aligned<BITS>(wsrc + st * BITS, wreg[0]);
        load_aligned<BITS>(wsrc + st1 * BITS, wreg[1]);
        const int sx_ = xhalf ? st1 : st;
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[sx_ * 4 + v];
        }
        const long long base = static_cast<long long>(col_s) * groups;
        screg[0] = table_bits(a.scale, base + st / per);
        bireg[0] = table_bits(a.bias, base + st / per);
        screg[1] = table_bits(a.scale, base + st1 / per);
        bireg[1] = table_bits(a.bias, base + st1 / per);
    };
    auto stash = [&]<int J>(std::integral_constant<int, J>, int st, int buf) {
        u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + line * kLdw + side * 8);
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int t = S * 16 + v * 8 + h * 2;
                    d[h] = __builtin_bit_cast(uint32_t, code_pair<DotBF16>(piece_bits<BITS>(wreg[J], t),
                                                                          piece_bits<BITS>(wreg[J], t + 1)));
                }
                dst[v] = u32x4{d[0], d[1], d[2], d[3]};
            }
        };
        if (side == 0) decode(std::integral_constant<int, 0>{}); else decode(std::integral_constant<int, 1>{});
        if (xhalf == J && xlive) {
            u32x4* xdst = reinterpret_cast<u32x4*>(xs[buf] + xrow * kLdw);
#pragma unroll
            for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
        }
        if (side == 0 && st % per == per - 1) {
            sc[(st / per) % 3][line] = table_float(a.scale, screg[J]);
            bi[(st / per) % 3][line] = table_float(a.bias, bireg[J]);
        }
    };
    auto sumx = [&](int st, int buf) {
        if (st % per == 0) run = 0.f;
        const u32x4* src = reinterpret_cast<const u32x4*>(xs[buf] + line * kLdw);
#pragma unroll
        for (int v = 0; v < 4; ++v) {
            const u32x4 u = src[v];
            const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int h = 0; h < 4; ++h) {
                const bf16x2 p = __builtin_bit_cast(bf16x2, q[h]);
                run += DotBF16::lo(p);
                run += DotBF16::hi(p);
            }
        }
        if (st % per == per - 1) sx[(st / per) & 1][line] = run;
    };

    // Wave (wm, wn) owns rows 32 wm.. and columns 64 wn..: 2 x 4 tiles. A lane holds, of a tile, column lane % 16 and
    // the rows 2 i + lane / 16 (i < 8); a fragment is the row (or column) lane % 16 over 16 k, the same in both halves.
    const int wm = wave & 3;
    const int wn = wave >> 2;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left;
    const bool live1 = wm * 32 + 16 < rows_left;
    f32x8 acc[2][4];
    f32x8 d[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            acc[mt][nt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
            d[mt][nt] = acc[mt][nt];
        }
    }
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    auto apply_bias = [&](int slot, int g) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            if (mt == 1 && !live1) continue;
            float s_x[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) s_x[i] = sx[g][wm * 32 + mt * 16 + 2 * i + rh];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const float bias = bi[slot][wn * 64 + nt * 16 + rl];
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][nt][i] = fmaf(s_x[i], bias, acc[mt][nt][i]);
            }
        }
    };
    fetch(0);
    stash(std::integral_constant<int, 0>{}, 0, 0);
    __syncthreads();
    auto step = [&]<int P>(std::integral_constant<int, P>, int s) {
        constexpr int cur = P;
        if (live0 && s > 0 && (s - 1) % per == per - 1) apply_bias(((s - 1) / per) % 3, ((s - 1) / per) & 1);
        if constexpr (P == 1) fetch(s + 1);
        if (live0) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                bf16x16 bf[4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) bf[nt] = frag16(ws[cur] + (wn * 64 + nt * 16 + rl) * kLdw + t * 8);
                const bool first = t == 0 && s % per == 0;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
                    const bf16x16 af = frag16(xs[cur] + (wm * 32 + mt * 16 + rl) * kLdw + t * 8);
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        d[mt][nt] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf[nt], first ? zero : d[mt][nt]);
                    }
                }
            }
            if (s % per == per - 1) {
                const int g = (s / per) % 3;
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
                        const float scale = sc[g][wn * 64 + nt * 16 + rl];
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mt][nt][i] = fmaf(d[mt][nt][i], scale, acc[mt][nt][i]);
                    }
                }
            }
        }
        if (sums) sumx(s, cur);
        stash(std::integral_constant<int, P ^ 1>{}, s + 1, cur ^ 1);
        __syncthreads();
    };
    for (int s = 0; s < stages; s += 2) {
        step(std::integral_constant<int, 0>{}, s);
        if (s + 1 < stages) step(std::integral_constant<int, 1>{}, s + 1);
    }
    if (live0 && (stages - 1) % per == per - 1) apply_bias(((stages - 1) / per) % 3, ((stages - 1) / per) & 1);
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int col = n0 + wn * 64 + nt * 16 + rl;
                if (col < a.n) a.out[out_row(a, row) * a.n + col] = acc[mt][nt][i];
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

template <int BITS>
hipError_t launch_wmma_gemm_bits(const Affine& a, hipStream_t stream, int items) {
    const dim3 grid((a.n + kGemmN - 1) / kGemmN, (a.m + kGemmM - 1) / kGemmM, items);
    affine_wmma_gemm<BITS><<<grid, dim3(256), 0, stream>>>(a);
    return hipGetLastError();
}

inline hipError_t launch_affine_wmma_gemm(const Affine& a, hipStream_t stream, int items = 1) {
    switch (a.bits) {
        case 2: return launch_wmma_gemm_bits<2>(a, stream, items);
        case 3: return launch_wmma_gemm_bits<3>(a, stream, items);
        case 4: return launch_wmma_gemm_bits<4>(a, stream, items);
        case 5: return launch_wmma_gemm_bits<5>(a, stream, items);
        case 6: return launch_wmma_gemm_bits<6>(a, stream, items);
        case 8: return launch_wmma_gemm_bits<8>(a, stream, items);
        default: return hipErrorInvalidValue;
    }
}

}  // namespace rocm
}  // namespace tf
