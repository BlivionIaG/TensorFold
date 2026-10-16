// EXL3 codebook, tile bitstream, strips index, and the Zig dequantize (exl3_format.zig).
// Stream bits are MSB-first in little-endian int16 pairs. One fp16 rounding per codebook value.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#ifdef __CUDACC__
#include <cuda_fp16.h>
#define EXL3_HD __host__ __device__ __forceinline__
#else
#define EXL3_HD inline
#endif

namespace tf_exl3 {

constexpr int kInst3 = 0;
constexpr int kMcg = 1;
constexpr int kMul1 = 2;

constexpr uint32_t kInst3Mul = 89226354u;
constexpr uint32_t kInst3Add = 64248484u;
constexpr uint32_t kMcgMul = 0xCBAC1FEDu;
constexpr uint32_t kMul1Mul = 0x83DCD12Du;
constexpr uint32_t kMask = 0x8FFF8FFFu;
constexpr uint32_t kFlip = 0x3B603B60u;
constexpr uint16_t kMul1ScaleBits = 0x1EEE;
constexpr uint16_t kMul1BiasBits = 0xC931;
constexpr int kHadamard = 128;
constexpr int kTilesPerStrip = 8;

// 1/sqrt(128) in binary64, and that value rounded to binary32.
constexpr uint64_t kHadInvBits = 0x3fb6a09e667f3bccull;
constexpr uint32_t kHadScaleF32Bits = 0x3db504f3u;

EXL3_HD int tile_words(int halves) { return halves * 8; }

// stripsTile in exl3_strips.zig: [N/128, K/16, 8, words].
EXL3_HD int strips_tile(int tiles_k, int tk, int tn) {
    const int strip = tn / kTilesPerStrip;
    const int local = tn % kTilesPerStrip;
    return strip * tiles_k * kTilesPerStrip + tk * kTilesPerStrip + local;
}

EXL3_HD int stored_tile(int tiles_n, int tk, int tn) { return tk * tiles_n + tn; }

EXL3_HD double bits_f64(uint64_t bits) {
#if defined(__CUDA_ARCH__)
    return __longlong_as_double(static_cast<long long>(bits));
#else
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
#endif
}

EXL3_HD float had_scale_f32() {
#if defined(__CUDA_ARCH__)
    return __uint_as_float(kHadScaleF32Bits);
#else
    float f;
    const uint32_t bits = kHadScaleF32Bits;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
#endif
}

EXL3_HD double had_inv() { return bits_f64(kHadInvBits); }

EXL3_HD double f16_to_f64(uint16_t h) {
    const unsigned sign = (h >> 15) & 1u;
    const unsigned exp = (h >> 10) & 0x1fu;
    const unsigned frac = h & 0x3ffu;
    double mag;
    if (exp == 0) {
        mag = static_cast<double>(frac) * 0x1p-24;
    } else if (exp == 31) {
        mag = frac ? bits_f64(0x7ff8000000000000ull) : bits_f64(0x7ff0000000000000ull);
    } else {
        const double sig = 1.0 + static_cast<double>(frac) * 0x1p-10;
        mag = sig * bits_f64(static_cast<uint64_t>(static_cast<int>(exp) - 15 + 1023) << 52);
    }
    return sign ? -mag : mag;
}

// One round-to-nearest-even of a binary64 value into fp16 bits. Zig's @floatCast(f16).
EXL3_HD uint16_t f64_to_f16_bits(double x) {
#if defined(__CUDA_ARCH__)
    const uint64_t u = static_cast<uint64_t>(__double_as_longlong(x));
#else
    uint64_t u = 0;
    std::memcpy(&u, &x, sizeof(u));
#endif
    const uint16_t sign = static_cast<uint16_t>((u >> 48) & 0x8000u);
    const int exp64 = static_cast<int>((u >> 52) & 0x7ffu);
    const uint64_t frac = u & 0x000fffffffffffffull;
    if (exp64 == 0x7ff) {
        if (frac == 0) return static_cast<uint16_t>(sign | 0x7c00u);
        return static_cast<uint16_t>(sign | 0x7e00u);
    }
    if (exp64 == 0) return sign;
    const int unbiased = exp64 - 1023;
    if (unbiased > 15) return static_cast<uint16_t>(sign | 0x7c00u);
    if (unbiased < -14) {
        if (unbiased < -25) return sign;
        const int rshift = 28 - unbiased;
        const uint64_t sig = frac | (1ull << 52);
        const uint64_t half = 1ull << (rshift - 1);
        const uint64_t rem = sig & ((1ull << rshift) - 1ull);
        uint64_t mant = sig >> rshift;
        if (rem > half || (rem == half && (mant & 1ull))) mant += 1ull;
        if (mant >= 0x400ull) return static_cast<uint16_t>(sign | (1u << 10));
        return static_cast<uint16_t>(sign | static_cast<uint16_t>(mant));
    }
    uint64_t mant = frac >> 42;
    const uint64_t rem = frac & ((1ull << 42) - 1ull);
    const uint64_t half = 1ull << 41;
    if (rem > half || (rem == half && (mant & 1ull))) {
        mant += 1ull;
        if (mant == 0x400ull) {
            const int bumped = unbiased + 1;
            if (bumped > 15) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | (static_cast<unsigned>(bumped + 15) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<unsigned>(unbiased + 15) << 10) | static_cast<uint16_t>(mant));
}

// E(p): exclusive end of value p's 16-bit window. Odd halves are mul1 half-bit widths.
EXL3_HD uint32_t stream_end(int halves, int p) {
    const uint32_t p1 = static_cast<uint32_t>(p + 1);
    if (halves & 1) return (p1 * static_cast<uint32_t>(halves) - (p1 & 1u)) / 2u;
    return p1 * static_cast<uint32_t>(halves / 2);
}

EXL3_HD uint32_t stream_bit(const int16_t* words, uint32_t index) {
    const uint32_t w32 = index >> 5;
    const uint32_t within = 31u - (index & 31u);
    const uint16_t lo = static_cast<uint16_t>(words[w32 * 2]);
    const uint16_t hi = static_cast<uint16_t>(words[w32 * 2 + 1]);
    const uint32_t word = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
    return (word >> within) & 1u;
}

EXL3_HD uint32_t tile_state(const int16_t* tile, int halves, int p) {
    const uint32_t ring = static_cast<uint32_t>(256 * halves / 2);
    const uint32_t end = stream_end(halves, p);
    const uint32_t start = (end + ring - 16u) % ring;
    uint32_t s = 0;
    for (int b = 0; b < 16; ++b) s = (s << 1) | stream_bit(tile, (start + static_cast<uint32_t>(b)) % ring);
    return s;
}

// m16n8k16 fragment map: lane p/8, element p%8.
EXL3_HD void placement(int p, int* row, int* col) {
    const int lane = p >> 3;
    const int j = p & 7;
    *row = 2 * (lane & 3) + (j & 1) + 8 * ((j >> 1) & 1);
    *col = (lane >> 2) + 8 * (j >> 2);
}

EXL3_HD uint16_t codebook_bits(int codebook, uint32_t state) {
    const uint32_t s = state & 0xffffu;
    if (codebook == kMul1) {
        const uint32_t x = s * kMul1Mul;
        const int h = 1024 + static_cast<int>(x & 255u) + static_cast<int>((x >> 8) & 255u) +
                      static_cast<int>((x >> 16) & 255u) + static_cast<int>((x >> 24) & 255u);
        const double scale = f16_to_f64(kMul1ScaleBits);
        const double bias = f16_to_f64(kMul1BiasBits);
        return f64_to_f16_bits(static_cast<double>(h) * scale + bias);
    }
    uint32_t x = (codebook == kMcg) ? s * kMcgMul : s * kInst3Mul + kInst3Add;
    x = (x & kMask) ^ kFlip;
    const double lo = f16_to_f64(static_cast<uint16_t>(x & 0xffffu));
    const double hi = f16_to_f64(static_cast<uint16_t>(x >> 16));
    return f64_to_f16_bits(lo + hi);
}

// Stored order [K/16, N/16, words] -> fp16 W_q [K, N], row-major.
inline void unpack_stored(int halves, int codebook, const int16_t* trellis, int tiles_k, int tiles_n, uint16_t* wq) {
    const int N = tiles_n * 16;
    const int words = tile_words(halves);
    for (int tk = 0; tk < tiles_k; ++tk) {
        for (int tn = 0; tn < tiles_n; ++tn) {
            const int16_t* tile = trellis + (static_cast<size_t>(stored_tile(tiles_n, tk, tn)) * static_cast<size_t>(words));
            for (int p = 0; p < 256; ++p) {
                int row = 0, col = 0;
                placement(p, &row, &col);
                wq[(static_cast<size_t>(tk * 16 + row) * static_cast<size_t>(N)) + static_cast<size_t>(tn * 16 + col)] =
                    codebook_bits(codebook, tile_state(tile, halves, p));
            }
        }
    }
}

inline void reorder_strips(const int16_t* stored, int tiles_k, int tiles_n, int words, int16_t* out) {
    for (int tk = 0; tk < tiles_k; ++tk) {
        for (int tn = 0; tn < tiles_n; ++tn) {
            const size_t from = static_cast<size_t>(stored_tile(tiles_n, tk, tn)) * static_cast<size_t>(words);
            const size_t to = static_cast<size_t>(strips_tile(tiles_k, tk, tn)) * static_cast<size_t>(words);
            std::memcpy(out + to, stored + from, static_cast<size_t>(words) * sizeof(int16_t));
        }
    }
}

inline int popcount(unsigned v) {
    int c = 0;
    while (v) {
        c += static_cast<int>(v & 1u);
        v >>= 1;
    }
    return c;
}

// In-place Hadamard along axis 0 (rows) or 1 (cols). Separate mul and add, matching exl3_format.rotate.
inline void rotate(double* x, int rows, int cols, int axis) {
    const double inv = had_inv();
    double scratch[kHadamard];
    const int blocks_r = rows / kHadamard;
    const int blocks_c = cols / kHadamard;
    if (axis == 0) {
        for (int c = 0; c < cols; ++c) {
            for (int b = 0; b < blocks_r; ++b) {
                const int base_r = b * kHadamard;
                for (int i = 0; i < kHadamard; ++i) {
                    double sum = 0;
                    for (int j = 0; j < kHadamard; ++j) {
                        const double sign = (popcount(static_cast<unsigned>(i & j)) & 1) ? -1.0 : 1.0;
                        const double h = sign * inv;
                        const double prod = h * x[(static_cast<size_t>(base_r + j) * static_cast<size_t>(cols)) + static_cast<size_t>(c)];
                        sum = sum + prod;
                    }
                    scratch[i] = sum;
                }
                for (int i = 0; i < kHadamard; ++i)
                    x[(static_cast<size_t>(base_r + i) * static_cast<size_t>(cols)) + static_cast<size_t>(c)] = scratch[i];
            }
        }
    } else {
        for (int r = 0; r < rows; ++r) {
            for (int b = 0; b < blocks_c; ++b) {
                const int base_c = b * kHadamard;
                for (int i = 0; i < kHadamard; ++i) {
                    double sum = 0;
                    for (int j = 0; j < kHadamard; ++j) {
                        const double sign = (popcount(static_cast<unsigned>(i & j)) & 1) ? -1.0 : 1.0;
                        const double h = sign * inv;
                        const double prod = h * x[(static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(base_c + j)];
                        sum = sum + prod;
                    }
                    scratch[i] = sum;
                }
                for (int i = 0; i < kHadamard; ++i)
                    x[(static_cast<size_t>(r) * static_cast<size_t>(cols)) + static_cast<size_t>(base_c + i)] = scratch[i];
            }
        }
    }
}

// W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh), in f64. wq receives the fp16 rotated-domain weights.
inline void dequantize(int halves, int codebook, const int16_t* trellis, int tiles_k, int tiles_n, const uint16_t* suh,
                       const uint16_t* svh, uint16_t* wq, double* W) {
    const int K = tiles_k * 16;
    const int N = tiles_n * 16;
    unpack_stored(halves, codebook, trellis, tiles_k, tiles_n, wq);
    const size_t cells = static_cast<size_t>(K) * static_cast<size_t>(N);
    for (size_t i = 0; i < cells; ++i) W[i] = f16_to_f64(wq[i]);
    rotate(W, K, N, 0);
    for (size_t i = 0; i < cells; ++i) {
        const double s = f16_to_f64(suh[i / static_cast<size_t>(N)]);
        W[i] = W[i] * s;
    }
    rotate(W, K, N, 1);
    for (size_t i = 0; i < cells; ++i) {
        const double s = f16_to_f64(svh[i % static_cast<size_t>(N)]);
        W[i] = W[i] * s;
    }
}

// y[r, j] = fp16(sum_i fp16_to_f64(x[r, i]) * W[i, j]), i ascending. Direct port of exl3_cpu.gemv per row.
inline void gemm_f16(const double* W, const uint16_t* x, int rows, int K, int N, uint16_t* y) {
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < N; ++j) {
            double sum = 0;
            for (int i = 0; i < K; ++i) {
                const double prod = f16_to_f64(x[static_cast<size_t>(r) * static_cast<size_t>(K) + static_cast<size_t>(i)]) *
                                    W[static_cast<size_t>(i) * static_cast<size_t>(N) + static_cast<size_t>(j)];
                sum = sum + prod;
            }
            y[static_cast<size_t>(r) * static_cast<size_t>(N) + static_cast<size_t>(j)] = f64_to_f16_bits(sum);
        }
    }
}

#ifdef __CUDACC__

// Unnormalized Walsh-Hadamard, 4 values a lane (index 4*lane+j). Matches (-1)^popcount(i&j).
__device__ __forceinline__ void fwht128(float (&v)[4], int lane) {
    const float a = v[0] + v[1];
    const float b = v[0] - v[1];
    const float c = v[2] + v[3];
    const float d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor_sync(0xffffffffu, v[j], m);
            v[j] = (lane & m) ? (o - v[j]) : (v[j] + o);
        }
    }
}

// One warp (lane = threadIdx.x, blockDim.x == 32). One 128-column block of one row.
// y = (((x * suh) @ H_K) @ W_q) @ H_N * svh. Trellis is strips order for this matrix alone.
// xhat, acc and tile are 128, 128 and 256 shared floats, owned by the caller.
__device__ __forceinline__ void project_nblock(const __half* __restrict__ x_row, const int16_t* __restrict__ trellis,
                                               const __half* __restrict__ suh, const __half* __restrict__ svh,
                                               __half* __restrict__ y_row, int K, int N, int halves, int codebook, int nblock,
                                               int lane, float* xhat, float* acc, float* tile) {
    const int tiles_k = K >> 4;
    const int words = tile_words(halves);
    const float scale = had_scale_f32();

    acc[lane] = 0.f;
    acc[lane + 32] = 0.f;
    acc[lane + 64] = 0.f;
    acc[lane + 96] = 0.f;
    __syncwarp();

    for (int kb = 0; kb < K; kb += kHadamard) {
        float v[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int k = kb + 4 * lane + j;
            v[j] = __half2float(x_row[k]) * __half2float(suh[k]);
        }
        fwht128(v, lane);
#pragma unroll
        for (int j = 0; j < 4; ++j) xhat[4 * lane + j] = v[j] * scale;
        __syncwarp();

        for (int tk_local = 0; tk_local < 8; ++tk_local) {
            const int tk = (kb >> 4) + tk_local;
            for (int local = 0; local < 8; ++local) {
                const int tn = nblock * 8 + local;
                const int16_t* tw = trellis + static_cast<size_t>(strips_tile(tiles_k, tk, tn)) * static_cast<size_t>(words);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int p = lane * 8 + j;
                    int row = 0, col = 0;
                    placement(p, &row, &col);
                    tile[row * 16 + col] = static_cast<float>(f16_to_f64(codebook_bits(codebook, tile_state(tw, halves, p))));
                }
                __syncwarp();
                if (lane < 16) {
                    float sum = acc[local * 16 + lane];
#pragma unroll
                    for (int r = 0; r < 16; ++r) sum = fmaf(xhat[tk_local * 16 + r], tile[r * 16 + lane], sum);
                    acc[local * 16 + lane] = sum;
                }
                __syncwarp();
            }
        }
    }

    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = acc[4 * lane + j];
    fwht128(o, lane);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = nblock * kHadamard + 4 * lane + j;
        y_row[col] = __float2half_rn(o[j] * scale * __half2float(svh[col]));
    }
}

#endif  // __CUDACC__

}  // namespace tf_exl3
