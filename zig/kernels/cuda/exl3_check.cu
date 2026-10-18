// Host port of zig/src/core/exl3_format.zig dequantize + gemv, compared with the GPU kernels.
// Expert epilogue is identity: no SiLU and no clamp. Gate/up and down are separate launches of one matrix each.
// Tolerance: each fp16 value passes if |a-b| <= 2e-3 or the values are at most 2 fp16 ulps apart.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "exl3_decode.cuh"

using tf_exl3::kHadamard;
using tf_exl3::kInst3;
using tf_exl3::kMcg;
using tf_exl3::kMul1;

extern "C" __global__ void exl3_linear_f16(const __half* x, const int16_t* trellis, const __half* suh, const __half* svh, __half* y,
                                           int rows, int K, int N, int halves, int codebook);
extern "C" __global__ void exl3_experts_gate_up_f16(const __half* x, const int* ids, const int16_t* trellis, const __half* suh,
                                                    const __half* svh, __half* y, int tokens, int K, int N, int n_experts_in_batch,
                                                    int topk, int halves, int codebook);
extern "C" __global__ void exl3_experts_down_f16(const __half* x, const int* ids, const int16_t* trellis, const __half* suh,
                                                 const __half* svh, __half* y, int tokens, int K, int N, int n_experts_in_batch,
                                                 int topk, int halves, int codebook);
extern "C" __global__ void exl3_experts_combine_f16(const __half* y, const float* weights, __half* out, int tokens, int N, int topk);

__global__ void exl3_check_unpack(const int16_t* trellis, uint16_t* wq, int K, int N, int halves, int codebook) {
    const int tn = static_cast<int>(blockIdx.x);
    const int tk = static_cast<int>(blockIdx.y);
    const int lane = static_cast<int>(threadIdx.x);
    if (lane >= 32) return;
    const int words = tf_exl3::tile_words(halves);
    const int16_t* tw = trellis + static_cast<size_t>(tf_exl3::strips_tile(K >> 4, tk, tn)) * static_cast<size_t>(words);
    for (int j = 0; j < 8; ++j) {
        const int p = lane * 8 + j;
        int row = 0, col = 0;
        tf_exl3::placement(p, &row, &col);
        wq[static_cast<size_t>(tk * 16 + row) * static_cast<size_t>(N) + static_cast<size_t>(tn * 16 + col)] =
            tf_exl3::codebook_bits(codebook, tf_exl3::tile_state(tw, halves, p));
    }
}

static void fail(const char* msg) {
    std::fprintf(stderr, "FAIL %s\n", msg);
    std::exit(1);
}

static void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FAIL cuda %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

static void launch_ok(const char* what) {
    ck(cudaGetLastError(), what);
    ck(cudaDeviceSynchronize(), what);
}

static uint64_t mix(uint64_t h, uint64_t v) {
    h += v;
    h *= 1099511628211ull;
    return h;
}

static void fill_trellis(uint32_t seed, int16_t* out, size_t n) {
    uint32_t rng = seed;
    for (size_t i = 0; i < n; ++i) {
        rng = rng * 1664525u + 1013904223u;
        out[i] = static_cast<int16_t>(static_cast<uint16_t>(rng >> 16));
    }
}

static void fill_scale(uint16_t* out, int n, double base, double step) {
    for (int i = 0; i < n; ++i) out[i] = tf_exl3::f64_to_f16_bits(base + step * static_cast<double>(i % 11));
}

static void fill_x(uint16_t* out, int rows, int K) {
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < K; ++c) {
            const int t = static_cast<int>((r * 3 + c * 5) % 17) - 8;
            out[static_cast<size_t>(r) * static_cast<size_t>(K) + static_cast<size_t>(c)] =
                tf_exl3::f64_to_f16_bits(static_cast<double>(t) * 0.125);
        }
    }
}

static int ordered_f16(uint16_t h) {
    if (h & 0x8000u) return 0x8000 - static_cast<int>(h & 0x7fffu);
    return static_cast<int>(h) + 0x8000;
}

static void compare_f16(const char* tag, const uint16_t* got, const uint16_t* exp, size_t n, double* max_abs) {
    for (size_t i = 0; i < n; ++i) {
        const double a = tf_exl3::f16_to_f64(got[i]);
        const double b = tf_exl3::f16_to_f64(exp[i]);
        const double d = std::fabs(a - b);
        if (d > *max_abs) *max_abs = d;
        const int ulp = std::abs(ordered_f16(got[i]) - ordered_f16(exp[i]));
        if (!(d <= 2e-3 || ulp <= 2)) {
            std::fprintf(stderr, "FAIL %s i %zu got %04x ref %04x abs %.8g ulp %d\n", tag, i, got[i], exp[i], d, ulp);
            std::exit(1);
        }
    }
}

static void expect_hash(const char* name, int halves, int codebook, uint32_t seed, uint64_t want_y, uint64_t want_dq, uint64_t want_wq,
                        uint32_t want_st) {
    constexpr int K = 128, N = 128, tiles = 8, rows = 128;
    const int words = tf_exl3::tile_words(halves);
    std::vector<int16_t> trellis(static_cast<size_t>(tiles * tiles * words));
    std::vector<uint16_t> suh(K), svh(N), x(static_cast<size_t>(rows) * K), wq(static_cast<size_t>(K) * N), y(static_cast<size_t>(rows) * N);
    std::vector<double> W(static_cast<size_t>(K) * N);
    fill_trellis(seed, trellis.data(), trellis.size());
    fill_scale(suh.data(), K, 0.75, 0.02);
    fill_scale(svh.data(), N, 0.5, 0.03);
    fill_x(x.data(), rows, K);
    if (tf_exl3::tile_state(trellis.data(), halves, 0) != want_st) fail(name);
    tf_exl3::dequantize(halves, codebook, trellis.data(), tiles, tiles, suh.data(), svh.data(), wq.data(), W.data());
    uint64_t wqh = 0, dqh = 0;
    for (uint16_t b : wq) wqh = mix(wqh, b);
    for (double v : W) {
        uint64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        dqh = mix(dqh, bits);
    }
    tf_exl3::gemm_f16(W.data(), x.data(), rows, K, N, y.data());
    uint64_t yh = 0;
    for (uint16_t b : y) yh = mix(yh, b);
    if (wqh != want_wq || dqh != want_dq || yh != want_y) {
        std::fprintf(stderr, "FAIL host %s y %llu dq %llu wq %llu\n", name, static_cast<unsigned long long>(yh),
                     static_cast<unsigned long long>(dqh), static_cast<unsigned long long>(wqh));
        std::exit(1);
    }
}

static void run_linear(const char* name, int halves, int codebook, uint32_t seed, int K, int N, const int* row_list, int nlist,
                       double* max_abs) {
    const int tiles_k = K >> 4, tiles_n = N >> 4;
    const int words = tf_exl3::tile_words(halves);
    const size_t tiles = static_cast<size_t>(tiles_k) * static_cast<size_t>(tiles_n);
    const size_t cells = static_cast<size_t>(K) * static_cast<size_t>(N);
    int max_rows = 0;
    for (int i = 0; i < nlist; ++i)
        if (row_list[i] > max_rows) max_rows = row_list[i];
    std::vector<int16_t> stored(tiles * static_cast<size_t>(words)), strips(stored.size());
    std::vector<uint16_t> suh(static_cast<size_t>(K)), svh(static_cast<size_t>(N)), x(static_cast<size_t>(max_rows) * static_cast<size_t>(K));
    std::vector<uint16_t> wq(cells), y_ref(static_cast<size_t>(max_rows) * static_cast<size_t>(N)), y_got(y_ref.size()), wq_got(cells);
    std::vector<double> W(cells);
    fill_trellis(seed, stored.data(), stored.size());
    fill_scale(suh.data(), K, 0.75, 0.02);
    fill_scale(svh.data(), N, 0.5, 0.03);
    fill_x(x.data(), max_rows, K);
    tf_exl3::dequantize(halves, codebook, stored.data(), tiles_k, tiles_n, suh.data(), svh.data(), wq.data(), W.data());
    tf_exl3::reorder_strips(stored.data(), tiles_k, tiles_n, words, strips.data());
    tf_exl3::gemm_f16(W.data(), x.data(), max_rows, K, N, y_ref.data());

    int16_t* tr_d = nullptr;
    __half *x_d = nullptr, *suh_d = nullptr, *svh_d = nullptr, *y_d = nullptr;
    uint16_t* wq_d = nullptr;
    ck(cudaMalloc(&tr_d, strips.size() * sizeof(int16_t)), "malloc trellis");
    ck(cudaMalloc(&x_d, x.size() * sizeof(uint16_t)), "malloc x");
    ck(cudaMalloc(&suh_d, suh.size() * sizeof(uint16_t)), "malloc suh");
    ck(cudaMalloc(&svh_d, svh.size() * sizeof(uint16_t)), "malloc svh");
    ck(cudaMalloc(&y_d, y_ref.size() * sizeof(uint16_t)), "malloc y");
    ck(cudaMalloc(&wq_d, wq.size() * sizeof(uint16_t)), "malloc wq");
    ck(cudaMemcpy(tr_d, strips.data(), strips.size() * sizeof(int16_t), cudaMemcpyHostToDevice), "h2d trellis");
    ck(cudaMemcpy(x_d, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d x");
    ck(cudaMemcpy(suh_d, suh.data(), suh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d suh");
    ck(cudaMemcpy(svh_d, svh.data(), svh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d svh");

    exl3_check_unpack<<<dim3(tiles_n, tiles_k), 32>>>(tr_d, wq_d, K, N, halves, codebook);
    launch_ok("unpack");
    ck(cudaMemcpy(wq_got.data(), wq_d, wq.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h wq");
    for (size_t i = 0; i < wq.size(); ++i) {
        if (wq_got[i] != wq[i]) {
            std::fprintf(stderr, "FAIL %s unpack i %zu got %04x ref %04x\n", name, i, wq_got[i], wq[i]);
            std::exit(1);
        }
    }

    for (int ri = 0; ri < nlist; ++ri) {
        const int rows = row_list[ri];
        ck(cudaMemset(y_d, 0, y_ref.size() * sizeof(uint16_t)), "zero y");
        exl3_linear_f16<<<dim3(N / 128, rows), 512>>>(x_d, tr_d, suh_d, svh_d, y_d, rows, K, N, halves, codebook);
        launch_ok(name);
        ck(cudaMemcpy(y_got.data(), y_d, static_cast<size_t>(rows) * static_cast<size_t>(N) * sizeof(uint16_t), cudaMemcpyDeviceToHost),
           "d2h y");
        char tag[64];
        std::snprintf(tag, sizeof(tag), "%s rows %d", name, rows);
        compare_f16(tag, y_got.data(), y_ref.data(), static_cast<size_t>(rows) * static_cast<size_t>(N), max_abs);
    }
    cudaFree(tr_d);
    cudaFree(x_d);
    cudaFree(suh_d);
    cudaFree(svh_d);
    cudaFree(y_d);
    cudaFree(wq_d);
}

static void combine_ref(const uint16_t* y, const float* w, int topk, int N, uint16_t* out) {
    for (int n = 0; n < N; ++n) {
        float acc = 0.f;
        for (int s = 0; s < topk; ++s) {
            const float yv = static_cast<float>(tf_exl3::f16_to_f64(y[static_cast<size_t>(s) * static_cast<size_t>(N) + static_cast<size_t>(n)]));
            acc = std::fmaf(w[s], yv, acc);
        }
        out[n] = tf_exl3::f64_to_f16_bits(static_cast<double>(acc));
    }
}

static void run_experts(double* max_abs) {
    constexpr int E = 2, K = 128, N = 128, tokens = 1, topk = 2, halves = 5, codebook = kMul1;
    constexpr int tiles = 8;
    const int words = tf_exl3::tile_words(halves);
    const size_t one = static_cast<size_t>(tiles * tiles * words);
    const size_t cells = static_cast<size_t>(K) * N;
    std::vector<int16_t> strips(one * E);
    std::vector<uint16_t> suh(static_cast<size_t>(E) * K), svh(static_cast<size_t>(E) * N);
    std::vector<uint16_t> x_gate(K), x_down(K);
    std::vector<double> W(cells);
    std::vector<uint16_t> wq(cells);
    std::vector<uint16_t> gate_ref(static_cast<size_t>(E) * N), down_ref(static_cast<size_t>(E) * N);
    for (int e = 0; e < E; ++e) {
        std::vector<int16_t> stored(one);
        fill_trellis(0x111111u + static_cast<uint32_t>(e) * 0x1000u, stored.data(), stored.size());
        tf_exl3::reorder_strips(stored.data(), tiles, tiles, words, strips.data() + one * static_cast<size_t>(e));
        fill_scale(suh.data() + static_cast<size_t>(e) * K, K, 0.75 + 0.5 * e, 0.02);
        fill_scale(svh.data() + static_cast<size_t>(e) * N, N, 0.5 + 0.25 * e, 0.03);
    }
    fill_x(x_gate.data(), 1, K);
    for (int i = 0; i < K; ++i) x_down[static_cast<size_t>(i)] = tf_exl3::f64_to_f16_bits((static_cast<double>(i % 9) - 4.0) * 0.25);
    for (int e = 0; e < E; ++e) {
        std::vector<int16_t> stored(one);
        fill_trellis(0x111111u + static_cast<uint32_t>(e) * 0x1000u, stored.data(), stored.size());
        tf_exl3::dequantize(halves, codebook, stored.data(), tiles, tiles, suh.data() + static_cast<size_t>(e) * K,
                            svh.data() + static_cast<size_t>(e) * N, wq.data(), W.data());
        tf_exl3::gemm_f16(W.data(), x_gate.data(), 1, K, N, gate_ref.data() + static_cast<size_t>(e) * N);
        tf_exl3::gemm_f16(W.data(), x_down.data(), 1, K, N, down_ref.data() + static_cast<size_t>(e) * N);
    }
    if (std::memcmp(gate_ref.data(), gate_ref.data() + N, N * sizeof(uint16_t)) == 0) fail("experts identical");

    int16_t* tr_d = nullptr;
    __half *suh_d = nullptr, *svh_d = nullptr, *xg_d = nullptr, *xd_d = nullptr, *y_d = nullptr, *out_d = nullptr;
    int *ids_a = nullptr, *ids_b = nullptr;
    float* w_d = nullptr;
    ck(cudaMalloc(&tr_d, strips.size() * sizeof(int16_t)), "ex trellis");
    ck(cudaMalloc(&suh_d, suh.size() * sizeof(uint16_t)), "ex suh");
    ck(cudaMalloc(&svh_d, svh.size() * sizeof(uint16_t)), "ex svh");
    ck(cudaMalloc(&xg_d, K * sizeof(uint16_t)), "ex xg");
    ck(cudaMalloc(&xd_d, K * sizeof(uint16_t)), "ex xd");
    ck(cudaMalloc(&y_d, static_cast<size_t>(topk) * N * sizeof(uint16_t)), "ex y");
    ck(cudaMalloc(&out_d, N * sizeof(uint16_t)), "ex out");
    ck(cudaMalloc(&ids_a, topk * sizeof(int)), "ids a");
    ck(cudaMalloc(&ids_b, topk * sizeof(int)), "ids b");
    ck(cudaMalloc(&w_d, topk * sizeof(float)), "weights");
    const int ha[2] = {1, 0};
    const int hb[2] = {0, 1};
    const float wts[2] = {0.25f, 0.75f};
    ck(cudaMemcpy(tr_d, strips.data(), strips.size() * sizeof(int16_t), cudaMemcpyHostToDevice), "h2d ex");
    ck(cudaMemcpy(suh_d, suh.data(), suh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d suh");
    ck(cudaMemcpy(svh_d, svh.data(), svh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d svh");
    ck(cudaMemcpy(xg_d, x_gate.data(), K * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d xg");
    ck(cudaMemcpy(xd_d, x_down.data(), K * sizeof(uint16_t), cudaMemcpyHostToDevice), "h2d xd");
    ck(cudaMemcpy(ids_a, ha, sizeof(ha), cudaMemcpyHostToDevice), "h2d ids a");
    ck(cudaMemcpy(ids_b, hb, sizeof(hb), cudaMemcpyHostToDevice), "h2d ids b");
    ck(cudaMemcpy(w_d, wts, sizeof(wts), cudaMemcpyHostToDevice), "h2d w");

    auto launch_pair = [&](auto kernel, const __half* x, const uint16_t* ref, const char* tag_a, const char* tag_b) {
        std::vector<uint16_t> got(static_cast<size_t>(topk) * N), comb_got(N), comb_exp(N);
        const int order_a[2] = {1, 0};
        const int order_b[2] = {0, 1};
        ck(cudaMemset(y_d, 0, static_cast<size_t>(topk) * N * sizeof(uint16_t)), "zero slots");
        kernel<<<dim3(N / 128, tokens * topk), 512>>>(x, ids_a, tr_d, suh_d, svh_d, y_d, tokens, K, N, E, topk, halves, codebook);
        launch_ok(tag_a);
        ck(cudaMemcpy(got.data(), y_d, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h slots");
        for (int s = 0; s < topk; ++s)
            compare_f16(tag_a, got.data() + static_cast<size_t>(s) * N, ref + static_cast<size_t>(order_a[s]) * N, N, max_abs);
        exl3_experts_combine_f16<<<dim3(1, tokens), 128>>>(y_d, w_d, out_d, tokens, N, topk);
        launch_ok("combine");
        ck(cudaMemcpy(comb_got.data(), out_d, N * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h combine");
        std::vector<uint16_t> slot_ref(static_cast<size_t>(topk) * N);
        for (int s = 0; s < topk; ++s)
            std::memcpy(slot_ref.data() + static_cast<size_t>(s) * N, ref + static_cast<size_t>(order_a[s]) * N, N * sizeof(uint16_t));
        combine_ref(slot_ref.data(), wts, topk, N, comb_exp.data());
        char ctag[64];
        std::snprintf(ctag, sizeof(ctag), "%s combine", tag_a);
        compare_f16(ctag, comb_got.data(), comb_exp.data(), N, max_abs);

        ck(cudaMemset(y_d, 0, static_cast<size_t>(topk) * N * sizeof(uint16_t)), "zero slots b");
        kernel<<<dim3(N / 128, tokens * topk), 512>>>(x, ids_b, tr_d, suh_d, svh_d, y_d, tokens, K, N, E, topk, halves, codebook);
        launch_ok(tag_b);
        ck(cudaMemcpy(got.data(), y_d, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "d2h slots b");
        for (int s = 0; s < topk; ++s)
            compare_f16(tag_b, got.data() + static_cast<size_t>(s) * N, ref + static_cast<size_t>(order_b[s]) * N, N, max_abs);
    };
    launch_pair(exl3_experts_gate_up_f16, xg_d, gate_ref.data(), "gate ids 1,0", "gate ids 0,1");
    launch_pair(exl3_experts_down_f16, xd_d, down_ref.data(), "down ids 1,0", "down ids 0,1");

    cudaFree(tr_d);
    cudaFree(suh_d);
    cudaFree(svh_d);
    cudaFree(xg_d);
    cudaFree(xd_d);
    cudaFree(y_d);
    cudaFree(out_d);
    cudaFree(ids_a);
    cudaFree(ids_b);
    cudaFree(w_d);
}

int main() {
    int ndev = 0;
    ck(cudaGetDeviceCount(&ndev), "device count");
    if (ndev < 1) fail("no cuda device");

    const uint64_t cb_hash[3] = {5346088324104631537ull, 6430218787447001709ull, 6852455333796569056ull};
    for (int cb = 0; cb < 3; ++cb) {
        uint64_t h = 0;
        for (int s = 0; s < 65536; ++s) h = mix(h, tf_exl3::codebook_bits(cb, static_cast<uint32_t>(s)));
        if (h != cb_hash[cb]) fail("codebook hash");
    }
    if (tf_exl3::strips_tile(4, 0, 0) != 0 || tf_exl3::strips_tile(4, 1, 0) != 8 || tf_exl3::strips_tile(4, 0, 1) != 1 ||
        tf_exl3::strips_tile(4, 0, 8) != 32 || tf_exl3::strips_tile(4, 1, 11) != 43)
        fail("strips index");
    expect_hash("mul1", 5, kMul1, 0xC0FFEEu, 4164895221326122322ull, 17662402908242580269ull, 8074557589523908949ull, 22130u);
    expect_hash("inst3", 2, kInst3, 0xA11CEu, 1214903717108637301ull, 4103559239593590070ull, 14191163093994138433ull, 51430u);
    expect_hash("mcg", 4, kMcg, 0xBEEu, 16465806618434633307ull, 5395953528798628363ull, 16774814512002529146ull, 35385u);

    double max_abs = 0;
    const int rows_sq[] = {1, 2, 7, 16, 128};
    const int rows_wide[] = {1, 7, 16};
    run_linear("mul1", 5, kMul1, 0xC0FFEEu, 128, 128, rows_sq, 5, &max_abs);
    run_linear("inst3", 2, kInst3, 0xA11CEu, 128, 128, rows_sq, 5, &max_abs);
    run_linear("mcg", 4, kMcg, 0xBEEu, 128, 128, rows_sq, 5, &max_abs);
    run_linear("mul1-n256", 5, kMul1, 0x51C0FFEEu, 128, 256, rows_wide, 3, &max_abs);
    run_linear("inst3-n256", 2, kInst3, 0x51A11CEu, 128, 256, rows_wide, 3, &max_abs);
    run_linear("mcg-n256", 4, kMcg, 0x51BEEu, 128, 256, rows_wide, 3, &max_abs);
    run_experts(&max_abs);

    std::printf("epilogue identity\n");
    std::printf("PASS max_abs %.8g\n", max_abs);
    return 0;
}
