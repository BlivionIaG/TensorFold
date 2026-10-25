// The Flash Next family's EXL3 expert-parallel glue (cuda_exl3_tp.zig): a rank's masked picks, and the summed MoE
// branch placed where the pass's write-back reads the per-slot outputs.

// A rank holds the expert ids [LO, HI) (global ids: rank 0 the routed 0..255, rank 1 the routed 256..511 and the
// shared expert 512). Another rank's slot becomes pick -1 (Experts.run skips it) with weight 0, so the rank's combined
// output is the fp32 fmaf chain over its own slots in slot order (a skipped slot adds weight 0 times a finite value).
extern "C" __global__ void fn_exl3_tp_mask(const int* __restrict__ pick, const float* __restrict__ wts,
                                           int* __restrict__ mpick, float* __restrict__ mwts, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pick[i];
    const bool own = p >= lo && p < hi;
    mpick[i] = own ? p : -1;
    mwts[i] = own ? wts[i] : 0.0f;
}

// The ranks' summed branch [rows, D] into the write-back's slot layout: y[r, 0, :] = sum, the other slots 0, and the
// weights {1, 0, ..., 0}. hc_writeback (MODE 2) then reads fma(sum, 1, 0) = sum and adds 0 * 0 for every other slot,
// so it writes the sum's bf16 exactly. Program (row, slot), 256 threads over D.
extern "C" __global__ void __launch_bounds__(256) fn_exl3_tp_place(const float* __restrict__ sum, float* __restrict__ y,
                                                                   float* __restrict__ wts, int d, int slots) {
    const int r = blockIdx.x, k = blockIdx.y;
    float* out = y + (static_cast<long long>(r) * slots + k) * d;
    const float* in = sum + static_cast<long long>(r) * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) out[i] = k == 0 ? in[i] : 0.0f;
    if (k == 0 && threadIdx.x < slots) wts[r * slots + threadIdx.x] = threadIdx.x == 0 ? 1.0f : 0.0f;
}

// Balanced decode ownership (TF_FLASHNEXT_EXL3_BALANCE): both ranks hold every routed expert, so a row's ten routed picks and the shared
// expert (id 512, rank 1 only) are dealt between the ranks by count instead of by id range. A pick starts on its home rank (id < SPLIT: 0,
// else 1); when one rank has two or more picks more than the other, half the difference moves, the lowest-weight picks first (ties by
// slot). Both ranks run this on the same picks and weights, so they agree; the rank keeps its own picks and masks the rest to -1, weight 0.
// One thread a row, S slots a row (the last is the shared expert).
extern "C" __global__ void fn_exl3_tp_balance(const int* __restrict__ pick, const float* __restrict__ wts,
                                              int* __restrict__ mpick, float* __restrict__ mwts, int rows, int slots, int split,
                                              int rank) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    int owner[16];
    int load[2] = {0, 0};
    for (int k = 0; k < slots; ++k) {
        const int p = pick[r * slots + k];
        owner[k] = p < 0 ? -1 : (p >= 512 ? 1 : (p < split ? 0 : 1));
        if (owner[k] >= 0) ++load[owner[k]];
    }
    const int from = load[0] > load[1] ? 0 : 1;
    const int to = 1 - from;
    int moves = (load[from] - load[to]) / 2;
    while (moves-- > 0) {
        int best = -1;
        for (int k = 0; k < slots; ++k) {
            if (owner[k] != from || pick[r * slots + k] >= 512) continue;
            if (best < 0 || wts[r * slots + k] < wts[r * slots + best]) best = k;
        }
        if (best < 0) break;
        owner[best] = to;
    }
    for (int k = 0; k < slots; ++k) {
        const int i = r * slots + k;
        const bool own = owner[k] == rank;
        mpick[i] = own ? pick[i] : -1;
        mwts[i] = own ? wts[i] : 0.0f;
    }
}

// Residency-aware balanced ownership. Each rank publishes which experts of the NEXT layer's pool it holds (1.0 or 0.0 a flag, then a marker) in
// its own segment of the all-reduce's tail (the other rank's segment is zero, so the sum is both); the next layer reads the peer's flags from
// that tail and its own from the pool's line_of table. Both ranks therefore decide on identical inputs. `tail` is [2][513] floats.
extern "C" __global__ void fn_exl3_tp_publish(float* __restrict__ tail, const int* __restrict__ next_line_of, float marker, int rank) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 512) return;
    const float own = i < 512 ? (next_line_of[i] >= 0 ? 1.0f : 0.0f) : marker;
    tail[rank * 513 + i] = own;
    tail[(1 - rank) * 513 + i] = 0.0f;
}

// Deal a row's picks by residency and load. A pick resident on one rank only runs there (a hit, cost 1); resident on both, on the lighter
// rank; resident on neither (a miss, cost 3 there), on its home rank unless that rank is more than two units heavier. The shared expert
// (id 512) stays on rank 1. Without a valid tail (marker mismatch) every pick stays on its home rank.
extern "C" __global__ void fn_exl3_tp_deal(const int* __restrict__ pick, const float* __restrict__ wts, int* __restrict__ mpick,
                                           float* __restrict__ mwts, const int* __restrict__ own_line_of,
                                           const float* __restrict__ tail, int rows, int slots, int split, int rank, float marker) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const bool valid = tail[512] == marker && tail[513 + 512] == marker;
    int owner[16];
    int res[16];
    int load[2] = {0, 0};
    for (int k = 0; k < slots; ++k) {
        const int p = pick[r * slots + k];
        owner[k] = -1;
        res[k] = 0;
        if (p < 0) continue;
        const int home = p >= 512 ? 1 : (p < split ? 0 : 1);
        if (p >= 512 || !valid) {
            owner[k] = home;
            ++load[home];
            continue;
        }
        const int mine = own_line_of[p] >= 0 ? 1 : 0;
        const int peer = tail[(1 - rank) * 513 + p] > 0.5f ? 1 : 0;
        res[k] = (rank == 0 ? mine : peer) | ((rank == 0 ? peer : mine) << 1);
    }
    if (valid) {
        for (int k = 0; k < slots; ++k) {  // resident on exactly one rank
            if (owner[k] >= 0 || pick[r * slots + k] < 0) continue;
            if (res[k] == 1) { owner[k] = 0; ++load[0]; }
            else if (res[k] == 2) { owner[k] = 1; ++load[1]; }
        }
        for (int k = 0; k < slots; ++k) {  // resident on both
            if (owner[k] >= 0 || pick[r * slots + k] < 0 || res[k] != 3) continue;
            const int p = pick[r * slots + k];
            const int home = p < split ? 0 : 1;
            const int to = load[home] > load[1 - home] + 1 ? 1 - home : home;  // keep the partition unless the home rank is clearly heavier
            owner[k] = to;
            ++load[to];
        }
        for (int k = 0; k < slots; ++k) {  // resident nowhere: a miss wherever it runs
            if (owner[k] >= 0 || pick[r * slots + k] < 0) continue;
            const int p = pick[r * slots + k];
            const int home = p < split ? 0 : 1;
            const int to = load[home] > load[1 - home] + 2 ? 1 - home : home;
            owner[k] = to;
            load[to] += 3;
        }
    }
    for (int k = 0; k < slots; ++k) {
        const int i = r * slots + k;
        const bool own = owner[k] == rank;
        mpick[i] = own ? pick[i] : -1;
        mwts[i] = own ? wts[i] : 0.0f;
    }
}
