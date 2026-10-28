// The Flash Next family's EXL3 expert-parallel glue (cuda_exl3_tp.zig): a rank's masked picks.

// The other rank's slots (outside [LO, HI)) become -1, which the experts' kernels skip; slots keep their positions.
extern "C" __global__ void fn_exl3_tp_mask(const int* __restrict__ pick, int* __restrict__ mpick, int n, int lo, int hi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int p = pick[i];
    mpick[i] = (p >= lo && p < hi) ? p : -1;
}

// After the ranks swapped blocks: a slot takes this rank's value, the peer's (recv) or zero, plus 0.0f as the sum did.
extern "C" __global__ void __launch_bounds__(256) fn_exl3_tp_merge(float* __restrict__ y, const float* __restrict__ recv,
        const int* __restrict__ pick, int D, int lo, int hi, int table_n) {
    const int i = blockIdx.x;
    const int p = pick[i];
    float4* out = reinterpret_cast<float4*>(y + (size_t)i * D);
    const float4* peer = reinterpret_cast<const float4*>(recv + (size_t)i * D);
    for (int d = threadIdx.x; d < D / 4; d += blockDim.x) {
        float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (p >= 0 && p < table_n) v = (p >= lo && p < hi) ? out[d] : peer[d];
        v.x = __fadd_rn(v.x, 0.0f);
        v.y = __fadd_rn(v.y, 0.0f);
        v.z = __fadd_rn(v.z, 0.0f);
        v.w = __fadd_rn(v.w, 0.0f);
        out[d] = v;
    }
}
