// One wave per (batch, value head, value row) walks the sequence in order.
// State for that row stays in registers. Dk is 16 or 128, so each lane owns
// elements lane, lane+32, ... and adds them in that index order. The warp
// sum is a xor-shuffle of 16, 8, 4, 2, 1, and every lane then uses lane 0's
// sum, which is the reduction the Python reference repeats.

#include "gated_delta.hpp"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

namespace tf {
namespace rocm {

__device__ float lane0_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

template <int DK>
__global__ void __launch_bounds__(32)
gated_delta_kernel(const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v,
                   const float* __restrict__ gate, const float* __restrict__ beta, float* __restrict__ state,
                   float* __restrict__ y, int length, int key_heads, int value_heads, int dv) {
    constexpr int SPAN = (DK + 31) / 32;
    const int lane = threadIdx.x;
    const int row = blockIdx.x;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (row >= dv) return;
    const int key_head = head / (value_heads / key_heads);

    float s[4] = {0.f, 0.f, 0.f, 0.f};
    const size_t state_row =
        ((static_cast<size_t>(batch) * value_heads + head) * dv + row) * DK;
#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) s[i] = state[state_row + idx];
    }

    for (int t = 0; t < length; ++t) {
        const size_t step = (static_cast<size_t>(batch) * length + t) * value_heads + head;
        const float decay = gate[step];
        const float step_beta = beta[step];
        const float value = v[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
                              static_cast<size_t>(head) * dv + row];
        const size_t key =
            ((static_cast<size_t>(batch) * length + t) * key_heads + key_head) * DK;

        float remembered = 0.f;
        float ks[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) {
                ks[i] = k[key + idx];
                s[i] = s[i] * decay;
                remembered = remembered + s[i] * ks[i];
            }
        }
        const float delta = (value - lane0_sum(remembered)) * step_beta;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) s[i] = s[i] + ks[i] * delta;
        }

        float readout = 0.f;
#pragma unroll
        for (int i = 0; i < SPAN; ++i) {
            const int idx = lane + i * 32;
            if (idx < DK) readout = readout + s[i] * q[key + idx];
        }
        // Every lane has to enter the shuffle. Lane 0 owns the stored sum.
        readout = lane0_sum(readout);
        if (lane == 0) {
            y[(static_cast<size_t>(batch) * length + t) * value_heads * dv +
              static_cast<size_t>(head) * dv + row] = readout;
        }
    }

#pragma unroll
    for (int i = 0; i < SPAN; ++i) {
        const int idx = lane + i * 32;
        if (idx < DK) state[state_row + idx] = s[i];
    }
}

}  // namespace rocm
}  // namespace tf

void gated_delta_launch(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                        float* state, float* y, int batch, int length, int key_heads, int value_heads, int dk,
                        int dv) {
    if ((dk != 16 && dk != 128) || dv < 1 || key_heads < 1 || value_heads % key_heads != 0 || batch < 1 ||
        length < 1) {
        throw std::invalid_argument("gated delta wants dk 16 or 128 and value heads a multiple of key heads");
    }
    dim3 block(32);
    dim3 grid(dv, value_heads, batch);
    hipError_t err;
    if (dk == 128) {
        tf::rocm::gated_delta_kernel<128><<<grid, block>>>(q, k, v, gate, beta, state, y, length, key_heads,
                                                           value_heads, dv);
    } else {
        tf::rocm::gated_delta_kernel<16><<<grid, block>>>(q, k, v, gate, beta, state, y, length, key_heads,
                                                          value_heads, dv);
    }
    err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("gated delta kernel: ") + hipGetErrorString(err));
}
