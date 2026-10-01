#include "act.hpp"

#include <stdexcept>
#include <string>

namespace tf {
namespace rocm {

__device__ float warp_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) x += __shfl_xor(x, mask, 32);
    return __shfl(x, 0, 32);
}

// Sequential squares along the row, then one scale. The warp tree is the reduction.
__global__ void rms_kernel(const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y,
                           int width, float eps) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const float* in = x + static_cast<long long>(row) * width;
    float sum = 0.f;
    for (int i = lane; i < width; i += 32) sum += in[i] * in[i];
    float inv = rsqrtf(warp_sum(sum) / static_cast<float>(width) + eps);
    float* out = y + static_cast<long long>(row) * width;
    for (int i = lane; i < width; i += 32) {
        float value = in[i] * inv;
        if (weight != nullptr) value *= weight[i];
        out[i] = value;
    }
}

// Length 1. Taps run in order, then the state shifts toward the new sample.
__global__ void conv_decode_kernel(const float* __restrict__ x, const float* __restrict__ weight,
                                   float* __restrict__ state, float* __restrict__ y, int channels, int kernel) {
    int channel = blockIdx.x * blockDim.x + threadIdx.x;
    int batch = blockIdx.y;
    if (channel >= channels) return;
    const float* row = weight + static_cast<long long>(channel) * kernel;
    float history[7];
    int kept = kernel - 1;
    float acc = 0.f;
    for (int tap = 0; tap < kept; ++tap) {
        history[tap] = state[(static_cast<long long>(batch) * kept + tap) * channels + channel];
        acc += history[tap] * row[tap];
    }
    float sample = x[static_cast<long long>(batch) * channels + channel];
    acc += sample * row[kept];
    float gate = 1.f / (1.f + expf(-acc));
    y[(static_cast<long long>(batch) * channels) + channel] = acc * gate;
    for (int tap = 0; tap < kept - 1; ++tap) {
        state[(static_cast<long long>(batch) * kept + tap) * channels + channel] = history[tap + 1];
    }
    if (kept > 0) state[(static_cast<long long>(batch) * kept + kept - 1) * channels + channel] = sample;
}

__global__ void rope_decode_kernel(const float* __restrict__ x, float* __restrict__ y, int width, int rotary,
                                   int pos, float theta) {
    int row = blockIdx.x;
    int lane = threadIdx.x;
    const float* in = x + static_cast<long long>(row) * width;
    float* out = y + static_cast<long long>(row) * width;
    int half = rotary >> 1;
    for (int i = lane; i < half; i += 32) {
        float freq = 1.f / powf(theta, static_cast<float>(i) / static_cast<float>(half));
        float ang = static_cast<float>(pos) * freq;
        float c = cosf(ang);
        float s = sinf(ang);
        float x1 = in[i];
        float x2 = in[half + i];
        out[i] = x1 * c - x2 * s;
        out[half + i] = x1 * s + x2 * c;
    }
    for (int i = rotary + lane; i < width; i += 32) out[i] = in[i];
}

}  // namespace rocm
}  // namespace tf

void rms_launch(const float* x, const float* weight, float* y, int rows, int width, float eps, hipStream_t stream) {
    tf::rocm::rms_kernel<<<rows, 32, 0, stream>>>(x, weight, y, width, eps);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("rms kernel: ") + hipGetErrorString(err));
}

void conv_decode_launch(const float* x, const float* weight, float* state, float* y, int batch, int channels,
                        int kernel, hipStream_t stream) {
    dim3 block(128);
    dim3 grid((channels + 127) / 128, batch);
    tf::rocm::conv_decode_kernel<<<grid, block, 0, stream>>>(x, weight, state, y, channels, kernel);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("conv kernel: ") + hipGetErrorString(err));
}

void rope_decode_launch(const float* x, float* y, int rows, int width, int rotary, int pos, float theta,
                        hipStream_t stream) {
    tf::rocm::rope_decode_kernel<<<rows, 32, 0, stream>>>(x, y, width, rotary, pos, theta);
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) throw std::runtime_error(std::string("rope kernel: ") + hipGetErrorString(err));
}
