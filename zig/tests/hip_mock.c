/* Host-only HIP ABI fixture, never linked into the runtime. */
#ifndef OMIT_INIT
int hipInit(unsigned int flags) {
    return flags == 0 ? INIT_RESULT : 1;
}
#endif
int hipGetDeviceCount(int *count) {
    *count = 2;
    return 0;
}

#ifdef FULL_RUNTIME
/* The complete runtime table: device memory is host memory, modules hold no functions, launches do nothing. */
#include <stdlib.h>
#include <string.h>

enum { OUT_OF_MEMORY = 2, INVALID_DEVICE = 101, NOT_FOUND = 500 };
static int token;

int hipDeviceGet(int *device, int ordinal) {
    if (ordinal < 0 || ordinal > 1) return INVALID_DEVICE;
    *device = ordinal;
    return 0;
}
int hipDevicePrimaryCtxRetain(void **ctx, int device) {
    *ctx = &token;
    return device < 0 || device > 1 ? INVALID_DEVICE : 0;
}
int hipDevicePrimaryCtxRelease(int device) { return device < 0 || device > 1 ? INVALID_DEVICE : 0; }
int hipCtxSetCurrent(void *ctx) { return ctx ? 0 : INVALID_DEVICE; }
int hipDeviceSynchronize(void) { return 0; }

int hipMalloc(void **ptr, size_t size) {
    *ptr = size >> 40 ? NULL : malloc(size);
    return *ptr ? 0 : OUT_OF_MEMORY;
}
int hipFree(void *ptr) {
    free(ptr);
    return 0;
}
int hipHostMalloc(void **ptr, size_t size, unsigned int flags) {
    return flags ? 1 : hipMalloc(ptr, size);
}
int hipHostFree(void *ptr) { return hipFree(ptr); }

int hipMemcpyHtoD(void *dst, const void *src, size_t size) {
    memcpy(dst, src, size);
    return 0;
}
int hipMemcpyDtoH(void *dst, const void *src, size_t size) { return hipMemcpyHtoD(dst, src, size); }
int hipMemcpyHtoDAsync(void *dst, const void *src, size_t size, void *stream) {
    return stream ? hipMemcpyHtoD(dst, src, size) : 1;
}
int hipMemcpyDtoHAsync(void *dst, const void *src, size_t size, void *stream) {
    return hipMemcpyHtoDAsync(dst, src, size, stream);
}
int hipMemset(void *dst, int value, size_t size) {
    memset(dst, value, size);
    return 0;
}
int hipMemsetD8Async(void *dst, unsigned char value, size_t size, void *stream) {
    return stream ? hipMemset(dst, value, size) : 1;
}

int hipStreamCreateWithFlags(void **stream, unsigned int flags) {
    *stream = &token;
    return flags == 1 ? 0 : 1;
}
int hipStreamDestroy(void *stream) { return stream ? 0 : 1; }
#ifndef OMIT_STREAM_SYNCHRONIZE
int hipStreamSynchronize(void *stream) { return stream == NULL || stream == &token ? 0 : 1; }
#endif

int hipModuleLoadData(void **module, const void *image) {
    *module = &token;
    return image ? 0 : 1;
}
int hipModuleUnload(void *module) { return module ? 0 : 1; }
int hipModuleGetFunction(void **function, void *module, const char *name) {
    (void)module;
    (void)name;
    *function = NULL;
    return NOT_FOUND;
}
int hipModuleLaunchKernel(void *f, unsigned gx, unsigned gy, unsigned gz, unsigned bx, unsigned by, unsigned bz,
                          unsigned shared, void *stream, void **params, void **extra) {
    (void)gx, (void)gy, (void)gz, (void)bx, (void)by, (void)bz, (void)shared, (void)stream, (void)params, (void)extra;
    return f ? 0 : 1;
}

const char *hipGetErrorName(int result) {
    switch (result) {
    case OUT_OF_MEMORY: return "hipErrorOutOfMemory";
    case NOT_FOUND: return "hipErrorNotFound";
    default: return "hipErrorUnknown";
    }
}
const char *hipGetErrorString(int result) { return result ? "mock failure" : "no error"; }
#endif
