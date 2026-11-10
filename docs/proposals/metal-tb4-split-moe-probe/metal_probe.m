// Offline Metal probes for docs/proposals/metal-tb4-split-moe.md. No eGPU is used or needed.
// build: DEVELOPER_DIR=/Library/Developer/CommandLineTools clang -O2 -fobjc-arc -framework Metal -framework Foundation metal_probe.m -o metal_probe
// run:   ./metal_probe wrap | latency | poll | vis | all    (vis spins ~1 s per cell and takes about a minute)
#import <Metal/Metal.h>
#include <fcntl.h>
#include <libkern/OSCacheControl.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint64_t now_ns(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW); }
static int cmp_u64(const void *a, const void *b) { uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b; return x < y ? -1 : x > y; }
static void stats(const char *what, uint64_t *v, int n) {
    qsort(v, n, sizeof *v, cmp_u64);
    printf("%-58s n=%-5d p10 %7.1f  p50 %7.1f  p90 %7.1f  p99 %7.1f  max %8.1f us\n", what, n, v[n / 10] / 1e3, v[n / 2] / 1e3,
           v[n * 9 / 10] / 1e3, v[n * 99 / 100] / 1e3, v[n - 1] / 1e3);
}
#define PAGE ((size_t)16384)

#define SPIN(NAME, TYPE, LOAD) "kernel void " NAME "(" TYPE " flag [[buffer(0)]], device atomic_uint* ack [[buffer(1)]], constant uint& want [[buffer(2)]], constant uint& bound [[buffer(3)]], device atomic_uint* iters [[buffer(4)]], uint t [[thread_position_in_grid]]) {\n" \
  "  if (t != 0) return; uint i = 0; while (i < bound && (" LOAD ") != want) ++i; atomic_store_explicit(iters, i, memory_order_relaxed); atomic_store_explicit(ack, i == bound ? 0xFFFFFFFFu : want, memory_order_relaxed); }\n"
static const char *kSrc =
"#include <metal_stdlib>\nusing namespace metal;\n"
"kernel void nop(device uint* p [[buffer(0)]], uint t [[thread_position_in_grid]]) { if (t == 0) p[0] += 1; }\n"
"kernel void flagw(device atomic_uint* p [[buffer(0)]], constant uint& v [[buffer(1)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t == 0) atomic_store_explicit(p, v, memory_order_relaxed); }\n"
"kernel void touch(device uint* p [[buffer(0)]], constant uint& n [[buffer(1)]], uint t [[thread_position_in_grid]]) { if (t < n) p[t] = p[t] * 3u + 1u; }\n"
// visibility variants (macro SPIN above), one resident thread: spin until LOAD(flag) == want or the bound runs out; iters = how long it looked
SPIN("spin_load", "device atomic_uint*", "atomic_load_explicit(flag, memory_order_relaxed)")
SPIN("spin_rmw", "device atomic_uint*", "atomic_fetch_add_explicit(flag, 0u, memory_order_relaxed)")
SPIN("spin_vol", "volatile device uint*", "*flag")
SPIN("spin_coh", "device coherent(device) atomic_uint*", "atomic_load_explicit(flag, memory_order_relaxed)")
// expert-block stand-in: tg reads `count` uint4 of one expert's weights (ids[tg>>7], chunk tg&127), xor-folds them; 13.5 MiB per layer over two dispatches
"kernel void stream8(device const uint4* base [[buffer(0)]], constant uint* ids [[buffer(1)]], device atomic_uint* part [[buffer(2)]], constant uint2& span [[buffer(3)]], uint tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {\n"
"  device const uint4* p = base + (ulong)ids[tg >> 7] * 110592ul + span.x + (tg & 127u) * span.y; uint acc = 0;\n"
"  for (uint i = t; i < span.y; i += 256u) { uint4 v = p[i]; acc += v.x ^ v.y ^ v.z ^ v.w; }\n"
"  acc = simd_sum(acc); if ((t & 31u) == 0) atomic_fetch_add_explicit(part + (tg & 255u), acc, memory_order_relaxed); }\n"
"kernel void stamp(device uint* out [[buffer(0)]], device atomic_uint* ack [[buffer(1)]], constant uint& seq [[buffer(2)]], uint t [[thread_index_in_threadgroup]]) {\n"
"  for (uint i = t; i < 8192u; i += 256u) out[i] = seq;\n"
"  threadgroup_barrier(mem_flags::mem_device); if (t == 0) atomic_store_explicit(ack, seq, memory_order_relaxed); }\n"
"kernel void stamp_only(device uint* out [[buffer(0)]], constant uint& seq [[buffer(1)]], uint t [[thread_index_in_threadgroup]]) { for (uint i = t; i < 8192u; i += 256u) out[i] = seq; }\n"
"kernel void poll_k(device const uint* flag [[buffer(0)]], device uint* rec [[buffer(1)]], device uint* cnt [[buffer(2)]], uint t [[thread_position_in_grid]]) {\n"
"  if (t != 0) return; uint k = cnt[0]; rec[k] = flag[0]; cnt[0] = k + 1; }\n"
"kernel void copy1(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint t [[thread_position_in_grid]]) { out[t] = in[t]; }\n"
"kernel void ack1(device atomic_uint* ack [[buffer(0)]], constant uint& seq [[buffer(1)]], uint t [[thread_position_in_grid]]) { if (t == 0) atomic_store_explicit(ack, seq, memory_order_relaxed); }\n";

typedef struct { void *p; size_t len; int fd; char name[64]; } Shm;
static Shm shm_make(const char *name, size_t len, int create) {
    Shm s = {0}; strlcpy(s.name, name, sizeof s.name); s.len = len;
    s.fd = shm_open(name, create ? (O_CREAT | O_RDWR | O_TRUNC) : O_RDWR, 0600);
    if (s.fd < 0) { perror("shm_open"); exit(1); }
    if (create && ftruncate(s.fd, len)) { perror("ftruncate"); exit(1); }
    s.p = mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_SHARED, s.fd, 0);
    if (s.p == MAP_FAILED) { perror("mmap"); exit(1); }
    return s;
}

static id<MTLBuffer> wrap(id<MTLDevice> dev, void *p, size_t len) {
    return [dev newBufferWithBytesNoCopy:p length:len options:MTLResourceStorageModeShared deallocator:nil];
}

// ---------------------------------------------------------------- wrap
static void run_wrap(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLComputePipelineState> touch) {
    printf("== wrap: newBufferWithBytesNoCopy over shm-backed memory (page size %zu)\n", (size_t)getpagesize());
    printf("device %s, recommendedMaxWorkingSetSize %.1f GiB, maxBufferLength %.1f GiB, hasUnifiedMemory %d\n", [[dev name] UTF8String],
           dev.recommendedMaxWorkingSetSize / 1073741824.0, dev.maxBufferLength / 1073741824.0, (int)dev.hasUnifiedMemory);
    size_t sizes[] = {PAGE, 1 << 20, 64 << 20, (size_t)1 << 30};
    for (int k = 0; k < 4; ++k) {
        size_t len = sizes[k];
        Shm s = shm_make("/tfsplit_wrap", len, 1);
        shm_unlink(s.name);
        uint64_t t0 = now_ns();
        id<MTLBuffer> b = wrap(dev, s.p, len);
        uint64_t t1 = now_ns();
        printf("shm MAP_SHARED %10zu B: wrap %s, contents==ptr %d, wrap time %.1f us", len, b ? "OK  " : "FAIL", b && b.contents == s.p, (t1 - t0) / 1e3);
        if (b) {
            uint32_t *w = s.p; uint32_t n = 4096;
            for (uint32_t i = 0; i < n; ++i) w[i] = i;
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:touch]; [e setBuffer:b offset:0 atIndex:0]; [e setBytes:&n length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding]; [cb commit]; [cb waitUntilCompleted];
            int bad = 0; for (uint32_t i = 0; i < n; ++i) bad += w[i] != i * 3u + 1u;
            printf(", GPU r/w roundtrip mismatches %d, cb status %ld\n", bad, (long)cb.status);
        } else printf("\n");
        b = nil; munmap(s.p, len); close(s.fd);
    }
    // unaligned cases
    Shm s = shm_make("/tfsplit_wrap", 4 * PAGE, 1); shm_unlink(s.name);
    id<MTLBuffer> b1 = wrap(dev, (char *)s.p + 4096, PAGE);
    id<MTLBuffer> b2 = wrap(dev, s.p, PAGE + 4096);
    id<MTLBuffer> b3 = wrap(dev, (char *)s.p + PAGE, 2 * PAGE);
    printf("ptr 4 KiB past a 16 KiB boundary: %s;  length 20 KiB (not a page multiple): %s;  ptr 16 KiB-aligned inside block: %s\n", b1 ? "OK" : "FAIL (nil)", b2 ? "OK" : "FAIL (nil)", b3 ? "OK" : "FAIL (nil)");
    if (b1) { ((volatile uint32_t *)((char *)s.p + 4096))[1] = 77; printf("  +4 KiB buffer: contents()==ptr %d, length %lu, CPU store visible through contents: %d\n", b1.contents == (char *)s.p + 4096, (unsigned long)b1.length, ((uint32_t *)b1.contents)[1] == 77); }
    if (b2) { ((volatile uint32_t *)s.p)[1] = 78; printf("  20 KiB buffer: contents()==ptr %d, length %lu, CPU store visible through contents: %d\n", b2.contents == s.p, (unsigned long)b2.length, ((uint32_t *)b2.contents)[1] == 78); }
    // MAP_FIXED over a PROT_NONE reservation, the way hipmac places a sysmem block at its GPU VA
    size_t arena = (size_t)64 << 20;
    void *res = mmap(NULL, arena, PROT_NONE, MAP_PRIVATE | MAP_ANON, -1, 0);
    int fd = shm_open("/tfsplit_fixed", O_CREAT | O_RDWR | O_TRUNC, 0600); shm_unlink("/tfsplit_fixed"); ftruncate(fd, 16 << 20);
    void *fx = mmap((char *)res + (8 << 20), 16 << 20, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_FIXED, fd, 0);
    id<MTLBuffer> b4 = fx == MAP_FAILED ? nil : wrap(dev, fx, 16 << 20);
    printf("MAP_FIXED shm block inside a PROT_NONE arena (hipmac layout): mmap %s, wrap %s\n", fx == MAP_FAILED ? "FAIL" : "OK", b4 ? "OK" : "FAIL");
    // a second mapping of the same shm object in this process: writes through one visible through the wrapped one
    Shm a = shm_make("/tfsplit_alias", PAGE * 4, 1); shm_unlink(a.name);
    void *alias = mmap(NULL, a.len, PROT_READ | PROT_WRITE, MAP_SHARED, a.fd, 0);
    id<MTLBuffer> ba = wrap(dev, a.p, a.len);
    ((volatile uint32_t *)alias)[5] = 0xC0FFEEu;
    printf("alias mapping store seen through wrapped buffer: %s\n", ((uint32_t *)ba.contents)[5] == 0xC0FFEEu ? "yes" : "NO");
    // hipmac sysmem blocks keep a DMA table at the head of the mapping: first page is not payload; wrapping the whole block (table included) is what a client holds
    printf("wrap of whole block incl. head table page: %s (the table page stays CPU-visible; payload offsets start at a page boundary)\n", ba ? "OK" : "FAIL");
    // MTLResidencySet or hazard: untracked
    id<MTLBuffer> bu = [dev newBufferWithBytesNoCopy:a.p length:a.len options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked deallocator:nil];
    printf("wrap with MTLResourceHazardTrackingModeUntracked: %s\n", bu ? "OK" : "FAIL");
}

static int wait_eq(volatile uint32_t *p, uint32_t v, uint64_t limit_ns) {
    uint64_t t0 = now_ns();
    for (uint32_t n = 0; *p != v; ++n)
        if ((n & 255) == 0 && now_ns() - t0 > limit_ns) return 0;
    return 1;
}

// ---------------------------------------------------------------- latency
static void run_latency(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib) {
    printf("== latency: Metal submit/complete/wait costs on this GPU\n");
    NSError *err; id<MTLComputePipelineState> nop = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nop"] error:&err];
    id<MTLComputePipelineState> flagw = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"flagw"] error:&err];
    Shm s = shm_make("/tfsplit_lat", 4 * PAGE, 1); shm_unlink(s.name);
    id<MTLBuffer> buf = wrap(dev, s.p, s.len);
    volatile uint32_t *w = s.p;
    enum { N = 2000 };
    static uint64_t t[N];
    // 1 empty-ish command buffer, commit + waitUntilCompleted
    for (int i = 0; i < 200 + N; ++i) {
        uint64_t t0 = now_ns();
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:nop]; [e setBuffer:buf offset:0 atIndex:0]; [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (i >= 200) t[i - 200] = now_ns() - t0;
    }
    stats("1 dispatch: encode+commit+waitUntilCompleted", t, N);
    // 2 completion handler + semaphore
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    for (int i = 0; i < 200 + N; ++i) {
        uint64_t t0 = now_ns();
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:nop]; [e setBuffer:buf offset:0 atIndex:0]; [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
        [cb addCompletedHandler:^(id<MTLCommandBuffer> c) { dispatch_semaphore_signal(sem); }]; [cb commit]; dispatch_semaphore_wait(sem, DISPATCH_TIME_FOREVER);
        if (i >= 200) t[i - 200] = now_ns() - t0;
    }
    stats("1 dispatch: commit + completion handler -> semaphore", t, N);
    // 3 pre-encoded command buffer, commit only (reuse cost of a per-token buffer re-encoded each time is shown by 1)
    // 4 commit -> kernel stores flag -> CPU sees it by polling (no completion wait)
    int tmo = 0, got = 0;
    for (int i = 0; i < 200 + N && tmo < 3; ++i) {
        uint32_t v = (uint32_t)(i + 1);
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:flagw]; [e setBuffer:buf offset:0 atIndex:0]; [e setBytes:&v length:4 atIndex:1];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
        uint64_t t0 = now_ns(); [cb commit]; int ok = wait_eq(w, v, 50000000ull);
        uint64_t t1 = now_ns(); [cb waitUntilCompleted];
        if (!ok) { ++tmo; printf("  TIMEOUT: store not visible to CPU within 50 ms of commit; visible after waitUntilCompleted: %d\n", w[0] == v); continue; }
        if (i >= 200) t[got++] = t1 - t0;
    }
    if (got) stats("commit -> kernel's store seen by CPU poll (commit..visible)", t, got);
    // 5 MTLSharedEvent: GPU waits on event, CPU signals, kernel stores flag, CPU polls it
    id<MTLSharedEvent> ev = [dev newSharedEvent]; uint64_t evv = 0;
    for (int i = 0; i < 100 + 500; ++i) {
        uint32_t v = (uint32_t)(1000000 + i); evv++;
        id<MTLCommandBuffer> cb = [q commandBuffer]; [cb encodeWaitForEvent:ev value:evv];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:flagw]; [e setBuffer:buf offset:0 atIndex:0]; [e setBytes:&v length:4 atIndex:1];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding]; [cb commit];
        usleep(300); // let the command buffer reach its wait
        uint64_t t0 = now_ns(); ev.signaledValue = evv; int ok = wait_eq(w, v, 50000000ull);
        uint64_t t1 = now_ns(); [cb waitUntilCompleted];
        if (!ok) { printf("  TIMEOUT: event path store not visible within 50 ms; visible after waitUntilCompleted: %d\n", w[0] == v); break; }
        if (i >= 100) t[i - 100] = t1 - t0;
    }
    stats("event: CPU signaledValue= -> waiting cb's kernel store visible", t, 500);
    // 6 dependent dispatches inside one encoder: what each extra kernel of a layer costs once the GPU is awake
    for (int n = 1; n <= 1000; n *= 10) {
        for (int rep = 0; rep < 12; ++rep) {
            id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            [e setComputePipelineState:nop]; [e setBuffer:buf offset:0 atIndex:0];
            for (int i = 0; i < n; ++i) [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            [e endEncoding]; uint64_t t0 = now_ns(); [cb commit]; [cb waitUntilCompleted]; t[rep] = now_ns() - t0;
        }
        char what[96]; snprintf(what, sizeof what, "%4d dependent 1-thread dispatches, one encoder, commit..completed", n); stats(what, t, 12);
    }
    // 7 per-layer event-chained command buffers with a 13.5 MiB expert stream (the Metal half of one MoE layer), triggered by a CPU relay
    id<MTLComputePipelineState> st8 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stream8"] error:&err];
    id<MTLComputePipelineState> ack1 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ack1"] error:&err];
    size_t region = (size_t)2 << 30; // 2 GiB of "expert" weights (anonymous memory, wrapped), 1213 experts of 1.6875 MiB
    void *wm = mmap(NULL, region, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0); memset(wm, 0x5a, region);
    id<MTLBuffer> wbuf = wrap(dev, wm, region);
    Shm c = shm_make("/tfsplit_chain", 4 * PAGE, 1); shm_unlink(c.name);
    id<MTLBuffer> cbuf = wrap(dev, c.p, c.len); volatile uint32_t *cw = c.p; // ack at 0, parts at 1024.. (256 words)
    enum { K = 40, ROUNDS = 20 };
    static uint64_t lt[K * ROUNDS]; int m = 0; uint32_t ids[8]; uint32_t rng = 12345;
    id<MTLBuffer> idbuf = [dev newBufferWithLength:K * 32 options:MTLResourceStorageModeShared];
    id<MTLSharedEvent> ev2 = [dev newSharedEvent]; uint64_t base = 0;
    for (int r = 0; r < ROUNDS + 3; ++r) {
        memset((void *)cw, 0, 64);
        uint32_t *idp = idbuf.contents;
        id<MTLCommandBuffer> cbs[K];
        for (uint32_t l = 1; l <= K; ++l) {
            for (int j = 0; j < 8; ++j) { rng = rng * 1664525u + 1013904223u; ids[j] = (rng >> 8) % 1200; idp[(l - 1) * 8 + j] = ids[j]; }
            id<MTLCommandBuffer> cb = [q commandBuffer]; [cb encodeWaitForEvent:ev2 value:base + l];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            uint32_t sp1[2] = {0, 576}, sp2[2] = {73728, 288};
            [e setComputePipelineState:st8]; [e setBuffer:wbuf offset:0 atIndex:0]; [e setBuffer:idbuf offset:(l - 1) * 32 atIndex:1]; [e setBuffer:cbuf offset:1024 atIndex:2]; [e setBytes:sp1 length:8 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e setBytes:sp2 length:8 atIndex:3];
            [e dispatchThreadgroups:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            [e setComputePipelineState:ack1]; [e setBuffer:cbuf offset:0 atIndex:0]; [e setBytes:&l length:4 atIndex:1];
            [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
            [cb commit]; cbs[l - 1] = cb;
        }
        usleep(2000);
        for (uint32_t l = 1; l <= K; ++l) {
            usleep(150); // the Vega's attention + router time between two MoE layers
            uint64_t t0 = now_ns(); ev2.signaledValue = base + l;
            if (!wait_eq(&cw[0], l, 100000000ull)) { printf("  TIMEOUT layer %u\n", l); break; }
            uint64_t t1 = now_ns(); if (r >= 3) lt[m++] = t1 - t0;
        }
        [cbs[K - 1] waitUntilCompleted]; base += K;
    }
    stats("MoE layer on Metal: event signal -> 13.5 MiB expert stream (2 dispatches) -> ack seen", lt, m);
    printf("  13.5 MiB / median = %.1f GB/s effective including wake and launch\n", 13.5 * 1048576.0 / (lt[m / 2] / 1e9) / 1e9);
    // 9 host data -> GPU: the CPU stores a new payload, then signals the event a pre-committed command buffer waits on; that buffer copies the payload back
    id<MTLComputePipelineState> copy1 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"copy1"] error:&err];
    for (int mode = 0; mode < 2; ++mode) { // 0: cb committed (and waiting on the event) before the store, 1: cb committed after the store
        uint32_t *in = (uint32_t *)((char *)c.p + PAGE), *out = (uint32_t *)((char *)c.p + 2 * PAGE);
        uint64_t stale = 0, trials_bad = 0; int N9 = 3000;
        for (uint32_t i = 1; i <= (uint32_t)N9; ++i) {
            id<MTLCommandBuffer> cb = [q commandBuffer];
            if (mode == 0) [cb encodeWaitForEvent:ev2 value:base + i];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; [e setComputePipelineState:copy1];
            [e setBuffer:cbuf offset:PAGE atIndex:0]; [e setBuffer:cbuf offset:2 * PAGE atIndex:1];
            [e dispatchThreads:MTLSizeMake(1024, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [e endEncoding];
            if (mode == 0) { [cb commit]; usleep(100); }
            for (int j = 0; j < 1024; ++j) in[j] = i; // fresh data from "the Vega"
            if (mode == 0) ev2.signaledValue = base + i; else [cb commit];
            [cb waitUntilCompleted];
            uint64_t bad = 0; for (int j = 0; j < 1024; ++j) bad += out[j] != i; stale += bad; trials_bad += bad != 0;
        }
        base += N9;
        printf("host store -> GPU read, %s: %llu stale words (%llu of %d trials)\n", mode == 0 ? "cb pre-committed, gated on an event signalled after the store" : "cb committed after the store", (unsigned long long)stale, (unsigned long long)trials_bad, N9);
    }
    // 8 data-then-flag order: 256 threads stamp 32 KiB; the flag is stored (a) by thread 0 of the same kernel after barrier(mem_device),
    //   (b) by a second dispatch in the same encoder; the CPU checks the data the moment it sees the flag
    id<MTLComputePipelineState> stamp = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stamp"] error:&err];
    id<MTLComputePipelineState> stamp_only = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"stamp_only"] error:&err];
    uint32_t *sp = (uint32_t *)((char *)c.p + PAGE);
    for (int mode = 0; mode < 2; ++mode) {
        uint64_t stale = 0, checked = 0, bad_trials = 0;
        for (uint32_t i = 1; i <= 3000; ++i) {
            memset(sp, 0, 8192 * 4);
            id<MTLCommandBuffer> cb = [q commandBuffer]; [cb encodeWaitForEvent:ev2 value:base + i];
            id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
            if (mode == 0) {
                [e setComputePipelineState:stamp]; [e setBuffer:cbuf offset:PAGE atIndex:0]; [e setBuffer:cbuf offset:512 atIndex:1]; [e setBytes:&i length:4 atIndex:2];
                [e dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            } else {
                [e setComputePipelineState:stamp_only]; [e setBuffer:cbuf offset:PAGE atIndex:0]; [e setBytes:&i length:4 atIndex:1];
                [e dispatchThreads:MTLSizeMake(256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
                [e setComputePipelineState:ack1]; [e setBuffer:cbuf offset:512 atIndex:0]; [e setBytes:&i length:4 atIndex:1];
                [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
            }
            [e endEncoding]; [cb commit];
            cw[128] = 0; usleep(100); ev2.signaledValue = base + i;
            if (!wait_eq(&cw[128], i, 100000000ull)) { printf("  TIMEOUT stamp %u\n", i); break; }
            uint64_t bad = 0; for (int j = 0; j < 8192; ++j) bad += sp[j] != i;
            stale += bad; bad_trials += bad != 0; checked += 8192; [cb waitUntilCompleted];
        }
        base += 3000;
        printf("flag after data, %s: %llu stale words of %llu seen by the CPU right after the flag (%llu of 3000 trials)\n", mode == 0 ? "same kernel, barrier(mem_device)" : "separate dispatch", (unsigned long long)stale, (unsigned long long)checked, (unsigned long long)bad_trials);
    }
}


// ---------------------------------------------------------------- poll: a CHAIN of tiny kernels, each reading the flag once (a new kernel = fresh cache state?)
static void run_poll(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib) {
    printf("== poll: chain of dependent 1-thread poll kernels in one command buffer; CPU stores the flag mid-flight\n");
    NSError *err; id<MTLComputePipelineState> pk = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"poll_k"] error:&err];
    enum { NK = 20000 };
    Shm s = shm_make("/tfsplit_poll", 8 * PAGE, 1); shm_unlink(s.name);
    id<MTLBuffer> b = wrap(dev, s.p, s.len); volatile uint32_t *w = s.p; // flag at 0, cnt at 64 (w[16]), rec at PAGE
    volatile uint32_t *rec = (volatile uint32_t *)((char *)s.p + PAGE);
    static uint64_t t[40]; int got = 0;
    for (int trial = 0; trial < 40; ++trial) {
        w[0] = 0; w[16] = 0; memset((void *)rec, 0, NK * 4);
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; [e setComputePipelineState:pk];
        [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:b offset:PAGE atIndex:1]; [e setBuffer:b offset:64 atIndex:2];
        for (int i = 0; i < NK; ++i) [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e endEncoding]; uint64_t t0 = now_ns(); [cb commit];
        while (now_ns() - t0 < 5000000ull) {} // the CPU cannot see the GPU's counter mid-flight (see below): store 5 ms in
        uint64_t ts = now_ns(); __atomic_store_n(&w[0], 1, __ATOMIC_SEQ_CST);
        [cb waitUntilCompleted]; uint64_t te = now_ns();
        double per = (te - t0 - 70000.0) / (double)NK; // per-kernel period of the chain (launch ~70 us taken off)
        uint32_t k0 = (uint32_t)((ts - t0 - 70000.0) / per); // kernel index running when the store happened
        uint32_t k1 = NK; for (uint32_t k = 0; k < NK; ++k) if (rec[k] == 1) { k1 = k; break; }
        if (trial < 5 || trial == 39) printf("  trial %2d: kernel index at store ~%u, first kernel that saw the flag %u (%s), chain period ~%.2f us\n", trial, k0, k1, k1 == NK ? "never" : "seen", per / 1e3);
        if (k1 < NK) t[got++] = (uint64_t)((k1 > k0 ? k1 - k0 : 0) * per);
        (void)ts;
    }
    // GPU store visible to the CPU while the same command buffer keeps running? marker stored by dispatch 1, dispatch 2 spins ~50 ms
    id<MTLComputePipelineState> ack1 = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"ack1"] error:&err];
    id<MTLComputePipelineState> sl = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"spin_load"] error:&err];
    for (int trial = 0; trial < 5; ++trial) {
        w[0] = 0; w[16] = 0; w[32] = 0; w[48] = 0; uint32_t one = 1, never = 0xDEAD, bound = 500000;
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ack1]; [e setBuffer:b offset:128 atIndex:0]; [e setBytes:&one length:4 atIndex:1]; [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
        [e setComputePipelineState:sl]; [e setBuffer:b offset:192 atIndex:0]; [e setBuffer:b offset:256 atIndex:1]; [e setBytes:&never length:4 atIndex:2]; [e setBytes:&bound length:4 atIndex:3]; [e setBuffer:b offset:320 atIndex:4];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
        uint64_t t0 = now_ns(); [cb commit]; int ok = wait_eq(&w[32], 1, 200000000ull); uint64_t tm = now_ns(); [cb waitUntilCompleted]; uint64_t te = now_ns();
        printf("  mid-cb visibility trial %d: marker %s at %.1f us after commit; command buffer completed at %.1f us\n", trial, ok ? "seen" : "NOT seen", (tm - t0) / 1e3, (te - t0) / 1e3);
    }
    if (got) stats("flag store -> first poll kernel that saw it (kernels x period)", t, got); else printf("  no poll kernel ever saw the flag\n");
}

// ---------------------------------------------------------------- vis: can a resident GPU thread see a CPU store, and how soon
static inline void dc_civac(void *p) { __asm__ volatile("dc civac, %0\n\tdsb sy" ::"r"(p) : "memory"); }

static void run_vis(id<MTLDevice> dev, id<MTLCommandQueue> q, id<MTLLibrary> lib) {
    printf("== vis: CPU store -> resident GPU thread (flag, ack, iters on separate 64-byte lines), by load flavour\n");
    const char *names[] = {"spin_load", "spin_rmw", "spin_vol", "spin_coh"};
    NSError *err;
    for (int k = 0; k < 4; ++k) {
        id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:[NSString stringWithUTF8String:names[k]]] error:&err];
        Shm s = shm_make("/tfsplit_vis", 2 * PAGE, 1); shm_unlink(s.name);
        id<MTLBuffer> b = wrap(dev, s.p, s.len); volatile uint32_t *w = s.p; // flag 0, ack 64 (w[16]), iters 128 (w[32])
        // calibrate: a spinner that never sees its flag
        uint32_t want = 7, bound = 1u << 22; w[0] = 0; w[16] = 0;
        id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        [e setComputePipelineState:ps]; [e setBuffer:b offset:0 atIndex:0]; [e setBuffer:b offset:64 atIndex:1]; [e setBytes:&want length:4 atIndex:2]; [e setBytes:&bound length:4 atIndex:3]; [e setBuffer:b offset:128 atIndex:4];
        [e dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e endEncoding];
        uint64_t t0 = now_ns(); [cb commit]; [cb waitUntilCompleted]; double ns_iter = (now_ns() - t0 - 70000.0) / bound;
        printf("%-9s %.2f ns/iteration (calibrated)\n", names[k], ns_iter);
        bound = (uint32_t)(400e6 / ns_iter); // about 400 ms of looking
        for (int flush = 0; flush < 2; ++flush) {
            static uint64_t t[8]; int got = 0, miss = 0;
            for (uint32_t i = 1; i <= 8; ++i) {
                w[0] = 0; w[16] = 0; w[32] = 0; dc_civac((void *)&w[0]); dc_civac((void *)&w[16]);
                id<MTLCommandBuffer> c2 = [q commandBuffer]; id<MTLComputeCommandEncoder> e2 = [c2 computeCommandEncoder];
                [e2 setComputePipelineState:ps]; [e2 setBuffer:b offset:0 atIndex:0]; [e2 setBuffer:b offset:64 atIndex:1]; [e2 setBytes:&i length:4 atIndex:2]; [e2 setBytes:&bound length:4 atIndex:3]; [e2 setBuffer:b offset:128 atIndex:4];
                [e2 dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)]; [e2 endEncoding]; [c2 commit];
                usleep(20000); // the kernel is surely running and spinning by now
                uint64_t ts = now_ns(); __atomic_store_n(&w[0], i, __ATOMIC_SEQ_CST); if (flush) dc_civac((void *)&w[0]);
                int ok = wait_eq(&w[16], i, 1000000000ull); uint64_t te = now_ns();
                [c2 waitUntilCompleted];
                if (ok) { t[got++] = te - ts; } else { ++miss; if (miss == 1) printf("    first miss: after the cb ended ack=%#x iters=%u of bound %u\n", w[16], w[32], bound); }
            }
            char what[96]; snprintf(what, sizeof what, "%-9s flush=%d CPU store -> GPU ack (missed %d/8 within 1 s)", names[k], flush, miss);
            if (got) stats(what, t, got); else printf("%s: never seen by the resident thread\n", what);
        }
    }
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IOLBF, 0); alarm(120); // a lost peer or a GPU that never shows a store must not hang the probe
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
        NSError *err; MTLCompileOptions *co = [MTLCompileOptions new]; co.languageVersion = MTLLanguageVersion4_0;
        id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:kSrc] options:co error:&err];
        if (!lib) { printf("compile: %s\n", [[err description] UTF8String]); return 1; }
        id<MTLComputePipelineState> touch = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"touch"] error:&err];
        const char *what = argc > 1 ? argv[1] : "all";
        if (!strcmp(what, "wrap") || !strcmp(what, "all")) run_wrap(dev, q, touch);
        if (!strcmp(what, "latency") || !strcmp(what, "all")) run_latency(dev, q, lib);
        if (!strcmp(what, "poll") || !strcmp(what, "all")) run_poll(dev, q, lib);
        if (!strcmp(what, "vis") || !strcmp(what, "all")) run_vis(dev, q, lib);
    }
    return 0;
}
