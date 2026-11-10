// hipprof: a HIP runtime shim that times what an engine launches, on the card or the mock.
//
// TensorFold opens libamdhip64 by TENSORFOLD_HIP; point that at this library and name the real one in HIPPROF_REAL:
//   HIPPROF_REAL=<hipmac>/build/libamdhip64.dylib TENSORFOLD_HIP=<this>.dylib TF_POLICY=graphs=off tensorfold-native serve ...
// Every symbol of zig/src/hip/abi.zig (and the two it looks up besides) is forwarded unchanged (the arguments are integers and pointers,
// at most 11 of them, so one signature carries them all). hipModuleLaunchKernel is timed:
//   HIPPROF_TRACE=file lists the first 3000 launches (HIPPROF_TRACE_N=n: the first n) (name, grid, block, m n k) in order
//   each   (default) launch, then hipStreamSynchronize, wall clock of both: one kernel's time + the runtime's fixed submit/complete cost
//   rep    HIPPROF_REP=N (default 16): after the timed launch, the same launch N more times back to back, one sync: the pipelined time of
//          a kernel in a stream (what a graph replay pays). Only kernels whose name contains HIPPROF_REP_MATCH (default "affine_") and
//          that rewrite the same bytes each time (stream and row tiles, not the recurrences).
//   events HIPPROF_EVENTS=1: no sync; hipEventRecord on the stream before and after every launch, read back later (hipmac's GPU timestamps, 100 MHz):
//          the GPU time of each kernel in the pipeline (each record costs about 1.5 us of GPU time) and the gap to the previous kernel
//   graph  HIPPROF_GRAPH=1: hipGraphLaunch is followed by a stream sync and timed (a whole decode round).
// HIPPROF_START=1: nothing is timed (or replayed) until the first SIGUSR1, so a model load runs at full speed.
// SIGUSR1 clears the table (and starts the timing), SIGUSR2 writes it to HIPPROF_LOG (default hipprof.txt; also written at exit).
// Launches inside a stream capture are never timed (the sync would end the capture).
// For the affine tiles the table also gives m, n, k of the launch (the first kernel argument is quant/mlx.hpp's Affine; the second,
// of the stream tile, its StreamSides) and the bytes of weights and group tables that read.

#include <dlfcn.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef uintptr_t u;
typedef u (*fn)(u, u, u, u, u, u, u, u, u, u, u);

static void *lib;
static int rep_n = 16;
static const char *rep_match = "affine_";
static int graph_on;
static volatile int capturing;
static volatile int active = 1;  // HIPPROF_START: off until the first SIGUSR1
static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int selfpipe[2];

static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e6 + t.tv_nsec * 1e-3;
}

static struct {
    u h;
    char n[200];
} fm[2048];
static int nf;

typedef struct {
    char name[200];
    unsigned g[3], b[3], sh;
    int m, n, k, sides;  // affine launches
    long long count;
    double total, min, rep_total, gap_total;
    long long gap_count;
    long long rep_count;
} Row;
static Row rows[1024];
static int nrows;


// ---- event timing ----
#define EVN 2048
typedef struct {
    int fi;  // index into fm (the kernel's name)
    unsigned g[3], b[3], sh;
    int m, n, k, sides;
    u e0, e1;
    u stream;
} Pend;
static Pend pend[EVN];
static int npend;
static int events_on, ev_ready;
static fn ev_create, ev_record, ev_sync, ev_elapsed;

static fn get(const char *n) {
    if (!lib) {
        const char *real = getenv("HIPPROF_REAL");
        lib = dlopen(real ? real : "libamdhip64.dylib", RTLD_NOW | RTLD_LOCAL);
        if (!lib) {
            fprintf(stderr, "hipprof: cannot open %s: %s\n", real ? real : "libamdhip64.dylib", dlerror());
            exit(2);
        }
    }
    return (fn)dlsym(lib, n);
}

static const char *name_of(u h) {
    for (int i = 0; i < nf; i++)
        if (fm[i].h == h) return fm[i].n;
    return "?";
}

static Row *row_for(const char *name, const unsigned *g, const unsigned *b, unsigned sh, int m, int n, int k, int sides) {
    for (int i = 0; i < nrows; i++) {
        Row *r = &rows[i];
        if (!strcmp(r->name, name) && !memcmp(r->g, g, 12) && !memcmp(r->b, b, 12) && r->sh == sh && r->m == m && r->n == n && r->k == k && r->sides == sides)
            return r;
    }
    if (nrows == 1024) return &rows[1023];
    Row *r = &rows[nrows++];
    memset(r, 0, sizeof *r);
    strncpy(r->name, name, 199);
    memcpy(r->g, g, 12);
    memcpy(r->b, b, 12);
    r->sh = sh;
    r->m = m, r->n = n, r->k = k, r->sides = sides;
    r->min = 1e18;
    return r;
}


// the pending event pairs of the launches, read back (mu held): GPU time of each kernel and the gap to the one before it
static void drain(void) {
    u prev_e1 = 0, prev_stream = 0;
    for (int i = 0; i < npend; i++) {
        Pend *p = &pend[i];
        float ms = 0, gap = 0;
        ev_sync(p->e1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        ev_elapsed((u)&ms, p->e0, p->e1, 0, 0, 0, 0, 0, 0, 0, 0);
        Row *r = row_for(fm[p->fi].n, p->g, p->b, p->sh, p->m, p->n, p->k, p->sides);
        r->count++;
        r->total += ms * 1000.0;
        if (ms * 1000.0 < r->min) r->min = ms * 1000.0;
        if (prev_e1 && prev_stream == p->stream) {
            ev_elapsed((u)&gap, prev_e1, p->e0, 0, 0, 0, 0, 0, 0, 0, 0);
            if (gap >= 0 && gap < 5) {
                r->gap_total += gap * 1000.0;
                r->gap_count++;
            }
        }
        prev_e1 = p->e1;
        prev_stream = p->stream;
    }
    npend = 0;
}

static int cmp_rows(const void *a, const void *b) {
    double x = ((const Row *)a)->total + ((const Row *)a)->rep_total, y = ((const Row *)b)->total + ((const Row *)b)->rep_total;
    return x < y ? 1 : x > y ? -1 : 0;
}

static void dump(void) {
    const char *path = getenv("HIPPROF_LOG");
    FILE *f = fopen(path ? path : "hipprof.txt", "w");
    if (!f) return;
    pthread_mutex_lock(&mu);
    if (events_on && ev_ready) drain();
    Row *copy = malloc(sizeof rows[0] * (nrows ? nrows : 1));
    int n = nrows;
    memcpy(copy, rows, sizeof rows[0] * n);
    pthread_mutex_unlock(&mu);
    qsort(copy, n, sizeof *copy, cmp_rows);
    fprintf(f, "# name\tcount\tgrid\tblock\tshared\tm\tn\tk\tsides\tavg_us\tmin_us\ttotal_ms\trep_avg_us\tweight_bytes\tgap_us\n");
    for (int i = 0; i < n; i++) {
        Row *r = &copy[i];
        double bytes = r->k > 0 ? (double)r->n * r->k * 0.5625 : 0;  // 4-bit codes + a 2-byte scale and bias a group of 64
        fprintf(f, "%s\t%lld\t%u,%u,%u\t%u,%u,%u\t%u\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.3f\t%.2f\t%.0f\t%.2f\n", r->name, r->count, r->g[0], r->g[1], r->g[2], r->b[0], r->b[1], r->b[2], r->sh, r->m,
                r->n, r->k, r->sides, r->count ? r->total / r->count : 0, r->count ? r->min : 0, r->total / 1000, r->rep_count ? r->rep_total / r->rep_count : 0, bytes, r->gap_count ? r->gap_total / r->gap_count : 0);
    }
    free(copy);
    fclose(f);
}

static void on_signal(int sig) {
    char c = sig == SIGUSR1 ? 'c' : 'd';
    ssize_t w = write(selfpipe[1], &c, 1);
    (void)w;
}

static void *watcher(void *arg) {
    (void)arg;
    char c;
    while (read(selfpipe[0], &c, 1) == 1) {
        if (c == 'c') {
            pthread_mutex_lock(&mu);
            nrows = 0;
            npend = 0;
            active = 1;
            pthread_mutex_unlock(&mu);
        } else {
            dump();
        }
    }
    return 0;
}

__attribute__((constructor)) static void init(void) {
    if (getenv("HIPPROF_REP")) rep_n = atoi(getenv("HIPPROF_REP"));
    if (getenv("HIPPROF_REP_MATCH")) rep_match = getenv("HIPPROF_REP_MATCH");
    graph_on = getenv("HIPPROF_GRAPH") != NULL;
    events_on = getenv("HIPPROF_EVENTS") != NULL;
    if (getenv("HIPPROF_START")) active = 0;
    if (pipe(selfpipe)) return;
    signal(SIGUSR1, on_signal);
    signal(SIGUSR2, on_signal);
    pthread_t t;
    pthread_create(&t, 0, watcher, 0);
    pthread_detach(t);
}

__attribute__((destructor)) static void fini(void) { dump(); }

#define NEXT(name) fn p = get(#name)

__attribute__((visibility("default"))) u hipModuleGetFunction(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {
    NEXT(hipModuleGetFunction);
    u r = p(a, b, c, d, e, f, g, h, i, j, k);
    if (!r && nf < 2048) {
        fm[nf].h = *(u *)a;
        strncpy(fm[nf].n, (const char *)c, 199);
        nf++;
    }
    return r;
}

__attribute__((visibility("default"))) u hipStreamBeginCapture(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {
    NEXT(hipStreamBeginCapture);
    u r = p(a, b, c, d, e, f, g, h, i, j, k);
    if (!r) capturing++;
    return r;
}

__attribute__((visibility("default"))) u hipStreamEndCapture(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {
    NEXT(hipStreamEndCapture);
    u r = p(a, b, c, d, e, f, g, h, i, j, k);
    if (capturing > 0) capturing--;
    return r;
}

__attribute__((visibility("default"))) u hipGraphLaunch(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {
    NEXT(hipGraphLaunch);
    if (!graph_on) return p(a, b, c, d, e, f, g, h, i, j, k);
    fn sync = get("hipStreamSynchronize");
    double t0 = now();
    u r = p(a, b, c, d, e, f, g, h, i, j, k);
    sync(b, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    double dt = now() - t0;
    unsigned z[3] = {0, 0, 0};
    pthread_mutex_lock(&mu);
    Row *row = row_for("GRAPH_LAUNCH", z, z, 0, 0, 0, 0, 0);
    row->count++;
    row->total += dt;
    if (dt < row->min) row->min = dt;
    pthread_mutex_unlock(&mu);
    return r;
}

// hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra)
__attribute__((visibility("default"))) u hipModuleLaunchKernel(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {
    NEXT(hipModuleLaunchKernel);
    if (capturing || !active) return p(a, b, c, d, e, f, g, h, i, j, k);
    fn sync = get("hipStreamSynchronize");
    const char *name = name_of(a);
    int m = 0, n = 0, kk = 0, sides = 0;
    if (strstr(name, "affine_") && j) {
        char **params = (char **)j;
        const char *aff = params[0];
        memcpy(&m, aff + 56, 4);
        memcpy(&n, aff + 60, 4);
        memcpy(&kk, aff + 64, 4);
        if (strstr(name, "stream") && params[1]) {
            const char *sd = params[1];
            int cnt, sn[4];
            memcpy(&cnt, sd + 164, 4);
            memcpy(sn, sd + 128, 16);
            if (cnt > 0 && cnt <= 4) {
                n = 0;
                for (int x = 0; x < cnt; x++) n += sn[x];
                sides = cnt;
            }
        }
    }
    static FILE *trace;
    static int trace_n;
    if (!trace && getenv("HIPPROF_TRACE")) trace = fopen(getenv("HIPPROF_TRACE"), "w");
    if (trace && trace_n < (getenv("HIPPROF_TRACE_N") ? atoi(getenv("HIPPROF_TRACE_N")) : 3000)) {
        fprintf(trace, "%s g=%u,%u,%u b=%u,%u,%u m=%d n=%d k=%d\n", name, (unsigned)b, (unsigned)c, (unsigned)d, (unsigned)e, (unsigned)f,
                (unsigned)g, m, n, kk);
        fflush(trace);
        trace_n++;
    }
    if (events_on) {
        pthread_mutex_lock(&mu);
        if (!ev_ready) {
            ev_create = get("hipEventCreate");
            ev_record = get("hipEventRecord");
            ev_sync = get("hipEventSynchronize");
            ev_elapsed = get("hipEventElapsedTime");
            for (int x = 0; x < EVN; x++) {
                ev_create((u)&pend[x].e0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                ev_create((u)&pend[x].e1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
            }
            ev_ready = 1;
        }
        if (npend == EVN) drain();
        Pend *pp = &pend[npend++];
        int fi = 0;
        for (; fi < nf; fi++)
            if (fm[fi].h == a) break;
        pp->fi = fi < nf ? fi : 0;
        memcpy(pp->g, (unsigned[3]){(unsigned)b, (unsigned)c, (unsigned)d}, 12);
        memcpy(pp->b, (unsigned[3]){(unsigned)e, (unsigned)f, (unsigned)g}, 12);
        pp->sh = (unsigned)h, pp->m = m, pp->n = n, pp->k = kk, pp->sides = sides, pp->stream = i;
        ev_record(pp->e0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        u r = p(a, b, c, d, e, f, g, h, i, j, k);
        ev_record(pp->e1, i, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        pthread_mutex_unlock(&mu);
        return r;
    }
    if (getenv("HIPPROF_XCHK") && strstr(name, "affine_dot2_stream") && j && m * kk <= 16384) {
        static int seen;
        if (seen++ < 60) {
            static uint16_t xb[16384];
            u xp;
            memcpy(&xp, ((char **)j)[0], 8);
            get("hipMemcpyDtoH")((u)xb, xp, (u)(2 * m * kk), 0, 0, 0, 0, 0, 0, 0, 0);
            int bad = 0;
            float mx = 0;
            for (int x = 0; x < m * kk; x++) {
                int ex = (xb[x] >> 10) & 31;
                if (ex == 31) bad++;
                else {
                    float v = (xb[x] & 0x3ff) / 1024.f * (ex ? 1 << (ex > 15 ? ex - 15 : 0) : 0) + 0;
                    if (v > mx) mx = v;
                }
            }
            fprintf(stderr, "hipprof: #%d %s k=%d n=%d m=%d x non-finite %d of %d, rough max %g\n", seen, name, kk, n, m, bad, m * kk, mx);
        }
    }
    double t0 = now();
    u r = p(a, b, c, d, e, f, g, h, i, j, k);
    sync(i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    double dt = now() - t0;
    if (getenv("HIPPROF_SLOW_US") && dt > atof(getenv("HIPPROF_SLOW_US"))) fprintf(stderr, "hipprof: %.0f us  %s g=%u,%u,%u b=%u,%u,%u m=%d n=%d k=%d\n", dt, name, (unsigned)b, (unsigned)c, (unsigned)d, (unsigned)e, (unsigned)f, (unsigned)g, m, n, kk);
    double rep = 0;
    if (rep_n > 0 && strstr(name, rep_match)) {
        double t1 = now();
        for (int x = 0; x < rep_n; x++) p(a, b, c, d, e, f, g, h, i, j, k);
        sync(i, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
        rep = (now() - t1) / rep_n;
    }
    unsigned gr[3] = {(unsigned)b, (unsigned)c, (unsigned)d}, bl[3] = {(unsigned)e, (unsigned)f, (unsigned)g};
    pthread_mutex_lock(&mu);
    Row *row = row_for(name, gr, bl, (unsigned)h, m, n, kk, sides);
    row->count++;
    row->total += dt;
    if (dt < row->min) row->min = dt;
    if (rep > 0) {
        row->rep_count++;
        row->rep_total += rep;
    }
    pthread_mutex_unlock(&mu);
    return r;
}

// every other symbol the engine binds, forwarded as it is
#define FWD(name)                                                                                                                  \
    __attribute__((visibility("default"))) u name(u a, u b, u c, u d, u e, u f, u g, u h, u i, u j, u k) {                         \
        NEXT(name);                                                                                                                \
        return p(a, b, c, d, e, f, g, h, i, j, k);                                                                                 \
    }
#include "hipprof_fwd.inc"
