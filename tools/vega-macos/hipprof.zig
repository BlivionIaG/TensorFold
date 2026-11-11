//! hipprof: a HIP runtime shim that times what an engine launches; its modes and variables: tools/vega-macos/README.md.
//! TENSORFOLD_HIP names this library and HIPPROF_REAL the runtime it forwards to; the whole HIP table is exported.
const std = @import("std");
const builtin = @import("builtin");
const table = @import("hipprof_table.zig");
const fwd = @import("hipprof_fwd.zig");

const U = usize;
const Fn = *const fn (U, U, U, U, U, U, U, U, U, U, U) callconv(.c) U;
const darwin = builtin.os.tag.isDarwin();
const rtld_now: c_int = 2;
const rtld_local: c_int = if (darwin) 4 else 0;
const rtld_nodelete: c_int = if (darwin) 0x80 else 0x1000;
const func_cap = 2048;
const event_cap = 2048;
const default_trace_n = 3000;

const DlInfo = extern struct { fname: ?[*:0]const u8, fbase: ?*anyopaque, sname: ?[*:0]const u8, saddr: ?*anyopaque };
extern "c" fn dlopen(path: ?[*:0]const u8, mode: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;
extern "c" fn dladdr(addr: *const anyopaque, info: *DlInfo) c_int;
extern "c" fn signal(sig: c_int, handler: *const fn (c_int) callconv(.c) void) ?*anyopaque;
extern "c" fn atexit(func: *const fn () callconv(.c) void) c_int;
extern "c" fn atoi(s: [*:0]const u8) c_int;
extern "c" fn atof(s: [*:0]const u8) f64;
extern "c" fn pipe(fds: *[2]c_int) c_int;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*std.c.FILE;
extern "c" fn fprintf(stream: *std.c.FILE, fmt: [*:0]const u8, ...) c_int;
extern "c" fn fflush(stream: *std.c.FILE) c_int;
extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

var lib: ?*anyopaque = null;
var lib_mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var mu: std.c.pthread_mutex_t = std.c.PTHREAD_MUTEX_INITIALIZER;
var rep_n: c_int = 16;
var rep_match: [*:0]const u8 = "affine_";
var graph_on = false;
var events_on = false;
var ev_ready = false;
var capturing = std.atomic.Value(i32).init(0);
var active = std.atomic.Value(i32).init(1); // HIPPROF_START: off until the first SIGUSR1
var selfpipe: [2]c_int = .{ -1, -1 };
var tab: table.Table = std.mem.zeroes(table.Table);

const Func = struct { h: U, n: [table.name_cap]u8 };
var fm: [func_cap]Func = std.mem.zeroes([func_cap]Func);
var nf: usize = 0;

/// A launch whose events are recorded and not yet read back.
const Pend = struct { fi: usize, shape: table.Shape, e0: U, e1: U, stream: U };
var pend: [event_cap]Pend = std.mem.zeroes([event_cap]Pend);
var npend: usize = 0;

fn lock(m: *std.c.pthread_mutex_t) void {
    _ = std.c.pthread_mutex_lock(m);
}

fn unlock(m: *std.c.pthread_mutex_t) void {
    _ = std.c.pthread_mutex_unlock(m);
}

fn now() f64 {
    var t: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &t);
    return @as(f64, @floatFromInt(t.sec)) * 1e6 + @as(f64, @floatFromInt(t.nsec)) * 1e-3;
}

fn stderrLine(buf: []const u8, len: c_int) void {
    if (len < 0) return;
    _ = std.c.write(2, buf.ptr, @min(@as(usize, @intCast(len)), buf.len - 1));
}

fn cannotOpen(real_path: [*:0]const u8) noreturn {
    var buf: [512]u8 = undefined;
    stderrLine(&buf, snprintf(&buf, buf.len, "hipprof: cannot open %s: %s\n", real_path, dlerror() orelse "?"));
    std.c.exit(2);
}

fn libHandle() *anyopaque {
    lock(&lib_mu);
    defer unlock(&lib_mu);
    if (lib) |l| return l;
    const path: [*:0]const u8 = if (std.c.getenv("HIPPROF_REAL")) |p| p else "libamdhip64.dylib";
    lib = dlopen(path, rtld_now | rtld_local) orelse cannotOpen(path);
    return lib.?;
}

/// The runtime's own `name`, looked up once.
fn real(comptime name: [:0]const u8) Fn {
    const S = struct {
        const symbol = name; // captured, so every name gets its own cache
        var cached: std.atomic.Value(U) = .init(0);
    };
    var p = S.cached.load(.acquire);
    if (p == 0) {
        const sym = dlsym(libHandle(), S.symbol) orelse {
            var buf: [256]u8 = undefined;
            stderrLine(&buf, snprintf(&buf, buf.len, "hipprof: the runtime has no %s\n", S.symbol.ptr));
            std.c.exit(2);
        };
        p = @intFromPtr(sym);
        S.cached.store(p, .release);
    }
    return @ptrFromInt(p);
}

fn call1(p: Fn, a: U) U {
    return p(a, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}

fn nameOf(h: U) [*:0]const u8 {
    for (fm[0..nf]) |*f| {
        if (f.h == h) return @ptrCast(&f.n);
    }
    return "?";
}

/// The pending event pairs read back (`mu` held): the GPU time of each kernel and the gap to the one before it.
fn drain() void {
    const ev_sync = real("hipEventSynchronize");
    const ev_elapsed = real("hipEventElapsedTime");
    var prev_e1: U = 0;
    var prev_stream: U = 0;
    for (pend[0..npend]) |*p| {
        var ms: f32 = 0;
        var gap: f32 = 0;
        _ = call1(ev_sync, p.e1);
        _ = ev_elapsed(@intFromPtr(&ms), p.e0, p.e1, 0, 0, 0, 0, 0, 0, 0, 0);
        const row = tab.rowFor(std.mem.sliceTo(&fm[p.fi].n, 0), p.shape);
        const us = @as(f64, ms) * 1000.0;
        table.Table.addTimed(row, us);
        if (prev_e1 != 0 and prev_stream == p.stream) {
            _ = ev_elapsed(@intFromPtr(&gap), prev_e1, p.e0, 0, 0, 0, 0, 0, 0, 0, 0);
            if (gap >= 0 and gap < 5) {
                row.gap_total += @as(f64, gap) * 1000.0;
                row.gap_count += 1;
            }
        }
        prev_e1 = p.e1;
        prev_stream = p.stream;
    }
    npend = 0;
}

fn dump() void {
    const path: [*:0]const u8 = if (std.c.getenv("HIPPROF_LOG")) |p| p else "hipprof.txt";
    const f = fopen(path, "w") orelse return;
    defer _ = std.c.fclose(f);
    lock(&mu);
    if (events_on and ev_ready) drain();
    const copy = std.heap.c_allocator.alloc(table.Row, tab.n) catch {
        unlock(&mu);
        return;
    };
    defer std.heap.c_allocator.free(copy);
    @memcpy(copy, tab.rows[0..tab.n]);
    unlock(&mu);
    table.sortRows(copy);
    _ = std.c.fwrite(table.header.ptr, 1, table.header.len, f);
    var line: [640]u8 = undefined;
    for (copy) |*r| {
        const text = table.formatRow(&line, r) orelse continue;
        _ = std.c.fwrite(text.ptr, 1, text.len, f);
    }
}

fn onSignal(sig: c_int) callconv(.c) void {
    const c: u8 = if (sig == @backingInt(std.c.SIG.USR1)) 'c' else 'd';
    _ = std.c.write(selfpipe[1], @ptrCast(&c), 1);
}

fn watcher(_: ?*anyopaque) callconv(.c) ?*anyopaque {
    var c: u8 = undefined;
    while (std.c.read(selfpipe[0], @ptrCast(&c), 1) == 1) {
        if (c == 'c') {
            lock(&mu);
            tab.n = 0;
            npend = 0;
            active.store(1, .monotonic);
            unlock(&mu);
        } else {
            dump();
        }
    }
    return null;
}

fn fini() callconv(.c) void {
    dump();
}

fn init() callconv(.c) void {
    // The engine opens the runtime to read the device's caps and closes it again before the real open: an unloaded shim
    // leaves its watcher thread and its signal handlers in unmapped text. Pin the image.
    var di: DlInfo = undefined;
    if (dladdr(@ptrCast(&init), &di) != 0) {
        if (di.fname) |name| _ = dlopen(name, rtld_now | rtld_nodelete);
    }
    _ = atexit(&fini);
    if (std.c.getenv("HIPPROF_REP")) |s| rep_n = atoi(s);
    if (std.c.getenv("HIPPROF_REP_MATCH")) |s| rep_match = s;
    graph_on = std.c.getenv("HIPPROF_GRAPH") != null;
    events_on = std.c.getenv("HIPPROF_EVENTS") != null;
    if (std.c.getenv("HIPPROF_START") != null) active.store(0, .monotonic);
    if (pipe(&selfpipe) != 0) return;
    _ = signal(@backingInt(std.c.SIG.USR1), onSignal);
    _ = signal(@backingInt(std.c.SIG.USR2), onSignal);
    var t: std.c.pthread_t = undefined;
    if (std.c.pthread_create(&t, null, watcher, null) == .SUCCESS) _ = std.c.pthread_detach(t);
}

const init_entry: *const fn () callconv(.c) void = &init;

comptime {
    @export(&init_entry, .{ .name = "hipprof_init_entry", .visibility = .hidden, .section = if (darwin) "__DATA,__mod_init_func" else ".init_array" });
    for (fwd.names) |name| @export(&Forward(name).call, .{ .name = name });
}

fn Forward(comptime name: [:0]const u8) type {
    return struct {
        fn call(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
            return real(name)(a, b, c, d, e, f, g, h, i, j, k);
        }
    };
}

/// A HIP result is an int in the low half of the return register.
fn ok(r: U) bool {
    return @as(u32, @truncate(r)) == 0;
}

export fn hipModuleGetFunction(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
    const r = real("hipModuleGetFunction")(a, b, c, d, e, f, g, h, i, j, k);
    if (ok(r) and nf < func_cap) {
        fm[nf].h = @as(*const U, @ptrFromInt(a)).*;
        table.copyName(&fm[nf].n, std.mem.sliceTo(@as([*:0]const u8, @ptrFromInt(c)), 0));
        nf += 1;
    }
    return r;
}

export fn hipStreamBeginCapture(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
    const r = real("hipStreamBeginCapture")(a, b, c, d, e, f, g, h, i, j, k);
    if (ok(r)) _ = capturing.fetchAdd(1, .monotonic);
    return r;
}

export fn hipStreamEndCapture(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
    const r = real("hipStreamEndCapture")(a, b, c, d, e, f, g, h, i, j, k);
    if (capturing.load(.monotonic) > 0) _ = capturing.fetchSub(1, .monotonic);
    return r;
}

/// HIPPROF_GRAPH: a graph launch followed by a stream sync and timed, a whole decode round.
export fn hipGraphLaunch(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
    const p = real("hipGraphLaunch");
    if (!graph_on) return p(a, b, c, d, e, f, g, h, i, j, k);
    const t0 = now();
    const r = p(a, b, c, d, e, f, g, h, i, j, k);
    _ = call1(real("hipStreamSynchronize"), b);
    const dt = now() - t0;
    const z: [3]u32 = .{ 0, 0, 0 };
    lock(&mu);
    const row = tab.rowFor("GRAPH_LAUNCH", .{ .g = z, .b = z, .sh = 0, .m = 0, .n = 0, .k = 0, .sides = 0 });
    table.Table.addTimed(row, dt);
    unlock(&mu);
    return r;
}

var trace: ?*std.c.FILE = null;
var trace_n: c_int = 0;
var xchk_seen: c_int = 0;
var xchk_buf: [16384]u16 = undefined;

fn traceLaunch(name: [*:0]const u8, sh: table.Shape) void {
    if (trace == null) {
        if (std.c.getenv("HIPPROF_TRACE")) |path| trace = fopen(path, "w");
    }
    const f = trace orelse return;
    const limit: c_int = if (std.c.getenv("HIPPROF_TRACE_N")) |s| atoi(s) else default_trace_n;
    if (trace_n >= limit) return;
    _ = fprintf(f, "%s g=%u,%u,%u b=%u,%u,%u m=%d n=%d k=%d\n", name, sh.g[0], sh.g[1], sh.g[2], sh.b[0], sh.b[1], sh.b[2], sh.m, sh.n, sh.k);
    _ = fflush(f);
    trace_n += 1;
}

/// HIPPROF_EVENTS: record events around the launch on its stream and read them back later; no sync.
fn launchEvents(p: Fn, fi: usize, sh: table.Shape, a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) U {
    const ev_create = real("hipEventCreate");
    const ev_record = real("hipEventRecord");
    lock(&mu);
    if (!ev_ready) {
        for (&pend) |*x| {
            _ = call1(ev_create, @intFromPtr(&x.e0));
            _ = call1(ev_create, @intFromPtr(&x.e1));
        }
        ev_ready = true;
    }
    if (npend == event_cap) drain();
    const pp = &pend[npend];
    npend += 1;
    pp.fi = fi;
    pp.shape = sh;
    pp.stream = i;
    _ = ev_record(pp.e0, i, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const r = p(a, b, c, d, e, f, g, h, i, j, k);
    _ = ev_record(pp.e1, i, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    unlock(&mu);
    return r;
}

/// HIPPROF_XCHK: the first 60 stream-tile launches' fp16 activations scanned for non-finite values.
fn xcheck(name: [*:0]const u8, m: i32, n: i32, kk: i32, params: [*]const ?[*]const u8) void {
    const seen = xchk_seen;
    xchk_seen += 1;
    if (seen >= 60) return;
    const first = params[0] orelse return;
    const xp = std.mem.readInt(u64, first[0..8], .little);
    const count: usize = @intCast(m *% kk);
    _ = real("hipMemcpyDtoH")(@intFromPtr(&xchk_buf), xp, 2 * count, 0, 0, 0, 0, 0, 0, 0, 0);
    const res = table.xcheck(xchk_buf[0..count]);
    var buf: [512]u8 = undefined;
    stderrLine(&buf, snprintf(&buf, buf.len, "hipprof: #%d %s k=%d n=%d m=%d x non-finite %d of %d, rough max %g\n", seen + 1, name, kk, n, m, res.non_finite, @as(c_int, @intCast(count)), @as(f64, res.rough_max)));
}

/// hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra)
export fn hipModuleLaunchKernel(a: U, b: U, c: U, d: U, e: U, f: U, g: U, h: U, i: U, j: U, k: U) callconv(.c) U {
    const p = real("hipModuleLaunchKernel");
    if (capturing.load(.monotonic) != 0 or active.load(.monotonic) == 0) return p(a, b, c, d, e, f, g, h, i, j, k);
    const sync = real("hipStreamSynchronize");
    const name = nameOf(a);
    const name_s = std.mem.sliceTo(name, 0);
    const params: ?[*]const ?[*]const u8 = @ptrFromInt(j);
    const dims = table.kernelDims(name_s, params);
    const sh: table.Shape = .{
        .g = .{ @truncate(b), @truncate(c), @truncate(d) },
        .b = .{ @truncate(e), @truncate(f), @truncate(g) },
        .sh = @truncate(h),
        .m = dims.m,
        .n = dims.n,
        .k = dims.k,
        .sides = dims.sides,
    };
    traceLaunch(name, sh);
    if (events_on) {
        var fi: usize = 0;
        while (fi < nf and fm[fi].h != a) fi += 1;
        return launchEvents(p, if (fi < nf) fi else 0, sh, a, b, c, d, e, f, g, h, i, j, k);
    }
    const kk = dims.k;
    if (std.c.getenv("HIPPROF_XCHK") != null and std.mem.indexOf(u8, name_s, "affine_dot2_stream") != null and j != 0 and
        dims.m *% kk <= 16384 and dims.m *% kk >= 0) xcheck(name, dims.m, dims.n, kk, params.?);
    const t0 = now();
    const r = p(a, b, c, d, e, f, g, h, i, j, k);
    _ = call1(sync, i);
    const dt = now() - t0;
    if (std.c.getenv("HIPPROF_SLOW_US")) |slow| {
        if (dt > atof(slow)) {
            var buf: [512]u8 = undefined;
            stderrLine(&buf, snprintf(&buf, buf.len, "hipprof: %.0f us  %s g=%u,%u,%u b=%u,%u,%u m=%d n=%d k=%d\n", dt, name, sh.g[0], sh.g[1], sh.g[2], sh.b[0], sh.b[1], sh.b[2], sh.m, sh.n, sh.k));
        }
    }
    var rep: f64 = 0;
    if (rep_n > 0 and std.mem.indexOf(u8, name_s, std.mem.sliceTo(rep_match, 0)) != null) {
        const t1 = now();
        for (0..@intCast(rep_n)) |_| _ = p(a, b, c, d, e, f, g, h, i, j, k);
        _ = call1(sync, i);
        rep = (now() - t1) / @as(f64, @floatFromInt(rep_n));
    }
    lock(&mu);
    const row = tab.rowFor(name_s, sh);
    table.Table.addTimed(row, dt);
    if (rep > 0) {
        row.rep_count += 1;
        row.rep_total += rep;
    }
    unlock(&mu);
    return r;
}
