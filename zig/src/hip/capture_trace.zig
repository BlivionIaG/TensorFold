//! A HIP runtime with no device that follows hipmac's stream-capture rules and records what a round launches: every kernel (symbol, geometry, the
//! complete argument pack, the arena's state at that moment) and the dependency edges the capture gives them. Test support for `hazard.zig`: the
//! addresses a round computes have no memory behind them, kernels run nowhere.
//!
//! The rules are those of `hipmac/runtime/zig/stream.zig`: a launch on a stream depends on that stream's newest nodes (its "tails") and becomes its only
//! tail; an event records a copy of its stream's tails; a stream that waits on an event has the event's tails added to its own, and a stream that
//! was not part of the capture joins it with only those. A capture that ends with a node no tail of the origin stream follows is "unjoined" and fails.
//! Outside a capture every stream behaves the same, so an eager round yields the stream order the device would run it in.

const std = @import("std");
const abi = @import("abi.zig");
const runtime = @import("runtime.zig");
const launch = @import("launch.zig");
const arena = @import("arena.zig");

const Result = abi.Result;
const ok: Result = 0;
const failed: Result = 1;

pub const Node = struct {
    /// The kernel's symbol, as `Module.function` was asked for it.
    name: []const u8,
    grid: [3]u32,
    block: [3]u32,
    shared: u32,
    /// Index of the stream it was launched on (creation order, 0 = the first made).
    stream: u32,
    /// The argument pack, byte for byte, and where each argument starts in it.
    bytes: []u8,
    offsets: []u16,
    /// How many items the arena's log held at the launch (see `arena.Log`).
    arena_items: usize,
    /// The nodes it waits for, by index in `Trace.nodes`.
    deps: []u32,
};

const Strm = struct {
    index: u32,
    in_capture: bool = false,
    tails: std.ArrayList(u32) = .empty,
};

const Evt = struct {
    recorded: bool = false,
    tails: std.ArrayList(u32) = .empty,
};

const Fn = struct { name: []const u8 };

/// The runtime and what it recorded; one at a time per thread (the C-ABI entry points reach it through `active`).
pub const Trace = struct {
    gpa: std.mem.Allocator,
    nodes: std.ArrayList(Node) = .empty,
    streams: std.ArrayList(*Strm) = .empty,
    events: std.ArrayList(*Evt) = .empty,
    fns: std.ArrayList(*Fn) = .empty,
    /// The arena log of the round being recorded, to stamp nodes with.
    arena_log: ?*const arena.Log = null,
    capturing: bool = false,
    origin: ?*Strm = null,
    /// The first node of the capture in progress.
    capture_first: usize = 0,
    invalidated: bool = false,
    /// Launches during a capture on a stream that is not part of it (HIP would run them at once, outside the graph).
    stray: usize = 0,
    /// Launches whose arguments were not seen (the observer was not installed).
    unobserved: usize = 0,
    next_va: u64 = 0x1000_0000,
    pending: ?Pending = null,
    previous_observer: ?launch.Observer = null,

    const Pending = struct { name: []const u8, bytes: []u8, offsets: []u16 };

    pub fn init(gpa: std.mem.Allocator) Trace {
        return .{ .gpa = gpa };
    }

    pub fn deinit(t: *Trace) void {
        for (t.nodes.items) |n| {
            t.gpa.free(n.bytes);
            t.gpa.free(n.offsets);
            t.gpa.free(n.deps);
        }
        t.nodes.deinit(t.gpa);
        for (t.streams.items) |s| {
            s.tails.deinit(t.gpa);
            t.gpa.destroy(s);
        }
        t.streams.deinit(t.gpa);
        for (t.events.items) |e| {
            e.tails.deinit(t.gpa);
            t.gpa.destroy(e);
        }
        t.events.deinit(t.gpa);
        for (t.fns.items) |f| {
            t.gpa.free(f.name);
            t.gpa.destroy(f);
        }
        t.fns.deinit(t.gpa);
        null_stream.tails.deinit(t.gpa);
        null_stream.tails = .empty;
        if (t.pending) |p| {
            t.gpa.free(p.bytes);
            t.gpa.free(p.offsets);
        }
        t.* = undefined;
    }

    /// Makes this the runtime the C entry points reach and the launch observer; `uninstall` before it goes out of scope.
    pub fn install(t: *Trace) void {
        active = t;
        t.previous_observer = launch.observer;
        launch.observer = .{ .ctx = t, .seen = seen };
    }

    pub fn uninstall(t: *Trace) void {
        launch.observer = t.previous_observer;
        active = null;
    }

    /// A runtime whose every call goes to this trace.
    pub fn runtimeOf(t: *Trace) runtime.Runtime {
        _ = t;
        return .{ .lib = undefined, .api = api };
    }

    /// Everything recorded since the last `clear` is dropped, and the streams' and events' edges with it (the streams, events and functions stay).
    pub fn clear(t: *Trace) void {
        for (t.nodes.items) |n| {
            t.gpa.free(n.bytes);
            t.gpa.free(n.offsets);
            t.gpa.free(n.deps);
        }
        t.nodes.clearRetainingCapacity();
        for (t.streams.items) |s| s.tails.clearRetainingCapacity();
        for (t.events.items) |e| {
            e.recorded = false;
            e.tails.clearRetainingCapacity();
        }
        null_stream.tails.clearRetainingCapacity();
        t.stray = 0;
        t.unobserved = 0;
        t.invalidated = false;
    }

    /// For every node, the set of nodes it follows (through any path of `deps`), as bit rows of `words` u64 each.
    pub fn ancestors(t: *const Trace, gpa: std.mem.Allocator) !Closure {
        const n = t.nodes.items.len;
        const words = (n + 63) / 64;
        const bits = try gpa.alloc(u64, @max(n * words, 1));
        @memset(bits, 0);
        for (t.nodes.items, 0..) |node, i| {
            const row = bits[i * words ..][0..words];
            for (node.deps) |d| {
                row[d / 64] |= @as(u64, 1) << @intCast(d % 64);
                const from = bits[d * words ..][0..words];
                for (row, from) |*r, f| r.* |= f;
            }
        }
        return .{ .bits = bits, .words = words };
    }

    pub fn nameOf(t: *const Trace, node: usize) []const u8 {
        return t.nodes.items[node].name;
    }

    /// The `i`-th argument of node `node` as `T` (the pack holds its bytes).
    pub fn arg(t: *const Trace, comptime T: type, node: usize, i: usize) T {
        const n = t.nodes.items[node];
        return std.mem.bytesToValue(T, n.bytes[n.offsets[i]..][0..@sizeOf(T)]);
    }

    fn seen(ctx: *anyopaque, f: @import("module.zig").Function, cfg: launch.Config, _: @import("stream.zig").Stream, args: *const launch.Args) void {
        _ = cfg;
        const t: *Trace = @ptrCast(@alignCast(ctx));
        const rec: *Fn = @ptrCast(@alignCast(f.handle.?));
        const bytes = t.gpa.dupe(u8, args.storage[0..args.used]) catch @panic("out of memory");
        const offsets = t.gpa.dupe(u16, args.offsets[0..args.count]) catch @panic("out of memory");
        if (t.pending) |p| {
            t.gpa.free(p.bytes);
            t.gpa.free(p.offsets);
        }
        t.pending = .{ .name = rec.name, .bytes = bytes, .offsets = offsets };
    }

    fn launchOn(t: *Trace, f: abi.Function, grid: [3]u32, block: [3]u32, shared: u32, s: *Strm) Result {
        const rec: *Fn = @ptrCast(@alignCast(f.?));
        if (t.capturing and !s.in_capture) t.stray += 1;
        const deps = t.gpa.dupe(u32, s.tails.items) catch return failed;
        var bytes: []u8 = &.{};
        var offsets: []u16 = &.{};
        if (t.pending) |p| {
            bytes = p.bytes;
            offsets = p.offsets;
            t.pending = null;
        } else {
            t.unobserved += 1;
            bytes = t.gpa.alloc(u8, 0) catch return failed;
            offsets = t.gpa.alloc(u16, 0) catch return failed;
        }
        t.nodes.append(t.gpa, .{
            .name = rec.name,
            .grid = grid,
            .block = block,
            .shared = shared,
            .stream = s.index,
            .bytes = bytes,
            .offsets = offsets,
            .arena_items = if (t.arena_log) |l| l.items.items.len else 0,
            .deps = deps,
        }) catch return failed;
        s.tails.clearRetainingCapacity();
        s.tails.append(t.gpa, @intCast(t.nodes.items.len - 1)) catch return failed;
        return ok;
    }
};

/// `ancestors`' answer: whether node `later` follows node `earlier` in the graph.
pub const Closure = struct {
    bits: []u64,
    words: usize,

    pub fn deinit(c: Closure, gpa: std.mem.Allocator) void {
        gpa.free(c.bits);
    }

    pub fn follows(c: Closure, later: usize, earlier: usize) bool {
        return c.bits[later * c.words + earlier / 64] >> @intCast(earlier % 64) & 1 != 0;
    }
};

threadlocal var active: ?*Trace = null;

fn tr() *Trace {
    return active orelse @panic("capture_trace: no active trace");
}

/// The null stream (handle 0) is a stream like the others, never part of a capture.
fn strm(s: abi.Stream) *Strm {
    return if (s) |p| @ptrCast(@alignCast(p)) else &null_stream;
}

threadlocal var null_stream: Strm = .{ .index = std.math.maxInt(u32) };

fn evt(e: abi.Event) *Evt {
    return @ptrCast(@alignCast(e.?));
}

fn hipInit(_: c_uint) callconv(.c) Result {
    return ok;
}
fn hipGetDeviceCount(n: *c_int) callconv(.c) Result {
    n.* = 1;
    return ok;
}
fn hipDeviceGet(d: *abi.Device, _: c_int) callconv(.c) Result {
    d.* = 0;
    return ok;
}
fn hipDevicePrimaryCtxRetain(c: *abi.Context, _: abi.Device) callconv(.c) Result {
    c.* = @ptrFromInt(0x40);
    return ok;
}
fn hipDevicePrimaryCtxRelease(_: abi.Device) callconv(.c) Result {
    return ok;
}
fn hipCtxSetCurrent(_: abi.Context) callconv(.c) Result {
    return ok;
}
fn hipDeviceSynchronize() callconv(.c) Result {
    return ok;
}
fn hipMalloc(p: *abi.DevicePtr, n: usize) callconv(.c) Result {
    const t = tr();
    p.* = @ptrFromInt(t.next_va);
    t.next_va += std.mem.alignForward(u64, @max(n, 1), 1 << 20);
    return ok;
}
fn hipFree(_: abi.DevicePtr) callconv(.c) Result {
    return ok;
}
fn hipHostMalloc(p: *abi.DevicePtr, n: usize, _: c_uint) callconv(.c) Result {
    p.* = std.c.malloc(@max(n, 1)) orelse return failed;
    return ok;
}
fn hipHostFree(p: abi.DevicePtr) callconv(.c) Result {
    std.c.free(p);
    return ok;
}
fn hipMemcpyHtoDAsync(_: abi.DevicePtr, _: [*]const u8, _: usize, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipMemcpyDtoHAsync(_: [*]u8, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipMemcpyHtoD(_: abi.DevicePtr, _: [*]const u8, _: usize) callconv(.c) Result {
    return ok;
}
fn hipMemcpyDtoH(_: [*]u8, _: abi.DevicePtr, _: usize) callconv(.c) Result {
    return ok;
}
fn hipMemset(_: abi.DevicePtr, _: c_int, _: usize) callconv(.c) Result {
    return ok;
}
fn hipMemsetAsync(_: abi.DevicePtr, _: c_int, _: usize, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipStreamCreateWithFlags(out: *abi.Stream, _: c_uint) callconv(.c) Result {
    const t = tr();
    const s = t.gpa.create(Strm) catch return failed;
    s.* = .{ .index = @intCast(t.streams.items.len) };
    t.streams.append(t.gpa, s) catch return failed;
    out.* = @ptrCast(s);
    return ok;
}
fn hipStreamDestroy(_: abi.Stream) callconv(.c) Result {
    return ok;
}
/// A host wait inside a capture invalidates it (hipmac's `hm_stream_sync`).
fn hipStreamSynchronize(s: abi.Stream) callconv(.c) Result {
    if (strm(s).in_capture) {
        tr().invalidated = true;
        return failed;
    }
    return ok;
}
fn hipModuleLoadData(m: *abi.Module, _: [*]const u8) callconv(.c) Result {
    m.* = @ptrFromInt(0x80);
    return ok;
}
fn hipModuleUnload(_: abi.Module) callconv(.c) Result {
    return ok;
}
fn hipModuleGetFunction(out: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) Result {
    const t = tr();
    const f = t.gpa.create(Fn) catch return failed;
    f.* = .{ .name = t.gpa.dupe(u8, std.mem.span(name)) catch return failed };
    t.fns.append(t.gpa, f) catch return failed;
    out.* = @ptrCast(f);
    return ok;
}
fn hipStreamQuery(_: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipStreamWaitEvent(s: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) Result {
    const t = tr();
    const st = strm(s);
    const ev = evt(e);
    if (!ev.recorded) return ok;
    if (t.capturing and !st.in_capture) {
        st.in_capture = true;
        st.tails.clearRetainingCapacity();
    }
    for (ev.tails.items) |n| {
        if (std.mem.indexOfScalar(u32, st.tails.items, n) == null) st.tails.append(t.gpa, n) catch return failed;
    }
    return ok;
}
fn hipEventCreateWithFlags(out: *abi.Event, _: c_uint) callconv(.c) Result {
    const t = tr();
    const e = t.gpa.create(Evt) catch return failed;
    e.* = .{};
    t.events.append(t.gpa, e) catch return failed;
    out.* = @ptrCast(e);
    return ok;
}
fn hipEventDestroy(_: abi.Event) callconv(.c) Result {
    return ok;
}
fn hipEventRecord(e: abi.Event, s: abi.Stream) callconv(.c) Result {
    const t = tr();
    const ev = evt(e);
    ev.tails.clearRetainingCapacity();
    ev.tails.appendSlice(t.gpa, strm(s).tails.items) catch return failed;
    ev.recorded = true;
    return ok;
}
fn hipEventSynchronize(_: abi.Event) callconv(.c) Result {
    return ok;
}
fn hipEventQuery(_: abi.Event) callconv(.c) Result {
    return ok;
}
fn hipEventElapsedTime(ms: *f32, _: abi.Event, _: abi.Event) callconv(.c) Result {
    ms.* = 0;
    return ok;
}
fn hipMemGetInfo(free_bytes: *usize, total: *usize) callconv(.c) Result {
    free_bytes.* = 6 << 30;
    total.* = 8 << 30;
    return ok;
}
fn hipModuleLaunchKernel(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) Result {
    return tr().launchOn(f, .{ gx, gy, gz }, .{ bx, by, bz }, shared, strm(s));
}
fn hipModuleLaunchCooperativeKernel(f: abi.Function, gx: c_uint, gy: c_uint, gz: c_uint, bx: c_uint, by: c_uint, bz: c_uint, shared: c_uint, s: abi.Stream, _: ?[*]?*anyopaque) callconv(.c) Result {
    return tr().launchOn(f, .{ gx, gy, gz }, .{ bx, by, bz }, shared, strm(s));
}
fn hipGetErrorName(_: Result) callconv(.c) ?[*:0]const u8 {
    return "capture_trace";
}
fn hipRuntimeGetVersion(v: *c_int) callconv(.c) Result {
    v.* = 70000000;
    return ok;
}
fn hipDeviceGetName(buf: [*]u8, n: c_int, _: abi.Device) callconv(.c) Result {
    const name = "gfx900 (fake)";
    const len: usize = @min(name.len, @as(usize, @intCast(n)) - 1);
    @memcpy(buf[0..len], name[0..len]);
    buf[len] = 0;
    return ok;
}
fn hipDeviceGetAttribute(v: *c_int, a: abi.DeviceAttribute, _: c_int) callconv(.c) Result {
    v.* = switch (a) {
        .multiprocessor_count => 56,
        .warp_size => 64,
        .max_threads_per_block => 1024,
        .max_shared_memory_per_block, .shared_mem_per_block_optin => 65536,
        .max_shared_memory_per_multiprocessor => 65536,
        .l2_cache_size => 4 << 20,
        .cooperative_launch => 1,
        else => 0,
    };
    return ok;
}
fn hipHostGetDevicePointer(out: *abi.DevicePtr, host: ?*anyopaque, _: c_uint) callconv(.c) Result {
    out.* = host;
    return ok;
}
fn hipMemcpyDtoD(_: abi.DevicePtr, _: abi.DevicePtr, _: usize) callconv(.c) Result {
    return ok;
}
fn hipMemcpyDtoDAsync(_: abi.DevicePtr, _: abi.DevicePtr, _: usize, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipMemsetD32(_: abi.DevicePtr, _: c_int, _: usize) callconv(.c) Result {
    return ok;
}
fn hipMemsetD32Async(_: abi.DevicePtr, _: c_int, _: usize, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipModuleGetGlobal(p: *abi.DevicePtr, n: *usize, _: abi.Module, _: [*:0]const u8) callconv(.c) Result {
    const t = tr();
    p.* = @ptrFromInt(t.next_va);
    t.next_va += 1 << 20;
    n.* = 1 << 16;
    return ok;
}
fn hipFuncGetAttribute(v: *c_int, a: abi.FunctionAttribute, _: abi.Function) callconv(.c) Result {
    v.* = switch (a) {
        .max_threads_per_block => 1024,
        .num_regs => 64,
        .max_dynamic_shared_size_bytes => 65536,
        else => 0,
    };
    return ok;
}
fn hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(n: *c_int, _: abi.Function, _: c_int, _: usize) callconv(.c) Result {
    n.* = 4;
    return ok;
}
fn hipStreamBeginCapture(s: abi.Stream, _: abi.CaptureMode) callconv(.c) Result {
    const t = tr();
    const st = strm(s);
    if (t.capturing) return failed;
    t.capturing = true;
    t.invalidated = false;
    t.origin = st;
    t.capture_first = t.nodes.items.len;
    st.in_capture = true;
    st.tails.clearRetainingCapacity();
    return ok;
}
/// Like HIP: a capture whose side streams were never joined back (a node no tail of the origin follows) ends invalidated.
fn hipStreamEndCapture(s: abi.Stream, g: *abi.Graph) callconv(.c) Result {
    const t = tr();
    g.* = null;
    const st = strm(s);
    if (!t.capturing or t.origin != st) return failed;
    var bad = t.invalidated;
    if (!bad) {
        var c = t.ancestors(t.gpa) catch return failed;
        defer c.deinit(t.gpa);
        const first = t.capture_first;
        for (first..t.nodes.items.len) |i| {
            var reached = false;
            for (st.tails.items) |tail| if (tail == i or c.follows(tail, i)) {
                reached = true;
                break;
            };
            if (!reached) bad = true;
        }
    }
    t.capturing = false;
    t.origin = null;
    for (t.streams.items) |x| {
        x.in_capture = false;
        x.tails.clearRetainingCapacity();
    }
    for (t.events.items) |e| {
        e.recorded = false;
        e.tails.clearRetainingCapacity();
    }
    if (bad) return failed;
    g.* = @ptrFromInt(0x1000);
    return ok;
}
fn hipStreamIsCapturing(s: abi.Stream, status: *abi.CaptureStatus) callconv(.c) Result {
    status.* = if (strm(s).in_capture) .active else .none;
    return ok;
}
fn hipGraphCreate(g: *abi.Graph, _: c_uint) callconv(.c) Result {
    g.* = @ptrFromInt(0x1000);
    return ok;
}
fn hipGraphDestroy(_: abi.Graph) callconv(.c) Result {
    return ok;
}
fn hipGraphAddKernelNode(_: *abi.GraphNode, _: abi.Graph, _: ?[*]const abi.GraphNode, _: usize, _: *const abi.KernelNodeParams) callconv(.c) Result {
    return failed;
}
fn hipGraphKernelNodeSetParams(_: abi.GraphNode, _: *const abi.KernelNodeParams) callconv(.c) Result {
    return failed;
}
fn hipGraphAddDependencies(_: abi.Graph, _: [*]const abi.GraphNode, _: [*]const abi.GraphNode, _: usize) callconv(.c) Result {
    return failed;
}
fn hipGraphGetNodes(_: abi.Graph, _: ?[*]abi.GraphNode, n: *usize) callconv(.c) Result {
    n.* = 0;
    return ok;
}
fn hipGraphInstantiateWithFlags(e: *abi.GraphExec, _: abi.Graph, _: c_ulonglong) callconv(.c) Result {
    e.* = @ptrFromInt(0x2000);
    return ok;
}
fn hipGraphUpload(_: abi.GraphExec, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipGraphLaunch(_: abi.GraphExec, _: abi.Stream) callconv(.c) Result {
    return ok;
}
fn hipGraphExecDestroy(_: abi.GraphExec) callconv(.c) Result {
    return ok;
}
fn hipGraphExecUpdate(_: abi.GraphExec, _: abi.Graph, n: *abi.GraphNode, r: *abi.ExecUpdateResult) callconv(.c) Result {
    n.* = null;
    r.* = .success;
    return ok;
}
fn hipGraphExecKernelNodeSetParams(_: abi.GraphExec, _: abi.GraphNode, _: *const abi.KernelNodeParams) callconv(.c) Result {
    return failed;
}

const api: abi.Api = .{
    .hipInit = hipInit,
    .hipGetDeviceCount = hipGetDeviceCount,
    .hipDeviceGet = hipDeviceGet,
    .hipDevicePrimaryCtxRetain = hipDevicePrimaryCtxRetain,
    .hipDevicePrimaryCtxRelease = hipDevicePrimaryCtxRelease,
    .hipCtxSetCurrent = hipCtxSetCurrent,
    .hipDeviceSynchronize = hipDeviceSynchronize,
    .hipMalloc = hipMalloc,
    .hipFree = hipFree,
    .hipHostMalloc = hipHostMalloc,
    .hipHostFree = hipHostFree,
    .hipMemcpyHtoDAsync = hipMemcpyHtoDAsync,
    .hipMemcpyDtoHAsync = hipMemcpyDtoHAsync,
    .hipMemcpyHtoD = hipMemcpyHtoD,
    .hipMemcpyDtoH = hipMemcpyDtoH,
    .hipMemset = hipMemset,
    .hipMemsetAsync = hipMemsetAsync,
    .hipStreamCreateWithFlags = hipStreamCreateWithFlags,
    .hipStreamDestroy = hipStreamDestroy,
    .hipStreamSynchronize = hipStreamSynchronize,
    .hipModuleLoadData = hipModuleLoadData,
    .hipModuleUnload = hipModuleUnload,
    .hipModuleGetFunction = hipModuleGetFunction,
    .hipStreamQuery = hipStreamQuery,
    .hipStreamWaitEvent = hipStreamWaitEvent,
    .hipEventCreateWithFlags = hipEventCreateWithFlags,
    .hipEventDestroy = hipEventDestroy,
    .hipEventRecord = hipEventRecord,
    .hipEventSynchronize = hipEventSynchronize,
    .hipEventQuery = hipEventQuery,
    .hipEventElapsedTime = hipEventElapsedTime,
    .hipMemGetInfo = hipMemGetInfo,
    .hipModuleLaunchKernel = hipModuleLaunchKernel,
    .hipGetErrorName = hipGetErrorName,
    .hipRuntimeGetVersion = hipRuntimeGetVersion,
    .hipDeviceGetName = hipDeviceGetName,
    .hipDeviceGetAttribute = hipDeviceGetAttribute,
    .hipHostGetDevicePointer = hipHostGetDevicePointer,
    .hipMemcpyDtoD = hipMemcpyDtoD,
    .hipMemcpyDtoDAsync = hipMemcpyDtoDAsync,
    .hipMemsetD32 = hipMemsetD32,
    .hipMemsetD32Async = hipMemsetD32Async,
    .hipModuleGetGlobal = hipModuleGetGlobal,
    .hipFuncGetAttribute = hipFuncGetAttribute,
    .hipModuleOccupancyMaxActiveBlocksPerMultiprocessor = hipModuleOccupancyMaxActiveBlocksPerMultiprocessor,
    .hipModuleLaunchCooperativeKernel = hipModuleLaunchCooperativeKernel,
    .hipStreamBeginCapture = hipStreamBeginCapture,
    .hipStreamEndCapture = hipStreamEndCapture,
    .hipStreamIsCapturing = hipStreamIsCapturing,
    .hipGraphCreate = hipGraphCreate,
    .hipGraphDestroy = hipGraphDestroy,
    .hipGraphAddKernelNode = hipGraphAddKernelNode,
    .hipGraphKernelNodeSetParams = hipGraphKernelNodeSetParams,
    .hipGraphAddDependencies = hipGraphAddDependencies,
    .hipGraphGetNodes = hipGraphGetNodes,
    .hipGraphInstantiateWithFlags = hipGraphInstantiateWithFlags,
    .hipGraphUpload = hipGraphUpload,
    .hipGraphLaunch = hipGraphLaunch,
    .hipGraphExecDestroy = hipGraphExecDestroy,
    .hipGraphExecUpdate = hipGraphExecUpdate,
    .hipGraphExecKernelNodeSetParams = hipGraphExecKernelNodeSetParams,
};

// ---- tests ----

const Module = @import("module.zig").Module;
const Function = @import("module.zig").Function;
const Stream = @import("stream.zig").Stream;
const Event = @import("stream.zig").Event;
const graph = @import("graph.zig");
const fork_mod = @import("fork.zig");

/// A fake runtime with one probe kernel; built in place (the module and the streams point at the runtime).
const Probe = struct {
    trace: Trace,
    rt: runtime.Runtime,
    mod: Module,
    f: Function,

    fn start(p: *Probe, gpa: std.mem.Allocator) !void {
        p.trace = .init(gpa);
        p.trace.install();
        p.rt = p.trace.runtimeOf();
        const image: [8]u8 align(8) = @splat(0);
        p.mod = try Module.load(&p.rt, &image);
        p.f = try p.mod.function("probe");
    }

    fn stop(p: *Probe) void {
        p.mod.unload();
        p.trace.uninstall();
        p.trace.deinit();
    }

    /// A launch of the probe on `s` with one pointer argument.
    fn kick(p: *Probe, s: Stream, at: u64) !void {
        var args: launch.Args = .{};
        try args.add(at);
        try launch.launch(p.f, .{ .grid = .{ .x = 4 }, .block = .{ .x = 64 } }, s, &args);
    }
};

test "capture_trace: launches follow their stream, an event's fork and join order the others" {
    var p: Probe = undefined;
    try p.start(std.testing.allocator);
    defer p.stop();
    var main = try Stream.init(&p.rt);
    defer main.deinit();
    var side = try Stream.init(&p.rt);
    defer side.deinit();
    var fork_ev = try Event.init(&p.rt, false);
    defer fork_ev.deinit();
    var join_ev = try Event.init(&p.rt, false);
    defer join_ev.deinit();
    try graph.beginCapture(main, .thread_local);
    try p.kick(main, 0x10); // 0
    try fork_ev.record(main);
    try side.wait(fork_ev);
    try p.kick(side, 0x20); // 1
    try p.kick(main, 0x30); // 2
    try join_ev.record(side);
    try main.wait(join_ev);
    try p.kick(main, 0x40); // 3
    var g = try graph.endCapture(main);
    g.deinit();
    const nodes = p.trace.nodes.items;
    try std.testing.expectEqual(@as(usize, 4), nodes.len);
    try std.testing.expectEqualSlices(u32, &.{}, nodes[0].deps);
    try std.testing.expectEqualSlices(u32, &.{0}, nodes[1].deps);
    try std.testing.expectEqualSlices(u32, &.{0}, nodes[2].deps);
    try std.testing.expectEqual(@as(usize, 2), nodes[3].deps.len);
    try std.testing.expect(std.mem.indexOfScalar(u32, nodes[3].deps, 1) != null);
    try std.testing.expect(std.mem.indexOfScalar(u32, nodes[3].deps, 2) != null);
    try std.testing.expectEqual(@as(u64, 0x20), p.trace.arg(u64, 1, 0));
    var c = try p.trace.ancestors(std.testing.allocator);
    defer c.deinit(std.testing.allocator);
    try std.testing.expect(c.follows(3, 0) and c.follows(3, 1) and c.follows(3, 2));
    try std.testing.expect(!c.follows(1, 2) and !c.follows(2, 1));
    try std.testing.expectEqual(@as(usize, 0), p.trace.stray);
}

test "capture_trace: a side stream that is never joined fails the capture" {
    var p: Probe = undefined;
    try p.start(std.testing.allocator);
    defer p.stop();
    var main = try Stream.init(&p.rt);
    defer main.deinit();
    var side = try Stream.init(&p.rt);
    defer side.deinit();
    var fork_ev = try Event.init(&p.rt, false);
    defer fork_ev.deinit();
    try graph.beginCapture(main, .thread_local);
    try p.kick(main, 0x10);
    try fork_ev.record(main);
    try side.wait(fork_ev);
    try p.kick(side, 0x20);
    try p.kick(main, 0x30);
    try std.testing.expectError(error.HipFailed, graph.endCapture(main));
    // the next capture starts clean
    try graph.beginCapture(main, .thread_local);
    try p.kick(main, 0x10);
    var g = try graph.endCapture(main);
    g.deinit();
}

test "capture_trace: a launch on a stream the capture never reached is stray" {
    var p: Probe = undefined;
    try p.start(std.testing.allocator);
    defer p.stop();
    var main = try Stream.init(&p.rt);
    defer main.deinit();
    var other = try Stream.init(&p.rt);
    defer other.deinit();
    try graph.beginCapture(main, .thread_local);
    try p.kick(main, 0x10);
    try p.kick(other, 0x20);
    try std.testing.expectEqual(@as(usize, 1), p.trace.stray);
    try std.testing.expectError(error.HipFailed, graph.endCapture(main));
}

test "capture_trace: a host wait inside a capture invalidates it" {
    var p: Probe = undefined;
    try p.start(std.testing.allocator);
    defer p.stop();
    var main = try Stream.init(&p.rt);
    defer main.deinit();
    try graph.beginCapture(main, .thread_local);
    try p.kick(main, 0x10);
    try std.testing.expect(std.meta.isError(main.synchronize()));
    try std.testing.expect(p.trace.invalidated);
    try std.testing.expectError(error.HipFailed, graph.endCapture(main));
}

test "fork: a branch joins back, is reusable, and does nothing without a fork or on the counting stream" {
    var p: Probe = undefined;
    try p.start(std.testing.allocator);
    defer p.stop();
    var main = try Stream.init(&p.rt);
    defer main.deinit();
    var f = try fork_mod.Fork.init(&p.rt);
    defer f.deinit();
    for (0..2) |_| {
        p.trace.clear();
        try graph.beginCapture(main, .thread_local);
        try p.kick(main, 0x10); // 0
        const b = try fork_mod.Branch.begin(&f, main);
        try p.kick(.{ .r = &p.rt, .handle = b.side }, 0x20); // 1
        try p.kick(main, 0x30); // 2
        try b.end();
        try p.kick(main, 0x40); // 3
        var g = try graph.endCapture(main);
        g.deinit();
        const nodes = p.trace.nodes.items;
        try std.testing.expectEqual(@as(usize, 4), nodes.len);
        try std.testing.expectEqual(f.side.handle, b.side);
        try std.testing.expectEqualSlices(u32, &.{0}, nodes[1].deps);
        try std.testing.expectEqualSlices(u32, &.{0}, nodes[2].deps);
        try std.testing.expectEqual(@as(usize, 2), nodes[3].deps.len);
    }
    // no fork: the branch is the stream itself, and nothing is recorded
    const events = p.trace.events.items.len;
    const none = try fork_mod.Branch.begin(null, main);
    try std.testing.expectEqual(main.handle, none.side);
    try none.end();
    const counted = try fork_mod.Branch.begin(&f, .{ .r = &p.rt, .handle = @import("launches/util.zig").counting });
    try std.testing.expectEqual(@import("launches/util.zig").counting, counted.side);
    try counted.end();
    try std.testing.expectEqual(events, p.trace.events.items.len);
}
