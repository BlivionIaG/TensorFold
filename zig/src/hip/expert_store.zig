//! A MoE model's routed experts on one HIP device: a host tier, a pinned window read in place, and cache lines.
const std = @import("std");
const core = @import("core");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const stream_mod = @import("stream.zig");
const stream_value = @import("stream_value.zig");
const moe = @import("moe.zig");
const abi = @import("abi.zig");

const service_mod = core.expert_service;
const Runtime = runtime.Runtime;
const DeviceBuffer = memory.DeviceBuffer;
const HostBuffer = memory.HostBuffer;
const Stream = stream_mod.Stream;

pub const Error = stream_value.Error || std.mem.Allocator.Error || error{ BadShape, NotSized, StageFailed };

pub const parts = core.expert_window.parts;
pub const Shape = core.expert_window.Shape;
pub const Stats = core.expert_stats.Stats;

/// What a layer's views need: its table, its projections' blob offsets, the stage hook, and an address for stubs.
pub const Ref = struct { ptab: u64, fused_off: u64, down_off: u64, anchor: u64, hook: moe.Hook };

/// Device memory is allocated in pieces of at most this many bytes: one contiguous block of gigabytes may not exist.
const chunk_bytes: usize = 512 << 20;

/// The cache lines in device memory, and the scratch a prompt span's experts are copied into by a copy stream.
const Cache = struct {
    chunks: []DeviceBuffer,
    per_chunk: usize,
    scratch: DeviceBuffer,
    copy: Stream,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    r: *const Runtime,
    service: service_mod.Service,
    /// The window's pinned, device-mapped memory; the tier's table, trace and slots live in it.
    win: HostBuffer,
    /// The shared experts, one blob a layer, always resident.
    shared: DeviceBuffer,
    values: stream_value.ValueOps,
    cache: ?Cache = null,

    /// A window of about `window_bytes` pinned (less when the runtime refuses, down to what the halves need).
    pub fn init(gpa: std.mem.Allocator, r: *const Runtime, sh: Shape, window_bytes: usize) Error!Store {
        try sh.check();
        const values = try stream_value.ValueOps.resolve(r);
        const Window = core.expert_window.Window;
        const least = Window.bytesFor(sh, Window.leastSlots(sh));
        var want = window_bytes;
        var win: HostBuffer = undefined;
        while (true) {
            const slots = Window.slotsFor(sh, want);
            if (slots < Window.leastSlots(sh)) return error.BadShape;
            if (HostBuffer.allocMapped(r, Window.bytesFor(sh, slots))) |w| {
                win = w;
                break;
            } else |err| {
                if (want / 4 * 3 < least) return err;
                want = want / 4 * 3;
            }
        }
        errdefer win.free();
        var shared = try DeviceBuffer.alloc(r, sh.layers * sh.blobBytes());
        errdefer shared.free();
        var tier = try service_mod.Tier.init(gpa, sh, win.bytes, try win.device(), shared.base());
        errdefer tier.deinit();
        return .{ .gpa = gpa, .r = r, .service = try service_mod.Service.init(tier), .win = win, .shared = shared, .values = values };
    }

    pub fn deinit(s: *Store) void {
        s.service.deinit();
        if (s.cache) |*c| {
            c.copy.deinit();
            c.scratch.free();
            for (c.chunks) |*chunk| chunk.free();
            s.gpa.free(c.chunks);
        }
        s.shared.free();
        s.win.free();
        s.* = undefined;
    }

    pub fn shape(s: *const Store) Shape {
        return s.service.tier.shape;
    }

    pub fn blobBytes(s: *const Store) usize {
        return s.shape().blobBytes();
    }

    /// Host bytes of the tier and device bytes of what is always resident (the shared experts).
    pub fn hostBytes(sh: Shape) usize {
        return service_mod.Tier.hostBytes(sh);
    }

    pub fn residentBytes(sh: Shape) usize {
        return service_mod.Tier.residentBytes(sh);
    }

    /// Layer `layer`'s experts: `tensors` are the six parts of every expert, the shared expert last.
    pub fn addLayer(s: *Store, layer: usize, tensors: [parts][]const u8) Error!void {
        const tier = &s.service.tier;
        try tier.addLayer(layer, tensors);
        const one = try s.gpa.alloc(u8, s.blobBytes());
        defer s.gpa.free(one);
        service_mod.Tier.sharedBlob(tier.shape, tensors, one);
        try s.shared.upload(layer * one.len, one);
    }

    /// The addresses layer `layer`'s views carry.
    pub fn ref(s: *Store, layer: usize) Ref {
        const tier = &s.service.tier;
        return .{
            .ptab = tier.win.layerTable(tier.shape, layer),
            .fused_off = 0,
            .down_off = tier.shape.downOffset(),
            .anchor = tier.win.dev,
            .hook = .{ .ctx = s, .post = post, .layer = @intCast(layer), .trace = tier.win.dev + tier.win.trace_off },
        };
    }

    /// `total` device lines (a span's scratch, one layer's experts, among them); `protect_pct`: SLRU's protected share.
    pub fn attachPool(s: *Store, total: usize, copy_experts: usize, protect_pct: u8) Error!void {
        std.debug.assert(s.cache == null);
        const sh = s.shape();
        const blob = sh.blobBytes();
        if (total <= 2 * sh.experts) return error.NotSized;
        const lines = total - sh.experts;
        const per_chunk = @max(1, chunk_bytes / blob);
        const chunks = try s.gpa.alloc(DeviceBuffer, (lines + per_chunk - 1) / per_chunk);
        errdefer s.gpa.free(chunks);
        var made: usize = 0;
        errdefer for (chunks[0..made]) |*c| c.free();
        for (chunks, 0..) |*c, i| {
            c.* = try DeviceBuffer.alloc(s.r, @min(per_chunk, lines - i * per_chunk) * blob);
            made += 1;
        }
        var scratch = try DeviceBuffer.alloc(s.r, sh.experts * blob);
        errdefer scratch.free();
        var copy = try Stream.init(s.r);
        errdefer copy.deinit();
        try s.service.tier.attachPool(lines, copy_experts, protect_pct);
        s.cache = .{ .chunks = chunks, .per_chunk = per_chunk, .scratch = scratch, .copy = copy };
    }

    pub fn lineCount(s: *const Store) usize {
        return s.service.tier.lineCount();
    }

    /// Starts the worker that serves the layers' stages; the first offloaded forward needs it.
    pub fn start(s: *Store) Error!void {
        try s.service.start(s.ports());
    }

    /// A round's first stage may be served: the worker has answered every stage posted so far.
    pub fn beginRound(s: *Store) void {
        s.service.beginRound();
    }

    /// The round over its first `rows` rows is finished: its time joins the counters, the admission budget renews.
    pub fn endRound(s: *Store, rows: usize, ns: u64) Error!void {
        try s.service.endRound(rows, ns);
    }

    /// The device's side of a stage: where a cache line is, and the copy of a window slot into the scratch.
    fn ports(s: *Store) service_mod.Ports {
        return .{
            .ctx = s,
            .line_addr = lineAddr,
            .scratch_base = if (s.cache) |c| c.scratch.base() else 0,
            .scratch_copy = scratchCopy,
            .scratch_wait = scratchWait,
        };
    }

    fn lineAddr(ctx: *anyopaque, slot: usize) u64 {
        const s: *Store = @ptrCast(@alignCast(ctx));
        const c = &s.cache.?;
        return c.chunks[slot / c.per_chunk].base() + (slot % c.per_chunk) * s.blobBytes();
    }

    fn scratchCopy(ctx: *anyopaque, at: usize, window_off: usize, len: usize) error{CopyFailed}!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        const c = &s.cache.?;
        c.scratch.uploadAsync(at, s.win.view(window_off, window_off + len), c.copy) catch return error.CopyFailed;
    }

    fn scratchWait(ctx: *anyopaque) error{CopyFailed}!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.cache.?.copy.synchronize() catch return error.CopyFailed;
    }

    /// A layer's stage in the stream, after its picks were traced: the device asks (`req`), then waits for `ack`.
    fn post(ctx: *anyopaque, stream: abi.Stream, layer: u32, rows: usize, prefill: bool) moe.Error!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        // a dry run measuring scratch launches nothing and waits for no one
        if (stream == @import("launches/util.zig").counting) return;
        const n = s.service.post(.{ .layer = layer, .rows = @intCast(rows), .prefill = prefill }) catch return error.BadShape;
        try s.values.write(stream, s.service.reqWord(), n);
        try s.values.waitAtLeast(stream, s.service.ackWord(), n);
    }
};
