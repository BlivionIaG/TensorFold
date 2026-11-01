//! The elastic K/V arena (TF_FLASHNEXT_KV_ELASTIC=1): a virtual range at full size, its chunks mapped on demand.
const std = @import("std");
const cuda = @import("cuda");
const quiet = @import("builtin").is_test;

/// Where the chunks come from: a reserved range at `base`, `chunk` bytes at a time; tests stand in for the driver.
pub const Maps = struct {
    ctx: *anyopaque,
    base: u64,
    chunk: usize,
    map: *const fn (ctx: *anyopaque, at: u64, len: usize) bool,
    unmap: *const fn (ctx: *anyopaque, at: u64, len: usize) bool,
    close: *const fn (ctx: *anyopaque) void,
};

/// Reserves `bytes` of virtual addresses (rounded up to a chunk) and says how they are mapped.
pub const Open = *const fn (gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) anyerror!Maps;

/// The knob's reading: `min_pages` null is the default (a page and a run for every lane).
pub const Elastic = struct { min_pages: ?usize = null, open: Open = openVmm };

/// Parses TF_FLASHNEXT_KV_ELASTIC: unset, empty or 0 is off, 1 is on.
pub fn parseOn(text: ?[]const u8) !bool {
    const v = std.mem.trim(u8, text orelse return false, " \t");
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadFlashNextKvElastic;
}

/// Parses TF_FLASHNEXT_KV_ELASTIC_MIN_PAGES: null when unset.
pub fn parseMin(text: ?[]const u8) !?usize {
    const v = std.mem.trim(u8, text orelse return null, " \t");
    if (v.len == 0) return null;
    return std.fmt.parseInt(usize, v, 10) catch return error.BadFlashNextKvElastic;
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

/// TF_FLASHNEXT_KV_ELASTIC=1 with TF_FLASHNEXT_KV_ELASTIC_MIN_PAGES=n: the elastic arena, off unless asked.
pub fn fromEnv() !?Elastic {
    if (!try parseOn(env("TF_FLASHNEXT_KV_ELASTIC"))) return null;
    return .{ .min_pages = try parseMin(env("TF_FLASHNEXT_KV_ELASTIC_MIN_PAGES")) };
}

/// What the ranks compare at their hello: 0 off, else 1 and the named minimum (0: the default), as both must agree.
pub fn knob() !u64 {
    const e = try fromEnv() orelse return 0;
    return 1 + (e.min_pages orelse 0);
}

/// Zero fills of the new chunks on the engine's stream, and the wait that proves them.
pub const Zero = struct {
    d: *const cuda.Driver,
    stream: cuda.Stream,

    fn fill(z: Zero, at: u64, len: usize) !void {
        const view: cuda.DeviceBuffer = .{ .d = z.d, .ptr = at, .len = len, .borrowed = true };
        try view.fill8(0, z.stream.handle);
    }

    fn wait(z: Zero) !void {
        try z.stream.synchronize();
    }

    fn drain(z: Zero) void {
        cuda.completion.drain(z.d, z.stream, "flashnext: a failed K/V chunk fill did not drain");
    }
};

/// The arena's chunks, from its base up: mapped whole and zeroed before anyone may use them, unmapped from the top.
pub const Span = struct {
    maps: Maps,
    zero: Zero,
    total: usize, // bytes the pages take; the last chunk may reach past them
    chunks: usize = 0, // mapped from the base: [0, chunks)

    pub fn open(gpa: std.mem.Allocator, d: *const cuda.Driver, open_fn: Open, zero: Zero, total: usize) !Span {
        const maps = try open_fn(gpa, d, total);
        return .{ .maps = maps, .zero = zero, .total = total };
    }

    /// Chunks that hold the first `bytes` of the arena (all of them past its end).
    pub fn chunksFor(s: *const Span, bytes: usize) usize {
        return (@min(bytes, s.total) + s.maps.chunk - 1) / s.maps.chunk;
    }

    /// Device bytes held now: whole chunks, the last one included whole.
    pub fn mappedBytes(s: *const Span) usize {
        return s.chunks * s.maps.chunk;
    }

    /// Whole bytes of pages the mapped chunks cover: what a page must fit inside.
    pub fn usableBytes(s: *const Span) usize {
        return @min(s.mappedBytes(), s.total);
    }

    fn addr(s: *const Span, chunk: usize) u64 {
        return s.maps.base + @as(u64, chunk) * s.maps.chunk;
    }

    /// Maps chunks up to `want`, zeroes each and waits for the fills: all of them or none.
    pub fn grow(s: *Span, want: usize) !void {
        if (want <= s.chunks) return;
        const first = s.chunks;
        const c = s.maps.chunk;
        try cuda.memory.mapHold((want - first) * c);
        errdefer cuda.memory.mapFree((want - first) * c);
        var done = first;
        errdefer {
            while (done > first) {
                done -= 1;
                _ = s.maps.unmap(s.maps.ctx, s.addr(done), c);
            }
        }
        while (done < want) : (done += 1) {
            if (!s.maps.map(s.maps.ctx, s.addr(done), c)) return error.FlashNextKvMapFailed;
        }
        errdefer s.zero.drain(); // the fills must be done before the memory goes back
        for (first..want) |i| try s.zero.fill(s.addr(i), c);
        try s.zero.wait();
        s.chunks = want;
    }

    /// Unmaps chunks down to `want`, the top one first; an unmap that fails stops there, the rest still counted.
    pub fn shrink(s: *Span, want: usize) !void {
        while (s.chunks > want) {
            if (!s.maps.unmap(s.maps.ctx, s.addr(s.chunks - 1), s.maps.chunk)) return error.FlashNextKvMapFailed;
            s.chunks -= 1;
            cuda.memory.mapFree(s.maps.chunk);
        }
    }

    /// Everything unmapped and the reservation given back, once nothing runs on the arena.
    pub fn close(s: *Span) void {
        s.shrink(0) catch {};
        s.maps.close(s.maps.ctx);
    }
};

/// The driver's virtual memory behind a `Maps`.
const Real = struct {
    gpa: std.mem.Allocator,
    vmm: cuda.arena.Vmm,

    fn map(ctx: *anyopaque, at: u64, len: usize) bool {
        const r: *Real = @ptrCast(@alignCast(ctx));
        r.vmm.mapAt(at, len) catch return r.failed("map");
        return true;
    }

    fn unmap(ctx: *anyopaque, at: u64, len: usize) bool {
        const r: *Real = @ptrCast(@alignCast(ctx));
        r.vmm.unmapAt(at, len) catch return r.failed("unmap");
        return true;
    }

    fn failed(r: *const Real, what: []const u8) bool {
        if (!quiet) if (r.vmm.failed) |f| std.log.warn("flashnext: elastic K/V {s} failed: {s} (CUresult {d})", .{ what, f.call, f.code });
        return false;
    }

    fn close(ctx: *anyopaque) void {
        const r: *Real = @ptrCast(@alignCast(ctx));
        r.vmm.close();
        r.gpa.destroy(r);
    }
};

pub fn openVmm(gpa: std.mem.Allocator, d: *const cuda.Driver, bytes: usize) anyerror!Maps {
    var device: cuda.abi.Device = 0;
    try d.check(d.api.cuCtxGetDevice(&device), "cuCtxGetDevice");
    const r = try gpa.create(Real);
    errdefer gpa.destroy(r);
    const vmm = cuda.arena.Vmm.open(d, device, bytes) catch |e| {
        if (!quiet) std.log.warn("flashnext: elastic K/V needs CUDA virtual memory: {t}", .{e});
        return error.FlashNextKvElasticUnavailable;
    };
    r.* = .{ .gpa = gpa, .vmm = vmm };
    return .{ .ctx = r, .base = vmm.va, .chunk = vmm.granularity, .map = Real.map, .unmap = Real.unmap, .close = Real.close };
}
