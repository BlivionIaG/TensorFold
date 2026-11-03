//! One EXL3 pool's lines in a fixed virtual range: memory is mapped, in driver chunks, only behind the active ones.

const std = @import("std");
const abi = @import("abi.zig");
const arena = @import("arena.zig");
const Driver = @import("driver.zig").Driver;

pub const Error = arena.Error || error{Invalid};

/// Maps and unmaps one chunk at a fixed address: arena.Vmm.mapAt/unmapAt, or a test double.
pub const Mapper = struct {
    ctx: *anyopaque,
    map: *const fn (ctx: *anyopaque, at: u64, len: usize) Error!void,
    unmap: *const fn (ctx: *anyopaque, at: u64, len: usize) Error!void,
};

/// Chunk arithmetic: lines sit back to back, so less than one chunk per pool is mapped without being used.
pub const Window = struct {
    granularity: usize,
    line_bytes: usize,
    lines: usize,

    /// Chunks that back the first `active` lines.
    pub fn chunksFor(w: Window, active: usize) usize {
        return std.math.divCeil(usize, active * w.line_bytes, w.granularity) catch unreachable;
    }

    /// Virtual bytes to reserve for the capacity.
    pub fn reserveBytes(w: Window) usize {
        return std.mem.alignForward(usize, w.lines * w.line_bytes, w.granularity);
    }

    /// Bytes of the lines that are backed when `chunks` chunks are mapped.
    pub fn usableBytes(w: Window, chunks: usize) usize {
        return @min(chunks * w.granularity, w.lines * w.line_bytes);
    }

    /// Mapped bytes no line of the first `active` uses.
    pub fn wasteBytes(w: Window, active: usize) usize {
        return w.chunksFor(active) * w.granularity - active * w.line_bytes;
    }
};

pub const Elastic = struct {
    win: Window,
    base: u64,
    mapper: Mapper,
    /// Chunks mapped, counted from `base`.
    mapped: usize = 0,
    /// The driver reservation behind a real mapper (null: a test double owns the addresses).
    backing: ?*arena.Vmm = null,

    /// Reserves virtual addresses for `lines` lines of `line_bytes` and maps nothing yet.
    pub fn open(gpa: std.mem.Allocator, d: *const Driver, lines: usize, line_bytes: usize) !Elastic {
        var device: abi.Device = 0;
        try d.check(d.api.cuCtxGetDevice(&device), "cuCtxGetDevice");
        const vmm = try gpa.create(arena.Vmm);
        errdefer gpa.destroy(vmm);
        vmm.* = try arena.Vmm.open(d, device, lines * line_bytes);
        errdefer vmm.close();
        return .{
            .win = .{ .granularity = vmm.granularity, .line_bytes = line_bytes, .lines = lines },
            .base = vmm.va,
            .mapper = .{ .ctx = vmm, .map = vmmMap, .unmap = vmmUnmap },
            .backing = vmm,
        };
    }

    fn vmmMap(ctx: *anyopaque, at: u64, len: usize) Error!void {
        const v: *arena.Vmm = @ptrCast(@alignCast(ctx));
        v.mapAt(at, len) catch |e| {
            if (v.failed) |f| std.log.warn("exl3 elastic lines: {s} failed ({d})", .{ f.call, f.code });
            return e;
        };
    }

    fn vmmUnmap(ctx: *anyopaque, at: u64, len: usize) Error!void {
        const v: *arena.Vmm = @ptrCast(@alignCast(ctx));
        return v.unmapAt(at, len);
    }

    /// Unmaps whatever is mapped and gives the reservation back.
    pub fn close(e: *Elastic, gpa: std.mem.Allocator) void {
        e.move(0) catch {};
        if (e.backing) |vmm| {
            vmm.close();
            gpa.destroy(vmm);
        }
        e.* = undefined;
    }

    fn chunkAt(e: *const Elastic, index: usize) u64 {
        return e.base + @as(u64, index) * e.win.granularity;
    }

    /// Bytes of the lines that have memory now.
    pub fn mappedBytes(e: *const Elastic) usize {
        return e.win.usableBytes(e.mapped);
    }

    /// Backs exactly the first `active` lines or nothing: a failed map or unmap puts this call's chunks back.
    pub fn resize(e: *Elastic, active: usize) Error!void {
        const before = e.mapped;
        e.move(active) catch |err| {
            e.moveChunks(before) catch {};
            return err;
        };
    }

    /// Maps (ascending) or unmaps (descending) chunks until the first `active` lines are backed; stops at a failure.
    fn move(e: *Elastic, active: usize) Error!void {
        if (active > e.win.lines) return error.Invalid;
        try e.moveChunks(e.win.chunksFor(active));
    }

    fn moveChunks(e: *Elastic, need: usize) Error!void {
        while (e.mapped < need) {
            try e.mapper.map(e.mapper.ctx, e.chunkAt(e.mapped), e.win.granularity);
            e.mapped += 1;
        }
        while (e.mapped > need) {
            try e.mapper.unmap(e.mapper.ctx, e.chunkAt(e.mapped - 1), e.win.granularity);
            e.mapped -= 1;
        }
    }
};

const Fake = struct {
    var mapped_at: [16]u64 = undefined;
    var maps: usize = 0;
    var unmapped_at: [16]u64 = undefined;
    var unmaps: usize = 0;
    var live: [16]bool = @splat(false); // the chunks mapped now, by index from the base
    var fail_map_after: usize = std.math.maxInt(usize);
    var fail_unmap_after: usize = std.math.maxInt(usize);

    fn reset() void {
        maps = 0;
        unmaps = 0;
        live = @splat(false);
        fail_map_after = std.math.maxInt(usize);
        fail_unmap_after = std.math.maxInt(usize);
    }

    fn liveCount() usize {
        var n: usize = 0;
        for (live) |m| n += @intFromBool(m);
        return n;
    }

    fn map(_: *anyopaque, at: u64, len: usize) Error!void {
        std.debug.assert(len == 2 << 20);
        if (maps >= fail_map_after) return error.CudaFailed;
        const index: usize = @intCast((at - (1 << 40)) / (2 << 20));
        std.debug.assert(!live[index]);
        live[index] = true;
        mapped_at[maps % mapped_at.len] = at;
        maps += 1;
    }

    fn unmap(_: *anyopaque, at: u64, len: usize) Error!void {
        std.debug.assert(len == 2 << 20);
        if (unmaps >= fail_unmap_after) return error.CudaFailed;
        const index: usize = @intCast((at - (1 << 40)) / (2 << 20));
        std.debug.assert(live[index]);
        live[index] = false;
        unmapped_at[unmaps % unmapped_at.len] = at;
        unmaps += 1;
    }

    fn elastic(lines: usize) Elastic {
        const win: Window = .{ .granularity = 2 << 20, .line_bytes = 1_843_200, .lines = lines };
        return .{ .win = win, .base = 1 << 40, .mapper = .{ .ctx = undefined, .map = map, .unmap = unmap } };
    }
};

test "elastic lines: chunks back the first lines back to back, and waste is less than one chunk" {
    const w: Window = .{ .granularity = 2 << 20, .line_bytes = 1_843_200, .lines = 8 };
    try std.testing.expectEqual(@as(usize, 16 << 20), w.reserveBytes());
    // 1.758 MiB a line: n lines need n chunks up to eight lines (7.03 chunks), then fewer than a chunk per line
    for (0..9) |active| try std.testing.expectEqual(active, w.chunksFor(active));
    try std.testing.expectEqual(@as(usize, 8), w.chunksFor(9));
    try std.testing.expectEqual(@as(usize, 15), w.chunksFor(16));
    for (0..w.lines + 1) |active| try std.testing.expect(w.wasteBytes(active) < w.granularity);
    try std.testing.expectEqual(@as(usize, 2 * (1 << 20) - 1_843_200), w.wasteBytes(1));
    // the last chunk may extend past the capacity: only the lines' bytes are usable
    try std.testing.expectEqual(@as(usize, 8 * 1_843_200), w.usableBytes(w.chunksFor(8)));
    try std.testing.expectEqual(@as(usize, 2 << 20), w.usableBytes(1));
    // the same arithmetic for 2 MiB-aligned lines wastes nothing
    const exact: Window = .{ .granularity = 2 << 20, .line_bytes = 2 << 20, .lines = 4 };
    for (0..5) |active| try std.testing.expectEqual(@as(usize, 0), exact.wasteBytes(active));
}

test "elastic lines: growing maps chunks upward, shrinking unmaps them downward, addresses never move" {
    Fake.reset();
    var e = Fake.elastic(8);
    const g: u64 = 2 << 20;
    try e.resize(3); // 3 lines = 5.27 MiB = 3 chunks
    try std.testing.expectEqual(@as(usize, 3), e.mapped);
    try std.testing.expectEqual([3]u64{ 1 << 40, (1 << 40) + g, (1 << 40) + 2 * g }, Fake.mapped_at[0..3].*);
    try std.testing.expectEqual(@as(usize, 3 * (2 << 20)), e.mappedBytes());
    try e.resize(3); // no change: no driver call
    try std.testing.expectEqual(@as(usize, 3), Fake.maps);
    try e.resize(1);
    try std.testing.expectEqual([2]u64{ (1 << 40) + 2 * g, (1 << 40) + g }, Fake.unmapped_at[0..2].*);
    try std.testing.expectEqual(@as(usize, 1), e.mapped);
    try e.resize(2); // 2 lines = 3.5 MiB: the second chunk comes back at the address it had
    try std.testing.expectEqual((1 << 40) + g, Fake.mapped_at[3]);
    try e.resize(8);
    try std.testing.expectEqual(@as(usize, 8), e.mapped);
    try std.testing.expectEqual(@as(usize, 8 * 1_843_200), e.mappedBytes());
    try std.testing.expectEqual((1 << 40) + 7 * g, Fake.mapped_at[Fake.maps - 1]);
    try std.testing.expectError(error.Invalid, e.resize(9));
    try std.testing.expectEqual(@as(usize, 8), e.mapped);
}

test "elastic lines: a failed map or unmap puts the chunks back, so the mapping is the one it was" {
    Fake.reset();
    var e = Fake.elastic(8);
    try e.resize(5);
    const before = Fake.live;
    // a grow whose third new chunk fails: the two it mapped go back
    Fake.fail_map_after = Fake.maps + 2;
    try std.testing.expectError(error.CudaFailed, e.resize(8));
    try std.testing.expectEqual(@as(usize, 5), e.mapped);
    try std.testing.expectEqual(before, Fake.live);
    Fake.fail_map_after = std.math.maxInt(usize);
    // a shrink whose second unmap fails: the chunk it unmapped comes back at its address
    Fake.fail_unmap_after = Fake.unmaps + 1;
    try std.testing.expectError(error.CudaFailed, e.resize(1));
    try std.testing.expectEqual(@as(usize, 5), e.mapped);
    try std.testing.expectEqual(before, Fake.live);
    // the control: without the putting back, the same failure leaves a half-done shrink
    Fake.fail_unmap_after = Fake.unmaps + 1;
    try std.testing.expectError(error.CudaFailed, e.move(1));
    try std.testing.expectEqual(@as(usize, 4), e.mapped);
    try std.testing.expect(!std.meta.eql(before, Fake.live));
    // when the putting back fails as well, `mapped` still counts what is in place, and a retry finishes
    Fake.fail_unmap_after = Fake.unmaps + 1;
    Fake.fail_map_after = Fake.maps;
    try std.testing.expectError(error.CudaFailed, e.resize(1));
    try std.testing.expectEqual(@as(usize, 3), e.mapped);
    try std.testing.expectEqual(@as(usize, 3), Fake.liveCount());
    Fake.fail_unmap_after = std.math.maxInt(usize);
    Fake.fail_map_after = std.math.maxInt(usize);
    try e.resize(1);
    try std.testing.expectEqual(@as(usize, 1), e.mapped);
    try std.testing.expectEqual(@as(usize, 1), Fake.liveCount());
}

test "elastic lines: a Lines whose resize fails keeps the memory it had, and `mem` still spans exactly that" {
    Fake.reset();
    var driver = @import("paging_test_device.zig").Device.driver();
    var l: @import("exl3_pool.zig").Lines = undefined;
    l.copy = .{ .d = &driver, .handle = @import("paging_test_device.zig").Device.stream };
    l.line_pending = &.{};
    l.stage_busy = &.{};
    var epochs: [8]u64 = @splat(7);
    l.line_epoch = &epochs;
    l.spec.lines = 8;
    l.window = Fake.elastic(8);
    l.mem = .{ .d = &driver, .ptr = 1 << 40, .len = 0, .borrowed = true };
    try l.setMapped(5);
    const bytes = l.mem.len;
    try std.testing.expectEqual(l.window.?.mappedBytes(), bytes);
    try std.testing.expectEqual(@as(u64, 0), epochs[0]);
    const before = Fake.live;
    Fake.fail_map_after = Fake.maps + 1;
    try std.testing.expectError(error.CudaFailed, l.setMapped(8));
    try std.testing.expectEqual(bytes, l.mem.len);
    try std.testing.expectEqual(before, Fake.live);
    Fake.fail_map_after = std.math.maxInt(usize);
    Fake.fail_unmap_after = Fake.unmaps + 1;
    try std.testing.expectError(error.CudaFailed, l.setMapped(1));
    try std.testing.expectEqual(bytes, l.mem.len);
    try std.testing.expectEqual(before, Fake.live);
    try std.testing.expectEqual(@as(u64, 1 << 40), l.mem.ptr); // the base never moved
    Fake.fail_unmap_after = std.math.maxInt(usize);
    try l.setMapped(1);
    try std.testing.expectEqual(@as(usize, 2 << 20), l.mem.len);
}
