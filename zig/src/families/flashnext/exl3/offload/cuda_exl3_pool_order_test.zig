//! The pool on a recording driver: what a pass reads is settled before it runs, however the device schedules streams.

const std = @import("std");
const cuda = @import("cuda");
const fpool = @import("cuda_exl3_pool.zig");
const abi = cuda.abi;
const testing = std.testing;

const max_ops = 12288;
const words = max_ops / 64;
const base: u64 = 1 << 20;
const arena = 1 << 16;

const Op = struct { kind: enum { copy, word, record, wait, pass }, stream: usize, dst: u64 = 0, bytes: []const u8 = &.{}, value: u32 = 0 };

/// The device as the driver calls describe it: an op starts after its stream's earlier ops and the events it waited on.
const Dev = struct {
    var gpa: std.mem.Allocator = undefined;
    var ops: std.ArrayList(Op) = .empty;
    var pred: []usize = &.{}; // row i: the ops that finished before op i starts
    var settled: [words]usize = @splat(0); // what a host wait proved finished: every later op follows it
    var last: [8]?usize = @splat(null); // a stream's newest op
    var recorded: [512]?usize = @splat(null); // an event's newest record
    var streams: usize = 0;
    var events: usize = 0;
    var top: u64 = base;

    fn start(a: std.mem.Allocator) !cuda.Driver {
        gpa = a;
        ops = .empty;
        pred = try a.alloc(usize, max_ops * words);
        settled = @splat(0);
        last = @splat(null);
        recorded = @splat(null);
        streams = 0;
        events = 0;
        top = base;
        var d: cuda.Driver = undefined;
        d.api.cuMemAlloc_v2 = alloc;
        d.api.cuMemFree_v2 = free;
        d.api.cuMemHostAlloc = hostAlloc;
        d.api.cuMemFreeHost = hostFree;
        d.api.cuMemsetD8_v2 = fill;
        d.api.cuMemsetD32Async = word;
        d.api.cuMemcpyHtoDAsync_v2 = copy;
        d.api.cuStreamCreate = streamCreate;
        d.api.cuStreamDestroy_v2 = streamDestroy;
        d.api.cuStreamSynchronize = streamSync;
        d.api.cuStreamWaitEvent = wait;
        d.api.cuEventCreate = eventCreate;
        d.api.cuEventDestroy_v2 = eventDestroy;
        d.api.cuEventRecord = record;
        d.api.cuEventSynchronize = eventSync;
        d.api.cuEventQuery = query;
        return d;
    }

    fn stop() void {
        for (ops.items) |op| gpa.free(op.bytes);
        ops.deinit(gpa);
        gpa.free(pred);
    }

    fn row(i: usize) []usize {
        return pred[i * words ..][0..words];
    }

    /// True when op `a` has finished before op `b` starts, on every schedule the driver calls allow.
    fn before(a: usize, b: usize) bool {
        return (row(b)[a / 64] >> @intCast(a % 64)) & 1 != 0;
    }

    fn follow(dst: []usize, i: usize) void {
        for (dst, row(i)) |*d, s| d.* |= s;
        dst[i / 64] |= @as(usize, 1) << @intCast(i % 64);
    }

    fn push(op: Op, after: ?usize) usize {
        const i = ops.items.len;
        if (i == max_ops) @panic("the recording driver's op log is full");
        ops.append(gpa, op) catch @panic("out of memory");
        @memcpy(row(i), &settled);
        if (last[op.stream]) |p| follow(row(i), p);
        if (after) |p| follow(row(i), p);
        last[op.stream] = i;
        return i;
    }

    fn alloc(p: *abi.DevicePtr, len: usize) callconv(.c) abi.Result {
        p.* = top;
        top += (len + 255) & ~@as(u64, 255);
        return abi.success;
    }
    fn free(_: abi.DevicePtr) callconv(.c) abi.Result {
        return abi.success;
    }
    fn hostAlloc(p: *?*anyopaque, len: usize, _: c_uint) callconv(.c) abi.Result {
        p.* = std.c.malloc(len);
        return if (p.* == null) abi.error_not_found else abi.success;
    }
    fn hostFree(p: ?*anyopaque) callconv(.c) abi.Result {
        std.c.free(p);
        return abi.success;
    }
    fn fill(_: abi.DevicePtr, value: u8, _: usize) callconv(.c) abi.Result {
        std.debug.assert(value == 0); // the tables start absent: the model's memory starts zeroed
        return abi.success;
    }
    fn word(dst: abi.DevicePtr, value: c_uint, n: usize, s: abi.Stream) callconv(.c) abi.Result {
        std.debug.assert(n == 1);
        _ = push(.{ .kind = .word, .stream = @intFromPtr(s), .dst = dst, .value = value }, null);
        return abi.success;
    }
    fn copy(dst: abi.DevicePtr, src: ?*const anyopaque, len: usize, s: abi.Stream) callconv(.c) abi.Result {
        const bytes = gpa.dupe(u8, @as([*]const u8, @ptrCast(src.?))[0..len]) catch return abi.error_not_found;
        _ = push(.{ .kind = .copy, .stream = @intFromPtr(s), .dst = dst, .bytes = bytes }, null);
        return abi.success;
    }
    fn streamCreate(s: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
        streams += 1;
        s.* = @ptrFromInt(streams);
        return abi.success;
    }
    fn streamDestroy(_: abi.Stream) callconv(.c) abi.Result {
        return abi.success;
    }
    fn eventCreate(e: *abi.Event, _: c_uint) callconv(.c) abi.Result {
        events += 1;
        e.* = @ptrFromInt(events);
        return abi.success;
    }
    fn eventDestroy(_: abi.Event) callconv(.c) abi.Result {
        return abi.success;
    }
    fn record(e: abi.Event, s: abi.Stream) callconv(.c) abi.Result {
        recorded[@intFromPtr(e)] = push(.{ .kind = .record, .stream = @intFromPtr(s) }, null);
        return abi.success;
    }
    fn wait(s: abi.Stream, e: abi.Event, _: c_uint) callconv(.c) abi.Result {
        _ = push(.{ .kind = .wait, .stream = @intFromPtr(s) }, recorded[@intFromPtr(e)]);
        return abi.success;
    }
    /// The slowest device a host may meet: an event is known finished only once a host wait covered its record.
    fn query(e: abi.Event) callconv(.c) abi.Result {
        const r = recorded[@intFromPtr(e)] orelse return abi.success;
        return if ((settled[r / 64] >> @intCast(r % 64)) & 1 != 0) abi.success else abi.error_not_ready;
    }
    fn eventSync(e: abi.Event) callconv(.c) abi.Result {
        if (recorded[@intFromPtr(e)]) |r| follow(&settled, r);
        return abi.success;
    }
    fn streamSync(s: abi.Stream) callconv(.c) abi.Result {
        if (last[@intFromPtr(s)]) |p| follow(&settled, p);
        return abi.success;
    }
};

const Range = struct { at: u64, len: usize };

/// One compute pass as the engine would launch it: the experts it computes, their lines, its picks on the device.
const Pass = struct { op: usize, ids: []u16, lines: []u32, picks: u64, want: []i32, lo: usize, hi: usize = 0, strict: bool };

const ids_n = 24;
const slots = 6;
const max_rows = 3;

const Run = struct {
    copies: std.ArrayList(fpool.round.Copy) = .empty,
    passes: std.ArrayList(Pass) = .empty,
    split: usize = 0,
    natural: usize = 0,
    waits: u64 = 0,

    fn deinit(r: *Run, gpa: std.mem.Allocator) void {
        for (r.passes.items) |p| {
            gpa.free(p.ids);
            gpa.free(p.lines);
            gpa.free(p.want);
        }
        r.passes.deinit(gpa);
        r.copies.deinit(gpa);
    }
};

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn unsetenv(name: [*:0]const u8) c_int;

fn overlaps(op: Op, len: usize, rg: Range) ?[2]u64 {
    const lo = @max(op.dst, rg.at);
    const hi = @min(op.dst + len, rg.at + rg.len);
    return if (lo < hi) .{ lo, hi } else null;
}

/// Replays what finished before the pass, then holds its experts, table words and picks against the source.
fn settle(pool: *const fpool.Pool, routed: []const [3][]const u8, p: Pass, mem: []u8) !void {
    @memset(mem, 0);
    for (Dev.ops.items[0..p.op], 0..) |op, i| {
        if (!Dev.before(i, p.op)) continue;
        if (op.kind == .copy) @memcpy(mem[op.dst - base ..][0..op.bytes.len], op.bytes);
        if (op.kind == .word) std.mem.writeInt(u32, mem[op.dst - base ..][0..4], op.value, .little);
    }
    var ranges: [max_rows * slots * 9 + 1]Range = undefined;
    var n: usize = 0;
    const l = &pool.lines;
    for (p.ids, p.lines) |id, line| for (0..3) |w| {
        const entry = l.tables.ptr + l.tlay.ptr[w] + @as(u64, id) * 8;
        const width = l.tables.ptr + l.tlay.k2[w] + @as(u64, id) * 4;
        const want = routed[id][w];
        try testing.expectEqual(l.at(line, w), std.mem.readInt(u64, mem[entry - base ..][0..8], .little));
        try testing.expectEqual(pool.k2[id][w], std.mem.readInt(i32, mem[width - base ..][0..4], .little));
        try testing.expectEqualSlices(u8, want, mem[l.at(line, w) - base ..][0..want.len]);
        ranges[n..][0..3].* = .{ .{ .at = entry, .len = 8 }, .{ .at = width, .len = 4 }, .{ .at = l.at(line, w), .len = want.len } };
        n += 3;
    };
    if (p.picks != 0) {
        try testing.expectEqualSlices(u8, std.mem.sliceAsBytes(p.want), mem[p.picks - base ..][0 .. p.want.len * 4]);
        ranges[n] = .{ .at = p.picks, .len = p.want.len * 4 };
        n += 1;
    }
    // a write its own round did not order against a hit-first pass may not reach what it reads; any other, equal bytes
    for (Dev.ops.items, 0..) |op, i| {
        if (op.kind != .copy and op.kind != .word) continue;
        if (i == p.op or Dev.before(i, p.op) or Dev.before(p.op, i)) continue;
        var value: [4]u8 = undefined;
        std.mem.writeInt(u32, &value, op.value, .little);
        const bytes: []const u8 = if (op.kind == .word) &value else op.bytes;
        for (ranges[0..n]) |rg| {
            const cut = overlaps(op, bytes.len, rg) orelse continue;
            if (p.strict and i >= p.lo and i < p.hi) return error.UnorderedWrite;
            if (!std.mem.eql(u8, bytes[cut[0] - op.dst .. cut[1] - op.dst], mem[cut[0] - base .. cut[1] - base])) return error.UnorderedWrite;
        }
    }
}

/// 96 rounds of one to three rows through a ten-line pool; `cut` waits for the compute stream as decode does a layer.
fn drive(gpa: std.mem.Allocator, batch: bool, cut: bool, hit_first: usize, out: *Run) !void {
    var d = try Dev.start(gpa);
    defer Dev.stop();
    const store = try gpa.alloc(u8, ids_n * 96);
    defer gpa.free(store);
    for (store, 0..) |*b, i| b.* = @truncate(i *% 31 +% i / 96);
    var routed: [fpool.shape.experts][3][]const u8 = @splat(.{ &.{}, &.{}, &.{} });
    for (routed[0..ids_n], 0..) |*e, i| e.* = .{ store[i * 96 ..][0..16], store[i * 96 + 16 ..][0..32], store[i * 96 + 48 ..][0..48] };
    // set or cleared either way: the test never depends on the caller's environment (batch needs host planning)
    _ = unsetenv("TF_FLASHNEXT_EXL3_GRAPH");
    if (batch) _ = setenv("TF_FLASHNEXT_TABLE_BATCH", "1", 1) else _ = unsetenv("TF_FLASHNEXT_TABLE_BATCH");
    const pool = fpool.Pool.init(gpa, undefined, &d, .{ .routed = &routed, .shared = null }, std.mem.zeroes(fpool.Scales), .{
        .layer = 3,
        .lines = fpool.min_lines,
        .dims = 16,
        .width = 16,
        .max_rows = max_rows,
        .slots = slots,
        .direct = true,
        .hit_first = hit_first,
    }, null);
    _ = unsetenv("TF_FLASHNEXT_TABLE_BATCH");
    var p = try pool;
    defer p.deinit();
    try testing.expectEqual(batch, p.lines.batch);
    const s = try cuda.Stream.init(&d, true);
    var prng = std.Random.DefaultPrng.init(0x4817);
    const rng = prng.random();
    var member: [ids_n]bool = undefined;
    for (0..96) |round| {
        var picks: [max_rows * slots]i32 = undefined;
        const rows: usize = if (round % 4 == 3) max_rows else 1 + round % 2;
        for (picks[0 .. rows * slots]) |*pk| pk.* = if (rng.uintLessThan(u8, 9) == 0) -1 else @min(rng.uintLessThan(u8, ids_n), rng.uintLessThan(u8, ids_n));
        const lo = Dev.ops.items.len;
        const r = try p.planHitFirst(picks[0 .. rows * slots], rows);
        try out.copies.appendSlice(gpa, r.copies[0..r.n_copies]);
        out.split += @intFromBool(p.hit_first);
        out.natural += @intFromBool(r.masked() and !p.hit_first);
        for (0..r.n_passes) |j| {
            try p.stage(r, j, s);
            @memset(&member, false);
            for (r.passIds(j)) |id| member[id] = true;
            const want = try gpa.dupe(i32, picks[0 .. rows * slots]);
            errdefer gpa.free(want);
            for (want) |*pk| pk.* = if (pk.* >= 0 and member[@intCast(pk.*)]) pk.* else -1;
            const ids = try gpa.dupe(u16, r.passIds(j));
            errdefer gpa.free(ids);
            const lines = try gpa.dupe(u32, r.passLines(j));
            errdefer gpa.free(lines);
            const op = Dev.push(.{ .kind = .pass, .stream = @intFromPtr(s.handle) }, null);
            try out.passes.append(gpa, .{ .op = op, .ids = ids, .lines = lines, .picks = p.picksFor(r, j, 0), .want = want, .lo = lo, .strict = p.hit_first });
            try p.release(r, j, s);
        }
        for (out.passes.items[out.passes.items.len - r.n_passes ..]) |*pass| pass.hi = Dev.ops.items.len;
        if (cut) try s.synchronize();
    }
    out.waits = p.lines.stats.gate_waits;
    const mem = try gpa.alloc(u8, arena);
    defer gpa.free(mem);
    try testing.expect(Dev.top - base <= arena);
    for (out.passes.items) |pass| try settle(&p, &routed, pass, mem);
}

test "exl3: hit first keeps every pass's lines, table words and picks settled, batched tables or not, cut or not" {
    const gpa = testing.allocator;
    for ([_]bool{ false, true }) |batch| for ([_]bool{ true, false }) |cut| {
        var plain: Run = .{};
        defer plain.deinit(gpa);
        try drive(gpa, batch, cut, 0, &plain);
        var first: Run = .{};
        defer first.deinit(gpa);
        try drive(gpa, batch, cut, 1, &first);
        // the same copies into the same lines over the same victims, round after round: only the passes differ
        try testing.expectEqualSlices(fpool.round.Copy, plain.copies.items, first.copies.items);
        try testing.expectEqual(@as(usize, 0), plain.split);
        try testing.expect(first.split > 20 and first.natural > 5);
        try testing.expectEqual(plain.natural, first.natural);
        try testing.expectEqual(plain.passes.items.len + first.split, first.passes.items.len);
    };
}
