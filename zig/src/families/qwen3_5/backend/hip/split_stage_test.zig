//! `HipLayerStage` through the real `Pipeline` on the recording runtime: its launches between the waits and writes.

const std = @import("std");
const hip = @import("hip");
const split = hip.split;
const view = @import("model/view.zig");
const stage = @import("split_stage.zig");
const fake = @import("split_fake.zig");
const base = @import("split_test.zig");
const Eng = @import("range_test.zig").Eng;

const T = std.testing;
const gpa = T.allocator;
const pipeline = split.pipeline;
const wire = split.wire;

/// A stream device with the stream collapsed to call order: `enqueue` runs the work now and a write lands at once.
const Recorder = struct {
    const Op = union(enum) { wait: struct { offset: usize, value: u32 }, work, write: struct { offset: usize, value: u32 } };
    ops: std.ArrayList(Op) = .empty,
    mem: ?[]align(16 * 1024) u8 = null,

    fn deinit(r: *Recorder) void {
        r.ops.deinit(gpa);
        if (r.mem) |m| gpa.free(m);
    }

    fn me(p: *anyopaque) *Recorder {
        return @ptrCast(@alignCast(p));
    }

    fn caps(_: *anyopaque) split.device.Caps {
        return .{ .backend = "hip", .name = "recorder", .arch = "gfx1030", .vram_bytes = 1 << 30, .peak_bandwidth = 1 << 30, .act = .f16, .launch_ns = 0, .max_windows = 1 };
    }

    fn alloc(_: *anyopaque, _: usize) split.device.Error!split.device.DevPtr {
        return error.Failed;
    }

    fn free(_: *anyopaque, _: split.device.DevPtr) void {}

    fn upload(_: *anyopaque, _: split.device.DevPtr, _: []const u8) split.device.Error!void {
        return error.Failed;
    }

    fn download(_: *anyopaque, _: []u8, _: split.device.DevPtr) split.device.Error!void {
        return error.Failed;
    }

    fn allocWindow(p: *anyopaque, len: usize) split.device.Error!split.device.Window {
        const r = me(p);
        const mem = gpa.alignedAlloc(u8, .fromByteUnits(16 * 1024), (len + 16383) / 16384 * 16384) catch return error.OutOfMemory;
        @memset(mem, 0);
        r.mem = mem;
        return .{ .ptr = mem.ptr, .len = mem.len, .id = 1 };
    }

    fn freeWindow(_: *anyopaque, _: split.device.Window) void {}

    fn enqueue(p: *anyopaque, work: split.device.Work) split.device.Error!void {
        me(p).ops.append(gpa, .work) catch return error.OutOfMemory;
        work.run(work.ctx) catch return error.Failed;
    }

    fn enqueueWait(p: *anyopaque, _: split.device.Window, offset: usize, value: u32, _: u64) split.device.Error!void {
        me(p).ops.append(gpa, .{ .wait = .{ .offset = offset, .value = value } }) catch return error.OutOfMemory;
    }

    fn enqueueWrite(p: *anyopaque, w: split.device.Window, offset: usize, value: u32) split.device.Error!void {
        me(p).ops.append(gpa, .{ .write = .{ .offset = offset, .value = value } }) catch return error.OutOfMemory;
        @atomicStore(u32, w.word(offset), value, .release);
    }

    fn sync(_: *anyopaque, _: u64) split.device.Error!void {}

    fn timestamp(_: *anyopaque) split.device.Error!u64 {
        return 0;
    }

    fn deinitFn(_: *anyopaque) void {}

    const vtable: split.device.VTable = .{
        .caps = caps,
        .alloc = alloc,
        .free = free,
        .upload = upload,
        .download = download,
        .alloc_window = allocWindow,
        .free_window = freeWindow,
        .enqueue = enqueue,
        .enqueue_wait = enqueueWait,
        .enqueue_write = enqueueWrite,
        .sync = sync,
        .timestamp_ns = timestamp,
        .deinit = deinitFn,
    };

    fn device(r: *Recorder) split.device.ExternalDevice {
        return .{ .ptr = r, .vtable = &vtable };
    }
};

/// The host stage: a hidden row from a token (stamped, level A or B), or a token from a hidden row; no arithmetic.
const Host = struct {
    steps: usize = 0,

    fn step(ptr: *anyopaque, seq: u32, _: []const u8, _: u32, _: []const u8, out: []u8, _: pipeline.Level) anyerror!usize {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.steps += 1;
        if (out.len >= wire.stampedLen(4 * 2560)) {
            var row: [4 * 2560]u8 = @splat(0);
            wire.packStamped(out, &row, seq);
            return row.len;
        }
        std.mem.writeInt(u32, out[0..4], 9, .little);
        return 4;
    }

    fn finish(_: *anyopaque, _: u32) anyerror!void {}
};

fn pipelineSteps(order: pipeline.Order, level: pipeline.Level, range: view.Range, steps: usize) !void {
    const t = try Eng.open(range, 64, 1);
    defer t.close();
    try t.e.size(64, 0);
    var st = try stage.HipLayerStage.init(t.e, 16);
    defer st.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = 4 * 2560 };
    const window = try dev.allocWindow(2 * pipeline.Pipeline.slotBytes(cfg));
    var scratch: [4 * 2560]u8 = undefined;
    var host: Host = .{};
    const peer: ?pipeline.Peer = if (order == .external_only) null else .{ .ptr = &host, .step = Host.step, .finish = Host.finish };
    var p = pipeline.Pipeline.init(dev, window, cfg, st.external(), peer, &scratch);
    const slot = pipeline.Pipeline.slotBytes(cfg);
    // the model the stage's launches are compared with: its own range, the hand-offs the order gives it
    const m = t.e.model();
    var marks: base.Marks = .{};
    var rig = try base.Rig.init();
    defer rig.deinit();
    const want = try base.decode(&rig, m, base.handoff(m, order == .metal_first), &marks);
    defer gpa.free(want);
    for (0..steps) |i| {
        fake.clear();
        const before = rec.ops.items.len;
        _ = try p.step(@intCast(100 + i));
        // every step is the wait on the down slot's word, the stage's launches, then the write of the up slot's
        const ops = rec.ops.items[before..];
        try T.expectEqual(@as(usize, 3), ops.len);
        try T.expectEqual(@as(usize, 0), ops[0].wait.offset);
        try T.expectEqual(@as(u32, @intCast(i + 1)), ops[0].wait.value);
        try T.expect(ops[1] == .work);
        try T.expectEqual(slot, ops[2].write.offset);
        try T.expectEqual(@as(u32, @intCast(i + 1)), ops[2].write.value);
        try base.expectSeq(want, fake.launches());
        try T.expectEqual(i + 1, st.pos);
    }
    try T.expectEqual(steps, rec.ops.items.len / 3);
}

test "split stage: the stage's step sits between the pipeline's wait and write, in every order" {
    try pipelineSteps(.external_first, .a, .{ .first = 0, .last = 16 }, 3);
    try pipelineSteps(.metal_first, .a, .{ .first = 16 }, 3);
    try pipelineSteps(.metal_first, .b, .{ .first = 8 }, 2);
    try pipelineSteps(.external_only, .a, .whole, 2);
}

test "split stage: a stage refuses a step that does not fit its range" {
    const t = try Eng.open(.{ .first = 16 }, 64, 1);
    defer t.close();
    try t.e.size(64, 0);
    var st = try stage.HipLayerStage.init(t.e, 4);
    defer st.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * 2560 };
    const window = try dev.allocWindow(2 * pipeline.Pipeline.slotBytes(cfg));
    const n = pipeline.Pipeline.slotBytes(cfg);
    var scratch: [4 * 2560]u8 = undefined;
    const io: pipeline.StepIo = .{ .in = .init(window.ptr, n), .out = .init(window.ptr + n, n), .seq = 1, .stamped = true, .retries = 10, .scratch = &scratch, .hidden_bytes = 4 * 2560 };
    try st.record(io);
    // a row of another width is refused
    var bad = io;
    bad.hidden_bytes = 4 * 2048;
    try T.expectError(error.BadStep, st.record(bad));
    // a step of several rows or several sequences is refused: the stage decodes one position of one sequence
    var wide = io;
    wide.rows = 2;
    try T.expectError(error.BadStep, st.record(wide));
    var two = io;
    two.slots = &.{ 0, 1 };
    try T.expectError(error.BadStep, st.record(two));
    // the caches hold 4 positions: the fifth step is refused by name, and nothing advanced
    for (0..3) |_| try st.record(io);
    try T.expectError(error.ContextFull, st.record(io));
    try T.expectEqual(@as(usize, 4), st.pos);
    // a new sequence starts over
    try st.reset();
    try T.expectEqual(@as(usize, 0), st.pos);
    try st.record(io);
}

fn count(name: []const u8) usize {
    var n: usize = 0;
    for (fake.launches()) |l| n += @intFromBool(std.mem.eql(u8, std.mem.span(l), name));
    return n;
}

test "split stage: warm-up records the plain step and the stamped one, then starts over" {
    const tail = try Eng.open(.{ .first = 16 }, 64, 1);
    defer tail.close();
    try tail.e.size(64, 0);
    var st = try stage.HipLayerStage.init(tail.e, 8);
    defer st.deinit();
    fake.clear();
    try st.warm();
    try T.expectEqual(@as(usize, 1), count("tf_split_import_row"));
    try T.expectEqual(@as(usize, 1), count("tf_split_consume_stamped"));
    try T.expectEqual(@as(usize, 0), st.pos);
    // the first stage has no hidden row to read, so only the plain step runs
    const head = try Eng.open(.{ .first = 0, .last = 16 }, 64, 1);
    defer head.close();
    try head.e.size(64, 0);
    var first = try stage.HipLayerStage.init(head.e, 8);
    defer first.deinit();
    fake.clear();
    try first.warm();
    try T.expectEqual(@as(usize, 0), count("tf_split_consume_stamped"));
    try T.expectEqual(@as(usize, 1), count("tf_split_export_row"));
    try T.expectEqual(@as(usize, 0), first.pos);
}
