//! The engine opened over a layer range and `HipLayerStage` through the real `Pipeline`, on the recording runtime
//! (split_fake.zig): allocation sizes, the refusals, and the stage's launches between the pipeline's waits and writes.

const std = @import("std");
const hip = @import("hip");
const split = @import("split");
const view = @import("model/view.zig");
const weights = @import("model/weights.zig");
const bridge = @import("model/bridge.zig");
const state = @import("forward/state.zig");
const pages = @import("forward/pages.zig");
const plan = @import("forward/plan.zig");
const fwd = @import("forward/forward.zig");
const round_graphs = @import("engine/round_graphs.zig");
const Engine = @import("engine/engine.zig").Engine;
const hip_lanes = @import("engine/hip_lanes.zig");
const stage = @import("split_stage.zig");
const fake = @import("split_fake.zig");
const base = @import("split_test.zig");

const T = std.testing;
const gpa = T.allocator;
const pipeline = split.pipeline;
const wire = split.wire;

/// An engine over the 4B's shape and a layer range, sized on the stand-in runtime; the weights are fake addresses.
const Eng = struct {
    e: *Engine,

    fn open(range: view.Range, capacity: usize, rows: usize) !Eng {
        if (!fake.available()) return error.SkipZigTest;
        const e = try gpa.create(Engine);
        errdefer gpa.destroy(e);
        var f = try fake.Fixture.init(gpa, range, .f16);
        errdefer f.deinit();
        const b = try gpa.create(bridge.Bridge);
        errdefer gpa.destroy(b);
        e.* = .{
            .gpa = gpa,
            .driver = fake.runtime(),
            .ctx = undefined,
            .lib = undefined,
            .stream = undefined,
            .weights = .{ .gpa = gpa, .spec = fake.spec_4b, .range = f.model.range },
            .bridge = b,
            .o = .{ .capacity = capacity, .batch_rows = rows, .prompt_rows = 64, .range = range },
            .sized = false,
            .stopped = false,
            .act = .f16,
            .dtype = .f16,
            .rounds = undefined,
            .prompts = undefined,
            .ids = undefined,
            .ids_dev = undefined,
            .pool = undefined,
            .drawer = undefined,
            .graphs = round_graphs.Graphs.init(gpa),
        };
        e.lib = try hip.Launcher.load(&e.driver, (hip.Policy{}).choices(), hip.kernels.images);
        e.stream = try hip.Stream.init(&e.driver);
        b.* = .{
            .gpa = gpa,
            .layers = f.layers,
            .q_weight = try hip.DeviceBuffer.alloc(&e.driver, 512),
            .k_weight = try hip.DeviceBuffer.alloc(&e.driver, 512),
            .model = f.model,
        };
        return .{ .e = e };
    }

    /// `Engine.deinit` without the runtime and context the stand-in does not own.
    fn close(t: Eng) void {
        const e = t.e;
        if (e.sized) {
            e.round.deinit(e);
            e.drawer.deinit();
            e.ids_dev.free();
            e.ids.free();
            e.prompts.deinit();
            e.rounds.deinit();
            e.pool.deinit();
        }
        e.graphs.deinit();
        e.bridge.deinit();
        e.weights.deinit();
        e.stream.deinit();
        e.lib.unload();
        gpa.destroy(e);
    }
};

/// Device bytes `size` holds for an engine over `range` (it checks them against its own plan and returns them).
fn sizedBytes(range: view.Range, capacity: usize, rows: usize) !usize {
    const t = try Eng.open(range, capacity, rows);
    defer t.close();
    const before = hip.usage(false).device;
    try t.e.size(capacity, 0);
    return hip.usage(false).device - before;
}

test "split stage: an engine over half the layers sizes to what its range holds" {
    const whole = try sizedBytes(.whole, 256, 4);
    const lo = try sizedBytes(.{ .first = 0, .last = 16 }, 256, 4);
    const hi = try sizedBytes(.{ .first = 16 }, 256, 4);
    try T.expect(lo < whole and hi < whole);
    // the pool and the linear state are the layers' own: the halves hold the whole's KV pages between them
    const pool = struct {
        fn of(range: view.Range) !usize {
            var f = try fake.Fixture.init(gpa, range, .f16);
            defer f.deinit();
            return pages.Pool.deviceBytes(&f.model, 10);
        }
    };
    try T.expectEqual(try pool.of(.whole), try pool.of(.{ .first = 0, .last = 16 }) + try pool.of(.{ .first = 16 }));
    try T.expect(try pool.of(.{ .first = 0, .last = 16 }) > 0);
    // a range of linear layers only holds no KV pages at all
    try T.expectEqual(@as(usize, 0), try pool.of(.{ .first = 0, .last = 3 }));
}

test "split stage: a ranged engine's caches hold state for its layers only" {
    var f = try fake.Fixture.init(gpa, .{ .first = 4, .last = 12 }, .f16);
    defer f.deinit();
    var rt = fake.runtime();
    var pool = try pages.Pool.init(gpa, &rt, &f.model, 2, false);
    defer pool.deinit();
    var c = try state.Caches.init(gpa, &pool, &f.model, 128);
    defer c.deinit(gpa);
    for (c.layers, 0..) |l, i| {
        const owned = i >= 4 and i < 12;
        try T.expectEqual(owned and !fake.spec_4b.full(i), l == .linear);
        try T.expectEqual(owned and fake.spec_4b.full(i), l == .full);
        try T.expectEqual(!owned, l == .absent);
        try T.expectEqual(owned and fake.spec_4b.full(i), pool.keys[i].len != 0);
    }
    // 8 layers, 2 of them full attention: 6 linear states and 2 layers of pages
    var whole = try fake.Fixture.init(gpa, .whole, .f16);
    defer whole.deinit();
    try T.expect(state.Caches.deviceBytes(&f.model, 128) < state.Caches.deviceBytes(&whole.model, 128));
}

test "split stage: a range is refused where it cannot run" {
    var rt = fake.runtime();
    _ = &rt;
    try T.expectError(error.BadLayerRange, (view.Range{ .first = 20, .last = 40 }).check(32));
    // a ranged engine runs no lane round, no whole-model prefill and takes no drafter taps
    const t = try Eng.open(.{ .first = 0, .last = 16 }, 128, 1);
    defer t.close();
    try t.e.size(128, 0);
    try T.expect(t.e.ranged());
    var caches = try t.e.newCaches(8);
    defer caches.deinit(gpa);
    try T.expectError(error.RangedEngine, t.e.verify(&.{}, &.{}, &.{}));
    try T.expectError(error.RangedEngine, t.e.choose(&.{}, null));
    try T.expectError(error.RangedEngine, t.e.spanRows(&caches, &.{1}, 0, 1, null));
    try T.expectError(error.RangedEngine, t.e.prefill(&caches, &.{1}, 0, null, undefined, null));
    try T.expectError(error.RangeWithDraft, t.e.setTaps(&.{3}));
    // the lane host (drafting, verify rounds, snapshots) is refused before it builds anything
    try T.expectError(error.RangeWithDraft, hip_lanes.Hip.init(gpa, t.e));
}

/// A stream device's window protocol with the stream collapsed to the call order: `enqueue` runs the work now, as a HIP
/// stream records launches, and the stream's write lands at once, so a step's operations are in the log in stream order.
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

    fn unsupported() split.device.Error {
        return error.Failed;
    }

    fn alloc(_: *anyopaque, _: usize) split.device.Error!split.device.DevPtr {
        return unsupported();
    }

    fn free(_: *anyopaque, _: split.device.DevPtr) void {}

    fn upload(_: *anyopaque, _: split.device.DevPtr, _: []const u8) split.device.Error!void {
        return unsupported();
    }

    fn download(_: *anyopaque, _: []u8, _: split.device.DevPtr) split.device.Error!void {
        return unsupported();
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

    fn step(ptr: *anyopaque, seq: u32, _: []const u8, rows: u32, _: []const u8, out: []u8, _: pipeline.Level) anyerror!usize {
        const h: *Host = @ptrCast(@alignCast(ptr));
        h.steps += 1;
        if (out.len >= wire.stampedLen(4 * 2560)) {
            const zero: [4 * 4 * 2560]u8 = @splat(0);
            wire.packStampedRows(out, &zero, rows, 4 * 2560, seq);
            return rows * 4 * 2560;
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
    var st = try stage.HipLayerStage.init(t.e, 16, .{});
    defer st.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = 4 * 2560 };
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
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
    var st = try stage.HipLayerStage.init(t.e, 4, .{});
    defer st.deinit();
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * 2560 };
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    const n = pipeline.Pipeline.slotBytes(cfg);
    var scratch: [4 * 2560]u8 = undefined;
    const io: pipeline.StepIo = .{ .in = .init(window.ptr, n), .out = .init(window.ptr + n, n), .seq = 1, .stamped = true, .retries = 10, .scratch = &scratch, .hidden_bytes = 4 * 2560 };
    try st.record(io);
    // a row of another width, and a last stage handed a token's worth of window
    var bad = io;
    bad.hidden_bytes = 4 * 2048;
    try T.expectError(error.BadStep, st.record(bad));
    // the caches hold 4 positions: the fifth step is refused by name, and nothing advanced
    for (0..3) |_| try st.record(io);
    try T.expectError(error.ContextFull, st.record(io));
    try T.expectEqual(@as(usize, 4), st.pos);
    // a new sequence starts over
    try st.reset();
    try T.expectEqual(@as(usize, 0), st.pos);
    try st.record(io);
}

/// A pipeline of `max_rows` 4 over `rec`'s window at `offset`, its stage and its host.
const Lane = struct {
    st: stage.HipLayerStage,
    host: Host = .{},
    scratch: [4 * 4 * 2560]u8 = undefined,
    p: pipeline.Pipeline = undefined,

    fn start(l: *Lane, e: *Engine, dev: split.device.ExternalDevice, window: split.device.Window, order: pipeline.Order, level: pipeline.Level, offset: usize) !void {
        l.st = try stage.HipLayerStage.init(e, 16, .{ .max_rows = 4 });
        const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = 4 * 2560, .max_rows = 4, .window_offset = offset };
        const peer: ?pipeline.Peer = if (order == .external_only) null else .{ .ptr = &l.host, .step = Host.step, .finish = Host.finish };
        l.p = pipeline.Pipeline.init(dev, window, cfg, l.st.external(), peer, &l.scratch);
    }
};

/// One step of `rows` positions through the pipeline: the launches are the range's prompt span and its hand-offs, and the
/// boundary kernels read and write the window's rows where the protocol puts them, one consume launch a stamped row.
fn pipelineRows(order: pipeline.Order, level: pipeline.Level, range: view.Range, rows: usize) !void {
    const t = try Eng.open(range, 64, 1);
    defer t.close();
    try t.e.size(64, 0);
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = 4 * 2560, .max_rows = 4 };
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    var lane: Lane = undefined;
    try lane.start(t.e, dev, window, order, level, 0);
    defer lane.st.deinit();
    const m = t.e.model();
    const stamped = order == .metal_first;
    var rig = try base.Rig.init();
    defer rig.deinit();
    var marks: base.Marks = .{};
    const want = try base.span(&rig, m, base.handoff(m, stamped), rows, 0, true, &marks);
    defer gpa.free(want);
    fake.clear();
    _ = try lane.p.stepRows((&[4]u32{ 5, 6, 7, 8 })[0..rows]);
    try base.expectSeq(want, fake.launches());
    try T.expectEqual(rows, lane.st.pos);

    const hidden: u32 = 2560;
    const got = fake.boundaries();
    const in_at = @intFromPtr(lane.p.down.data().ptr);
    const out_at = @intFromPtr(lane.p.up.data().ptr);
    var at: usize = 0;
    if (!m.range.embeds() and stamped) {
        // each row on its own groups, its own status word, its own activation row
        for (0..rows) |r| {
            const c = got[at];
            try T.expect(c.kind == .consume);
            try T.expectEqual(in_at + r * wire.stampedLen(4 * hidden), c.window);
            try T.expectEqual(@as(u32, hidden), c.n);
            try T.expectEqual(@as(u32, 1), c.seq);
            try T.expectEqual(got[0].status + 4 * r, c.status);
            try T.expectEqual(got[0].device + r * hidden * 2, c.device);
            at += 1;
        }
    } else if (!m.range.embeds()) {
        // plain rows are contiguous: one import for all of them
        try T.expect(got[at].kind == .import);
        try T.expectEqual(in_at, got[at].window);
        try T.expectEqual(@as(u32, @intCast(rows)) * hidden, got[at].n);
        at += 1;
    }
    if (!m.headed()) {
        try T.expect(got[at].kind == .export_);
        try T.expectEqual(out_at, got[at].window);
        try T.expectEqual(@as(u32, @intCast(rows)) * hidden, got[at].n);
        at += 1;
    }
    try T.expectEqual(at, got.len);
    // the stage's status words: one a stamped row, none otherwise
    try T.expectEqual(if (stamped) rows else 0, lane.st.stamped_rows);
    // a longer step than the pipeline's maximum is refused by the pipeline, and one beyond the stage's by the stage
    var over: pipeline.StepIo = .{ .in = lane.p.down, .out = lane.p.up, .seq = 2, .rows = 5, .stamped = stamped, .retries = 1, .scratch = &lane.scratch, .hidden_bytes = 4 * 2560 };
    try T.expectError(error.BadStep, lane.st.record(over));
    over.rows = 0;
    try T.expectError(error.BadStep, lane.st.record(over));
}

test "split stage: a step of several rows is one span, one import (or a consume a row) and one export or head" {
    try pipelineRows(.external_first, .a, .{ .first = 0, .last = 16 }, 3);
    try pipelineRows(.metal_first, .a, .{ .first = 16 }, 3);
    try pipelineRows(.metal_first, .b, .{ .first = 8, .last = 24 }, 4);
    try pipelineRows(.metal_first, .a, .{ .first = 16 }, 2);
    try pipelineRows(.external_only, .a, .whole, 3);
}

test "split stage: sequences on one engine take their own plan buffers, in turn, and prompt chunks go as spans" {
    const range: view.Range = .{ .first = 16 };
    const t = try Eng.open(range, 64, 1);
    defer t.close();
    try t.e.size(64, 0);
    var rec: Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * 2560, .max_rows = 4 };
    const slot = pipeline.Pipeline.slotBytes(cfg);
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 2));
    var lanes: [2]Lane = undefined;
    for (&lanes, 0..) |*l, i| try l.start(t.e, dev, window, .metal_first, .a, 2 * slot * i);
    defer for (&lanes) |*l| l.st.deinit();

    // no two of the stages' pinned or device plan words overlap
    const Span = struct { at: usize, len: usize };
    var spans: [8]Span = undefined;
    var n: usize = 0;
    for (&lanes) |*l| for (l.st.plans) |*b| {
        spans[n] = .{ .at = @intFromPtr(b.host.bytes.ptr), .len = b.host.bytes.len };
        spans[n + 1] = .{ .at = b.dev.base(), .len = plan.Buffer.deviceBytes(1, 32) };
        n += 2;
    };
    try T.expectEqual(@as(usize, 8), n);
    for (spans[0..n], 0..) |a, i| for (spans[i + 1 .. n]) |b| try T.expect(a.at + a.len <= b.at or b.at + b.len <= a.at);

    // a 5- and a 3-token prompt in chunks of 2, then two decode steps each, interleaved by `streams`
    var out: [2][2]u32 = undefined;
    const prompts = [2][]const u32{ &.{ 1, 2, 3, 4, 5 }, &.{ 6, 7, 8 } };
    const works = [2]split.streams.Work{ .{ .prompt = prompts[0], .out = &out[0] }, .{ .prompt = prompts[1], .out = &out[1] } };
    fake.clear();
    try split.streams.generate(&.{ &lanes[0].p, &lanes[1].p }, &works, 2);
    try T.expectEqual(@as(usize, 7), lanes[0].st.pos);
    try T.expectEqual(@as(usize, 5), lanes[1].st.pos);
    // a one-row step takes the decode round's plan, whether it is a decode step or a prompt's last chunk (the 5-token
    // prompt ends on one row at position 4, the 3-token on one at 2): the stage's three one-row steps fill buffers 0, 1, 0
    // of their own stage, and the word after the token's is the position
    const pos_word = plan.Layout.of(1, 2, 32).pos;
    try T.expectEqual([2]u32{ 6, 5 }, [2]u32{ lanes[0].st.plans[0].host.slice(u32)[pos_word], lanes[0].st.plans[1].host.slice(u32)[pos_word] });
    try T.expectEqual([2]u32{ 4, 3 }, [2]u32{ lanes[1].st.plans[0].host.slice(u32)[pos_word], lanes[1].st.plans[1].host.slice(u32)[pos_word] });
    // a stamped row is consumed once, whichever step carried it: 5 + 3 prompt positions and 2 + 2 decode steps
    var consumes: usize = 0;
    for (fake.boundaries()) |b| {
        try T.expect(b.kind == .consume);
        try T.expectEqual(@as(u32, 2560), b.n);
        consumes += 1;
    }
    try T.expectEqual(@as(usize, 12), consumes);
}

test "split stage: warm-up runs every row count's span and the decode step, plain and stamped, then starts over" {
    const t = try Eng.open(.{ .first = 16 }, 64, 1);
    defer t.close();
    try t.e.size(64, 0);
    var st = try stage.HipLayerStage.init(t.e, 16, .{ .max_rows = 4 });
    defer st.deinit();
    fake.clear();
    try st.warm();
    var imports: usize = 0;
    var consumes: usize = 0;
    for (fake.boundaries()) |b| switch (b.kind) {
        .import => imports += 1,
        .consume => consumes += 1,
        .export_ => return error.TestUnexpectedResult,
    };
    // stamped: the decode step's row and the spans of 2, 3 and 4 rows; plain: those four, then the spans of the smallest
    // and the largest count at the far end of the caches
    try T.expectEqual(@as(usize, 1 + 2 + 3 + 4), consumes);
    try T.expectEqual(@as(usize, 4 + 2), imports);
    try T.expectEqual(@as(usize, 0), st.pos);
    try T.expectEqual(@as(usize, 0), st.stamped_rows);
    // the stage serves a sequence from its first position again
    var rec: Recorder = .{};
    defer rec.deinit();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * 2560, .max_rows = 4 };
    const n = pipeline.Pipeline.slotBytes(cfg);
    const window = try rec.device().allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    var scratch: [4 * 4 * 2560]u8 = undefined;
    const io: pipeline.StepIo = .{ .in = .init(window.ptr, n), .out = .init(window.ptr + n, n), .seq = 1, .rows = 4, .stamped = true, .retries = 10, .scratch = &scratch, .hidden_bytes = 4 * 2560 };
    try st.record(io);
    try T.expectEqual(@as(usize, 4), st.pos);
}
