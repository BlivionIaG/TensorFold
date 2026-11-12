//! A stage serving several sequence slots through the real `Pipeline` on the recording runtime (split_fake.zig): a decode
//! batch is one round over rows of different caches, each row's input its own consume launch, positions advance for the
//! named slots only, and several lanes over one engine keep to their own plan buffers. Nothing is computed.

const std = @import("std");
const split = @import("split");
const view = @import("model/view.zig");
const plan = @import("forward/plan.zig");
const fake = @import("split_fake.zig");
const base = @import("split_test.zig");
const sst = @import("split_stage_test.zig");
const stage = @import("split_stage.zig");

const T = std.testing;
const pipeline = split.pipeline;
const wire = split.wire;

const Eng = sst.Eng;
const Lane = sst.Lane;

const hidden: u32 = 2560;

/// The words a decode round of `n` windows wrote into pinned plan words `w`: row i is window i, one token at the slot's
/// position, its caches' descriptor.
fn expectWindows(w: []const u32, n: usize, slots: []const u8, positions: []const usize, lane: *const Lane) !void {
    const l = plan.Layout.of(n, n + 1, 32);
    for (0..n) |i| {
        try T.expectEqual(@as(u32, @intCast(positions[slots[i]])), w[l.pos + i]);
        try T.expectEqual(@as(u32, @intCast(i)), w[l.first + i]);
        try T.expectEqual(@as(u32, 1), w[l.count + i]);
        try T.expectEqual(@as(u32, @intCast(i)), w[l.slot + i]);
        const desc = @as(u64, w[l.desc + 2 * i]) | @as(u64, w[l.desc + 2 * i + 1]) << 32;
        try T.expectEqual(lane.st.seqs[slots[i]].caches.desc.base(), desc);
    }
}

/// Eight sequences each prompted `slot + 1` positions (so every one is at its own position), then a batch of `n` of
/// them in reverse slot order, then a batch of the even slots only: one round each, a consume launch a row from the
/// window row of the same index, tokens for rows, and only the named slots advance.
fn batch(order: pipeline.Order, level: pipeline.Level, range: view.Range, n: usize) !void {
    const t = try Eng.open(range, 64, 8);
    defer t.close();
    try t.e.size(64, 0);
    var rec: sst.Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = 4 * hidden, .max_rows = 8 };
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    var lane: Lane = undefined;
    try lane.start(t.e, dev, window, order, level, 0, .{ .max_rows = 8, .slots = 8 });
    defer lane.st.deinit();
    const m = t.e.model();
    const stamped = order == .metal_first;

    const refilled = fake.refills();
    const tokens = [8]u32{ 11, 12, 13, 14, 15, 16, 17, 18 };
    var positions: [8]usize = undefined;
    for (0..8) |i| {
        try lane.p.submitSlots(tokens[0 .. i + 1], &.{@intCast(i)});
        _ = try lane.p.collect();
        positions[i] = i + 1;
        try T.expectEqual(i + 1, lane.st.position(i));
    }

    var rig = try base.Rig.init();
    defer rig.deinit();
    var marks: base.Marks = .{};
    const want = try base.decodeRows(&rig, m, base.handoff(m, stamped), n, &marks);
    defer T.allocator.free(want);

    var slots: [8]u8 = undefined;
    for (slots[0..n], 0..) |*s, i| s.* = @intCast(n - 1 - i);
    fake.clear();
    const turn = lane.st.turn;
    try lane.p.submitSlots(tokens[0..n], slots[0..n]);
    var out: [8]u32 = undefined;
    try lane.p.collectAll(out[0..n]);
    // one round: the decode launches of `n` rows, their hand-offs per row
    try base.expectSeq(want, fake.launches());
    try expectWindows(lane.st.plans[turn].host.slice(u32), n, slots[0..n], &positions, &lane);

    const got = fake.boundaries();
    const in_at = @intFromPtr(lane.p.down.data().ptr);
    var at: usize = 0;
    if (!m.range.embeds()) {
        for (0..n) |r| {
            const c = got[at];
            try T.expect(c.kind == .consume);
            try T.expectEqual(in_at + r * wire.stampedLen(4 * hidden), c.window);
            try T.expectEqual(hidden, c.n);
            try T.expectEqual(got[0].status + 4 * r, c.status);
            try T.expectEqual(got[0].device + r * hidden * 2, c.device);
            at += 1;
        }
    }
    if (!m.headed()) {
        try T.expect(got[at].kind == .export_);
        try T.expectEqual(@as(u32, @intCast(n)) * hidden, got[at].n);
        at += 1;
    }
    try T.expectEqual(at, got.len);

    // only the named slots advanced
    for (0..8) |i| {
        const named = i < n;
        try T.expectEqual(i + 1 + @intFromBool(named), lane.st.position(i));
    }
    const first_positions = positions;
    for (slots[0..n]) |s| positions[s] += 1;

    // a sequence that skips a step keeps its position: the even slots below `n` alone go next, on the other plan buffer
    var even: [8]u8 = undefined;
    var k: usize = 0;
    for (0..n) |i| if (i % 2 == 0) {
        even[k] = @intCast(i);
        k += 1;
    };
    const before: [8]usize = .{ lane.st.position(0), lane.st.position(1), lane.st.position(2), lane.st.position(3), lane.st.position(4), lane.st.position(5), lane.st.position(6), lane.st.position(7) };
    const turn2 = lane.st.turn;
    try T.expect(turn2 != turn);
    fake.clear();
    try lane.p.submitSlots(tokens[0..k], even[0..k]);
    try lane.p.collectAll(out[0..k]);
    for (0..8) |i| {
        const named = std.mem.indexOfScalar(u8, even[0..k], @intCast(i)) != null;
        try T.expectEqual(before[i] + @intFromBool(named), lane.st.position(i));
    }
    try T.expectEqual(refilled, fake.refills());
    // the first batch's words are still where it left them: the second filled the other buffer
    try expectWindows(lane.st.plans[turn].host.slice(u32), n, slots[0..n], &first_positions, &lane);
}

test "split stage: a decode batch is one round over rows of different caches, in every order and level" {
    for ([_]usize{ 1, 2, 3, 8 }) |n| {
        try batch(.external_first, .a, .{ .first = 0, .last = 16 }, n);
        try batch(.metal_first, .a, .{ .first = 16 }, n);
        try batch(.metal_first, .b, .{ .first = 8, .last = 24 }, n);
        try batch(.external_only, .a, .whole, n);
    }
}

test "split stage: a batch is refused when its slots do not fit the stage" {
    const t = try Eng.open(.{ .first = 16 }, 64, 4);
    defer t.close();
    try t.e.size(64, 0);
    // more slots than the engine's rows, or than a lane has, are refused when the stage opens
    try T.expectError(error.TooManySlots, stage.HipLayerStage.init(t.e, 8, .{ .max_rows = 8, .slots = 5 }));
    try T.expectError(error.TooManySlots, stage.HipLayerStage.init(t.e, 8, .{ .slots = 0 }));
    var st = try stage.HipLayerStage.init(t.e, 8, .{ .max_rows = 4, .slots = 3 });
    defer st.deinit();
    var rec: sst.Recorder = .{};
    defer rec.deinit();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * hidden, .max_rows = 4 };
    const n = pipeline.Pipeline.slotBytes(cfg);
    const window = try rec.device().allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    var scratch: [4 * 4 * 2560]u8 = undefined;
    var io: pipeline.StepIo = .{ .in = .init(window.ptr, n), .out = .init(window.ptr + n, n), .seq = 1, .rows = 2, .slots = &.{ 0, 2 }, .stamped = true, .retries = 10, .scratch = &scratch, .hidden_bytes = 4 * hidden };
    try st.record(io);
    // a slot the stage does not have, one named twice, a batch whose rows and slots differ
    io.slots = &.{ 0, 3 };
    try T.expectError(error.BadStep, st.record(io));
    io.slots = &.{ 1, 1 };
    try T.expectError(error.BadStep, st.record(io));
    io.rows = 3;
    io.slots = &.{ 0, 1 };
    try T.expectError(error.BadStep, st.record(io));
    // a position past the caches' end is refused by name, and no slot advanced
    io.rows = 1;
    io.slots = &.{1};
    for (0..8) |_| try st.record(io);
    try T.expectError(error.ContextFull, st.record(io));
    try T.expectEqual(@as(usize, 8), st.position(1));
    try T.expectEqual(@as(usize, 1), st.position(0));
    try st.resetSlot(1);
    try T.expectEqual(@as(usize, 0), st.position(1));
    try T.expectEqual(@as(usize, 1), st.position(0));
}

test "split stage: warm-up runs a decode batch of every size up to the slots" {
    const t = try Eng.open(.{ .first = 16 }, 64, 4);
    defer t.close();
    try t.e.size(64, 0);
    var one = try stage.HipLayerStage.init(t.e, 16, .{ .max_rows = 4 });
    defer one.deinit();
    fake.clear();
    const refilled = fake.refills();
    try one.warm();
    const single = fake.launches().len;
    var four = try stage.HipLayerStage.init(t.e, 16, .{ .max_rows = 4, .slots = 4 });
    defer four.deinit();
    fake.clear();
    try four.warm();
    // no step's plan words were refilled before the stream's copy of the step before had run
    try T.expectEqual(refilled, fake.refills());
    // the batches of 2, 3 and 4 rows, plain and stamped, and the widest at the far end add rounds of their own
    try T.expect(fake.launches().len > single);
    var consumes: usize = 0;
    for (fake.boundaries()) |b| consumes += @intFromBool(b.kind == .consume);
    // stamped: the decode row, the batches (2 + 3 + 4) and the spans of 2 to 4 rows (2 + 3 + 4)
    try T.expectEqual(@as(usize, 1 + 9 + 9), consumes);
    for (0..4) |slot| try T.expectEqual(@as(usize, 0), four.position(slot));
}

test "split stage: lanes of several sequences over one engine keep their own plan buffers" {
    const t = try Eng.open(.{ .first = 16 }, 64, 4);
    defer t.close();
    try t.e.size(64, 0);
    var rec: sst.Recorder = .{};
    defer rec.deinit();
    const dev = rec.device();
    const cfg: pipeline.Config = .{ .order = .metal_first, .hidden_bytes = 4 * hidden, .max_rows = 4 };
    const slot_bytes = pipeline.Pipeline.slotBytes(cfg);
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 2));
    var lanes: [2]Lane = undefined;
    for (&lanes, 0..) |*l, i| try l.start(t.e, dev, window, .metal_first, .a, 2 * slot_bytes * i, .{ .max_rows = 4, .slots = 3 });
    defer for (&lanes) |*l| l.st.deinit();

    var outs: [5][2]u32 = undefined;
    const prompts = [5][]const u32{ &.{ 1, 2, 3 }, &.{ 4, 5 }, &.{ 6, 7, 8, 9 }, &.{ 10, 11, 12 }, &.{13} };
    var works: [5]split.streams.Work = undefined;
    for (&works, prompts, &outs) |*w, p, *o| w.* = .{ .prompt = p, .out = o };
    fake.clear();
    const refilled = fake.refills();
    try split.streams.generateLanes(&.{
        .{ .pipe = &lanes[0].p, .works = works[0..3] },
        .{ .pipe = &lanes[1].p, .works = works[3..5] },
    }, 2);
    // a lane's steps took their plan buffers in turn, each refilled only after a step collected since its last copy
    try T.expectEqual(refilled, fake.refills());
    // every sequence ran its prompt and its two decode steps in its own slot
    const lens = [5]usize{ 3, 2, 4, 3, 1 };
    for (lens[0..3], 0..) |len, slot| try T.expectEqual(len + 2, lanes[0].st.position(slot));
    for (lens[3..5], 0..) |len, slot| try T.expectEqual(len + 2, lanes[1].st.position(slot));
    // each stamped row is consumed once: every prompt position and every batch row
    var consumes: usize = 0;
    for (fake.boundaries()) |b| consumes += @intFromBool(b.kind == .consume);
    try T.expectEqual(@as(usize, 3 + 2 + 4 + 3 + 1 + 2 * 3 + 2 * 2), consumes);
    // no two plan buffers of the two stages overlap, pinned or device
    const Span = struct { at: usize, len: usize };
    var spans: [8]Span = undefined;
    var k: usize = 0;
    for (&lanes) |*l| for (l.st.plans) |*b| {
        spans[k] = .{ .at = @intFromPtr(b.host.bytes.ptr), .len = b.host.bytes.len };
        spans[k + 1] = .{ .at = b.dev.base(), .len = plan.Buffer.deviceBytes(4, 32) };
        k += 2;
    };
    for (spans[0..k], 0..) |a, i| for (spans[i + 1 .. k]) |b| try T.expect(a.at + a.len <= b.at or b.at + b.len <= a.at);
}
