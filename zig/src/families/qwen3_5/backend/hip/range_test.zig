//! The engine opened over a layer range, on the recording runtime (split_fake.zig): allocation sizes and the refusals.

const std = @import("std");
const hip = @import("hip");
const view = @import("model/view.zig");
const bridge = @import("model/bridge.zig");
const state = @import("forward/state.zig");
const pages = @import("forward/pages.zig");
const round_graphs = @import("engine/round_graphs.zig");
const Engine = @import("engine/engine.zig").Engine;
const hip_lanes = @import("engine/hip_lanes.zig");
const fake = @import("split_fake.zig");

const T = std.testing;
const gpa = T.allocator;

/// An engine over the 4B's shape and a layer range, sized on the stand-in runtime; the weights are fake addresses.
pub const Eng = struct {
    e: *Engine,

    pub fn open(range: view.Range, capacity: usize, rows: usize) !Eng {
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
    pub fn close(t: Eng) void {
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

test "layer range: an engine over half the layers sizes to what its range holds" {
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

test "layer range: a ranged engine's caches hold state for its layers only" {
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

test "layer range: a range is refused where it cannot run" {
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
