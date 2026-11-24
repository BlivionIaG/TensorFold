//! Real-checkpoint LoopCD: guided rows equal across chunk seams, shared forwards, restores, drafts and serving.
const std = @import("std");
const mtl = @import("metal");
const lanes = @import("lanes");
const api = @import("engine_api");
const native = @import("inkling2.zig");
const serving = @import("inkling2_serving_test.zig");
const Model = @import("../families/inkling2/metal/model.zig").Model;
const backend = @import("../families/inkling2/metal/backend.zig");
const fwd = @import("../families/inkling2/metal/forward.zig");
const st = @import("../families/inkling2/metal/state.zig");
const snapshot = @import("../families/inkling2/metal/snapshot.zig");

const gpa = std.testing.allocator;
const io = std.testing.io;
const strengths = [_]f32{ -1, 0.3, 1 };
const length = 150;
const capacity = 256;

/// Every row's logits of `ids` fed after `cache`'s rows: a first piece of `first` rows, then pieces of `step`.
fn rows(m: *const Model, s: *st.Scratch, cache: *st.Cache, ids: []const u32, first: usize, step: usize, out: []f32) !void {
    const v = m.config.text.unpadded_vocab;
    var at: usize = 0;
    while (at < ids.len) {
        const n = @min(if (at == 0) first else step, ids.len - at);
        try fwd.run(m, s, cache, ids[at..][0..n], .{ .all = true });
        @memcpy(out[at * v ..][0 .. n * v], s.logits_all.slice(f32, n * v));
        cache.len += n;
        at += n;
    }
}

/// The rows of `ids` from an empty cache.
fn fresh(m: *const Model, s: *st.Scratch, ids: []const u32, first: usize, step: usize, out: []f32) !void {
    var cache = try st.Cache.init(m.device, capacity, &m.weight_plan);
    defer cache.deinit();
    try rows(m, s, &cache, ids, first, step, out);
}

fn expectRows(want: []const f32, got: []const f32) !void {
    try std.testing.expectEqualSlices(u32, std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(want)), std.mem.bytesAsSlice(u32, std.mem.sliceAsBytes(got)));
}

test "LoopCD guided rows equal across chunk seams, two-stream forwards and restored states at every strength" {
    const dir = try serving.checkpoint();
    const m = try Model.load(gpa, io, dir);
    defer m.deinit();
    var s = try st.Scratch.init(m.device, st.max_rows);
    defer s.deinit();
    const v = m.config.text.unpadded_vocab;
    const ids = serving.prompt(length, 5);
    const other = serving.prompt(length, 6);
    const plain = try gpa.alloc(f32, length * v);
    defer gpa.free(plain);
    try fresh(m, &s, &ids, st.prompt_rows, st.prompt_rows, plain);
    const whole = try gpa.alloc(f32, length * v);
    defer gpa.free(whole);
    const got = try gpa.alloc(f32, length * v);
    defer gpa.free(got);
    const solo = try gpa.alloc(f32, length * v);
    defer gpa.free(solo);
    for (strengths) |w| {
        m.loopcd = w;
        defer m.loopcd = null;
        try fresh(m, &s, &ids, st.prompt_rows, st.prompt_rows, whole);
        // The guidance is live: every strength moves the rows off the unguided ones.
        try std.testing.expect(!std.mem.eql(u8, std.mem.sliceAsBytes(plain), std.mem.sliceAsBytes(whole)));
        // Pieces of 1, 7, 15, 31 and 63 rows, and the 64-row seam moved to 1, 63 and 65.
        const cuts = [_][2]usize{ .{ 1, 1 }, .{ 7, 7 }, .{ 15, 15 }, .{ 31, 31 }, .{ 63, 63 }, .{ 1, 64 }, .{ 63, 64 }, .{ 64, 1 } };
        for (cuts) |cut| {
            try fresh(m, &s, &ids, cut[0], cut[1], got);
            try expectRows(whole, got);
        }
        // Two streams in one forward, 16 rows each after their own prefixes, equal their solo rows.
        try fresh(m, &s, &other, st.prompt_rows, st.prompt_rows, solo);
        var a = try st.Cache.init(m.device, capacity, &m.weight_plan);
        defer a.deinit();
        var b = try st.Cache.init(m.device, capacity, &m.weight_plan);
        defer b.deinit();
        try rows(m, &s, &a, ids[0..40], 40, 40, got);
        try rows(m, &s, &b, other[0..57], 57, 57, got);
        var pair: [2 * st.window_rows]u32 = undefined;
        @memcpy(pair[0..st.window_rows], ids[40..][0..st.window_rows]);
        @memcpy(pair[st.window_rows..], other[57..][0..st.window_rows]);
        try fwd.runShared(m, &s, &.{ .{ .cache = &a, .rows = st.window_rows }, .{ .cache = &b, .rows = st.window_rows } }, &pair, .{ .all = true });
        const shared = s.logits_all.slice(f32, 2 * st.window_rows * v);
        try expectRows(whole[40 * v ..][0 .. st.window_rows * v], shared[0 .. st.window_rows * v]);
        try expectRows(solo[57 * v ..][0 .. st.window_rows * v], shared[st.window_rows * v ..]);
        // A state kept at the chunk grid and restored into an empty cache continues to the same rows.
        var kept = try st.Cache.init(m.device, capacity, &m.weight_plan);
        defer kept.deinit();
        try rows(m, &s, &kept, ids[0..st.prompt_rows], st.prompt_rows, st.prompt_rows, got);
        var snap = try snapshot.save(m.device, m.queue, &kept);
        defer snap.deinit();
        var restored = try st.Cache.init(m.device, capacity, &m.weight_plan);
        defer restored.deinit();
        try snapshot.restore(&restored, m.queue, &snap);
        try rows(m, &s, &restored, ids[st.prompt_rows..], st.prompt_rows, st.prompt_rows, got);
        try expectRows(whole[st.prompt_rows * v ..], got[0 .. (length - st.prompt_rows) * v]);
    }
}

fn opened(dir: []const u8, drafter: ?[]const u8, w: f32) !api.Opened {
    var why: []const u8 = "";
    const result = try native.open(gpa, gpa, io, .{
        .dir = dir,
        .model_type = "inkling2_loop",
        .context = serving.context,
        .lanes = 4,
        .drafts = drafter != null,
        .drafter = drafter,
        .prompt_cache_gib = 1,
        .decode_share = 0.5,
        .loopcd = true,
        .loopcd_w = w,
    }, &why);
    if (result == null) {
        std.debug.print("Mini Metal LoopCD open refused: {s}\n", .{why});
        return error.MiniMetalOpenRefused;
    }
    return result.?;
}

/// The serial guided reference: one stream, prefill then one token a step, on a model guided at `w`.
fn reference(dir: []const u8, w: f32, ids: []const u32) !serving.Reference {
    const model = try Model.load(gpa, io, dir);
    defer model.deinit();
    model.loopcd = w;
    const target = try backend.Metal.init(gpa, model, .{ .capacity = serving.context, .streams = 1, .budget = 2 * st.cacheBytes(serving.context) });
    defer target.deinit();
    try target.warm();
    const b = target.backend();
    try b.vtable.prepare_features.?(b.ptr, &serving.taps);
    return serving.serial(b, ids);
}

test "LoopCD guided serving equals its serial reference concurrent, drafted and restored from the prompt cache" {
    const dir = try serving.checkpoint();
    const drafter = @import("sys").getenv("TF_INKLING2_DSPARK_DIR");
    const ids = serving.prompt(serving.history, 7);
    var turn: [serving.history + 16]u32 = undefined;
    @memcpy(turn[0..serving.history], &ids);
    @memcpy(turn[serving.history..], &serving.prompt(16, 8));
    for ([_]f32{ 0.3, 1 }) |w| {
        const want = try reference(dir, w, &turn);
        const engine = try opened(dir, drafter, w);
        defer engine.close(engine.ctx);
        var first = try serving.serve(engine, 1, &.{ .prompt = &turn, .max_tokens = serving.made, .drafts = false, .logprobs = 3, .history_len = ids.len });
        defer first.deinit();
        try std.testing.expectEqual(@as(?u32, 0), first.cached);
        try serving.expectTokens(&want, &first);
        var restored = try serving.serve(engine, 2, &.{ .prompt = &turn, .max_tokens = serving.made, .drafts = false, .logprobs = 3, .history_len = ids.len });
        defer restored.deinit();
        try std.testing.expectEqual(@as(?u32, ids.len), restored.cached);
        try serving.expectTokens(&want, &restored);
        if (drafter != null) {
            var drafted = try serving.serve(engine, 3, &.{ .prompt = &turn, .max_tokens = serving.made, .drafts = true, .logprobs = 3, .history_len = ids.len });
            defer drafted.deinit();
            try serving.expectTokens(&want, &drafted);
            try std.testing.expect(drafted.accepted > 0);
        }
        // Four requests at once: each reply equals the serial one.
        var replies: [4]serving.Reply = @splat(.{});
        defer for (&replies) |*r| r.deinit();
        for (&replies, 0..) |*r, i| try engine.engine.submit(10 + i, &.{ .prompt = &turn, .max_tokens = serving.made, .drafts = drafter != null and i % 2 == 1, .logprobs = 3, .history_len = 0 }, .{ .ctx = r, .event = serving.Reply.event });
        for (&replies) |*r| {
            try std.testing.expectEqual(api.Reason.length, try r.wait());
            try serving.expectTokens(&want, r);
        }
    }
}

test {
    _ = @import("inkling2_loopcd_probe_test.zig");
}
