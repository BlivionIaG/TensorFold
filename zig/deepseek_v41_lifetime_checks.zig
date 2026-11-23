//! Native backend allocation lifetime without adding engine imports to source-only operator checks.
const std = @import("std");
const mtl = @import("metal");
const Model = @import("tensorfold").deepseek_v41.Model;
const Backend = @import("tensorfold").deepseek_v41_backend.Backend;

/// The rounds the depth policy has priced: one plain one-row round and the three drafted windows of a machine.
fn measured(backend: *Backend) void {
    backend.mtp.noteRound(1, 19.0);
    backend.mtp.noteRound(2, 27.0);
    backend.mtp.noteRound(3, 37.0);
    backend.mtp.noteRound(4, 41.0);
    backend.mtp.noteDraft(4.0);
}

test "V4.1 backend releases all per-slot host state on unload" {
    var model: Model = undefined;
    model.lanes = 4;
    var backend = try Backend.init(std.testing.allocator, &model);
    backend.deinit();
}

test "drafter ID growth preserves pending Markov picks" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const picks = try device.buffer(8 * @sizeOf(u32), mtl.ResourceOptions.shared);
    defer picks.deinit();
    const expected = [_]u32{ 11, 23, 37, 41, 53, 67, 79, 83 };
    @memcpy(picks.slice(u32, expected.len), &expected);
    var model: Model = undefined;
    model.device = device;
    model.residency = try device.residencySet(4);
    model.residency.add(picks);
    model.drafter_ids = null;
    model.markov_ids = picks;
    defer {
        model.residency.deinit();
        if (model.drafter_ids) |ids| ids.deinit();
        if (model.markov_ids) |ids| {
            if (ids.gpuAddress() != picks.gpuAddress()) ids.deinit();
        }
    }
    _ = try model.drafterId(97);
    try std.testing.expectEqualSlices(u32, &expected, model.markov_ids.?.slice(u32, expected.len));
    for ([_]u32{ 5, 17 }) |rows| {
        const ids = try model.drafterRound(97, rows);
        try ids.requireRows(rows, 1, 1, @sizeOf(u32));
        try std.testing.expectEqualSlices(u32, &expected, model.markov_ids.?.slice(u32, expected.len));
    }
}

test "the window's depth follows the rounds' own ms and landed drafts" {
    var model: Model = undefined;
    model.lanes = 1;
    model.mtp_enabled = true;
    // Only bound-weight presence is read; this host policy performs no GPU operation.
    model.weights.dspark = @as(@TypeOf(model.weights.dspark.?), undefined);
    model.caches = &.{};
    var backend = try Backend.init(std.testing.allocator, &model);
    defer backend.deinit();
    const offer: Backend.Offer = .{ .src = .trained, .have = 3 };
    // No round measured yet: the first window is the plain one that prices a plain round.
    try std.testing.expectEqual(@as(u32, 0), Backend.depthFor(&backend, 0, offer));
    measured(&backend);
    backend.depth_now[0] = std.math.maxInt(u32); // the keep that ends the round
    // Three drafts at 0.75 a draft pay the four-row window, and the round's other hook reads the same choice.
    try std.testing.expectEqual(@as(u32, 3), Backend.depthFor(&backend, 0, offer));
    try std.testing.expectEqual(@as(u32, 1), Backend.depthFor(&backend, 0, .{ .src = .lookup, .have = 1 }));
    // Prose: one draft in ten lands, so the plain round is the cheaper token.
    backend.depth_now[0] = std.math.maxInt(u32);
    for (&backend.mtp.q[0]) |*c| c.* = .{ .v = 0.1, .n = 8 };
    try std.testing.expectEqual(@as(u32, 0), Backend.depthFor(&backend, 0, offer));
}

test "suffix drafts follow committed context" {
    const gpa = std.testing.allocator;
    var model: Model = undefined;
    model.lanes = 1;
    model.mtp_enabled = false;
    model.weights.dspark = null;
    model.caches = &.{};
    var backend = try Backend.init(gpa, &model);
    defer backend.deinit();
    measured(&backend);
    var stream = try @import("lanes").Stream.init(gpa, .{
        .id = "committed",
        .prompt = &.{ 1, 2, 3, 4, 5, 9 },
        .max_new = 32,
    });
    defer stream.deinit(gpa);
    _ = try stream.commit(gpa, &.{ 1, 2, 3 }, &.{});
    stream.pending = 3;
    try backend.by.put(gpa, &stream, 0);
    try Backend.draft(&backend, &.{.{
        .stream = &stream,
        .follow = &.{},
        .rows = &.{},
        .start = 6,
        .position = stream.context.items.len,
        .depth = 3,
    }});
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, backend.held[0]);
}

test "lookup proposals take precedence over bound trained drafts" {
    const gpa = std.testing.allocator;
    var model: Model = undefined;
    model.lanes = 1;
    model.mtp_enabled = true;
    model.weights.dspark = @as(@TypeOf(model.weights.dspark.?), undefined);
    model.caches = &.{};
    var backend = try Backend.init(gpa, &model);
    defer backend.deinit();
    measured(&backend);
    var stream = try @import("lanes").Stream.init(gpa, .{
        .id = "lookup-preferred",
        .prompt = &.{ 1, 2, 3, 4, 5, 9 },
        .max_new = 32,
    });
    defer stream.deinit(gpa);
    _ = try stream.commit(gpa, &.{ 1, 2, 3 }, &.{});
    stream.pending = 3;
    try backend.by.put(gpa, &stream, 0);
    backend.mtp_cache[0] = .{ .pending = 3, .drafts = .{ 90, 91, 92, 93 }, .count = 3 };
    const held = (try Backend.tree(&backend, &stream, gpa)) orelse return error.MissingLookupProposal;
    const tokens = held.tokens orelse return error.MissingLookupTokens;
    defer gpa.free(tokens);
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, tokens);
    try Backend.draft(&backend, &.{.{
        .stream = &stream,
        .follow = &.{},
        .rows = &.{},
        .start = 6,
        .position = stream.context.items.len,
        .depth = 3,
    }});
    try std.testing.expectEqualSlices(u32, &.{ 4, 5, 9 }, backend.held[0]);
}
