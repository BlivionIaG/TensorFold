//! The toy model over several sequences on one mock device: each sequence's tokens, overlap, failure and batches.
const std = @import("std");
const pipeline = @import("pipeline.zig");
const device = @import("device.zig");
const streams = @import("streams.zig");
const toy = @import("toy.zig");

const Params = toy.Params;
const Stage = toy.Stage;
const Rig = toy.Rig;
const reference = toy.reference;
const layers = toy.layers;
const vocab = toy.vocab;

test "several sequences share a device and each gets its own reference's tokens" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompts = [3][]const u32{ &.{ 5, 17, 3, 90 }, &.{ 8, 8, 61, 2, 9, 40, 11 }, &.{71} };
    var want: [3][14]u32 = undefined;
    for (prompts, &want) |pr, *w| try reference(params, pr, w);
    for ([_]pipeline.Order{ .external_first, .metal_first }) |order| {
        for ([_]pipeline.Level{ .a, .b }) |level| {
            if (order == .external_first and level == .b) continue;
            for ([_]usize{ 1, 4 }) |chunk| {
                const r = try Rig.create(std.testing.allocator, order, level, level == .b, 6, 3, 0);
                defer r.destroy();
                var got: [3][14]u32 = undefined;
                var works: [3]streams.Work = undefined;
                var pipes: [3]*pipeline.Pipeline = undefined;
                for (0..3) |i| {
                    works[i] = .{ .prompt = prompts[i], .out = &got[i] };
                    pipes[i] = &r.pipes[i];
                }
                try streams.generate(&pipes, &works, chunk);
                for (want, got) |w, g| try std.testing.expectEqualSlices(u32, &w, &g);
            }
        }
    }
}

test "while the host works on one sequence the device works on another" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 5, 17 };
    var want: [8]u32 = undefined;
    try reference(params, &prompt, &want);
    // each side takes 3 ms a step: one sequence at a time costs both, two sequences cost about the slower one each
    const delay: u64 = 3_000_000;
    var elapsed: [2]u64 = undefined;
    for ([_]usize{ 1, 2 }, &elapsed) |n, *e| {
        const r = try Rig.create(std.testing.allocator, .metal_first, .a, false, 6, n, delay);
        defer r.destroy();
        var got: [2][8]u32 = undefined;
        var works: [2]streams.Work = undefined;
        var pipes: [2]*pipeline.Pipeline = undefined;
        for (0..n) |i| {
            works[i] = .{ .prompt = &prompt, .out = &got[i] };
            pipes[i] = &r.pipes[i];
        }
        const t0 = device.nowNs();
        try streams.generate(pipes[0..n], works[0..n], 1);
        e.* = device.nowNs() - t0;
        for (got[0..n]) |g| try std.testing.expectEqualSlices(u32, &want, &g);
    }
    // two sequences in less than 1.8 times one: serial would be twice (the margin is for a loaded machine)
    try std.testing.expect(elapsed[1] * 5 < elapsed[0] * 9);
}

test "a failed step releases every pipeline on the device, and after a reset the sequences run again" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompts = [2][]const u32{ &.{ 5, 17, 3 }, &.{ 8, 61 } };
    var want: [2][6]u32 = undefined;
    for (prompts, &want) |pr, *w| try reference(params, pr, w);
    const r = try Rig.create(std.testing.allocator, .metal_first, .a, false, 6, 2, 0);
    defer r.destroy();
    var got: [2][6]u32 = undefined;
    var works = [2]streams.Work{ .{ .prompt = &.{ 5, vocab }, .out = &got[0] }, .{ .prompt = prompts[1], .out = &got[1] } };
    const pipes = [2]*pipeline.Pipeline{ &r.pipes[0], &r.pipes[1] };
    try std.testing.expectError(error.PeerLost, streams.generate(&pipes, &works, 1));
    for (pipes) |p| p.reset();
    // the layers' state belongs to the stages: a sequence that starts over starts from zero
    for (r.ext[0..2], r.host[0..2]) |*e, *h| {
        e.state = @splat(@splat(@splat(0)));
        h.state = @splat(@splat(@splat(0)));
    }
    works[0].prompt = prompts[0];
    try streams.generate(&pipes, &works, 1);
    for (want, got) |w, g| try std.testing.expectEqualSlices(u32, &w, &g);
}

test "decode batches: sequences of different lengths in lanes of up to eight each get their own reference's tokens" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    // eight prompts of different lengths and amounts to generate: the batches shrink as sequences finish
    const prompts = [8][]const u32{ &.{ 5, 17, 3, 90 }, &.{ 8, 8, 61, 2, 9 }, &.{71}, &.{ 12, 44 }, &.{ 33, 19, 7 }, &.{ 60, 1, 2, 3, 4, 5 }, &.{ 40, 41 }, &.{ 90, 91, 92 } };
    const lens = [8]usize{ 9, 14, 3, 12, 1, 7, 11, 5 };
    var want: [8][14]u32 = undefined;
    for (prompts, lens, &want) |pr, n, *w| try reference(params, pr, w[0..n]);
    // lane layouts: one lane of eight, two of four, three of (3, 3, 2); every order and level
    const layouts = [_][]const usize{ &.{8}, &.{ 4, 4 }, &.{ 3, 3, 2 } };
    for (layouts) |layout| {
        for ([_]pipeline.Order{ .external_first, .metal_first, .external_only }) |order| {
            for ([_]pipeline.Level{ .a, .b }) |level| {
                if (order != .metal_first and level == .b) continue;
                const r = try Rig.create(std.testing.allocator, order, level, level == .b, if (order == .external_only) layers else 6, layout.len, 0);
                defer r.destroy();
                var got: [8][14]u32 = undefined;
                var works: [8]streams.Work = undefined;
                for (&works, prompts, lens, &got) |*w, pr, n, *g| w.* = .{ .prompt = pr, .out = g[0..n] };
                var lanes: [3]streams.Lane = undefined;
                var at: usize = 0;
                for (layout, 0..) |n, i| {
                    lanes[i] = .{ .pipe = &r.pipes[i], .works = works[at..][0..n] };
                    at += n;
                }
                try streams.generateLanes(lanes[0..layout.len], 3);
                for (lens, want, got) |n, w, g| try std.testing.expectEqualSlices(u32, w[0..n], g[0..n]);
            }
        }
    }
}

test "a decode batch of one row of each sequence leaves the tokens and the state each sequence alone would" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    var batch: Stage = .{ .params = params, .first = 0, .last = layers, .embed = true, .head = true };
    var alone: Stage = .{ .params = params, .first = 0, .last = layers, .embed = true, .head = true };
    // give the slots different histories
    const history = [3][]const u32{ &.{ 5, 17, 3 }, &.{8}, &.{ 71, 2, 9, 40, 11 } };
    var out: [4 * 3]u8 = undefined;
    for (history, 0..) |h, slot| {
        var bytes: [5 * 4]u8 = undefined;
        for (h, 0..) |t, i| std.mem.writeInt(u32, bytes[4 * i ..][0..4], t, .little);
        _ = try batch.runSlots(&.{@intCast(slot)}, h.len, bytes[0 .. 4 * h.len], &out);
        _ = try alone.runSlots(&.{@intCast(slot)}, h.len, bytes[0 .. 4 * h.len], &out);
    }
    for (0..4) |round| {
        var in: [3 * 4]u8 = undefined;
        for (0..3) |i| std.mem.writeInt(u32, in[4 * i ..][0..4], @intCast(10 * round + i + 1), .little);
        // the slots in a different order than their numbers, so a row is not mistaken for its slot
        const slots = [3]u8{ 2, 0, 1 };
        try std.testing.expectEqual(@as(usize, 12), try batch.runSlots(&slots, 3, &in, &out));
        for (slots, 0..) |slot, row| {
            var one: [4]u8 = undefined;
            try std.testing.expectEqual(@as(usize, 4), try alone.runSlots(&.{slot}, 1, in[4 * row ..][0..4], &one));
            try std.testing.expectEqualSlices(u8, &one, out[4 * row ..][0..4]);
        }
    }
    try std.testing.expect(std.mem.eql(u8, std.mem.asBytes(&alone.state), std.mem.asBytes(&batch.state)));
}

test "a batch names each sequence once, and its tokens are collected a row" {
    const r = try Rig.create(std.testing.allocator, .external_only, .a, false, layers, 1, 0);
    defer r.destroy();
    const p = &r.pipes[0];
    try std.testing.expectError(error.BadStep, p.submitSlots(&.{ 1, 2 }, &.{ 3, 3 }));
    try std.testing.expectError(error.BadStep, p.submitSlots(&.{ 1, 2, 3 }, &.{ 0, 1 }));
    try std.testing.expectError(error.BadStep, p.submitSlots(&.{ 1, 2 }, &.{}));
    try p.submitSlots(&.{ 1, 2 }, &.{ 3, 5 });
    try std.testing.expectError(error.BadStep, p.collect());
    var short: [1]u32 = undefined;
    try std.testing.expectError(error.BadStep, p.collectAll(&short));
    var two: [2]u32 = undefined;
    try p.collectAll(&two);
    try std.testing.expectError(error.BadStep, p.collectAll(&two));
}
