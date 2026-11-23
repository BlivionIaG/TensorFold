//! The producer's statistics against synthetic captures: the weighting, the mask, and the file they become.

const std = @import("std");
const derive = @import("glp_derive");
const glp = @import("glp");

/// A capture row of one prompt per layer: every column of a layer holds the same value.
fn onePrompt(a: std.mem.Allocator, layers: usize, hidden: usize, values: []const f32) ![]f32 {
    const sums = try a.alloc(f32, layers * hidden);
    for (values, 0..) |v, i| @memset(sums[i * hidden ..][0..hidden], v);
    return sums;
}

/// The layers a GLP file holds from a derived set: layer 0 is not representable, so it is left out.
fn fileLayers(a: std.mem.Allocator, d: derive.Derived) ![]glp.Layer {
    var out: std.ArrayList(glp.Layer) = .empty;
    errdefer out.deinit(a);
    for (d.layers) |l| if (l.id != 0) try out.append(a, .{ .id = l.id, .dir = l.dir });
    return out.toOwnedSlice(a);
}

test "every prompt counts once in a set's mean, not once per token" {
    const a = std.testing.allocator;
    var set = try derive.Mean.init(a, 1, 2);
    defer set.deinit(a);
    // A prompt of two rows of 1.0, then a prompt of a hundred rows of 3.0.
    try set.addPrompt(&.{ 2, 2 }, 2);
    try set.addPrompt(&.{ 300, 300 }, 100);
    var mean = [_]f32{ 0, 0 };
    set.mean(0, &mean);
    // The two prompts weigh the same: (1 + 3) / 2, not (2 * 1 + 100 * 3) / 102.
    try std.testing.expectApproxEqAbs(@as(f32, 2), mean[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 2), mean[1], 1e-6);
    // The peak is the largest per-prompt mean, and it is the mask screen's input.
    try std.testing.expectApproxEqAbs(@as(f32, 3), set.peak[0], 1e-6);
}

test "the direction is the difference of the two sets' means, one unit long" {
    const a = std.testing.allocator;
    var harmful = try derive.Mean.init(a, 2, 2);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 2, 2);
    defer harmless.deinit(a);
    // Layer 0 differs by (2, 0); layer 1 by (0, 4).
    try harmful.addPrompt(&.{ 3, 0, 0, 4 }, 1);
    try harmless.addPrompt(&.{ 1, 0, 0, 0 }, 1);
    var derived = try derive.derive(a, harmful, harmless, 2, 2, 0);
    defer derived.deinit(a);
    try std.testing.expectEqual(@as(usize, 2), derived.layers.len);
    try std.testing.expectEqual(@as(u32, 0), derived.layers[0].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1), derived.layers[0].dir[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), derived.layers[0].dir[1], 1e-6);
    try std.testing.expectEqual(@as(u32, 1), derived.layers[1].id);
    try std.testing.expectApproxEqAbs(@as(f32, 1), derived.layers[1].dir[1], 1e-6);
    for (derived.layers) |l| {
        var norm: f32 = 0;
        for (l.dir) |v| norm += v * v;
        try std.testing.expectApproxEqAbs(@as(f32, 1), @sqrt(norm), 1e-5);
    }
}

test "a layer whose two means agree is dropped, and an empty set is refused" {
    const a = std.testing.allocator;
    var harmful = try derive.Mean.init(a, 3, 2);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 3, 2);
    defer harmless.deinit(a);
    try harmful.addPrompt(&.{ 1, 1, 5, 5, 7, 7 }, 1);
    try harmless.addPrompt(&.{ 1, 1, 2, 2, 7, 7 }, 1);
    var derived = try derive.derive(a, harmful, harmless, 3, 2, 0);
    defer derived.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), derived.layers.len);
    try std.testing.expectEqual(@as(u32, 1), derived.layers[0].id);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2 }, derived.dropped);
    var empty = try derive.Mean.init(a, 1, 2);
    defer empty.deinit(a);
    try std.testing.expectError(error.NoPrompts, derive.derive(a, harmful, empty, 3, 2, 0));
}

test "the mask leaves out the dims the activations are largest in, and renormalises the rest" {
    const a = std.testing.allocator;
    const hidden = 400;
    var harmful = try derive.Mean.init(a, 1, hidden);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 1, hidden);
    defer harmless.deinit(a);
    // One massive dim that differs between the sets, a small refusal signal on dim 1, and low noise elsewhere.
    const h = try a.alloc(f32, hidden);
    defer a.free(h);
    const b = try a.alloc(f32, hidden);
    defer a.free(b);
    for (0..hidden) |i| {
        h[i] = 1;
        b[i] = 0.8;
    }
    h[0] = 900;
    h[1] = 3;
    b[0] = 100;
    b[1] = 1;
    try harmful.addPrompt(h, 1);
    try harmless.addPrompt(b, 1);
    var derived = try derive.derive(a, harmful, harmless, 1, hidden, 0.5 / 100.0);
    defer derived.deinit(a);
    try std.testing.expectEqual(@as(usize, 1), derived.layers.len);
    const dir = derived.layers[0].dir;
    // 0.5% of 400 is two dims: dim 0 (the massive one) and dim 1 (the next largest).
    try std.testing.expectEqual(@as(f32, 0), dir[0]);
    try std.testing.expectEqual(@as(f32, 0), dir[1]);
    var norm: f64 = 0;
    for (dir) |v| norm += @as(f64, v) * v;
    try std.testing.expectApproxEqAbs(@as(f64, 1), norm, 1e-5);
    // Unmasked, the direction is the massive dim's.
    var plain = try derive.derive(a, harmful, harmless, 1, hidden, 0);
    defer plain.deinit(a);
    try std.testing.expect(plain.layers[0].dir[0] > 0.99);
    // The screen reads the peak-to-median ratio, which here is the massive dim's.
    const scratch = try a.alloc(f32, hidden);
    defer a.free(scratch);
    try std.testing.expect(derive.massiveRatio(harmful, harmless, 0, hidden, scratch) > derive.massive_threshold);
}

test "the recommended mask leaves 26 of 5120 dims out, as the receipt says" {
    const a = std.testing.allocator;
    const hidden = 5120;
    var harmful = try derive.Mean.init(a, 1, hidden);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 1, hidden);
    defer harmless.deinit(a);
    const h = try a.alloc(f32, hidden);
    defer a.free(h);
    const b = try a.alloc(f32, hidden);
    defer a.free(b);
    for (0..hidden) |i| {
        h[i] = @floatFromInt(i + 1);
        b[i] = 0;
    }
    try harmful.addPrompt(h, 1);
    try harmless.addPrompt(b, 1);
    var derived = try derive.derive(a, harmful, harmless, 1, hidden, derive.mask_recommended);
    defer derived.deinit(a);
    var zeros: usize = 0;
    for (derived.layers[0].dir) |v| zeros += @intFromBool(v == 0);
    try std.testing.expectEqual(@as(usize, 26), zeros);
    // The dims left out are the largest ones, so the direction has no mass past dim 5120-26.
    for (derived.layers[0].dir[hidden - 26 ..]) |v| try std.testing.expectEqual(@as(f32, 0), v);
}

test "the file the producer writes passes the reader and the model it names" {
    const a = std.testing.allocator;
    const hidden = 16;
    var harmful = try derive.Mean.init(a, 4, hidden);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, 4, hidden);
    defer harmless.deinit(a);
    const harmful_sums = try onePrompt(a, 4, hidden, &.{ 1, 2, 3, 4 });
    defer a.free(harmful_sums);
    const harmless_sums = try onePrompt(a, 4, hidden, &.{ 0, 1, 1, 0 });
    defer a.free(harmless_sums);
    try harmful.addPrompt(harmful_sums, 1);
    try harmless.addPrompt(harmless_sums, 1);
    var derived = try derive.derive(a, harmful, harmless, 4, hidden, 0);
    defer derived.deinit(a);

    const layers = try fileLayers(a, derived);
    defer a.free(layers);
    try std.testing.expectEqual(@as(usize, 3), layers.len);
    const bytes = try glp.write(a, .{
        .mode = .project,
        .alpha = 1.0,
        .base_model = "example/dsv41-pack",
        .base_revision = "0123456789abcdef0123456789abcdef01234567",
        .method = derive.method,
        .contrast = derive.contrast,
        .layers = layers,
    });
    defer a.free(bytes);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const vector = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(glp.Mode.project, vector.mode);
    try std.testing.expectEqualStrings(glp.site_name, vector.hook_point);
    try std.testing.expectEqual(glp.spec_version, vector.spec_version);
    try std.testing.expectEqualStrings(derive.method, vector.method);
    try std.testing.expectEqualStrings(derive.contrast, vector.contrast);
    try std.testing.expectEqual(@as(f32, 1.0), vector.alpha);
    try std.testing.expectEqual(@as(usize, hidden), vector.width);
    try std.testing.expectEqual(@as(usize, 3), vector.layers.len);
    try std.testing.expectEqual(@as(u32, 1), vector.layers[0].id);
    for (vector.layers, derived.layers[1..]) |read, made| {
        for (made.dir, read.dir) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    }
    // The checkpoint it names admits it, and one of another shape does not.
    try glp.checkModel(vector, .{ .hidden = hidden, .layers = 4, .repo = "example/dsv41-pack", .revision = "0123456789abcdef0123456789abcdef01234567" }, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(vector, .{ .hidden = hidden, .layers = 3 }, &why));
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(vector, .{ .hidden = 8 }, &why));
    var other: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(vector, .{ .repo = "other/checkpoint" }, &other));
    try std.testing.expect(other.len > 0);
}

test "a masked direction at the checkpoint's width reads back the same bytes" {
    const a = std.testing.allocator;
    const hidden = 5120;
    const layers_n = 3;
    var harmful = try derive.Mean.init(a, layers_n, hidden);
    defer harmful.deinit(a);
    var harmless = try derive.Mean.init(a, layers_n, hidden);
    defer harmless.deinit(a);
    const h = try onePrompt(a, layers_n, hidden, &.{ 1, 2, 3 });
    defer a.free(h);
    const b = try onePrompt(a, layers_n, hidden, &.{ 0, 1, 1 });
    defer a.free(b);
    // A massive dim on the last layer, so the mask has something to leave out.
    b[2 * hidden] = 500;
    try harmful.addPrompt(h, 1);
    try harmless.addPrompt(b, 1);
    var derived = try derive.derive(a, harmful, harmless, layers_n, hidden, derive.mask_recommended);
    defer derived.deinit(a);
    const layers = try fileLayers(a, derived);
    defer a.free(layers);
    const bytes = try glp.write(a, .{
        .mode = .project,
        .alpha = 1.0,
        .base_model = "example/dsv41-pack",
        .method = derive.method,
        .contrast = derive.contrast,
        .layers = layers,
    });
    defer a.free(bytes);
    // Two layers of 5120 fp32 dirs, in a container that is metadata plus the tensors.
    try std.testing.expect(bytes.len > 2 * hidden * 4 and bytes.len < 2 * hidden * 4 + 8192);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const vector = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(@as(usize, hidden), vector.width);
    try std.testing.expectEqual(@as(usize, 2), vector.layers.len);
    try std.testing.expectEqual(@as(u32, 2), vector.layers[1].id);
    for (derived.layers[1].dir, vector.layers[0].dir) |want, got| try std.testing.expectApproxEqAbs(want, got, 1e-6);
    // Its own digest covers the directions, and the reader verified it.
    try std.testing.expect(vector.digest_claimed);
}
