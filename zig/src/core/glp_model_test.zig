//! Model, strength and handoff-policy tests adapted from #1835 c5947a8d2e for the GLPv3 fold site.

const std = @import("std");
const glp = @import("glp.zig");
const t = @import("glp_fixture.zig");

const io = std.testing.io;

/// The text model DeepSeek-V4.1 has: 40 backbone layers beyond which the DSpark blocks sit.
const dsv41: glp.Model = .{ .hidden = 8, .layers = 40 };

test "a vector whose width is not the model's is refused" {
    var got = try t.accept(.{});
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    var why: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(got.read.vector, .{ .hidden = 5120, .layers = 40 }, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "8 wide") != null);
    try glp.checkModel(got.read.vector, dsv41, &why);
}

test "the backbone's last layer is steerable and the first DSpark block is not" {
    inline for ([_]struct { id: u32, ok: bool }{ .{ .id = 39, .ok = true }, .{ .id = 40, .ok = false }, .{ .id = 42, .ok = false } }) |c| {
        var got = try t.accept(.{ .layers = &.{ 1, c.id } });
        defer std.testing.allocator.free(got.bytes);
        defer got.read.deinit();
        var why: glp.Why = .{};
        if (c.ok) {
            try glp.checkModel(got.read.vector, dsv41, &why);
        } else {
            try std.testing.expectError(error.BadGlpVector, glp.checkModel(got.read.vector, dsv41, &why));
            try std.testing.expect(std.mem.indexOf(u8, why.text(), "40 backbone layers") != null);
            try std.testing.expect(std.mem.indexOf(u8, why.text(), "deepest steerable layer is 39") != null);
        }
    }
}

test "a vector is for exactly one repo: a conversion, a fork or a similar name is another checkpoint" {
    var got = try t.accept(.{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017" });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const v = got.read.vector;
    var why: glp.Why = .{};
    try glp.checkModel(v, .{ .hidden = 8, .layers = 40, .repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017" }, &why);
    // the hub names a repo without regard to case
    try glp.checkModel(v, .{ .hidden = 8, .layers = 40, .repo = "DeepSeek-AI/deepseek-v4.1-flash-1017" }, &why);
    // a longer name that begins with the vector's is not that checkpoint: converted weights, a fork, an abliteration
    for ([_][]const u8{
        "someone/DeepSeek-V4.1-Flash-1017-EXL3",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017-abliterated",
        "someone/DeepSeek-V4.1-Flash-10170",
        "deepseek-ai/DeepSeek-V4.1-Flash-0731",
        "someone/DeepSeek-V4.1-Flash-1017",
        "DeepSeek-V4.1-Flash-1017",
    }) |repo| {
        try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, .{ .hidden = 8, .layers = 40, .repo = repo }, &why));
        try std.testing.expect(std.mem.indexOf(u8, why.text(), "the model is") != null);
    }
    // the vector's name may be the shorter one too
    var short = try t.accept(.{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash" });
    defer std.testing.allocator.free(short.bytes);
    defer short.read.deinit();
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(short.read.vector, .{ .hidden = 8, .layers = 40, .repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017" }, &why));
    // a checkpoint whose source is unknown is not matched by name; its geometry is all that is checked
    try glp.checkModel(v, dsv41, &why);
}

test "the hub's URL names the same checkpoint as its repo id" {
    var got = try t.accept(.{ .base_model = null, .extra = &.{
        .{ .key = "general.base_model.count", .value = .{ .u32v = 1 } },
        .{ .key = "general.base_model.0.repo_url", .value = .{ .text = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017" } },
    } });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    var why: glp.Why = .{};
    try std.testing.expectEqualStrings("deepseek-ai/DeepSeek-V4.1-Flash-1017", got.read.vector.base_model);
    try glp.checkModel(got.read.vector, .{ .hidden = 8, .layers = 40, .repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017" }, &why);
}

test "the same repo at another commit is refused, a shorter commit name is the same commit" {
    const commit = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";
    const bytes = try t.writeFixture(.{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017", .base_revision = commit[0..12] });
    defer std.testing.allocator.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    const repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
    var why: glp.Why = .{};
    try glp.checkModel(read.vector, .{ .hidden = 8, .layers = 40, .repo = repo, .revision = commit }, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(read.vector, .{ .hidden = 8, .layers = 40, .repo = repo, .revision = "ffffffffffffffffffffffffffffffffffffffff" }, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "revision") != null);
    // the served commit may be cut as short as git's seven characters, and no shorter
    try glp.checkModel(read.vector, .{ .hidden = 8, .layers = 40, .repo = repo, .revision = commit[0..7] }, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(read.vector, .{ .hidden = 8, .layers = 40, .repo = repo, .revision = commit[0..6] }, &why));
    // another repo is refused by name before any commit is compared
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(read.vector, .{ .hidden = 8, .layers = 40, .repo = "someone/DeepSeek-V4.1-Flash-1017-EXL3", .revision = commit }, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "the model is") != null);
}

test "an additive edit that BF16 cannot hold is refused, a projection is not sized by the file" {
    var add = try t.accept(.{ .mode = "add", .value = 3.0e38 });
    defer std.testing.allocator.free(add.bytes);
    defer add.read.deinit();
    var why: glp.Why = .{};
    try glp.checkStrength(add.read.vector, 1.0, &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkStrength(add.read.vector, 2.0, &why));
    try std.testing.expectError(error.BadGlpVector, glp.checkStrength(add.read.vector, -1.0, &why));
    try std.testing.expectError(error.BadGlpVector, glp.checkStrength(add.read.vector, std.math.nan(f32), &why));
    var project = try t.accept(.{});
    defer std.testing.allocator.free(project.bytes);
    defer project.read.deinit();
    try glp.checkStrength(project.read.vector, 1000.0, &why);
}

test "the strength is finite and from 0, and only 0 is ever zero" {
    try std.testing.expectEqual(@as(?f32, 0.5), glp.strength(0.5));
    try std.testing.expectEqual(@as(?f32, 3.5), glp.strength(3.5));
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(glp.strength(0.0).?)));
    // minus zero is the strength 0, as the plus one: the sign is not part of the setting
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(glp.strength(-0.0).?)));
    for ([_]f64{ -1, -1e-30, std.math.nan(f64), std.math.inf(f64), -std.math.inf(f64), 1e39, 1e300 }) |x| try std.testing.expect(glp.strength(x) == null);
    // a value that FP32 would flush to zero, or hold as a subnormal, is not a way to ask for 0
    for ([_]f64{ 1e-100, 1e-40, 1e-39 }) |x| try std.testing.expect(glp.strength(x) == null);
    try std.testing.expect(glp.strength(std.math.floatMin(f32)) != null);
}

test "an alpha of 0 is off and any other strength is on" {
    var got = try t.accept(.{});
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const off: glp.Glp = .{ .vector = got.read.vector, .alpha = 0 };
    const on: glp.Glp = .{ .vector = got.read.vector, .alpha = 0.25 };
    try std.testing.expect(off.off());
    try std.testing.expect(!on.off());
}

test "a hand-off identity binds the file, the mode, the strength and the checkpoint" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var one = try t.accept(.{});
    defer a.free(one.bytes);
    defer one.read.deinit();
    var other = try t.accept(.{ .layers = &.{ 11, 13 } });
    defer a.free(other.bytes);
    defer other.read.deinit();
    const base: glp.Glp = .{ .vector = one.read.vector, .alpha = 3.5 };
    const cp: []const u8 = "checkpoint-0123456789abcdef";

    // no steering is the checkpoint itself: the same bytes, nothing allocated
    const plain = try glp.identity(std.testing.failing_allocator, cp, null);
    try std.testing.expectEqual(cp.ptr, plain.ptr);

    const id = try glp.identity(ar, cp, base);
    try std.testing.expect(std.mem.startsWith(u8, id, cp));
    try std.testing.expectEqualStrings("+glp-", id[cp.len..][0..5]);
    try std.testing.expectEqual(cp.len + "+glp-".len + 32, id.len);
    try std.testing.expectEqualStrings(id, try glp.identity(ar, cp, base));

    var variants: std.ArrayList([]const u8) = .empty;
    try variants.append(ar, id);
    // another vector file, another strength, zero, another mode over the same file, another checkpoint
    try variants.append(ar, try glp.identity(ar, cp, .{ .vector = other.read.vector, .alpha = 3.5 }));
    try variants.append(ar, try glp.identity(ar, cp, .{ .vector = one.read.vector, .alpha = 3.25 }));
    try variants.append(ar, try glp.identity(ar, cp, .{ .vector = one.read.vector, .alpha = 0 }));
    var as_add = one.read.vector;
    as_add.mode = .add;
    try variants.append(ar, try glp.identity(ar, cp, .{ .vector = as_add, .alpha = 3.5 }));
    try variants.append(ar, try glp.identity(ar, "checkpoint-fedcba9876543210", base));
    for (variants.items, 0..) |x, i| for (variants.items[0..i]) |y| try std.testing.expect(!std.mem.eql(u8, x, y));

    // the sign of zero is not part of the strength
    const zero = try glp.identity(ar, cp, .{ .vector = one.read.vector, .alpha = 0 });
    try std.testing.expectEqualStrings(zero, try glp.identity(ar, cp, .{ .vector = one.read.vector, .alpha = -0.0 }));
}

test "a vector file loads with its own strength, or the operator's" {
    const a = std.testing.allocator;
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const bytes = try t.writeFixture(.{ .alpha = 2.0 });
    defer a.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "v.gguf", .data = bytes });
    const path = try std.fs.path.join(ar, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "v.gguf" });
    var why: glp.Why = .{};
    const own = try glp.load(ar, io, path, null, &why);
    try std.testing.expectEqual(@as(f32, 2.0), own.alpha);
    try std.testing.expectEqual(glp.Mode.project, own.vector.mode);
    const given = try glp.load(ar, io, path, 0.5, &why);
    try std.testing.expectEqual(@as(f32, 0.5), given.alpha);
    const zero = try glp.load(ar, io, path, 0, &why);
    try std.testing.expect(zero.off());
    try std.testing.expectError(error.BadGlpVector, glp.load(ar, io, path, -1, &why));
    try std.testing.expectError(error.BadGlpVector, glp.load(ar, io, path, std.math.nan(f32), &why));
    try std.testing.expectError(error.BadGlpVector, glp.load(ar, io, ".zig-cache/tmp/no-such-vector.gguf", null, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "no such file") != null);
}
