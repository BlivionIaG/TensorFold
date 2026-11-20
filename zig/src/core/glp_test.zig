//! GLPv3 acceptance/refusal cases adapted from #1835 c5947a8d2e, using synthetic containers only.

const std = @import("std");
const glp = @import("glp.zig");
const gguf = @import("glp_container.zig");
const t = @import("glp_fixture.zig");

const Shape = t.Shape;

test "a v3 project vector with the metadata a producer writes is accepted" {
    var got = try t.accept(.{});
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const v = got.read.vector;
    var want: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(got.bytes, &want, .{});
    try std.testing.expectEqualSlices(u8, &want, &v.file_sha256);
    try std.testing.expectEqual(glp.Mode.project, v.mode);
    try std.testing.expectEqual(glp.Site.hc_fold_pre_engram, v.site);
    try std.testing.expectEqual(@as(u32, 3), v.spec_version);
    try std.testing.expectEqual(@as(f32, 3.5), v.alpha);
    try std.testing.expectEqual(@as(usize, 8), v.width);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
    try std.testing.expectEqualStrings("hc_fold_pre_engram", v.hook_point);
    try std.testing.expectEqualStrings("deepseek-ai/DeepSeek-V4.1-Flash-1017", v.base_model);
    try std.testing.expect(v.digest_claimed);
    try std.testing.expect(!v.transferred);
}

test "an add vector is accepted as add, with any length of direction" {
    var got = try t.accept(.{ .mode = "add", .scale = 0.25 });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    try std.testing.expectEqual(glp.Mode.add, got.read.vector.mode);
    // an additive direction is applied as it stands: its length is the producer's strength
    try std.testing.expectEqual(t.component(.{ .scale = 0.25 }, 11, 3), got.read.vector.layers[0].dir[3]);
}

test "the directions are the file's bytes: a project direction is never rescaled" {
    // a squared norm 8e-4 off one is inside the unit bound and is read exactly as it was written
    const shape: Shape = .{ .scale = 1.0004 };
    var got = try t.accept(shape);
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    for (got.read.vector.layers) |l| {
        for (l.dir, 0..) |x, i| try std.testing.expectEqual(t.component(shape, l.id, i), x);
    }
}

test "a project direction that is not a unit vector is refused, never rescaled" {
    try t.expectRefused(.{ .scale = 1.001 }, "unit vector");
    try t.expectRefused(.{ .scale = 0.9 }, "unit vector");
    try t.expectRefused(.{ .scale = 2.0 }, "squared norm");
}

test "a mode other than add or project is refused, not coerced" {
    for ([_][]const u8{ "scale", "ablate", "Project", "ADD", "" }) |mode| try t.expectRefused(.{ .mode = mode }, "glp.mode");
    try t.expectRefused(.{ .mode = null }, "names no glp.mode");
    // the standard mode key and a second one: which applies would depend on the reader, so the container is refused
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.mode", .value = .{ .text = "add" } }} }, "twice");
}

test "a mode that is not text is refused" {
    // the standard mode key is dropped and an integer one written in its place
    try t.expectRefused(.{ .mode = null, .extra = &.{.{ .key = "glp.mode", .value = .{ .u32v = 1 } }} }, "glp.mode is not text");
}

test "a spec version other than 3 is refused, whatever the hook" {
    for ([_]u32{ 0, 1, 2, 4, 99 }) |version| try t.expectRefused(.{ .spec_version = version }, "glp.spec_version");
    try t.expectRefused(.{ .spec_version = null }, "no glp.spec_version");
    try t.expectRefused(.{ .numeric_text = .version }, "glp.spec_version is not an unsigned integer");
    // a version 1 file at the v1 flat hook is a file this reader has no site for
    try t.expectRefused(.{ .spec_version = 1, .hook_point = "residual_stream_post_layer" }, "glp.spec_version 1");
}

test "a hook other than the fold site is refused, the v1 flat residual included" {
    for ([_][]const u8{ "residual_stream_post_layer", "ffn_out_pre_residual", "attn_out_pre_residual", "hc_fold_post_engram", "" }) |hook| {
        try t.expectRefused(.{ .hook_point = hook }, "hook point");
    }
    try t.expectRefused(.{ .hook_point = null }, "names no glp.hook_point");
}

test "a rank above one is refused, in the key and in suffix tensors" {
    for ([_]u32{ 0, 2, 4 }) |rank| try t.expectRefused(.{ .rank = rank }, "glp.rank");
    try t.expectRefused(.{ .numeric_text = .rank }, "glp.rank is not an unsigned integer");
    const names = [_][]const u8{ "direction.11", "direction.11.1" };
    try t.expectRefused(.{ .layers = &.{ 11, 11 }, .tensor_names = &names }, "rank above one");
    // a rank key absent is rank one; a bool orthonormal flag is accepted, another type is not
    var got = try t.accept(.{ .rank = null, .extra = &.{.{ .key = "glp.orthonormal", .value = .{ .boolv = true } }} });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.orthonormal", .value = .{ .u32v = 1 } }} }, "glp.orthonormal is not a boolean");
}

test "strength scales are refused, not ignored" {
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.dir_scales", .value = .{ .text = "1.0" } }} }, "glp.dir_scales");
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.layer_scales", .value = .{ .text = "11:1.0,12:1.0" } }} }, "glp.layer_scales");
}

test "a legacy dspark file is refused, alone or beside glp keys" {
    const legacy = [_]t.Extra{.{ .key = "dspark.mode", .value = .{ .text = "project" } }};
    try t.expectRefused(.{ .spec_version = null, .mode = null, .hook_point = null, .extra = &legacy }, "dspark");
    try t.expectRefused(.{ .extra = &legacy }, "dspark");
}

test "informational and unknown glp keys are read past, whatever their type, without changing the vector" {
    var plain = try t.accept(.{});
    defer std.testing.allocator.free(plain.bytes);
    defer plain.read.deinit();
    var got = try t.accept(.{ .extra = &.{
        .{ .key = "glp.dose_cos", .value = .{ .text = "11:0.1,12:0.2" } },
        .{ .key = "glp.resid_norm_median", .value = .{ .text = "11:9.5,12:9.7" } },
        .{ .key = "glp.dose_ladder", .value = .{ .text = "0.5,1,2" } },
        .{ .key = "glp.future_count", .value = .{ .u32v = 7 } },
        .{ .key = "glp.future_ratio", .value = .{ .f32v = 0.25 } },
        .{ .key = "glp.future_flag", .value = .{ .boolv = true } },
        .{ .key = "producer.note", .value = .{ .text = "made by a test" } },
    } });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    // the notes change the file, not what is applied
    try std.testing.expect(!std.mem.eql(u8, &plain.read.vector.file_sha256, &got.read.vector.file_sha256));
    try std.testing.expectEqualSlices(u8, &plain.read.vector.sha256, &got.read.vector.sha256);
    try std.testing.expectEqual(plain.read.vector.mode, got.read.vector.mode);
    try std.testing.expectEqual(plain.read.vector.alpha, got.read.vector.alpha);
    try std.testing.expectEqual(plain.read.vector.layers.len, got.read.vector.layers.len);
    for (plain.read.vector.layers, got.read.vector.layers) |x, y| {
        try std.testing.expectEqual(x.id, y.id);
        try std.testing.expectEqualSlices(f32, x.dir, y.dir);
    }
    // what changes the edit stays refused beside them: a version, mode, site, rank or scale is never read past
    const notes = [_]t.Extra{.{ .key = "glp.dose_cos", .value = .{ .text = "11:0.1" } }};
    try t.expectRefused(.{ .spec_version = 2, .extra = &notes }, "glp.spec_version");
    try t.expectRefused(.{ .rank = 2, .extra = &notes }, "glp.rank");
    try t.expectRefused(.{ .mode = "scale", .extra = &notes }, "glp.mode");
    try t.expectRefused(.{ .hook_point = "ffn_out_pre_residual", .extra = &notes }, "hook point");
    try t.expectRefused(.{ .extra = &.{ notes[0], .{ .key = "glp.dir_scales", .value = .{ .text = "1" } } } }, "glp.dir_scales");
}

test "the container must be a control vector" {
    try t.expectRefused(.{ .architecture = null }, "general.architecture");
    try t.expectRefused(.{ .architecture = "llama" }, "controlvector");
}

test "direction.N is layer N, with no offset, and layer 0 is not steerable" {
    const shape: Shape = .{ .layers = &.{ 12, 11, 30 } };
    var got = try t.accept(shape);
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const v = got.read.vector;
    // the ids ascend whatever order the file lists them in
    try std.testing.expectEqual(@as(u32, 11), v.layers[0].id);
    try std.testing.expectEqual(@as(u32, 12), v.layers[1].id);
    try std.testing.expectEqual(@as(u32, 30), v.layers[2].id);
    // each id holds the numbers of the tensor that carries its own number, not a neighbour's
    for ([_]u32{ 11, 12, 30 }) |id| {
        for (v.layer(id).?, 0..) |x, i| try std.testing.expectEqual(t.component(shape, id, i), x);
    }
    try std.testing.expect(std.mem.indexOfDiff(f32, v.layer(11).?, v.layer(12).?) != null);
    try std.testing.expect(v.layer(0) == null);
    try std.testing.expect(v.layer(13) == null);
    try std.testing.expect(v.layer(31) == null);
    try t.expectRefused(.{ .layers = &.{ 0, 11 } }, "direction.0");
}

test "a layer id list that disagrees with the tensors is refused, a one-layer shift included" {
    try t.expectRefused(.{ .layers = &.{ 11, 12 }, .layer_ids = "10,11" }, "does not match");
    // llama.cpp's generator names layer il as direction.{il+1}: a list one layer late is what that shift looks like
    try t.expectRefused(.{ .layers = &.{ 11, 12 }, .layer_ids = "12,13" }, "does not match");
    try t.expectRefused(.{ .layers = &.{ 11, 12 }, .layer_ids = "11" }, "names 1 layers and holds 2");
    try t.expectRefused(.{ .layers = &.{ 11, 12 }, .layer_ids = "11,12,13" }, "more layers");
    try t.expectRefused(.{ .layers = &.{ 11, 12 }, .layer_ids = "11,x" }, "not a number list");
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.layer_ids_zero_based", .value = .{ .u32v = 1 } }}, .layer_ids_key = false }, "is not text");
    var got = try t.accept(.{ .layer_ids_key = false });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
}

test "tensors that are not one-layer fp32 directions are refused" {
    try t.expectRefused(.{ .kind = 1 }, "not fp32");
    try t.expectRefused(.{ .two_dims = true }, "not one dimensional");
    try t.expectRefused(.{ .tensor_names = &.{ "direction.11", "weight.12" } }, "not a direction");
    try t.expectRefused(.{ .tensor_names = &.{ "direction.11", "direction.x" } }, "does not name a layer");
    try t.expectRefused(.{ .layers = &.{ 5, 5 }, .tensor_names = &.{ "direction.5", "direction.05" } }, "direction.N");
    try t.expectRefused(.{ .layers = &.{ 5, 5 }, .tensor_names = &.{ "direction.5", "direction.5" } }, "two directions for layer 5");
}

test "a direction that is not finite is refused, and a zero one only for a projection" {
    for ([_]f32{ std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |value| {
        try t.expectRefused(.{ .value = value, .mode = "add" }, "not finite");
        try t.expectRefused(.{ .value = value }, "not finite");
    }
    try t.expectRefused(.{ .value = 0 }, "all zero");
    // an additive zero direction is a no-op, still a vector: it is read, every number zero
    var zero = try t.accept(.{ .mode = "add", .value = 0 });
    defer std.testing.allocator.free(zero.bytes);
    defer zero.read.deinit();
    try std.testing.expectEqual(glp.Mode.add, zero.read.vector.mode);
    for (zero.read.vector.layers) |l| for (l.dir) |x| try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(x)));
}

test "a content digest that does not cover the vectors is refused" {
    try t.expectRefused(.{ .digest = .wrong }, "digest does not match");
    try t.expectRefused(.{ .digest = .number }, "glp.content_sha256 is not text");
    var got = try t.accept(.{ .digest = .absent });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    try std.testing.expect(!got.read.vector.digest_claimed);
}

test "a changed direction byte no longer matches the digest the file claims" {
    const bytes = try t.writeFixture(.{ .mode = "add" });
    defer std.testing.allocator.free(bytes);
    // the last byte is the top byte of the last direction's last component: its exponent moves, the value stays finite
    bytes[bytes.len - 1] ^= 0x01;
    const why = try t.reason(bytes);
    defer std.testing.allocator.free(why);
    try std.testing.expect(std.mem.indexOf(u8, why, "digest does not match") != null);
}

test "the strength is a finite float from 0, and 0 is read as exactly 0" {
    for ([_]f32{ -1, std.math.nan(f32), std.math.inf(f32), -std.math.inf(f32) }) |alpha| try t.expectRefused(.{ .alpha = alpha }, "glp.alpha_default");
    try t.expectRefused(.{ .numeric_text = .alpha }, "not a floating-point strength");
    var zero = try t.accept(.{ .alpha = 0 });
    defer std.testing.allocator.free(zero.bytes);
    defer zero.read.deinit();
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(zero.read.vector.alpha)));
    var negative_zero = try t.accept(.{ .alpha = -0.0 });
    defer std.testing.allocator.free(negative_zero.bytes);
    defer negative_zero.read.deinit();
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(negative_zero.read.vector.alpha)));
    var wide = try t.accept(.{ .alpha = 2.5, .numeric_text = .alpha_f64 });
    defer std.testing.allocator.free(wide.bytes);
    defer wide.read.deinit();
    try std.testing.expectEqual(@as(f32, 2.5), wide.read.vector.alpha);
}

test "a double or subnormal strength FP32 cannot hold as asked is refused, not read as 0 or as a live strength" {
    // doubles that overflow FP32, or underflow to +0 (which would read as "off") or to a subnormal the GPU flushes
    for ([_]f64{ 1e300, 1e-50, 1e-300, 4.9e-324, 1e-39 }) |bad| {
        const bytes = try t.writeFixture(.{ .alpha = 1, .numeric_text = .alpha_f64 });
        defer std.testing.allocator.free(bytes);
        const at = std.mem.indexOf(u8, bytes, &std.mem.toBytes(@as(f64, 1.0))).?;
        @memcpy(bytes[at..][0..8], &std.mem.toBytes(bad));
        const why = try t.reason(bytes);
        defer std.testing.allocator.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, "glp.alpha_default") != null);
    }
    // FP32 subnormals, from the smallest to just under the smallest normal, are refused the same way
    for ([_]f32{ std.math.floatTrueMin(f32), 1e-40, std.math.floatMin(f32) / 2 }) |bad| {
        try t.expectRefused(.{ .alpha = bad }, "glp.alpha_default");
        try t.expectRefused(.{ .alpha = bad, .numeric_text = .alpha_f64 }, "glp.alpha_default");
    }
    // the smallest normal strength is a strength, as a float or as the double that equals it
    for ([_]bool{ false, true }) |as_double| {
        var smallest = try t.accept(.{ .alpha = std.math.floatMin(f32), .numeric_text = if (as_double) .alpha_f64 else null });
        defer std.testing.allocator.free(smallest.bytes);
        defer smallest.read.deinit();
        try std.testing.expectEqual(std.math.floatMin(f32), smallest.read.vector.alpha);
    }
}

test "a hostile or truncated file is refused, not read past" {
    const bytes = try t.writeFixture(.{});
    defer std.testing.allocator.free(bytes);
    for ([_]usize{ 0, 4, 7, 20, bytes.len / 2, bytes.len - 64, bytes.len - 1 }) |len| {
        const why = try t.reason(bytes[0..len]);
        std.testing.allocator.free(why);
    }
    const wrong = try std.testing.allocator.dupe(u8, bytes);
    defer std.testing.allocator.free(wrong);
    wrong[0] = 'X';
    std.testing.allocator.free(try t.reason(wrong));
}

test "a key named twice and a nested array are refused by the container" {
    const a = std.testing.allocator;
    const twice = try gguf.write(a, &.{ .{ .key = "a", .value = .{ .u32v = 1 } }, .{ .key = "a", .value = .{ .u32v = 2 } } }, &.{});
    defer a.free(twice);
    var why: glp.Why = .{};
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    try std.testing.expectError(error.BadGlpVector, glp.read(arena.allocator(), twice, &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "twice") != null);
    const inner = [_]gguf.Value{.{ .u32v = 1 }};
    const nested = [_]gguf.Value{.{ .array = .{ .elem = .u32v, .items = &inner } }};
    const bytes = try gguf.write(a, &.{.{ .key = "n", .value = .{ .array = .{ .elem = .array, .items = &nested } } }}, &.{});
    defer a.free(bytes);
    try std.testing.expectError(error.BadGguf, t.parseFile(a, bytes));
}

test "an array that claims more items than the file has is refused" {
    const a = std.testing.allocator;
    const bytes = try gguf.write(a, &.{.{ .key = "n", .value = .{ .array = .{ .elem = .u8v, .items = &.{.{ .u8v = 1 }} } } }}, &.{});
    defer a.free(bytes);
    // key header: magic 4 + version 4 + counts 16, key length 8 + "n", type word 4, element type 4, then the count
    const count_at = 4 + 4 + 16 + 8 + 1 + 4 + 4;
    std.mem.writeInt(u64, bytes[count_at..][0..8], 1 << 20, .little);
    try std.testing.expectError(error.BadGguf, t.parseFile(a, bytes));
}

test "the container keeps every value type it is given" {
    const a = std.testing.allocator;
    const entries = [_]gguf.Entry{
        .{ .key = "a.u8", .value = .{ .u8v = 7 } },
        .{ .key = "a.i8", .value = .{ .i8v = -7 } },
        .{ .key = "a.u16", .value = .{ .u16v = 1000 } },
        .{ .key = "a.i16", .value = .{ .i16v = -1000 } },
        .{ .key = "a.u32", .value = .{ .u32v = 100000 } },
        .{ .key = "a.i32", .value = .{ .i32v = -100000 } },
        .{ .key = "a.f32", .value = .{ .f32v = 0.5 } },
        .{ .key = "a.bool", .value = .{ .boolv = true } },
        .{ .key = "a.string", .value = .{ .string = "hello" } },
        .{ .key = "a.array", .value = .{ .array = .{ .elem = .u32v, .items = &.{ .{ .u32v = 1 }, .{ .u32v = 2 } } } } },
        .{ .key = "a.u64", .value = .{ .u64v = 1 << 40 } },
        .{ .key = "a.i64", .value = .{ .i64v = -(1 << 40) } },
        .{ .key = "a.f64", .value = .{ .f64v = 0.25 } },
    };
    const bytes = try gguf.write(a, &entries, &.{});
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqual(@as(usize, 13), parsed.file.entries.len);
    try std.testing.expectEqual(@as(u8, 7), parsed.file.find("a.u8").?.u8v);
    try std.testing.expectEqual(@as(i8, -7), parsed.file.find("a.i8").?.i8v);
    try std.testing.expectEqual(@as(i32, -100000), parsed.file.find("a.i32").?.i32v);
    try std.testing.expectEqual(@as(f32, 0.5), parsed.file.find("a.f32").?.f32v);
    try std.testing.expectEqual(true, parsed.file.find("a.bool").?.boolv);
    try std.testing.expectEqualStrings("hello", parsed.file.find("a.string").?.string);
    const array = parsed.file.find("a.array").?.array;
    try std.testing.expectEqual(gguf.Type.u32v, array.elem);
    // an array is walked past, never kept: its items would cost a value each, and no GLP key reads one
    try std.testing.expectEqual(@as(u64, 2), array.skipped);
    try std.testing.expectEqual(@as(usize, 0), array.items.len);
    try std.testing.expectEqual(@as(i64, -(1 << 40)), parsed.file.find("a.i64").?.i64v);
    try std.testing.expectEqual(@as(f64, 0.25), parsed.file.find("a.f64").?.f64v);
}

/// The vector a producer writes for two layers, handed over in the order a caller would hold them.
fn sample(a: std.mem.Allocator, mode: glp.Mode) ![]u8 {
    const dir11 = try t.unitDir(a, 8, 1);
    defer a.free(dir11);
    const dir10 = try t.unitDir(a, 8, 0);
    defer a.free(dir10);
    const layers = [_]glp.Layer{ .{ .id = 11, .dir = dir11 }, .{ .id = 10, .dir = dir10 } };
    return glp.write(a, .{
        .mode = mode,
        .alpha = 0.75,
        .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        .base_revision = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
        .method = "difference_of_means_per_layer",
        .contrast = "matched_prompts_unpaired",
        .layers = &layers,
    });
}

test "a written vector is read back and written again to the same bytes, for both modes" {
    const a = std.testing.allocator;
    for ([_]glp.Mode{ .add, .project }) |mode| {
        const bytes = try sample(a, mode);
        defer a.free(bytes);
        var read = try t.readOk(bytes);
        defer read.deinit();
        try std.testing.expectEqual(mode, read.vector.mode);
        try std.testing.expectEqual(@as(u32, 10), read.vector.layers[0].id);
        const again = try glp.writeVector(a, read.vector);
        defer a.free(again);
        try std.testing.expectEqualSlices(u8, bytes, again);
        var reread = try t.readOk(again);
        defer reread.deinit();
        try std.testing.expectEqualSlices(u8, &read.vector.sha256, &reread.vector.sha256);
        try std.testing.expectEqualSlices(u8, &read.vector.file_sha256, &reread.vector.file_sha256);
    }
}

test "a written vector's digest covers its tensor bytes, and its mode is in the container" {
    const a = std.testing.allocator;
    const bytes = try sample(a, .project);
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (parsed.file.tensors) |tensor| h.update(parsed.file.bytesOf(tensor).?);
    var sha: [32]u8 = undefined;
    h.final(&sha);
    const hex = std.fmt.bytesToHex(sha, .lower);
    try std.testing.expectEqualStrings(hex[0..], parsed.file.str("glp.content_sha256").?);
    try std.testing.expectEqualStrings("project", parsed.file.str("glp.mode").?);
    try std.testing.expectEqualStrings("hc_fold_pre_engram", parsed.file.str("glp.hook_point").?);
    try std.testing.expectEqual(@as(u64, 3), parsed.file.num("glp.spec_version").?);
}

test "a producer refuses what the reader would refuse" {
    const a = std.testing.allocator;
    const unit = try t.unitDir(a, 4, 0);
    defer a.free(unit);
    const nan = try a.alloc(f32, 4);
    defer a.free(nan);
    @memset(nan, std.math.nan(f32));
    const zero = try a.alloc(f32, 4);
    defer a.free(zero);
    @memset(zero, 0);
    const half = try a.alloc(f32, 4);
    defer a.free(half);
    @memset(half, 0.25);
    const short = try t.unitDir(a, 3, 0);
    defer a.free(short);
    const l_nan = [_]glp.Layer{.{ .id = 10, .dir = nan }};
    const l_zero = [_]glp.Layer{.{ .id = 10, .dir = zero }};
    const l_first = [_]glp.Layer{.{ .id = 0, .dir = unit }};
    const l_twice = [_]glp.Layer{ .{ .id = 5, .dir = unit }, .{ .id = 5, .dir = unit } };
    const l_width = [_]glp.Layer{ .{ .id = 5, .dir = unit }, .{ .id = 6, .dir = short } };
    const l_unit = [_]glp.Layer{.{ .id = 5, .dir = unit }};
    const l_half = [_]glp.Layer{.{ .id = 5, .dir = half }};
    for ([_][]const glp.Layer{ &l_nan, &l_first, &l_twice, &l_width, &.{} }) |layers| {
        try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .mode = .add, .layers = layers }));
    }
    // a zero direction is no projection: the unit rule refuses it, and it is a no-op only as an addition
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .mode = .project, .layers = &l_zero }));
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .mode = .add, .alpha = -1, .layers = &l_unit }));
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .mode = .add, .alpha = std.math.nan(f32), .layers = &l_unit }));
    try std.testing.expectError(error.BadGlpVector, glp.write(a, .{ .mode = .project, .layers = &l_half }));
    // the same non-unit direction is a fine additive one
    const ok = try glp.write(a, .{ .mode = .add, .layers = &l_half });
    a.free(ok);
}

test "an additive zero direction is written, read back and written again to the same bytes" {
    const a = std.testing.allocator;
    const zero = try a.alloc(f32, 8);
    defer a.free(zero);
    @memset(zero, 0);
    const layers = [_]glp.Layer{.{ .id = 7, .dir = zero }};
    const bytes = try glp.write(a, .{ .mode = .add, .alpha = 2.0, .layers = &layers });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    try std.testing.expectEqual(glp.Mode.add, read.vector.mode);
    try std.testing.expectEqual(@as(f32, 2.0), read.vector.alpha);
    try std.testing.expectEqualSlices(f32, zero, read.vector.layer(7).?);
    const again = try glp.writeVector(a, read.vector);
    defer a.free(again);
    try std.testing.expectEqualSlices(u8, bytes, again);
    // the stored tensor bytes are all zero and the digest covers them
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    const stored = parsed.file.bytesOf(parsed.file.tensor("direction.7").?).?;
    try std.testing.expectEqual(@as(usize, 32), stored.len);
    for (stored) |byte| try std.testing.expectEqual(@as(u8, 0), byte);
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(stored, &sha, .{});
    try std.testing.expectEqualSlices(u8, &sha, &read.vector.sha256);
}

test "a tensor whose offset reaches the top of the address space is refused, not read" {
    const a = std.testing.allocator;
    const plain = try t.writeFixture(.{});
    defer a.free(plain);
    var parsed = try t.parseFile(a, plain);
    defer parsed.deinit();
    const data_start: u64 = parsed.file.data_start;
    // the last 32-byte aligned address, then offsets that put the tensor's start there, before it and at the very top
    const top = std.math.maxInt(u64) - 31;
    for ([_]u64{ top - data_start, top - data_start - 96, std.math.maxInt(u64) - data_start, std.math.maxInt(u64) - data_start - 1 }) |offset| {
        const bytes = try t.writeFixture(.{ .first_offset = offset });
        defer a.free(bytes);
        const why = try t.reason(bytes);
        defer a.free(why);
        try std.testing.expect(std.mem.indexOf(u8, why, "points past the file") != null);
        // the container's own accessor answers no rather than handing back bytes from wherever the sum wrapped to
        var again = try t.parseFile(a, bytes);
        defer again.deinit();
        try std.testing.expect(again.file.bytesOf(again.file.tensors[0]) == null);
    }
}

test "a small file with a huge ignored array cannot make the reader hold more than the file is long" {
    const a = std.testing.allocator;
    const bytes = try t.writeFixture(.{ .extra = &.{
        .{ .key = "glp.dose_cos", .value = .{ .zeros = 1 << 20 } },
        .{ .key = "producer.names", .value = .{ .texts = &.{ "a", "bb", "ccc" } } },
    } });
    defer a.free(bytes);
    var counting = std.testing.FailingAllocator.init(a, .{});
    var arena = std.heap.ArenaAllocator.init(counting.allocator());
    defer arena.deinit();
    var why: glp.Why = .{};
    const v = try glp.read(arena.allocator(), bytes, &why);
    try std.testing.expectEqual(@as(usize, 2), v.layers.len);
    // a kept item costs a 32-byte value: this array held 1 MiB of items, which would be 32 MiB in the arena
    try std.testing.expect(counting.allocated_bytes < 256 * 1024);
}

test "an array of strings whose count or lengths run past the file is refused" {
    const a = std.testing.allocator;
    const bytes = try gguf.write(a, &.{.{ .key = "n", .value = .{ .array = .{ .elem = .string, .items = &.{.{ .string = "abc" }} } } }}, &.{});
    defer a.free(bytes);
    // key header: magic 4 + version 4 + counts 16, key length 8 + "n", type word 4, element type 4, then the count
    const count_at = 4 + 4 + 16 + 8 + 1 + 4 + 4;
    const many = try a.dupe(u8, bytes);
    defer a.free(many);
    std.mem.writeInt(u64, many[count_at..][0..8], 1 << 40, .little);
    try std.testing.expectError(error.BadGguf, t.parseFile(a, many));
    // the first item's length word follows the count
    const long = try a.dupe(u8, bytes);
    defer a.free(long);
    std.mem.writeInt(u64, long[count_at + 8 ..][0..8], 1 << 40, .little);
    try std.testing.expectError(error.BadGguf, t.parseFile(a, long));
    var intact = try t.parseFile(a, bytes);
    defer intact.deinit();
    try std.testing.expectEqual(@as(u64, 1), intact.file.find("n").?.array.skipped);
}
