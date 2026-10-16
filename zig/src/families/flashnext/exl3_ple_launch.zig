//! PLE launch walks `Admitted.ple_layer_ids` and no other layer.
//! An absent or empty list is `MissingField` before a read. One caller-owned
//! buffer is reused; the n-gram table is not built per layer.

const std = @import("std");
const admit = @import("exl3_admit.zig");
const ple = @import("exl3_ple.zig");

/// A layer the host could gather. Launch ignores every id that admission left out.
/// `shards` is null or empty when that layer's n-gram shard is not on disk.
pub const Layer = struct {
    id: u16,
    shards: ?[]const ple.Shard = null,
    rows: []const u64,
};

const Found = struct {
    shards: []const ple.Shard,
    rows: []const u64,
};

fn find(layers: []const Layer, id: u16) error{MissingShard}!Found {
    for (layers) |layer| {
        if (layer.id != id) continue;
        const shards = layer.shards orelse return error.MissingShard;
        if (shards.len == 0) return error.MissingShard;
        return .{ .shards = shards, .rows = layer.rows };
    }
    return error.MissingShard;
}

/// Gathers rows for each admitted id, in list order, through `ple.gatherFrom`.
/// `reader` is the `Disk` or hot-row reader those gathers already take.
/// Shard presence is checked before any read, so a missing shard does not
/// pull a different layer. `dest` is reused and must hold one layer's rows.
pub fn launch(
    admitted: admit.Admitted,
    layers: []const Layer,
    words_per_row: u32,
    reader: anytype,
    dest: []i16,
) !void {
    const ids = try admitted.pleLaunch();
    if (ids.len == 0) return error.MissingField;

    for (ids) |id| _ = try find(layers, id);

    for (ids) |id| {
        const layer = try find(layers, id);
        const need = layer.rows.len * @as(usize, words_per_row);
        try ple.gatherFrom(layer.shards, words_per_row, layer.rows, reader, dest[0..need]);
    }
}

fn contains(layers: []const u32, id: u32) bool {
    for (layers) |layer| if (layer == id) return true;
    return false;
}

/// Ple ids missing from the pinned `key_proj` layers or the pinned `value_proj` layers.
/// Count is checked before any store; a short `out` is `error.Length`.
pub fn projectionDisagreement(
    ple_ids: []const u32,
    key_layers: []const u32,
    value_layers: []const u32,
    out: []u32,
) error{Length}!usize {
    var n: usize = 0;
    for (ple_ids) |id| {
        if (!contains(key_layers, id) or !contains(value_layers, id)) n +|= 1;
    }
    if (n > out.len) return error.Length;
    var w: usize = 0;
    for (ple_ids) |id| {
        if (!contains(key_layers, id) or !contains(value_layers, id)) {
            out[w] = id;
            w +|= 1;
        }
    }
    return n;
}

const words: u32 = 51;
const row_b: u64 = 102;

fn fixture(ids: []const u16, on: bool) admit.Admitted {
    return .{
        .model_type = "qwen4_exp",
        .quant_method = "exl3",
        .hidden = 0,
        .layers = 0,
        .experts = 0,
        .topk = 0,
        .expert_intermediate = 0,
        .vocab = 0,
        .ngram_shards = 0,
        .ngram_bits = 5,
        .ple = .page_cache,
        .kv = .bf16,
        .ranks = 1,
        .cache_bytes = 0,
        .ssd_experts = false,
        .ngram_host = true,
        .ple_layer_ids = ids,
        .ple_listed = on,
    };
}

fn paint(buf: []u8, at: usize, global: u64) void {
    var w: usize = 0;
    while (w < words) : (w +|= 1) {
        const v: i16 = @intCast(global * 64 + w);
        std.mem.writeInt(i16, buf[at + w * 2 ..][0..2], v, .little);
    }
}

/// In-memory stand-in for the gather read. A span of `reject` bytes is the whole shard.
const probe = struct {
    var image: []const []const u8 = &.{};
    var reject: u64 = 0;
    var file: [8]u32 = @splat(0);
    var offset: [8]u64 = @splat(0);
    var len: [8]u64 = @splat(0);
    var n: usize = 0;

    fn reset(files: []const []const u8, whole: u64) void {
        image = files;
        reject = whole;
        n = 0;
    }

    pub fn read(_: @This(), file_id: u32, at: u64, dest: []u8) anyerror!void {
        if (reject != 0 and @as(u64, dest.len) == reject) return error.WholeShard;
        const src = image[file_id];
        const start: usize = @intCast(at);
        if (start + dest.len > src.len) return error.ShortRead;
        @memcpy(dest, src[start..][0..dest.len]);
        file[n] = file_id;
        offset[n] = at;
        len[n] = dest.len;
        n +|= 1;
    }
};

test "absent or empty ple_layer_ids is MissingField before any read" {
    var image: [row_b]u8 = @splat(0);
    const files = [_][]const u8{&image};
    probe.reset(&files, row_b);
    const shard = [_]ple.Shard{.{ .file = 0, .offset = 0, .rows = 1 }};
    const layers = [_]Layer{.{ .id = 2, .shards = &shard, .rows = &.{0} }};
    var dest: [words]i16 = @splat(-1);

    try std.testing.expectError(error.MissingField, launch(fixture(&.{}, false), &layers, words, probe{}, &dest));
    try std.testing.expectError(error.MissingField, launch(fixture(&.{}, true), &layers, words, probe{}, &dest));
    try std.testing.expectEqual(@as(usize, 0), probe.n);
    try std.testing.expectEqual(@as(i16, -1), dest[0]);
}

test "launch reads admitted layer 2 and does not read the layer 0 decoy" {
    var image: [2 * row_b]u8 = @splat(0);
    paint(&image, 0, 7);
    paint(&image, row_b, 9);
    const files = [_][]const u8{&image};
    probe.reset(&files, 2 * row_b);

    const layer2 = [_]ple.Shard{.{ .file = 0, .offset = 0, .rows = 1 }};
    const decoy = [_]ple.Shard{.{ .file = 0, .offset = row_b, .rows = 1 }};
    const layers = [_]Layer{
        .{ .id = 0, .shards = &decoy, .rows = &.{0} },
        .{ .id = 2, .shards = &layer2, .rows = &.{0} },
    };
    var dest: [words]i16 = @splat(-1);
    try launch(fixture(&.{2}, true), &layers, words, probe{}, &dest);

    try std.testing.expectEqual(@as(usize, 1), probe.n);
    try std.testing.expectEqual(@as(u32, 0), probe.file[0]);
    try std.testing.expectEqual(@as(u64, 0), probe.offset[0]);
    try std.testing.expectEqual(row_b, probe.len[0]);
    try std.testing.expect(probe.len[0] != probe.reject);
    try std.testing.expectEqual(@as(i16, 7 * 64), dest[0]);
    try std.testing.expectEqual(@as(i16, 7 * 64 + 50), dest[50]);
    try std.testing.expect(dest[0] != 9 * 64);
}

test "a missing shard returns before a different layer is read" {
    var image: [row_b]u8 = @splat(0);
    paint(&image, 0, 3);
    const files = [_][]const u8{&image};
    const present = [_]ple.Shard{.{ .file = 0, .offset = 0, .rows = 1 }};
    const layers = [_]Layer{
        .{ .id = 0, .shards = null, .rows = &.{0} },
        .{ .id = 2, .shards = &present, .rows = &.{0} },
    };
    var dest: [words]i16 = @splat(-1);

    probe.reset(&files, row_b);
    try std.testing.expectError(error.MissingShard, launch(fixture(&.{ 0, 2 }, true), &layers, words, probe{}, &dest));
    try std.testing.expectEqual(@as(usize, 0), probe.n);

    probe.reset(&files, row_b);
    try std.testing.expectError(error.MissingShard, launch(fixture(&.{ 2, 0 }, true), &layers, words, probe{}, &dest));
    try std.testing.expectEqual(@as(usize, 0), probe.n);
    try std.testing.expectEqual(@as(i16, -1), dest[0]);
}

test "a whole-shard read is still WholeShard from Disk" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const alloc = std.testing.allocator;

    var image: [2 * row_b]u8 = @splat(0);
    paint(&image, 0, 1);
    paint(&image, row_b, 2);
    try tmp.dir.writeFile(io, .{ .sub_path = "shard.bin", .data = &image });
    const path = try tmp.dir.realPathFileAlloc(io, "shard.bin", alloc);
    defer alloc.free(path);
    const z = try std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
    defer alloc.free(z);

    const fds = [_]c_int{try ple.Disk.openPath(z)};
    var disk = ple.Disk{ .fds = &fds, .whole = 2 * row_b };
    defer disk.close();

    const shard = [_]ple.Shard{.{ .file = 0, .offset = 0, .rows = 2 }};
    const layers = [_]Layer{.{ .id = 2, .shards = &shard, .rows = &.{ 0, 1 } }};
    var dest: [2 * words]i16 = @splat(-1);
    try std.testing.expectError(error.WholeShard, launch(fixture(&.{2}, true), &layers, words, &disk, &dest));
}

test "projection disagreement lists ple ids missing from key or value" {
    var out: [4]u32 = @splat(9);
    const n = try projectionDisagreement(&.{ 2, 6, 8, 4 }, &.{ 2, 6 }, &.{ 2, 8 }, &out);
    try std.testing.expectEqual(@as(usize, 3), n);
    try std.testing.expectEqualSlices(u32, &.{ 6, 8, 4 }, out[0..n]);
    try std.testing.expectEqual(@as(u32, 9), out[3]);

    var exact: [3]u32 = @splat(9);
    const n_exact = try projectionDisagreement(&.{ 2, 6, 8, 4 }, &.{ 2, 6 }, &.{ 2, 8 }, &exact);
    try std.testing.expectEqual(@as(usize, 3), n_exact);
    try std.testing.expectEqualSlices(u32, &.{ 6, 8, 4 }, &exact);

    var small: [2]u32 = @splat(9);
    try std.testing.expectError(error.Length, projectionDisagreement(&.{ 2, 6, 8, 4 }, &.{ 2, 6 }, &.{ 2, 8 }, &small));
    try std.testing.expectEqual(@as(u32, 9), small[0]);
    try std.testing.expectEqual(@as(u32, 9), small[1]);

    var none: [2]u32 = @splat(9);
    const z = try projectionDisagreement(&.{ 2, 6 }, &.{ 6, 2 }, &.{ 2, 6 }, &none);
    try std.testing.expectEqual(@as(usize, 0), z);
    try std.testing.expectEqual(@as(u32, 9), none[0]);

    var one: [1]u32 = @splat(9);
    const both = try projectionDisagreement(&.{4}, &.{}, &.{ 4, 1 }, &one);
    try std.testing.expectEqual(@as(usize, 1), both);
    try std.testing.expectEqual(@as(u32, 4), one[0]);
}
