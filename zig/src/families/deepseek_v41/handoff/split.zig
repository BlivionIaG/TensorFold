//! Layer-split prefill (opt-in, mixed CUDA/Metal arithmetic): the boundary stream's rows, the cut rule, the chunk
//! schedule both ends share, and the manifest of a split hand-off (boundary stream first, then layers [0, cut)).
const std = @import("std");
const fabric = @import("fabric");
const g = @import("geometry.zig");
const plan = @import("plan.zig");
const p2 = @import("p2.zig");
const manifest = fabric.manifest;

/// Bytes of one boundary row's residual: four planes of the 5120-wide stream, bf16, the layer's post applied.
pub const stream_bytes: u64 = 4 * 5120 * 2;
/// Bytes of the next site's pre coefficients (four f32) that ride behind the stream.
pub const pre_bytes: u64 = 4 * 4;
/// One boundary row on the wire.
pub const row_bytes: u64 = stream_bytes + pre_bytes;
/// A row as bf16 elements (the manifest's latent width; the pre's f32 bytes ride in the last eight).
pub const row_elems: u64 = row_bytes / 2;

/// Manifest layer index of the boundary stream is `layer_base + cut`: it names the cut and stays clear of model layers.
pub const layer_base: u32 = 1000;

/// The default cut: the Sparks run layers [0, 14), the M5 layers 14..19 (Engram 14 among them), then its own cone.
pub const default_cut: u32 = 14;
/// The route's length gate (tokens): below it the pipeline's fill, hop and cone are not hidden.
pub const default_min_tokens: u64 = 16384;
/// The most boundary rows one export stages (host memory: 41 KB a row).
pub const default_max_tokens: u64 = 131072;
/// The first chunks of a split prompt, rows: a short ramp fills the pipeline before the full chunk.
pub const ramp = [_]u32{ 512, 1024, 2048 };
/// A chunk the engines may not leave behind a split chunk: the cone and the short path need more than this.
pub const min_tail_rows: u64 = 17;

/// A cut must start the M5 on an index source below the decoder: the selection a layer reuses is made at its source,
/// and only the boundary rows cross (a cut inside a source's group would need its selection too).
pub fn cutOk(geo: g.Geometry, cut: u32) bool {
    const src = geo.sourceIndex(cut) orelse return false;
    return geo.sources[src].ratio == 2 and cut > 0 and cut < geo.layers;
}

/// Whether a manifest layer is the boundary stream, and the cut it names.
pub fn cutOfLayer(index: u32) ?u32 {
    return if (index > layer_base and index < layer_base + 64) index - layer_base else null;
}

/// The cut a split manifest names, or null for any other hand-off.
pub fn cutOf(m: manifest.Manifest) ?u32 {
    for (m.layers) |l| if (cutOfLayer(l.index)) |c| return c;
    return null;
}

/// The boundary stream as a manifest layer: `rows` rows of `row_elems` bf16, the rows being the block dimension.
pub fn boundaryLayer(arena: std.mem.Allocator, rows: u64, cut: u32) !manifest.Layer {
    const shape = try arena.dupe(u64, &.{ rows, row_elems });
    const dims = try arena.dupe(manifest.Dim, &.{ .block, .latent });
    return .{ .index = layer_base + cut, .kind = .mla, .shape = shape, .dims = dims, .dtype = .bfloat16, .latent_size = row_elems, .rope_size = 0 };
}

/// The state's plan of a split export of `upto` prompt tokens cut at `cut`: layers [0, cut) and no drafter rings.
pub fn stateShape(upto: u64, cut: u32) plan.Shape {
    return .{ .upto = upto, .cut = cut };
}

/// A split hand-off's manifest layers: the boundary stream (position 0, so its frames come first), then the state.
pub fn describe(arena: std.mem.Allocator, upto: u64, cut: u32) ![]manifest.Layer {
    const shape = stateShape(upto, cut);
    try shape.check();
    if (!cutOk(shape.geo, cut)) return error.BadShape;
    const specs = try plan.specs(arena, shape);
    const state = try p2.describe(arena, specs);
    const out = try arena.alloc(manifest.Layer, 1 + state.len);
    out[0] = try boundaryLayer(arena, upto, cut);
    @memcpy(out[1..], state);
    return out;
}

/// Chunk starts and sizes keep to the compressors' 128-row window (the normal path starts every segment on one) but the last.
pub const grid_rows: u64 = 128;

/// Rows a split prompt's i-th chunk takes: the ramp, then `max` (the grid's multiple at or below it, 128 at least).
pub fn chunkRows(i: u32, max: u32) u32 {
    const full = @max(@as(u32, grid_rows), max / @as(u32, grid_rows) * @as(u32, grid_rows));
    return @min(full, if (i < ramp.len) ramp[i] else full);
}

pub const Chunk = struct { row0: u64, rows: u32 };

/// The producer's chunks of `n` rows (`max` at most a chunk): the ramp then full chunks, every one but the last a
/// multiple of 16 rows (so each starts on the grid) and the last never under 17 rows.
pub const Chunks = struct {
    n: u64,
    max: u32,
    at: u64 = 0,
    i: u32 = 0,

    pub fn next(c: *Chunks) ?Chunk {
        if (c.at >= c.n) return null;
        const left = c.n - c.at;
        var rows: u64 = chunkRows(c.i, c.max);
        if (rows >= left) {
            rows = left;
        } else if (left - rows < min_tail_rows) {
            // leave a short tail, the chunk before it on the grid
            const take = if (left > 2 * min_tail_rows) left - min_tail_rows else left;
            const aligned = take / grid_rows * grid_rows;
            rows = if (aligned >= min_tail_rows) aligned else take;
        }
        const out: Chunk = .{ .row0 = c.at, .rows = @intCast(rows) };
        c.at += rows;
        c.i += 1;
        return out;
    }
};

/// The consumer's next pass over the stage: rows [at, at + rows) of the `ready` ones, at most `max` a pass, and never
/// leaving 1..16 rows of the stream behind (the cone and the short path need 17 or none). Null: wait for rows.
pub fn passRows(at: u64, ready: u64, upto: u64, max: u32) ?u32 {
    if (ready <= at or at >= upto) return null;
    var rows: u64 = @min(@min(ready, upto) - at, max);
    const left = upto - (at + rows);
    if (left > 0 and left < min_tail_rows) {
        if (ready >= upto and rows + left <= max) rows += left else if (rows > min_tail_rows) rows -= min_tail_rows - left else return null;
    }
    return @intCast(rows);
}

/// What the reply's runtime block says of a split prompt: the arithmetic is two engines'.
pub const label = "layer-split: mixed CUDA/Metal arithmetic";

test "the cut rule takes the compressing index sources below the decoder, nothing else" {
    try std.testing.expect(cutOk(g.release, 8) and cutOk(g.release, 14) and cutOk(g.release, 2));
    for ([_]u32{ 0, 1, 7, 10, 11, 13, 15, 20, 24, 40 }) |bad| try std.testing.expect(!cutOk(g.release, bad));
    try std.testing.expectEqual(@as(?u32, 14), cutOfLayer(layer_base + 14));
    try std.testing.expectEqual(@as(?u32, null), cutOfLayer(41));
    try std.testing.expectEqual(@as(?u32, null), cutOfLayer(layer_base));
    try std.testing.expectEqual(@as(u64, 40976), row_bytes);
    try std.testing.expectEqual(@as(u64, 20488), row_elems);
}

test "chunks ramp, then run full, and a stream never ends on a stub" {
    var it: Chunks = .{ .n = 20000, .max = 4096 };
    var got: [16]Chunk = undefined;
    var k: usize = 0;
    var total: u64 = 0;
    while (it.next()) |c| : (k += 1) {
        got[k] = c;
        try std.testing.expectEqual(total, c.row0);
        total += c.rows;
    }
    try std.testing.expectEqual(@as(u64, 20000), total);
    try std.testing.expectEqual(@as(u32, 512), got[0].rows);
    try std.testing.expectEqual(@as(u32, 1024), got[1].rows);
    try std.testing.expectEqual(@as(u32, 2048), got[2].rows);
    try std.testing.expectEqual(@as(u32, 4096), got[3].rows);
    // every stream length: the chunks tile it and the last has at least 17 rows (or is the only chunk)
    for ([_]u64{ 1, 16, 17, 511, 512, 513, 529, 530, 3585, 3600, 4097, 4112, 4113, 4129, 16384, 16385, 32768, 32769, 131071 }) |n| {
        var c: Chunks = .{ .n = n, .max = 4096 };
        var sum: u64 = 0;
        var last: u32 = 0;
        var count: u32 = 0;
        while (c.next()) |x| {
            try std.testing.expectEqual(sum, x.row0);
            try std.testing.expect(x.rows >= 1 and x.rows <= 4096);
            sum += x.rows;
            last = x.rows;
            count += 1;
        }
        try std.testing.expectEqual(n, sum);
        try std.testing.expect(count == 1 or last >= min_tail_rows);
    }
    // every chunk but the last starts on the grid and is a multiple of 16 rows, whatever the maximum
    for ([_]u32{ 512, 1000, 2048, 4099 }) |max| for ([_]u64{ 5000, 5009, 5024, 20003, 131071 }) |n| {
        var c: Chunks = .{ .n = n, .max = max };
        while (c.next()) |x| {
            try std.testing.expectEqual(@as(u64, 0), x.row0 % grid_rows);
            if (x.row0 + x.rows < n) try std.testing.expectEqual(@as(u32, 0), x.rows % @as(u32, grid_rows));
            try std.testing.expect(x.rows <= max);
        }
    };
    var small: Chunks = .{ .n = 100000, .max = 1024 };
    try std.testing.expectEqual(@as(u32, 512), small.next().?.rows);
    try std.testing.expectEqual(@as(u32, 1024), small.next().?.rows);
    try std.testing.expectEqual(@as(u32, 1024), small.next().?.rows);
}

test "the consumer runs what has arrived, in passes that leave no stub of the stream" {
    try std.testing.expectEqual(@as(?u32, null), passRows(0, 0, 10000, 4096));
    try std.testing.expectEqual(@as(?u32, 512), passRows(0, 512, 10000, 4096));
    try std.testing.expectEqual(@as(?u32, 4096), passRows(512, 9000, 10000, 4096));
    try std.testing.expectEqual(@as(?u32, null), passRows(10000, 10000, 10000, 4096)); // all of it run
    // the stream's end: a pass of the rest, and not one that leaves 1..16 rows
    try std.testing.expectEqual(@as(?u32, 1000), passRows(9000, 10000, 10000, 4096));
    try std.testing.expectEqual(@as(?u32, 4095), passRows(0, 4100, 4112, 4096)); // 16 would be left: leave 17
    try std.testing.expectEqual(@as(?u32, 4095), passRows(0, 4110, 4112, 8192)); // not all there: 17 stay for the next pass
    try std.testing.expectEqual(@as(?u32, 4112), passRows(0, 4112, 4112, 8192));
    var at: u64 = 0;
    var passes: u32 = 0;
    for ([_]u64{ 4113, 4112, 4100, 8209, 8200, 100000 }) |upto| {
        at = 0;
        while (passRows(at, upto, upto, 4096)) |r| : (passes += 1) {
            at += r;
            const left = upto - at;
            try std.testing.expect(left == 0 or left >= min_tail_rows);
        }
        try std.testing.expectEqual(upto, at);
    }
}

test "a split manifest is the boundary stream first, then layers below the cut and no drafter rings" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const layers = try describe(a, 20000, 14);
    try std.testing.expectEqual(@as(u32, layer_base + 14), layers[0].index);
    try std.testing.expectEqual(@as(u64, 20000), layers[0].rows());
    try std.testing.expectEqual(row_bytes, layers[0].rowBytes());
    try std.testing.expectEqual(@as(?u32, 14), cutOfLayer(layers[0].index));
    var seen: [40]bool = @splat(false);
    for (layers[1..]) |l| {
        try std.testing.expect(l.kind == .recurrent and l.index < 14);
        seen[l.index] = true;
    }
    for (seen[0..14]) |s| try std.testing.expect(s);
    try std.testing.expectError(error.BadShape, describe(a, 20000, 11));
    // the bytes of the state are a prefix-layers share of the whole export's
    const whole = try plan.specs(a, .{ .upto = 20000 });
    const part = try plan.specs(a, stateShape(20000, 14));
    var wb: u64 = 0;
    var pb: u64 = 0;
    for (whole) |s| wb += s.bytes();
    for (part) |s| pb += s.bytes();
    try std.testing.expect(pb > 0 and pb < wb);
    for (part) |s| try std.testing.expect(s.layer < 14 and s.kind != .ring);
    try std.testing.expect(plan.find(stateShape(20000, 14), 14, "swa") == null);
    try std.testing.expect(plan.find(stateShape(20000, 14), 13, "swa") != null);
    try std.testing.expect(plan.find(stateShape(20000, 14), 14, "comp@0") == null);
    var bad = stateShape(20000, 14);
    bad.ds_rows = 128;
    try std.testing.expectError(error.BadShape, bad.check());
}
