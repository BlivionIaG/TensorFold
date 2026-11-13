//! placement.json (schema tensorfold-split/1): who owns each layer, embedding and head, timeouts, the prediction.
const std = @import("std");
const strict = @import("strict.zig");

const Allocator = std.mem.Allocator;
const Value = std.json.Value;

pub const schema = "tensorfold-split/1";
const schema_family = "tensorfold-split/";

pub const Diag = strict.Diag;

pub const Error = error{
    /// `schema` is not a tensorfold-split schema.
    UnknownSchema,
    /// A tensorfold-split schema this build does not read.
    UnsupportedVersion,
    /// A layer owner (or embedding/head) other than "metal" or "external".
    UnknownOwner,
    /// Two compact ranges claim the same layer.
    OverlappingRanges,
    /// The compact ranges leave a layer below the highest one unowned.
    GapInRanges,
    /// A compact range is not `[start, end]` with start <= end.
    InvalidRange,
    /// The placement covers a different number of layers than the model has.
    LayerCountMismatch,
    /// `expert_layers` names a layer the placement does not have, or is not strictly ascending.
    ExpertLayersInvalid,
} || strict.Error;

pub const Owner = enum { metal, external };
pub const Prefill = enum { metal, external, split };
pub const Fallback = enum { none, metal };

/// What the planner chose; the words are the ones `tensorfold info` prints.
pub const Mode = enum { all_metal, resident, layer_split, layer_split_pipelined, expert_split, reverse_expert_split };

pub const Device = struct {
    /// The external backend's name (hip, cuda, mock).
    backend: []const u8,
    index: u32,
    /// Size of the host window in MiB (64 is the hipmac block minimum).
    window_mib: u32,
};

/// Every wait is bounded (milliseconds); on expiry the request fails with a named error.
pub const Timeouts = struct {
    /// One crossing: the peer must answer within this.
    hop: u32,
    /// One whole decode step or prefill chunk.
    step: u32,
};

pub const Prediction = struct {
    mode: Mode,
    decode_tok_s: f64,
    streams: u32,
    context: u32,
};

pub const Placement = struct {
    device: Device,
    /// The owner of each layer, in order.
    layers: []const Owner,
    /// Layers whose routed experts are on the external device, strictly ascending (expert split).
    expert_layers: []const u32,
    embedding: Owner,
    head: Owner,
    prefill: Prefill,
    fallback: Fallback,
    timeouts_ms: Timeouts,
    prediction: ?Prediction,

    /// Layers owned by `o`.
    pub fn count(p: Placement, o: Owner) usize {
        var n: usize = 0;
        for (p.layers) |l| n += @intFromBool(l == o);
        return n;
    }
};

/// Everything checkable without the model: positive window and timeouts, ascending expert layers, sane prediction.
pub fn validate(p: Placement, model_layers: usize) Error!void {
    if (p.layers.len == 0 or p.layers.len != model_layers) return error.LayerCountMismatch;
    if (p.device.backend.len == 0 or p.device.window_mib == 0) return error.InvalidValue;
    if (p.timeouts_ms.hop == 0 or p.timeouts_ms.step == 0) return error.InvalidValue;
    var prev: ?u32 = null;
    for (p.expert_layers) |l| {
        if (l >= p.layers.len) return error.ExpertLayersInvalid;
        if (prev) |q| if (l <= q) return error.ExpertLayersInvalid;
        prev = l;
    }
    if (p.prediction) |pr| {
        if (!(pr.decode_tok_s >= 0) or pr.streams == 0 or pr.context == 0) return error.InvalidValue;
    }
}

/// The `[start, end)` range of layers owned by `o` when contiguous, else null (no layer: an empty range at 0).
fn rangeOf(layers: []const Owner, o: Owner) ?[2]u32 {
    var start: ?u32 = null;
    var end: u32 = 0;
    for (layers, 0..) |l, i| {
        if (l != o) continue;
        const idx: u32 = @intCast(i);
        if (start == null) start = idx else if (end != idx) return null;
        end = idx + 1;
    }
    return .{ start orelse 0, end };
}

/// Pretty JSON: layers as `{"external": [0, 14], "metal": [14, 40]}` when each owner is contiguous, else a list.
pub fn write(gpa: Allocator, p: Placement) Allocator.Error![]u8 {
    var aw: std.Io.Writer.Allocating = .init(gpa);
    defer aw.deinit();
    writeJson(p, &aw.writer) catch return error.OutOfMemory;
    return aw.toOwnedSlice();
}

fn writeJson(p: Placement, w: *std.Io.Writer) !void {
    var jw: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_2 } };
    try jw.beginObject();
    try jw.objectField("schema");
    try jw.write(schema);
    try jw.objectField("device");
    try jw.write(p.device);
    try jw.objectField("layers");
    const ext = rangeOf(p.layers, .external);
    const met = rangeOf(p.layers, .metal);
    if (ext != null and met != null and ext.?[1] - ext.?[0] + met.?[1] - met.?[0] == p.layers.len) {
        try jw.beginObject();
        if (ext.?[1] > ext.?[0]) {
            try jw.objectField("external");
            try jw.print("[{d}, {d}]", .{ ext.?[0], ext.?[1] });
        }
        if (met.?[1] > met.?[0]) {
            try jw.objectField("metal");
            try jw.print("[{d}, {d}]", .{ met.?[0], met.?[1] });
        }
        try jw.endObject();
    } else {
        try jw.beginWriteRaw();
        try w.writeByte('[');
        for (p.layers, 0..) |l, i| try w.print("{s}\"{s}\"", .{ if (i == 0) "" else ", ", @tagName(l) });
        try w.writeByte(']');
        jw.endWriteRaw();
    }
    try jw.objectField("expert_layers");
    try jw.beginWriteRaw();
    try w.writeByte('[');
    for (p.expert_layers, 0..) |l, i| try w.print("{s}{d}", .{ if (i == 0) "" else ", ", l });
    try w.writeByte(']');
    jw.endWriteRaw();
    try jw.objectField("embedding");
    try jw.write(p.embedding);
    try jw.objectField("head");
    try jw.write(p.head);
    try jw.objectField("prefill");
    try jw.write(p.prefill);
    try jw.objectField("fallback");
    try jw.write(p.fallback);
    try jw.objectField("timeouts_ms");
    try jw.write(p.timeouts_ms);
    if (p.prediction) |pr| {
        try jw.objectField("prediction");
        try jw.write(pr);
    }
    try jw.endObject();
}

/// Strict parse and `validate` against the file's own layer count. Strings and slices live in the result's arena.
pub fn parse(gpa: Allocator, bytes: []const u8) Error!std.json.Parsed(Placement) {
    var diag: Diag = .{};
    return parseDiag(gpa, bytes, &diag);
}

/// As `parse`; on failure `diag.path()` names the offending member.
pub fn parseDiag(gpa: Allocator, bytes: []const u8, diag: *Diag) Error!std.json.Parsed(Placement) {
    const arena = try gpa.create(std.heap.ArenaAllocator);
    errdefer gpa.destroy(arena);
    arena.* = .init(gpa);
    errdefer arena.deinit();
    const a = arena.allocator();

    const root = try strict.parseDocument(a, bytes);
    var obj = try strict.object(root, "", diag);
    const sv = try strict.member(obj, "schema", "", diag);
    if (sv != .string) return strict.fail(diag, "schema", error.WrongType);
    if (!std.mem.eql(u8, sv.string, schema)) {
        if (std.mem.startsWith(u8, sv.string, schema_family)) return strict.fail(diag, "schema", error.UnsupportedVersion);
        return strict.fail(diag, "schema", error.UnknownSchema);
    }
    // Members with their own grammar are read here; the rest go through the generic strict reader.
    const layers = try parseLayers(a, try strict.member(obj, "layers", "", diag), diag);
    const embedding = try parseOwner(try strict.member(obj, "embedding", "", diag), "embedding", diag);
    const head = try parseOwner(try strict.member(obj, "head", "", diag), "head", diag);
    for ([_][]const u8{ "schema", "layers", "embedding", "head" }) |k| _ = obj.orderedRemove(k);
    const Rest = struct {
        device: Device,
        expert_layers: []const u32,
        prefill: Prefill,
        fallback: Fallback,
        timeouts_ms: Timeouts,
        prediction: ?Prediction,
    };
    const rest = try strict.parseValue(Rest, a, .{ .object = obj }, "", diag);
    const p: Placement = .{
        .device = rest.device,
        .layers = layers,
        .expert_layers = rest.expert_layers,
        .embedding = embedding,
        .head = head,
        .prefill = rest.prefill,
        .fallback = rest.fallback,
        .timeouts_ms = rest.timeouts_ms,
        .prediction = rest.prediction,
    };
    try validate(p, layers.len);
    return .{ .arena = arena, .value = p };
}

fn parseOwner(v: Value, path: []const u8, diag: *Diag) Error!Owner {
    if (v != .string) return strict.fail(diag, path, error.WrongType);
    return std.meta.stringToEnum(Owner, v.string) orelse strict.fail(diag, path, error.UnknownOwner);
}

fn parseLayers(a: Allocator, v: Value, diag: *Diag) Error![]const Owner {
    switch (v) {
        .array => |arr| {
            if (arr.items.len == 0) return strict.fail(diag, "layers", error.InvalidValue);
            const out = try a.alloc(Owner, arr.items.len);
            for (out, arr.items, 0..) |*o, item, i| {
                var buf: [64]u8 = undefined;
                o.* = try parseOwner(item, std.fmt.bufPrint(&buf, "layers[{d}]", .{i}) catch "layers", diag);
            }
            return out;
        },
        .object => |obj| return parseRanges(a, obj, diag),
        else => return strict.fail(diag, "layers", error.WrongType),
    }
}

/// The compact form: each key an owner, each value `[start, end]` (end exclusive).
fn parseRanges(a: Allocator, obj: std.json.ObjectMap, diag: *Diag) Error![]const Owner {
    const Range = struct { owner: Owner, start: usize, end: usize };
    var ranges: [2]Range = undefined;
    var n: usize = 0;
    var total: usize = 0;
    var it = obj.iterator();
    while (it.next()) |kv| {
        var buf: [96]u8 = undefined;
        const path = strict.join(&buf, "layers", kv.key_ptr.*);
        const owner = std.meta.stringToEnum(Owner, kv.key_ptr.*) orelse return strict.fail(diag, path, error.UnknownOwner);
        const pair = kv.value_ptr.*;
        if (pair != .array) return strict.fail(diag, path, error.WrongType);
        if (pair.array.items.len != 2) return strict.fail(diag, path, error.InvalidRange);
        const lo = pair.array.items[0];
        const hi = pair.array.items[1];
        if (lo != .integer or hi != .integer) return strict.fail(diag, path, error.WrongType);
        if (lo.integer < 0 or hi.integer < lo.integer) return strict.fail(diag, path, error.InvalidRange);
        ranges[n] = .{ .owner = owner, .start = @intCast(lo.integer), .end = @intCast(hi.integer) };
        n += 1;
        total = @max(total, @as(usize, @intCast(hi.integer)));
    }
    if (total == 0) return strict.fail(diag, "layers", error.InvalidValue);
    // the layer list is as long as the highest end; refuse an absurd one before allocating
    if (total > 1 << 20) return strict.fail(diag, "layers", error.InvalidRange);
    const out = try a.alloc(Owner, total);
    const claimed = try a.alloc(bool, total);
    @memset(claimed, false);
    for (ranges[0..n]) |r| for (r.start..r.end) |i| {
        if (claimed[i]) return strict.fail(diag, "layers", error.OverlappingRanges);
        claimed[i] = true;
        out[i] = r.owner;
    };
    for (claimed) |c| if (!c) return strict.fail(diag, "layers", error.GapInRanges);
    return out;
}

const example =
    \\{
    \\  "schema": "tensorfold-split/1",
    \\  "device": { "backend": "hip", "index": 0, "window_mib": 64 },
    \\  "layers": { "external": [0, 14], "metal": [14, 40] },
    \\  "expert_layers": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    \\  "embedding": "external",
    \\  "head": "metal",
    \\  "prefill": "metal",
    \\  "fallback": "none",
    \\  "timeouts_ms": { "hop": 50, "step": 2000 },
    \\  "prediction": { "mode": "layer_split_pipelined", "decode_tok_s": 131.2, "streams": 2, "context": 8192 }
    \\}
;

test "the compact and the per-layer form parse to the same placement" {
    const gpa = std.testing.allocator;
    const compact = try parse(gpa, example);
    defer compact.deinit();
    const p = compact.value;
    try std.testing.expectEqual(@as(usize, 40), p.layers.len);
    try std.testing.expectEqual(@as(usize, 14), p.count(.external));
    try std.testing.expectEqual(Owner.external, p.layers[13]);
    try std.testing.expectEqual(Owner.metal, p.layers[14]);
    try std.testing.expectEqual(@as(usize, 14), p.expert_layers.len);
    try std.testing.expectEqual(Owner.external, p.embedding);
    try std.testing.expectEqual(Owner.metal, p.head);
    try std.testing.expectEqual(Mode.layer_split_pipelined, p.prediction.?.mode);

    var list: std.ArrayList(u8) = .empty;
    defer list.deinit(gpa);
    for (0..40) |i| try list.print(gpa, "{s}\"{s}\"", .{ if (i == 0) "" else ", ", if (i < 14) "external" else "metal" });
    const text = try std.fmt.allocPrint(gpa, "{{\"schema\":\"tensorfold-split/1\",\"device\":{{\"backend\":\"hip\",\"index\":0,\"window_mib\":64}},\"layers\":[{s}]," ++
        "\"expert_layers\":[],\"embedding\":\"external\",\"head\":\"metal\",\"prefill\":\"metal\",\"fallback\":\"none\",\"timeouts_ms\":{{\"hop\":50,\"step\":2000}}}}", .{list.items});
    defer gpa.free(text);
    const listed = try parse(gpa, text);
    defer listed.deinit();
    try std.testing.expectEqualSlices(Owner, p.layers, listed.value.layers);
    try std.testing.expectEqual(@as(?Prediction, null), listed.value.prediction);
}

test "write emits the compact form when it can and round-trips" {
    const gpa = std.testing.allocator;
    const first = try parse(gpa, example);
    defer first.deinit();
    const text = try write(gpa, first.value);
    defer gpa.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"external\": [0, 14]") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "\"metal\": [14, 40]") != null);
    const second = try parse(gpa, text);
    defer second.deinit();
    const a = first.value;
    const b = second.value;
    try std.testing.expectEqualSlices(Owner, a.layers, b.layers);
    try std.testing.expectEqualSlices(u32, a.expert_layers, b.expert_layers);
    try std.testing.expectEqualStrings(a.device.backend, b.device.backend);
    try std.testing.expectEqual(a.device.window_mib, b.device.window_mib);
    try std.testing.expectEqual(a.timeouts_ms, b.timeouts_ms);
    try std.testing.expectEqual(a.prediction.?, b.prediction.?);
    try std.testing.expectEqual(a.prefill, b.prefill);
    const again = try write(gpa, b);
    defer gpa.free(again);
    try std.testing.expectEqualStrings(text, again);
}

test "an interleaved placement is written as a list and a single-owner one as one range" {
    const gpa = std.testing.allocator;
    const layers = [_]Owner{ .metal, .external, .metal, .external };
    var p: Placement = .{
        .device = .{ .backend = "mock", .index = 1, .window_mib = 64 },
        .layers = &layers,
        .expert_layers = &.{ 1, 3 },
        .embedding = .metal,
        .head = .metal,
        .prefill = .split,
        .fallback = .metal,
        .timeouts_ms = .{ .hop = 10, .step = 100 },
        .prediction = null,
    };
    const text = try write(gpa, p);
    defer gpa.free(text);
    try std.testing.expect(std.mem.indexOf(u8, text, "[\"metal\", \"external\", \"metal\", \"external\"]") != null);
    const back = try parse(gpa, text);
    defer back.deinit();
    try std.testing.expectEqualSlices(Owner, &layers, back.value.layers);
    try std.testing.expectEqual(Fallback.metal, back.value.fallback);
    try std.testing.expectEqual(@as(u32, 1), back.value.device.index);

    p.layers = &.{ .metal, .metal, .metal };
    p.expert_layers = &.{};
    const only = try write(gpa, p);
    defer gpa.free(only);
    try std.testing.expect(std.mem.indexOf(u8, only, "\"metal\": [0, 3]") != null);
    try std.testing.expect(std.mem.indexOf(u8, only, "\"external\"") == null);
    const only_back = try parse(gpa, only);
    defer only_back.deinit();
    try std.testing.expectEqual(@as(usize, 3), only_back.value.count(.metal));
}

fn expectBad(gpa: Allocator, from: []const u8, to: []const u8, err: Error, path: []const u8) !void {
    const text = try std.mem.replaceOwned(u8, gpa, example, from, to);
    defer gpa.free(text);
    try std.testing.expect(!std.mem.eql(u8, text, example));
    var diag: Diag = .{};
    try std.testing.expectError(err, parseDiag(gpa, text, &diag));
    try std.testing.expectEqualStrings(path, diag.path());
}

test "planted-bad placement files get named errors" {
    const gpa = std.testing.allocator;
    const ext = "\"external\": [0, 14]";
    const met = "\"metal\": [14, 40]";
    try expectBad(gpa, "tensorfold-split/1", "tensorfold-cluster/1", error.UnknownSchema, "schema");
    try expectBad(gpa, "tensorfold-split/1", "tensorfold-split/2", error.UnsupportedVersion, "schema");
    try expectBad(gpa, ext, "\"external\": [0, 15]", error.OverlappingRanges, "layers");
    try expectBad(gpa, ext, "\"external\": [0, 13]", error.GapInRanges, "layers");
    try expectBad(gpa, ext, "\"external\": [5, 3]", error.InvalidRange, "layers.external");
    try expectBad(gpa, ext, "\"external\": [0]", error.InvalidRange, "layers.external");
    try expectBad(gpa, ext, "\"gpu\": [0, 14]", error.UnknownOwner, "layers.gpu");
    try expectBad(gpa, met, "\"metal\": \"14-40\"", error.WrongType, "layers.metal");
    try expectBad(gpa, "\"embedding\": \"external\"", "\"embedding\": \"gpu\"", error.UnknownOwner, "embedding");
    try expectBad(gpa, "\"head\": \"metal\"", "\"head\": 1", error.WrongType, "head");
    try expectBad(gpa, "\"prefill\": \"metal\"", "\"prefill\": \"both\"", error.InvalidValue, "prefill");
    try expectBad(gpa, "\"fallback\": \"none\"", "\"fallback\": \"cpu\"", error.InvalidValue, "fallback");
    try expectBad(gpa, "\"hop\": 50, ", "", error.MissingField, "timeouts_ms.hop");
    try expectBad(gpa, "\"hop\": 50", "\"hop\": 0", error.InvalidValue, "");
    try expectBad(gpa, "\"index\": 0", "\"index\": 0, \"extra\": true", error.UnknownField, "device.extra");
    try expectBad(gpa, "\"window_mib\": 64", "\"window_mib\": -1", error.InvalidValue, "device.window_mib");
    try expectBad(gpa, "\"expert_layers\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]", "\"expert_layers\": [0, 1, 1]", error.ExpertLayersInvalid, "");
    try expectBad(gpa, "\"expert_layers\": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]", "\"expert_layers\": [40]", error.ExpertLayersInvalid, "");
    try expectBad(gpa, "\"mode\": \"layer_split_pipelined\"", "\"mode\": \"turbo\"", error.InvalidValue, "prediction.mode");
    try expectBad(gpa, "\"streams\": 2", "\"streams\": 0", error.InvalidValue, "");
    try expectBad(gpa, "\"fallback\": \"none\",", "", error.MissingField, "fallback");
    var diag: Diag = .{};
    try std.testing.expectError(error.InvalidJson, parseDiag(gpa, "{", &diag));
    try std.testing.expectError(error.WrongType, parseDiag(gpa, "[]", &diag));
    try std.testing.expectError(error.WrongType, parseDiag(gpa, "{\"schema\": 3}", &diag));
}

test "validate checks the layer count against the model" {
    const gpa = std.testing.allocator;
    const parsed = try parse(gpa, example);
    defer parsed.deinit();
    try validate(parsed.value, 40);
    try std.testing.expectError(error.LayerCountMismatch, validate(parsed.value, 32));
    try std.testing.expectError(error.LayerCountMismatch, validate(parsed.value, 41));
}
