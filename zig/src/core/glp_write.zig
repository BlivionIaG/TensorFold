//! GLPv3 GGUF writer adapted from #1835 c5947a8d2e; format credits are in the GLP recipe.

const std = @import("std");
const gguf = @import("glp_container.zig");
const t = @import("glp_types.zig");
const read = @import("glp_read.zig");
const Error = t.Error;

/// What a producer writes: the vector's fields and one direction per layer.
pub const WriteSpec = struct {
    /// The operation the directions are for; a project vector's directions must be unit vectors.
    mode: t.Mode,
    alpha: f32 = 1.0,
    derived_at: ?[]const u8 = null,
    /// The base repo ("Org/Name", or the hub's URL for it), written as general.base_model.0 name, organization and URL.
    base_model: []const u8 = "",
    /// The base checkpoint's commit, written as general.base_model.0.version: the spec's pin.
    base_revision: []const u8 = "",
    method: []const u8 = "",
    contrast: []const u8 = "",
    /// glp.validation: the named suites and their numbers, as the spec's "Calibrating the dose" asks.
    validation: ?[]const u8 = null,
    /// glp.dose_cos: per-layer L:v pairs, mean |h.d|/|h| over the derivation's capture rows.
    dose_cos: ?[]const u8 = null,
    /// glp.resid_norm_median: per-layer L:v pairs, median |h| of the same rows.
    resid_norm_median: ?[]const u8 = null,
    /// What the file says about itself; the keys a consumer reads before it reads the directions.
    publication: t.Publication = .{},
    layers: []const t.Layer,
};

fn byId(_: void, x: t.Layer, y: t.Layer) bool {
    return x.id < y.id;
}

/// One more metadata key, when the producer named one.
fn appendText(written: *std.ArrayList(gguf.Entry), tmp: std.mem.Allocator, key: []const u8, text: ?[]const u8) !void {
    const value = text orelse return;
    try written.append(tmp, .{ .key = key, .value = .{ .string = value } });
}

/// Write canonical general.base_model.0.* provenance; version holds the commit, never a producer-specific key.
fn appendBase(written: *std.ArrayList(gguf.Entry), tmp: std.mem.Allocator, spec: WriteSpec) !void {
    if (spec.base_model.len == 0 and spec.base_revision.len == 0) return;
    try written.append(tmp, .{ .key = "general.base_model.count", .value = .{ .u32v = 1 } });
    const repo = t.repoId(spec.base_model);
    if (repo.len > 0) {
        if (std.mem.startsWith(u8, repo, "http://") or std.mem.startsWith(u8, repo, "https://")) {
            try written.append(tmp, .{ .key = "general.base_model.0.repo_url", .value = .{ .string = repo } });
        } else if (std.mem.lastIndexOfScalar(u8, repo, '/')) |owner| {
            const url = try std.fmt.allocPrint(tmp, "{s}{s}", .{ t.hub_prefix, repo });
            try written.append(tmp, .{ .key = "general.base_model.0.name", .value = .{ .string = repo[owner + 1 ..] } });
            try written.append(tmp, .{ .key = "general.base_model.0.organization", .value = .{ .string = repo[0..owner] } });
            try written.append(tmp, .{ .key = "general.base_model.0.repo_url", .value = .{ .string = url } });
        } else {
            try written.append(tmp, .{ .key = "general.base_model.0.name", .value = .{ .string = repo } });
        }
    }
    if (spec.base_revision.len > 0) {
        try written.append(tmp, .{ .key = "general.base_model.0.version", .value = .{ .string = spec.base_revision } });
    }
}

/// Write a vector file the way the spec names it; the returned bytes are owned by the caller.
pub fn write(a: std.mem.Allocator, spec: WriteSpec) Error![]u8 {
    if (spec.layers.len == 0) return error.BadGlpVector;
    if (!std.math.isFinite(spec.alpha) or spec.alpha < 0) return error.BadGlpVector;

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const tmp = arena.allocator();

    // The file lists its layers in ascending order, whatever order the caller hands them over in.
    const sorted = try tmp.alloc(t.Layer, spec.layers.len);
    @memcpy(sorted, spec.layers);
    std.mem.sort(t.Layer, sorted, {}, byId);

    const width = sorted[0].dir.len;
    for (sorted, 0..) |l, i| {
        if (i > 0 and l.id == sorted[i - 1].id) return error.BadGlpVector;
        if (l.id == 0 or l.dir.len == 0 or l.dir.len != width) return error.BadGlpVector;
        var n2: f64 = 0;
        for (l.dir) |v| {
            if (!std.math.isFinite(v)) return error.BadGlpVector;
            n2 += @as(f64, v) * v;
        }
        // an additive direction may be zero (a no-op); a projection's must be a unit vector, which a zero one is not
        if (spec.mode == .project and @abs(n2 - 1) > read.unit_tolerance) return error.BadGlpVector;
    }

    var payload: std.ArrayList(u8) = .empty;
    for (sorted) |l| {
        for (l.dir) |v| {
            var buf: [4]u8 = undefined;
            std.mem.writeInt(u32, &buf, @bitCast(v), .little);
            try payload.appendSlice(tmp, &buf);
        }
    }

    // The digest covers the concatenated tensor bytes, which is what the reader hashes to check the file.
    var sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(payload.items, &sha, .{});
    const hex = std.fmt.bytesToHex(sha, .lower);

    var ids: std.ArrayList(u8) = .empty;
    for (sorted, 0..) |l, i| {
        if (i > 0) try ids.append(tmp, ',');
        try ids.print(tmp, "{d}", .{l.id});
    }

    const dims = try tmp.alloc(u64, 1);
    dims[0] = width;
    const names = try tmp.alloc([32]u8, sorted.len);
    const tensors = try tmp.alloc(gguf.OutTensor, sorted.len);
    var start: usize = 0;
    for (sorted, tensors, 0..) |l, *tensor, i| {
        const name = std.fmt.bufPrint(names[i][0..], "direction.{d}", .{l.id}) catch return error.BadGlpVector;
        tensor.* = .{ .name = name, .dims = dims, .kind = gguf.f32_type, .data = payload.items[start..][0 .. l.dir.len * 4] };
        start += l.dir.len * 4;
    }

    const entries = [_]gguf.Entry{
        .{ .key = "general.architecture", .value = .{ .string = "controlvector" } },
        .{ .key = "general.alignment", .value = .{ .u32v = 32 } },
        .{ .key = "controlvector.layer_count", .value = .{ .u32v = @intCast(sorted.len) } },
        .{ .key = "glp.spec_version", .value = .{ .u32v = t.spec_version } },
        .{ .key = "glp.mode", .value = .{ .string = @tagName(spec.mode) } },
        .{ .key = "glp.rank", .value = .{ .u32v = 1 } },
        .{ .key = "glp.alpha_default", .value = .{ .f32v = spec.alpha } },
        .{ .key = "glp.hook_point", .value = .{ .string = t.site_name } },
        .{ .key = "glp.derived_at", .value = .{ .string = spec.derived_at orelse t.site_name } },
        .{ .key = "glp.method", .value = .{ .string = spec.method } },
        .{ .key = "glp.contrast", .value = .{ .string = spec.contrast } },
        .{ .key = "glp.layer_ids_zero_based", .value = .{ .string = ids.items } },
        .{ .key = "glp.content_sha256", .value = .{ .string = hex[0..] } },
    };
    var written: std.ArrayList(gguf.Entry) = .empty;
    try written.appendSlice(tmp, &entries);
    try appendBase(&written, tmp, spec);
    const about = spec.publication;
    try appendText(&written, tmp, "general.name", about.name);
    try appendText(&written, tmp, "general.description", about.description);
    try appendText(&written, tmp, "general.author", about.author);
    try appendText(&written, tmp, "general.license", about.license);
    try appendText(&written, tmp, "controlvector.model_hint", about.model_hint);
    try appendText(&written, tmp, "glp.measured_effect", about.measured_effect);
    try appendText(&written, tmp, "glp.created", about.created);
    // Diagnostics the spec names: informational, never gated, absent unless the producer measured them.
    try appendText(&written, tmp, "glp.validation", spec.validation);
    try appendText(&written, tmp, "glp.dose_cos", spec.dose_cos);
    try appendText(&written, tmp, "glp.resid_norm_median", spec.resid_norm_median);
    return gguf.write(a, written.items, tensors) catch |err| switch (err) {
        error.BadGguf => error.BadGlpVector,
        error.OutOfMemory => error.OutOfMemory,
    };
}

/// Write a vector read earlier back out, so a file can be rewritten with the same fields.
pub fn writeVector(a: std.mem.Allocator, v: t.Vector) Error![]u8 {
    return write(a, .{
        .mode = v.mode,
        .alpha = v.alpha,
        .derived_at = v.derived_at,
        .base_model = v.base_model,
        .base_revision = v.base_revision,
        .method = v.method,
        .contrast = v.contrast,
        .publication = v.publication,
        .layers = v.layers,
    });
}
