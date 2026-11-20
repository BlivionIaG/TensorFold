//! Reading a GLP file: a rank-one add or project vector at the spec's v3 hc fold site, accepted whole or refused.
//! What changes how the edit is applied (version, mode, site, rank, scales, legacy keys) is refused, never ignored.

const std = @import("std");
const gguf = @import("glp_container.zig");
const t = @import("glp_types.zig");
const Why = t.Why;
const Error = t.Error;

/// A project direction's squared norm may differ from 1 by this much (the reference reader's orthonormal bound).
pub const unit_tolerance: f64 = 1e-3;

// Other glp.* keys are informational (the spec names glp.dose_cos and glp.resid_norm_median and gates none of them):
// they are read past, so a producer's extra notes never refuse a file whose mode, site, rank and version are known.

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) Error {
    why.set(fmt, args);
    return error.BadGlpVector;
}

/// The text a key holds, null when the key is absent; a key of another type is refused, not skipped.
fn text(file: gguf.File, key: []const u8, why: *Why) Error!?[]const u8 {
    const v = file.find(key) orelse return null;
    return switch (v) {
        .string => |s| s,
        else => return refuse(why, "{s} is not text", .{key}),
    };
}

/// The unsigned integer a key holds, null when the key is absent; text, a float or a negative number is refused.
fn count(file: gguf.File, key: []const u8, why: *Why) Error!?u64 {
    if (file.find(key) == null) return null;
    return file.num(key) orelse return refuse(why, "{s} is not an unsigned integer", .{key});
}

/// The layer id a tensor name carries: `direction.N` with N written canonically, or a refusal naming why.
fn directionId(name: []const u8, why: *Why) Error!u32 {
    const prefix = "direction.";
    if (!std.mem.startsWith(u8, name, prefix)) {
        return refuse(why, "the file holds the tensor \"{f}\", which is not a direction", .{std.zig.fmtString(name)});
    }
    const tail = name[prefix.len..];
    if (std.mem.indexOfScalar(u8, tail, '.') != null) {
        return refuse(why, "the file holds \"{f}\": a rank above one; this reader implements rank 1", .{std.zig.fmtString(name)});
    }
    for (tail) |ch| if (ch < '0' or ch > '9') return refuse(why, "\"{f}\" does not name a layer", .{std.zig.fmtString(name)});
    if (tail.len == 0 or (tail.len > 1 and tail[0] == '0')) {
        return refuse(why, "\"{f}\" does not name a layer as direction.N", .{std.zig.fmtString(name)});
    }
    return std.fmt.parseInt(u32, tail, 10) catch return refuse(why, "\"{f}\" names a layer past 32 bits", .{std.zig.fmtString(name)});
}

fn norm2(dir: []const f32) f64 {
    var sum: f64 = 0;
    for (dir) |v| sum += @as(f64, v) * v;
    return sum;
}

/// glp.alpha_default as an FP32 strength, held to the rule --glp-alpha is: finite, from 0, and exactly 0 or a normal
/// float. A double FP32 would flush to 0, or a subnormal, is refused: neither may read as off nor as a live strength.
/// 1 when the file names none.
fn alphaDefault(file: gguf.File, why: *Why) Error!f32 {
    const v = file.find("glp.alpha_default") orelse return 1.0;
    const wide: f64 = switch (v) {
        .f32v => |x| x,
        .f64v => |x| x,
        else => return refuse(why, "glp.alpha_default is not a floating-point strength", .{}),
    };
    const alpha = t.strength(wide) orelse
        return refuse(why, "glp.alpha_default is not a finite strength from 0 that FP32 holds: zero, or a normal float", .{});
    return alpha;
}

/// The digest over the tensors' bytes, in the file's order; null when a tensor points past the file.
fn digest(file: gguf.File) ?[32]u8 {
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    for (file.tensors) |tensor| h.update(file.bytesOf(tensor) orelse return null);
    var out: [32]u8 = undefined;
    h.final(&out);
    return out;
}

/// Keys that carry another reading of the file, or an interpretation this reader lacks, are refused here.
fn checkKeys(file: gguf.File, why: *Why) Error!void {
    for (file.entries) |e| {
        if (std.mem.startsWith(u8, e.key, "dspark.")) {
            return refuse(why, "the file carries dspark.* keys: a pre-GLP internal build, to be re-exported as GLP", .{});
        }
        if (std.mem.eql(u8, e.key, "glp.dir_scales") or std.mem.eql(u8, e.key, "glp.layer_scales")) {
            return refuse(why, "the file carries {s}: per-direction and per-layer scales are not implemented", .{e.key});
        }
    }
}

/// The mode and the hook the file declares, once its version, rank and architecture are ones this reader applies.
const Declared = struct { mode: t.Mode, hook: []const u8 };

fn declared(file: gguf.File, why: *Why) Error!Declared {
    const arch = try text(file, "general.architecture", why) orelse return refuse(why, "the file names no general.architecture", .{});
    if (!std.mem.eql(u8, arch, "controlvector")) return refuse(why, "the file's architecture is not controlvector", .{});

    const version = try count(file, "glp.spec_version", why) orelse
        return refuse(why, "the file carries no glp.spec_version; this reader implements version {d}", .{t.spec_version});
    if (version != t.spec_version) {
        return refuse(why, "glp.spec_version {d}; this reader implements version {d}, the hc fold site's", .{ version, t.spec_version });
    }

    // The reference runtime refuses a file with no glp.mode ("refusing to guess"): an add file read as a projection,
    // or the reverse, is another edit, so the mode is required and never defaulted.
    const mode_text = try text(file, "glp.mode", why) orelse return refuse(why, "the file names no glp.mode", .{});
    const mode = std.meta.stringToEnum(t.Mode, mode_text) orelse
        return refuse(why, "glp.mode \"{f}\" is not one this reader applies: add or project", .{std.zig.fmtString(mode_text)});

    const rank = try count(file, "glp.rank", why) orelse 1;
    if (rank != 1) return refuse(why, "glp.rank {d}; this reader implements rank 1", .{rank});
    if (file.find("glp.orthonormal")) |v| if (std.meta.activeTag(v) != .boolv) return refuse(why, "glp.orthonormal is not a boolean", .{});

    const hook = try text(file, "glp.hook_point", why) orelse return refuse(why, "the file names no glp.hook_point", .{});
    if (!std.mem.eql(u8, hook, t.site_name)) {
        return refuse(why, "the hook point \"{f}\" is not applied; this reader applies at {s} only", .{ std.zig.fmtString(hook), t.site_name });
    }
    return .{ .mode = mode, .hook = hook };
}

const Found = struct { layers: []t.Layer, width: usize };

/// The layers the file's tensors carry, ascending; each direction is finite, a project one a nonzero unit vector.
fn directions(a: std.mem.Allocator, file: gguf.File, mode: t.Mode, why: *Why) Error!Found {
    var layers: std.ArrayList(t.Layer) = .empty;
    var width: usize = 0;
    for (file.tensors) |tensor| {
        const id = try directionId(tensor.name, why);
        if (id == 0) return refuse(why, "the file holds direction.0; layer 0 is not steerable", .{});
        if (tensor.kind != gguf.f32_type) return refuse(why, "{s} is not fp32", .{tensor.name});
        if (tensor.dims.len != 1) return refuse(why, "{s} is not one dimensional", .{tensor.name});
        if (width == 0) width = @intCast(tensor.dims[0]);
        if (width != tensor.dims[0]) return refuse(why, "{s} is {d} wide; the other directions are {d}", .{ tensor.name, tensor.dims[0], width });
        const raw = file.bytesOf(tensor) orelse return refuse(why, "{s} points past the file", .{tensor.name});
        const dir = try a.alloc(f32, width);
        for (dir, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        for (dir) |v| if (!std.math.isFinite(v)) return refuse(why, "layer {d}'s direction holds a value that is not finite", .{id});
        // An additive direction may be any finite vector, zero included (a no-op); only a projection needs a unit one.
        if (mode == .project) {
            const n2 = norm2(dir);
            if (n2 == 0) return refuse(why, "layer {d}'s project direction is all zero", .{id});
            if (@abs(n2 - 1) > unit_tolerance) {
                return refuse(why, "layer {d}'s project direction has squared norm {d:.6}; it must be a unit vector, never rescaled", .{ id, n2 });
            }
        }
        try layers.append(a, .{ .id = id, .dir = dir });
    }
    if (layers.items.len == 0) return refuse(why, "the file holds no direction tensors", .{});
    std.mem.sort(t.Layer, layers.items, {}, struct {
        fn lessThan(_: void, x: t.Layer, y: t.Layer) bool {
            return x.id < y.id;
        }
    }.lessThan);
    for (layers.items[1..], 1..) |l, i| {
        if (l.id == layers.items[i - 1].id) return refuse(why, "the file holds two directions for layer {d}", .{l.id});
    }
    return .{ .layers = layers.items, .width = width };
}

/// glp.layer_ids_zero_based, when the file carries it, names exactly the layers the tensors do, in any order.
fn checkLayerIds(a: std.mem.Allocator, file: gguf.File, layers: []const t.Layer, why: *Why) Error!void {
    const list = try text(file, "glp.layer_ids_zero_based", why) orelse return;
    var ids: std.ArrayList(u32) = .empty;
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |part| {
        const id = std.fmt.parseInt(u32, std.mem.trim(u8, part, " "), 10) catch return refuse(why, "the file's layer id list is not a number list", .{});
        try ids.append(a, id);
    }
    if (ids.items.len > layers.len) return refuse(why, "the file names more layers than it holds", .{});
    if (ids.items.len < layers.len) return refuse(why, "the file names {d} layers and holds {d}", .{ ids.items.len, layers.len });
    std.mem.sort(u32, ids.items, {}, std.sort.asc(u32));
    for (ids.items, layers) |id, l| {
        if (id != l.id) return refuse(why, "the file's layer id {d} does not match its tensor's {d}", .{ id, l.id });
    }
}

/// The base checkpoint as the spec's provenance table gives it.
const Base = struct { repo: []const u8 = "", revision: []const u8 = "" };

/// The base checkpoint's repo and commit pin from GGUF's general.base_model.0.* keys, the spec's provenance fields:
/// the repo from repo_url, else organization and name; the pin from version. No producer-specific glp.* key names a
/// base, so none can override the pin. Fields that disagree, or name more than one base, are refused.
fn baseModel(a: std.mem.Allocator, file: gguf.File, why: *Why) Error!Base {
    const prefix = "general.base_model.0.";
    const url = try text(file, prefix ++ "repo_url", why);
    const org = try text(file, prefix ++ "organization", why);
    const name = try text(file, prefix ++ "name", why);
    const version = try text(file, prefix ++ "version", why);
    if (try count(file, "general.base_model.count", why)) |n| {
        if (n > 1) return refuse(why, "the file names {d} base models; a vector is for one checkpoint", .{n});
        if (n == 0 and (url != null or org != null or name != null or version != null)) {
            return refuse(why, "general.base_model.count is 0 but the file names a base model", .{});
        }
    }
    var repo: []const u8 = "";
    if (url) |u| repo = t.repoId(u);
    if (org != null and name != null and org.?.len > 0 and name.?.len > 0) {
        const joined = try std.fmt.allocPrint(a, "{s}/{s}", .{ org.?, name.? });
        if (repo.len == 0) {
            repo = joined;
        } else if (std.mem.startsWith(u8, url.?, t.hub_prefix) and !std.ascii.eqlIgnoreCase(repo, joined)) {
            return refuse(why, "general.base_model.0 names {f} by its URL and {f} by its organization and name", .{ std.zig.fmtString(repo), std.zig.fmtString(joined) });
        }
    } else if (repo.len == 0) {
        // a name with no owner cannot be proven the same repo as an owner's: kept as given, refused on a mismatch
        repo = name orelse "";
    }
    return .{ .repo = repo, .revision = version orelse "" };
}

/// Read a vector whole and accept it, or refuse it with a reason; the strings and layers live in `a`.
pub fn read(a: std.mem.Allocator, bytes: []const u8, why: *Why) Error!t.Vector {
    if (bytes.len < 8) return refuse(why, "the file is too short to be a control vector", .{});
    var problem: gguf.Problem = .{};
    const file = gguf.parse(a, bytes, &problem) catch |err| {
        if (err == error.OutOfMemory) return error.OutOfMemory;
        return refuse(why, "{s}", .{problem.text()});
    };
    try checkKeys(file, why);
    const decl = try declared(file, why);
    const alpha = try alphaDefault(file, why);
    const found = try directions(a, file, decl.mode, why);
    try checkLayerIds(a, file, found.layers, why);

    const sha = digest(file) orelse return refuse(why, "the file's tensors point past its end", .{});
    if (try text(file, "glp.content_sha256", why)) |claimed| {
        const hex = std.fmt.bytesToHex(sha, .lower);
        if (!std.ascii.eqlIgnoreCase(hex[0..], claimed)) return refuse(why, "the file's content digest does not match its vectors", .{});
    }

    const derived = try text(file, "glp.derived_at", why) orelse decl.hook;
    var file_sha256: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &file_sha256, .{});
    const base = try baseModel(a, file, why);
    return .{
        .mode = decl.mode,
        .site = .hc_fold_pre_engram,
        .spec_version = t.spec_version,
        .alpha = alpha,
        .hook_point = decl.hook,
        .derived_at = derived,
        .base_model = base.repo,
        .base_revision = base.revision,
        .method = try text(file, "glp.method", why) orelse "",
        .contrast = try text(file, "glp.contrast", why) orelse "",
        .width = found.width,
        .layers = found.layers,
        .sha256 = sha,
        .file_sha256 = file_sha256,
        .digest_claimed = file.find("glp.content_sha256") != null,
        .transferred = !std.mem.eql(u8, derived, decl.hook),
        .publication = .{
            .name = try text(file, "general.name", why),
            .description = try text(file, "general.description", why),
            .author = try text(file, "general.author", why),
            .license = try text(file, "general.license", why),
            .model_hint = try text(file, "controlvector.model_hint", why),
            .measured_effect = try text(file, "glp.measured_effect", why),
            .created = try text(file, "glp.created", why),
        },
    };
}
