//! Native GLPv3 controls from #1835; concepts: Matt Suiche/weightless and Arditi et al. (see GLP recipe).

const std = @import("std");
const t = @import("glp_types.zig");

pub const Why = t.Why;
pub const Error = t.Error;
pub const Mode = t.Mode;
pub const Site = t.Site;
pub const Layer = t.Layer;
pub const Publication = t.Publication;
pub const Vector = t.Vector;
pub const Glp = t.Glp;
pub const spec_version = t.spec_version;
pub const site_name = t.site_name;
pub const read = @import("glp_read.zig").read;
pub const unit_tolerance = @import("glp_read.zig").unit_tolerance;
pub const WriteSpec = @import("glp_write.zig").WriteSpec;
pub const write = @import("glp_write.zig").write;
pub const writeVector = @import("glp_write.zig").writeVector;

/// The largest finite BF16 magnitude: an add edit past it would turn into an infinity at the stream's storage.
const bf16_max: f64 = 3.3895313892515355e38;

/// What the checkpoint says about itself, as far as it says: null or empty where it does not.
pub const Model = struct {
    /// The text model's hidden size.
    hidden: ?usize = null,
    /// The backbone's block count: the DSpark/MTP blocks that follow it are never steerable.
    layers: ?usize = null,
    /// Exact hub repo ID (Org/Name), empty for local conversions; forks and similarly named repos are not aliases.
    repo: []const u8 = "",
    /// The commit the checkpoint's snapshot is, empty when it is not a hub snapshot.
    revision: []const u8 = "",
};

fn refuse(why: *Why, comptime fmt: []const u8, args: anytype) Error {
    why.set(fmt, args);
    return error.BadGlpVector;
}

/// Check the vector against the model it would steer: its width, its layers and the checkpoint it came from.
pub fn checkModel(v: Vector, model: Model, why: *Why) Error!void {
    if (model.hidden) |hidden| {
        if (v.width != hidden) return refuse(why, "the vector's directions are {d} wide; the model is {d}", .{ v.width, hidden });
    }
    if (model.layers) |depth| {
        if (v.layers.len == 0) return refuse(why, "the vector holds no layers", .{});
        // direction.N steers backbone block N; the ids ascend, so the last is the deepest.
        const deepest = v.layers[v.layers.len - 1].id;
        if (deepest >= depth) {
            return refuse(why, "the vector steers layer {d} of a model with {d} backbone layers: its deepest steerable layer is {d}", .{ deepest, depth, depth -| 1 });
        }
    }
    if (model.repo.len == 0 or v.base_model.len == 0) return;
    const base = t.repoId(v.base_model);
    if (!std.ascii.eqlIgnoreCase(base, model.repo)) return refuse(why, "the vector is for {s}; the model is {s}", .{ base, model.repo });
    if (v.base_revision.len == 0 or model.revision.len == 0) return;
    if (!revisionMatches(v.base_revision, model.revision)) {
        return refuse(why, "the vector is for revision {s} of {s}; the model is revision {s}", .{ v.base_revision, base, model.revision });
    }
}

/// The strength must keep an additive edit inside what BF16 can hold; a projection's size depends on the stream.
pub fn checkStrength(v: Vector, alpha: f32, why: *Why) Error!void {
    if (!std.math.isFinite(alpha) or alpha < 0) return refuse(why, "the strength is not finite and from 0", .{});
    if (v.mode != .add) return;
    for (v.layers) |l| {
        var top: f64 = 0;
        for (l.dir) |x| top = @max(top, @abs(@as(f64, x)));
        if (top * alpha > bf16_max) return refuse(why, "layer {d}'s direction times the strength is past what BF16 holds", .{l.id});
    }
}

/// An FP32 strength from a number the operator or a file gave: finite, from 0, exactly 0 or a normal float.
pub const strength = t.strength;

/// The bytes a vector file may hold; a 40-layer 5120-wide vector is about 0.8 MiB.
pub const file_limit: usize = 64 << 20;

/// Read a vector file whole and accept it; its strings and layers live in `a`, which must outlive the vector.
pub fn readFile(a: std.mem.Allocator, io: std.Io, path: []const u8, why: *Why) Error!Vector {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(file_limit)) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return refuse(why, "cannot read '{s}': {s}", .{ path, readReason(e) });
    };
    return read(a, bytes, why);
}

/// Load immutable file data with explicit or file-default strength; checkModel supplies checkpoint admission.
pub fn load(a: std.mem.Allocator, io: std.Io, path: []const u8, alpha: ?f32, why: *Why) Error!Glp {
    const vector = try readFile(a, io, path, why);
    const effective: f32 = if (alpha) |x| strength(x) orelse return refuse(why, "the strength is not finite and from 0", .{}) else vector.alpha;
    try checkStrength(vector, effective, why);
    return .{ .vector = vector, .alpha = effective };
}

/// Why a file could not be read, in words the operator reads.
fn readReason(e: anyerror) []const u8 {
    return switch (e) {
        error.FileNotFound => "no such file",
        error.AccessDenied => "permission denied",
        error.IsDir => "the path is a directory",
        error.BadPathName => "the path is not a file name",
        error.NameTooLong => "the path is too long",
        error.StreamTooLong => "the file is over the size limit",
        else => "the file cannot be read",
    };
}

/// The shortest commit name that may stand for a longer one: git's own abbreviation floor.
const short_commit = 7;

/// Two commit names agree when they are equal, or when one is the other cut short at no fewer than seven characters.
fn revisionMatches(a: []const u8, b: []const u8) bool {
    if (std.ascii.eqlIgnoreCase(a, b)) return true;
    if (@min(a.len, b.len) < short_commit) return false;
    return std.ascii.startsWithIgnoreCase(a, b) or std.ascii.startsWithIgnoreCase(b, a);
}

/// Preserve plain checkpoint IDs; otherwise bind file SHA-256, mode, hook and effective FP32 strength.
pub fn identity(a: std.mem.Allocator, checkpoint: []const u8, g: ?Glp) std.mem.Allocator.Error![]const u8 {
    const steer = g orelse return checkpoint;
    var h: std.crypto.hash.sha2.Sha256 = .init(.{});
    h.update("tensorfold.glp.identity.v1\x00");
    h.update(&steer.vector.file_sha256);
    h.update(@tagName(steer.vector.mode));
    h.update("\x00");
    h.update(@tagName(steer.vector.site));
    h.update("\x00");
    const alpha: f32 = if (steer.alpha == 0) 0 else steer.alpha;
    var bits: [4]u8 = undefined;
    std.mem.writeInt(u32, &bits, @bitCast(alpha), .little);
    h.update(&bits);
    var tag: [32]u8 = undefined;
    h.final(&tag);
    return std.fmt.allocPrint(a, "{s}+glp-{x}", .{ checkpoint, tag[0..16] });
}

test {
    _ = @import("glp_test.zig");
    _ = @import("glp_model_test.zig");
    _ = @import("glp_publication_test.zig");
}
