//! Immutable CUDA GLPv3 plan; format: Matt Suiche/weightless, ablation: Arditi et al. (see GLP recipe).

const std = @import("std");
const glp_ops = @import("dsv41_kernels").glp; // the kernels module's own, never a second copy of its file

pub const Mode = glp_ops.Mode;

/// The persistent role holding every steered layer's direction, [layers, width] fp32 (uploaded once at load).
pub const dirs_role = "s.glp.dirs";

pub const Error = error{ BadGlpMode, BadGlpStrength, BadGlpWidth, BadGlpLayers, GlpRank, GlpUnsupported, GlpMismatch } || std.mem.Allocator.Error;

pub const Layer = struct { id: u32, dir: []const f32 };

pub const Plan = struct {
    mode: Mode,
    /// the effective strength: finite and above 0 (0 never makes a plan)
    alpha: f32,
    width: u32,
    /// the SHA-256 of the vector file's exact bytes: what ranks and hand-offs compare
    file_sha256: [32]u8,
    /// ascending by id, each id at least 1; the directions are the source's (it outlives the plan)
    layers: []Layer,

    /// Validate and copy the structural GLP source; exact zero strength returns no plan and allocates nothing.
    pub fn init(gpa: std.mem.Allocator, source: anytype) Error!?Plan {
        const alpha: f32 = source.alpha;
        if (!std.math.isFinite(alpha) or alpha < 0) return error.BadGlpStrength;
        if (alpha == 0) return null;
        const mode: Mode = if (std.mem.eql(u8, @tagName(source.vector.mode), "add")) .add else if (std.mem.eql(u8, @tagName(source.vector.mode), "project")) .project else return error.BadGlpMode;
        const width: usize = source.vector.width;
        if (width == 0 or width % 8 != 0 or width > glp_ops.max_width) return error.BadGlpWidth;
        const src = source.vector.layers;
        if (src.len == 0) return error.BadGlpLayers;
        const layers = try gpa.alloc(Layer, src.len);
        errdefer gpa.free(layers);
        for (src, layers, 0..) |l, *out, i| {
            if (l.id == 0 or (i > 0 and l.id <= layers[i - 1].id)) return error.BadGlpLayers;
            if (l.dir.len != width) return error.BadGlpWidth;
            out.* = .{ .id = l.id, .dir = l.dir };
        }
        return .{ .mode = mode, .alpha = alpha, .width = @intCast(width), .file_sha256 = source.vector.file_sha256, .layers = layers };
    }

    pub fn deinit(p: *Plan, gpa: std.mem.Allocator) void {
        gpa.free(p.layers);
        p.* = undefined;
    }

    /// The model's geometry: the vector is as wide as the stream and steers backbone blocks only.
    pub fn admit(p: *const Plan, backbone_layers: u32, hidden: u32) Error!void {
        if (p.width != hidden) return error.BadGlpWidth;
        for (p.layers) |l| if (l.id >= backbone_layers) return error.BadGlpLayers;
    }

    /// The direction table's role sized to the whole table in the buffer plan `bp` (its views reach only so far).
    pub fn reserve(p: *const Plan, bp: anytype) !void {
        const g = try bp.sizes.getOrPut(bp.a, dirs_role);
        if (!g.found_existing) {
            g.key_ptr.* = try bp.a.dupe(u8, dirs_role);
            g.value_ptr.* = 0;
        }
        g.value_ptr.* = @max(g.value_ptr.*, p.tableBytes());
    }

    /// A copy owned by the arena `a` (the layers and their directions): it outlives what it was built from.
    pub fn clone(p: *const Plan, a: std.mem.Allocator) std.mem.Allocator.Error!Plan {
        const layers = try a.alloc(Layer, p.layers.len);
        for (p.layers, layers) |l, *out| out.* = .{ .id = l.id, .dir = try a.dupe(f32, l.dir) };
        var out = p.*;
        out.layers = layers;
        return out;
    }

    /// The row of `id` in the direction table, null when the vector leaves the layer alone.
    pub fn slot(p: *const Plan, id: u32) ?u32 {
        var lo: usize = 0;
        var hi: usize = p.layers.len;
        while (lo < hi) {
            const mid = lo + (hi - lo) / 2;
            const at = p.layers[mid].id;
            if (at == id) return @intCast(mid);
            if (at < id) lo = mid + 1 else hi = mid;
        }
        return null;
    }

    pub fn has(p: *const Plan, id: u32) bool {
        return p.slot(id) != null;
    }

    /// The direction table's bytes: every layer's row, fp32 little endian, in `layers` order.
    pub fn tableBytes(p: *const Plan) u64 {
        return @as(u64, p.layers.len) * p.width * 4;
    }

    /// The table to upload at load (one owned copy).
    pub fn table(p: *const Plan, gpa: std.mem.Allocator) ![]u8 {
        const out = try gpa.alloc(u8, @intCast(p.tableBytes()));
        for (p.layers, 0..) |l, i| @memcpy(out[i * p.width * 4 ..][0 .. p.width * 4], std.mem.sliceAsBytes(l.dir));
        return out;
    }

    /// This plan's words (see `policyWords`, which the ranks agree on); null: none.
    pub fn words(p: ?*const Plan) [8]i64 {
        const x = p orelse return @splat(0);
        return pack(x.file_sha256, @backingInt(x.mode), x.alpha, x.layers.len);
    }

    /// `checkpoint` bound to this vector's identity: a stored session never restores into another steering.
    pub fn bind(checkpoint: [32]u8, p: ?*const Plan) [32]u8 {
        const x = p orelse return checkpoint;
        var h: std.crypto.hash.sha2.Sha256 = .init(.{});
        h.update(&checkpoint);
        h.update("tf-glp-v3 hc_fold_pre_engram");
        const w = words(x);
        h.update(std.mem.sliceAsBytes(w[1..]));
        var out: [32]u8 = undefined;
        h.final(&out);
        return out;
    }
};

fn pack(sha: [32]u8, mode: i64, alpha: f32, layers: usize) [8]i64 {
    var w: [8]i64 = undefined;
    w[0] = 1;
    for (0..4) |i| w[1 + i] = std.mem.readInt(i64, sha[8 * i ..][0..8], .little);
    w[5] = mode;
    w[6] = @as(i64, @as(u32, @bitCast(if (alpha == 0) @as(f32, 0) else alpha))); // -0 is 0, as in the identity
    w[7] = @intCast(layers);
    return w;
}

/// Bind loaded-policy metadata even at alpha zero, where no GPU plan exists; all-zero words mean no vector.
pub fn policyWords(source: anytype) [8]i64 {
    const s = source orelse return @splat(0);
    const project = std.mem.eql(u8, @tagName(s.vector.mode), "project");
    return pack(s.vector.file_sha256, @intFromBool(project), s.alpha, s.vector.layers.len);
}

const testing = std.testing;

const FakeMode = enum { add, project };
const FakeLayer = struct { id: u32, dir: []const f32 };
fn fake(mode: FakeMode, alpha: f32, layers: []const FakeLayer) struct { alpha: f32, vector: struct { mode: FakeMode, width: usize, layers: []const FakeLayer, file_sha256: [32]u8 } } {
    return .{ .alpha = alpha, .vector = .{ .mode = mode, .width = 8, .layers = layers, .file_sha256 = @splat(7) } };
}

test "plan: alpha 0 makes no plan, a source's layers keep their order, a ragged one is refused" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ok = [_]FakeLayer{ .{ .id = 3, .dir = &d }, .{ .id = 9, .dir = &d } };
    try testing.expect((try Plan.init(testing.allocator, fake(.add, 0, &ok))) == null);
    var p = (try Plan.init(testing.allocator, fake(.project, 0.5, &ok))).?;
    defer p.deinit(testing.allocator);
    try testing.expectEqual(Mode.project, p.mode);
    try testing.expectEqual(@as(?u32, 1), p.slot(9));
    try testing.expectEqual(@as(?u32, null), p.slot(4));
    try testing.expect(p.has(3) and !p.has(0));
    try testing.expectError(error.BadGlpLayers, p.admit(9, 8));
    try testing.expectError(error.BadGlpWidth, p.admit(40, 16));
    try p.admit(10, 8);
    const dup = [_]FakeLayer{ .{ .id = 3, .dir = &d }, .{ .id = 3, .dir = &d } };
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(.add, 1, &dup)));
    const zero = [_]FakeLayer{.{ .id = 0, .dir = &d }};
    try testing.expectError(error.BadGlpLayers, Plan.init(testing.allocator, fake(.add, 1, &zero)));
    const short = [_]FakeLayer{.{ .id = 1, .dir = d[0..4] }};
    try testing.expectError(error.BadGlpWidth, Plan.init(testing.allocator, fake(.add, 1, &short)));
    try testing.expectError(error.BadGlpStrength, Plan.init(testing.allocator, fake(.add, -1, &ok)));
    try testing.expectError(error.BadGlpStrength, Plan.init(testing.allocator, fake(.add, std.math.nan(f32), &ok)));
}

test "plan: the table holds each layer's row, and ranks' words and the stored identity separate mode and strength" {
    const d0 = [_]f32{ 1, 2, 3, 4, 5, 6, 7, 8 };
    const d1 = [_]f32{ 8, 7, 6, 5, 4, 3, 2, 1 };
    const ls = [_]FakeLayer{ .{ .id = 2, .dir = &d0 }, .{ .id = 5, .dir = &d1 } };
    var a = (try Plan.init(testing.allocator, fake(.add, 1, &ls))).?;
    defer a.deinit(testing.allocator);
    var b = (try Plan.init(testing.allocator, fake(.project, 1, &ls))).?;
    defer b.deinit(testing.allocator);
    var c = (try Plan.init(testing.allocator, fake(.add, 2, &ls))).?;
    defer c.deinit(testing.allocator);
    const t = try a.table(testing.allocator);
    defer testing.allocator.free(t);
    try testing.expectEqual(@as(usize, 2 * 8 * 4), t.len);
    try testing.expectEqual(@as(f32, 8), std.mem.bytesAsSlice(f32, t)[8]);
    try testing.expect(!std.mem.eql(i64, &Plan.words(&a), &Plan.words(&b)));
    try testing.expect(!std.mem.eql(i64, &Plan.words(&a), &Plan.words(&c)));
    try testing.expect(!std.mem.eql(i64, &Plan.words(&a), &Plan.words(null)));
    const base: [32]u8 = @splat(1);
    try testing.expectEqual(base, Plan.bind(base, null));
    try testing.expect(!std.mem.eql(u8, &base, &Plan.bind(base, &a)));
    try testing.expect(!std.mem.eql(u8, &Plan.bind(base, &a), &Plan.bind(base, &b)));
}

test "plan: the policy words are the plan's when active, and a strength of 0 is still a policy (a plan of none)" {
    const d = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0 };
    const ls = [_]FakeLayer{.{ .id = 2, .dir = &d }};
    var p = (try Plan.init(testing.allocator, fake(.project, 0.5, &ls))).?;
    defer p.deinit(testing.allocator);
    try testing.expectEqual(Plan.words(&p), policyWords(@as(?@TypeOf(fake(.add, 0, &ls)), fake(.project, 0.5, &ls))));
    const none = policyWords(@as(?@TypeOf(fake(.add, 0, &ls)), null));
    try testing.expectEqual(@as([8]i64, @splat(0)), none);
    const zero = policyWords(@as(?@TypeOf(fake(.add, 0, &ls)), fake(.add, 0, &ls)));
    try testing.expect(!std.mem.eql(i64, &zero, &none));
    try testing.expectEqual(zero, policyWords(@as(?@TypeOf(fake(.add, 0, &ls)), fake(.add, -0.0, &ls))));
}
