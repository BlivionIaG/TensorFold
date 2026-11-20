//! DeepSeek V4.1 steering on Metal (GLP v3 hc_fold_pre_engram): the plan on the device and its two in-place edits.
//!
//! The GLP container format and the weightless runtime concept are Matt Suiche's
//! (https://github.com/msuiche/weightless, docs/spec/GLP.md); directional ablation is Arditi et al., "Refusal in
//! Language Models Is Mediated by a Single Direction" (https://arxiv.org/abs/2406.11717). The plan's shape follows
//! the Qwen 27B steering plan of the same project (upload once, one buffer a covered layer, a layer-to-direction
//! lookup); the kernels are this file's own.
//!
//! A vector's direction N edits backbone block N twice: the block's attention writer (one copy, before its hc_post) and
//! the four-copy stream its FFN fold leaves (before the next block's Engram, or the head after the last block).
//! Nothing here is built when steering is off or the strength is 0: `Plan.init` then returns null.
const std = @import("std");
const mtl = @import("metal");
const sources = @import("kernel_sources");

pub const Mode = enum { add, project };

/// One direction: the backbone block it steers and the fp32 vector (the model's hidden width).
pub const Layer = struct { id: u32, dir: []const f32 };

/// What a plan is built from. The loader's own types are converted by field (`init` takes any struct with these).
pub const Source = struct { mode: Mode, alpha: f32, layers: []const Layer };

/// The model the plan must fit: its hidden width and the backbone blocks (drafter blocks are never steered).
pub const Geometry = struct { hidden: u32, layers: u32 };

pub const threads = 256;

/// The kernels' `TfGlpArgs` record, field for field.
pub const Args = extern struct { dim: u32, rows: u32, alpha: f32, project: u32 };

pub const Pipelines = struct {
    fold: mtl.Pipeline,
    writer: mtl.Pipeline,

    pub fn load(device: mtl.Device) !Pipelines {
        return loadSource(device, sources.dsv41.glp);
    }

    /// The same kernels from other text: the contract's planted controls prefix a TF_CTL define.
    pub fn loadSource(device: mtl.Device, text: []const u8) !Pipelines {
        const lib = try mtl.Library.fromSource(device, text, mtl.CompileOptions.mlx());
        defer lib.deinit();
        const fold = try mtl.Pipeline.init(device, lib, "tf_dsv41_glp_fold", false);
        errdefer fold.deinit();
        return .{ .fold = fold, .writer = try mtl.Pipeline.init(device, lib, "tf_dsv41_glp_writer", false) };
    }

    pub fn deinit(p: *Pipelines) void {
        p.fold.deinit();
        p.writer.deinit();
    }
};

pub const Error = error{ BadSteerShape, BadSteerLayer, BadSteerWidth, BadSteerDirection, BadSteerAlpha, BadSteerMode };

fn modeOf(v: anytype) Error!Mode {
    return std.meta.stringToEnum(Mode, @tagName(v)) orelse error.BadSteerMode;
}

/// The refusals a plan makes at load, before anything is allocated: no direction for block 0 or past the backbone,
/// a direction that is not the hidden width or not finite, a block named twice, a strength not finite or negative.
pub fn validate(src: anytype, geom: Geometry) Error!void {
    _ = try modeOf(src.mode);
    if (geom.hidden == 0 or geom.layers < 2 or src.layers.len == 0) return error.BadSteerShape;
    if (!std.math.isFinite(src.alpha) or src.alpha < 0) return error.BadSteerAlpha;
    for (src.layers, 0..) |l, i| {
        // block 0 has no direction in the format; ids from the drafter's blocks onward are not the backbone's
        if (l.id == 0 or l.id >= geom.layers) return error.BadSteerLayer;
        if (l.dir.len != geom.hidden) return error.BadSteerWidth;
        for (l.dir) |v| if (!std.math.isFinite(v)) return error.BadSteerDirection;
        for (src.layers[0..i]) |p| if (p.id == l.id) return error.BadSteerLayer;
    }
}

/// Device bytes `Plan.init` allocates for a validated source: none at strength 0 (no plan), else every direction in
/// one buffer, rounded up to the 16 KiB page the device allocates in.
pub fn residentBytes(src: anytype, geom: Geometry) usize {
    if (src.alpha == 0) return 0;
    return std.mem.alignForward(usize, src.layers.len * @as(usize, geom.hidden) * 4, 16384);
}

/// The vector in force: every direction in one device buffer, uploaded once and kept for the model's life.
pub const Plan = struct {
    allocator: std.mem.Allocator,
    mode: Mode,
    alpha: f32,
    hidden: u32,
    /// the block's direction row in `buf`, `none` for blocks the vector does not cover
    row: []u32,
    buf: mtl.Buffer,
    pipes: Pipelines,

    const none = std.math.maxInt(u32);

    /// The plan for a vector, or null when its strength is 0 (every edit bypassed, nothing built or dispatched).
    pub fn init(a: std.mem.Allocator, device: mtl.Device, src: anytype, geom: Geometry) !?*Plan {
        try validate(src, geom);
        if (src.alpha == 0) return null;
        const p = try a.create(Plan);
        errdefer a.destroy(p);
        const row = try a.alloc(u32, geom.layers);
        errdefer a.free(row);
        @memset(row, none);
        const buf = try device.buffer(src.layers.len * @as(usize, geom.hidden) * 4, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
        errdefer buf.deinit();
        const dirs = buf.slice(f32, src.layers.len * geom.hidden);
        for (src.layers, 0..) |l, i| {
            @memcpy(dirs[i * geom.hidden ..][0..geom.hidden], l.dir);
            row[l.id] = @intCast(i);
        }
        var pipes = try Pipelines.load(device);
        errdefer pipes.deinit();
        p.* = .{ .allocator = a, .mode = try modeOf(src.mode), .alpha = src.alpha, .hidden = geom.hidden, .row = row, .buf = buf, .pipes = pipes };
        return p;
    }

    pub fn deinit(p: *Plan) void {
        p.pipes.deinit();
        p.buf.deinit();
        p.allocator.free(p.row);
        p.allocator.destroy(p);
    }

    /// The direction buffer joins the model's residency set (the caller commits the set).
    pub fn registerResidency(p: *const Plan, set: mtl.ResidencySet) void {
        set.add(p.buf);
    }

    /// Whether the vector steers backbone block `layer`.
    pub fn covers(p: *const Plan, layer: u32) bool {
        return layer < p.row.len and p.row[layer] != none;
    }

    fn launch(p: *const Plan, e: mtl.ComputeEncoder, pipe: mtl.Pipeline, layer: u32, x: mtl.Buffer, off: usize, rows: u32) void {
        if (rows == 0) return;
        std.debug.assert(p.covers(layer));
        e.setPipeline(pipe);
        e.setBuffer(x, off, 0);
        e.setBuffer(p.buf, @as(usize, p.row[layer]) * p.hidden * 4, 1);
        e.setValue(Args{ .dim = p.hidden, .rows = rows, .alpha = p.alpha, .project = @intFromBool(p.mode == .project) }, 2);
        e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(threads, 1, 1));
    }

    /// Block `layer`'s four-copy stream [rows][4][hidden] bf16 at `off` of `x`, edited in place; the caller barriers.
    pub fn fold(p: *const Plan, e: mtl.ComputeEncoder, layer: u32, x: mtl.Buffer, off: usize, rows: u32) void {
        p.launch(e, p.pipes.fold, layer, x, off, rows);
    }

    /// Block `layer`'s attention writer [rows][hidden] bf16 at `off` of `x`, edited in place; the caller barriers.
    pub fn writer(p: *const Plan, e: mtl.ComputeEncoder, layer: u32, x: mtl.Buffer, off: usize, rows: u32) void {
        p.launch(e, p.pipes.writer, layer, x, off, rows);
    }
};

test "a vector the model cannot take is refused before anything is built" {
    const geom: Geometry = .{ .hidden = 4, .layers = 6 };
    const d = [4]f32{ 1, 0, 0, 0 };
    const good = [_]Layer{.{ .id = 1, .dir = &d }, .{ .id = 5, .dir = &d }};
    try validate(Source{ .mode = .project, .alpha = 1, .layers = &good }, geom);
    const bad = [_]struct { l: []const Layer, a: f32, e: Error }{
        .{ .l = &.{.{ .id = 0, .dir = &d }}, .a = 1, .e = error.BadSteerLayer },
        .{ .l = &.{.{ .id = 6, .dir = &d }}, .a = 1, .e = error.BadSteerLayer },
        .{ .l = &.{ .{ .id = 2, .dir = &d }, .{ .id = 2, .dir = &d } }, .a = 1, .e = error.BadSteerLayer },
        .{ .l = &.{.{ .id = 2, .dir = d[0..3] }}, .a = 1, .e = error.BadSteerWidth },
        .{ .l = &.{.{ .id = 2, .dir = &.{ 1, std.math.nan(f32), 0, 0 } }}, .a = 1, .e = error.BadSteerDirection },
        .{ .l = &good, .a = -1, .e = error.BadSteerAlpha },
        .{ .l = &good, .a = std.math.inf(f32), .e = error.BadSteerAlpha },
        .{ .l = &.{}, .a = 1, .e = error.BadSteerShape },
    };
    for (bad) |b| try std.testing.expectError(b.e, validate(Source{ .mode = .add, .alpha = b.a, .layers = b.l }, geom));
}

test "the loader's own mode and layer types convert by field name" {
    const Other = struct { mode: enum { add, project }, alpha: f32, layers: []const struct { id: u32, dir: []const f32 } };
    const d = [2]f32{ 0, 1 };
    try validate(Other{ .mode = .project, .alpha = 0.5, .layers = &.{.{ .id = 1, .dir = &d }} }, .{ .hidden = 2, .layers = 3 });
}
