//! LoopCD's combine (metal_loop.zig `loopcd`): h' = h_R + w (h_R - h_1), BF16 planes, one rounding.
const std = @import("std");
const mtl = @import("metal");
const loop = @import("tensorfold").inkling2_metal.loop;
const sources = @import("kernel_sources");
const u = @import("inkling2_util.zig");
const rig = @import("rig.zig");
const c = u.c;
const g = u.g;

/// The strength the gates run: an exact binary fraction, so every exact input's value stays exact in fp32.
pub const w: f32 = 0.5;
/// The planes a call walks: the core's banked block planes plus the running state (at most one a pass here).
const max_planes = 4;
const shapes = [_]c.Shape{
    .{ .rows = 1, .n = 2304, .k = 2 },
    .{ .rows = 32, .n = 2304, .k = 1 },
    .{ .rows = 3, .n = 96, .k = 3 },
    .{ .rows = 17, .n = 128, .k = 1 },
};

fn noSums(shape: c.Shape) usize {
    _ = shape;
    return 0;
}

fn depth(shape: c.Shape) usize {
    _ = shape;
    return 2; // the difference and the fused multiply-add, both in fp32
}

const Combine = struct {
    arena: std.heap.ArenaAllocator,
    rig: rig.Rig,
    ops: loop.Ops,
    shape: c.Shape = .{ .rows = 0, .n = 0, .k = 0 },
    /// The final pass's planes, as prepared (the output is restored from them before every run).
    hr: []u16 = &.{},
    h1: []u16 = &.{},
    x: rig.Ref = undefined,
    y: rig.Ref = undefined,
    out: rig.Ref = undefined,

    pub fn count(p: *const Combine) usize {
        return p.shape.rows * p.shape.k * p.shape.n;
    }

    pub fn prepare(p: *Combine, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n == 0 or shape.n > loop.max_hidden or shape.k == 0 or shape.k > max_planes) return false;
        const a = u.reset(&p.arena, &p.rig);
        p.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rnd = prng.random();
        const n = p.count();
        p.hr = try a.alloc(u16, n);
        p.h1 = try a.alloc(u16, n);
        for (p.hr) |*v| v.* = u.uniform(rnd, 4);
        for (p.h1) |*v| v.* = u.uniform(rnd, 4);
        if (fill == .nonfinite) {
            p.hr[0] = u.nan_bits;
            p.h1[1] = u.pinf_bits;
            p.hr[2] = u.ninf_bits;
        }
        p.x = try p.rig.input(std.mem.sliceAsBytes(p.hr));
        p.y = try p.rig.input(std.mem.sliceAsBytes(p.h1));
        p.out = try p.rig.output(n * 2);
        return true;
    }

    pub fn run(p: *Combine) !void {
        // The engine writes over the final pass's plane; each run starts from the prepared values.
        @memcpy(p.rig.bytes(p.out, p.count() * 2), std.mem.sliceAsBytes(p.hr));
        const pass = p.rig.begin();
        try p.ops.loopcd(pass.e, .{
            .final = u.view(p.out),
            .first = u.view(p.y),
            .planes = @intCast(p.shape.k),
            .stride = @intCast(p.shape.rows * p.shape.n),
            .rows = @intCast(p.shape.rows),
            .hidden = @intCast(p.shape.n),
            .w = w,
        });
        try p.rig.finish(pass);
    }

    pub fn reference(p: *Combine, want: []f64, mag: []f64) !void {
        for (0..p.count()) |i| {
            const r = u.val(p.hr[i]);
            const q = u.val(p.h1[i]);
            want[i] = @import("tensorfold").loopcd.combine(w, r, q);
            mag[i] = @abs(r) + @abs(@as(f64, w) * (r - q));
        }
    }

    pub fn simulate(p: *Combine, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = order; // elementwise: every output stands alone
        for (0..p.count()) |i| out[i] = replay(acc, u.val(p.hr[i]), u.val(p.h1[i]));
    }

    fn replay(acc: c.Accum, r: f64, q: f64) f64 {
        return switch (acc) {
            .f32 => blk: {
                const rf: f32 = @floatCast(r);
                const qf: f32 = @floatCast(q);
                break :blk @mulAdd(f32, w, rf - qf, rf);
            },
            .f16 => blk: {
                const rf: f16 = @floatCast(r);
                const qf: f16 = @floatCast(q);
                break :blk @floatCast(@mulAdd(f16, @as(f16, @floatCast(w)), rf - qf, rf));
            },
            .bf16 => blk: {
                const rf: f32 = @floatCast(u.bf(r));
                const qf: f32 = @floatCast(u.bf(q));
                break :blk u.bf(@mulAdd(f32, @as(f32, @floatCast(w)), rf - qf, rf));
            },
            .none => 0,
        };
    }

    const vt = g.hooks(Combine);
    fn kernel(p: *Combine, name: []const u8, ctl: bool) g.Kernel {
        return .{ .ctx = p, .vt = &vt, .control = if (ctl) .shapes else null, .spec = .{
            .name = name,
            .accum = .none,
            .depth = noSums,
            .ops = 1,
            .out = .bf16,
            .in_place = true,
            .tiles = .{ .{}, .{ .size = 256, .step = 256, .min = 256 }, .{} },
            .served = &shapes,
        } };
    }
};

const control: struct { from: []const u8, to: []const u8 } = .{
    .from = "out[o] = bfloat(fma(a.w, r - p, r));",
    .to = "out[o] = bfloat(fma(a.w, r + p, r));",
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    held: std.ArrayList(*Combine) = .empty,

    fn add(s: *Suite, only: ?[]const u8, name: []const u8, control_edit: ?@TypeOf(control)) !void {
        if (only) |filter| if (std.mem.indexOf(u8, name, filter) == null) return;
        const p = try s.a.create(Combine);
        errdefer s.a.destroy(p);
        const ops = if (control_edit) |ctl| blk: {
            const source = try u.mutate(s.a, sources.inkling2_loop_ops, ctl.from, ctl.to);
            defer s.a.free(source);
            break :blk try loop.Ops.loadSource(s.device, .{}, source);
        } else try loop.Ops.load(s.device, .{});
        p.* = .{ .arena = .init(s.a), .rig = .{ .a = s.a, .device = s.device, .queue = s.queue }, .ops = ops };
        errdefer p.arena.deinit();
        // The name's prefix selects this library in the limits tables.
        const k = p.kernel(name, control_edit != null);
        try s.held.ensureUnusedCapacity(s.a, 1);
        try s.kernels.append(s.a, k);
        s.held.appendAssumeCapacity(p);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(only, "inkling2 loopcd combine inkling2_loop_loopcd_combine", null);
        try s.add(only, "inkling2 loopcd combine control: adds the reference inkling2_loop_loopcd_combine", control);
    }

    pub fn deinit(s: *Suite) void {
        for (s.held.items) |p| {
            p.ops.deinit();
            p.rig.deinit();
            p.arena.deinit();
            s.a.destroy(p);
        }
        s.held.deinit(s.a);
    }
};
