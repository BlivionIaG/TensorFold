//! `Lines` over an engine's real expert pools: the decoder layers' and the MTP layer's, in that order.
const std = @import("std");
const prompt = @import("../../cuda_prompt.zig");
const pool_ = @import("../offload/cuda_exl3_pool.zig");
const dplan = @import("../dplan/cuda_exl3_dplan.zig");
const pass = @import("../model/cuda_exl3_pass.zig");
const rates = @import("elastic_rates.zig");
const policy = @import("elastic_policy.zig");
const lines = @import("cuda_exl3_lines.zig");

pub const Engine = struct {
    gpa: std.mem.Allocator,
    e: *prompt.Engine,
    pools: [policy.max_pools]*pool_.Pool = undefined,
    n: usize = 0,
    lines: lines.Lines = undefined,

    /// The engine's elastic pools with `floor` lines at least each; null when it has none (nothing to trade).
    pub fn create(gpa: std.mem.Allocator, e: *prompt.Engine, floor: u32) !?*Engine {
        const x = try gpa.create(Engine);
        errdefer gpa.destroy(x);
        x.* = .{ .gpa = gpa, .e = e };
        for (e.layers[0..e.n_layers]) |*l| if (l.exl3) |ex| if (ex.pool) |p| try x.add(p);
        if (e.exl3_dplan.mtp_pool) |p| try x.add(p);
        if (x.n == 0) {
            gpa.destroy(x);
            return null;
        }
        x.lines = try lines.Lines.init(x.backend(), floor);
        return x;
    }

    fn add(x: *Engine, p: *pool_.Pool) !void {
        if (!p.opt.elastic) return error.ElasticMemoryNeedsBothElasticModes;
        if (x.n == x.pools.len) return error.Invalid;
        x.pools[x.n] = p;
        x.n += 1;
    }

    pub fn destroy(x: *Engine) void {
        x.gpa.destroy(x);
    }

    /// Every pool's cumulative decode counters (the device planner's pending rounds included; the quiescent rank's).
    pub fn sample(x: *Engine, out: *rates.Sample) !void {
        out.n = x.n;
        for (x.pools[0..x.n], out.pools[0..x.n]) |p, *o| {
            const d = if (dplan.current(&x.e.exl3_dplan)) |dp| try dp.samplePoolAfterConsumers(p) else p.telemetry_device;
            o.* = .{ .rounds = d.plans, .hit = d.hit_picks, .miss = d.miss_picks, .copies = d.copies };
        }
    }

    fn backend(x: *Engine) lines.Backend {
        return .{ .ctx = x, .count = x.n, .shape = shape, .set = set, .free = free };
    }

    fn shape(ctx: *anyopaque, i: usize) policy.Line {
        const x: *Engine = @ptrCast(@alignCast(ctx));
        const p = x.pools[i];
        return .{ .active = @intCast(p.active), .floor = 0, .cap = @intCast(p.opt.lines), .bytes = p.lineBytes(), .chunk = p.lines.window.?.win.granularity };
    }

    /// The device planner's `setActive` for a pool it plans (residency comes home first), else the pool's own.
    fn set(ctx: *anyopaque, i: usize, n: u32) anyerror!void {
        const x: *Engine = @ptrCast(@alignCast(ctx));
        if (x.e.exl3_prefill.deep) |dp| if (dp.ring) |r| if (r.live) return error.ElasticRefused; // a prefill owns the mapped lines
        const p = x.pools[i];
        if (dplan.current(&x.e.exl3_dplan)) |dp| if (dp.indexOf(p)) |j| return dp.setActive(j, n);
        try dplan.toHost(&x.e.exl3_dplan);
        return p.setActive(n);
    }

    fn free(ctx: *anyopaque) u64 {
        const x: *Engine = @ptrCast(@alignCast(ctx));
        return pass.freeBytes(x.e.driver) catch 0;
    }
};
