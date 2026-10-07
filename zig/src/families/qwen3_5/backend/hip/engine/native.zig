//! Qwen3.5 / Qwen3.6 for the native server on HIP: the engine sized to GPU memory and the lane backend over it.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const Engine = @import("engine.zig").Engine;
const memory = @import("memory.zig");
const worker = @import("worker.zig");
const hip_lanes = @import("hip_lanes.zig");

/// The MLX affine widths the family loads.
const mlx_formats: []const []const u8 = &.{ "mlx-q2", "mlx-q3", "mlx-q4", "mlx-q5", "mlx-q6", "mlx-q8" };

/// The dense and the sparse checkpoints are two registry entries over one engine.
pub fn Native(comptime type_name: []const u8) type {
    return struct {
        pub const model_type = type_name;
        pub const formats = mlx_formats;
        /// Positions a stream holds when neither --context nor the checkpoint's config names a window.
        pub const default_context: i64 = 32768;
        /// The engine cuts prompts itself.
        pub const prefill_step: u32 = 0;
        pub const open = openEngine;
        pub const follow = followRank;
    };
}

/// What the serve flags ask of the engine, as the host read them.
pub const Options = struct {
    /// The window asked for, or the model's: the loaded window is below it when the memory does not fit it.
    window: usize,
    /// --context named it: a window the memory does not fit is refused, not fitted down.
    fixed: bool = false,
    /// Requests served at once.
    streams: usize,
    /// This process is `rank` of `world`.
    rank: u32 = 0,
    world: u32 = 1,
    /// Bytes kept prompt states may hold (null: the policy's, else the plan's) and their snapshot slots.
    cache_gib: ?f64 = null,
    keep: ?i64 = null,
};

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    /// The window the engine fits, and whether it is below the one asked for.
    window: usize,
    fitted: bool,
};

/// A flag or a resource the engine cannot take: `problem` says why.
pub const Refused = error{Refused};

fn refuse(a: std.mem.Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// One rank's loaded and sized engine: the window and prompt cache every rank agreed on.
const Prepared = struct {
    e: *Engine,
    plan: memory.Plan,
    /// Snapshots the prompt cache may hold (the plan's `cache_budget` is 0 when the cache is off).
    keep: usize,
    /// The pages of the KV pool and the share of them the prompt cache may hold.
    pool: memory.Pool,
};

/// Loads this rank's share of the model, plans the memory with the other ranks and sizes the scratch.
fn prepare(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) (Refused || std.mem.Allocator.Error)!Prepared {
    const rows = hip_lanes.Hip.max_window;
    const e = Engine.load(gpa, io, dir, .{
        .batch_rows = @max(32, @min(o.streams * rows, 128)),
        .slack = rows + 1,
        .device = dev.index,
        .policy = dev.policy,
        .rank = o.rank,
        .world = o.world,
        .id = if (dev.group) |g| g.id else null,
    }) catch |err| return refuse(a, problem, "the native HIP engine cannot load {s} ({s})", .{ dir, @errorName(err) });
    errdefer e.deinit();
    const window = o.window;
    var plan = e.plan(o.streams, window) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    if (o.world > 1) {
        // the window every rank fits, then the prompt cache the least of them holds
        const fit = e.least(.{ plan.window, 0 }) catch |err| return refuse(a, problem, "the ranks could not agree on a window ({s})", .{@errorName(err)});
        plan = e.plan(o.streams, fit[0]) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
        const cache = e.least(.{ plan.window, plan.cache_budget }) catch |err| return refuse(a, problem, "the ranks could not agree on a prompt cache ({s})", .{@errorName(err)});
        plan.cache_budget = cache[1];
    }
    const policy = dev.policy;
    // --prompt-cache-gib and --checkpoint-slots override the policy's prefix bytes and slots; zero of either: off
    if (o.cache_gib) |g| plan.cache_budget = @intFromFloat(g * (1 << 30)) else if (policy.prefix.bytes > 0) plan.cache_budget = policy.prefix.bytes;
    const keep: usize = if (o.keep) |n| @intCast(n) else policy.prefix.slots;
    if (keep == 0) plan.cache_budget = 0;
    if (plan.window == 0) {
        return refuse(a, problem, "the weights leave no room for a request on this GPU ({d:.2} GiB of {d:.2} GiB); use a smaller checkpoint, --lanes or more ranks (--tp)", .{ gibs(plan.weights), gibs(plan.total) });
    }
    if (plan.window < window and o.fixed) {
        return refuse(a, problem, "--context {d} does not fit this GPU's memory: {d} lanes and a kept copy of a prompt fit {d} tokens beside the weights; lower --context or --lanes, or add ranks (--tp)", .{ window, o.streams, plan.window });
    }
    // the pool every rank can hold: the least of the ranks' pages, each rank's own bytes a page
    var pool = memory.pool(e.weights.spec, e.act.size(), o.streams, e.o.batch_rows, plan.capacity, plan.cache_budget, keep);
    if (o.world > 1) {
        const fit = e.least(.{ pool.pages, pool.cache_pages }) catch |err| return refuse(a, problem, "the ranks could not agree on a page pool ({s})", .{@errorName(err)});
        pool.pages = fit[0];
        pool.cache_pages = fit[1];
    }
    e.size(plan.capacity, pool.pages) catch |err| return refuse(a, problem, "the native HIP engine cannot allocate its scratch and pages ({s})", .{@errorName(err)});
    std.debug.print("[tensorfold] HIP rank {d} of {d}: weights {d:.2} GiB, scratch {d:.2} GiB, context window {d} tokens, prompt cache {d:.2} GiB ({d} snapshots), {d} pages of {d:.2} MiB, reserve {d:.2} GiB of {d:.2} GiB\n", .{ o.rank, o.world, gibs(plan.weights), gibs(plan.scratch), plan.window, gibs(plan.cache_budget), pool.snaps, pool.pages, @as(f64, @floatFromInt(e.pool.pageBytes())) / (1 << 20), gibs(plan.reserve), gibs(plan.total) });
    return .{ .e = e, .plan = plan, .keep = keep, .pool = pool };
}

const Owned = struct {
    gpa: std.mem.Allocator,
    e: *Engine,
    backend: *hip_lanes.Hip,
};

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    // the other ranks hear that the rounds are over, then every rank frees its share
    own.backend.deinit();
    own.e.deinit();
    own.gpa.destroy(own);
}

/// The engine for `dir` on `dev`; under tensor parallelism this is rank 0, which serves.
fn openEngine(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!Loaded {
    const p = try prepare(a, gpa, io, dev, dir, o, problem);
    errdefer p.e.deinit();
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = p.e, .backend = try hip_lanes.Hip.init(gpa, p.e) };
    errdefer own.backend.deinit();
    if (p.plan.cache_budget > 0) own.backend.keepPages(p.pool.cache_pages, p.pool.snaps);
    // a lone rank times its forwards for the depth rule; the ranks of a group draft without costs
    if (dev.group) |g| own.backend.withLink(&g.link) else own.backend.measure();
    return .{
        .backend = own.backend.backend(),
        .facts = own.backend.facts(),
        .rows = hip_lanes.Hip.max_window,
        .ctx = own,
        .deinit = release,
        .window = p.plan.window,
        .fitted = p.plan.window < o.window,
    };
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops.
fn followRank(a: std.mem.Allocator, gpa: std.mem.Allocator, io: std.Io, dev: hip.Device, dir: []const u8, o: Options, problem: *[]const u8) anyerror!void {
    const p = try prepare(a, gpa, io, dev, dir, o, problem);
    defer p.e.deinit();
    var w = try worker.Worker.init(gpa, p.e);
    defer w.deinit();
    try w.follow(&dev.group.?.link);
}
