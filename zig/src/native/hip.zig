//! The engines a native server opens on HIP: Qwen3.5 / Qwen3.6 MLX affine checkpoints (dense and MoE) on the lane
//! core, every request's rows verified in shared rounds on one RDNA2 or RDNA3 GPU.

const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const qwen35 = @import("qwen35");
const Allocator = std.mem.Allocator;

pub const backends: []const []const u8 = &.{"hip"};
pub const families: []const api.Family = &.{
    .{ .model_type = "qwen3_5", .formats = &.{ "mlx-q2", "mlx-q3", "mlx-q4", "mlx-q5", "mlx-q6", "mlx-q8" } },
    .{ .model_type = "qwen3_5_moe", .formats = &.{ "mlx-q2", "mlx-q3", "mlx-q4", "mlx-q5", "mlx-q6", "mlx-q8" } },
};

/// The GPU family gate entries name ("rdna2" for gfx103x, "rdna3" for gfx11 / gfx12); null without a usable GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    var d = hip.Driver.open() catch return null;
    defer d.close();
    var ctx = hip.Context.init(&d, 0) catch return null;
    defer ctx.deinit();
    const family = hip.rocm.familyOf(ctx.capability() catch return null) orelse return null;
    return a.dupe(u8, @tagName(family)) catch null;
}

/// The model's window (config.json's max_position_embeddings, text_config's first), 0 when it names none.
fn modelContext(a: Allocator, io: std.Io, dir: []const u8) i64 {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return 0;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return 0;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return 0;
    if (doc != .object) return 0;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    const limit = text.object.get("max_position_embeddings") orelse doc.object.get("max_position_embeddings") orelse return 0;
    return if (limit == .integer and limit.integer > 0) limit.integer else 0;
}

/// Prompt caches kept for later turns, within the plan's byte budget.
const kept_prompts = 8;

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// Positions a stream holds when neither --context nor memory says otherwise.
const default_context = 32768;

const Host = struct {
    gpa: Allocator,
    e: *qwen35.engine.Engine,
    backend: *qwen35.hip_lanes.Hip,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.backend.deinit();
        h.e.deinit();
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no HIP engine reads the checkpoint.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5") and !std.mem.eql(u8, o.model_type, "qwen3_5_moe")) {
        problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
        return null;
    }
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse @min(if (native > 0) native else default_context, default_context);
    if (window <= 0 or (native > 0 and window > native)) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} exceeds this model's {d}-token window", .{ window, native });
        return null;
    }
    const rows = qwen35.hip_lanes.Hip.max_window;
    const streams = @max(o.lanes, 1);
    const e = qwen35.engine.Engine.load(gpa, io, o.dir, .{ .batch_rows = @max(32, @min(streams * rows, 128)), .slack = rows + 1 }) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native HIP engine cannot load {s} ({s})", .{ o.dir, @errorName(err) });
        return null;
    };
    var served = false;
    defer if (!served) e.deinit();
    const plan = e.plan(streams, @intCast(window)) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
        return null;
    };
    if (plan.window == 0) {
        problem.* = try std.fmt.allocPrint(a, "the weights leave no room for a request on this GPU ({d:.2} GiB of {d:.2} GiB); use a smaller checkpoint or --lanes", .{ gibs(plan.weights), gibs(plan.total) });
        return null;
    }
    if (o.context != null and plan.window < @as(usize, @intCast(window))) {
        problem.* = try std.fmt.allocPrint(a, "--context {d} does not fit this GPU's memory: {d} lanes and a kept copy of a prompt fit {d} tokens beside the weights; lower --context or --lanes", .{ window, streams, plan.window });
        return null;
    }
    e.size(plan.capacity) catch |err| {
        problem.* = try std.fmt.allocPrint(a, "the native HIP engine cannot allocate its scratch ({s})", .{@errorName(err)});
        return null;
    };
    std.debug.print("[tensorfold] HIP: weights {d:.2} GiB, scratch {d:.2} GiB, context window {d} tokens, prompt cache {d:.2} GiB, reserve {d:.2} GiB of {d:.2} GiB\n", .{ gibs(plan.weights), gibs(plan.scratch), plan.window, gibs(plan.cache_budget), gibs(plan.reserve), gibs(plan.total) });
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.e = e;
    h.backend = try qwen35.hip_lanes.Hip.init(gpa, e);
    errdefer h.backend.deinit();
    h.backend.keepPrompts(kept_prompts, plan.cache_budget);
    h.cfg = try lanes.Config.init(gpa, h.backend.facts(), rows, rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = streams, .context_window = @intCast(plan.window), .context_fitted = plan.window < @as(usize, @intCast(window)) });
    try h.host.start();
    served = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}
