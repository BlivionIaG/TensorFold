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
    const caps = ctx.caps() catch return null;
    return a.dupe(u8, @tagName(caps.family)) catch null;
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

/// Prompt caches kept for later turns when `--checkpoint-slots` names none.
const kept_prompts = 8;

fn gibs(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// Positions a stream holds when neither --context nor the checkpoint's config names a window.
const default_context = 32768;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

/// A flag or a resource the engine cannot take: `problem` says why.
const Refused = error{Refused};

fn refuse(a: Allocator, problem: *[]const u8, comptime fmt: []const u8, args: anytype) Refused {
    problem.* = std.fmt.allocPrint(a, fmt, args) catch fmt;
    return error.Refused;
}

/// The tensor-parallel flags as the Python ROCm server checks them.
fn checkGroup(a: Allocator, o: api.Open, problem: *[]const u8) Refused!void {
    if (o.tp != 1 and o.tp != 2 and o.tp != 4 and o.tp != 8) return refuse(a, problem, "--tp {d} is not a supported ROCm world size; choose 1, 2, 4 or 8", .{o.tp});
    if (o.tp > 1 and o.master.len == 0) return refuse(a, problem, "--tp > 1 needs --master: rank 0's address on the link between the machines", .{});
    if (o.tp == 1 and o.rank != 0) return refuse(a, problem, "--rank must be 0 when --tp 1", .{});
    if (o.rank >= o.tp) return refuse(a, problem, "--rank {d} not in [0, --tp {d})", .{ o.rank, o.tp });
    if (o.keep) |n| if (n < 0) return refuse(a, problem, "--checkpoint-slots must be 0 or more", .{});
    if (o.cache_gib) |g| if (g < 0) return refuse(a, problem, "--prompt-cache-gib must be 0 or more", .{});
}

/// The ranks of a tensor-parallel group: the TCP link that carries rank 0's steps and RCCL's unique id. RCCL's
/// library stays loaded until the engine is gone (its bootstrap thread starts at the id).
const Group = struct {
    rccl: hip.rccl.Rccl,
    link: hip.link.Link,
    id: hip.rccl.UniqueId,

    /// Rank 0 listens on `master`:`master_port` until the others have connected; they get its id.
    /// Rank 0's policy goes to the others, which adopt it; a rank whose GPU differs from rank 0's is refused.
    fn join(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, caps: hip.Caps, policy: *hip.Policy, problem: *[]const u8) (Refused || Allocator.Error)!*Group {
        const g = try gpa.create(Group);
        errdefer gpa.destroy(g);
        g.rccl = hip.rccl.Rccl.open(policy.rccl_lib.slice()) catch |err| return refuse(a, problem, "tensor parallelism needs RCCL ({s})", .{@errorName(err)});
        errdefer g.rccl.close();
        const mine: hip.rccl.UniqueId = if (o.rank == 0) g.rccl.uniqueId() catch |err| return refuse(a, problem, "RCCL gave no id ({s})", .{@errorName(err)}) else undefined;
        const host = if (std.mem.eql(u8, o.master, "localhost")) "127.0.0.1" else o.master;
        const pair = hip.link.Link.open(io, o.rank, o.tp, host, o.master_port, mine, .{ .caps = caps.id(), .policy = policy.words() }) catch |err| return switch (err) {
            error.MixedGpus => refuse(a, problem, "the ranks' GPUs differ: a group needs one kind of GPU", .{}),
            else => refuse(a, problem, "the ranks' link at {s}:{d} failed ({s})", .{ o.master, o.master_port, @errorName(err) }),
        };
        g.link, g.id, const hello = pair;
        if (o.rank != 0) policy.* = hip.Policy.fromWords(hello.policy, policy.*);
        return g;
    }

    fn close(g: *Group, gpa: Allocator) void {
        g.link.close();
        g.rccl.close();
        gpa.destroy(g);
    }
};

/// What the card at `device` can do, or null when it is no usable GPU.
fn capsOf(device: c_int) ?hip.Caps {
    var d = hip.Driver.open() catch return null;
    defer d.close();
    var ctx = hip.Context.init(&d, device) catch return null;
    defer ctx.deinit();
    return ctx.caps() catch null;
}

/// The card of `rank`: with every card visible rank r takes card r, with one card a process that card.
fn deviceOf(rank: u32) c_int {
    var d = hip.Driver.open() catch return 0;
    defer d.close();
    const count = d.deviceCount() catch return 0;
    return if (count > 0) @intCast(rank % @as(u32, @intCast(count))) else 0;
}

/// One rank's loaded and sized engine: the window and prompt cache every rank agreed on.
const Prepared = struct {
    e: *qwen35.engine.Engine,
    plan: qwen35.memory.Plan,
    /// The window asked for, or the model's: `plan.window` is below it when the memory does not fit it.
    asked: usize,
    /// Kept prompt entries (the plan's `cache_budget` is 0 when the cache is off).
    keep: usize,
    streams: usize,
    group: ?*Group,
    /// The policy's line, for the server's info.
    policy_line: []const u8,

    fn deinit(p: *Prepared, gpa: Allocator) void {
        gpa.free(p.policy_line);
        p.e.deinit();
        if (p.group) |g| g.close(gpa);
    }
};

/// Joins the group, loads this rank's share of the model, plans the memory with the other ranks and sizes the scratch.
fn prepare(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) (Refused || Allocator.Error)!Prepared {
    if (!std.mem.eql(u8, o.model_type, "qwen3_5") and !std.mem.eql(u8, o.model_type, "qwen3_5_moe")) {
        return refuse(a, problem, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    }
    try checkGroup(a, o, problem);
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse if (native > 0) native else default_context;
    if (window <= 0 or (native > 0 and window > native)) return refuse(a, problem, "--context {d} exceeds this model's {d}-token window", .{ window, native });
    const rows = qwen35.hip_lanes.Hip.max_window;
    // `--parallel auto` is one lane, as on the Python ROCm engine
    const streams: usize = if (o.lanes_auto) 1 else @max(o.lanes, 1);
    const device = deviceOf(o.rank);
    const caps = capsOf(device) orelse return refuse(a, problem, "HIP device {d} is not a GPU this engine supports", .{device});
    // rank 0 resolves the policy: the GPU's defaults, the flags, the old variables, TF_POLICY; the others adopt it
    var notes: hip.Policy.Notes = .{};
    var policy = hip.Policy.resolve(caps, o.policy, .current, &notes) catch |err| return refuse(a, problem, "the policy \"{s}\" is refused ({s})", .{ o.policy, @errorName(err) });
    const group: ?*Group = if (o.tp > 1) try Group.join(a, gpa, io, o, caps, &policy, problem) else null;
    errdefer if (group) |g| g.close(gpa);
    const policy_line = try std.fmt.allocPrint(gpa, "{f}", .{policy});
    errdefer gpa.free(policy_line);
    std.debug.print("[tensorfold] HIP rank {d} of {d}: policy {s}{s} {s}\n", .{ o.rank, o.tp, policy_line, if (o.rank > 0) " (rank 0's)" else "", notes.text() });
    if (o.p2p) |on| _ = setenv("NCCL_P2P_DISABLE", if (on) "0" else "1", 1);
    const e = qwen35.engine.Engine.load(gpa, io, o.dir, .{
        .batch_rows = @max(32, @min(streams * rows, 128)),
        .slack = rows + 1,
        .device = device,
        .policy = policy,
        .rank = o.rank,
        .world = o.tp,
        .id = if (group) |g| g.id else null,
    }) catch |err| return refuse(a, problem, "the native HIP engine cannot load {s} ({s})", .{ o.dir, @errorName(err) });
    errdefer e.deinit();
    var plan = e.plan(streams, @intCast(window)) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
    if (o.tp > 1) {
        // the window every rank fits, then the prompt cache the least of them holds
        const fit = e.least(.{ plan.window, 0 }) catch |err| return refuse(a, problem, "the ranks could not agree on a window ({s})", .{@errorName(err)});
        plan = e.plan(streams, fit[0]) catch |err| return refuse(a, problem, "the native HIP engine cannot read the GPU's memory ({s})", .{@errorName(err)});
        const cache = e.least(.{ plan.window, plan.cache_budget }) catch |err| return refuse(a, problem, "the ranks could not agree on a prompt cache ({s})", .{@errorName(err)});
        plan.cache_budget = cache[1];
    }
    // --prompt-cache-gib names the bytes and --checkpoint-slots the entries; zero of either turns the cache off
    if (o.cache_gib) |g| plan.cache_budget = @intFromFloat(g * (1 << 30));
    const keep: usize = if (o.keep) |n| @intCast(n) else kept_prompts;
    if (keep == 0) plan.cache_budget = 0;
    if (plan.window == 0) {
        return refuse(a, problem, "the weights leave no room for a request on this GPU ({d:.2} GiB of {d:.2} GiB); use a smaller checkpoint, --lanes or more ranks (--tp)", .{ gibs(plan.weights), gibs(plan.total) });
    }
    if (o.context != null and plan.window < @as(usize, @intCast(window))) {
        return refuse(a, problem, "--context {d} does not fit this GPU's memory: {d} lanes and a kept copy of a prompt fit {d} tokens beside the weights; lower --context or --lanes, or add ranks (--tp)", .{ window, streams, plan.window });
    }
    e.size(plan.capacity) catch |err| return refuse(a, problem, "the native HIP engine cannot allocate its scratch ({s})", .{@errorName(err)});
    std.debug.print("[tensorfold] HIP rank {d} of {d}: weights {d:.2} GiB, scratch {d:.2} GiB, context window {d} tokens, prompt cache {d:.2} GiB, reserve {d:.2} GiB of {d:.2} GiB\n", .{ o.rank, o.tp, gibs(plan.weights), gibs(plan.scratch), plan.window, gibs(plan.cache_budget), gibs(plan.reserve), gibs(plan.total) });
    return .{ .e = e, .plan = plan, .asked = @intCast(window), .keep = keep, .streams = streams, .group = group, .policy_line = policy_line };
}

const Host = struct {
    gpa: Allocator,
    e: *qwen35.engine.Engine,
    group: ?*Group,
    backend: *qwen35.hip_lanes.Hip,
    cfg: lanes.Config,
    policy_line: []const u8,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(ctx: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(ctx));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        // the other ranks hear that the rounds are over, then every rank frees its share
        h.backend.deinit();
        h.e.deinit();
        if (h.group) |g| g.close(h.gpa);
        h.gpa.free(h.policy_line);
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no HIP engine reads the checkpoint. Under tensor
/// parallelism this is rank 0, which serves; the others run `follow`.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    var p = prepare(a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    var served = false;
    defer if (!served) p.deinit(gpa);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.policy_line = try gpa.dupe(u8, p.policy_line);
    errdefer gpa.free(h.policy_line);
    h.e = p.e;
    h.group = p.group;
    h.backend = try qwen35.hip_lanes.Hip.init(gpa, p.e);
    errdefer h.backend.deinit();
    h.backend.keepPrompts(if (p.plan.cache_budget == 0) 0 else p.keep, p.plan.cache_budget);
    // a lone rank times its forwards for the depth rule; the ranks of a group draft without costs
    if (p.group) |g| h.backend.withLink(&g.link) else h.backend.measure();
    h.cfg = try lanes.Config.init(gpa, h.backend.facts(), qwen35.hip_lanes.Hip.max_window, qwen35.hip_lanes.Hip.max_window - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, h.backend.backend(), h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = @intCast(p.streams), .context_window = @intCast(p.plan.window), .context_fitted = p.plan.window < p.asked, .policy = h.policy_line });
    try h.host.start();
    served = true;
    gpa.free(p.policy_line);
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops; false with `problem` set
/// when it cannot start.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !bool {
    var p = prepare(a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => return false,
        else => |x| return x,
    };
    defer p.deinit(gpa);
    var w = try qwen35.worker.Worker.init(gpa, p.e);
    defer w.deinit();
    try w.follow(&p.group.?.link);
    return true;
}
