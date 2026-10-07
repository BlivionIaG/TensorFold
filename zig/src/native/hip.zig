//! The engines a native server opens on HIP: this file owns the GPU, its policy, the tensor-parallel group, the round loop
//! and the lane host; each family in `registry` brings its lane backend.

const std = @import("std");
const hip = @import("hip");
const api = @import("engine_api");
const lanes = @import("lanes");
const qwen35 = @import("qwen3_5");
const Allocator = std.mem.Allocator;

/// The HIP families: namespaces with `model_type`, `formats`, `default_context`, `prefill_step`, `open` and `follow`.
const registry = .{ qwen35.native, qwen35.native_moe };

pub const backends: []const []const u8 = &.{"hip"};
pub const families: []const api.Family = blk: {
    var out: [registry.len]api.Family = undefined;
    for (registry, 0..) |F, i| out[i] = .{ .model_type = F.model_type, .formats = F.formats };
    const final = out;
    break :blk &final;
};

/// The GPU family gate entries name ("rdna2" for gfx103x, "rdna3" for gfx11 / gfx12); null without a usable GPU.
pub fn chip(a: Allocator) ?[]const u8 {
    const caps = hip.Device.capsOf(0) orelse return null;
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
    if (o.prompt_cache_gib) |g| if (g < 0) return refuse(a, problem, "--prompt-cache-gib must be 0 or more", .{});
}

/// What every family needs before it loads: the card, the policy, the group, and the window and lanes asked for.
const Ready = struct {
    dev: hip.Device,
    options: Options,
    /// The policy's line, for the server's info.
    policy_line: []const u8,

    const Options = qwen35.native_engine.Options;

    fn deinit(r: Ready, gpa: Allocator) void {
        gpa.free(r.policy_line);
        if (r.dev.group) |g| g.close(gpa);
    }
};

/// Resolves the policy on rank 0 (the GPU's defaults, the flags, the old variables, TF_POLICY), joins the group and
/// gives the other ranks rank 0's policy.
fn ready(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) (Refused || Allocator.Error)!Ready {
    try checkGroup(a, o, problem);
    const native = modelContext(a, io, o.dir);
    const window: i64 = o.context orelse if (native > 0) native else F.default_context;
    if (window <= 0 or (native > 0 and window > native)) return refuse(a, problem, "--context {d} exceeds this model's {d}-token window", .{ window, native });
    const index = hip.Device.ordinal(o.rank);
    const caps = hip.Device.capsOf(index) orelse return refuse(a, problem, "HIP device {d} is not a GPU this engine supports", .{index});
    var notes: hip.Policy.Notes = .{};
    var policy = hip.Policy.resolve(caps, o.policy, .current, &notes) catch |err| return refuse(a, problem, "the policy \"{s}\" is refused ({s})", .{ o.policy, @errorName(err) });
    const group: ?*hip.Group = if (o.tp > 1) hip.Group.join(gpa, io, o.rank, o.tp, o.master, o.master_port, caps, &policy) catch |err| return switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        error.NoRccl => refuse(a, problem, "tensor parallelism needs RCCL", .{}),
        error.NoId => refuse(a, problem, "RCCL gave no id", .{}),
        error.MixedGpus => refuse(a, problem, "the ranks' GPUs differ: a group needs one kind of GPU", .{}),
        else => refuse(a, problem, "the ranks' link at {s}:{d} failed ({s})", .{ o.master, o.master_port, @errorName(err) }),
    } else null;
    errdefer if (group) |g| g.close(gpa);
    const policy_line = try std.fmt.allocPrint(gpa, "{f}", .{policy});
    std.debug.print("[tensorfold] HIP rank {d} of {d}: policy {s}{s} {s}\n", .{ o.rank, o.tp, policy_line, if (o.rank > 0) " (rank 0's)" else "", notes.text() });
    if (o.p2p) |on| _ = setenv("NCCL_P2P_DISABLE", if (on) "0" else "1", 1);
    return .{
        .dev = .{ .index = index, .caps = caps, .policy = policy, .group = group },
        .options = .{
            .window = @intCast(window),
            .fixed = o.context != null,
            // `--parallel auto` is one lane, as on the Python ROCm engine
            .streams = if (o.lanes_fixed) @max(o.lanes, 1) else 1,
            .rank = o.rank,
            .world = o.tp,
            .cache_gib = o.prompt_cache_gib,
            .keep = o.keep,
        },
        .policy_line = policy_line,
    };
}

/// One loaded model behind the lane host: everything the engine thread reads lives here.
const Host = struct {
    gpa: Allocator,
    group: ?*hip.Group,
    policy_line: []const u8,
    family: *anyopaque,
    release: *const fn (*anyopaque) void,
    cfg: lanes.Config,
    clock: lanes.backend.WallClock,
    core: lanes.Engine,
    host: api.LaneHost,

    fn close(p: *anyopaque) void {
        const h: *Host = @ptrCast(@alignCast(p));
        h.host.stop();
        h.core.deinit();
        h.cfg.deinit(h.gpa);
        h.release(h.family);
        if (h.group) |g| g.close(h.gpa);
        h.gpa.free(h.policy_line);
        h.gpa.destroy(h);
    }
};

/// The engine for `o.dir`, or null with `problem` set when no HIP family reads the checkpoint. Under tensor
/// parallelism this is rank 0, which serves; the others run `follow`.
pub fn open(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) return openWith(F, a, gpa, io, o, problem);
    }
    problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return null;
}

fn openWith(comptime F: type, a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !?api.Opened {
    const r = ready(F, a, gpa, io, o, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    var served = false;
    defer if (!served) r.deinit(gpa);
    const loaded = F.open(a, gpa, io, r.dev, o.dir, r.options, problem) catch |err| switch (err) {
        error.Refused => return null,
        else => |x| return x,
    };
    errdefer loaded.deinit(loaded.ctx);
    const h = try gpa.create(Host);
    errdefer gpa.destroy(h);
    h.gpa = gpa;
    h.group = r.dev.group;
    h.policy_line = r.policy_line;
    h.family = loaded.ctx;
    h.release = loaded.deinit;
    h.cfg = try lanes.Config.init(gpa, loaded.facts, loaded.rows, loaded.rows - 1);
    errdefer h.cfg.deinit(gpa);
    h.clock = .{ .io = io };
    h.core = lanes.Engine.init(gpa, &h.cfg, loaded.backend, h.clock.clock());
    errdefer h.core.deinit();
    h.host = api.LaneHost.init(gpa, io, &h.core, .{ .lanes = @intCast(r.options.streams), .context_window = @intCast(loaded.window), .context_fitted = loaded.fitted, .prefill_step = F.prefill_step, .policy = h.policy_line });
    try h.host.start();
    served = true;
    return .{ .engine = h.host.engine(), .close = Host.close, .ctx = h };
}

/// A rank above 0: holds its share of the model and runs rank 0's steps until rank 0 stops; false with `problem` set
/// when it cannot start.
pub fn follow(a: Allocator, gpa: Allocator, io: std.Io, o: api.Open, problem: *[]const u8) !bool {
    inline for (registry) |F| {
        if (std.mem.eql(u8, o.model_type, F.model_type)) {
            const r = ready(F, a, gpa, io, o, problem) catch |err| switch (err) {
                error.Refused => return false,
                else => |x| return x,
            };
            defer r.deinit(gpa);
            F.follow(a, gpa, io, r.dev, o.dir, r.options, problem) catch |err| switch (err) {
                error.Refused => return false,
                else => |x| return x,
            };
            return true;
        }
    }
    problem.* = try std.fmt.allocPrint(a, "the native HIP engine has no backend for {s} checkpoints yet; serve with --engine python", .{o.model_type});
    return false;
}

test "every registered family is listed for capabilities" {
    try std.testing.expectEqual(@as(usize, registry.len), families.len);
    try std.testing.expectEqualStrings("qwen3_5", families[0].model_type);
    try std.testing.expectEqualStrings("qwen3_5_moe", families[1].model_type);
}
