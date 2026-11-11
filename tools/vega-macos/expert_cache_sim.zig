//! tf-expert-cache-sim: offline replay of real Qwen3.6-35B-A3B decode routing traces against expert-cache policies (CPU only).
//!
//! Model: one global pool of CAP lines (default 2876 = 4.74 GiB of 1,769,472 B experts) shared by 40 layers x 256 experts (key = layer*256+expert).
//! Every (token, layer) call asks for the layer's top-8 (lockstep mode: the union of the sequences' picks). Residents are hits and are touched
//! first, then misses are admitted one by one ("admit on miss" = free: the miss read writes through into the chosen line). A line touched or
//! admitted in the current call is never the victim (the production `stageAt` rule, zig/src/hip/expert_store.zig). Policies may also refuse to
//! admit (TinyLFU; ARC never does). The cache and all policy state are emptied at each prompt (or lockstep group) start.
//!
//! usage: tf-expert-cache-sim [TRACES_DIR] [--long DIR] [--cap N] [--jobs N] [--only SUBSTR]
//!   TRACES_DIR holds meta.json + *.npz (ids (T,40,8), kind: 1 = decode, 0 = prefill, as written by routing_trace.py); the default is the study
//!   worktree's traces/. --long defaults to <TRACES_DIR>/../traces_long (set to "none" to skip).
//! Datasets: short = every prompt of TRACES_DIR on its own; long = long generations; lock4 = short prompts, 4 at a time in lockstep, per-layer union.
//! The report is the one of the Python tool this replaced (expert_cache_sim.py), line for line, down to the number formats.
const std = @import("std");
const l = @import("expert_cache_sim_list.zig");
const data = @import("expert_cache_sim_data.zig");
const cfg = @import("expert_cache_sim_config.zig");
const run = @import("expert_cache_sim_run.zig");
const report = @import("expert_cache_sim_report.zig");

const cap_default = 2876;
const default_traces = "~/Projects/Infra/tf-study-moe-cache/docs/proposals/metal-tb4-split-moe-probe/traces";

/// Exits with a message on stderr; the Python original ended in a traceback.
fn die(comptime fmt: []const u8, args: anytype) noreturn {
    std.debug.print("expert_cache_sim: " ++ fmt ++ "\n", args);
    std.process.exit(1);
}

/// `os.path.expanduser` for `~` and `~/...`.
fn expandUser(a: std.mem.Allocator, home: ?[]const u8, path: []const u8) ![]const u8 {
    if (!std.mem.eql(u8, path, "~") and !std.mem.startsWith(u8, path, "~/")) return path;
    const h = home orelse return path;
    return std.fmt.allocPrint(a, "{s}{s}", .{ std.mem.trimEnd(u8, h, "/"), path[1..] });
}

pub const Options = struct {
    /// the traces directory, `~` already expanded
    traces: []const u8,
    /// `--long`: null or empty = <traces>/../traces_long, "none" = skip
    long: ?[]const u8 = null,
    cap: usize = cap_default,
    /// `--jobs` as given (the report echoes it) and as a number
    jobs_text: []const u8,
    jobs: usize,
    /// `--only`: the configurations whose label contains this
    only: []const u8 = "",
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const io = init.io;
    const args = try init.minimal.args.toSlice(arena);
    var long: ?[]const u8 = null;
    var cap: usize = cap_default;
    var jobs_text: []const u8 = try std.fmt.allocPrint(arena, "{d}", .{std.Thread.getCpuCount() catch 4});
    var only: []const u8 = "";
    var positional: ?[]const u8 = null;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const arg = args[i];
        if (std.mem.eql(u8, arg, "--long") or std.mem.eql(u8, arg, "--cap") or std.mem.eql(u8, arg, "--jobs") or std.mem.eql(u8, arg, "--only")) {
            i += 1;
            if (i == args.len) die("{s} needs a value", .{arg});
            if (std.mem.eql(u8, arg, "--long")) long = args[i] else if (std.mem.eql(u8, arg, "--jobs")) jobs_text = args[i] else if (std.mem.eql(u8, arg, "--only")) only = args[i] else cap = std.fmt.parseInt(usize, args[i], 10) catch die("--cap {s}: not a number of lines", .{args[i]});
        } else if (positional == null) positional = arg;
    }
    const jobs = std.fmt.parseInt(usize, jobs_text, 10) catch die("--jobs {s}: not a number", .{jobs_text});
    if (jobs == 0) die("--jobs must be at least 1", .{});
    const traces = try expandUser(arena, init.environ_map.get("HOME"), positional orelse default_traces);
    var stdout_buf: [16384]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &stdout_buf);
    const out = &stdout.interface;
    execute(arena, io, out, .{ .traces = traces, .long = long, .cap = cap, .jobs_text = jobs_text, .jobs = jobs, .only = only }) catch |e| {
        out.flush() catch {};
        switch (e) {
            error.NoBaseline => die("tables 3-4 need the \"LRU global\" configuration (it does not match --only)", .{}),
            error.LoadFailed, error.JobFailed => std.process.exit(1), // already reported
            else => die("{t}", .{e}),
        }
    };
    try out.flush();
}

fn elapsedSeconds(io: std.Io, since: std.Io.Timestamp) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - since.toNanoseconds())) / 1e9;
}

/// The whole run: load, replay, write the report to `out`.
pub fn execute(arena: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, opt: Options) !void {
    const t0 = std.Io.Clock.awake.now(io);
    var ldir: []const u8 = opt.long orelse "";
    if (ldir.len == 0) {
        const cwd = try std.process.currentPathAlloc(io, arena);
        const abs = try std.fs.path.resolve(arena, &.{ cwd, opt.traces });
        ldir = try std.fs.path.join(arena, &.{ std.fs.path.dirname(abs) orelse "/", "traces_long" });
    }
    const cap = opt.cap;
    const short = data.load(arena, io, opt.traces) catch |e| {
        std.debug.print("expert_cache_sim: cannot load {s}: {t}\n", .{ opt.traces, e });
        return error.LoadFailed;
    };
    var long: ?data.Set = null;
    if (!std.mem.eql(u8, ldir, "none")) {
        const st = std.Io.Dir.cwd().statFile(io, ldir, .{}) catch null;
        if (st != null and st.?.kind == .directory) long = data.load(arena, io, ldir) catch |e| {
            std.debug.print("expert_cache_sim: cannot load {s}: {t}\n", .{ ldir, e });
            return error.LoadFailed;
        };
    }
    const seeds = try data.buildSeed(arena, short, cap);
    const groups = try data.lockGroups(arena, short);
    var all_configs = try cfg.build(arena);
    var n_conf: usize = 0;
    for (all_configs) |c| {
        if (std.mem.indexOf(u8, c.label, opt.only) != null) {
            all_configs[n_conf] = c;
            n_conf += 1;
        }
    }
    const configs = all_configs[0..n_conf];

    const gib: f64 = @as(f64, @floatFromInt(cap * 1769472)) / 1073741824.0;
    try out.print("# expert cache replay, {d} lines (", .{cap});
    try report.fixed(out, gib, 2);
    try out.writeAll(" GiB), cold start per prompt\n");
    try out.print("short: {d} prompts, {d} decode tokens; ", .{ short.traces.len, short.decodeTokens() });
    if (long) |lg| try out.print("long: {d} generations, {d} decode tokens; ", .{ lg.traces.len, lg.decodeTokens() });
    try out.print("lock4: {d} groups of 4 short prompts in lockstep\n\n", .{groups.len});

    var jobs: std.ArrayList(run.Job) = .empty;
    for (configs, 0..) |c, ci| {
        const sets = [_]?data.Set{ short, long };
        for (sets, 0..) |maybe, di| {
            const set = maybe orelse continue;
            for (set.by_name, 0..) |ti, item| try jobs.append(arena, .{ .config = @intCast(ci), .dataset = @fromBackingInt(@intCast(di)), .item = @intCast(item), .weight = @intCast(set.traces[ti].decodeTokens()) });
        }
        if (c.init == .cold) for (0..groups.len) |g| try jobs.append(arena, .{ .config = @intCast(ci), .dataset = .lock4, .item = @intCast(g), .weight = 4 * 287 });
    }
    // heaviest first so the workers stay busy
    std.mem.sort(run.Job, jobs.items, {}, struct {
        fn heavier(_: void, a: run.Job, b: run.Job) bool {
            return a.weight > b.weight;
        }
    }.heavier);
    const results = try arena.alloc(run.Result, jobs.items.len);
    var ctx: run.Ctx = .{ .cap = cap, .configs = configs, .short = short, .long = long, .seeds = seeds, .groups = groups, .jobs = jobs.items, .results = results };
    try run.runAll(&ctx, opt.jobs);
    try out.writeAll("(replay took ");
    try report.fixed(out, elapsedSeconds(io, t0), 0);
    try out.print(" s on {s} workers)\n\n", .{opt.jobs_text});

    var res = try report.Results.init(arena, configs, short, long, jobs.items, results);
    try report.tables(arena, out, &res);
    try out.writeAll("\n(total wall time ");
    try report.fixed(out, elapsedSeconds(io, t0), 0);
    try out.writeAll(" s)\n");
}
