//! GPU test runner for the Zig HIP port: `tf-hip-test <command> [flags]`, one PASS or FAIL line a case group, exit 1 on failure.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const runtime = @import("runtime/runtime.zig");
const kernels = @import("kernels/kernels.zig");
const launches = @import("bench/launches.zig");
const overhead = @import("bench/overhead.zig");

const usage =
    \\usage: tf-hip-test [--policy K=V,...] [-v] <command>
    \\  runtime [--filter F]                      the HIP runtime on this GPU: info, smoke, graph, cooperative, library, image
    \\  kernels [--filter F] [--bench] [--oracle DIR] [--reps N]
    \\                                            every kernel against its reference: the affine entries of the registry against
    \\                                            float64 and their families, rows and tiles byte for byte, the DeltaNet, the
    \\                                            decode tails and router; --oracle DIR adds the Python-oracle affine fixtures
    \\  bench launches [n] [reps]                 host cost of one kernel call: the library's C launchers vs the Zig launches
    \\  bench overhead [n] [reps]                 dependent one-thread kernels: plain stream vs one graph
    \\  Case groups of `kernels`: affine decode|prefill|sweep, rows decode|prefill, gdn, decode router|tail|group|pair|chain,
    \\  oracle affine; with --bench also affine short and gdn bench. --filter takes a substring of a group's name.
    \\  --policy sets the run's Policy (TF_POLICY and the old variables apply too); -v prints each group's steps.
    \\Old command names stay as aliases for one release: info smoke graph cooperative library image affine gemm gemv decode exact
    \\gdn launches overhead.
    \\
;

const Command = enum { runtime, kernels, bench_launches, bench_overhead, info };

/// What an invocation asks for, whether by the new names or an old alias.
const Plan = struct {
    command: Command,
    filter: []const u8 = "",
    bench: bool = false,
    oracle: ?[]const u8 = null,
    reps: ?usize = null,
    n: ?usize = null,
};

const Bad = error{ BadArguments, UnknownCommand };

fn number(text: []const u8) Bad!usize {
    return std.fmt.parseInt(usize, text, 10) catch error.BadArguments;
}

fn value(rest: []const []const u8, i: *usize) Bad![]const u8 {
    i.* += 1;
    if (i.* >= rest.len) return error.BadArguments;
    return rest[i.*];
}

/// The old commands as filters of the new ones: positional [reps] [filter] arguments are read, the old tile-by-tile
/// comparisons are the registry's entries now.
fn oldAlias(cmd: []const u8, rest: []const []const u8) Bad!?Plan {
    const eql = std.mem.eql;
    const first: []const u8 = if (rest.len > 0) rest[0] else "";
    if (eql(u8, cmd, "info")) return .{ .command = .info };
    for ([_][]const u8{ "smoke", "graph", "cooperative", "library", "image" }) |name| {
        if (eql(u8, cmd, name)) return .{ .command = .runtime, .filter = name };
    }
    if (eql(u8, cmd, "affine")) return .{ .command = .kernels, .filter = "oracle", .oracle = if (rest.len > 0) first else return error.BadArguments };
    if (eql(u8, cmd, "gemm")) {
        if (eql(u8, first, "sweep")) return .{ .command = .kernels, .filter = "affine sweep" };
        if (eql(u8, first, "tiers")) return .{ .command = .kernels, .filter = "rows prefill" };
        if (eql(u8, first, "short")) return .{ .command = .kernels, .filter = "affine short", .bench = true };
        return .{ .command = .kernels, .filter = "affine prefill", .bench = true };
    }
    if (eql(u8, cmd, "gemv")) return .{ .command = .kernels, .filter = "affine decode", .bench = true };
    if (eql(u8, cmd, "exact")) return .{ .command = .kernels, .filter = "rows decode" };
    if (eql(u8, cmd, "gdn")) return .{ .command = .kernels, .filter = if (eql(u8, first, "bench")) "gdn bench" else "gdn", .bench = eql(u8, first, "bench") };
    if (eql(u8, cmd, "decode")) {
        const which: []const u8 = if (rest.len > 1) rest[1] else "";
        const filters = [_][]const u8{ "decode router", "decode tail", "decode group", "decode pair", "decode chain" };
        for (filters) |f| if (which.len > 0 and std.mem.endsWith(u8, f, which)) return .{ .command = .kernels, .filter = f, .bench = true };
        return .{ .command = .kernels, .filter = "decode ", .bench = true };
    }
    if (eql(u8, cmd, "launches")) return .{ .command = .bench_launches, .n = if (rest.len > 0) try number(first) else null, .reps = if (rest.len > 1) try number(rest[1]) else null };
    if (eql(u8, cmd, "overhead")) return .{ .command = .bench_overhead, .n = if (rest.len > 0) try number(first) else null, .reps = if (rest.len > 1) try number(rest[1]) else null };
    return null;
}

fn parse(cmd: []const u8, rest: []const []const u8) Bad!Plan {
    if (try oldAlias(cmd, rest)) |plan| {
        std.debug.print("note: `{s}` is an alias of the {t} command, with {s}\n", .{ cmd, plan.command, if (plan.filter.len > 0) plan.filter else "all groups" });
        return plan;
    }
    var plan: Plan = undefined;
    var i: usize = 0;
    if (std.mem.eql(u8, cmd, "runtime")) {
        plan = .{ .command = .runtime };
    } else if (std.mem.eql(u8, cmd, "kernels")) {
        plan = .{ .command = .kernels };
    } else if (std.mem.eql(u8, cmd, "bench")) {
        if (rest.len == 0) return error.BadArguments;
        plan = .{ .command = if (std.mem.eql(u8, rest[0], "launches")) .bench_launches else if (std.mem.eql(u8, rest[0], "overhead")) .bench_overhead else return error.BadArguments };
        if (rest.len > 1) plan.n = try number(rest[1]);
        if (rest.len > 2) plan.reps = try number(rest[2]);
        return plan;
    } else return error.UnknownCommand;
    while (i < rest.len) : (i += 1) {
        const a = rest[i];
        if (std.mem.eql(u8, a, "--filter")) {
            plan.filter = try value(rest, &i);
        } else if (std.mem.eql(u8, a, "--bench") and plan.command == .kernels) {
            plan.bench = true;
        } else if (std.mem.eql(u8, a, "--oracle") and plan.command == .kernels) {
            plan.oracle = try value(rest, &i);
        } else if (std.mem.eql(u8, a, "--reps") and plan.command == .kernels) {
            plan.reps = try number(try value(rest, &i));
        } else return error.BadArguments;
    }
    return plan;
}

pub fn main(init: std.process.Init) !u8 {
    const raw = try init.minimal.args.toSlice(init.arena.allocator());
    var policy: []const u8 = "";
    var words: std.ArrayList([]const u8) = .empty;
    var i: usize = 1;
    while (i < raw.len) : (i += 1) {
        if (std.mem.eql(u8, raw[i], "--policy") and i + 1 < raw.len) {
            i += 1;
            policy = raw[i];
        } else if (std.mem.eql(u8, raw[i], "-v") or std.mem.eql(u8, raw[i], "--verbose")) {
            check.verbose = true;
        } else try words.append(init.arena.allocator(), raw[i]);
    }
    if (words.items.len == 0) {
        std.debug.print("{s}", .{usage});
        return 2;
    }
    const plan = parse(words.items[0], words.items[1..]) catch |e| {
        std.debug.print("{s}\n{s}", .{ @errorName(e), usage });
        return 2;
    };
    var driver = try hip.Driver.open();
    defer driver.close();
    var ctx = try hip.Context.init(&driver, 0);
    defer ctx.deinit();
    const gpu: check.Gpu = .{ .d = &driver, .ctx = &ctx, .gpa = init.gpa, .io = init.io, .policy = policy };
    return execute(gpu, plan) catch |e| {
        std.debug.print("FAIL {t}: {t}\n", .{ plan.command, e });
        return 1;
    };
}

fn execute(gpu: check.Gpu, plan: Plan) !u8 {
    switch (plan.command) {
        .info => {
            try runtime.info(gpu);
            return 0;
        },
        .runtime => return runtime.run(gpu, plan.filter),
        .kernels => return kernels.run(gpu, .{ .filter = plan.filter, .bench = plan.bench, .oracle = plan.oracle, .reps = plan.reps orelse 5 }),
        .bench_launches => try launches.run(gpu, plan.n orelse 2000, plan.reps orelse 15),
        .bench_overhead => try overhead.run(gpu, plan.n orelse 1000, plan.reps orelse 20),
    }
    return 0;
}
