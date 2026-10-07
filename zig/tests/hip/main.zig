//! GPU test runner for the Zig HIP port: `tf-hip-test <command> [flags]`, a PASS or FAIL line a group, exit 1 on FAIL.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const runtime = @import("runtime/runtime.zig");
const kernels = @import("kernels/kernels.zig");
const launches = @import("bench/launches.zig");
const overhead = @import("bench/overhead.zig");
const args = @import("args.zig");

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
    const plan = args.parse(words.items[0], words.items[1..]) catch |e| {
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

fn execute(gpu: check.Gpu, plan: args.Plan) !u8 {
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
