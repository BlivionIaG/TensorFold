//! `tf-hip-test kernels`: every kernel against its reference, the affine ones through the registry's entries.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const prod = @import("product.zig");
const cases = @import("cases.zig");
const affine = @import("affine.zig");
const rows = @import("rows.zig");
const decode = @import("decode.zig");
const gdn = @import("gdn.zig");
const oracle = @import("oracle.zig");
const Gpu = check.Gpu;
const Rig = prod.Rig;

pub const Options = struct {
    filter: []const u8 = "",
    bench: bool = false,
    /// A directory of Python-oracle fixtures to run the affine products against.
    oracle: ?[]const u8 = null,
    reps: usize = 5,
};

const State = struct { gpu: Gpu, opts: Options, rig: *Rig };

fn affineDecode(s: State) !void {
    try affine.list(s.rig, &cases.decode, "decode");
}

fn affinePrefill(s: State) !void {
    try affine.list(s.rig, &cases.prefill, "prefill");
}

fn affineSweep(s: State) !void {
    try affine.sweep(s.rig);
}

fn affineShort(s: State) !void {
    try affine.short(s.rig);
}

fn rowsDecode(s: State) !void {
    try rows.decode(s.rig);
}

fn rowsPrefill(s: State) !void {
    try rows.prefill(s.rig);
}

fn gdnCheck(s: State) !void {
    try gdn.run(s.gpu);
}

fn gdnBench(s: State) !void {
    try gdn.bench(s.gpu, 32768);
}

fn oracleCheck(s: State) !void {
    try oracle.affine(s.gpu, s.opts.oracle.?);
}

fn decodeGroup(comptime which: []const u8) fn (State) anyerror!void {
    return struct {
        fn f(s: State) anyerror!void {
            try decode.run(s.gpu, which, s.opts.bench, s.opts.reps);
        }
    }.f;
}

/// Runs the case groups the filter takes; the exit code is 0 when none failed.
pub fn run(gpu: Gpu, opts: Options) !u8 {
    const caps = try gpu.ctx.caps();
    var lib = try hip.rocm.Library.open(gpu.d, caps, try check.policyOf(gpu));
    defer lib.close();
    const launcher = &(lib.zig orelse return error.LibraryUnavailable);
    var rig: Rig = .{
        .gpu = gpu,
        .launcher = launcher,
        .kernels = &launcher.affine,
        .stream = try hip.Stream.init(gpu.d, true),
        .fp16 = caps.family == .rdna2,
        .bench = opts.bench,
        .reps = opts.reps,
    };
    defer rig.stream.deinit();
    const state: State = .{ .gpu = gpu, .opts = opts, .rig = &rig };
    var t: check.Tally = .{ .filter = opts.filter };
    t.group("affine decode", state, affineDecode);
    t.group("affine prefill", state, affinePrefill);
    t.group("affine sweep", state, affineSweep);
    t.group("rows decode", state, rowsDecode);
    t.group("rows prefill", state, rowsPrefill);
    t.group("gdn", state, gdnCheck);
    t.group("decode router", state, decodeGroup("router"));
    t.group("decode tail", state, decodeGroup("tail"));
    t.group("decode group", state, decodeGroup("group"));
    t.group("decode pair", state, decodeGroup("pair"));
    t.group("decode chain", state, decodeGroup("chain"));
    if (opts.oracle != null) t.group("oracle affine", state, oracleCheck);
    if (opts.bench) {
        t.group("affine short", state, affineShort);
        t.group("gdn bench", state, gdnBench);
    }
    return t.summary("kernels");
}
