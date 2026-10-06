//! The decode-step kernels of decode.hip against the launches they replace (TF_DECODE_FUSE=old): the router's logits
//! and the residual tails, each against a float64 reference on random data, with microseconds a launch. Weights
//! rotate over many copies so each launch reads them from DRAM. `tf-hip-test decode [reps] [name filter]`.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const Gpu = check.Gpu;
const L = @typeInfo(@FieldType(hip.rocm.Library, "zig")).optional.child;

const Rng = struct {
    state: u64,

    fn next(r: *Rng) u64 {
        r.state ^= r.state >> 12;
        r.state ^= r.state << 25;
        r.state ^= r.state >> 27;
        return r.state *% 0x2545F4914F6CDD1D;
    }

    fn unit(r: *Rng) f32 {
        const v: i32 = @intCast(r.next() >> 40 & 0x7ff);
        return @as(f32, @floatFromInt(v - 1024)) / 1024.0;
    }
};

fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

fn f16Value(b: u16) f64 {
    const h: f16 = @bitCast(b);
    return h;
}

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn bf16Value(b: u16) f64 {
    return @as(f32, @bitCast(@as(u32, b) << 16));
}

const Rig = struct {
    gpu: Gpu,
    fast: L,
    old: L,
    stream: hip.Stream,
    fp16: bool,
    rng: Rng,
    reps: usize,
};

fn elapsed(t: *Rig, start: hip.Event, stop: hip.Event, iters: usize) !f64 {
    try stop.record(t.stream);
    try stop.synchronize();
    return @as(f64, try hip.Event.elapsedMs(start, stop)) * 1000.0 / @as(f64, @floatFromInt(iters));
}

/// The router's logits for `rows` rows over (experts, d) fp32 weights.
fn router(t: *Rig, rows: usize, d: usize, experts: usize) !void {
    const gpa = t.gpu.gpa;
    const copies = 96;
    const hx = try gpa.alloc(u16, rows * d);
    defer gpa.free(hx);
    for (hx) |*v| v.* = if (t.fp16) f16Bits(t.rng.unit()) else bf16Bits(t.rng.unit());
    const hw = try gpa.alloc(f32, experts * d);
    defer gpa.free(hw);
    for (hw) |*v| v.* = t.rng.unit() / 8.0;
    var x = try hip.DeviceBuffer.fromHost(t.gpu.d, std.mem.sliceAsBytes(hx));
    defer x.free();
    var w = try hip.DeviceBuffer.alloc(t.gpu.d, copies * hw.len * 4);
    defer w.free();
    for (0..copies) |i| try w.upload(i * hw.len * 4, std.mem.sliceAsBytes(hw));
    var out: [2]hip.DeviceBuffer = undefined;
    for (&out) |*o| o.* = try hip.DeviceBuffer.alloc(t.gpu.d, rows * experts * 4);
    defer for (&out) |*o| o.free();
    var start = try hip.Event.init(t.gpu.d, true);
    defer start.deinit();
    var stop = try hip.Event.init(t.gpu.d, true);
    defer stop.deinit();
    const kind: c_int = if (t.fp16) 1 else 2;
    var us: [2]f64 = undefined;
    var max_err: [2]f64 = .{ 0, 0 };
    for ([2]*const L{ &t.old, &t.fast }, 0..) |l, v| {
        const iters = copies * t.reps;
        var best: f64 = std.math.inf(f64);
        for (0..3) |round| {
            try start.record(t.stream);
            for (0..iters) |i| try l.tf_moe_router(@ptrFromInt(x.ptr), kind, @ptrFromInt(w.ptr + (i % copies) * hw.len * 4), @ptrFromInt(out[v].ptr), @intCast(rows), @intCast(d), @intCast(experts), t.stream.handle);
            const u = try elapsed(t, start, stop, iters);
            if (round > 0) best = @min(best, u);
        }
        us[v] = best;
        const got = try gpa.alloc(f32, rows * experts);
        defer gpa.free(got);
        try out[v].download(0, std.mem.sliceAsBytes(got));
        for (0..rows) |r| for (0..experts) |e| {
            var y: f64 = 0;
            var norm: f64 = 0;
            for (0..d) |i| {
                const xv = if (t.fp16) f16Value(hx[r * d + i]) else bf16Value(hx[r * d + i]);
                const term = xv * @as(f64, hw[e * d + i]);
                y += term;
                norm += @abs(term);
            }
            max_err[v] = @max(max_err[v], @abs(@as(f64, got[r * experts + e]) - y) / norm);
        };
    }
    std.debug.print("RESULT decode router r{d} d{d} e{d}: old {d:.1} us, new {d:.1} us, x{d:.2}; max|y-ref|/sum|terms| old {e:.1} new {e:.1}\n", .{ rows, d, experts, us[0], us[1], us[0] / us[1], max_err[0], max_err[1] });
    try check.expect(max_err[1] <= 2 * max_err[0] + 1e-9, "decode router r{d}: further from the float64 reference than twice the previous kernel", .{rows});
}

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
    const reps: usize = if (args.len > 0) try std.fmt.parseInt(usize, args[0], 10) else 3;
    const filter: []const u8 = if (args.len > 1) args[1] else "";
    const family = hip.rocm.familyOf(try gpu.ctx.capability()) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(gpu.d, family);
    defer lib.close();
    const launcher = lib.zig orelse return error.LibraryUnavailable;
    var t: Rig = .{
        .gpu = gpu,
        .fast = launcher,
        .old = launcher,
        .stream = try hip.Stream.init(gpu.d, true),
        .fp16 = family == .rdna2,
        .rng = .{ .state = 0x9E3779B97F4A7C15 },
        .reps = reps,
    };
    defer t.stream.deinit();
    t.fast.fuse = true;
    t.old.fuse = false;
    var ran: usize = 0;
    if (filter.len == 0 or std.mem.indexOf(u8, "router", filter) != null) {
        for ([_]usize{ 1, 2, 4, 8 }) |rows| try router(&t, rows, 2048, 257);
        ran += 1;
    }
    try check.expect(ran > 0, "decode: no case matches '{s}'", .{filter});
    check.pass("decode: {d} kernels, each within twice the previous launches' error of the float64 reference", .{ran});
}
