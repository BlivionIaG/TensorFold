//! `tf-hip-bench overhead|launches [n] [reps]`: the host cost of a launch, on a stream and in a graph, and per kernel.

const std = @import("std");
const hip = @import("hip");
const probe = @import("hip_probe");

const usage = "usage: tf-hip-bench overhead [n] [reps] | launches [n] [reps]\n" ++
    "  overhead: n dependent one-thread launches on a stream, then the same chain replayed from one graph\n" ++
    "  launches: the host cost of one call of each model kernel through the Zig launches, enqueued and done\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 2) {
        std.debug.print(usage, .{});
        return 2;
    }
    const n = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else null;
    const reps = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else null;
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    const b: Bench = .{ .r = &r, .gpa = init.gpa, .io = init.io };
    if (std.mem.eql(u8, args[1], "overhead")) return b.overhead(n orelse 1000, reps orelse 20);
    if (std.mem.eql(u8, args[1], "launches")) return b.launches(n orelse 2000, reps orelse 15);
    std.debug.print(usage, .{});
    return 2;
}

const Bench = struct {
    r: *hip.Runtime,
    gpa: std.mem.Allocator,
    io: std.Io,

    fn now(b: Bench) i96 {
        return std.Io.Clock.awake.now(b.io).toNanoseconds();
    }

    /// Microseconds a call from `t0` over `n` calls.
    fn per(b: Bench, t0: i96, n: usize) f64 {
        return @as(f64, @floatFromInt(b.now() - t0)) / 1000 / @as(f64, @floatFromInt(n));
    }

    /// `n` dependent one-thread launches, then the same chain captured once and replayed: microseconds a launch.
    fn overhead(b: Bench, n: usize, reps: usize) !u8 {
        var module = try hip.Module.load(b.r, &probe.bytes);
        defer module.unload();
        const step = try module.function("tf_hip_step");
        var stream = try hip.Stream.init(b.r);
        defer stream.deinit();
        var counter = try hip.DeviceBuffer.alloc(b.r, 8);
        defer counter.free();
        try counter.fill8(0);
        const single: hip.launch.Config = .{ .grid = .{ .x = 1 }, .block = .{ .x = 1 } };
        const times = try b.gpa.alloc(f64, reps);
        defer b.gpa.free(times);
        for (times) |*t| {
            const t0 = b.now();
            for (0..n) |_| try launchStep(step, single, stream, counter);
            try stream.synchronize();
            t.* = b.per(t0, n);
        }
        const plain = median(times);
        try hip.graph.beginCapture(stream, .thread_local);
        for (0..n) |_| try launchStep(step, single, stream, counter);
        var g = try hip.graph.endCapture(stream);
        defer g.deinit();
        var exec = try g.instantiate();
        defer exec.deinit();
        try exec.upload(stream);
        for (times) |*t| {
            const t0 = b.now();
            try exec.launchOn(stream);
            try stream.synchronize();
            t.* = b.per(t0, n);
        }
        const graphed = median(times);
        var ran: u64 = 0;
        try counter.download(0, std.mem.asBytes(&ran));
        const want: u64 = @intCast(2 * n * reps);
        if (ran != want) {
            std.debug.print("FAIL every launch runs once: {d} steps, expected {d}\n", .{ ran, want });
            return 1;
        }
        std.debug.print("RESULT {d} dependent launches: {d:.2} us each on a stream, {d:.2} us each in a graph\n", .{ n, plain, graphed });
        return 0;
    }

    /// The host cost of one call of each kernel through the Zig launches: enqueued, and done with the GPU's work.
    fn launches(b: Bench, n: usize, reps: usize) !u8 {
        const caps = try hip.Device.kernelCaps(b.r, 0);
        const fp16 = caps.act == .f16;
        var l = try hip.Launcher.load(b.r, .{}, hip.kernels.images);
        defer l.unload();
        var stream = try hip.Stream.init(b.r);
        defer stream.deinit();
        var bufs: Buffers = undefined;
        inline for (@typeInfo(Buffers).@"struct".field_names, .{ 1 << 20, 8 << 20, 8 << 20, 1 << 20, 1 << 20, 8 << 20, 64 }) |name, bytes| {
            @field(bufs, name) = try hip.DeviceBuffer.alloc(b.r, bytes);
            try @field(bufs, name).fill8(0);
        }
        defer inline for (@typeInfo(Buffers).@"struct".field_names) |name| @field(bufs, name).free();
        const times = try b.gpa.alloc(f64, reps);
        defer b.gpa.free(times);
        const enqueue = try b.gpa.alloc(f64, reps);
        defer b.gpa.free(enqueue);
        for (std.enums.values(Kernel)) |k| {
            for (times, enqueue) |*t, *e| {
                const t0 = b.now();
                for (0..n) |_| try one(&l, k, bufs, fp16, stream.handle);
                e.* = b.per(t0, n);
                try stream.synchronize();
                t.* = b.per(t0, n);
            }
            std.debug.print("RESULT {t}: {d:.2} us a call enqueued, {d:.2} us done\n", .{ k, median(enqueue), median(times) });
        }
        return 0;
    }
};

fn launchStep(step: hip.Function, cfg: hip.launch.Config, stream: hip.Stream, counter: hip.DeviceBuffer) !void {
    var a: hip.launch.Args = .{};
    try a.add(counter.base());
    try a.add(@as(u64, 1));
    try hip.launch.launch(step, cfg, stream, &a);
}

fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    return xs[xs.len / 2];
}

fn f32s(x: hip.DeviceBuffer) ?[*]f32 {
    return @ptrCast(@alignCast(x.ptr));
}

const Kernel = enum { rms, affine_row, affine_rows, conv, gated_delta, causal, mix };

const Buffers = struct { act: hip.DeviceBuffer, f32a: hip.DeviceBuffer, f32b: hip.DeviceBuffer, words: hip.DeviceBuffer, table: hip.DeviceBuffer, cache: hip.DeviceBuffer, pos: hip.DeviceBuffer };

/// One call of `k` on zeroed buffers, with a decode row's shapes (a 1024-wide row, a 6144-channel conv, one head).
fn one(l: *const hip.Launcher, k: Kernel, b: Buffers, fp16: bool, s: hip.abi.Stream) !void {
    const act: c_int = if (fp16) 1 else 2;
    switch (k) {
        .rms => try l.tf_rms(b.act.ptr, f32s(b.f32a), b.act.ptr, act, 1, 1024, 1e-6, s),
        .affine_row, .affine_rows => {
            const m: c_int = if (k == .affine_row) 1 else 4;
            try l.tf_affine(b.act.ptr, b.words.ptr, b.table.ptr, b.table.ptr, 1, b.f32b.ptr, m, 1024, 1024, 8, 64, 0, @intFromBool(fp16), s, null, 1, 0);
        },
        .conv => try l.tf_conv_decode(f32s(b.f32a), f32s(b.f32b), f32s(b.f32a), f32s(b.f32b), 1, 6144, 4, s),
        .gated_delta => {
            const f: ?[*]f32 = f32s(b.f32a);
            try l.tf_gated_delta(f, f, f, f, f, f32s(b.f32b), f, 1, 1, 16, 16, 128, 128, s, null);
        },
        .causal => {
            const sh: c_longlong = 256 * 256;
            const o: ?[*]f32 = f32s(b.f32b);
            try l.tf_causal(f32s(b.f32a), b.cache.ptr, b.cache.ptr, o, 1, 1, 256, 8, 2, 256, 0.0625, 0, 0, sh, 256, 0, sh, 256, if (fp16) 0 else 1, o, o, o, s, @ptrCast(@alignCast(b.pos.ptr)));
        },
        .mix => inline for (.{ Kernel.rms, .affine_row, .conv, .gated_delta, .rms, .affine_row, .causal }) |each| try one(l, each, b, fp16, s),
    }
}
