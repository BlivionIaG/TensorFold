//! The chunked DeltaNet prefill and the token-serial kernel against a float64 recurrence on random inputs
//! and, with `--bench`, the two kernels' time at the model's head counts.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const Gpu = check.Gpu;

const d = 128;

const Shape = struct { length: usize, key_heads: usize, value_heads: usize };

const Rng = struct {
    s: u64,

    fn next(r: *Rng) u64 {
        r.s ^= r.s >> 12;
        r.s ^= r.s << 25;
        r.s ^= r.s >> 27;
        return r.s *% 2685821657736338717;
    }

    fn uniform(r: *Rng) f64 {
        return (@as(f64, @floatFromInt(r.next() >> 11)) + 0.5) / 9007199254740992.0;
    }

    fn normal(r: *Rng) f64 {
        const a = r.uniform();
        const b = r.uniform();
        return @sqrt(-2 * @log(a)) * @cos(2 * std.math.pi * b);
    }
};

const Inputs = struct {
    q: []f32,
    k: []f32,
    v: []f32,
    gate: []f32,
    beta: []f32,
    state: []f32,

    /// L2-normalized q and k, gates exp(-A softplus(x)) in (0, 1], beta = sigmoid, a state of 0.1 scale.
    fn make(gpa: std.mem.Allocator, sh: Shape, seed: u64) !Inputs {
        var rng: Rng = .{ .s = seed };
        const qk = sh.length * sh.key_heads * d;
        const vy = sh.length * sh.value_heads * d;
        const gb = sh.length * sh.value_heads;
        var in: Inputs = undefined;
        in.q = try gpa.alloc(f32, qk);
        in.k = try gpa.alloc(f32, qk);
        in.v = try gpa.alloc(f32, vy);
        in.gate = try gpa.alloc(f32, gb);
        in.beta = try gpa.alloc(f32, gb);
        in.state = try gpa.alloc(f32, sh.value_heads * d * d);
        for ([_][]f32{ in.q, in.k }) |a| {
            var row: usize = 0;
            while (row < a.len / d) : (row += 1) {
                var ss: f64 = 0;
                for (a[row * d ..][0..d]) |*x| {
                    const z = rng.normal();
                    x.* = @floatCast(z);
                    ss += z * z;
                }
                const inv: f32 = @floatCast(1 / @sqrt(ss));
                for (a[row * d ..][0..d]) |*x| x.* *= inv;
            }
        }
        for (in.v) |*x| x.* = @floatCast(rng.normal());
        for (in.state) |*x| x.* = @floatCast(0.1 * rng.normal());
        const heads = sh.value_heads;
        for (in.gate, in.beta, 0..) |*g, *b, i| {
            const h = i % heads;
            const a_log = 0.5 + 7.5 * @as(f64, @floatFromInt(h)) / @as(f64, @floatFromInt(heads));
            const x = rng.normal() - 2;
            const sp = if (x > 20) x else @log(1 + @exp(x));
            g.* = @floatCast(@exp(-a_log * sp));
            b.* = @floatCast(1 / (1 + @exp(-rng.normal())));
        }
        return in;
    }

    fn free(in: Inputs, gpa: std.mem.Allocator) void {
        for ([_][]f32{ in.q, in.k, in.v, in.gate, in.beta, in.state }) |a| gpa.free(a);
    }
};

/// The recurrence in float64: y (L, Hv, d) and the final state.
fn reference(gpa: std.mem.Allocator, sh: Shape, in: Inputs, y: []f64, state: []f64) !void {
    const s = try gpa.alloc(f64, d * d);
    defer gpa.free(s);
    const group = sh.value_heads / sh.key_heads;
    for (0..sh.value_heads) |h| {
        for (s, in.state[h * d * d ..][0 .. d * d]) |*a, b| a.* = b;
        const kh = h / group;
        for (0..sh.length) |t| {
            const gb = t * sh.value_heads + h;
            const decay: f64 = in.gate[gb];
            const beta: f64 = in.beta[gb];
            const k = in.k[(t * sh.key_heads + kh) * d ..][0..d];
            const q = in.q[(t * sh.key_heads + kh) * d ..][0..d];
            const v = in.v[gb * d ..][0..d];
            for (0..d) |r| {
                const row = s[r * d ..][0..d];
                var kv: f64 = 0;
                for (row, k) |*a, kk| {
                    a.* *= decay;
                    kv += a.* * kk;
                }
                const delta = (v[r] - kv) * beta;
                var out: f64 = 0;
                for (row, k, q) |*a, kk, qq| {
                    a.* += kk * delta;
                    out += a.* * qq;
                }
                y[gb * d + r] = out;
            }
        }
        @memcpy(state[h * d * d ..][0 .. d * d], s);
    }
}

const Err = struct { abs: f64, rel: f64, rms: f64 };

/// Largest absolute error, that over the reference's largest value, and the relative RMS error.
fn compare(got: []const f32, want: []const f64) Err {
    var scale: f64 = 0;
    var max: f64 = 0;
    var se: f64 = 0;
    var sw: f64 = 0;
    for (got, want) |g, w| {
        const e = @abs(@as(f64, g) - w);
        max = @max(max, e);
        scale = @max(scale, @abs(w));
        se += e * e;
        sw += w * w;
    }
    return .{ .abs = max, .rel = max / scale, .rms = @sqrt(se / sw) };
}

const Device = struct {
    gpu: Gpu,
    lib: hip.rocm.Library,
    stream: hip.Stream,
    arena: hip.Arena,

    fn open(gpu: Gpu, arena_bytes: usize) !*Device {
        const caps = try gpu.ctx.caps();
        const dev = try gpu.gpa.create(Device);
        errdefer gpu.gpa.destroy(dev);
        dev.gpu = gpu;
        dev.lib = try hip.rocm.Library.open(gpu.d, caps, try check.policyOf(gpu));
        errdefer dev.lib.close();
        dev.stream = try hip.Stream.init(gpu.d, true);
        errdefer dev.stream.deinit();
        dev.arena = try hip.Arena.init(gpu.d, arena_bytes);
        return dev;
    }

    fn close(dev: *Device) void {
        dev.arena.deinit();
        dev.stream.deinit();
        dev.lib.close();
        dev.gpu.gpa.destroy(dev);
    }

    fn ops(dev: *Device) hip.ops.Ops {
        return .{ .lib = &dev.lib, .stream = dev.stream.handle, .arena = &dev.arena };
    }
};

const Run = struct {
    q: hip.DeviceBuffer,
    k: hip.DeviceBuffer,
    v: hip.DeviceBuffer,
    gate: hip.DeviceBuffer,
    beta: hip.DeviceBuffer,
    y: hip.DeviceBuffer,
    state: hip.DeviceBuffer,

    fn upload(gpu: Gpu, in: Inputs, sh: Shape) !Run {
        var r: Run = undefined;
        r.q = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.q));
        errdefer r.q.free();
        r.k = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.k));
        errdefer r.k.free();
        r.v = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.v));
        errdefer r.v.free();
        r.gate = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.gate));
        errdefer r.gate.free();
        r.beta = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.beta));
        errdefer r.beta.free();
        r.y = try hip.DeviceBuffer.alloc(gpu.d, sh.length * sh.value_heads * d * 4);
        errdefer r.y.free();
        r.state = try hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(in.state));
        return r;
    }

    fn free(r: *Run) void {
        for ([_]*hip.DeviceBuffer{ &r.q, &r.k, &r.v, &r.gate, &r.beta, &r.y, &r.state }) |b| b.free();
    }

    fn reset(r: Run, in: Inputs) !void {
        try r.state.upload(0, std.mem.sliceAsBytes(in.state));
        try r.y.fill8(0, null);
    }

    /// The serial kernel straight from the library (`serial`), or Ops.gatedDelta's choice.
    fn launch(r: Run, dev: *Device, sh: Shape, serial: bool) !void {
        if (serial) {
            const f = struct {
                fn of(b: hip.DeviceBuffer) ?[*]f32 {
                    return @ptrFromInt(b.ptr);
                }
            }.of;
            try dev.lib.call("tf_gated_delta", .{ f(r.q), f(r.k), f(r.v), f(r.gate), f(r.beta), f(r.state), f(r.y), 1, @as(c_int, @intCast(sh.length)), @as(c_int, @intCast(sh.key_heads)), @as(c_int, @intCast(sh.value_heads)), d, d, dev.stream.handle, null });
        } else {
            dev.arena.reset();
            try dev.ops().gatedDelta(r.q.ptr, r.k.ptr, r.v.ptr, r.gate.ptr, r.beta.ptr, r.state.ptr, r.y.ptr, sh.length, sh.key_heads, sh.value_heads, d, d, null);
        }
    }
};

fn download(gpu: Gpu, b: hip.DeviceBuffer, out: []f32) !void {
    _ = gpu;
    try b.download(0, std.mem.sliceAsBytes(out));
}

/// The chunked kernel against the float64 recurrence (`check`), and the two kernels' time at the model's head counts (`bench`).
pub fn run(gpu: Gpu) !void {
    const shapes = [_]Shape{
        .{ .length = 64, .key_heads = 2, .value_heads = 4 },
        .{ .length = 200, .key_heads = 2, .value_heads = 4 },
        .{ .length = 1000, .key_heads = 2, .value_heads = 4 },
        .{ .length = 4170, .key_heads = 2, .value_heads = 4 },
    };
    const dev = try Device.open(gpu, 1 << 30);
    defer dev.close();
    for (shapes) |sh| {
        const in = try Inputs.make(gpu.gpa, sh, 0x9e3779b97f4a7c15 + sh.length);
        defer in.free(gpu.gpa);
        const ry = try gpu.gpa.alloc(f64, sh.length * sh.value_heads * d);
        defer gpu.gpa.free(ry);
        const rs = try gpu.gpa.alloc(f64, sh.value_heads * d * d);
        defer gpu.gpa.free(rs);
        try reference(gpu.gpa, sh, in, ry, rs);
        var run_dev = try Run.upload(gpu, in, sh);
        defer run_dev.free();
        const y = try gpu.gpa.alloc(f32, ry.len);
        defer gpu.gpa.free(y);
        const st = try gpu.gpa.alloc(f32, rs.len);
        defer gpu.gpa.free(st);
        var worst: f64 = 0;
        for ([_]bool{ true, false }) |serial| {
            try run_dev.reset(in);
            try run_dev.launch(dev, sh, serial);
            try dev.stream.synchronize();
            try download(gpu, run_dev.y, y);
            try download(gpu, run_dev.state, st);
            const ey = compare(y, ry);
            const es = compare(st, rs);
            if (!serial) worst = @max(ey.rel, es.rel);
            check.step("gdn L={d} Hk={d} Hv={d} {s}: y abs {e:.2} rel {e:.2} rms {e:.2} | state abs {e:.2} rel {e:.2} rms {e:.2}\n", .{ sh.length, sh.key_heads, sh.value_heads, if (serial) "serial " else "chunked", ey.abs, ey.rel, ey.rms, es.abs, es.rel, es.rms });
        }
        try check.expect(worst < 5e-3, "chunked L={d}: error {e} over the reference's scale", .{ sh.length, worst });
    }
    check.pass("gdn: the chunked DeltaNet prefill within 5e-3 of the float64 recurrence at four lengths", .{});
}

pub fn bench(gpu: Gpu, rows: usize) !void {
    const sh: Shape = .{ .length = rows, .key_heads = 16, .value_heads = 32 };
    const dev = try Device.open(gpu, 1 << 30);
    defer dev.close();
    var rng: Rng = .{ .s = 12345 };
    const in: Inputs = .{
        .q = try gpu.gpa.alloc(f32, rows * sh.key_heads * d),
        .k = try gpu.gpa.alloc(f32, rows * sh.key_heads * d),
        .v = try gpu.gpa.alloc(f32, rows * sh.value_heads * d),
        .gate = try gpu.gpa.alloc(f32, rows * sh.value_heads),
        .beta = try gpu.gpa.alloc(f32, rows * sh.value_heads),
        .state = try gpu.gpa.alloc(f32, sh.value_heads * d * d),
    };
    defer in.free(gpu.gpa);
    for ([_][]f32{ in.q, in.k, in.v, in.state }) |a| for (a) |*x| {
        x.* = @floatCast((rng.uniform() - 0.5) * 0.2);
    };
    for (in.gate, in.beta) |*g, *b| {
        g.* = @floatCast(0.8 + 0.2 * rng.uniform());
        b.* = @floatCast(rng.uniform());
    }
    var run_dev = try Run.upload(gpu, in, sh);
    defer run_dev.free();
    for ([_]bool{ true, false }) |serial| {
        var times: [5]f64 = undefined;
        for (0..6) |rep| {
            try run_dev.reset(in);
            try dev.stream.synchronize();
            const t0 = check.now(gpu.io);
            try run_dev.launch(dev, sh, serial);
            try dev.stream.synchronize();
            const dt = @as(f64, @floatFromInt(check.now(gpu.io) - t0)) / 1e6;
            if (rep > 0) times[rep - 1] = dt;
        }
        std.debug.print("RESULT gdn bench {d} rows, {s}: {d:.2} ms\n", .{ rows, if (serial) "serial " else "chunked", check.median(&times) });
    }
}
