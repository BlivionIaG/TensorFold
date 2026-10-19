//! The stream tile with Nemotron's relu^2 as its epilogue under the contract on HIP: the plain tile then tf_nemo_relu2.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const nonfiniteMatches = @import("relu2_nonfinite.zig").matches;

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const Kernels = @FieldType(hip.Launcher, "affine");
/// launches/affine.zig's Arg, as relu2Go takes it.
pub const Arg = @typeInfo(@TypeOf(Kernels.relu2Go)).@"fn".param_types[2].?;

const bits = 4; // Nano's MLX checkpoints
const group = 64;

/// A group's codes by dot2 pairs, its scale and bias, then the groups' sum: one fp32 chain over k.
fn depth(s: c.Shape) usize {
    return s.k / 2 + s.k / group + 8;
}

const names = [2][]const u8{
    "hip affine_dot2_stream_relu2 (4-bit, fp16)",
    "hip affine_dot2_stream_relu2 (4-bit, bf16)",
};
const bf16_why = "the bound's two roundings to bf16 cap the miss of a sum relu turns to 0 at 85x";

pub fn spec(k: kit.Kind) c.Spec {
    return .{
        .name = names[@intFromBool(k == .bf16)],
        .accum = .f32,
        .depth = depth,
        .ops = 1,
        // the sum rounded to the kind, then squared
        .mids = 2,
        .out = kit.elem(k),
        .replaces = "affine_dot2_stream then tf_nemo_relu2",
        .nan = .{ .stated = "the unfused bytes; only the shipped stream tile may turn an infinite sum into NaN" },
        // a one-row layer's routed fc1 and shared up, and two rows
        .served = &.{
            .{ .rows = 1, .n = 1856, .k = 2688 },
            .{ .rows = 1, .n = 3712, .k = 2688 },
            .{ .rows = 2, .n = 1856, .k = 2688 },
        },
        .sums = if (k == .bf16) .{ .f16 = 40, .bf16 = 40, .why = bf16_why } else .{},
    };
}

/// x [rows][k] of the kind times the MLX 4-bit matrix [n][k]: relu(round(y))^2 [rows][n] of the kind.
pub const Relu2 = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    k: kit.Kind,
    plain: bool = false,
    arena: std.heap.ArenaAllocator,
    f: ?contract.affine.Affine = null,
    arg: Arg = undefined,
    sum: u64 = 0,
    base: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Relu2 {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .k = kit.act(), .arena = .init(a) };
    }

    pub fn deinit(self: *Relu2) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Relu2) g.Kernel {
        var s = spec(self.k);
        if (self.plain) {
            s.name = "hip affine_dot2_stream (4-bit, fp32 out)";
            s.out = .f32;
            s.ops = 0;
            s.mids = 0;
            s.replaces = null;
            s.nan = .propagate;
            s.sums = .{};
            s.known = .{ .gate = .nonfinite, .finding = "shipped affine_dot2_stream: an infinite input becomes NaN" };
        }
        return .{ .spec = s, .ctx = self, .vt = if (self.plain) &plain_vt else &vt };
    }

    const vt = g.hooks(Relu2);
    const plain_vt = blk: {
        var v = vt;
        v.check = null;
        v.replaced = null;
        break :blk v;
    };

    pub fn prepare(self: *Relu2, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k % group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        var f = try contract.affine.Affine.init(a, shape, bits, group);
        try f.fill(fill, seed);
        // gate 4's rows sum to a negative product, which relu would hide: x negated makes it positive
        if (fill == .stress) for (f.x) |*b| {
            b.* ^= 0x8000;
        };
        if (self.k == .f16) for (f.x) |*b| {
            if (@abs(c.fromBf16(b.*)) < 0x1p-14) b.* = 0; // fp16 normals only, so each bf16 value is exact
        };
        self.f = f;
        const xb = try a.alloc(u8, f.x.len * 2);
        for (f.x, 0..) |b, i| kit.put(self.k, xb, i, c.fromBf16(b));
        const w = try a.alloc(u32, shape.n * shape.k * bits / 32);
        f.words(w);
        const x = try self.rig.input(xb);
        const words = try self.rig.input(std.mem.sliceAsBytes(w));
        const scale = try self.rig.input(std.mem.sliceAsBytes(f.scales));
        const bias = try self.rig.input(std.mem.sliceAsBytes(f.biases));
        const out = try self.rig.output(f.outputs() * @as(usize, if (self.plain) 4 else 2));
        self.sum = try self.rig.scratch(f.outputs() * 4);
        self.base = try self.rig.scratch(f.outputs() * 2);
        self.arg = .{
            .x = x,
            .words = words,
            .scale = .{ .p = scale, .kind = 1 }, // bf16 tables
            .bias = .{ .p = bias, .kind = 1 },
            .out = if (self.plain) out else 0,
            .m = @intCast(shape.rows),
            .n = @intCast(shape.n),
            .k = @intCast(shape.k),
            .bits = bits,
            .group = group,
            .fp16 = @intFromBool(self.k == .f16),
            .out16 = if (self.plain) 0 else out,
        };
        return true;
    }

    pub fn run(self: *Relu2) !void {
        if (self.plain)
            try self.l.affine.streamGo(self.l.r, self.arg, .{}, self.arg.n, 1, self.rig.s.handle)
        else
            try self.l.affine.relu2Go(self.l.r, self.arg, 1, self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The plain stream tile's fp32 sums, then tf_nemo_relu2 into the kind.
    pub fn replaced(self: *Relu2) ![]const u8 {
        var plain = self.arg;
        plain.out = self.sum;
        plain.out16 = 0;
        try self.l.affine.streamGo(self.l.r, plain, .{}, plain.n, 1, self.rig.s.handle);
        const n = self.f.?.outputs();
        var a: hip.launch.Args = .{};
        for ([_]u64{ self.sum, self.base }) |v| try a.add(v);
        try a.add(@backingInt(self.k));
        try a.add(@as(c_longlong, @intCast(n)));
        const cfg: hip.launch.Config = .{ .grid = .{ .x = @intCast((n + 255) / 256) }, .block = .{ .x = 256 } };
        try hip.launch.launch(self.l.nemo.relu2, cfg, self.rig.s, &a);
        return self.rig.bytes(self.base, n * 2);
    }

    /// All original plants remain; only the independently observed shipped infinity mismatch is inherited.
    pub fn check(self: *Relu2, at: c.Gate) !?[]const u8 {
        if (at != .nonfinite) return null;
        const a = self.arena.allocator();
        const out = try a.dupe(u8, try self.rig.outputBytes());
        const base = try self.replaced();
        if (!std.mem.eql(u8, out, base)) return "non-finite output differs from the unfused kernels' bytes";
        const n = self.count();
        const raw = try a.alloc(f64, n);
        const mag = try a.alloc(f64, n);
        try self.f.?.reference(raw, mag);
        const sums = try self.rig.bytes(self.sum, n * 4);
        var inherited: usize = 0;
        for (raw, 0..) |w, i| {
            const y = c.read(kit.elem(self.k), out, i);
            const sum = c.read(.f32, sums, i);
            if (!nonfiniteMatches(w, sum, y)) return "class mismatch outside the shipped stream infinity finding";
            const r = if (std.math.isNan(w)) w else @max(w, 0);
            if (c.classOf(y) != c.classOf(r * r)) inherited += 1;
        }
        if (inherited > 0) {
            const msg = "INHERITED {s}: {d} shipped stream infinity mismatches\n";
            std.debug.print(msg, .{ spec(self.k).name, inherited });
        }
        return null;
    }

    pub fn count(self: *Relu2) usize {
        return self.f.?.outputs();
    }

    /// relu(y)^2 of the float64 product (NaN kept as torch keeps it), the square's slope on the product's magnitude.
    pub fn reference(self: *Relu2, want: []f64, mag: []f64) !void {
        try self.f.?.reference(want, mag);
        if (self.plain) return;
        for (want, mag) |*w, *m| {
            const r = if (std.math.isNan(w.*)) w.* else @max(w.*, 0);
            m.* *= 2 * @abs(r);
            w.* = r * r;
        }
    }

    pub fn simulate(self: *Relu2, acc: c.Accum, order: g.Order, out: []f64) !void {
        try self.f.?.simulate(acc, order, out);
        if (self.plain) return;
        for (out) |*w| {
            const r = if (std.math.isNan(w.*)) w.* else @max(w.*, 0);
            w.* = r * r;
        }
    }
};
