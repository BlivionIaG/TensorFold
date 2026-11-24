//! LoopCD's combine (h' = h_R + w (h_R - h_1)) under an FP64 oracle and a mutation control.
const std = @import("std");
const mtl = @import("metal");
const contract = @import("contract");
const c = contract.spec;
const g = contract.gates;
const rig_ = @import("rig.zig");
const host = @import("tensorfold").inkling2_metal_host.loop;
const p = host.plan;
const sources = @import("inkling2_metal");

/// An exact binary fraction, so every bf16 input's contrast stays exact in fp32 until the output rounds.
pub const w: f32 = 0.5;
const label = "metal inkling2 loop loopcd combine tf_inkling2_loop_loopcd_combine";
const symbol = "tf_inkling2_loop_loopcd_combine";
const mutation = .{ .from = "fma(a.w, r - p, r)", .to = "fma(a.w, r + p, r)" };

fn depth(_: c.Shape) usize {
    return 2;
}

pub const Probe = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    pipes: host.Pipelines,
    shape: c.Shape = undefined,
    served: [4]c.Shape = .{
        .{ .rows = 1, .n = 2304, .k = 2 },
        .{ .rows = 32, .n = 2304, .k = 1 },
        .{ .rows = 3, .n = 96, .k = 3 },
        .{ .rows = 17, .n = 128, .k = 1 },
    },
    hr: rig_.Ref = undefined,
    h1: rig_.Ref = undefined,
    out: rig_.Ref = undefined,
    bits: [2][]u16 = @splat(&.{}),

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, text: []const u8) !Probe {
        return .{
            .a = a,
            .arena = .init(a),
            .rig = .{ .a = a, .device = device, .queue = queue },
            .pipes = try host.Pipelines.loadSource(device, text),
        };
    }
    pub fn deinit(self: *Probe) void {
        self.rig.deinit();
        self.arena.deinit();
        self.pipes.deinit();
    }

    const vt = g.hooks(Probe);
    pub fn kernel(self: *Probe) g.Kernel {
        return .{ .ctx = self, .vt = &vt, .spec = .{
            .name = label,
            .accum = .none,
            .depth = depth,
            .ops = 1,
            .out = .bf16,
            .served = &self.served,
            .sums = .{},
        } };
    }
    pub fn count(self: *Probe) usize {
        return self.shape.rows * self.shape.k * self.shape.n;
    }

    pub fn prepare(self: *Probe, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.rows > 64 or shape.n < 32 or shape.n > p.hidden or shape.k == 0 or shape.k > 4) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        self.shape = shape;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const n = self.count();
        for (0..2) |i| {
            self.bits[i] = try self.arena.allocator().alloc(u16, n);
            for (self.bits[i]) |*b| b.* = c.toBf16(rng.float(f32) * 4 - 2);
        }
        if (fill == .nonfinite) {
            self.bits[0][0] = 0x7fc0;
            self.bits[1][1] = 0x7f80;
            self.bits[0][2] = 0xff80;
        }
        self.hr = try self.rig.input(std.mem.sliceAsBytes(self.bits[0]));
        self.h1 = try self.rig.input(std.mem.sliceAsBytes(self.bits[1]));
        self.out = try self.rig.output(n * 2);
        return true;
    }

    fn at(ref: rig_.Ref) host.Ref {
        return .{ .buf = ref.buf, .off = ref.off };
    }
    pub fn run(self: *Probe) !void {
        @memcpy(self.rig.bytes(self.out, self.count() * 2), std.mem.sliceAsBytes(self.bits[0]));
        const pass = self.rig.begin();
        host.combine(&self.pipes, pass.e, at(self.out), at(self.h1), .{
            .rows = @intCast(self.shape.rows),
            .hidden = @intCast(self.shape.n),
            .planes = @intCast(self.shape.k),
            .stride = @intCast(self.shape.rows * self.shape.n),
            .w = w,
        });
        try self.rig.finish(pass);
    }

    pub fn reference(self: *Probe, want: []f64, mag: []f64) !void {
        for (0..self.count()) |i| {
            const r = c.fromBf16(self.bits[0][i]);
            const q = c.fromBf16(self.bits[1][i]);
            want[i] = r + @as(f64, w) * (r - q);
            mag[i] = @abs(r) + @abs(@as(f64, w) * (r - q));
        }
    }

    pub fn simulate(self: *Probe, acc: c.Accum, order: g.Order, out: []f64) !void {
        _ = acc;
        _ = order; // elementwise: every output stands alone
        for (0..self.count()) |i| out[i] = c.fromBf16(self.bits[0][i]) + @as(f64, w) * (c.fromBf16(self.bits[0][i]) - c.fromBf16(self.bits[1][i]));
    }
};

pub const Suite = struct {
    a: std.mem.Allocator,
    device: mtl.Device,
    queue: mtl.Queue,
    kernels: *std.ArrayList(g.Kernel),
    probes: std.ArrayList(*Probe) = .empty,

    pub fn add(s: *Suite, control: bool, only: ?[]const u8) !void {
        const name = if (control) label ++ " control" else label;
        if (only) |text| if (std.mem.indexOf(u8, name, text) == null) return;
        const probe = try s.a.create(Probe);
        errdefer s.a.destroy(probe);
        // The control plants the fault when the shipped kernel is exact, and restores the reference when a mutation is shipped.
        const text = if (control) blk: {
            if (std.mem.indexOf(u8, sources.loop, mutation.to) != null) break :blk try std.mem.replaceOwned(u8, s.a, sources.loop, mutation.to, mutation.from);
            break :blk try std.mem.replaceOwned(u8, s.a, sources.loop, mutation.from, mutation.to);
        } else try s.a.dupe(u8, sources.loop);
        defer s.a.free(text);
        probe.* = try Probe.init(s.a, s.device, s.queue, text);
        errdefer probe.deinit();
        try s.probes.append(s.a, probe);
        var k = probe.kernel();
        if (control) {
            k.spec.name = name;
            k.control = .shapes;
        }
        try s.kernels.append(s.a, k);
    }

    pub fn collect(s: *Suite, only: ?[]const u8) !void {
        try s.add(false, only);
        try s.add(true, only);
    }

    pub fn deinit(s: *Suite) void {
        for (s.probes.items) |probe| {
            probe.deinit();
            s.a.destroy(probe);
        }
        s.probes.deinit(s.a);
    }
};
