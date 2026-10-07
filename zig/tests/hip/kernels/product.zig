//! One random packed affine product on the host and the device, launched by any registry entry, with its reference.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const data = @import("data.zig");
const ref = @import("reference.zig");
const cases = @import("cases.zig");
const Gpu = check.Gpu;
const Kernels = hip.affine.Kernels;
const Entry = hip.affine.Entry;

pub const Launcher = @typeInfo(@FieldType(hip.rocm.Library, "zig")).optional.child;

/// The device a group runs on: the registry's kernels, a stream and the seeded generator.
pub const Rig = struct {
    gpu: Gpu,
    launcher: *const Launcher,
    kernels: *const Kernels,
    stream: hip.Stream,
    fp16: bool,
    rng: data.Rng = .{ .state = 0x9E3779B97F4A7C15 },
    /// Timings are printed.
    bench: bool = false,
    reps: usize = 5,
    /// A running hash of the bytes every launch wrote: the same before and after a change that keeps its bits.
    digest: u64 = 0,
};

pub const Product = struct {
    rig: *Rig,
    c: cases.Case,
    hx: []u16,
    hw: []u32,
    hs: []u16,
    hb: []u16,
    expert_of: []u32 = &.{},
    x: hip.DeviceBuffer,
    words: hip.DeviceBuffer,
    scale: hip.DeviceBuffer,
    bias: hip.DeviceBuffer,
    plan: ?hip.DeviceBuffer = null,
    members: ?hip.DeviceBuffer = null,
    /// A plan's items and the most rows in one; the output rows a launch writes.
    items: usize = 1,
    max_rows: usize,
    out_rows: usize,
    /// Copies of the tables back to back, so timed launches read weights from DRAM.
    copies: usize,
    copy_words: usize,
    copy_tab: usize,

    pub fn init(rig: *Rig, c: cases.Case) !Product {
        const gpa = rig.gpu.gpa;
        const experts = @max(c.experts, 1);
        const words_row = c.k * c.bits / 32;
        const groups = c.k / c.group;
        const pairs = c.rows * c.slots;
        var p: Product = undefined;
        p.rig = rig;
        p.c = c;
        p.items = 1;
        p.max_rows = c.rows;
        p.out_rows = if (c.routed()) pairs else c.rows;
        p.plan = null;
        p.members = null;
        p.expert_of = &.{};
        p.copy_words = std.mem.alignForward(usize, experts * c.n * words_row, 64);
        p.copy_tab = std.mem.alignForward(usize, experts * c.n * groups, 128);
        const copy_bytes = p.copy_words * 4 + p.copy_tab * 4;
        p.copies = if (rig.bench and !c.routed() and c.path == .decode) std.math.clamp((384 << 20) / copy_bytes + 1, 2, 400) else 1;
        const x_rows: usize = if (!c.routed()) c.rows else if (c.down) pairs else c.rows;
        p.hx = try data.fill(gpa, u16, x_rows * c.k, &rig.rng, if (rig.fp16) data.makeX16 else data.makeXB);
        errdefer gpa.free(p.hx);
        p.hw = try data.fill(gpa, u32, p.copy_words, &rig.rng, data.makeWord);
        errdefer gpa.free(p.hw);
        p.hs = try data.fill(gpa, u16, p.copy_tab, &rig.rng, data.makeScale);
        errdefer gpa.free(p.hs);
        p.hb = try data.fill(gpa, u16, p.copy_tab, &rig.rng, data.makeBias);
        errdefer gpa.free(p.hb);
        p.x = try data.toDevice(rig.gpu, p.hx);
        errdefer p.x.free();
        p.words = try hip.DeviceBuffer.alloc(rig.gpu.d, p.copies * p.copy_words * 4);
        errdefer p.words.free();
        p.scale = try hip.DeviceBuffer.alloc(rig.gpu.d, p.copies * p.copy_tab * 2);
        errdefer p.scale.free();
        p.bias = try hip.DeviceBuffer.alloc(rig.gpu.d, p.copies * p.copy_tab * 2);
        errdefer p.bias.free();
        for (0..p.copies) |i| {
            try p.words.upload(i * p.copy_words * 4, std.mem.sliceAsBytes(p.hw));
            try p.scale.upload(i * p.copy_tab * 2, std.mem.sliceAsBytes(p.hs));
            try p.bias.upload(i * p.copy_tab * 2, std.mem.sliceAsBytes(p.hb));
        }
        if (c.routed()) try p.makePlan(pairs);
        return p;
    }

    /// Each pair draws an expert; the pairs sorted by expert are cut into items of at most `item_rows`.
    fn makePlan(p: *Product, pairs: usize) !void {
        const gpa = p.rig.gpu.gpa;
        const c = p.c;
        p.expert_of = try gpa.alloc(u32, pairs);
        for (p.expert_of, 0..) |*e, i| e.* = if (c.even) @intCast(i % c.experts) else @intCast(p.rig.rng.next() % c.experts);
        const order = try gpa.alloc(i32, pairs);
        defer gpa.free(order);
        for (order, 0..) |*o, i| o.* = @intCast(i);
        std.mem.sort(i32, order, p.expert_of, struct {
            fn less(of: []u32, a: i32, b: i32) bool {
                const ea = of[@intCast(a)];
                const eb = of[@intCast(b)];
                return if (ea != eb) ea < eb else a < b;
            }
        }.less);
        const items = try gpa.alloc(i32, pairs * 3);
        defer gpa.free(items);
        var count: usize = 0;
        var i: usize = 0;
        p.max_rows = 1;
        while (i < pairs) {
            const e = p.expert_of[@intCast(order[i])];
            var j = i;
            while (j < pairs and j - i < c.item_rows and p.expert_of[@intCast(order[j])] == e) j += 1;
            items[count * 3 ..][0..3].* = .{ @intCast(e), @intCast(i), @intCast(j - i) };
            p.max_rows = @max(p.max_rows, j - i);
            count += 1;
            i = j;
        }
        p.items = count;
        p.plan = try data.toDevice(p.rig.gpu, items[0 .. count * 3]);
        p.members = try data.toDevice(p.rig.gpu, order);
    }

    pub fn deinit(p: *Product) void {
        const gpa = p.rig.gpu.gpa;
        gpa.free(p.hx);
        gpa.free(p.hw);
        gpa.free(p.hs);
        gpa.free(p.hb);
        gpa.free(p.expert_of);
        p.x.free();
        p.words.free();
        p.scale.free();
        p.bias.free();
        if (p.plan) |*b| b.free();
        if (p.members) |*b| b.free();
    }

    pub fn flops(p: Product) f64 {
        return 2.0 * @as(f64, @floatFromInt(p.out_rows)) * @as(f64, @floatFromInt(p.c.n)) * @as(f64, @floatFromInt(p.c.k));
    }

    /// The bytes of weights and tables a launch reads (every expert an item names, once).
    pub fn bytes(p: Product) f64 {
        const per = @as(f64, @floatFromInt(p.c.n)) * @as(f64, @floatFromInt(p.c.k * p.c.bits / 8 + 4 * (p.c.k / p.c.group)));
        if (!p.c.routed()) return per;
        var seen = std.bit_set.DynamicBitSetUnmanaged.initEmpty(p.rig.gpu.gpa, p.c.experts) catch return per;
        defer seen.deinit(p.rig.gpu.gpa);
        for (p.expert_of) |e| seen.set(e);
        return per * @as(f64, @floatFromInt(seen.count()));
    }

    pub fn arg(p: Product, out: u64, copy: usize) hip.affine.Arg {
        const c = p.c;
        return .{
            .x = p.x.ptr,
            .words = p.words.ptr + copy * p.copy_words * 4,
            .scale = .{ .p = p.scale.ptr + copy * p.copy_tab * 2, .kind = 1 },
            .bias = .{ .p = p.bias.ptr + copy * p.copy_tab * 2, .kind = 1 },
            .out = out,
            .m = @intCast(if (c.routed()) p.max_rows else c.rows),
            .n = @intCast(c.n),
            .k = @intCast(c.k),
            .bits = c.bits,
            .group = c.group,
            .fp16 = @intFromBool(p.rig.fp16),
            .route = if (c.routed()) .{ .items = p.plan.?.ptr, .members = p.members.?.ptr, .x_div = @intCast(if (c.down) 1 else c.slots) } else .{},
        };
    }

    pub fn outBytes(p: Product, rounded: bool) usize {
        return p.out_rows * p.c.n * @as(usize, if (rounded) 2 else 4);
    }

    pub fn problem(p: *const Product) ref.Problem {
        return .{ .fp16 = p.rig.fp16, .n = p.c.n, .k = p.c.k, .bits = p.c.bits, .group = p.c.group, .x = p.hx, .words = p.hw, .scale = p.hs, .bias = p.hb };
    }

    /// The float64 value of output (`row`, `col`).
    pub fn reference(p: *const Product, row: usize, col: usize) ref.Value {
        if (!p.c.routed()) return ref.reference(p.problem(), row, 0, col);
        return ref.reference(p.problem(), if (p.c.down) row else row / p.c.slots, p.expert_of[row], col);
    }
};

/// The shape a registry entry is asked about: a split entry is asked for two parts.
pub fn shapeFor(a: hip.affine.Arg, items: usize, e: *const Entry) hip.registry.Shape {
    var s = Kernels.shapeOf(a, @intCast(items));
    if (e.family == .split) s.parts = 2;
    return s;
}

pub fn shapeOf(p: *const Product, e: *const Entry) hip.registry.Shape {
    return shapeFor(p.arg(0, 0), p.items, e);
}

/// Whether `e` is an entry of the launch's (op, path) that the GPU has and whose shape takes it.
pub fn takesArg(k: *const Kernels, e: *const Entry, a: hip.affine.Arg, items: usize, path: hip.registry.Path) bool {
    const op: hip.registry.Op = if (a.route.items != 0) .routed else .project;
    if (e.format != .mlx or e.op != op or e.path != path) return false;
    var env = k.env();
    env.matrix_on = env.matrix;
    return e.caps(env) and e.fits(shapeFor(a, items, e));
}

/// Entries of the product's (op, path) that the GPU has and whose shapes take it.
pub fn takes(p: *const Product, e: *const Entry) bool {
    return takesArg(p.rig.kernels, e, p.arg(0, 0), p.items, p.c.path);
}

/// The entries the registry picks between at some row count under `env`, for the shape of `a`.
pub fn pickable(k: *const Kernels, a: hip.affine.Arg, items: usize, path: hip.registry.Path, env: hip.registry.Env, out: *std.ArrayList(*const Entry), gpa: std.mem.Allocator) !void {
    const op: hip.registry.Op = if (a.route.items != 0) .routed else .project;
    var shape = Kernels.shapeOf(a, @intCast(items));
    // a lane round keeps one family at every row count; outside one the decode tiles change with the rows
    shape.round = path == .decode;
    for (1..hip.registry.rule_rows + 1) |m| {
        shape.m = @intCast(m);
        const e = k.reg.select(env, .mlx, op, path, shape) orelse continue;
        if (std.mem.indexOfScalar(*const Entry, out.items, e) == null) try out.append(gpa, e);
    }
}

/// Launches `e` on the product into `out`, rounded to the activation type when `rounded`.
pub fn launchEntry(p: *const Product, e: *const Entry, out: hip.DeviceBuffer, rounded: bool, partial: u64, copy: usize) hip.Error!void {
    var a = p.arg(out.ptr, copy);
    if (rounded) {
        a.out16 = a.out;
        a.out = 0;
    }
    try e.launch(p.rig.kernels, .{
        .d = p.rig.gpu.d,
        .s = p.rig.stream.handle,
        .arg = a,
        .items = @intCast(p.items),
        .partial = partial,
        .parts = if (e.family == .split) 2 else 1,
    });
}

/// A hash of a device buffer's first `len` bytes, read through windows.
pub fn digestOf(gpu: Gpu, buf: hip.DeviceBuffer, len: usize) !u64 {
    const win = 32 << 20;
    const host = try gpu.gpa.alloc(u8, @min(win, len));
    defer gpu.gpa.free(host);
    var h: u64 = 0;
    var at: usize = 0;
    while (at < len) : (at += win) {
        const n = @min(win, len - at);
        try buf.download(at, host[0..n]);
        h = std.hash.Wyhash.hash(h, host[0..n]);
    }
    return h;
}
