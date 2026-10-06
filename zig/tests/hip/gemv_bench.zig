//! The decode tiles of 1 to 8 rows: the stream tile against the previous tiles (TF_AFFINE_GEMV=old) on the engine's
//! decode shapes, dense and routed. Each launch reads weights nobody read for a few hundred MB, so the times are DRAM's.
//! Per case: microseconds a launch and GB/s of weights, scales and biases, and each tile's error against a float64
//! dequant(W) . x on sampled outputs. `tf-hip-test gemv [reps] [name filter]`.

const std = @import("std");
const hip = @import("hip");
const check = @import("check.zig");
const ref = @import("gemm_ref.zig");
const Gpu = check.Gpu;

/// `experts` > 0 is a routed product: `rows` tokens of `slots` experts each drawn from the stacked tables.
const Case = struct { name: []const u8, rows: usize = 1, n: usize, k: usize, bits: c_int = 4, group: c_int = 64, experts: usize = 0, slots: usize = 9, x_div: bool = false };

const cases = [_]Case{
    .{ .name = "35b lm_head", .n = 248320, .k = 2048 },
    .{ .name = "35b qkv", .n = 8192, .k = 2048 },
    .{ .name = "35b o", .n = 2048, .k = 4096 },
    .{ .name = "35b shared gate_up", .n = 1024, .k = 2048 },
    .{ .name = "35b shared down", .n = 2048, .k = 512 },
    .{ .name = "35b kv", .n = 512, .k = 2048 },
    .{ .name = "35b routed gate_up", .n = 1024, .k = 2048, .experts = 256, .x_div = true },
    .{ .name = "35b routed down", .n = 2048, .k = 512, .experts = 256 },
    .{ .name = "9b qkv", .n = 8192, .k = 4096 },
    .{ .name = "9b gate_up", .n = 24576, .k = 4096 },
    .{ .name = "9b down", .n = 4096, .k = 12288 },
    .{ .name = "9b lm_head", .n = 248320, .k = 4096 },
    .{ .name = "9b down 6-bit", .n = 4096, .k = 12288, .bits = 6 },
    .{ .name = "9b down 8-bit", .n = 4096, .k = 12288, .bits = 8 },
    .{ .name = "9b gate_up 6-bit", .n = 24576, .k = 4096, .bits = 6 },
    .{ .name = "9b gate_up 8-bit", .n = 24576, .k = 4096, .bits = 8 },
    .{ .name = "3-bit", .n = 8192, .k = 2048, .bits = 3 },
    .{ .name = "2-bit", .n = 8192, .k = 2048, .bits = 2 },
    .{ .name = "5-bit", .n = 8192, .k = 2048, .bits = 5 },
    .{ .name = "group 32", .n = 8192, .k = 2048, .group = 32 },
    .{ .name = "group 128", .n = 8192, .k = 2048, .group = 128 },
    .{ .name = "ragged", .n = 1001, .k = 2112 },
    .{ .name = "stride k4032", .n = 24576, .k = 4032 },
    .{ .name = "stride k4160", .n = 24576, .k = 4160 },
    .{ .name = "stride k2048", .n = 49152, .k = 2048 },
    .{ .name = "stride k8192", .n = 12288, .k = 8192 },
    .{ .name = "35b qkv x2", .rows = 2, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x4", .rows = 4, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x8", .rows = 8, .n = 8192, .k = 2048 },
    .{ .name = "35b qkv x16", .rows = 16, .n = 8192, .k = 2048 },
    .{ .name = "9b gate_up x12", .rows = 12, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x16", .rows = 16, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x4", .rows = 4, .n = 24576, .k = 4096 },
    .{ .name = "9b gate_up x8", .rows = 8, .n = 24576, .k = 4096 },
    .{ .name = "35b routed gate_up x2", .rows = 2, .n = 1024, .k = 2048, .experts = 256, .x_div = true },
    .{ .name = "35b routed gate_up x4", .rows = 4, .n = 1024, .k = 2048, .experts = 256, .x_div = true },
    .{ .name = "35b routed down x4", .rows = 4, .n = 2048, .k = 512, .experts = 256 },
    .{ .name = "35b routed gate_up x8", .rows = 8, .n = 1024, .k = 2048, .experts = 256, .x_div = true },
};

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

fn bf16Bits(v: f32) u16 {
    return @intCast(@as(u32, @bitCast(v)) >> 16);
}

fn f16Bits(v: f32) u16 {
    const h: f16 = @floatCast(v);
    return @bitCast(h);
}

fn upload(gpu: Gpu, host: anytype) !hip.DeviceBuffer {
    return hip.DeviceBuffer.fromHost(gpu.d, std.mem.sliceAsBytes(host));
}

/// One product's data: `copies` copies of the (stacked) weights back to back, one x, and the plans.
const Job = struct {
    k: *const hip.affine.Kernels,
    d: *const hip.Driver,
    stream: hip.abi.Stream,
    arg: hip.affine.Arg,
    routed: bool,
    items: c_int = 0,
};

fn launch(j: Job) hip.Error!void {
    if (j.routed) return j.k.routedWith(j.d, j.arg, j.items, j.stream, .gemm);
    return j.k.run(j.d, j.arg, 0, j.stream, 0, 1, false);
}

fn runCase(gpu: Gpu, kernels: *const hip.affine.Kernels, stream: hip.Stream, fp16: bool, rng: *Rng, c: Case, reps: usize) !f64 {
    const gpa = gpu.gpa;
    const bits: usize = @intCast(c.bits);
    const group: usize = @intCast(c.group);
    const words_row = c.k * bits / 32;
    const groups = c.k / group;
    const stacked = @max(c.experts, 1);
    const copy_words = std.mem.alignForward(usize, stacked * c.n * words_row, 64);
    const copy_tab = std.mem.alignForward(usize, stacked * c.n * groups, 128);
    const copy_bytes = copy_words * 4 + copy_tab * 4;
    const copies = std.math.clamp((384 << 20) / copy_bytes + 1, 2, 400);
    const iters = @max(copies, 24) * reps / 5;

    // one copy's random data, uploaded to every copy
    const hw = try gpa.alloc(u32, copy_words);
    defer gpa.free(hw);
    for (hw) |*w| w.* = @truncate(rng.next() >> 16);
    const hs = try gpa.alloc(u16, copy_tab);
    defer gpa.free(hs);
    for (hs) |*v| v.* = bf16Bits((rng.unit() + 1.0) / 16.0);
    const hb = try gpa.alloc(u16, copy_tab);
    defer gpa.free(hb);
    for (hb) |*v| v.* = bf16Bits(rng.unit() / 2.0);
    var words = try hip.DeviceBuffer.alloc(gpu.d, copies * copy_words * 4);
    defer words.free();
    var scale = try hip.DeviceBuffer.alloc(gpu.d, copies * copy_tab * 2);
    defer scale.free();
    var bias = try hip.DeviceBuffer.alloc(gpu.d, copies * copy_tab * 2);
    defer bias.free();
    for (0..copies) |i| {
        try words.upload(i * copy_words * 4, std.mem.sliceAsBytes(hw));
        try scale.upload(i * copy_tab * 2, std.mem.sliceAsBytes(hs));
        try bias.upload(i * copy_tab * 2, std.mem.sliceAsBytes(hb));
    }

    // the activations: a token's row, or a routed down product's row a pair
    const slots = c.slots;
    const pairs = c.rows * slots;
    const x_rows: usize = if (c.experts > 0 and !c.x_div) pairs else c.rows;
    const hx = try gpa.alloc(u16, x_rows * c.k);
    defer gpa.free(hx);
    for (hx) |*v| v.* = if (fp16) f16Bits(rng.unit()) else bf16Bits(rng.unit());
    var x = try upload(gpu, hx);
    defer x.free();

    // routed plans: each token's slots draw experts; pairs sorted by expert, items of at most 8 rows
    const plan_len = if (c.experts > 0) iters else 1;
    var plans = try gpa.alloc(hip.DeviceBuffer, plan_len);
    var members = try gpa.alloc(hip.DeviceBuffer, plan_len);
    var counts = try gpa.alloc(c_int, plan_len);
    var max_rows = try gpa.alloc(c_int, plan_len);
    defer gpa.free(plans);
    defer gpa.free(members);
    defer gpa.free(counts);
    defer gpa.free(max_rows);
    var made: usize = 0;
    defer for (plans[0..made], members[0..made]) |*p, *m| {
        p.free();
        m.free();
    };
    var expert_of = try gpa.alloc(u32, pairs * plan_len);
    defer gpa.free(expert_of);
    var distinct: f64 = 1;
    if (c.experts > 0) {
        var total_distinct: usize = 0;
        for (0..plan_len) |pl| {
            const pe = expert_of[pl * pairs ..][0..pairs];
            for (pe) |*e| e.* = @intCast(rng.next() % c.experts);
            const order = try gpa.alloc(i32, pairs);
            defer gpa.free(order);
            for (order, 0..) |*o, i| o.* = @intCast(i);
            std.mem.sort(i32, order, pe, struct {
                fn less(ctx: []u32, a: i32, b: i32) bool {
                    const ea = ctx[@intCast(a)];
                    const eb = ctx[@intCast(b)];
                    return if (ea != eb) ea < eb else a < b;
                }
            }.less);
            const items = try gpa.alloc(i32, pairs * 3);
            defer gpa.free(items);
            var n_items: usize = 0;
            var i: usize = 0;
            var mx: c_int = 1;
            while (i < pairs) {
                const e = pe[@intCast(order[i])];
                var j = i;
                while (j < pairs and j - i < 8 and pe[@intCast(order[j])] == e) j += 1;
                items[n_items * 3 ..][0..3].* = .{ @intCast(e), @intCast(i), @intCast(j - i) };
                mx = @max(mx, @as(c_int, @intCast(j - i)));
                n_items += 1;
                if (i == 0 or pe[@intCast(order[i - 1])] != e) total_distinct += 1;
                i = j;
            }
            plans[pl] = try upload(gpu, items[0 .. n_items * 3]);
            members[pl] = try upload(gpu, order);
            made += 1;
            counts[pl] = @intCast(n_items);
            max_rows[pl] = mx;
        }
        distinct = @as(f64, @floatFromInt(total_distinct)) / @as(f64, @floatFromInt(plan_len));
    }
    const out_rows = if (c.experts > 0) pairs else c.rows;
    var out_old = try hip.DeviceBuffer.alloc(gpu.d, out_rows * c.n * 4);
    defer out_old.free();
    var out_new = try hip.DeviceBuffer.alloc(gpu.d, out_rows * c.n * 4);
    defer out_new.free();
    try out_old.fill8(0, null);
    try out_new.fill8(0, null);

    var old_kernels = kernels.*;
    old_kernels.stream_on = false;
    var new_kernels = kernels.*;
    new_kernels.stream_on = true;
    var start = try hip.Event.init(gpu.d, true);
    defer start.deinit();
    var stop = try hip.Event.init(gpu.d, true);
    defer stop.deinit();

    var us: [2]f64 = undefined;
    var stats: [2]ref.Stat = .{ .{}, .{} };
    const problem: ref.Problem = .{ .fp16 = fp16, .n = c.n, .k = c.k, .bits = bits, .group = group, .x = hx, .words = hw, .scale = hs, .bias = hb };
    for ([2]*const hip.affine.Kernels{ &old_kernels, &new_kernels }, 0..) |k, v| {
        const out = if (v == 0) out_old else out_new;
        var best: f64 = std.math.inf(f64);
        for (0..3) |round| {
            try start.record(stream);
            for (0..iters) |i| {
                const copy = i % copies;
                const pl = if (c.experts > 0) i else 0;
                const arg: hip.affine.Arg = .{
                    .x = x.ptr,
                    .words = words.ptr + copy * copy_words * 4,
                    .scale = .{ .p = scale.ptr + copy * copy_tab * 2, .kind = 1 },
                    .bias = .{ .p = bias.ptr + copy * copy_tab * 2, .kind = 1 },
                    .out = out.ptr,
                    .m = if (c.experts > 0) max_rows[pl] else @intCast(c.rows),
                    .n = @intCast(c.n),
                    .k = @intCast(c.k),
                    .bits = c.bits,
                    .group = c.group,
                    .fp16 = @intFromBool(fp16),
                    .route = if (c.experts > 0) .{ .items = plans[pl].ptr, .members = members[pl].ptr, .x_div = if (c.x_div) @intCast(slots) else 1 } else .{},
                };
                try launch(.{ .k = k, .d = gpu.d, .stream = stream.handle, .arg = arg, .routed = c.experts > 0, .items = if (c.experts > 0) counts[pl] else 1 });
            }
            try stop.record(stream);
            try stop.synchronize();
            const ms = try hip.Event.elapsedMs(start, stop);
            if (round > 0) best = @min(best, @as(f64, ms) * 1000.0 / @as(f64, @floatFromInt(iters)));
        }
        us[v] = best;
        // errors: the last launch's plan and copy; rerun it once so the output is that launch's
        const last = iters - 1;
        const pl = if (c.experts > 0) last else 0;
        const copy = last % copies;
        const arg: hip.affine.Arg = .{
            .x = x.ptr,
            .words = words.ptr + copy * copy_words * 4,
            .scale = .{ .p = scale.ptr + copy * copy_tab * 2, .kind = 1 },
            .bias = .{ .p = bias.ptr + copy * copy_tab * 2, .kind = 1 },
            .out = out.ptr,
            .m = if (c.experts > 0) max_rows[pl] else @intCast(c.rows),
            .n = @intCast(c.n),
            .k = @intCast(c.k),
            .bits = c.bits,
            .group = c.group,
            .fp16 = @intFromBool(fp16),
            .route = if (c.experts > 0) .{ .items = plans[pl].ptr, .members = members[pl].ptr, .x_div = if (c.x_div) @intCast(slots) else 1 } else .{},
        };
        try out.fill8(0, null);
        try launch(.{ .k = k, .d = gpu.d, .stream = stream.handle, .arg = arg, .routed = c.experts > 0, .items = if (c.experts > 0) counts[pl] else 1 });
        try stream.synchronize();
        const pick = 16;
        for (0..@min(pick, out_rows)) |ri| {
            const row = if (out_rows <= pick) ri else ri * (out_rows - 1) / (pick - 1);
            for (0..@min(pick, c.n)) |ci| {
                const col = if (c.n <= pick) ci else ci * (c.n - 1) / (pick - 1);
                const xr = if (c.experts > 0 and c.x_div) row / slots else row;
                const e = if (c.experts > 0) expert_of[pl * pairs + row] else 0;
                // the copy's data are the same random block in every copy
                const val = ref.reference(problem, xr, e, col);
                var y: f32 = 0;
                try out.download((row * c.n + col) * 4, std.mem.asBytes(&y));
                stats[v].add(y, val);
            }
        }
    }
    const per_expert = @as(f64, @floatFromInt(c.n)) * @as(f64, @floatFromInt(c.k * bits / 8 + 4 * groups));
    const moved = per_expert * distinct;
    std.debug.print("RESULT gemv {s} m{d} n{d} k{d} b{d} g{d}: old {d:.1} us {d:.0} GB/s, new {d:.1} us {d:.0} GB/s, x{d:.2}; max|y-ref|/sum|terms| old {e:.1} new {e:.1}, rms old {e:.2} new {e:.2}\n", .{
        c.name,           c.rows,              c.n,            c.k,                 c.bits,        c.group,
        us[0],            moved / us[0] / 1e3, us[1],          moved / us[1] / 1e3, us[0] / us[1], stats[0].max_rel,
        stats[1].max_rel, stats[0].rms(),      stats[1].rms(),
    });
    try check.expect(stats[1].max_rel <= 2 * stats[0].max_rel + 1e-9 and stats[1].rms() <= 2 * stats[0].rms() + 1e-9, "gemv {s}: the stream tile is further from the float64 reference than twice the previous tile", .{c.name});
    return us[0] / us[1];
}

pub fn run(gpu: Gpu, args: []const [:0]const u8) !void {
    const reps: usize = if (args.len > 0) try std.fmt.parseInt(usize, args[0], 10) else 5;
    const filter: []const u8 = if (args.len > 1) args[1] else "";
    const family = hip.rocm.familyOf(try gpu.ctx.capability()) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(gpu.d, family);
    defer lib.close();
    const launcher = &(lib.zig orelse return error.LibraryUnavailable);
    var stream = try hip.Stream.init(gpu.d, true);
    defer stream.deinit();
    var rng: Rng = .{ .state = 0x9E3779B97F4A7C15 };
    var ran: usize = 0;
    var worst: f64 = std.math.inf(f64);
    for (cases) |c| {
        if (filter.len > 0 and std.mem.indexOf(u8, c.name, filter) == null) continue;
        worst = @min(worst, try runCase(gpu, &launcher.affine, stream, family == .rdna2, &rng, c, reps));
        ran += 1;
    }
    try check.expect(ran > 0, "gemv: no case matches '{s}'", .{filter});
    check.pass("gemv: {d} decode products, the stream tile within twice the previous tiles' error of the float64 reference (slowest x{d:.2})", .{ ran, worst });
}
