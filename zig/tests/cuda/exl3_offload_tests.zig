//! Expert offload on a real device with the real kernels, for one layer of the EXL3 pack.
//! Host bank: the memory-mapped shards (pages come in when a line is staged; the 46 GB is never copied).
//! Device pool: `cuda.exl3_offload.Pool`, a few expert lines in strips order, filled by H2D copies on a copy stream.
//! Policy: `exl3_offload.scheduleUsedAt` (hit / copy / cpu), reaped with `reapAt` once the copy event is done.
//! Compute: `Exl3.hit` over the resident lines (ids are line indices, -1 skips the slot) plus `exl3_cpu.gemvAvx2`
//! for the misses, whose rows the host writes into `y` before `combine`.
//! (a) offload block == fully resident block (the exl3-moe 512-line stacks), same router picks.
//! (b) the lifetime count promotes a repeatedly used expert across 50 synthetic tokens, with real copies.
//! (c) a captured graph replays with rewritten ids and swapped line contents and equals eager.
//! Usage: tf-cuda-test exl3-offload <pack dir> [layers, comma separated, default 0,17]

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const moe_t = @import("exl3_moe_tests.zig");
const strips = @import("exl3_strips.zig");
const router = @import("exl3_router.zig");
const cpu = @import("exl3_cpu.zig");
const offload = @import("exl3_offload.zig");
const graph = @import("exl3_graph.zig");
const shape = @import("exl3_shape.zig");

const Gpu = check.Gpu;
const Pool = cuda.exl3_offload.Pool;
const hidden = 2560;
const inter = 640;
const experts = 512;
const topk = 10;
const projs = [3][]const u8{ "gate_proj", "up_proj", "down_proj" };

fn asI16(bytes: []u8) []i16 {
    return @as([*]i16, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 2];
}

// ---------------------------------------------------------------- host bank

/// Reads one expert's nine tensors out of the mapped shards into a staged line (strips order).
const Bank = struct {
    gpa: std.mem.Allocator,
    pack: *const real.Pack,
    layer: usize,
    spec: cuda.exl3_offload.Spec,
    layout: cuda.exl3_offload.Layout,
    aligned: []i16,

    fn init(gpa: std.mem.Allocator, pack: *const real.Pack, layer: usize) !Bank {
        var mats: [3]cuda.exl3_offload.Mat = undefined;
        var biggest: usize = 0;
        for (projs, &mats) |p, *m| {
            const tr = try moe_t.tensor(pack, "model.language_model.layers.{d}.mlp.experts.0.{s}.trellis", .{ layer, p });
            if (tr.rank != 3) return error.BadShape;
            m.* = .{ .k = tr.dim(0) * 16, .n = tr.dim(1) * 16, .halves = @intCast(tr.dim(2) / 8) };
            biggest = @max(biggest, tr.bytes.len / 2);
        }
        const spec: cuda.exl3_offload.Spec = .{ .gate = mats[0], .up = mats[1], .down = mats[2] };
        return .{ .gpa = gpa, .pack = pack, .layer = layer, .spec = spec, .layout = try cuda.exl3_offload.Layout.init(spec), .aligned = try gpa.alloc(i16, biggest) };
    }

    fn deinit(b: *Bank) void {
        b.gpa.free(b.aligned);
    }

    /// Fills `dst` (one staged line) with expert `id`. This is where the mapped pages fault in.
    fn fill(b: *Bank, id: usize, dst: []u8) !void {
        const mats = [3]cuda.exl3_offload.Mat{ b.spec.gate, b.spec.up, b.spec.down };
        for (projs, mats, b.layout.regions) |p, m, r| {
            const tr = try moe_t.tensor(b.pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.trellis", .{ b.layer, id, p });
            const su = try moe_t.tensor(b.pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.suh", .{ b.layer, id, p });
            const sv = try moe_t.tensor(b.pack, "model.language_model.layers.{d}.mlp.experts.{d}.{s}.svh", .{ b.layer, id, p });
            const words = m.halves * 8;
            if (tr.rank != 3 or tr.dim(0) * 16 != m.k or tr.dim(1) * 16 != m.n or tr.dim(2) != words) return error.NonUniformExperts;
            if (su.bytes.len != r.suh_bytes or sv.bytes.len != r.svh_bytes) return error.BadShape;
            const n = tr.bytes.len / 2;
            @memcpy(std.mem.sliceAsBytes(b.aligned[0..n]), tr.bytes);
            try strips.reorder(b.aligned[0..n], tr.dim(0), tr.dim(1), words, asI16(dst[r.trellis_off..][0..r.trellis_bytes]));
            @memcpy(dst[r.suh_off..][0..r.suh_bytes], su.bytes);
            @memcpy(dst[r.svh_off..][0..r.svh_bytes], sv.bytes);
        }
    }
};

// ---------------------------------------------------------------- shared inputs

const Router = struct {
    gate16: []f16,
    sgate16: []f16,

    fn load(gpa: std.mem.Allocator, pack: *const real.Pack, layer: usize) !Router {
        const g = try moe_t.f16Copy(gpa, try moe_t.tensor(pack, "model.language_model.layers.{d}.mlp.gate.weight", .{layer}));
        errdefer gpa.free(g);
        const s = try moe_t.f16Copy(gpa, try moe_t.tensor(pack, "model.language_model.layers.{d}.mlp.shared_expert_gate.weight", .{layer}));
        return .{ .gate16 = g, .sgate16 = s };
    }

    fn free(r: Router, gpa: std.mem.Allocator) void {
        gpa.free(r.gate16);
        gpa.free(r.sgate16);
    }

    /// Host router on an fp16 hidden: the top-10 ids and their fp32 weights.
    fn route(r: Router, x16: []const f16, picks: *[topk]u16, weights: *[topk]f32) !void {
        var x32: [hidden]f32 = undefined;
        for (&x32, x16) |*d, s| d.* = @floatCast(s);
        var logits: [experts + 1]f32 = undefined;
        var p11: [topk + 1]u16 = undefined;
        var w11: [topk + 1]f32 = undefined;
        var scratch: [experts]f32 = undefined;
        try router.launchDtype(.f16, 0, std.mem.sliceAsBytes(r.gate16), &x32, std.mem.sliceAsBytes(r.sgate16), &x32, topk, &logits, &p11, &w11, &scratch);
        picks.* = p11[0..topk].*;
        weights.* = w11[0..topk].*;
    }
};

const Dev = struct {
    x: cuda.DeviceBuffer,
    ids: cuda.DeviceBuffer,
    y_gate: cuda.DeviceBuffer,
    y_up: cuda.DeviceBuffer,
    y_glu: cuda.DeviceBuffer,
    y_down: cuda.DeviceBuffer,
    wts: cuda.DeviceBuffer,
    out: cuda.DeviceBuffer,

    fn init(d: *const cuda.Driver, tokens: usize) !Dev {
        var v: Dev = undefined;
        v.x = try cuda.DeviceBuffer.alloc(d, tokens * hidden * 2);
        errdefer v.x.free();
        v.ids = try cuda.DeviceBuffer.alloc(d, tokens * topk * 4);
        errdefer v.ids.free();
        v.y_gate = try cuda.DeviceBuffer.alloc(d, tokens * topk * inter * 2);
        errdefer v.y_gate.free();
        v.y_up = try cuda.DeviceBuffer.alloc(d, tokens * topk * inter * 2);
        errdefer v.y_up.free();
        v.y_glu = try cuda.DeviceBuffer.alloc(d, tokens * topk * inter * 2);
        errdefer v.y_glu.free();
        v.y_down = try cuda.DeviceBuffer.alloc(d, tokens * topk * hidden * 2);
        errdefer v.y_down.free();
        v.wts = try cuda.DeviceBuffer.alloc(d, tokens * topk * 4);
        errdefer v.wts.free();
        v.out = try cuda.DeviceBuffer.alloc(d, tokens * hidden * 2);
        return v;
    }

    fn free(v: *Dev) void {
        v.x.free();
        v.ids.free();
        v.y_gate.free();
        v.y_up.free();
        v.y_glu.free();
        v.y_down.free();
        v.wts.free();
        v.out.free();
    }
};

fn downloadF16(buf: cuda.DeviceBuffer, out: []f16) !void {
    try buf.download(0, std.mem.sliceAsBytes(out));
}

// ---------------------------------------------------------------- fully resident reference

const Resident = struct {
    gpu: Gpu,
    ex: *const cuda.exl3.Exl3,
    stream: cuda.Stream,
    gs: moe_t.Stack,
    us: moe_t.Stack,
    ds: moe_t.Stack,
    dev: Dev,

    fn init(gpu: Gpu, pack: *const real.Pack, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize, max_tokens: usize) !Resident {
        var gs = try moe_t.loadStack(gpu, pack, layer, "gate_proj");
        errdefer gs.free();
        var us = try moe_t.loadStack(gpu, pack, layer, "up_proj");
        errdefer us.free();
        var ds = try moe_t.loadStack(gpu, pack, layer, "down_proj");
        errdefer ds.free();
        return .{ .gpu = gpu, .ex = ex, .stream = stream, .gs = gs, .us = us, .ds = ds, .dev = try Dev.init(gpu.d, max_tokens) };
    }

    fn deinit(r: *Resident) void {
        r.dev.free();
        r.gs.free();
        r.us.free();
        r.ds.free();
    }

    /// All 512 lines on the device, ids are the expert ids.
    fn run(r: *Resident, tokens: usize, xs: []const f16, picks: []const u16, weights: []const f32, out: []f16, rows: []f16) !void {
        const ids = try r.gpu.gpa.alloc(i32, tokens * topk);
        defer r.gpu.gpa.free(ids);
        for (ids, picks) |*d, s| d.* = s;
        try r.dev.x.upload(0, std.mem.sliceAsBytes(xs));
        try r.dev.ids.upload(0, std.mem.sliceAsBytes(ids));
        try r.dev.wts.upload(0, std.mem.sliceAsBytes(weights));
        const dv = r.dev;
        try r.ex.hit(r.stream, dv.x.ptr, dv.ids.ptr, try r.gs.packed_(), try r.us.packed_(), try r.ds.packed_(), dv.y_gate.ptr, dv.y_up.ptr, dv.y_glu.ptr, dv.y_down.ptr, tokens, experts, topk);
        try r.ex.combine(r.stream, dv.y_down.ptr, dv.wts.ptr, dv.out.ptr, tokens, hidden, topk);
        try r.stream.synchronize();
        try downloadF16(dv.out, out[0 .. tokens * hidden]);
        try downloadF16(dv.y_down, rows[0 .. tokens * topk * hidden]);
    }
};

// ---------------------------------------------------------------- the offload layer

const Stats = struct {
    hit: u64 = 0,
    copy: u64 = 0,
    cpu: u64 = 0,
    cpu_experts: u64 = 0,
    evicted: u64 = 0,
};

const Offload = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    ex: *const cuda.exl3.Exl3,
    stream: cuda.Stream,
    pool: Pool,
    bank: *Bank,
    pack: *const real.Pack,
    layer: u8,
    slots: []u16,
    line: []offload.Line,
    stamp: []u32,
    vlayer: []u8,
    vram: offload.Vram,
    counts: *[shape.layers][shape.experts]u32,
    policy: offload.Policy,
    now: u32 = 0,
    pending: std.ArrayList(offload.Copy) = .empty,
    done: cuda.Event,
    have_done: bool = false,
    dev: ?Dev,
    cpu_scratch: []f64,
    stats: Stats = .{},
    fill_ns: i96 = 0,
    copy_ns: i96 = 0,

    fn init(gpu: Gpu, pack: *const real.Pack, bank: *Bank, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize, lines: usize, copy_lines: usize, max_tokens: usize) !Offload {
        const gpa = gpu.gpa;
        var pool = try Pool.init(gpa, gpu.d, bank.spec, lines);
        errdefer pool.deinit();
        const slots = try gpa.alloc(u16, lines);
        errdefer gpa.free(slots);
        const line = try gpa.alloc(offload.Line, lines);
        errdefer gpa.free(line);
        const stamp = try gpa.alloc(u32, lines);
        errdefer gpa.free(stamp);
        const vlayer = try gpa.alloc(u8, lines);
        errdefer gpa.free(vlayer);
        const counts = try gpa.create([shape.layers][shape.experts]u32);
        errdefer gpa.destroy(counts);
        counts.* = @splat(@splat(0));
        var done = try cuda.Event.init(gpu.d, false);
        errdefer done.deinit();
        const dev: ?Dev = if (max_tokens > 0) try Dev.init(gpu.d, max_tokens) else null;
        const k = pool.layout.line_bytes;
        const mx = @max(bank.spec.gate.k * bank.spec.gate.n, bank.spec.down.k * bank.spec.down.n);
        const scratch = try gpa.alloc(f64, 3 * mx);
        return .{
            .gpa = gpa, .io = gpu.io, .d = gpu.d, .ex = ex, .stream = stream, .pool = pool, .bank = bank, .pack = pack, .layer = @intCast(layer),
            .slots = slots, .line = line, .stamp = stamp, .vlayer = vlayer,
            .vram = offload.Vram.init(slots, line, stamp, vlayer),
            .counts = counts,
            .policy = .{ .slots = @intCast(lines), .expert_bytes = k, .copy_bytes = k * copy_lines },
            .done = done, .dev = dev, .cpu_scratch = scratch,
        };
    }

    fn deinit(o: *Offload) void {
        o.gpa.free(o.cpu_scratch);
        if (o.dev) |*v| v.free();
        o.done.deinit();
        o.pending.deinit(o.gpa);
        o.gpa.destroy(o.counts);
        o.gpa.free(o.vlayer);
        o.gpa.free(o.stamp);
        o.gpa.free(o.line);
        o.gpa.free(o.slots);
        o.pool.deinit();
    }

    /// Copies that were enqueued last round have landed: promote them from in flight to resident.
    fn reapLanded(o: *Offload) !void {
        for (o.pending.items) |c| {
            try o.pool.wait(c.line);
            try offload.reapAt(&o.vram, o.layer, c.id, o.now);
        }
        o.pending.clearRetainingCapacity();
    }

    /// Policy step for one round: observe, schedule, and enqueue the staged copies.
    fn plan(o: *Offload, picks: []const u16, decisions: []offload.Decision) !void {
        o.now += 1;
        try o.reapLanded();
        try offload.observe(&o.counts[o.layer], picks);
        var scratch: [topk * 16]u16 = undefined;
        // line contents before scheduling, to count evictions
        var before: [64]u16 = undefined;
        @memcpy(before[0..o.slots.len], o.slots);
        try offload.scheduleUsedAt(&o.vram, o.layer, picks, o.counts, o.policy, o.now, decisions, &scratch, offload.allOwned());
        var staged: [topk * 16]offload.Copy = undefined;
        const n = try offload.stagedCopies(o.vram, o.layer, picks, decisions, &staged);
        for (staged[0..n]) |c| {
            if (before[c.line] != offload.Cache.empty_id) o.stats.evicted += 1;
            const t0 = check.now(o.io);
            try o.bank.fill(c.id, o.pool.stagingLine(c.line));
            const t1 = check.now(o.io);
            try o.pool.enqueue(c.line, if (o.have_done) o.done else null);
            o.fill_ns += t1 - t0;
            try o.pending.append(o.gpa, c);
        }
        for (decisions) |dc| switch (dc) {
            .hit => o.stats.hit += 1,
            .copy => o.stats.copy += 1,
            .cpu => o.stats.cpu += 1,
        };
    }

    /// One routed expert on the CPU: gate, up, swiglu, down in f64 over the mapped bank, rounded to fp16.
    fn cpuExpert(o: *Offload, id: usize, x16: []const f16, row: []f16) !void {
        var x64: [hidden]f64 = undefined;
        for (&x64, x16) |*d, s| d.* = @floatCast(s);
        var g: [inter]f64 = undefined;
        var u: [inter]f64 = undefined;
        var glu: [inter]f64 = undefined;
        var y: [hidden]f64 = undefined;
        var mats: [3]real.Mat = undefined;
        var n_mats: usize = 0;
        defer for (mats[0..n_mats]) |m| moe_t.freeMat(o.gpa, m);
        for (projs) |p| {
            var nb: [160]u8 = undefined;
            mats[n_mats] = try real.mat(o.pack, o.gpa, try std.fmt.bufPrint(&nb, "model.language_model.layers.{d}.mlp.experts.{d}.{s}", .{ o.layer, id, p }));
            n_mats += 1;
        }
        const w = [3]real.Mat{ mats[0], mats[1], mats[2] };
        try cpu.gemvAvx2(.{ .halves = w[0].halves }, .mul1, w[0].trellis, w[0].tiles_k, w[0].tiles_n, w[0].suh, w[0].svh, &x64, &g, o.cpu_scratch);
        try cpu.gemvAvx2(.{ .halves = w[1].halves }, .mul1, w[1].trellis, w[1].tiles_k, w[1].tiles_n, w[1].suh, w[1].svh, &x64, &u, o.cpu_scratch);
        try cpu.swiglu(&g, &u, &glu, std.math.inf(f64));
        try cpu.gemvAvx2(.{ .halves = w[2].halves }, .mul1, w[2].trellis, w[2].tiles_k, w[2].tiles_n, w[2].suh, w[2].svh, &glu, &y, o.cpu_scratch);
        for (row, y) |*d, s| d.* = @floatCast(s);
        o.stats.cpu_experts += 1;
    }

    /// One decode round of `tokens` tokens (xs [tokens][hidden], picks and weights [tokens][topk]).
    fn run(o: *Offload, tokens: usize, xs: []const f16, picks: []const u16, weights: []const f32, decisions: []offload.Decision, out: []f16, rows: []f16) !void {
        try o.plan(picks, decisions);
        const dv = o.dev.?;
        var ids: [8 * topk]i32 = undefined;
        try offload.lineIds(o.vram, o.layer, picks, decisions, ids[0 .. tokens * topk]);

        // misses: the CPU row goes into y_down before the hit kernel, which skips the -1 slots
        for (0..tokens) |t| {
            for (0..topk) |k| {
                const s = t * topk + k;
                if (ids[s] >= 0) continue;
                const x = xs[t * hidden ..][0..hidden];
                try o.cpuExpert(picks[s], x, rows[s * hidden ..][0..hidden]);
                try dv.y_down.upload(s * hidden * 2, std.mem.sliceAsBytes(rows[s * hidden ..][0..hidden]));
            }
        }
        try dv.x.upload(0, std.mem.sliceAsBytes(xs));
        try dv.ids.upload(0, std.mem.sliceAsBytes(ids[0 .. tokens * topk]));
        try dv.wts.upload(0, std.mem.sliceAsBytes(weights));
        try o.ex.hit(o.stream, dv.x.ptr, dv.ids.ptr, try o.pool.packed_(0), try o.pool.packed_(1), try o.pool.packed_(2), dv.y_gate.ptr, dv.y_up.ptr, dv.y_glu.ptr, dv.y_down.ptr, tokens, o.pool.lines, topk);
        try o.ex.combine(o.stream, dv.y_down.ptr, dv.wts.ptr, dv.out.ptr, tokens, hidden, topk);
        try o.done.record(o.stream);
        o.have_done = true;
        try o.stream.synchronize();
        try downloadF16(dv.out, out[0 .. tokens * hidden]);
        try downloadF16(dv.y_down, rows[0 .. tokens * topk * hidden]);
    }
};

// ---------------------------------------------------------------- (a)

fn randomHidden(r: std.Random, x: []f16) void {
    for (x) |*v| v.* = @floatCast((r.float(f32) - 0.5) * 2.0);
}

const Scenario = struct { name: []const u8, lines: usize, copy_lines: usize, rounds: []const []const u8 };

fn runScenario(gpu: Gpu, pack: *const real.Pack, bank: *Bank, rt: Router, res: *Resident, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize, sc: Scenario) !void {
    const gpa = gpu.gpa;
    var rng = std.Random.DefaultPrng.init(0x0ff10ad + layer);
    var toks: [3][hidden]f16 = undefined;
    var tp: [3][topk]u16 = undefined;
    var tw: [3][topk]f32 = undefined;
    for (0..3) |i| {
        randomHidden(rng.random(), &toks[i]);
        try rt.route(&toks[i], &tp[i], &tw[i]);
    }
    std.debug.print("  [{s}] {d} lines ({d:.2} MiB), copy budget {d} lines/round; token A picks {any}\n", .{ sc.name, sc.lines, @as(f64, @floatFromInt(sc.lines * bank.layout.line_bytes)) / 1048576.0, sc.copy_lines, tp[0] });

    var o = try Offload.init(gpu, pack, bank, ex, stream, layer, sc.lines, sc.copy_lines, 3);
    defer o.deinit();

    var xs: [3 * hidden]f16 = undefined;
    var picks: [3 * topk]u16 = undefined;
    var wts: [3 * topk]f32 = undefined;
    var dec: [3 * topk]offload.Decision = undefined;
    const out_o = try gpa.alloc(f16, 3 * hidden);
    defer gpa.free(out_o);
    const out_r = try gpa.alloc(f16, 3 * hidden);
    defer gpa.free(out_r);
    const rows_o = try gpa.alloc(f16, 3 * topk * hidden);
    defer gpa.free(rows_o);
    const rows_r = try gpa.alloc(f16, 3 * topk * hidden);
    defer gpa.free(rows_r);

    var hit_rows_bad: usize = 0;
    var hit_rows: usize = 0;
    var worst_out: f64 = 0;
    var worst_row: f64 = 0;
    var all_hit_rounds: usize = 0;
    for (sc.rounds, 0..) |spec, ri| {
        const tokens = spec.len;
        for (spec, 0..) |ch, t| {
            const ti: usize = ch - 'A';
            @memcpy(xs[t * hidden ..][0..hidden], &toks[ti]);
            @memcpy(picks[t * topk ..][0..topk], &tp[ti]);
            @memcpy(wts[t * topk ..][0..topk], &tw[ti]);
        }
        const decs = dec[0 .. tokens * topk];
        try o.run(tokens, xs[0 .. tokens * hidden], picks[0 .. tokens * topk], wts[0 .. tokens * topk], decs, out_o, rows_o);
        try res.run(tokens, xs[0 .. tokens * hidden], picks[0 .. tokens * topk], wts[0 .. tokens * topk], out_r, rows_r);

        var nh: usize = 0;
        var nc: usize = 0;
        var nu: usize = 0;
        var row_worst: f64 = 0;
        for (decs, 0..) |dc, s| {
            const a = rows_o[s * hidden ..][0..hidden];
            const b = rows_r[s * hidden ..][0..hidden];
            switch (dc) {
                .hit => {
                    nh += 1;
                    hit_rows += 1;
                    if (!std.mem.eql(u8, std.mem.sliceAsBytes(a), std.mem.sliceAsBytes(b))) hit_rows_bad += 1;
                },
                .copy => nc += 1,
                .cpu => nu += 1,
            }
            if (dc != .hit) {
                var mx: f64 = 0;
                var pk: f64 = 0;
                for (a, b) |x, y| {
                    mx = @max(mx, @abs(@as(f64, @floatCast(x)) - @as(f64, @floatCast(y))));
                    pk = @max(pk, @abs(@as(f64, @floatCast(y))));
                }
                row_worst = @max(row_worst, mx / pk);
            }
        }
        worst_row = @max(worst_row, row_worst);
        var out_rel: f64 = 0;
        for (0..tokens) |t| {
            var mx: f64 = 0;
            var pk: f64 = 0;
            for (out_o[t * hidden ..][0..hidden], out_r[t * hidden ..][0..hidden]) |x, y| {
                mx = @max(mx, @abs(@as(f64, @floatCast(x)) - @as(f64, @floatCast(y))));
                pk = @max(pk, @abs(@as(f64, @floatCast(y))));
            }
            out_rel = @max(out_rel, mx / pk);
        }
        worst_out = @max(worst_out, out_rel);
        const same = std.mem.eql(u8, std.mem.sliceAsBytes(out_o[0 .. tokens * hidden]), std.mem.sliceAsBytes(out_r[0 .. tokens * hidden]));
        if (nh == tokens * topk) all_hit_rounds += 1;
        std.debug.print("    round {d} tokens {s}: hit {d} copy {d} cpu {d} | out vs resident {d:.5} of peak{s}; miss rows worst {d:.5} of row peak\n", .{ ri + 1, spec, nh, nc, nu, out_rel, if (same) " (bit-identical)" else "", row_worst });
        try check.expect(out_rel <= 0.005, "{s} round {d}: offload output differs from the resident output by {d}", .{ sc.name, ri + 1, out_rel });
        try check.expect(row_worst <= 0.01, "{s} round {d}: a CPU-miss row differs from the resident row", .{ sc.name, ri + 1 });
        if (nh == tokens * topk) try check.expect(same, "{s} round {d}: all-hit round is not bit-identical to resident", .{ sc.name, ri + 1 });
    }
    try o.reapLanded();
    std.debug.print("  [{s}] totals: hit {d} copy {d} cpu {d} slots; {d} H2D line copies ({d:.1} MiB, {d} replaced a resident line); {d} CPU expert evaluations; hit rows bit-identical to resident: {d}/{d}; worst round output {d:.5} of peak, worst miss row {d:.5}\n", .{
        sc.name, o.stats.hit, o.stats.copy, o.stats.cpu, o.pool.copies, @as(f64, @floatFromInt(o.pool.copied_bytes)) / 1048576.0, o.stats.evicted, o.stats.cpu_experts, hit_rows - hit_rows_bad, hit_rows, worst_out, worst_row,
    });
    try check.expect(hit_rows > 0 and o.stats.cpu > 0 and o.pool.copies > 0, "{s}: a path was not exercised (hit {d}, cpu {d}, copies {d})", .{ sc.name, hit_rows, o.stats.cpu, o.pool.copies });
    try check.expect(hit_rows_bad == 0, "{s}: {d} hit rows differ from the resident rows", .{ sc.name, hit_rows_bad });
    if (all_hit_rounds > 0) std.debug.print("  [{s}] {d} round(s) were all hits: output bit-identical to resident\n", .{ sc.name, all_hit_rounds });
}

// ---------------------------------------------------------------- (b)

fn verifyLine(o: *Offload, line: usize, id: usize, scratch: []u8, got: []u8) !void {
    try o.bank.fill(id, scratch);
    const lay = o.pool.layout;
    for (0..3) |p| {
        const r = lay.regions[p];
        const b = o.pool.bufs[p];
        try b.trellis.download(line * r.trellis_bytes, got[r.trellis_off..][0..r.trellis_bytes]);
        try b.suh.download(line * r.suh_bytes, got[r.suh_off..][0..r.suh_bytes]);
        try b.svh.download(line * r.svh_bytes, got[r.svh_off..][0..r.svh_bytes]);
    }
    try check.sameBytes("pool line vs bank", got, scratch);
}

fn promotion(gpu: Gpu, pack: *const real.Pack, bank: *Bank, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize) !void {
    const gpa = gpu.gpa;
    const lines = 8;
    var o = try Offload.init(gpu, pack, bank, ex, stream, layer, lines, 2, 0);
    defer o.deinit();
    const hot: u16 = 211;
    const warm_a: u16 = 333; // every 2nd token
    const warm_b: u16 = 17; // every 3rd token
    var rng = std.Random.DefaultPrng.init(0xb0b + layer);
    const r = rng.random();
    std.debug.print("  [promotion] {d} lines, 2 copies/round; expert {d} on every token, {d} every 2nd, {d} every 3rd, the rest random\n", .{ lines, hot, warm_a, warm_b });

    var first_resident: ?usize = null;
    var hot_evicted_after: usize = 0;
    var win_hit: usize = 0;
    var tot_hit: usize = 0;
    var last20_hit: usize = 0;
    var warm_a_resident_at: ?usize = null;
    const tokens = 50;
    for (0..tokens) |t| {
        var picks: [topk]u16 = undefined;
        var n: usize = 0;
        picks[n] = hot;
        n += 1;
        if (t % 2 == 0) {
            picks[n] = warm_a;
            n += 1;
        }
        if (t % 3 == 0) {
            picks[n] = warm_b;
            n += 1;
        }
        while (n < topk) {
            const c = r.intRangeLessThan(u16, 0, experts);
            if (std.mem.indexOfScalar(u16, picks[0..n], c) != null) continue;
            picks[n] = c;
            n += 1;
        }
        var dec: [topk]offload.Decision = undefined;
        try o.plan(&picks, &dec);
        var h: usize = 0;
        for (dec) |d| {
            if (d == .hit) h += 1;
        }
        win_hit += h;
        tot_hit += h;
        if (t >= tokens - 20) last20_hit += h;
        if (dec[0] == .hit and first_resident == null) first_resident = t;
        if (first_resident != null and t > first_resident.? and dec[0] != .hit) hot_evicted_after += 1;
        if (warm_a_resident_at == null and offload.residentAt(o.vram, o.layer, warm_a)) warm_a_resident_at = t;
        if ((t + 1) % 10 == 0) {
            std.debug.print("    tokens {d:>2}-{d:>2}: hit ratio this window {d:.3}, cumulative {d:.3}; resident now:", .{ t - 8, t + 1, @as(f64, @floatFromInt(win_hit)) / 100.0, @as(f64, @floatFromInt(tot_hit)) / @as(f64, @floatFromInt((t + 1) * topk)) });
            for (o.slots, o.line) |id, ln| {
                if (ln == .resident) std.debug.print(" {d}", .{id}) else if (ln == .inflight) std.debug.print(" {d}*", .{id});
            }
            std.debug.print("\n", .{});
            win_hit = 0;
        }
    }
    try o.reapLanded();
    const counts_hot = o.counts[o.layer][hot];
    std.debug.print("  [promotion] hot expert {d}: first hit at token {?d}, not a hit on {d} later token(s); lifetime count {d}; last-20-token hit ratio {d:.3} (the hot expert alone caps a token at 0.100 + warm)\n", .{ hot, first_resident, hot_evicted_after, counts_hot, @as(f64, @floatFromInt(last20_hit)) / 200.0 });
    try check.expect(first_resident != null and first_resident.? <= 4, "hot expert first hit at {?d}", .{first_resident});
    try check.expect(hot_evicted_after == 0, "hot expert missed {d} times after promotion", .{hot_evicted_after});
    try check.expect(offload.residentAt(o.vram, o.layer, hot), "hot expert is not resident at the end", .{});
    try check.expect(warm_a_resident_at != null, "the every-2nd-token expert never became resident", .{});
    // the early window has no hits; the late window has at least the hot expert on every token
    try check.expect(last20_hit >= 20, "last 20 tokens hit {d} slots, expected at least 20 (the hot expert each token)", .{last20_hit});

    // every resident line holds exactly the bank's bytes for its expert (real copies landed correctly)
    const scratch = try gpa.alloc(u8, o.pool.layout.line_bytes);
    defer gpa.free(scratch);
    const got = try gpa.alloc(u8, o.pool.layout.line_bytes);
    defer gpa.free(got);
    var verified: usize = 0;
    for (o.slots, o.line, 0..) |id, ln, i| {
        if (ln != .resident) continue;
        try verifyLine(&o, i, id, scratch, got);
        verified += 1;
    }
    const lm = @as(f64, @floatFromInt(o.pool.copied_bytes)) / 1048576.0;
    std.debug.print("  [promotion] {d} H2D line copies, {d:.1} MiB; mmap read + strips reorder {d:.2} ms/line; {d} resident lines re-read from the device and compared byte for byte with the bank\n", .{ o.pool.copies, lm, @as(f64, @floatFromInt(o.fill_ns)) / 1e6 / @as(f64, @floatFromInt(o.pool.copies)), verified });
    try check.expect(verified == lines, "only {d} of {d} lines resident", .{ verified, lines });
    check.pass("layer {d}: use-count promotes the repeatedly used expert (first hit token {?d}, never evicted after) over 50 tokens with real H2D copies", .{ layer, first_resident });
}

// ---------------------------------------------------------------- (c)

fn fillPool(o: *Offload, ids: []const u16) !void {
    for (ids, 0..) |id, line| {
        try o.bank.fill(id, o.pool.stagingLine(line));
        try o.pool.enqueue(line, null);
    }
    for (0..ids.len) |line| try o.pool.wait(line);
}

fn stampPoison(b: cuda.DeviceBuffer, stream: cuda.Stream) !void {
    try b.fill8(0xff, stream.handle);
}

fn graphReplay(gpu: Gpu, pack: *const real.Pack, bank: *Bank, rt: Router, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize) !void {
    const gpa = gpu.gpa;
    var o = try Offload.init(gpu, pack, bank, ex, stream, layer, topk, topk, 1);
    defer o.deinit();
    const dv = o.dev.?;
    var rng = std.Random.DefaultPrng.init(0xca97 + layer);
    var xa: [hidden]f16 = undefined;
    var xb: [hidden]f16 = undefined;
    randomHidden(rng.random(), &xa);
    randomHidden(rng.random(), &xb);
    var pa: [topk]u16 = undefined;
    var pb: [topk]u16 = undefined;
    var wa: [topk]f32 = undefined;
    var wb: [topk]f32 = undefined;
    try rt.route(&xa, &pa, &wa);
    try rt.route(&xb, &pb, &wb);

    const pk = [3]exl3_packed{ try o.pool.packed_(0), try o.pool.packed_(1), try o.pool.packed_(2) };
    const rows_bytes = topk * hidden * 2;

    const Capture = struct {
        fn body(oo: *Offload) !void {
            const d = oo.dev.?;
            try oo.ex.hit(oo.stream, d.x.ptr, d.ids.ptr, try oo.pool.packed_(0), try oo.pool.packed_(1), try oo.pool.packed_(2), d.y_gate.ptr, d.y_up.ptr, d.y_glu.ptr, d.y_down.ptr, 1, oo.pool.lines, topk);
            try oo.ex.combine(oo.stream, d.y_down.ptr, d.wts.ptr, d.out.ptr, 1, hidden, topk);
        }
    };

    // state 1: line i holds pick i of token A, slot k reads line k
    // state 2: every line swapped to token B's experts, slots read a permutation of lines
    // state 3: same lines as state 2, slots read another permutation (ids only)
    const States = struct { x: []const f16, lines: []const u16, ids: [topk]i32, w: [topk]f32 };
    var st1: States = .{ .x = &xa, .lines = &pa, .ids = undefined, .w = wa };
    var st2: States = .{ .x = &xb, .lines = &pb, .ids = undefined, .w = undefined };
    var st3: States = .{ .x = &xa, .lines = &pb, .ids = undefined, .w = undefined };
    for (0..topk) |k| {
        st1.ids[k] = @intCast(k);
        st2.ids[k] = @intCast((k * 3 + 1) % topk);
        st3.ids[k] = @intCast((k * 7 + 4) % topk);
        st2.w[k] = wb[@intCast(st2.ids[k])];
        st3.w[k] = wb[@intCast(st3.ids[k])] * 0.5;
    }

    const eager = try gpa.alloc(u8, 3 * (hidden * 2 + rows_bytes));
    defer gpa.free(eager);
    const outs = [3][]u8{ eager[0 .. hidden * 2], eager[(hidden * 2 + rows_bytes) ..][0 .. hidden * 2], eager[2 * (hidden * 2 + rows_bytes) ..][0 .. hidden * 2] };
    const rowsb = [3][]u8{ eager[hidden * 2 ..][0..rows_bytes], eager[(hidden * 2 + rows_bytes) + hidden * 2 ..][0..rows_bytes], eager[2 * (hidden * 2 + rows_bytes) + hidden * 2 ..][0..rows_bytes] };

    const all = [3]*const States{ &st1, &st2, &st3 };
    const swap_before = [3]bool{ true, true, false };
    var exec: ?cuda.graph.Exec = null;
    defer if (exec) |*e| e.deinit();
    var captured: ?cuda.graph.Graph = null;
    defer if (captured) |*c| c.deinit();

    var keys: [1]graph.Key = undefined;
    var n_keys: usize = 0;
    const key: graph.Key = .{ .rank = 0, .layer = @intCast(layer), .window = 1 };
    var replayed: usize = 0;
    for (all, 0..) |s, si| {
        if (swap_before[si]) {
            // swap what the lines hold: H2D copies of the new experts, the graph's pointers do not move
            try fillPool(&o, s.lines);
        }
        try dv.x.upload(0, std.mem.sliceAsBytes(s.x));
        try dv.ids.upload(0, std.mem.sliceAsBytes(&s.ids));
        try dv.wts.upload(0, std.mem.sliceAsBytes(&s.w));
        // eager
        try Capture.body(&o);
        try stream.synchronize();
        try dv.out.download(0, outs[si]);
        try dv.y_down.download(0, rowsb[si]);

        if (exec == null) {
            // capture once, with the output buffers poisoned so a replay that does not run shows up
            try stampPoison(dv.y_down, stream);
            try stampPoison(dv.out, stream);
            try stream.synchronize();
            try cuda.graph.beginCapture(stream, .thread_local);
            try Capture.body(&o);
            captured = try cuda.graph.endCapture(stream);
            const still = try check.download(gpu, dv.out);
            defer gpa.free(still);
            try check.expect(std.mem.allEqual(u8, still, 0xff), "capture ran the block", .{});
            exec = try captured.?.instantiate();
            try exec.?.upload(stream);
            try check.expect(graph.promote(keys[0..n_keys], key) == .recapture, "first window should capture", .{});
            keys[0] = key;
            n_keys = 1;
            std.debug.print("  [graph] captured {d} nodes (gate, up, swiglu, down, combine) once\n", .{try captured.?.nodeCount()});
        } else {
            try check.expect(graph.promote(keys[0..n_keys], key) == .rewrite_ids, "a stored window must rewrite ids, not recapture", .{});
        }
        try stampPoison(dv.y_down, stream);
        try stampPoison(dv.out, stream);
        try exec.?.launchOn(stream);
        try stream.synchronize();
        const g_out = try check.download(gpu, dv.out);
        defer gpa.free(g_out);
        const g_rows = try check.download(gpu, dv.y_down);
        defer gpa.free(g_rows);
        try check.sameBytes("graph replay out", g_out, outs[si]);
        try check.sameBytes("graph replay slot rows", g_rows, rowsb[si]);
        replayed += 1;
        std.debug.print("  [graph] state {d}: replay with ids {any} == eager, out and all {d} slot rows byte for byte\n", .{ si + 1, s.ids, topk });
    }
    try check.expect(!std.mem.eql(u8, outs[0], outs[1]) and !std.mem.eql(u8, outs[1], outs[2]), "states produced identical outputs; the comparison proves nothing", .{});

    // exl3_graph's rules for these bindings
    const bind_a = [_]graph.Binding{
        .{ .name = "expert_ids", .address = dv.ids.ptr, .replay = true },
        .{ .name = "gate_lines", .address = pk[0].trellis, .replay = false },
        .{ .name = "up_lines", .address = pk[1].trellis, .replay = false },
        .{ .name = "down_lines", .address = pk[2].trellis, .replay = false },
    };
    var bind_b = bind_a;
    bind_b[0].address = dv.ids.ptr + 4096;
    const cap_a: graph.Capture = .{ .window = 1, .bindings = &bind_a };
    const cap_b: graph.Capture = .{ .window = 1, .bindings = &bind_b };
    try check.expect(graph.isReplayName("expert_ids") and graph.valid(cap_a), "expert_ids must be a replay binding", .{});
    try check.expect(graph.sameTopology(cap_a, cap_b) and !graph.argumentsEqual(cap_a, cap_b), "a moved ids buffer keeps the topology", .{});
    var bind_c = bind_a;
    bind_c[1].address += 256;
    try check.expect(!graph.sameTopology(cap_a, .{ .window = 1, .bindings = &bind_c }), "moving the fixed line base must be a new topology", .{});

    // captureReady over the live policy state: all hits and nothing in flight is ready
    var lines_ready: [topk]offload.Line = @splat(.resident);
    var decs_hit: [topk]offload.Decision = @splat(.hit);
    try check.expect(graph.captureReady(false, &decs_hit, &lines_ready), "all hits with every line resident should be capture-ready", .{});
    decs_hit[3] = .copy;
    try check.expect(!graph.captureReady(false, &decs_hit, &lines_ready), "a copy decision must block capture", .{});
    decs_hit[3] = .cpu;
    lines_ready[2] = .inflight;
    try check.expect(!graph.captureReady(false, &decs_hit, &lines_ready), "an in-flight line must block capture", .{});
    check.pass("layer {d}: one captured graph replayed {d} states (ids rewritten, lines swapped by H2D) byte-identical to eager; exl3_graph replay/fixed/ready rules hold", .{ layer, replayed });
}

const exl3_packed = cuda.exl3.Packed;

// ---------------------------------------------------------------- driver

fn runLayer(gpu: Gpu, pack: *const real.Pack, ex: *const cuda.exl3.Exl3, stream: cuda.Stream, layer: usize) !void {
    const gpa = gpu.gpa;
    std.debug.print("== layer {d}\n", .{layer});
    var bank = try Bank.init(gpa, pack, layer);
    defer bank.deinit();
    std.debug.print("  host bank: mmap of the 7 shards; one line = {d} bytes ({d:.3} MiB), halves gate={d} up={d} down={d}\n", .{ bank.layout.line_bytes, @as(f64, @floatFromInt(bank.layout.line_bytes)) / 1048576.0, bank.spec.gate.halves, bank.spec.up.halves, bank.spec.down.halves });
    try check.expect(bank.spec.gate.k == hidden and bank.spec.gate.n == inter and bank.spec.down.k == inter and bank.spec.down.n == hidden, "expert shape", .{});
    const rt = try Router.load(gpa, pack, layer);
    defer rt.free(gpa);

    var res = try Resident.init(gpu, pack, ex, stream, layer, 3);
    defer res.deinit();

    // (a) 4 lines force misses; rounds mix tokens A, B, C (distinct hiddens, so distinct picks)
    try runScenario(gpu, pack, &bank, rt, &res, ex, stream, layer, .{
        .name = "4 lines",
        .lines = 4,
        .copy_lines = 2,
        .rounds = &.{ "A", "A", "A", "ABC", "BC", "BC", "BC", "AB" },
    });
    // a pool large enough for token A's 10 experts reaches an all-hit round: the output is then bit-identical
    try runScenario(gpu, pack, &bank, rt, &res, ex, stream, layer, .{
        .name = "12 lines",
        .lines = 12,
        .copy_lines = 4,
        .rounds = &.{ "A", "A", "A", "A", "AB" },
    });
    check.pass("layer {d}: offload block (H2D line pool + Exl3.hit + CPU miss rows + combine) matches the fully resident block", .{layer});

    try promotion(gpu, pack, &bank, ex, stream, layer);
    try graphReplay(gpu, pack, &bank, rt, ex, stream, layer);
}

pub fn run(gpu: Gpu, rest: []const [:0]const u8) !void {
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    var layer_list: [8]usize = undefined;
    var n_layers: usize = 0;
    const spec: []const u8 = if (rest.len > 1) rest[1] else "0,17";
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |tok| {
        if (n_layers == layer_list.len) return error.TooManyLayers;
        layer_list[n_layers] = try std.fmt.parseInt(usize, tok, 10);
        n_layers += 1;
    }

    var pack: real.Pack = .{ .files = undefined };
    defer pack.close(gpu.io);
    try pack.openAll(gpu, dir);
    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    for (layer_list[0..n_layers]) |l| try runLayer(gpu, &pack, &ex, stream, l);
}
