//! Every registered affine kernel against float64: for each product, each entry of its (op, path) that the GPU has and the
//! shape fits is launched on its own, its output compared with dequant(W) . x on sampled outputs, and the entries the
//! registry can pick between under the run's policy must write the same bytes. `--bench` adds each entry's time.

const std = @import("std");
const hip = @import("hip");
const check = @import("../check.zig");
const data = @import("data.zig");
const cases = @import("cases.zig");
const prod = @import("product.zig");
const Product = prod.Product;
const Rig = prod.Rig;
const Entry = hip.affine.Entry;
const Case = cases.Case;

/// The largest |y - ref| over the sum of the terms' magnitudes an fp32 sum may have: a few units of rounding a term.
fn bound(k: usize) f64 {
    return 0x1p-24 * (8 + 2 * @sqrt(@as(f64, @floatFromInt(k))));
}

/// What rounding the output to the activation type may add, relative to the value.
fn roundBound(fp16: bool) f64 {
    return if (fp16) 0x1p-10 else 0x1p-8;
}

const pick = 16;

/// Evenly spaced indices below `n`, ending on the last one: the edges are sampled too.
fn at(i: usize, count: usize, n: usize) usize {
    return if (count <= 1 or n <= count) @min(i, n - 1) else i * (n - 1) / (count - 1);
}

pub const Totals = struct { products: usize = 0, launches: usize = 0, worst: f64 = 0 };

const Ref = struct { row: usize, col: usize, v: @import("reference.zig").Value };

/// Largest error of the sampled outputs of `out`, over the bound that holds it.
fn worstOf(p: *const Product, out: hip.DeviceBuffer, rounded: bool, refs: []const Ref) !f64 {
    var worst: f64 = 0;
    const lim = bound(p.c.k);
    for (refs) |r| {
        var y: f64 = undefined;
        const at_ = (r.row * p.c.n + r.col);
        if (rounded) {
            var b: u16 = 0;
            try out.download(at_ * 2, std.mem.asBytes(&b));
            if (p.rig.fp16) {
                const h: f16 = @bitCast(b);
                y = h;
            } else {
                y = @as(f32, @bitCast(@as(u32, b) << 16));
            }
            // the activation type's own rounding, and the fp32 sum below it
            const allowed = roundBound(p.rig.fp16) * @abs(r.v.y) + lim * r.v.norm + 1e-7;
            worst = @max(worst, @abs(y - r.v.y) / allowed);
        } else {
            var f: f32 = 0;
            try out.download(at_ * 4, std.mem.asBytes(&f));
            worst = @max(worst, @abs(@as(f64, f) - r.v.y) / (lim * @max(r.v.norm, 1e-30)));
        }
    }
    return worst;
}

const Seen = struct { e: *const Entry, digest: u64 };

fn timeOf(p: *const Product, e: *const Entry, out: hip.DeviceBuffer, rounded: bool, partial: u64) !f64 {
    const rig = p.rig;
    var start = try hip.Event.init(rig.gpu.d, true);
    defer start.deinit();
    var stop = try hip.Event.init(rig.gpu.d, true);
    defer stop.deinit();
    const iters: usize = if (p.copies > 1) @max(p.copies, 24) * rig.reps / 5 else rig.reps;
    var best: f64 = std.math.inf(f64);
    for (0..3) |round| {
        try start.record(rig.stream);
        for (0..iters) |i| try prod.launchEntry(p, e, out, rounded, partial, i % p.copies);
        try stop.record(rig.stream);
        try stop.synchronize();
        const us = @as(f64, try hip.Event.elapsedMs(start, stop)) * 1000.0 / @as(f64, @floatFromInt(iters));
        if (round > 0) best = @min(best, us);
    }
    return best;
}

fn name(buf: []u8, c: Case, rows: usize) []const u8 {
    return std.fmt.bufPrint(buf, "{s} m{d} n{d} k{d} b{d} g{d}", .{ c.name, rows, c.n, c.k, c.bits, c.group }) catch "?";
}

/// One product: every entry that takes it, against the reference, and the pickable ones against each other.
pub fn product(rig: *Rig, c: Case, tot: *Totals) !void {
    const gpa = rig.gpu.gpa;
    var p = try Product.init(rig, c);
    defer p.deinit();
    var label_buf: [160]u8 = undefined;
    const label = name(&label_buf, c, p.out_rows);

    var refs: [pick * pick]Ref = undefined;
    var n_refs: usize = 0;
    for (0..@min(pick, p.out_rows)) |i| for (0..@min(pick, c.n)) |j| {
        const row = at(i, pick, p.out_rows);
        const col = at(j, pick, c.n);
        refs[n_refs] = .{ .row = row, .col = col, .v = p.reference(row, col) };
        n_refs += 1;
    };

    const env = rig.kernels.env();
    var pickable_set: std.ArrayList(*const Entry) = .empty;
    defer pickable_set.deinit(gpa);
    try prod.pickable(rig.kernels, p.arg(0, 0), p.items, c.path, env, &pickable_set, gpa);

    var out = try hip.DeviceBuffer.alloc(rig.gpu.d, p.outBytes(false));
    defer out.free();
    // a split entry keeps a dot and a sum of x per group
    var partial = try hip.DeviceBuffer.alloc(rig.gpu.d, if (c.path == .decode and !c.routed()) (c.k / c.group) * p.out_rows * c.n * 8 else 8);
    defer partial.free();
    var seen: [2][hip.registry.max_entries]Seen = undefined;
    var n_seen: [2]usize = .{ 0, 0 };
    var times: std.ArrayList(u8) = .empty;
    defer times.deinit(gpa);
    var ran: usize = 0;
    for (rig.kernels.reg.entries) |*e| {
        if (!prod.takes(&p, e)) continue;
        const shape = prod.shapeOf(&p, e);
        for ([_]bool{ false, true }) |rounded| {
            if (rounded and (!e.roundsAct(shape) or e.family == .split or c.routed())) continue;
            try out.fill8(0xA5, null);
            check.step("{s}: {s}{s}", .{ label, e.id, if (rounded) " (rounded)" else "" });
            try prod.launchEntry(&p, e, out, rounded, partial.ptr, 0);
            try rig.stream.synchronize();
            const worst = try worstOf(&p, out, rounded, refs[0..n_refs]);
            tot.worst = @max(tot.worst, worst);
            try check.expect(worst <= 1, "affine {s}: {s}{s} is {d:.1} times the error an fp32 sum may have against float64", .{ label, e.id, if (rounded) " (rounded)" else "", worst });
            const digest = try prod.digestOf(rig.gpu, out, p.outBytes(rounded));
            rig.digest = std.hash.Wyhash.hash(rig.digest, std.mem.asBytes(&digest));
            const r: usize = @intFromBool(rounded);
            if (std.mem.indexOfScalar(*const Entry, pickable_set.items, e) != null) {
                seen[r][n_seen[r]] = .{ .e = e, .digest = digest };
                n_seen[r] += 1;
            }
            ran += 1;
            if (rig.bench) {
                const us = try timeOf(&p, e, out, rounded, partial.ptr);
                const rate = if (c.path == .decode) p.bytes() / us / 1e3 else p.flops() / us / 1e6;
                try times.print(gpa, " {s}{s} {d:.1} us {d:.0} {s} |", .{ e.id[std.mem.lastIndexOfScalar(u8, e.id, '.').? + 1 ..], if (rounded) "*" else "", us, rate, if (c.path == .decode) "GB/s" else "TFLOPS" });
            }
        }
    }
    try check.expect(ran > 0, "affine {s}: no registered entry takes the product", .{label});
    // under the default rules the entries the registry picks between by rows write one set of bytes
    // (a routed decode plan keeps its rows by the item's rows: the rows group checks it)
    if (env.stream_on and env.gemm_on and !(c.routed() and c.path == .decode)) for (seen, n_seen) |picked, n| {
        if (n < 2) continue;
        for (picked[1..n]) |s| try check.expect(s.digest == picked[0].digest, "affine {s}: {s} writes other bytes than {s}, which the registry picks between by rows", .{ label, s.e.id, picked[0].e.id });
    };
    if (rig.bench) std.debug.print("RESULT bench {s} {s}:{s}\n", .{ @tagName(c.path), label, times.items });
    tot.products += 1;
    tot.launches += ran;
}

/// A table of products: the decode or prefill shapes.
pub fn list(rig: *Rig, table: []const Case, label: []const u8) !void {
    var tot: Totals = .{};
    for (table) |c| try product(rig, c, &tot);
    check.pass("affine {s}: {d} products, {d} kernel launches each within the fp32 bound of float64, the entries picked by rows byte-identical", .{ label, tot.products, tot.launches });
    std.debug.print("DIGEST affine {s} {x}\n", .{ label, rig.digest });
}

fn ragged(comptime path: hip.registry.Path) [6 * 3 * 4 * 3 + 6]Case {
    var out: [6 * 3 * 4 * 3 + 6]Case = undefined;
    var i: usize = 0;
    const rows = if (path == .prefill) [_]usize{ 64, 100, 129, 300 } else [_]usize{ 1, 3, 8, 16 };
    const cols = [_]usize{ 1, 33, 130, 257 };
    for (cases.widths) |bits| for (cases.groups) |group| for (rows, 0..) |m, r| {
        // k in groups: one, an odd stage count of 32-code stages (group 32 only) and a few
        const ks = [_]usize{ group, group * 3, if (group == 32) 2080 else 4 * @as(usize, group) };
        for (ks, 0..) |k, j| {
            out[i] = .{ .name = "ragged", .path = path, .rows = m, .n = cols[(r + j) % cols.len], .k = k, .bits = bits, .group = group };
            i += 1;
        }
    };
    // routed: short items over a few experts
    for (cases.widths) |bits| {
        out[i] = .{ .name = "ragged routed", .path = path, .rows = 300, .n = 200, .k = 256, .bits = bits, .experts = 40, .slots = 2, .item_rows = if (path == .prefill) 128 else 8 };
        i += 1;
    }
    return out;
}

/// Every width and group on ragged sizes, odd stage counts and short routed items, on both paths.
pub fn sweep(rig: *Rig) !void {
    var tot: Totals = .{};
    for (ragged(.prefill)) |c| try product(rig, c, &tot);
    for (ragged(.decode)) |c| try product(rig, c, &tot);
    check.pass("affine sweep: {d} ragged products (every width and group, odd stage counts, short routed items), {d} kernel launches", .{ tot.products, tot.launches });
}

/// Timings of every tile on a short prompt's rows (dense and routed): printed with `--bench` only.
pub fn short(rig: *Rig) !void {
    var tot: Totals = .{};
    for (cases.short) |base| {
        const rows = if (base.routed()) &cases.short_tokens else &cases.short_rows;
        for (rows) |m| {
            var c = base;
            c.rows = m;
            try product(rig, c, &tot);
        }
    }
    check.pass("affine short: {d} products on a short prompt's rows, {d} kernel launches", .{ tot.products, tot.launches });
}
