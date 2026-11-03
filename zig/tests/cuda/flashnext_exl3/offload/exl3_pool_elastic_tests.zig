//! exl3-pool elastic: outputs byte-equal to the resident layer while the active line count moves (device, host rounds).
const std = @import("std");
const cuda = @import("cuda");
const flash = @import("flashnext");
const check = @import("../../check.zig");
const data = @import("exl3_pool_data.zig");
const equal = @import("exl3_pool_equal_tests.zig");

const exl3e = cuda.exl3_experts;
const fpool = flash.exl3_pool;
const dplan = flash.exl3.dplan;
const Gpu = check.Gpu;
const LayerData = data.LayerData;
const Io = data.Io;

const dims = data.dims;
const slots = data.slots;
const max_rows = data.max_rows;
const routed_n = data.routed_n;
const capacity = 64;
const phases = [_]usize{ 64, 32, 10, 40, 10, 64 };

/// The layer's routed bytes in pinned memory, as the engine's bank is: the device planner copies from them directly.
const Pinned = struct {
    buf: cuda.HostBuffer,
    routed: [routed_n][3][]const u8,

    fn init(gpu: Gpu, ld: *const LayerData) !Pinned {
        var total: usize = 0;
        for (ld.routed) |r| for (r) |b| {
            total += (b.len + 255) / 256 * 256;
        };
        const buf = try cuda.HostBuffer.alloc(gpu.d, total);
        var out: Pinned = .{ .buf = buf, .routed = undefined };
        var at: usize = 0;
        for (&out.routed, ld.routed) |*dst, src| for (0..3) |w| {
            @memcpy(buf.bytes[at..][0..src[w].len], src[w]);
            dst[w] = buf.bytes[at..][0..src[w].len];
            at += (src[w].len + 255) / 256 * 256;
        };
        return out;
    }

    fn deinit(p: *Pinned) void {
        p.buf.free();
    }
};

fn freeBytes(gpu: Gpu) !usize {
    var free: usize = 0;
    var total: usize = 0;
    try gpu.d.check(gpu.d.api.cuMemGetInfo_v2(&free, &total), "cuMemGetInfo");
    return free;
}

fn compare(gpu: Gpu, s: cuda.Stream, vp: Io, vr: Io, rows: usize, what: []const u8) !void {
    try s.synchronize();
    const gpa = gpu.gpa;
    const yp = try vp.y(gpa, rows);
    defer gpa.free(yp);
    const yr = try vr.y(gpa, rows);
    defer gpa.free(yr);
    check.sameBytes(what, yp, yr) catch |e| {
        std.debug.print("    {s}: {d} rows differ\n", .{ what, rows });
        return e;
    };
    const op = try gpa.alloc(u8, rows * dims * 4);
    defer gpa.free(op);
    const or_ = try gpa.alloc(u8, rows * dims * 4);
    defer gpa.free(or_);
    try vp.out.download(0, op);
    try vr.out.download(0, or_);
    try check.sameBytes(what, op, or_);
}

/// A round whose misses cannot fit the lines: 16 rows of 10 experts no row shares.
fn distinctPicks(picks: []i32, rows: usize) void {
    for (0..rows) |r| {
        for (0..10) |k| picks[r * slots + k] = @intCast(r * 10 + k + 100);
        picks[r * slots + 10] = 512;
    }
}

pub fn elasticRun(gpu: Gpu, ex: *const exl3e.Experts, s: cuda.Stream, ld: *const LayerData, seed: u64) !void {
    const gpa = gpu.gpa;
    var bank = try Pinned.init(gpu, ld);
    defer bank.deinit();
    var pool = try fpool.Pool.init(gpa, gpu.io, gpu.d, .{ .routed = &bank.routed, .shared = ld.shared }, ld.scales(), .{
        .layer = @intCast(ld.layer),
        .lines = capacity,
        .prefetch_bytes = 0,
        .max_passes = fpool.passesFor(fpool.min_lines),
        .elastic = true,
    }, null);
    defer pool.deinit();
    const base = pool.lines.mem.ptr;
    var pools = [_]*fpool.Pool{&pool};
    const dp = try dplan.DevPlan.init(gpa, gpu.d, &pools);
    defer dp.deinit();
    var vp = try Io.init(gpu, pool.tables());
    defer vp.free();
    var vr = try Io.init(gpu, ld.res);
    defer vr.free();
    const z = data.Zipf.init(0xe1a5 + ld.layer, 1.1);
    var prng = std.Random.DefaultPrng.init(seed);
    const rng = prng.random();
    var x: [max_rows * dims]u16 = undefined;
    var picks: [max_rows * slots]i32 = undefined;
    var wts: [max_rows * slots]f32 = undefined;
    var rounds: usize = 0;
    var device_rounds: usize = 0;
    var forced: usize = 0;
    var previous: usize = capacity;
    for (phases, 0..) |active, phase| {
        if (phase != 0) {
            try s.synchronize();
            const free_before = try freeBytes(gpu);
            const chunks_before = pool.lines.window.?.mapped;
            try dp.setActive(0, active);
            try check.expect(pool.lines.mem.ptr == base, "the lines moved on a resize to {d}", .{active});
            try check.expect(pool.active == active and pool.lines.mem.len >= active * pool.lines.layout.line_bytes, "{d} lines but {d} active, {d} bytes", .{ active, pool.active, pool.lines.mem.len });
            const from = @min(active, previous);
            for (pool.slots_tab[from..], pool.line_tab[from..]) |id, state| {
                try check.expect(state == .empty and id == fpool.offload.Cache.empty_id, "a line past {d} kept expert {d}", .{ from, id });
            }
            const mapped = pool.lines.window.?;
            std.debug.print("  [elastic] {d} -> {d} lines: {d} -> {d} chunks of {d} MiB, device free {d} MiB -> {d} MiB\n", .{ previous, active, chunks_before, mapped.mapped, mapped.win.granularity >> 20, free_before >> 20, (try freeBytes(gpu)) >> 20 });
            previous = active;
        }
        for (0..14) |round| {
            const rows: usize = if (round == 5 or round == 11) max_rows else if (round == 8) 4 else 1;
            data.genRound(&z, rng, rows, &x, &picks, &wts);
            const force = round == 11;
            if (force) distinctPicks(&picks, rows);
            const n = rows * slots;
            try vp.set(x[0 .. rows * dims], picks[0..n], wts[0..n]);
            try vr.set(x[0 .. rows * dims], picks[0..n], wts[0..n]);
            try vp.sbuf.fill8(0xff, s.handle);
            if (round % 3 == 2 and !force) {
                try dp.toHost(); // a host-planned round on the active prefix
                _ = try data.poolRound(&pool, ex, s, vp, picks[0..n], rows, true);
            } else {
                const before = (try dp.peek()).inplace;
                try dp.toDevice(s);
                try dplan.runLayer(dp, s, 0, ex, pool.tables(), vp.x.ptr, vp.picks.ptr, rows, vp.sc, vp.wts.ptr, vp.out.ptr);
                device_rounds += 1;
                if (force) {
                    try s.synchronize();
                    const after = (try dp.peek()).inplace;
                    try check.expect(active > 10 or after > before, "a forced round at {d} lines ran no expert in place", .{active});
                    if (after > before) forced += 1;
                }
            }
            try data.call(ex, s, ld.res, vr, vr.picks.ptr, rows, true);
            try compare(gpu, s, vp, vr, rows, "elastic pool vs resident");
            rounds += 1;
        }
        try dp.toHost();
        const resident = try equal.verifyPool(gpu, &pool, ld);
        std.debug.print("  [elastic] {d} active lines: {d} resident lines and the tables match the checkpoint\n", .{ active, resident });
    }
    try check.expect(forced >= 2 and device_rounds > rounds / 2, "device rounds {d} of {d}, forced in-place rounds {d}", .{ device_rounds, rounds, forced });
    check.pass("layer {d}: {d} rounds ({d} device-planned, {d} forced into in-place experts) over active counts {any} of {d} lines; lines at {x} throughout; every output byte-identical to the fully resident layer", .{ ld.layer, rounds, device_rounds, forced, phases, capacity, base });
}
