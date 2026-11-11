//! TF_DSV41_CONE_CHECK=<sizes>: the cone prefill against the full path on slot 0, at each prompt size (comma list).
//!
//! For each size: prefill, last-token logits, kept state (storage, live, taps), greedy tokens; once through the full
//! path and once through the cone, then the comparison and each path's prefill time.
const std = @import("std");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const remote = @import("deepseek_v41_remote.zig");

const vocab = 129280;
const log = std.log.scoped(.cone_check);

fn envText(name: [:0]const u8) ?[]const u8 {
    return (@import("sys").environment.Process{}).get(name);
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

fn seconds(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds())) / 1e9;
}

/// One path's run: the prefill's last row, its kept state, the greedy ids after it.
const Run = struct {
    logits: []f32,
    snap: *ds.Model.Snapshot,
    picks: []u32,
    steps: []f32,
    prefill_s: f64,
};

fn prefill(m: *ds.Model, a: std.mem.Allocator, io: std.Io, ids: []const u32) !struct { []f32, f64 } {
    const rows = m.workspace.plan.rows;
    try m.reset();
    const t0 = seconds(io);
    var at: usize = 0;
    var last: []f32 = &.{};
    while (at < ids.len) {
        const end = @min(at + rows, ids.len);
        if (end == ids.len) last = try a.dupe(f32, try m.forward(ids[at..end], .last)) else _ = try m.forward(ids[at..end], .none);
        at = end;
    }
    return .{ last, seconds(io) - t0 };
}

fn run(m: *ds.Model, a: std.mem.Allocator, io: std.Io, ids: []const u32, decode_steps: usize, cone: bool) !Run {
    ds.forward_cone.force(m, cone);
    const pre = try prefill(m, a, io, ids);
    const snap = try m.snapshotSave(0);
    const picks = try a.alloc(u32, decode_steps);
    const steps = try a.alloc(f32, decode_steps * vocab);
    var logits: []const f32 = pre[0];
    for (0..decode_steps) |k| {
        picks[k] = argmax(logits);
        logits = try m.forward(&[1]u32{picks[k]}, .last);
        @memcpy(steps[k * vocab ..][0..vocab], logits);
    }
    return .{ .logits = pre[0], .snap = snap, .picks = picks, .steps = steps, .prefill_s = pre[1] };
}

fn diffBytes(want: []const u8, got: []const u8, first: *usize) usize {
    var n: usize = 0;
    for (want, got, 0..) |x, y, i| if (x != y) {
        if (n == 0) first.* = i;
        n += 1;
    };
    return n;
}

fn diffFloats(want: []const f32, got: []const f32) struct { usize, f32 } {
    var n: usize = 0;
    var worst: f32 = 0;
    for (want, got) |x, y| if (@as(u32, @bitCast(x)) != @as(u32, @bitCast(y))) {
        n += 1;
        worst = @max(worst, @abs(x - y));
    };
    return .{ n, worst };
}

fn sha(picks: []const u32) [8]u8 {
    var h: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(std.mem.sliceAsBytes(picks), &h, .{});
    return h[0..8].*;
}

fn compare(m: *const ds.Model, n: usize, full: Run, cone: Run, decode_steps: usize) void {
    const lg = diffFloats(full.logits, cone.logits);
    var steps_bad: usize = 0;
    var steps_worst: f32 = 0;
    for (0..decode_steps) |k| {
        const d = diffFloats(full.steps[k * vocab ..][0..vocab], cone.steps[k * vocab ..][0..vocab]);
        steps_bad += @intFromBool(d[0] != 0);
        steps_worst = @max(steps_worst, d[1]);
    }
    var first: usize = 0;
    const st = diffBytes(full.snap.storage, cone.snap.storage, &first);
    const st_first = first;
    first = 0;
    const lv = diffBytes(full.snap.live, cone.snap.live, &first);
    const lv_first = first;
    first = 0;
    const tp = diffBytes(full.snap.taps, cone.snap.taps, &first);
    const ring_bytes = @as(usize, m.caches[0].plan.layers) * ds.kv.cache.window * ds.kv.cache.ring_stride;
    const layer_bytes = ds.kv.cache.window * ds.kv.cache.ring_stride;
    log.info("n={d}: last-token logits differ in {d}/{d} values (worst {e}), argmax {d}/{d}", .{ n, lg[0], vocab, lg[1], argmax(full.logits), argmax(cone.logits) });
    log.info("n={d}: greedy {d} tokens sha full {x} cone {x}, {d} step rows differ (worst {e})", .{ n, decode_steps, sha(full.picks), sha(cone.picks), steps_bad, steps_worst });
    log.info("n={d}: storage {d} of {d} bytes differ (first {d}), live {d} of {d} (first {d}, ring layer {d}, ring {d} B), taps {d} of {d}; pos {d}/{d} taps_row {d}/{d} taps_n {d}/{d} last {d}/{d} base {d}/{d}", .{
        n,                   st,                 full.snap.storage.len, st_first,           lv,                 full.snap.live.len, lv_first,         lv_first / layer_bytes,   ring_bytes,               tp,
        full.snap.taps.len,  full.snap.position, cone.snap.position,    full.snap.taps_row, cone.snap.taps_row, full.snap.taps_n,   cone.snap.taps_n, full.snap.taps_rows_last, cone.snap.taps_rows_last, full.snap.taps_base,
        cone.snap.taps_base,
    });
    log.info("n={d}: prefill full {d:.2} s ({d:.0} tok/s), cone {d:.2} s ({d:.0} tok/s), x{d:.2}", .{
        n,
        full.prefill_s,
        @as(f64, @floatFromInt(n)) / full.prefill_s,
        cone.prefill_s,
        @as(f64, @floatFromInt(n)) / cone.prefill_s,
        full.prefill_s / cone.prefill_s,
    });
}

fn tokens(a: std.mem.Allocator, io: std.Io, n: usize) ![]u32 {
    const out = try a.alloc(u32, n);
    if (envText("TF_DSV41_CONE_TOKENS")) |path| {
        const src = try remote.parseIds(a, try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)));
        for (out, 0..) |*t, i| t.* = src[i % src.len];
        return out;
    }
    var x: u64 = 0x9e3779b97f4a7c15;
    for (out) |*t| {
        x = x *% 6364136223846793005 +% 1442695040888963407;
        t.* = @intCast(1000 + (x >> 33) % 100000);
    }
    return out;
}

/// Runs the check for the sizes in `spec`; a size the slot cannot hold is skipped.
pub fn runSizes(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, spec: []const u8) !void {
    const decode_steps: usize = if (envText("TF_DSV41_CONE_STEPS")) |s| try std.fmt.parseInt(usize, s, 10) else 128;
    var it = std.mem.tokenizeScalar(u8, spec, ',');
    // warm both paths: the first prefill pays pipeline builds and page-in
    {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const ids = try tokens(a, io, 1024);
        _ = try run(m, a, io, ids, 2, false);
        _ = try run(m, a, io, ids, 2, true);
    }
    while (it.next()) |word| {
        const n = try std.fmt.parseInt(usize, word, 10);
        if (n + decode_steps + 1 > m.caches[0].plan.capacity) {
            log.err("n={d} + {d} steps does not fit the context ({d}): skipped", .{ n, decode_steps, m.caches[0].plan.capacity });
            continue;
        }
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const ids = try tokens(a, io, n);
        const full = try run(m, a, io, ids, decode_steps, false);
        const cone = try run(m, a, io, ids, decode_steps, true);
        compare(m, n, full, cone, decode_steps);
        // the second timing of each path, state warm
        const full2 = try run(m, a, io, ids, 1, false);
        const cone2 = try run(m, a, io, ids, 1, true);
        log.info("n={d}: second prefill full {d:.2} s ({d:.0} tok/s), cone {d:.2} s ({d:.0} tok/s), x{d:.2}", .{
            n,
            full2.prefill_s,
            @as(f64, @floatFromInt(n)) / full2.prefill_s,
            cone2.prefill_s,
            @as(f64, @floatFromInt(n)) / cone2.prefill_s,
            full2.prefill_s / cone2.prefill_s,
        });
        m.snapshotDrop(full.snap);
        m.snapshotDrop(cone.snap);
        m.snapshotDrop(full2.snap);
        m.snapshotDrop(cone2.snap);
    }
    ds.forward_cone.force(m, false);
    try m.reset();
    log.info("done", .{});
}
