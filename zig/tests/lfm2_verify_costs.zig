//! LFM2 verify forwards timed by stream count and rows a stream: the round costs a DSpark verify budget weighs.
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm2_moe;
const st = q.state;
const c = q.config;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

/// Stream counts profiled; each runs every rows-a-stream width whose total fits a shared forward.
const counts = [_]usize{ 1, 2, 3, 4, 6, 8, 12, 16 };
const reps = 9;

fn readIds(gpa: std.mem.Allocator, io: std.Io, path: []const u8, list: *std.ArrayList(u32)) !void {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(1 << 22));
    defer gpa.free(bytes);
    var it = std.mem.tokenizeAny(u8, bytes, " \n\r\t,");
    while (it.next()) |t| try list.append(gpa, try std.fmt.parseInt(u32, t, 10));
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return error.ExpectedModelOutputHistory;
    var history: usize = if (args.len >= 4) try std.fmt.parseInt(usize, args[3], 10) else 1024;
    // Token id files: every file's ids end to end; stream i reads its history and then its rows from its own sixteenth.
    var real: std.ArrayList(u32) = .empty;
    defer real.deinit(gpa);
    for (args[@min(args.len, 4)..]) |path| try readIds(gpa, init.io, path, &real);
    const part = real.items.len / 16;
    if (real.items.len > 0) {
        if (part < st.window_rows + 64) return error.IdFilesTooShort;
        history = @min(history, part - st.window_rows);
    }
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const m = try q.Model.load(gpa, init.io, args[1]);
    defer m.deinit();
    var scratch = try st.Scratch.init(gpa, m.device, &m.config.attention, @max(history, st.batch_rows));
    defer scratch.deinit();
    var caches: [16]st.Cache = undefined;
    var made: usize = 0;
    defer for (caches[0..made]) |*k| k.deinit();
    const ids = try gpa.alloc(u32, history);
    defer gpa.free(ids);
    for (caches[0..], 0..) |*k, i| {
        k.* = try st.Cache.init(gpa, m.device, &m.config.attention, history + st.window_rows + 1);
        made += 1;
        for (ids, 0..) |*id, j| id.* = if (part > 0) real.items[i * part + j] else @intCast((j * 7919 + 13 + i * 104729) % 120000);
        try q.forward.run(m, &scratch, &.{.{ .cache = k, .rows = history }}, ids, false, .last);
    }
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const w = &out.writer;
    try w.print("# LFM2 verify forward ms (median of {d}) by streams and rows a stream, {d}-token histories, {s} rows\n", .{ reps, history, if (part > 0) "real" else "synthetic" });
    try w.print("# streams rows_each total_rows ms\n", .{});
    var segments: [st.batch_rows]q.forward.Segment = undefined;
    var tokens: [st.batch_rows]u32 = undefined;
    for (&tokens, 0..) |*t, i| t.* = @intCast((i * 31 + 7) % 120000);
    for (counts) |streams| {
        var each: usize = 1;
        while (each <= st.window_rows and streams * each <= st.batch_rows) : (each += 1) {
            for (segments[0..streams], caches[0..streams], 0..) |*s, *k, i| {
                s.* = .{ .cache = k, .rows = each };
                // real rows: the tokens after the stream's history, as a verify of drafts that all land
                if (part > 0) @memcpy(tokens[i * each ..][0..each], real.items[i * part + history ..][0..each]);
            }
            var times: [reps + 2]f64 = undefined;
            for (&times) |*t| {
                const t0 = now();
                try q.forward.run(m, &scratch, segments[0..streams], tokens[0 .. streams * each], true, .all);
                t.* = @as(f64, @floatFromInt(now() - t0)) / 1e6;
                for (caches[0..streams]) |*k| k.len = history;
            }
            const kept = times[2..]; // the first two calls warm a new width's pipelines
            std.mem.sort(f64, kept, {}, std.sort.asc(f64));
            const ms = kept[reps / 2];
            try w.print("{d} {d} {d} {d:.3}\n", .{ streams, each, streams * each, ms });
            std.debug.print("{d} streams x {d} rows ({d}): {d:.3} ms\n", .{ streams, each, streams * each, ms });
        }
    }
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[2], .data = out.written() });
}
