//! LFM2 prompt-chunk and one-row decode timing on the real weights (no server).
const std = @import("std");
const mtl = @import("metal");
const q = @import("tensorfold").lfm2;
const st = q.state;

fn now() i128 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(i128, ts.sec) * std.time.ns_per_s + ts.nsec;
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return error.ExpectedModelPromptChunkDecode;
    const length = try std.fmt.parseInt(usize, args[2], 10);
    const chunk = try std.fmt.parseInt(usize, args[3], 10);
    const steps = try std.fmt.parseInt(usize, args[4], 10);
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    {
        const d = try mtl.Device.init();
        const qq = try d.queue();
        empty(qq, "fresh device");
    }
    const m = try q.Model.load(init.gpa, init.io, args[1]);
    defer m.deinit();
    empty(m.queue, "model queue");
    std.debug.print("residency set: {}\n", .{m.resident != null});
    for (m.checkpoint.shards.items) |shard| bound(m.queue, shard.buffer, shard.bytes);
    var big = try st.Scratch.init(init.gpa, m.device, &m.config.attention, chunk);
    defer big.deinit();
    var one = try st.Scratch.init(init.gpa, m.device, &m.config.attention, 1);
    defer one.deinit();
    var cache = try st.Cache.init(init.gpa, m.device, &m.config.attention, length + steps + 1);
    defer cache.deinit();
    const ids = try init.gpa.alloc(u32, length + steps);
    defer init.gpa.free(ids);
    for (ids, 0..) |*id, i| id.* = @intCast((i * 7919 + 13) % 120000);
    const t0 = now();
    var at: usize = 0;
    while (at < length) {
        const rows = @min(chunk, length - at);
        try q.forward.run(m, &big, &.{.{ .cache = &cache, .rows = rows }}, ids[at..][0..rows], false, .last);
        at += rows;
    }
    const t1 = now();
    for (0..steps) |i| try q.forward.run(m, &one, &.{.{ .cache = &cache, .rows = 1 }}, ids[length + i ..][0..1], false, .all);
    const t2 = now();
    const pf: f64 = @as(f64, @floatFromInt(t1 - t0)) / 1e9;
    const dc: f64 = @as(f64, @floatFromInt(t2 - t1)) / 1e9;
    std.debug.print("prompt {d} in {d}-row chunks: {d:.2} s, {d:.0} tok/s; decode {d}: {d:.1} tok/s\n", .{ length, chunk, pf, @as(f64, @floatFromInt(length)) / pf, steps, @as(f64, @floatFromInt(steps)) / dc });
}

fn empty(queue: mtl.Queue, what: []const u8) void {
    const t = now();
    for (0..20) |_| {
        const cb = queue.commandBuffer();
        cb.compute(.serial).end();
        cb.commit();
        cb.wait();
    }
    std.debug.print("{s}: empty command buffer {d} us\n", .{ what, @divTrunc(now() - t, 20000) });
}

fn bound(queue: mtl.Queue, b: mtl.Buffer, bytes: usize) void {
    const t = now();
    for (0..20) |_| {
        const cb = queue.commandBuffer();
        const e = cb.compute(.serial);
        e.setBuffer(b, 0, 0);
        e.end();
        cb.commit();
        cb.wait();
    }
    std.debug.print("bound {d} MiB buffer: {d} us\n", .{ bytes >> 20, @divTrunc(now() - t, 20000) });
}
