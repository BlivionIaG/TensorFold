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
    try kernels(m);
    try prompts(m);
    try grouped(m);
    try sorting(m);
    try launches(m);
    try promptBudget(m);
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

const tile = @import("tensorfold").tile_qmm;
const row = @import("tensorfold").row_projection;

/// GPU time of 200 back-to-back calls of each decode projection kind on layer 10's real experts.
fn kernels(m: *q.Model) !void {
    const x = m.weights.blocks[10].ffn.moe;
    const dev = m.device;
    const in = try dev.buffer(16 * 11776 * 2, mtl.ResourceOptions.shared);
    defer in.deinit();
    for (in.slice(u16, 16 * 11776), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5)) >> 16);
    const out = try dev.buffer(16 * 11776 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const sums = try dev.buffer(16 * 184 * 4, mtl.ResourceOptions.shared);
    defer sums.deinit();
    const ids = try dev.buffer(64, mtl.ResourceOptions.shared);
    defer ids.deinit();
    for (ids.slice(u32, 4), 0..) |*v, i| v.* = @intCast(i * 13 + 1);
    const tq = q.forward.matrix;
    inline for (.{ "twin indexed 1536x2048", "row indexed 1536x2048", "twin dense 6144x2048", "row dense 6144x2048", "twin head 128000x2048 (6-bit)", "row head" }, 0..) |name, which| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..200) |_| switch (which) {
            0 => try tile.encodeTwinIndexed(e, &m.kernels.tiles, tq(x.gate.one), in, out, ids, 4, 4),
            1 => {
                const call = try row.callIndexed(&m.kernels.rows, x.gate, 4, false);
                e.setPipeline(call.pipeline);
                e.setBuffer(in, 0, 0);
                e.setBuffer(x.gate.one.w, x.gate.one.w_off, 1);
                e.setBuffer(x.gate.one.scales, x.gate.one.s_off, 2);
                e.setBuffer(x.gate.one.biases, x.gate.one.b_off, 3);
                e.setBytes(std.mem.asBytes(&call.dims), 4);
                e.setBuffer(out, 0, 5);
                e.setBuffer(ids, 0, 6);
                e.dispatchThreads(mtl.Size.of(call.groups * call.threads, 4, 1), mtl.Size.of(call.threads, 1, 1));
            },
            2, 4 => try tile.encodeTwin(e, &m.kernels.tiles, tq(if (which == 2) m.weights.blocks[0].mixer.conv.in else m.weights.embedding), in, 0, out, 0, 1),
            else => {
                const w = if (which == 3) m.weights.blocks[0].mixer.conv.in else m.weights.embedding;
                const call = try row.call(&m.kernels.rows, w, 1, false);
                e.setPipeline(call.pipeline);
                e.setBuffer(in, 0, 0);
                e.setBuffer(w.w, w.w_off, 1);
                e.setBuffer(w.scales, w.s_off, 2);
                e.setBuffer(w.biases, w.b_off, 3);
                e.setBytes(std.mem.asBytes(&call.dims), 4);
                e.setBuffer(out, 0, 5);
                e.dispatchThreads(mtl.Size.of(call.groups * call.threads, 1, 1), mtl.Size.of(call.threads, 1, 1));
            },
        };
        e.end();
        cb.commit();
        cb.wait();
        std.debug.print("{s}: {d:.1} us a call\n", .{ name, cb.gpuSeconds() * 1e6 / 200.0 });
    }
}

/// GPU TFLOPS of the prompt tiles on 512 rows of a dense 6144x2048 and a 2048x2048 matrix.
fn prompts(m: *q.Model) !void {
    const dev = m.device;
    const rows = 512;
    const in = try dev.buffer((rows + 32) * 11776 * 2, mtl.ResourceOptions.shared);
    defer in.deinit();
    for (in.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5)) >> 16);
    const out = try dev.buffer(rows * 11776 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const xs = try dev.buffer(rows * 184 * 4, mtl.ResourceOptions.shared);
    defer xs.deinit();
    const b = m.weights.blocks[0];
    for ([_]q.weights.Linear{ b.mixer.conv.in, b.mixer.conv.out, b.ffn.dense.gate }) |w| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        for (0..20) |_| try tile.encode(e, &m.kernels.tiles, q.forward.matrix(w), in, 0, xs, 0, out, 0, rows);
        e.end();
        cb.commit();
        cb.wait();
        const t = cb.gpuSeconds() / 20.0;
        std.debug.print("tiles {d}x{d}, {d} rows: {d:.0} us, {d:.2} TFLOPS\n", .{ w.n, w.k, rows, t * 1e6, 2.0 * @as(f64, @floatFromInt(rows * w.n * w.k)) / t / 1e12 });
    }
}

/// GPU TFLOPS of the grouped tiles on 8,192 rows, 128 an expert, over layer 10's gate experts.
fn grouped(m: *q.Model) !void {
    const dev = m.device;
    const rows = 8192;
    const x = m.weights.blocks[10].ffn.moe;
    const in = try dev.buffer((rows + 32) * 2048 * 2, mtl.ResourceOptions.shared);
    defer in.deinit();
    for (in.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 97)) / 97.0 - 0.5)) >> 16);
    const out = try dev.buffer(rows * 2048 * 2, mtl.ResourceOptions.shared);
    defer out.deinit();
    const xs = try dev.buffer(rows * 32 * 4, mtl.ResourceOptions.shared);
    defer xs.deinit();
    const starts = try dev.buffer(65 * 4, mtl.ResourceOptions.shared);
    defer starts.deinit();
    for (starts.slice(u32, 65), 0..) |*v, i| v.* = @intCast(i * 128);
    const work = try dev.buffer((rows / 32 + 66) * 4, mtl.ResourceOptions.shared);
    defer work.deinit();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    tile.encodeWork(e, &m.kernels.tiles, starts, 64, work);
    for (0..10) |_| try tile.encodeGrouped(e, &m.kernels.tiles, q.forward.matrix(x.gate.one), 64, in, xs, out, rows, starts, work);
    e.end();
    cb.commit();
    cb.wait();
    const t = cb.gpuSeconds() / 10.0;
    std.debug.print("grouped tiles 1536x2048 x64 experts, {d} rows: {d:.0} us, {d:.2} TFLOPS\n", .{ rows, t * 1e6, 2.0 * @as(f64, @floatFromInt(rows * 1536 * 2048)) / t / 1e12 });
}

/// GPU time of the expert sort on 8,192 picks.
fn sorting(m: *q.Model) !void {
    const dev = m.device;
    const slots: u32 = 8192;
    const ids = try dev.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer ids.deinit();
    for (ids.slice(u32, slots), 0..) |*v, i| v.* = @intCast((i * 2654435761) % 64);
    const starts = try dev.buffer(65 * 4, mtl.ResourceOptions.shared);
    defer starts.deinit();
    const order = try dev.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer order.deinit();
    const pos = try dev.buffer(slots * 4, mtl.ResourceOptions.shared);
    defer pos.deinit();
    const cb = m.queue.commandBuffer();
    const e = cb.compute(.serial);
    for (0..10) |_| {
        e.setPipeline(m.kernels.get("lfm2_sort"));
        e.setBuffer(ids, 0, 0);
        e.setValue(slots, 1);
        e.setBuffer(starts, 0, 2);
        e.setBuffer(order, 0, 3);
        e.setBuffer(pos, 0, 4);
        e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(64 * 16, 1, 1));
    }
    e.end();
    cb.commit();
    cb.wait();
    std.debug.print("sort of {d} picks: {d:.0} us\n", .{ slots, cb.gpuSeconds() * 1e6 / 10.0 });
}

/// GPU time a launch of the one-row norm, serial and concurrent encoders.
fn launches(m: *q.Model) !void {
    const dev = m.device;
    const h = try dev.buffer(2048 * 2 * 4, mtl.ResourceOptions.shared);
    defer h.deinit();
    @memset(h.contents()[0 .. 2048 * 2 * 4], 0);
    const w = m.weights.blocks[0].operator_norm;
    inline for (.{ .serial, .concurrent }) |kind| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(kind);
        for (0..1000) |_| {
            e.setPipeline(m.kernels.get("lfm2_norm"));
            e.setBuffer(h, 0, 0);
            e.setBuffer(h, 4096, 1);
            e.setBuffer(w.buffer, w.offset, 2);
            e.setValue(@as(f32, 1e-5), 3);
            e.setValue(@as(u32, 0), 4);
            e.setBuffer(h, 8192, 5);
            e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(512, 1, 1));
        }
        e.end();
        cb.commit();
        cb.wait();
        std.debug.print("one-row norm, {s}: {d:.1} us a launch\n", .{ @tagName(kind), cb.gpuSeconds() * 1e6 / 1000.0 });
    }
}


/// One 2,048-row prompt chunk's GPU budget: each kernel at its real shape (picks spread over the experts), times its launches.
fn promptBudget(m: *q.Model) !void {
    const rows = 2048;
    const slots = rows * 4;
    const dev = m.device;
    var sc = try st.Scratch.init(m.gpa, dev, &m.config.attention, rows);
    defer sc.deinit();
    var cache = try st.Cache.init(m.gpa, dev, &m.config.attention, rows + 64);
    defer cache.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    for (sc.picks.slice(u32, slots), 0..) |*v, i| v.* = @intCast((i % 4) * 16 + prng.random().uintLessThan(u32, 16));
    for (sc.x.slice(u16, rows * 2048), 0..) |*v, i| v.* = @truncate(@as(u32, @bitCast(@as(f32, @floatFromInt(i % 89)) / 89.0 - 0.5)) >> 16);
    const tq = q.forward.matrix;
    const t = &m.kernels.tiles;
    const conv = m.weights.blocks[0].mixer.conv;
    const attn = m.weights.blocks[2].mixer.attention;
    const moe = m.weights.blocks[10].ffn.moe;
    const dense = m.weights.blocks[0].ffn.dense;
    const Item = struct { name: []const u8, per: f64 };
    const items = [_]Item{
        .{ .name = "sums+tiles conv in", .per = 30 },   .{ .name = "sums+tiles conv out", .per = 30 },
        .{ .name = "sums+tiles attn q,k,v,o", .per = 10 }, .{ .name = "sums+tiles dense x3", .per = 2 },
        .{ .name = "route", .per = 38 },                .{ .name = "sort+work+gather", .per = 38 },
        .{ .name = "sums+grouped glu gate,up", .per = 38 }, .{ .name = "sums+grouped down", .per = 38 },
        .{ .name = "combine sorted", .per = 38 },      .{ .name = "norm", .per = 81 },
        .{ .name = "conv kernel", .per = 30 },         .{ .name = "attn mma 2k", .per = 10 },
    };
    var total: f64 = 0;
    inline for (items, 0..) |it, which| {
        const cb = m.queue.commandBuffer();
        const e = cb.compute(.serial);
        const reps = 3;
        for (0..reps) |_| switch (which) {
            0 => {
                tile.encodeSums(e, t, 2048, sc.x, 0, sc.sums, rows);
                try tile.encode(e, t, tq(conv.in), sc.x, 0, sc.sums, 0, sc.bcx, 0, rows);
            },
            1 => {
                tile.encodeSums(e, t, 2048, sc.y, 0, sc.sums, rows);
                try tile.encode(e, t, tq(conv.out), sc.y, 0, sc.sums, 0, sc.r, 0, rows);
            },
            2 => for ([_]q.weights.Linear{ attn.q, attn.k, attn.v, attn.out }) |w| {
                tile.encodeSums(e, t, 2048, sc.x, 0, sc.sums, rows);
                try tile.encode(e, t, tq(w), sc.x, 0, sc.sums, 0, sc.q, 0, rows);
            },
            3 => for ([_]q.weights.Linear{ dense.gate, dense.up, dense.down }) |w| {
                tile.encodeSums(e, t, w.k, sc.act, 0, sc.sums, rows);
                try tile.encode(e, t, tq(w), sc.act, 0, sc.sums, 0, sc.gate, 0, rows);
            },
            4 => {
                e.setPipeline(m.kernels.get("lfm2_route_logits"));
                e.setBuffer(sc.x, 0, 0);
                e.setBuffer(moe.router.buffer, moe.router.offset, 1);
                e.setBuffer(sc.route, 0, 2);
                e.dispatchGroups(mtl.Size.of(8, rows, 1), mtl.Size.of(256, 1, 1));
                e.setPipeline(m.kernels.get("lfm2_route_select"));
                e.setBuffer(sc.route, 0, 0);
                e.setBuffer(moe.bias.buffer, moe.bias.offset, 1);
                e.setValue(@as(f32, 1), 2);
                e.setBuffer(sc.weights, 0, 4);
                e.setBuffer(cache.logits, 0, 3); // keep the bench's spread picks
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(32, 1, 1));
            },
            5 => {
                e.setPipeline(m.kernels.get("lfm2_sort"));
                e.setBuffer(sc.picks, 0, 0);
                e.setValue(@as(u32, slots), 1);
                e.setBuffer(sc.starts, 0, 2);
                e.setBuffer(sc.order, 0, 3);
                e.setBuffer(sc.pos, 0, 4);
                e.dispatchGroups(mtl.Size.of(1, 1, 1), mtl.Size.of(64 * 16, 1, 1));
                tile.encodeWork(e, t, sc.starts, 64, sc.work);
                e.setPipeline(m.kernels.get("lfm2_gather"));
                e.setBuffer(sc.x, 0, 0);
                e.setBuffer(sc.order, 0, 1);
                e.setBuffer(sc.sorted, 0, 2);
                e.dispatchThreads(mtl.Size.of(2048, slots, 1), mtl.Size.of(256, 1, 1));
            },
            6 => {
                tile.encodeSums(e, t, 2048, sc.sorted, 0, sc.sums, slots);
                try tile.encodeGroupedGlu(e, t, tq(moe.gate.one), tq(moe.up.one), 64, sc.sorted, sc.sums, sc.act, slots, sc.starts, sc.work);
            },
            7 => {
                tile.encodeSums(e, t, 1536, sc.act, 0, sc.sums, slots);
                try tile.encodeGrouped(e, t, tq(moe.down.one), 64, sc.act, sc.sums, sc.experts, slots, sc.starts, sc.work);
            },
            8 => {
                e.setPipeline(m.kernels.get("lfm2_combine_sorted"));
                e.setBuffer(sc.experts, 0, 0);
                e.setBuffer(sc.weights, 0, 1);
                e.setBuffer(sc.r, 0, 2);
                e.setBuffer(sc.pos, 0, 3);
                e.dispatchThreads(mtl.Size.of(2048, rows, 1), mtl.Size.of(256, 1, 1));
            },
            9 => {
                e.setPipeline(m.kernels.get("lfm2_norm"));
                e.setBuffer(sc.h, 0, 0);
                e.setBuffer(sc.r, 0, 1);
                e.setBuffer(conv.weight.buffer, conv.weight.offset, 2);
                e.setValue(@as(f32, 1e-5), 3);
                e.setValue(@as(u32, 1), 4);
                e.setBuffer(sc.x, 0, 5);
                e.dispatchGroups(mtl.Size.of(rows, 1, 1), mtl.Size.of(512, 1, 1));
            },
            10 => {
                e.setPipeline(m.kernels.get("lfm2_conv"));
                e.setBuffer(sc.bcx, 0, 0);
                e.setBuffer(cache.blocks[0].conv, 0, 1);
                e.setBuffer(conv.weight.buffer, conv.weight.offset, 2);
                e.setBuffer(sc.y, 0, 3);
                e.setBuffer(sc.snapshots[0].?, 0, 4);
                e.dispatchThreads(mtl.Size.of(2048, rows, 1), mtl.Size.of(256, 1, 1));
            },
            else => {
                const kv = cache.blocks[2].attention;
                e.setPipeline(m.kernels.get("lfm2_attn_mma"));
                e.setBuffer(sc.queries, 0, 0);
                e.setBuffer(kv.keys, 0, 1);
                e.setBuffer(kv.values, 0, 2);
                e.setBytes(std.mem.asBytes(&[3]u32{ 0, @intCast(cache.capacity), @bitCast(@as(f32, 0.125)) }), 3);
                e.setBuffer(sc.pm, 0, 4);
                e.setBuffer(sc.pl, 0, 5);
                e.setBuffer(sc.po, 0, 6);
                e.setValue(@as(u32, rows), 7);
                e.dispatchGroups(mtl.Size.of(1, 8, rows / 8), mtl.Size.of(128, 1, 1));
                e.setPipeline(m.kernels.get("lfm2_attn_merge"));
                e.setBuffer(sc.pm, 0, 0);
                e.setBuffer(sc.pl, 0, 1);
                e.setBuffer(sc.po, 0, 2);
                e.setBytes(std.mem.asBytes(&[3]u32{ 0, @intCast(cache.capacity), @bitCast(@as(f32, 0.125)) }), 3);
                e.setBuffer(sc.y, 0, 4);
                e.dispatchGroups(mtl.Size.of(32, rows, 1), mtl.Size.of(32, 1, 1));
            },
        };
        e.end();
        cb.commit();
        cb.wait();
        const ms = cb.gpuSeconds() * 1e3 / reps;
        total += ms * it.per;
        std.debug.print("{s}: {d:.2} ms x {d} = {d:.0} ms\n", .{ it.name, ms, it.per, ms * it.per });
    }
    std.debug.print("prompt chunk budget: {d:.0} ms for {d} rows ({d:.0} tok/s)\n", .{ total, rows, rows / total * 1e3 });
}
