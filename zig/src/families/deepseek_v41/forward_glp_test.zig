//! Real GLP hooks against FP64, with missing-companion and shifted-layer controls; only fixture-used fields are set.
const std = @import("std");
const t = std.testing;
const mtl = @import("metal");
const model = @import("model.zig");
const workspace = @import("workspace.zig");
const forward_glp = @import("forward_glp.zig");
const glp = @import("metal/glp.zig");

const Model = model.Model;
const hidden = 5120;
const rows = 3;
const backbone = 40;
const alpha: f32 = 0.75;
const ids = [2]u32{ 2, 3 };
const branch_bytes = rows * hidden * 2;
const stream_bytes = rows * 4 * hidden * 2;

fn draw(s: *u64) f32 {
    s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
    const u: u32 = @intCast(s.* >> 40);
    return @as(f32, @floatFromInt(u)) / 8388608.0 - 1.0;
}

fn bf16(x: f32) u16 {
    const bits: u32 = @bitCast(x);
    return @intCast((bits +% 0x7fff +% ((bits >> 16) & 1)) >> 16);
}

fn real(h: u16) f64 {
    return @as(f32, @bitCast(@as(u32, h) << 16));
}

/// Independent FP64 outputs and magnitudes for BF16 rows under one block's direction.
const Want = struct {
    v: []f64,
    mag: []f64,

    fn make(mode: glp.Mode, d: []const f32, copies: usize, x: []const u16) !Want {
        const w: Want = .{ .v = try t.allocator.alloc(f64, x.len), .mag = try t.allocator.alloc(f64, x.len) };
        const a: f64 = alpha;
        const per = copies * hidden;
        for (0..x.len / per) |r| {
            const row = x[r * per ..][0..per];
            var dot: f64 = 0;
            var big: f64 = 0;
            for (0..hidden) |j| {
                var sum: f64 = 0;
                var abs: f64 = 0;
                for (0..copies) |c| {
                    const v = real(row[c * hidden + j]);
                    sum += v;
                    abs += @abs(v);
                }
                dot += sum / @as(f64, @floatFromInt(copies)) * d[j];
                big += @abs(@as(f64, d[j])) * abs / @as(f64, @floatFromInt(copies));
            }
            for (0..per) |i| {
                const v = real(row[i]);
                const dj: f64 = d[i % hidden];
                w.v[r * per + i] = if (mode == .add) v + a * dj else v - a * dot * dj;
                w.mag[r * per + i] = @abs(v) + a * @abs(dj) * (if (mode == .add) 1 else big);
            }
        }
        return w;
    }

    fn deinit(w: Want) void {
        t.allocator.free(w.v);
        t.allocator.free(w.mag);
    }

    /// Elements of `got` off the answer by more than a bf16 rounding and the fp32 chains' error.
    fn off(w: Want, got: []const u16) usize {
        var n: usize = 0;
        for (got, w.v, w.mag) |g, v, m| {
            if (@abs(real(g) - v) > 0x1p-8 * @abs(v) + 1e-5 * m) n += 1;
        }
        return n;
    }

    /// Elements where this answer and `o` differ by more than twice the tolerance: they tell the two blocks apart.
    fn apart(w: Want, o: Want) usize {
        var n: usize = 0;
        for (w.v, o.v, w.mag) |a, b, m| {
            if (@abs(a - b) > 2 * (0x1p-8 * @abs(a) + 1e-5 * m)) n += 1;
        }
        return n;
    }
};

const Fix = struct {
    mode: glp.Mode,
    queue: mtl.Queue,
    buf: mtl.Buffer,
    m: Model,
    plan: *glp.Plan,
    dirs: [2][hidden]f32,
    xw: [rows * hidden]u16,
    xs: [rows * 4 * hidden]u16,

    fn open(fx: *Fix, device: mtl.Device, queue: mtl.Queue, mode: glp.Mode) !void {
        var seed: u64 = 0x9e3779b97f4a7c15;
        for (&fx.dirs) |*d| for (d) |*v| {
            v.* = draw(&seed);
        };
        for (&fx.xw) |*v| v.* = bf16(2 * draw(&seed));
        for (&fx.xs) |*v| v.* = bf16(2 * draw(&seed));
        const layers = [_]glp.Layer{ .{ .id = ids[0], .dir = &fx.dirs[0] }, .{ .id = ids[1], .dir = &fx.dirs[1] } };
        fx.plan = (try glp.Plan.init(t.allocator, device, glp.Source{ .mode = mode, .alpha = alpha, .layers = &layers }, .{ .hidden = hidden, .layers = backbone })) orelse return error.NoPlan;
        errdefer fx.plan.deinit();
        fx.buf = try device.buffer(branch_bytes + 2 * stream_bytes, mtl.ResourceOptions.shared);
        // only the three workspace fields the wrappers view are laid out; the rest of the plan is never read
        var plan: workspace.Plan = undefined;
        plan.rows = rows;
        plan.lanes = 1;
        plan.bytes = branch_bytes + 2 * stream_bytes;
        plan.offsets = @splat(0);
        plan.offsets[@backingInt(workspace.Field.branch)] = 0;
        plan.offsets[@backingInt(workspace.Field.stream)] = branch_bytes;
        plan.offsets[@backingInt(workspace.Field.stream_next)] = branch_bytes + stream_bytes;
        fx.m.glp = fx.plan;
        fx.m.workspace = .{ .buffer = fx.buf, .shared_maps = fx.buf, .plan = plan };
        fx.m.pending_expand = false;
        fx.m.stream_alt = false;
        fx.mode = mode;
        fx.queue = queue;
    }

    fn close(fx: *Fix) void {
        fx.buf.deinit();
        fx.plan.deinit();
    }

    fn writerOut(fx: *const Fix) []const u16 {
        return fx.buf.slice(u16, rows * hidden);
    }

    fn foldOut(fx: *const Fix) []const u16 {
        return @as([*]const u16, @ptrCast(@alignCast(fx.buf.contents() + branch_bytes)))[0 .. rows * 4 * hidden];
    }

    /// The inputs reloaded, then the actual wrappers for the given block indices (null: that hook is omitted).
    fn run(fx: *Fix, writer: ?usize, fold: ?usize) !void {
        @memcpy(fx.buf.contents()[0..branch_bytes], std.mem.sliceAsBytes(&fx.xw));
        @memcpy(fx.buf.contents()[branch_bytes..][0..stream_bytes], std.mem.sliceAsBytes(&fx.xs));
        const cb = fx.queue.commandBuffer();
        const enc = cb.compute(.serial);
        if (writer) |l| forward_glp.writer(&fx.m, enc, l, rows);
        if (fold) |l| forward_glp.fold(&fx.m, enc, l, rows);
        enc.end();
        cb.commit();
        cb.wait();
        try t.expect(cb.failure() == null);
    }
};

/// One run's two comparisons against the block-`ids[0]` answers: each holds (`ok`) or is plainly wrong.
fn expectMatch(fx: *Fix, writer: ?usize, fold: ?usize, wr: Want, st: Want, w_ok: bool, f_ok: bool) !void {
    try fx.run(writer, fold);
    const wo = wr.off(fx.writerOut());
    const fo = st.off(fx.foldOut());
    if (w_ok) try t.expectEqual(@as(usize, 0), wo) else try t.expect(wo > rows * hidden / 4);
    if (f_ok) try t.expectEqual(@as(usize, 0), fo) else try t.expect(fo > rows * 4 * hidden / 4);
}

fn probe(device: mtl.Device, queue: mtl.Queue, mode: glp.Mode) !void {
    var fx: Fix = undefined;
    try fx.open(device, queue, mode);
    defer fx.close();
    const wr = [2]Want{ try Want.make(mode, &fx.dirs[0], 1, &fx.xw), try Want.make(mode, &fx.dirs[1], 1, &fx.xw) };
    defer for (wr) |w| w.deinit();
    const st = [2]Want{ try Want.make(mode, &fx.dirs[0], 4, &fx.xs), try Want.make(mode, &fx.dirs[1], 4, &fx.xs) };
    defer for (st) |w| w.deinit();
    // the two blocks' answers are far apart, so a wrong index cannot pass as the right one
    try t.expect(wr[0].apart(wr[1]) > rows * hidden / 4);
    try t.expect(st[0].apart(st[1]) > rows * 4 * hidden / 4);
    // each steered block: both hooks match their own direction's answer
    for (0..2) |k| try expectMatch(&fx, ids[k], ids[k], wr[k], st[k], true, true);
    // a block the vector does not cover leaves both buffers as they were
    try fx.run(4, 4);
    try t.expectEqualSlices(u16, &fx.xw, fx.writerOut());
    try t.expectEqualSlices(u16, &fx.xs, fx.foldOut());
    // planted controls, each judged by the same comparisons: an omitted hook, a shifted block index (either hook)
    try expectMatch(&fx, null, ids[0], wr[0], st[0], false, true);
    try expectMatch(&fx, ids[0], null, wr[0], st[0], true, false);
    try expectMatch(&fx, ids[1], ids[1], wr[0], st[0], false, false);
    try expectMatch(&fx, ids[1], ids[0], wr[0], st[0], false, true);
    try expectMatch(&fx, ids[0], ids[1], wr[0], st[0], true, false);
}

test "GLP hooks: the real writer and fold wrappers edit the right buffers with the right block's direction" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    for ([_]glp.Mode{ .add, .project }) |mode| try probe(device, queue, mode);
}
