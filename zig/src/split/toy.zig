//! A tiny dense model in exact integer arithmetic, so a layer split can be compared with an unsplit run bit for bit on
//! any backend (CPU here, Metal in the metal tests). It has what a split must preserve: an embedding, layers with
//! recurrent per-layer state that must stay with the device that owns the layer, a head and a greedy draw.
//!
//! layer l: y = W_l x; a = (y + s_l) >> 3; s_l = clamp(s_l + y); x = clamp(x + a).
//! head: logits = Wh x; the token is the first maximum. Input x0 = embed[token] * 1024.
const std = @import("std");
const pipeline = @import("pipeline.zig");
const wire = @import("wire.zig");
const device = @import("device.zig");

pub const hidden = 32;
pub const vocab = 97;
pub const layers = 12;
pub const x_limit: i32 = 1 << 20;
pub const s_limit: i32 = 1 << 28;
pub const hidden_bytes = hidden * 4;
/// The most positions a toy step carries.
pub const max_rows = 16;

pub const Params = struct {
    embed: [vocab][hidden]i32,
    w: [layers][hidden][hidden]i32,
    head: [vocab][hidden]i32,

    pub fn init(seed: u64) Params {
        var p: Params = undefined;
        var rng = std.Random.SplitMix64.init(seed);
        for (&p.embed) |*row| for (row) |*v| {
            v.* = small(&rng);
        };
        for (&p.w) |*m| for (m) |*row| for (row) |*v| {
            v.* = small(&rng);
        };
        for (&p.head) |*row| for (row) |*v| {
            v.* = small(&rng);
        };
        return p;
    }

    fn small(rng: *std.Random.SplitMix64) i32 {
        return @as(i32, @intCast(rng.next() % 7)) - 3;
    }
};

/// A contiguous run of layers with the embedding, the head, both or neither, and the state of the layers it owns.
pub const Stage = struct {
    params: *const Params,
    first: usize,
    last: usize,
    embed: bool,
    head: bool,
    /// The layers' state, one set a sequence slot.
    state: [pipeline.max_slots][layers][hidden]i32 = @splat(@splat(@splat(0))),

    pub fn external(self: *Stage) pipeline.ExternalStage {
        return .{ .ptr = self, .run = runFn };
    }

    /// The device side on the CPU: refuse an aborted step, validate a stamped row, run, leave the result in the out slot.
    fn runFn(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const self: *Stage = @ptrCast(@alignCast(ptr));
        const h = io.in.header().*;
        if (h.status != 0 or h.kind == @backingInt(wire.Kind.abort)) return error.PeerLost;
        const input = if (io.stamped) blk: {
            const rows = io.scratch[0 .. io.rows * io.hidden_bytes];
            _ = try wire.readStampedRows(rows, io.in.data(), io.rows, io.hidden_bytes, io.seq, io.retries);
            break :blk rows;
        } else io.in.data()[0..h.bytes];
        _ = try self.runSlots(io.slots, io.rows, input, io.out.data());
    }

    /// One position: `input` is a token (4 bytes) when the stage embeds, else a hidden row of i32; the output is a token
    /// when it has the head.
    pub fn run(self: *Stage, input: []const u8, out: []u8) !usize {
        return self.runRows(1, input, out);
    }

    /// `rows` consecutive positions of sequence 0, in order: the state of every layer advances once a row. The output is
    /// the token after the last row when the stage has the head, else the `rows` hidden rows.
    pub fn runRows(self: *Stage, rows: usize, input: []const u8, out: []u8) !usize {
        return self.runSlots(&.{0}, rows, input, out);
    }

    /// With one slot, `rows` consecutive positions of that sequence; with `rows` slots, the next position of each. The
    /// output with the head is the token after the last row (one slot) or a token a row (a batch), else the hidden rows.
    pub fn runSlots(self: *Stage, slots: []const u8, rows: usize, input: []const u8, out: []u8) !usize {
        if (rows == 0 or input.len != rows * (if (self.embed) @as(usize, 4) else hidden_bytes)) return error.BadHidden;
        if (slots.len != 1 and slots.len != rows) return error.BadHidden;
        for (slots) |sl| if (sl >= pipeline.max_slots) return error.BadHidden;
        const batch = slots.len > 1;
        for (0..rows) |r| {
            const slot: usize = slots[if (batch) r else 0];
            var x: [hidden]i32 = undefined;
            if (self.embed) {
                const t = std.mem.readInt(u32, input[4 * r ..][0..4], .little);
                if (t >= vocab) return error.BadToken;
                for (&x, self.params.embed[t]) |*d, e| d.* = e * 1024;
            } else {
                const row = input[r * hidden_bytes ..][0..hidden_bytes];
                for (&x, 0..) |*d, i| d.* = std.mem.readInt(i32, row[i * 4 ..][0..4], .little);
            }
            for (self.first..self.last) |l| layer(self, slot, l, &x);
            if (self.head) {
                if (batch) std.mem.writeInt(u32, out[4 * r ..][0..4], draw(self.params, &x), .little) else if (r + 1 == rows) std.mem.writeInt(u32, out[0..4], draw(self.params, &x), .little);
            } else for (x, 0..) |v, i| std.mem.writeInt(i32, out[r * hidden_bytes + i * 4 ..][0..4], v, .little);
        }
        return if (self.head) (if (batch) 4 * rows else 4) else rows * hidden_bytes;
    }

    fn layer(self: *Stage, slot: usize, l: usize, x: *[hidden]i32) void {
        const s = &self.state[slot][l];
        var y: [hidden]i32 = undefined;
        for (&y, self.params.w[l]) |*d, row| {
            var acc: i32 = 0;
            for (row, x) |wv, xv| acc += wv * xv;
            d.* = acc;
        }
        for (x, s, y) |*xv, *sv, yv| {
            const a = (yv + sv.*) >> 3;
            sv.* = std.math.clamp(sv.* + yv, -s_limit, s_limit);
            xv.* = std.math.clamp(xv.* + a, -x_limit, x_limit);
        }
    }
};

pub fn draw(params: *const Params, x: *const [hidden]i32) u32 {
    var best: u32 = 0;
    var best_logit: i32 = std.math.minInt(i32);
    for (params.head, 0..) |row, v| {
        var acc: i32 = 0;
        for (row, x) |wv, xv| acc += wv * xv;
        if (acc > best_logit) {
            best_logit = acc;
            best = @intCast(v);
        }
    }
    return best;
}

/// The unsplit reference: the whole model in one stage on the CPU, the prompt then `n` greedy tokens.
pub fn reference(params: *const Params, prompt: []const u32, out: []u32) !void {
    var st: Stage = .{ .params = params, .first = 0, .last = layers, .embed = true, .head = true };
    var buf: [hidden_bytes]u8 = undefined;
    var token: u32 = 0;
    for (prompt, 0..) |t, i| {
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, t, .little);
        _ = try st.run(&b, &buf);
        token = std.mem.readInt(u32, buf[0..4], .little);
        if (i + 1 == prompt.len) break;
    }
    for (out) |*o| {
        o.* = token;
        var b: [4]u8 = undefined;
        std.mem.writeInt(u32, &b, token, .little);
        _ = try st.run(&b, &buf);
        token = std.mem.readInt(u32, buf[0..4], .little);
    }
}

/// Drive a pipeline over `prompt`, then take `out.len` greedy tokens, the same sequence `reference` makes.
pub fn generate(p: *pipeline.Pipeline, prompt: []const u32, out: []u32) pipeline.Error!void {
    return generateChunked(p, prompt, out, 1);
}

/// `generate` with the prompt going through in steps of up to `chunk` positions.
pub fn generateChunked(p: *pipeline.Pipeline, prompt: []const u32, out: []u32, chunk: usize) pipeline.Error!void {
    var token: u32 = 0;
    var at: usize = 0;
    while (at < prompt.len) {
        const n = @min(chunk, prompt.len - at);
        token = try p.stepRows(prompt[at..][0..n]);
        at += n;
    }
    for (out) |*o| {
        o.* = token;
        token = try p.step(token);
    }
}

/// The host-side stage on the CPU. At level B it lands the stamped row late and out of order, as a GPU may.
pub const CpuPeer = struct {
    stage: *Stage,
    late: bool = false,
    thread: ?std.Thread = null,

    pub fn peer(self: *CpuPeer) pipeline.Peer {
        return .{ .ptr = self, .step = stepFn, .finish = finishFn };
    }

    fn stepFn(ptr: *anyopaque, seq: u32, slots: []const u8, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const self: *CpuPeer = @ptrCast(@alignCast(ptr));
        var plain: [max_rows * hidden_bytes]u8 = undefined;
        const n = try self.stage.runSlots(slots, rows, input, &plain);
        if (self.stage.head) {
            @memcpy(out[0..n], plain[0..n]);
            return n;
        }
        if (level == .a or !self.late) {
            wire.packStampedRows(out, plain[0..n], rows, hidden_bytes, seq);
            return n;
        }
        const Late = struct {
            fn go(dst: []u8, all_rows: [max_rows * hidden_bytes]u8, count: u32, stamp: u32) void {
                var all: [max_rows * wire.stampedLen(hidden_bytes)]u8 = undefined;
                wire.packStampedRows(&all, &all_rows, count, hidden_bytes, stamp);
                var g = wire.stampedRowsLen(count, hidden_bytes) / wire.group_bytes;
                while (g > 0) {
                    g -= 1;
                    device.pause(20_000);
                    @memcpy(dst[g * wire.group_bytes ..][0..wire.group_bytes], all[g * wire.group_bytes ..][0..wire.group_bytes]);
                }
            }
        };
        self.thread = try std.Thread.spawn(.{}, Late.go, .{ out, plain, rows, seq });
        return n;
    }

    fn finishFn(ptr: *anyopaque, _: u32) anyerror!void {
        const self: *CpuPeer = @ptrCast(@alignCast(ptr));
        if (self.thread) |t| t.join();
        self.thread = null;
    }
};

const mock = @import("mock.zig");
const streams = @import("streams.zig");

/// The device stage of a sequence, held back `ns` before it runs: a stand-in for a layer range that takes that long.
const SlowStage = struct {
    inner: *Stage,
    ns: u64,

    fn external(self: *SlowStage) pipeline.ExternalStage {
        return .{ .ptr = self, .run = runFn };
    }

    fn runFn(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const self: *SlowStage = @ptrCast(@alignCast(ptr));
        device.pause(self.ns);
        return Stage.runFn(self.inner, io);
    }
};

/// The host stage of a sequence, held back `ns` per step.
const SlowPeer = struct {
    inner: *CpuPeer,
    ns: u64,

    fn peer(self: *SlowPeer) pipeline.Peer {
        return .{ .ptr = self, .step = stepFn, .finish = finishFn };
    }

    fn stepFn(ptr: *anyopaque, seq: u32, slots: []const u8, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const self: *SlowPeer = @ptrCast(@alignCast(ptr));
        device.pause(self.ns);
        return CpuPeer.stepFn(self.inner, seq, slots, rows, input, out, level);
    }

    fn finishFn(ptr: *anyopaque, seq: u32) anyerror!void {
        return CpuPeer.finishFn(@as(*SlowPeer, @ptrCast(@alignCast(ptr))).inner, seq);
    }
};

/// A split of the toy model for `n` sequences: the stages of each side, one pair a sequence, over one mock device.
const Rig = struct {
    params: *Params,
    mock: *mock.Mock,
    window: device.Window,
    ext: [streams.max_streams]Stage = undefined,
    host: [streams.max_streams]Stage = undefined,
    cpu: [streams.max_streams]CpuPeer = undefined,
    slow_ext: [streams.max_streams]SlowStage = undefined,
    slow_host: [streams.max_streams]SlowPeer = undefined,
    scratch: [streams.max_streams][max_rows * hidden_bytes]u8 = undefined,
    pipes: [streams.max_streams]pipeline.Pipeline = undefined,
    n: usize,

    fn create(order: pipeline.Order, level: pipeline.Level, late: bool, k: usize, n: usize, delay_ns: u64) !*Rig {
        const gpa = std.testing.allocator;
        const r = try gpa.create(Rig);
        errdefer gpa.destroy(r);
        r.n = n;
        r.params = try gpa.create(Params);
        errdefer gpa.destroy(r.params);
        r.params.* = .init(42);
        r.mock = try mock.Mock.create(gpa, .{ .wait_delay_ns = 1_000, .write_delay_ns = 1_000 });
        const dev = r.mock.external();
        errdefer dev.deinit();
        const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = hidden_bytes, .max_rows = max_rows, .timeout_ns = 5_000_000_000 };
        r.window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, n));
        for (0..n) |i| {
            r.ext[i] = switch (order) {
                .external_first => .{ .params = r.params, .first = 0, .last = k, .embed = true, .head = false },
                .metal_first => .{ .params = r.params, .first = k, .last = layers, .embed = false, .head = true },
                .external_only => .{ .params = r.params, .first = 0, .last = layers, .embed = true, .head = true },
            };
            r.host[i] = switch (order) {
                .external_first => .{ .params = r.params, .first = k, .last = layers, .embed = false, .head = true },
                else => .{ .params = r.params, .first = 0, .last = k, .embed = true, .head = false },
            };
            r.cpu[i] = .{ .stage = &r.host[i], .late = late };
            r.slow_ext[i] = .{ .inner = &r.ext[i], .ns = delay_ns };
            r.slow_host[i] = .{ .inner = &r.cpu[i], .ns = delay_ns };
            var c = cfg;
            c.window_offset = i * 2 * pipeline.Pipeline.slotBytes(cfg);
            const peer: ?pipeline.Peer = if (order == .external_only) null else r.slow_host[i].peer();
            r.pipes[i] = .init(dev, r.window, c, r.slow_ext[i].external(), peer, &r.scratch[i]);
        }
        return r;
    }

    fn destroy(r: *Rig) void {
        const dev = r.mock.external();
        dev.sync(1_000_000_000) catch {};
        dev.freeWindow(r.window);
        dev.deinit();
        std.testing.allocator.destroy(r.params);
        std.testing.allocator.destroy(r);
    }
};

fn runSplit(order: pipeline.Order, level: pipeline.Level, late: bool, k: usize, chunk: usize, prompt: []const u32, out: []u32) !void {
    const r = try Rig.create(order, level, late, k, 1, 0);
    defer r.destroy();
    try generateChunked(&r.pipes[0], prompt, out, chunk);
}

test "every split of the toy model generates the unsplit reference's tokens" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 5, 17, 3, 90 };
    var want: [24]u32 = undefined;
    try reference(params, &prompt, &want);
    // the draw has to move around, or the check would pass on a constant stream
    var seen: [vocab]bool = @splat(false);
    for (want) |t| seen[t] = true;
    var distinct: usize = 0;
    for (seen) |b| distinct += @intFromBool(b);
    try std.testing.expect(distinct > 3);
    for ([_]pipeline.Order{ .external_first, .metal_first }) |order| {
        for ([_]usize{ 1, 5, 6, 11 }) |k| {
            var got: [24]u32 = undefined;
            try runSplit(order, .a, false, k, 1, &prompt, &got);
            try std.testing.expectEqualSlices(u32, &want, &got);
        }
    }
    var got: [24]u32 = undefined;
    try runSplit(.external_only, .a, false, layers, 1, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
}

test "level B accepts a hidden row whose flag outran its data" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 8, 8, 61 };
    var want: [12]u32 = undefined;
    try reference(params, &prompt, &want);
    var got: [12]u32 = undefined;
    try runSplit(.metal_first, .b, true, 4, 1, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
}

test "a prompt in chunks gives the tokens of the prompt a position at a time, in every order and level" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 5, 17, 3, 90, 12, 44, 8, 8, 61, 2, 70, 33, 19 };
    var want: [10]u32 = undefined;
    try reference(params, &prompt, &want);
    for ([_]usize{ 2, 5, 13, 16 }) |chunk| {
        for ([_]pipeline.Order{ .external_first, .metal_first, .external_only }) |order| {
            for ([_]pipeline.Level{ .a, .b }) |level| {
                if (order != .metal_first and level == .b) continue;
                var got: [10]u32 = undefined;
                try runSplit(order, level, level == .b, if (order == .external_only) layers else 7, chunk, &prompt, &got);
                try std.testing.expectEqualSlices(u32, &want, &got);
            }
        }
    }
}

test "several sequences share a device and each gets its own reference's tokens" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompts = [3][]const u32{ &.{ 5, 17, 3, 90 }, &.{ 8, 8, 61, 2, 9, 40, 11 }, &.{71} };
    var want: [3][14]u32 = undefined;
    for (prompts, &want) |pr, *w| try reference(params, pr, w);
    for ([_]pipeline.Order{ .external_first, .metal_first }) |order| {
        for ([_]pipeline.Level{ .a, .b }) |level| {
            if (order == .external_first and level == .b) continue;
            for ([_]usize{ 1, 4 }) |chunk| {
                const r = try Rig.create(order, level, level == .b, 6, 3, 0);
                defer r.destroy();
                var got: [3][14]u32 = undefined;
                var works: [3]streams.Work = undefined;
                var pipes: [3]*pipeline.Pipeline = undefined;
                for (0..3) |i| {
                    works[i] = .{ .prompt = prompts[i], .out = &got[i] };
                    pipes[i] = &r.pipes[i];
                }
                try streams.generate(&pipes, &works, chunk);
                for (want, got) |w, g| try std.testing.expectEqualSlices(u32, &w, &g);
            }
        }
    }
}

test "decode batches: sequences of different lengths in lanes of up to eight each get their own reference's tokens" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    // eight prompts of different lengths and different amounts to generate: the batches shrink as sequences finish
    const prompts = [8][]const u32{ &.{ 5, 17, 3, 90 }, &.{ 8, 8, 61, 2, 9 }, &.{71}, &.{ 12, 44 }, &.{ 33, 19, 7 }, &.{ 60, 1, 2, 3, 4, 5 }, &.{ 40, 41 }, &.{ 90, 91, 92 } };
    const lens = [8]usize{ 9, 14, 3, 12, 1, 7, 11, 5 };
    var want: [8][14]u32 = undefined;
    for (prompts, lens, &want) |pr, n, *w| try reference(params, pr, w[0..n]);
    // lane layouts: one lane of eight, two of four, three of (3, 3, 2); every order and level
    const layouts = [_][]const usize{ &.{8}, &.{ 4, 4 }, &.{ 3, 3, 2 } };
    for (layouts) |layout| {
        for ([_]pipeline.Order{ .external_first, .metal_first, .external_only }) |order| {
            for ([_]pipeline.Level{ .a, .b }) |level| {
                if (order != .metal_first and level == .b) continue;
                const r = try Rig.create(order, level, level == .b, if (order == .external_only) layers else 6, layout.len, 0);
                defer r.destroy();
                var got: [8][14]u32 = undefined;
                var works: [8]streams.Work = undefined;
                for (&works, prompts, lens, &got) |*w, pr, n, *g| w.* = .{ .prompt = pr, .out = g[0..n] };
                var lanes: [3]streams.Lane = undefined;
                var at: usize = 0;
                for (layout, 0..) |n, i| {
                    lanes[i] = .{ .pipe = &r.pipes[i], .works = works[at..][0..n] };
                    at += n;
                }
                try streams.generateLanes(lanes[0..layout.len], 3);
                for (lens, want, got) |n, w, g| try std.testing.expectEqualSlices(u32, w[0..n], g[0..n]);
            }
        }
    }
}

test "while the host works on one sequence the device works on another" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 5, 17 };
    var want: [8]u32 = undefined;
    try reference(params, &prompt, &want);
    // each side takes 3 ms a step: one sequence at a time costs both, two sequences cost about the slower one each
    const delay: u64 = 3_000_000;
    var elapsed: [2]u64 = undefined;
    for ([_]usize{ 1, 2 }, &elapsed) |n, *e| {
        const r = try Rig.create(.metal_first, .a, false, 6, n, delay);
        defer r.destroy();
        var got: [2][8]u32 = undefined;
        var works: [2]streams.Work = undefined;
        var pipes: [2]*pipeline.Pipeline = undefined;
        for (0..n) |i| {
            works[i] = .{ .prompt = &prompt, .out = &got[i] };
            pipes[i] = &r.pipes[i];
        }
        const t0 = device.nowNs();
        try streams.generate(pipes[0..n], works[0..n], 1);
        e.* = device.nowNs() - t0;
        for (got[0..n]) |g| try std.testing.expectEqualSlices(u32, &want, &g);
    }
    // two sequences in less than 1.8 times one: serial would be twice (the margin is for a loaded machine)
    try std.testing.expect(elapsed[1] * 5 < elapsed[0] * 9);
}

test "a cut link or a stuck device is a named error inside the timeout and the next run is valid" {
    const params = try std.testing.allocator.create(Params);
    defer std.testing.allocator.destroy(params);
    params.* = .init(42);
    const prompt = [_]u32{ 7, 8 };
    var want: [6]u32 = undefined;
    try reference(params, &prompt, &want);
    // a device whose writes take far longer than the host will wait
    const slow = try mock.Mock.create(std.testing.allocator, .{ .write_delay_ns = 400_000_000 });
    const dev = slow.external();
    defer dev.deinit();
    const cfg: pipeline.Config = .{ .order = .external_first, .hidden_bytes = hidden_bytes, .timeout_ns = 50_000_000 };
    const window = try dev.allocWindow(pipeline.Pipeline.windowBytes(cfg, 1));
    defer dev.freeWindow(window);
    var ext: Stage = .{ .params = params, .first = 0, .last = 5, .embed = true, .head = false };
    var host: Stage = .{ .params = params, .first = 5, .last = layers, .embed = false, .head = true };
    var cpu: CpuPeer = .{ .stage = &host };
    var scratch: [hidden_bytes]u8 = undefined;
    var p: pipeline.Pipeline = .init(dev, window, cfg, ext.external(), cpu.peer(), &scratch);
    const t0 = device.nowNs();
    try std.testing.expectError(error.Timeout, p.step(7));
    try std.testing.expect(device.nowNs() - t0 < 2_000_000_000);
    // pulled cable: the stream refuses new work with a named error
    slow.kill();
    p.reset();
    try std.testing.expectError(error.DeviceLost, p.step(8));
    // the next request on a healthy device starts clean and matches the reference
    var got: [6]u32 = undefined;
    try runSplit(.external_first, .a, false, 5, 1, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
}
