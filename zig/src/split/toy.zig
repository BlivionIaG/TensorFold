//! A tiny dense model in exact integer arithmetic: a split is compared with the unsplit run bit for bit on any backend.
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
    state: [layers][hidden]i32 = @splat(@splat(0)),

    pub fn external(self: *Stage) pipeline.ExternalStage {
        return .{ .ptr = self, .run = runFn };
    }

    /// The device side on the CPU: refuse an aborted step, validate a stamped row, run, leave the result in `out`.
    fn runFn(ptr: *anyopaque, io: pipeline.StepIo) anyerror!void {
        const self: *Stage = @ptrCast(@alignCast(ptr));
        const h = io.in.header().*;
        if (h.status != 0 or h.kind == @backingInt(wire.Kind.abort)) return error.PeerLost;
        const input = if (io.stamped) blk: {
            const row = io.scratch[0..io.hidden_bytes];
            _ = try wire.readStamped(row, io.in.data(), io.seq, io.retries);
            break :blk row;
        } else io.in.data()[0..h.bytes];
        _ = try self.run(input, io.out.data());
    }

    /// `input` is a token when the stage embeds, else a hidden row of i32; the output is a token when it has the head.
    pub fn run(self: *Stage, input: []const u8, out: []u8) !usize {
        var x: [hidden]i32 = undefined;
        if (self.embed) {
            const t = std.mem.readInt(u32, input[0..4], .little);
            if (t >= vocab) return error.BadToken;
            for (&x, self.params.embed[t]) |*d, e| d.* = e * 1024;
        } else {
            if (input.len != hidden_bytes) return error.BadHidden;
            for (&x, 0..) |*d, i| d.* = std.mem.readInt(i32, input[i * 4 ..][0..4], .little);
        }
        for (self.first..self.last) |l| layer(self, l, &x);
        if (self.head) {
            std.mem.writeInt(u32, out[0..4], draw(self.params, &x), .little);
            return 4;
        }
        for (x, 0..) |v, i| std.mem.writeInt(i32, out[i * 4 ..][0..4], v, .little);
        return hidden_bytes;
    }

    /// y = W_l x; a = (y + s_l) >> 3; s_l = clamp(s_l + y); x = clamp(x + a).
    fn layer(self: *Stage, l: usize, x: *[hidden]i32) void {
        const s = &self.state[l];
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
    var token: u32 = 0;
    for (prompt) |t| token = try p.step(t);
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

    fn stepFn(ptr: *anyopaque, seq: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const self: *CpuPeer = @ptrCast(@alignCast(ptr));
        var plain: [hidden_bytes]u8 = undefined;
        const n = try self.stage.run(input, &plain);
        if (self.stage.head) {
            @memcpy(out[0..n], plain[0..n]);
            return n;
        }
        if (level == .a or !self.late) {
            wire.packStamped(out, plain[0..n], seq);
            return n;
        }
        const Late = struct {
            fn go(dst: []u8, row: [hidden_bytes]u8, stamp: u32) void {
                var all: [wire.stampedLen(hidden_bytes)]u8 = undefined;
                wire.packStamped(&all, &row, stamp);
                var g = all.len / wire.group_bytes;
                while (g > 0) {
                    g -= 1;
                    device.pause(20_000);
                    @memcpy(dst[g * wire.group_bytes ..][0..wire.group_bytes], all[g * wire.group_bytes ..][0..wire.group_bytes]);
                }
            }
        };
        self.thread = try std.Thread.spawn(.{}, Late.go, .{ out, plain, seq });
        return n;
    }

    fn finishFn(ptr: *anyopaque, _: u32) anyerror!void {
        const self: *CpuPeer = @ptrCast(@alignCast(ptr));
        if (self.thread) |t| t.join();
        self.thread = null;
    }
};

const mock = @import("mock.zig");

/// The toy model split at layer `k` between a mock device and the CPU in `order`; `out` gets the generated tokens.
pub fn runSplit(gpa: std.mem.Allocator, order: pipeline.Order, level: pipeline.Level, late: bool, k: usize, prompt: []const u32, out: []u32) !void {
    const params = try gpa.create(Params);
    defer gpa.destroy(params);
    params.* = .init(42);
    const m = try mock.Mock.create(gpa, .{ .wait_delay_ns = 1_000, .write_delay_ns = 1_000 });
    const dev = m.external();
    defer dev.deinit();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = hidden_bytes, .timeout_ns = 5_000_000_000 };
    const window = try dev.allocWindow(pipeline.Pipeline.slotBytes(cfg) * 2);
    defer dev.freeWindow(window);
    var ext: Stage = switch (order) {
        .external_first => .{ .params = params, .first = 0, .last = k, .embed = true, .head = false },
        .metal_first => .{ .params = params, .first = k, .last = layers, .embed = false, .head = true },
        .external_only => .{ .params = params, .first = 0, .last = layers, .embed = true, .head = true },
    };
    var host: Stage = switch (order) {
        .external_first => .{ .params = params, .first = k, .last = layers, .embed = false, .head = true },
        else => .{ .params = params, .first = 0, .last = k, .embed = true, .head = false },
    };
    var cpu: CpuPeer = .{ .stage = &host, .late = late };
    var scratch: [hidden_bytes]u8 = undefined;
    var p: pipeline.Pipeline = .init(dev, window, cfg, ext.external(), if (order == .external_only) null else cpu.peer(), &scratch);
    try generate(&p, prompt, out);
    try dev.sync(1_000_000_000);
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
            try runSplit(std.testing.allocator, order, .a, false, k, &prompt, &got);
            try std.testing.expectEqualSlices(u32, &want, &got);
        }
    }
    var got: [24]u32 = undefined;
    try runSplit(std.testing.allocator, .external_only, .a, false, layers, &prompt, &got);
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
    try runSplit(std.testing.allocator, .metal_first, .b, true, 4, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
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
    const window = try dev.allocWindow(pipeline.Pipeline.slotBytes(cfg) * 2);
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
    try runSplit(std.testing.allocator, .external_first, .a, false, 5, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
}
