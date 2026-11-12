//! The toy model's stage on the Mac's GPU through `metal.Relay`: integer kernels, so a split equals the unsplit run.
const std = @import("std");
const metal = @import("metal");
const split = @import("split");

const toy = split.toy;
const pipeline = split.pipeline;
const wire = split.wire;

const H = toy.hidden;
const source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\constant int H = 32; constant int V = 97; constant int XL = 1 << 20; constant int SL = 1 << 28;
    \\kernel void toy_embed(device const int* embed [[buffer(0)]], device const uint* tok [[buffer(1)]],
    \\                      device int* x [[buffer(2)]], uint h [[thread_position_in_grid]]) {
    \\    x[h] = embed[tok[0] * H + h] * 1024;
    \\}
    \\kernel void toy_load(device const int* in [[buffer(0)]], device int* x [[buffer(1)]], uint h [[thread_position_in_grid]]) {
    \\    x[h] = in[h];
    \\}
    \\kernel void toy_layer(device const int* w [[buffer(0)]], device int* s [[buffer(1)]], device int* x [[buffer(2)]],
    \\                      uint h [[thread_position_in_threadgroup]]) {
    \\    int acc = 0;
    \\    for (int j = 0; j < H; j++) acc += w[h * H + j] * x[j];
    \\    threadgroup_barrier(mem_flags::mem_device);
    \\    int a = (acc + s[h]) >> 3;
    \\    s[h] = clamp(s[h] + acc, -SL, SL);
    \\    x[h] = clamp(x[h] + a, -XL, XL);
    \\}
    \\kernel void toy_head(device const int* head [[buffer(0)]], device const int* x [[buffer(1)]],
    \\                     device int* logits [[buffer(2)]], device uint* tok [[buffer(3)]], uint v [[thread_position_in_threadgroup]]) {
    \\    if (v < V) {
    \\        int acc = 0;
    \\        for (int j = 0; j < H; j++) acc += head[v * H + j] * x[j];
    \\        logits[v] = acc;
    \\    }
    \\    threadgroup_barrier(mem_flags::mem_device);
    \\    if (v == 0) {
    \\        uint best = 0; int bl = logits[0];
    \\        for (uint i = 1; i < V; i++) if (logits[i] > bl) { bl = logits[i]; best = i; }
    \\        tok[0] = best;
    \\    }
    \\}
    \\kernel void toy_pack(device const uint* x [[buffer(0)]], device uint4* out [[buffer(1)]], constant uint& stamp [[buffer(2)]],
    \\                     uint g [[thread_position_in_grid]]) {
    \\    uint a = 3 * g, b = 3 * g + 1, c = 3 * g + 2;
    \\    out[g] = uint4(a < H ? x[a] : 0, b < H ? x[b] : 0, c < H ? x[c] : 0, stamp);
    \\}
;

pub const Error = anyerror;

pub const MetalPeer = struct {
    device: metal.Device,
    queue: metal.Queue,
    relay: metal.Relay,
    library: metal.Library,
    window: metal.HostWindow,
    stage: *toy.Stage,
    layer_pl: metal.Pipeline,
    embed_pl: metal.Pipeline,
    load_pl: metal.Pipeline,
    head_pl: metal.Pipeline,
    pack_pl: metal.Pipeline,
    embed: metal.Buffer,
    weights: metal.Buffer,
    head: metal.Buffer,
    state: metal.Buffer,
    x: metal.Buffer,
    logits: metal.Buffer,
    tok_in: metal.Buffer,
    tok_out: metal.Buffer,
    stamp: metal.Buffer,
    // what the next encode reads: where in the window the input row is (-1: none) and where the packed row goes
    input_offset: ?usize = null,
    output_offset: usize = 0,
    pending: ?metal.Ticket = null,

    /// `window_ptr` and `window_len` are the memory the pipeline's slots live in (the external device's window).
    pub fn init(stage: *toy.Stage, window_ptr: [*]u8, window_len: usize) Error!*MetalPeer {
        const gpa = std.heap.c_allocator;
        const dev = try metal.Device.init();
        const self = try gpa.create(MetalPeer);
        errdefer gpa.destroy(self);
        const opts = metal.ResourceOptions.shared;
        const lib = try metal.Library.fromSource(dev, source, .{});
        const params = stage.params;
        self.* = .{
            .device = dev,
            .queue = try dev.queue(),
            .relay = undefined,
            .library = lib,
            .window = try metal.HostWindow.wrap(dev, window_ptr, window_len),
            .stage = stage,
            .layer_pl = try metal.Pipeline.init(dev, lib, "toy_layer", false),
            .embed_pl = try metal.Pipeline.init(dev, lib, "toy_embed", false),
            .load_pl = try metal.Pipeline.init(dev, lib, "toy_load", false),
            .head_pl = try metal.Pipeline.init(dev, lib, "toy_head", false),
            .pack_pl = try metal.Pipeline.init(dev, lib, "toy_pack", false),
            .embed = try dev.buffer(@sizeOf(@TypeOf(params.embed)), opts),
            .weights = try dev.buffer(@sizeOf(@TypeOf(params.w)), opts),
            .head = try dev.buffer(@sizeOf(@TypeOf(params.head)), opts),
            .state = try dev.buffer(@sizeOf(@TypeOf(stage.state)), opts),
            .x = try dev.buffer(H * 4, opts),
            .logits = try dev.buffer(toy.vocab * 4, opts),
            .tok_in = try dev.buffer(16, opts),
            .tok_out = try dev.buffer(16, opts),
            .stamp = try dev.buffer(16, opts),
        };
        self.relay = try metal.Relay.init(dev, self.queue);
        @memcpy(self.embed.contents()[0..self.embed.length()], std.mem.asBytes(&params.embed));
        @memcpy(self.weights.contents()[0..self.weights.length()], std.mem.asBytes(&params.w));
        @memcpy(self.head.contents()[0..self.head.length()], std.mem.asBytes(&params.head));
        @memset(self.state.contents()[0..self.state.length()], 0);
        return self;
    }

    pub fn deinit(self: *MetalPeer) void {
        self.relay.deinit();
        self.window.deinit();
        inline for (.{ "embed", "weights", "head", "state", "x", "logits", "tok_in", "tok_out", "stamp" }) |f| @field(self, f).deinit();
        inline for (.{ "layer_pl", "embed_pl", "load_pl", "head_pl", "pack_pl" }) |f| @field(self, f).deinit();
        self.library.deinit();
        self.queue.deinit();
        self.device.deinit();
        std.heap.c_allocator.destroy(self);
    }

    pub fn peer(self: *MetalPeer) pipeline.Peer {
        return .{ .ptr = self, .step = stepFn, .finish = finishFn };
    }

    fn encode(ctx: *anyopaque, cb: metal.CommandBuffer) void {
        const self: *MetalPeer = @ptrCast(@alignCast(ctx));
        const st = self.stage;
        const enc = cb.compute(.serial);
        defer enc.end();
        const one = metal.Size.of(1, 1, 1);
        const row = metal.Size.of(H, 1, 1);
        if (st.embed) {
            enc.setPipeline(self.embed_pl);
            enc.setBuffer(self.embed, 0, 0);
            enc.setBuffer(self.tok_in, 0, 1);
            enc.setBuffer(self.x, 0, 2);
            enc.dispatchThreads(row, row);
        } else if (self.input_offset) |off| {
            enc.setPipeline(self.load_pl);
            enc.setBuffer(self.window.buffer, off, 0);
            enc.setBuffer(self.x, 0, 1);
            enc.dispatchThreads(row, row);
        }
        for (st.first..st.last) |l| {
            enc.setPipeline(self.layer_pl);
            enc.setBuffer(self.weights, l * H * H * 4, 0);
            enc.setBuffer(self.state, l * H * 4, 1);
            enc.setBuffer(self.x, 0, 2);
            enc.dispatchGroups(one, row);
        }
        if (st.head) {
            enc.setPipeline(self.head_pl);
            enc.setBuffer(self.head, 0, 0);
            enc.setBuffer(self.x, 0, 1);
            enc.setBuffer(self.logits, 0, 2);
            enc.setBuffer(self.tok_out, 0, 3);
            enc.dispatchGroups(one, metal.Size.of(128, 1, 1));
        } else {
            enc.setPipeline(self.pack_pl);
            enc.setBuffer(self.x, 0, 0);
            enc.setBuffer(self.window.buffer, self.output_offset, 1);
            enc.setBuffer(self.stamp, 0, 2);
            const groups = wire.stampedLen(toy.hidden_bytes) / wire.group_bytes;
            enc.dispatchThreads(metal.Size.of(groups, 1, 1), metal.Size.of(groups, 1, 1));
        }
    }

    fn offsetOf(self: *MetalPeer, s: []const u8) Error!usize {
        const base = @intFromPtr(self.window.ptr);
        const at = @intFromPtr(s.ptr);
        if (at < base or at + s.len > base + self.window.len) return error.OutsideWindow;
        return at - base;
    }

    fn stepFn(ptr: *anyopaque, seq: u32, slots: []const u8, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        // decode rows of one sequence only
        if (rows != 1 or slots.len != 1) return error.BadStep;
        const self: *MetalPeer = @ptrCast(@alignCast(ptr));
        const st = self.stage;
        self.input_offset = null;
        if (st.embed) {
            std.mem.writeInt(u32, self.tok_in.contents()[0..4], std.mem.readInt(u32, input[0..4], .little), .little);
        } else {
            self.input_offset = try self.offsetOf(input);
        }
        if (!st.head) {
            self.output_offset = try self.offsetOf(out);
            std.mem.writeInt(u32, self.stamp.contents()[0..4], seq, .little);
        }
        const ticket = try self.relay.submit(self, encode);
        self.relay.open(ticket);
        self.pending = ticket;
        if (st.head or level == .a) try finishFn(ptr, seq);
        if (st.head) {
            @memcpy(out[0..4], self.tok_out.contents()[0..4]);
            return 4;
        }
        return toy.hidden_bytes;
    }

    fn finishFn(ptr: *anyopaque, _: u32) anyerror!void {
        const self: *MetalPeer = @ptrCast(@alignCast(ptr));
        const t = self.pending orelse return;
        self.pending = null;
        if (!self.relay.waitDone(t, 5_000)) return error.Timeout;
        if (self.relay.failed(t) != null) return error.BadStage;
    }
};

/// The toy model's parameters (seed 42), built on the heap.
pub const Fixture = struct {
    params: *toy.Params,

    pub fn init(gpa: std.mem.Allocator) !Fixture {
        const p = try gpa.create(toy.Params);
        p.* = .init(42);
        return .{ .params = p };
    }

    pub fn deinit(f: Fixture, gpa: std.mem.Allocator) void {
        gpa.destroy(f.params);
    }
};

/// Run the toy model over the mock device and Metal, `k` layers on one side and the rest on the other.
pub fn runSplit(gpa: std.mem.Allocator, f: Fixture, order: pipeline.Order, level: pipeline.Level, k: usize, prompt: []const u32, out: []u32) !void {
    const pool = metal.objc.Pool.push();
    defer pool.pop();
    const m = try split.mock.Mock.create(gpa, .{ .wait_delay_ns = 1_000, .write_delay_ns = 1_000 });
    const dev = m.external();
    defer dev.deinit();
    const cfg: pipeline.Config = .{ .order = order, .level = level, .hidden_bytes = toy.hidden_bytes };
    const window = try dev.allocWindow(pipeline.Pipeline.slotBytes(cfg) * 2);
    defer dev.freeWindow(window);
    var ext: toy.Stage = switch (order) {
        .external_first => .{ .params = f.params, .first = 0, .last = k, .embed = true, .head = false },
        .metal_first => .{ .params = f.params, .first = k, .last = toy.layers, .embed = false, .head = true },
        .external_only => .{ .params = f.params, .first = 0, .last = toy.layers, .embed = true, .head = true },
    };
    var host: toy.Stage = switch (order) {
        .external_first => .{ .params = f.params, .first = k, .last = toy.layers, .embed = false, .head = true },
        else => .{ .params = f.params, .first = 0, .last = k, .embed = true, .head = false },
    };
    const mp = try MetalPeer.init(&host, window.ptr, window.len);
    var scratch: [toy.hidden_bytes]u8 = undefined;
    defer mp.deinit();
    var p: pipeline.Pipeline = .init(dev, window, cfg, ext.external(), mp.peer(), &scratch);
    try toy.generate(&p, prompt, out);
    try dev.sync(1_000_000_000);
}

test "metal on either side of the split reproduces the unsplit tokens" {
    _ = metal.Device.init() catch return error.SkipZigTest;
    const f = try Fixture.init(std.testing.allocator);
    defer f.deinit(std.testing.allocator);
    const prompt = [_]u32{ 5, 17, 3, 90 };
    var want: [24]u32 = undefined;
    try toy.reference(f.params, &prompt, &want);
    for ([_]pipeline.Order{ .external_first, .metal_first }) |order| {
        for ([_]usize{ 1, 6, 11 }) |k| {
            var got: [24]u32 = undefined;
            try runSplit(std.testing.allocator, f, order, .a, k, &prompt, &got);
            try std.testing.expectEqualSlices(u32, &want, &got);
        }
    }
}

test "metal's stamped hidden row is accepted with the flag stored before the kernel finishes" {
    _ = metal.Device.init() catch return error.SkipZigTest;
    const f = try Fixture.init(std.testing.allocator);
    defer f.deinit(std.testing.allocator);
    const prompt = [_]u32{ 8, 8, 61, 2 };
    var want: [40]u32 = undefined;
    try toy.reference(f.params, &prompt, &want);
    var got: [40]u32 = undefined;
    try runSplit(std.testing.allocator, f, .metal_first, .b, 4, &prompt, &got);
    try std.testing.expectEqualSlices(u32, &want, &got);
}

test "all layers on metal alone equal the unsplit reference" {
    _ = metal.Device.init() catch return error.SkipZigTest;
    const f = try Fixture.init(std.testing.allocator);
    defer f.deinit(std.testing.allocator);
    const pool = metal.objc.Pool.push();
    defer pool.pop();
    const prompt = [_]u32{ 11, 2, 77 };
    var want: [16]u32 = undefined;
    try toy.reference(f.params, &prompt, &want);
    var all: toy.Stage = .{ .params = f.params, .first = 0, .last = toy.layers, .embed = true, .head = true };
    var page: [16 * 1024]u8 align(16 * 1024) = undefined;
    const mp = try MetalPeer.init(&all, &page, page.len);
    defer mp.deinit();
    const pe = mp.peer();
    var out: [4]u8 = undefined;
    var token: u32 = 0;
    var got: [16]u32 = undefined;
    var b: [4]u8 = undefined;
    var seq: u32 = 0;
    for (prompt) |t| {
        std.mem.writeInt(u32, &b, t, .little);
        seq += 1;
        _ = try pe.step(pe.ptr, seq, &.{0}, 1, &b, &out, .a);
        token = std.mem.readInt(u32, &out, .little);
    }
    for (&got) |*g| {
        g.* = token;
        std.mem.writeInt(u32, &b, token, .little);
        seq += 1;
        _ = try pe.step(pe.ptr, seq, &.{0}, 1, &b, &out, .a);
        token = std.mem.readInt(u32, &out, .little);
    }
    try std.testing.expectEqualSlices(u32, &want, &got);
}
