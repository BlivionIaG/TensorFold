const std = @import("std");
const service_mod = @import("expert_service.zig");

const Service = service_mod.Service;
const Tier = service_mod.Tier;
const Window = service_mod.Window;
const Ports = service_mod.Ports;
const shape: service_mod.Shape = .{ .layers = 2, .experts = 4, .picks = 3, .trace_rows = 2, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
const blob = 160;
const win_dev: u64 = 0x4000_0000;

const Nowhere = struct {
    fn line(_: *anyopaque, slot: usize) u64 {
        return 0x1000_0000 + slot * blob;
    }
    fn copy(_: *anyopaque, _: usize, _: usize, _: usize) error{CopyFailed}!void {}
    fn wait(_: *anyopaque) error{CopyFailed}!void {}
};

var anchor: u8 = 0;

fn ports() Ports {
    return .{ .ctx = &anchor, .line_addr = Nowhere.line, .scratch_base = 0x8000_0000, .scratch_copy = Nowhere.copy, .scratch_wait = Nowhere.wait };
}

fn parts(gpa: std.mem.Allocator, seed: u8) ![6][]u8 {
    var out: [6][]u8 = undefined;
    for (&out, shape.part_bytes, 0..) |*t, b, i| {
        t.* = try gpa.alloc(u8, b * (shape.experts + 1));
        for (t.*, 0..) |*x, j| x.* = @truncate(j *% 13 +% i *% 5 +% seed);
    }
    return out;
}

/// What the device's stream does for one stage: the picks into the trace, the serial into `req`, then a wait on `ack`.
fn card(s: *Service, n: u32, picks: []const i32) void {
    @memcpy(s.tier.win.traceWords()[0..picks.len], picks);
    @atomicStore(u32, s.reqWord(), n, .release);
    while (@atomicLoad(u32, s.ackWord(), .acquire) -% n >= 0x8000_0000) std.atomic.spinLoopHint();
}

fn open(gpa: std.mem.Allocator, memory: *[]align(64) u8, ts: *[6][]u8) !Service {
    memory.* = try gpa.alignedAlloc(u8, .@"64", Window.bytesFor(shape, Window.leastSlots(shape)));
    var tier = try Tier.init(gpa, shape, memory.*, win_dev, 0);
    ts.* = try parts(gpa, 3);
    try tier.addLayer(0, ts.*);
    try tier.addLayer(1, ts.*);
    return Service.init(tier);
}

test "the device's requests and the worker's answers pair up over more stages than the ring holds" {
    const gpa = std.testing.allocator;
    var memory: []align(64) u8 = undefined;
    var ts: [6][]u8 = undefined;
    var s = try open(gpa, &memory, &ts);
    defer {
        s.deinit();
        gpa.free(memory);
        for (ts) |t| gpa.free(t);
    }
    try std.testing.expectError(error.BadShape, s.post(.{ .layer = 0, .rows = 1, .prefill = false }));
    try s.start(ports());
    try std.testing.expectError(error.BadShape, s.post(.{ .layer = 0, .rows = 0, .prefill = false }));
    try std.testing.expectError(error.BadShape, s.post(.{ .layer = 0, .rows = 3, .prefill = false }));
    s.beginRound();
    const stages: u32 = service_mod.ring + 500;
    for (0..stages) |i| {
        const layer: u32 = @intCast(i & 1);
        const expert: i32 = @intCast(i % 4);
        const n = try s.post(.{ .layer = layer, .rows = 1, .prefill = false });
        try std.testing.expectEqual(@as(u32, @intCast(i + 1)), n);
        card(&s, n, &.{ expert, 4, 4 });
        // the answer is in: the table row of this layer names a window slot holding the expert's bytes
        const entry = s.tier.win.tableWords()[(layer * 5 + @as(u32, @intCast(expert))) * 2];
        try std.testing.expect(s.tier.win.contains(entry));
        const got = s.tier.win.bytes[entry - win_dev ..][0..blob];
        var want: [blob]u8 = undefined;
        var at: usize = 0;
        for (ts, shape.part_bytes) |t, b| {
            @memcpy(want[at..][0..b], t[@as(usize, @intCast(expert)) * b ..][0..b]);
            at += b;
        }
        try std.testing.expectEqualSlices(u8, &want, got);
    }
    try s.endRound(1, 5);
    try std.testing.expectEqual(@as(u64, stages), s.tier.stats.stages);
    try std.testing.expectEqual(@as(u64, 1), s.tier.stats.rounds);
}

test "a failed stage is still answered, and the round that held it is refused" {
    const gpa = std.testing.allocator;
    var memory: []align(64) u8 = undefined;
    var ts: [6][]u8 = undefined;
    var s = try open(gpa, &memory, &ts);
    defer {
        s.deinit();
        gpa.free(memory);
        for (ts) |t| gpa.free(t);
    }
    std.heap.page_allocator.free(s.tier.host[1].?);
    s.tier.host[1] = null;
    try s.start(ports());
    s.beginRound();
    const n = try s.post(.{ .layer = 1, .rows = 1, .prefill = false });
    card(&s, n, &.{ 1, 4, 4 });
    try std.testing.expectError(error.StageFailed, s.endRound(1, 5));
}
