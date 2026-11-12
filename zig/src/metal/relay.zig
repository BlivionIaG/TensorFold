//! Relay: host-to-Metal handoff at command-buffer boundaries, one parked buffer per step gated on a shared event.
const std = @import("std");
const device_mod = @import("device.zig");
const command = @import("command.zig");
const objc = @import("objc.zig");
const sync = @import("sync.zig");
const wake = @import("wake_probe.zig");

const Device = device_mod.Device;
const Queue = command.Queue;
const CommandBuffer = command.CommandBuffer;

/// Steps that may be committed and not yet recycled at once; submit refuses the next one.
pub const max_in_flight = 64;

pub const Error = device_mod.Error || error{ TooManyInFlight, Aborted };

/// One step: the CPU sets the event to `gate` to release the buffer, which sets it to `done` when it finishes.
pub const Ticket = struct { gate: u64, done: u64 };

/// The caller encodes its work into `cb`; called inside submit, so `ctx` may live on the caller's stack.
pub const Encode = *const fn (ctx: *anyopaque, cb: CommandBuffer) void;

const Slot = struct {
    /// The step this slot holds, 0 when empty.
    step: u64 = 0,
    cb: CommandBuffer = undefined,

    fn finished(self: Slot) bool {
        return switch (self.cb.status()) {
            .completed, .@"error" => true,
            else => false,
        };
    }
};

/// One growing event: step t waits for gate 2t-1, signals done 2t; only `abort` and `waitDone` leave the owner thread.
pub const Relay = struct {
    queue: Queue,
    event: sync.SharedEvent,
    steps: u64 = 0,
    aborted: std.atomic.Value(bool) = .init(false),
    /// The last `max_in_flight` steps' command buffers, retained until the slot is reused or deinit.
    slots: [max_in_flight]Slot = @splat(.{}),

    /// A relay over `queue` (borrowed: it must outlive the relay) with its own shared event, at step 0.
    pub fn init(device: Device, queue: Queue) Error!Relay {
        return .{ .queue = queue, .event = try device.sharedEvent() };
    }

    /// Commit the next step now, parked until `open`: wait for the gate, `encode`, signal done.
    pub fn submit(self: *Relay, ctx: *anyopaque, encode: Encode) Error!Ticket {
        if (self.aborted.load(.acquire)) return error.Aborted;
        const t = self.steps + 1;
        const slot = &self.slots[(t - 1) % max_in_flight];
        if (slot.step != 0) {
            if (!slot.finished()) return error.TooManyInFlight;
            slot.cb.release();
            slot.step = 0;
        }
        const ticket: Ticket = .{ .gate = 2 * t - 1, .done = 2 * t };
        const pool = objc.Pool.push();
        defer pool.pop();
        const cb = self.queue.commandBuffer();
        cb.waitFor(self.event, ticket.gate);
        encode(ctx, cb);
        cb.signal(self.event, ticket.done);
        slot.* = .{ .step = t, .cb = cb.retain() };
        cb.commit();
        self.steps = t;
        return ticket;
    }

    /// Release the gate of `t` after the CPU's stores; cumulative, so it also releases earlier tickets in order.
    pub fn open(self: *Relay, t: Ticket) void {
        self.event.set(t.gate);
    }

    /// Block until the buffer of `t` signals done, at most `timeout_ms`; false on timeout or after `abort`.
    pub fn waitDone(self: *Relay, t: Ticket, timeout_ms: u64) bool {
        const reached = self.event.wait(t.done, timeout_ms);
        return reached and !self.aborted.load(.acquire);
    }

    /// The command buffer error text when the buffer of `t` finished in error, else null; valid until the pool drains.
    pub fn failed(self: *Relay, t: Ticket) ?[*:0]const u8 {
        const step = t.done / 2;
        const slot = self.slots[(step - 1) % max_in_flight];
        if (slot.step != step) return null;
        return slot.cb.failure();
    }

    /// Failure path: raise the event to its maximum so parked buffers drain without their data; later calls fail.
    pub fn abort(self: *Relay) void {
        self.aborted.store(true, .release);
        self.event.set(std.math.maxInt(u64));
    }

    /// Abort, wait for every committed buffer to complete, release them and the event. Nothing parks forever.
    pub fn deinit(self: *Relay) void {
        self.abort();
        for (&self.slots) |*slot| {
            if (slot.step == 0) continue;
            slot.cb.wait();
            slot.cb.release();
            slot.step = 0;
        }
        self.event.deinit();
    }
};

const Library = @import("library.zig").Library;
const Pipeline = @import("library.zig").Pipeline;
const Buffer = device_mod.Buffer;
const HostWindow = @import("window.zig").HostWindow;
const types = @import("types.zig");
const testing = std.testing;

const kernels =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void copy_bump(device const uint* in [[buffer(0)]], device uint* out [[buffer(1)]], device atomic_uint* counter [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    \\  out[t] = in[t];
    \\  if (t == 0) atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed);
    \\}
    \\kernel void log_step(device atomic_uint* counter [[buffer(0)]], device uint* log [[buffer(1)]], constant uint& step [[buffer(2)]], uint t [[thread_position_in_grid]]) {
    \\  if (t == 0) log[atomic_fetch_add_explicit(counter, 1u, memory_order_relaxed)] = step;
    \\}
;

/// A device, a queue, the test kernels and a plain shared scratch buffer, or null without a Metal device.
const Rig = struct {
    device: Device,
    queue: Queue,
    library: Library,
    copy: Pipeline,
    log: Pipeline,
    window: HostWindow,
    scratch: Buffer,

    fn init() !Rig {
        const device = Device.init() catch return error.SkipZigTest;
        const queue = try device.queue();
        const library = try Library.fromSource(device, kernels, .{});
        return .{
            .device = device,
            .queue = queue,
            .library = library,
            .copy = try Pipeline.init(device, library, "copy_bump", false),
            .log = try Pipeline.init(device, library, "log_step", false),
            .window = try HostWindow.createShm(device, 1 << 20),
            .scratch = try device.buffer(1 << 20, types.ResourceOptions.shared),
        };
    }

    fn deinit(self: Rig) void {
        self.scratch.deinit();
        self.window.deinit();
        self.log.deinit();
        self.copy.deinit();
        self.library.deinit();
        self.queue.deinit();
        self.device.deinit();
    }

    /// Words of scratch: counter at 0, log from 16, copy results from 4096.
    fn words(self: Rig) []volatile u32 {
        return @as([*]volatile u32, @ptrCast(@alignCast(self.scratch.contents())))[0 .. (1 << 20) / 4];
    }
};

const Job = struct {
    rig: *const Rig,
    step: u32 = 0,

    fn copyBump(ctx: *anyopaque, cb: CommandBuffer) void {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const enc = cb.compute(.serial);
        enc.setPipeline(job.rig.copy);
        enc.setBuffer(job.rig.window.buffer, 0, 0);
        enc.setBuffer(job.rig.scratch, 4096 * 4, 1);
        enc.setBuffer(job.rig.scratch, 0, 2);
        enc.dispatchThreads(types.Size.of(1024, 1, 1), types.Size.of(256, 1, 1));
        enc.end();
    }

    fn logStep(ctx: *anyopaque, cb: CommandBuffer) void {
        const job: *Job = @ptrCast(@alignCast(ctx));
        const enc = cb.compute(.serial);
        enc.setPipeline(job.rig.log);
        enc.setBuffer(job.rig.scratch, 0, 0);
        enc.setBuffer(job.rig.scratch, 16 * 4, 1);
        enc.setValue(job.step, 2);
        enc.dispatchThreads(types.Size.of(1, 1, 1), types.Size.of(1, 1, 1));
        enc.end();
    }
};

test "relay: each step reads exactly the data the CPU stored before opening its gate" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    const in = rig.window.words();
    const out = rig.words();
    out[0] = 0;
    var stale: usize = 0;
    for (1..2001) |step| {
        const ticket = try relay.submit(&job, Job.copyBump);
        wake.sleepNs(100_000); // the buffer is parked on its gate by now
        const value: u32 = @intCast(step * 0x9E3779B1 % 0xFFFF_FFFF);
        for (0..1024) |i| in[i] = value +% @as(u32, @intCast(i));
        relay.open(ticket);
        try testing.expect(relay.waitDone(ticket, 5000));
        try testing.expect(relay.failed(ticket) == null);
        try testing.expectEqual(@as(u32, @intCast(step)), out[0]);
        for (0..1024) |i| stale += @intFromBool(out[4096 + i] != value +% @as(u32, @intCast(i)));
    }
    try testing.expectEqual(@as(usize, 0), stale);
}

fn expectLog(rig: *const Rig, expected: []const u32) !void {
    const w = rig.words();
    try testing.expectEqual(@as(u32, @intCast(expected.len)), w[0]);
    for (expected, 0..) |step, i| try testing.expectEqual(step, w[16 + i]);
}

test "relay: steps pre-committed ahead open in order" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    var tickets: [3]Ticket = undefined;
    for (&tickets, 1..) |*t, step| {
        job.step = @intCast(step);
        t.* = try relay.submit(&job, Job.logStep);
    }
    try testing.expectEqual(@as(u64, 1), tickets[0].gate);
    try testing.expectEqual(@as(u64, 4), tickets[1].done);
    // nothing runs before its gate: only step 1 is released, steps 2 and 3 stay parked
    wake.sleepNs(2_000_000);
    try expectLog(&rig, &.{});
    relay.open(tickets[0]);
    try testing.expect(relay.waitDone(tickets[0], 5000));
    try testing.expect(!relay.waitDone(tickets[1], 50));
    try expectLog(&rig, &.{1});
    for (tickets[1..]) |t| {
        relay.open(t);
        try testing.expect(relay.waitDone(t, 5000));
    }
    try expectLog(&rig, &.{ 1, 2, 3 });
}

test "relay: an out-of-order open releases the earlier gates too and the steps still run in submit order" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    var tickets: [3]Ticket = undefined;
    for (&tickets, 1..) |*t, step| {
        job.step = @intCast(step);
        t.* = try relay.submit(&job, Job.logStep);
    }
    relay.open(tickets[2]);
    try testing.expect(relay.waitDone(tickets[2], 5000));
    try expectLog(&rig, &.{ 1, 2, 3 });
    // opening an earlier ticket afterwards changes nothing, the event never goes back
    relay.open(tickets[0]);
    try testing.expect(relay.waitDone(tickets[1], 5000));
    try testing.expect(relay.event.value() >= tickets[2].done);
}

test "relay: abort drains parked buffers and fails later calls" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    var tickets: [3]Ticket = undefined;
    for (&tickets, 1..) |*t, step| {
        job.step = @intCast(step);
        t.* = try relay.submit(&job, Job.logStep);
    }
    wake.sleepNs(2_000_000);
    try expectLog(&rig, &.{});
    relay.abort();
    for (relay.slots[0..3]) |slot| {
        slot.cb.wait();
        try testing.expect(slot.cb.failure() == null);
    }
    try expectLog(&rig, &.{ 1, 2, 3 });
    try testing.expect(!relay.waitDone(tickets[2], 100));
    try testing.expectError(error.Aborted, relay.submit(&job, Job.logStep));
}

test "relay: deinit with unopened steps does not hang" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    var job: Job = .{ .rig = &rig };
    for (0..4) |_| _ = try relay.submit(&job, Job.logStep);
    relay.deinit();
    try testing.expectEqual(@as(u32, 4), rig.words()[0]);
}

test "relay: submit refuses a step whose slot still holds an unfinished buffer" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    for (0..max_in_flight) |_| _ = try relay.submit(&job, Job.logStep);
    try testing.expectError(error.TooManyInFlight, relay.submit(&job, Job.logStep));
    try testing.expectEqual(@as(u64, max_in_flight), relay.steps);
}

test "relay: slots are recycled once their buffers complete" {
    const pool = objc.Pool.push();
    defer pool.pop();
    const rig = try Rig.init();
    defer rig.deinit();
    rig.words()[0] = 0;
    var relay = try Relay.init(rig.device, rig.queue);
    defer relay.deinit();
    var job: Job = .{ .rig = &rig };
    for (0..3 * max_in_flight) |i| {
        job.step = @intCast(i % 16);
        const t = try relay.submit(&job, Job.logStep);
        relay.open(t);
        try testing.expect(relay.waitDone(t, 5000));
        // the counter is the log index: keep it inside the scratch buffer's log area
        rig.words()[0] = 0;
    }
}
