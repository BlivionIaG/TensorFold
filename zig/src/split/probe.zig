//! `offload probe`: measure the link and write the receipt the planner reads; the Mac's half comes in through `Host`.
const std = @import("std");
const device = @import("device.zig");
const receipt = @import("receipt.zig");

const Allocator = std.mem.Allocator;

pub const HostInfo = struct {
    name: []const u8,
    os_build: []const u8,
    max_working_set_bytes: u64,
    /// Sustained read bandwidth of a streaming kernel over a buffer far larger than any cache, bytes a second.
    stream_read_bps: f64,
};

/// The Mac's side of the probe, supplied by the Metal build.
pub const Host = struct {
    ptr: *anyopaque,
    info: *const fn (ptr: *anyopaque, a: Allocator) anyerror!HostInfo,
    /// Nanoseconds from the host opening a parked command buffer's gate to its completion, `out.len` samples.
    wake: *const fn (ptr: *anyopaque, out: []u64) anyerror!void,
    /// Nanoseconds to wrap a 64 MiB page-aligned window as a Metal buffer, `out.len` samples.
    wrap: *const fn (ptr: *anyopaque, out: []u64) anyerror!void,
};

pub const Options = struct {
    iterations: usize = 400,
    /// Bytes moved each way for the bandwidth figures; capped by the device's memory.
    bulk_bytes: usize = 64 << 20,
    /// The Mac's end-to-end decode bandwidth (bytes a second) when measured; else a share of the streaming read.
    mac_decode_bps: ?f64 = null,
};

/// Share of a streaming read's bandwidth the Mac's decode reaches end to end (assumed; `mac_decode_bps` replaces it).
pub const decode_over_stream: f64 = 0.74;

pub const Error = device.Error || error{ HostProbeFailed, TooFewSamples };

fn ns(x: u64) f64 {
    return @floatFromInt(x);
}

fn percentile(sorted: []const u64, p: f64) f64 {
    const i: usize = @intFromFloat(@floor(p * @as(f64, @floatFromInt(sorted.len - 1))));
    return ns(sorted[i]);
}

/// Median and 99th percentile of `samples` (sorted in place) as a latency.
pub fn latency(samples: []u64, source: receipt.Source) receipt.Latency {
    std.mem.sort(u64, samples, {}, std.sort.asc(u64));
    return .{ .p50 = percentile(samples, 0.5), .p99 = @max(percentile(samples, 0.99), percentile(samples, 0.5)), .source = source };
}

/// Round trips through a window word: the host stores `i`, the waiting device stream writes it back.
pub fn roundTrip(dev: device.ExternalDevice, out: []u64) Error!void {
    const w = try dev.allocWindow(16 * 1024);
    defer dev.freeWindow(w);
    const ask = w.word(0);
    const answer = w.word(64);
    for (out, 1..) |*slot, i| {
        const seq: u32 = @intCast(i);
        try dev.enqueueWait(w, 0, seq, 2_000_000_000);
        try dev.enqueueWrite(w, 64, seq);
        const start = device.nowNs();
        @atomicStore(u32, ask, seq, .release);
        while (@atomicLoad(u32, answer, .acquire) != seq) {
            if (device.nowNs() - start > 2_000_000_000) return error.Timeout;
            std.atomic.spinLoopHint();
        }
        slot.* = device.nowNs() - start;
    }
    try dev.sync(2_000_000_000);
}

/// Bytes a second for one bulk copy each way through device memory.
pub fn bulk(gpa: Allocator, dev: device.ExternalDevice, bytes: usize) (Error || Allocator.Error)![2]f64 {
    const n = @min(bytes, dev.caps().vram_bytes / 4);
    const buf = try gpa.alloc(u8, n);
    defer gpa.free(buf);
    @memset(buf, 0x5a);
    const p = try dev.alloc(n);
    defer dev.free(p);
    const t0 = device.nowNs();
    try dev.upload(p, buf);
    const t1 = device.nowNs();
    try dev.download(buf, p);
    const t2 = device.nowNs();
    const f: f64 = @floatFromInt(n);
    return .{ f / (@as(f64, @floatFromInt(t1 - t0)) * 1e-9), f / (@as(f64, @floatFromInt(t2 - t1)) * 1e-9) };
}

/// Decode efficiency (fraction of peak bandwidth) of an architecture, assumed until a run on that card replaces it.
pub fn decodeEfficiency(arch: []const u8) receipt.Labeled(f64) {
    if (std.mem.startsWith(u8, arch, "gfx900")) return .{ .value = 0.37, .source = .assumed };
    if (std.mem.startsWith(u8, arch, "gfx11") or std.mem.startsWith(u8, arch, "gfx12")) return .{ .value = 0.50, .source = .assumed };
    return .{ .value = 0.37, .source = .assumed };
}

/// Run the probe. Strings in the result live in `a` (an arena the caller owns).
pub fn run(a: Allocator, host: Host, dev: device.ExternalDevice, options: Options) (Error || Allocator.Error || anyerror)!receipt.Receipt {
    const n = options.iterations;
    if (n < 8) return error.TooFewSamples;
    const info = try host.info(host.ptr, a);
    const caps = dev.caps();

    const wake = try a.alloc(u64, n);
    try host.wake(host.ptr, wake);
    const wrap = try a.alloc(u64, @min(n, 64));
    try host.wrap(host.ptr, wrap);
    const rtt = try a.alloc(u64, n);
    try roundTrip(dev, rtt);
    const rates = try bulk(a, dev, options.bulk_bytes);

    const wake_l = latency(wake, .measured);
    const rtt_l = latency(rtt, .measured);
    std.mem.sort(u64, wrap, {}, std.sort.asc(u64));
    const to_device: receipt.Latency = .{ .p50 = rtt_l.p50 / 2, .p99 = rtt_l.p99 / 2, .source = .probed };
    const to_host: receipt.Latency = .{ .p50 = rtt_l.p50 / 2 + wake_l.p50, .p99 = rtt_l.p99 / 2 + wake_l.p99, .source = .probed };
    const decode_bps = options.mac_decode_bps orelse info.stream_read_bps * decode_over_stream;
    return .{
        .os_build = info.os_build,
        .mac = .{
            .name = info.name,
            .max_working_set_bytes = .{ .value = info.max_working_set_bytes, .source = .probed },
            .decode_bandwidth_bps = .{ .value = decode_bps, .source = if (options.mac_decode_bps != null) .measured else .assumed },
        },
        .device = .{
            .name = try a.dupe(u8, caps.name),
            .arch = try a.dupe(u8, caps.arch),
            .vram_bytes = .{ .value = caps.vram_bytes, .source = .probed },
            .peak_bandwidth_bps = .{ .value = @floatFromInt(caps.peak_bandwidth), .source = .documented },
            .decode_efficiency = decodeEfficiency(caps.arch),
            .act = switch (caps.act) {
                .f16 => .f16,
                .bf16 => .bf16,
            },
            .launch_ns = .{ .value = @floatFromInt(caps.launch_ns), .source = .documented },
            .backend = try a.dupe(u8, caps.backend),
        },
        .link = .{
            .h2d_bps = .{ .value = rates[0], .source = .measured },
            .d2h_bps = .{ .value = rates[1], .source = .measured },
            .hop_to_host_ns = to_host,
            .hop_to_device_ns = to_device,
            .metal_wake_ns = wake_l,
            .window_wrap_ns = .{ .value = percentile(wrap, 0.5), .source = .measured },
        },
    };
}

test "a probe over the mock device and a scripted host writes a receipt the planner can read" {
    const mock = @import("mock.zig");
    const Fake = struct {
        fn info(_: *anyopaque, _: Allocator) anyerror!HostInfo {
            return .{ .name = "test mac", .os_build = "TEST", .max_working_set_bytes = 40 << 30, .stream_read_bps = 220e9 };
        }
        fn wake(_: *anyopaque, out: []u64) anyerror!void {
            for (out, 0..) |*o, i| o.* = 70_000 + i;
        }
        fn wrap(_: *anyopaque, out: []u64) anyerror!void {
            for (out) |*o| o.* = 20_000;
        }
    };
    var dummy: u8 = 0;
    const host: Host = .{ .ptr = &dummy, .info = Fake.info, .wake = Fake.wake, .wrap = Fake.wrap };
    const m = try mock.Mock.create(std.testing.allocator, .{ .vram_bytes = 1 << 30 });
    const dev = m.external();
    defer dev.deinit();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const r = try run(arena.allocator(), host, dev, .{ .iterations = 64, .bulk_bytes = 1 << 20 });
    try std.testing.expect(r.link.hop_to_host_ns.p50 > r.link.metal_wake_ns.p50);
    try std.testing.expect(r.link.hop_to_device_ns.p99 >= r.link.hop_to_device_ns.p50);
    try std.testing.expectApproxEqAbs(@as(f64, 220e9 * decode_over_stream), r.mac.decode_bandwidth_bps.value, 1);
    try std.testing.expectEqual(receipt.Source.assumed, r.mac.decode_bandwidth_bps.source);
    const bytes = try receipt.write(std.testing.allocator, r);
    defer std.testing.allocator.free(bytes);
    var parsed = try receipt.parse(std.testing.allocator, bytes);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("mock", parsed.value.device.arch);
}
