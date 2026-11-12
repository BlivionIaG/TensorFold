//! The Mac's half of `offload probe`: Metal's report, a streaming read, the wake of a parked buffer, a window wrap.
const std = @import("std");
const metal = @import("metal");
const split = @import("split");

const probe = split.probe;
const Allocator = std.mem.Allocator;

const read_source =
    \\#include <metal_stdlib>
    \\using namespace metal;
    \\kernel void sum_read(device const uint4* in [[buffer(0)]], device uint* out [[buffer(1)]], constant uint& n [[buffer(2)]],
    \\                     uint t [[thread_position_in_grid]], uint nt [[threads_per_grid]]) {
    \\    uint4 acc = uint4(0);
    \\    for (uint i = t; i < n; i += nt) acc += in[i];
    \\    out[t] = acc.x ^ acc.y ^ acc.z ^ acc.w;
    \\}
;

pub const MetalHost = struct {
    pub fn host(self: *MetalHost) probe.Host {
        return .{ .ptr = self, .info = info, .wake = wake, .wrap = wrap };
    }

    fn info(_: *anyopaque, a: Allocator) anyerror!probe.HostInfo {
        const pool = metal.objc.Pool.push();
        defer pool.pop();
        const dev = try metal.Device.init();
        defer dev.deinit();
        return .{
            .name = try a.dupe(u8, std.mem.span(dev.name())),
            .os_build = try osBuild(a),
            .max_working_set_bytes = dev.maxWorkingSet(),
            .stream_read_bps = try streamRead(dev),
        };
    }

    fn wake(_: *anyopaque, out: []u64) anyerror!void {
        const pool = metal.objc.Pool.push();
        defer pool.pop();
        const dev = try metal.Device.init();
        defer dev.deinit();
        try metal.wake_probe.wake(dev, out.len, out);
    }

    fn wrap(_: *anyopaque, out: []u64) anyerror!void {
        const pool = metal.objc.Pool.push();
        defer pool.pop();
        const dev = try metal.Device.init();
        defer dev.deinit();
        const len: usize = 64 << 20;
        const mem = try std.posix.mmap(null, len, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
        defer std.posix.munmap(mem);
        for (out) |*o| {
            const t0 = metal.wake_probe.nowNs();
            const w = try metal.HostWindow.wrap(dev, mem.ptr, len);
            o.* = metal.wake_probe.nowNs() - t0;
            w.deinit();
        }
    }
};

/// kern.osversion, e.g. "26B123".
fn osBuild(a: Allocator) ![]const u8 {
    var buf: [64]u8 = undefined;
    var len: usize = buf.len;
    if (std.c.sysctlbyname("kern.osversion", &buf, &len, null, 0) != 0 or len == 0) return a.dupe(u8, "unknown");
    return a.dupe(u8, std.mem.sliceTo(buf[0..len], 0));
}

/// Sustained bytes a second one pass of a streaming kernel reads from a 512 MiB buffer, the best of five passes.
fn streamRead(dev: metal.Device) !f64 {
    const bytes: usize = 512 << 20;
    const lib = try metal.Library.fromSource(dev, read_source, .{});
    defer lib.deinit();
    const pl = try metal.Pipeline.init(dev, lib, "sum_read", false);
    defer pl.deinit();
    const input = try dev.buffer(bytes, metal.ResourceOptions.shared);
    defer input.deinit();
    @memset(input.contents()[0..bytes], 0x5a);
    const threads: usize = 1 << 20;
    const output = try dev.buffer(threads * 4, metal.ResourceOptions.shared);
    defer output.deinit();
    const queue = try dev.queue();
    defer queue.deinit();
    var best: f64 = std.math.floatMax(f64);
    for (0..5) |_| {
        const cb = queue.commandBuffer();
        const enc = cb.compute(.serial);
        enc.setPipeline(pl);
        enc.setBuffer(input, 0, 0);
        enc.setBuffer(output, 0, 1);
        enc.setValue(@as(u32, @intCast(bytes / 16)), 2);
        enc.dispatchThreads(metal.Size.of(threads, 1, 1), metal.Size.of(256, 1, 1));
        enc.end();
        cb.commit();
        cb.wait();
        best = @min(best, cb.gpuSeconds());
    }
    return @as(f64, @floatFromInt(bytes)) / best;
}

test "the Metal host reports a plausible Mac" {
    var h: MetalHost = .{};
    const host = h.host();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const info = host.info(host.ptr, arena.allocator()) catch |err| switch (err) {
        error.NoDevice => return error.SkipZigTest,
        else => return err,
    };
    try std.testing.expect(info.name.len > 0 and info.max_working_set_bytes > (1 << 30));
    // any Apple GPU streams at least 20 GB/s; other builds loading the machine only lower it, so the bound is loose
    try std.testing.expect(info.stream_read_bps > 20e9);
    var wrap: [8]u64 = undefined;
    try host.wrap(host.ptr, &wrap);
    for (wrap) |w| try std.testing.expect(w > 0);
}
