//! A probe `Host` for a build with no GPU of its own: the CPU stands in for the Mac, so its receipt is not a Mac's.
const std = @import("std");
const builtin = @import("builtin");
const device = @import("device.zig");
const probe = @import("probe.zig");

const Allocator = std.mem.Allocator;

const stream_bytes = 256 << 20;
const window_bytes = 64 << 20;

pub const CpuHost = struct {
    pub fn host(self: *CpuHost) probe.Host {
        return .{ .ptr = self, .info = info, .wake = wake, .wrap = wrap };
    }

    fn info(_: *anyopaque, _: Allocator) anyerror!probe.HostInfo {
        return .{
            .name = "cpu stand-in",
            .os_build = @tagName(builtin.os.tag),
            .max_working_set_bytes = try std.process.totalSystemMemory(),
            .stream_read_bps = try streamRead(),
        };
    }

    /// Time from a store to a spinning thread's completion (at least a nanosecond): a stand-in for an event-gated wake.
    fn wake(_: *anyopaque, out: []u64) anyerror!void {
        var gate: Gate = .{};
        const t = try std.Thread.spawn(.{}, Gate.waiter, .{&gate});
        defer {
            gate.stop.store(true, .release);
            t.join();
        }
        for (out, 1..) |*sample, i| {
            const seq: u32 = @intCast(i);
            const start = device.nowNs();
            gate.go.store(seq, .release);
            while (gate.done.load(.acquire) != seq) {
                if (device.nowNs() - start > 2_000_000_000) return error.Timeout;
                std.atomic.spinLoopHint();
            }
            sample.* = @max(device.nowNs() - start, 1);
        }
    }

    /// Time to map and unmap a 64 MiB page-aligned window.
    fn wrap(_: *anyopaque, out: []u64) anyerror!void {
        for (out) |*sample| {
            const start = device.nowNs();
            const mem = try std.posix.mmap(null, window_bytes, .{ .READ = true, .WRITE = true }, .{ .TYPE = .PRIVATE, .ANONYMOUS = true }, -1, 0);
            std.posix.munmap(mem);
            sample.* = device.nowNs() - start;
        }
    }
};

const Gate = struct {
    go: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),

    fn waiter(g: *Gate) void {
        var seen: u32 = 0;
        while (!g.stop.load(.acquire)) {
            const v = g.go.load(.acquire);
            if (v == seen) {
                std.atomic.spinLoopHint();
                continue;
            }
            seen = v;
            g.done.store(v, .release);
        }
    }
};

/// Bytes a second of the best of three sums over a buffer far larger than any cache.
fn streamRead() !f64 {
    const words = try std.heap.page_allocator.alloc(u64, stream_bytes / 8);
    defer std.heap.page_allocator.free(words);
    for (words, 0..) |*w, i| w.* = i;
    var best: u64 = std.math.maxInt(u64);
    for (0..3) |_| {
        const start = device.nowNs();
        var sum: [4]u64 = @splat(0);
        var i: usize = 0;
        while (i + 4 <= words.len) : (i += 4) for (&sum, words[i..][0..4]) |*s, w| {
            s.* +%= w;
        };
        std.mem.doNotOptimizeAway(sum);
        best = @min(best, device.nowNs() - start);
    }
    return @as(f64, stream_bytes) / (@as(f64, @floatFromInt(@max(best, 1))) * 1e-9);
}

test "the stand-in host reports memory, a stream rate, and wake and wrap samples" {
    var cpu: CpuHost = .{};
    const h = cpu.host();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const info = try h.info(h.ptr, arena.allocator());
    try std.testing.expect(info.max_working_set_bytes > stream_bytes and info.stream_read_bps > 1e8);
    var wakes: [32]u64 = undefined;
    try h.wake(h.ptr, &wakes);
    try std.testing.expect(std.mem.max(u64, &wakes) > 0 and std.mem.max(u64, &wakes) < 2_000_000_000);
    var wraps: [4]u64 = undefined;
    try h.wrap(h.ptr, &wraps);
    try std.testing.expect(std.mem.max(u64, &wraps) > 0);
}
