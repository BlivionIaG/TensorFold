//! TF_FLASHNEXT_MEM_PROBE=1: the device's real free bytes (cuMemGetInfo) at each allocation phase of an EXL3 start
//! and after each graph capture, so a memory ledger can be set against the device rather than the allocator's counters.
const std = @import("std");
const cuda = @import("cuda");

pub fn enabled() bool {
    const v = std.c.getenv("TF_FLASHNEXT_MEM_PROBE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

pub fn free(d: *const cuda.Driver) u64 {
    var f: usize = 0;
    var t: usize = 0;
    d.check(d.api.cuMemGetInfo_v2(&f, &t), "cuMemGetInfo") catch return 0;
    return f;
}

/// Logs the used bytes of the device (total - free) under `label`; returns the free bytes (0 when off).
pub fn at(d: *const cuda.Driver, label: []const u8) u64 {
    if (!enabled()) return 0;
    var f: usize = 0;
    var t: usize = 0;
    d.check(d.api.cuMemGetInfo_v2(&f, &t), "cuMemGetInfo") catch return 0;
    std.log.info("memprobe {s}: free={d} used={d}", .{ label, f, t - f });
    return f;
}

test "memprobe is off without the environment variable" {
    try std.testing.expect(!enabled() or std.c.getenv("TF_FLASHNEXT_MEM_PROBE") != null);
}
