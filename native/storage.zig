const std = @import("std");

pub const reserve_bytes: u64 = 64 * 1024 * 1024 * 1024;

// Darwin sys/statvfs.h; query space available to the current user.
const Statvfs = extern struct {
    bsize: c_ulong,
    frsize: c_ulong,
    blocks: u64,
    bfree: u64,
    bavail: u64,
    files: u64,
    ffree: u64,
    favail: u64,
    fsid: c_ulong,
    flag: c_ulong,
    namemax: c_ulong,
};
extern "c" fn statvfs(path: [*:0]const u8, result: *Statvfs) c_int;

fn checkCapacity(available: u64, incoming: u64) !void {
    if (available < reserve_bytes or incoming > available - reserve_bytes)
        return error.DiskReserveExceeded;
}

pub fn check(path: [:0]const u8, incoming: u64) !void {
    var buffer: [4096]u8 = undefined;
    const parent = try std.fmt.bufPrintSentinel(&buffer, "{s}", .{std.fs.path.dirname(path) orelse "."}, 0);
    var stats: Statvfs = undefined;
    if (statvfs(parent, &stats) != 0) return error.DiskSpaceUnavailable;
    checkCapacity(stats.bavail *| stats.frsize, incoming) catch |err| {
        std.debug.print("Write refused at {s}: preserving 64 GiB free disk space; prune generated fixtures before retrying.\n", .{path});
        return err;
    };
}

test "writes preserve the disk reserve including the incoming payload" {
    try checkCapacity(reserve_bytes + 4096, 4096);
    try std.testing.expectError(error.DiskReserveExceeded, checkCapacity(reserve_bytes + 4095, 4096));
    try std.testing.expectError(error.DiskReserveExceeded, checkCapacity(reserve_bytes - 1, 0));
    try std.testing.expectError(error.DiskReserveExceeded, checkCapacity(std.math.maxInt(u64), std.math.maxInt(u64)));
}

test "space checks use the output filesystem and fail closed" {
    try check("build/native-checks/storage-probe.npy", 0);
    try std.testing.expectError(error.DiskSpaceUnavailable, check("build/native-checks/missing-storage-parent/storage-probe.npy", 0));
}
