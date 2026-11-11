//! The session tier's options (read in sessions_gpu.zig) and the store policy they make.

const std = @import("std");
const kv = @import("kv");

const sessions = kv.sessions;

pub const Options = struct {
    /// TF_DSV41_SESSIONS (default 1)
    enabled: bool = true,
    disk: ?[]const u8 = null,
    disk_budget: u64 = 64 << 30,
    disk_min_tokens: u64 = 1024,
    /// TF_DSV41_SESSION_RESIDENT / _RESIDENT_GIB: idle chats and bytes kept in RAM (null: unlimited; 0: none idle)
    resident: ?u32 = null,
    resident_bytes: ?u64 = null,
    /// TF_DSV41_SESSION_RAM_MIB (default 256)
    ram_bytes: u64 = 256 << 20,
    staging_mib: u32 = 128,
    skip_covered: bool = false,
    delta: bool = false,
    /// TF_DSV41_SESSION_FRESH=1: purge the NVMe tier at boot (default: entries of this model and profile survive)
    fresh: bool = false,
    /// TF_DSV41_SESSION_SLOTS=1: the store with several live slots too (default off)
    several: bool = false,
    /// TF_DSV41_SAVE_QUEUE_MIB (default 256): MiB of saves a worker writes behind the engine (0: inline)
    save_queue_mib: u64 = 256,
};

/// The store's policy from the family's options: an unset residency stays unlimited, as sessions.Policy reads null.
pub fn policy(o: Options) sessions.store.Policy {
    return .{ .ram_bytes = o.ram_bytes, .chats = o.resident, .resident_bytes = o.resident_bytes, .skip_covered = o.skip_covered, .delta = o.delta, .write_through = o.disk != null, .save_queue_bytes = if (o.disk != null) o.save_queue_mib << 20 else 0 };
}

test "the default session options keep idle chats unlimited; an explicit 0 keeps none" {
    const unset = policy(.{});
    try std.testing.expectEqual(@as(?u32, null), unset.chats);
    try std.testing.expectEqual(@as(?u64, null), unset.resident_bytes);
    const zero = policy(.{ .resident = 0, .resident_bytes = 0 });
    try std.testing.expectEqual(@as(?u32, 0), zero.chats);
    try std.testing.expectEqual(@as(?u64, 0), zero.resident_bytes);
}

test "the save queue is on with a disk tier by default and off without one or at 0 MiB" {
    try std.testing.expectEqual(@as(u64, 0), policy(.{}).save_queue_bytes);
    try std.testing.expectEqual(@as(u64, 256 << 20), policy(.{ .disk = "/tier" }).save_queue_bytes);
    try std.testing.expectEqual(@as(u64, 0), policy(.{ .disk = "/tier", .save_queue_mib = 0 }).save_queue_bytes);
    try std.testing.expectEqual(@as(u64, 8 << 20), policy(.{ .disk = "/tier", .save_queue_mib = 8 }).save_queue_bytes);
}
