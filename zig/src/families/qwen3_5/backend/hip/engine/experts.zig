//! The offload of a MoE checkpoint's routed experts: whether a load does it, and the device cache sized after the plan.
const std = @import("std");
const hip = @import("hip");
const table = @import("../../../weights/table.zig");

pub const Mode = enum { auto, off, on };

/// What the user asked: `auto` offloads when the weights do not fit the budget, `on` always, `off` never.
pub const Settings = struct {
    mode: Mode = .auto,
    /// Experts the cache admits a round at most.
    copies: usize = 4096,
    /// Percent of the cache lines an SLRU protects; 0 is a plain LRU.
    protect: u8 = 75,
    /// Device bytes of the cache (null: what the plan leaves).
    cache_gib: ?f64 = null,
    /// Pinned host memory the device reads in place.
    window_mib: usize = 1536,

    /// `--ssd-experts GIB` names the cache and so turns the offload on.
    pub fn parse(env: anytype, cache_gib: ?f64) !Settings {
        const environment = @import("sys").environment;
        const d: Settings = .{};
        const offload = try environment.value(?u8, "TF_EXPERT_OFFLOAD", env.get("TF_EXPERT_OFFLOAD"), null);
        return .{
            .mode = if (cache_gib != null) .on else if (offload) |v| (if (v == 1) .on else .off) else .auto,
            .copies = try environment.value(usize, "TF_EXPERT_COPIES", env.get("TF_EXPERT_COPIES"), d.copies),
            .protect = @min(try environment.value(u8, "TF_EXPERT_PROTECT", env.get("TF_EXPERT_PROTECT"), d.protect), 95),
            .cache_gib = cache_gib,
            .window_mib = try environment.value(usize, "TF_EXPERT_WINDOW_MIB", env.get("TF_EXPERT_WINDOW_MIB"), d.window_mib),
        };
    }

    pub fn read(cache_gib: ?f64) !Settings {
        return parse(@import("sys").environment.Process{}, cache_gib);
    }
};

pub const Decision = struct { offload: bool, routed: u64 };

/// Whether this load keeps the routed experts in host memory: `total` is the rank's bytes, `room` the budget's.
pub fn decide(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, s: Settings, chat: bool, world: usize, total: u64, room: u64) !Decision {
    const routed = if (s.mode == .off or chat or world > 1) 0 else try table.routedBytes(gpa, io, dir);
    const offload = routed > 0 and (s.mode == .on or total > room);
    if (offload and !hip.kernels.expert_ptrs) {
        std.log.err("the checkpoint's weights do not fit and this GPU's kernels cannot read experts from host memory (caps expert_ptrs)", .{});
        return error.WeightsDoNotFit;
    }
    return .{ .offload = offload, .routed = routed };
}

/// What an offloaded checkpoint holds on the device: all but the routed experts, plus tables and the window.
pub fn resident(total: u64, routed: u64) u64 {
    return total - routed + routed / 256 + (128 << 20);
}

/// Device bytes the plan leaves untouched when the cache takes the rest: the driver's, the graphs' and a launch's own.
const margin: usize = 192 << 20;

fn gib(bytes: usize) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
}

/// Gives the offloaded experts their device cache after the rest of the plan is allocated, and starts the worker.
pub fn attach(s: *hip.expert_store.Store, pool: hip.admission.Pool, held: u64, o: Settings) !void {
    const blob = s.shape().blobBytes();
    const keys = s.shape().layers * s.shape().experts;
    const bytes: usize = if (o.cache_gib) |g| @intFromFloat(g * (1 << 30)) else pool.room(held) -| margin;
    const lines = @min(bytes / blob, keys);
    if (lines == 0) return error.OutOfDeviceMemory;
    try s.attachPool(lines, o.copies, o.protect);
    try s.start();
    const cached = s.lineCount();
    const sh = s.shape();
    std.log.info("experts: {d} layers x {d} routed in a {d:.2} GiB host tier, {d:.2} GiB resident; cache {d} lines = {d:.2} GiB ({d:.1}%)", .{
        sh.layers,
        sh.experts,
        gib(hip.expert_store.Store.hostBytes(sh)),
        gib(hip.expert_store.Store.residentBytes(sh)),
        cached,
        gib(cached * blob),
        100.0 * @as(f64, @floatFromInt(cached)) / @as(f64, @floatFromInt(keys)),
    });
}

const Env = struct {
    offload: ?[]const u8 = null,
    copies: ?[]const u8 = null,
    protect: ?[]const u8 = null,

    pub fn get(v: Env, name: []const u8) ?[]const u8 {
        if (std.mem.eql(u8, name, "TF_EXPERT_OFFLOAD")) return v.offload;
        if (std.mem.eql(u8, name, "TF_EXPERT_COPIES")) return v.copies;
        if (std.mem.eql(u8, name, "TF_EXPERT_PROTECT")) return v.protect;
        return null;
    }
};

test "unset settings are the defaults and auto offloads only when the weights do not fit" {
    const s = try Settings.parse(Env{}, null);
    try std.testing.expectEqual(Settings{}, s);
    try std.testing.expectEqual(Mode.auto, s.mode);
}

test "the switch, the copy budget and the protected share are read, the share capped at 95 percent" {
    const s = try Settings.parse(Env{ .offload = "0", .copies = "64", .protect = "100" }, null);
    try std.testing.expectEqual(Mode.off, s.mode);
    try std.testing.expectEqual(@as(usize, 64), s.copies);
    try std.testing.expectEqual(@as(u8, 95), s.protect);
    try std.testing.expectEqual(Mode.on, (try Settings.parse(Env{ .offload = "1" }, null)).mode);
}

test "an explicit cache size turns the offload on whatever the switch says" {
    const s = try Settings.parse(Env{ .offload = "0" }, 2.5);
    try std.testing.expectEqual(Mode.on, s.mode);
    try std.testing.expectEqual(@as(?f64, 2.5), s.cache_gib);
}

test "malformed settings are refused" {
    try std.testing.expectError(error.InvalidEnvironment, Settings.parse(Env{ .copies = "many" }, null));
    try std.testing.expectError(error.InvalidEnvironment, Settings.parse(Env{ .offload = "2" }, null));
    try std.testing.expectError(error.InvalidEnvironment, Settings.parse(Env{ .protect = "101" }, null));
}

test "the resident bytes of an offloaded checkpoint leave out its experts" {
    try std.testing.expectEqual(@as(u64, 10 << 30) - (8 << 30) + (8 << 30) / 256 + (128 << 20), resident(10 << 30, 8 << 30));
}
