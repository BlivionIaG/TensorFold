//! The knobs of elastic memory (TF_FLASHNEXT_ELASTIC_MEMORY=1) and the refusal when its two halves are not both on.
const std = @import("std");
const policy = @import("elastic_policy.zig");

pub const Settings = struct {
    policy: policy.Config = .{},
    /// Milliseconds between two ticks of the controller.
    tick_ms: u64 = 250,
};

/// Reads one variable: null when unset. Tests pass their own.
pub const Get = *const fn (name: []const u8) ?[]const u8;

fn numeric(comptime T: type, text: ?[]const u8, default: T, least: T, most: T) !T {
    const v = std.mem.trim(u8, text orelse return default, " \t");
    if (v.len == 0) return default;
    const n = std.fmt.parseInt(T, v, 10) catch return error.BadFlashNextElasticMemory;
    if (n < least or n > most) return error.BadFlashNextElasticMemory;
    return n;
}

/// TF_FLASHNEXT_ELASTIC_MEMORY: unset, empty or 0 is off, 1 is on, anything else is refused.
pub fn parseOn(text: ?[]const u8) !bool {
    const v = std.mem.trim(u8, text orelse return false, " \t");
    if (v.len == 0 or std.mem.eql(u8, v, "0")) return false;
    if (std.mem.eql(u8, v, "1")) return true;
    return error.BadFlashNextElasticMemory;
}

/// The settings from the environment (TF_FLASHNEXT_ELASTIC_ HIGH_PCT, LOW_PCT, FILL_PCT, IDLE_TICKS, ... STEP_MB).
pub fn parse(get: Get) !Settings {
    const prefix = "TF_FLASHNEXT_ELASTIC_";
    var s: Settings = .{};
    const d = s.policy;
    const pct = struct {
        fn of(g: Get, name: []const u8, default: u32) !u32 {
            var buf: [64]u8 = undefined;
            const key = std.fmt.bufPrint(&buf, prefix ++ "{s}", .{name}) catch unreachable;
            return (try numeric(u32, g(key), default / 10_000, 1, 100)) * 10_000;
        }
        fn count(g: Get, name: []const u8, default: u32, least: u32) !u32 {
            var buf: [64]u8 = undefined;
            const key = std.fmt.bufPrint(&buf, prefix ++ "{s}", .{name}) catch unreachable;
            return numeric(u32, g(key), default, least, 1 << 24);
        }
    };
    s.policy.high_ppm = try pct.of(get, "HIGH_PCT", d.high_ppm);
    s.policy.low_ppm = try pct.of(get, "LOW_PCT", d.low_ppm);
    s.policy.fill_ppm = try pct.of(get, "FILL_PCT", d.fill_ppm);
    s.policy.idle_ticks = try pct.count(get, "IDLE_TICKS", d.idle_ticks, 1);
    s.policy.dwell_ticks = try pct.count(get, "DWELL_TICKS", d.dwell_ticks, 0);
    s.policy.backoff_ticks = try pct.count(get, "BACKOFF_TICKS", d.backoff_ticks, 0);
    s.policy.hold_ticks = try pct.count(get, "HOLD_TICKS", d.hold_ticks, 0);
    s.tick_ms = try numeric(u64, get(prefix ++ "TICK_MS"), s.tick_ms, 1, 3_600_000);
    const mb = try numeric(u64, get(prefix ++ "STEP_MB"), d.step_bytes >> 20, 1, 1 << 20);
    s.policy.step_bytes = mb << 20;
    if (!s.policy.valid()) return error.BadFlashNextElasticMemory; // the watermarks are ordered: low < fill < high
    return s;
}

fn envGet(name: []const u8) ?[]const u8 {
    var buf: [96]u8 = undefined;
    if (name.len >= buf.len) return null;
    @memcpy(buf[0..name.len], name);
    buf[name.len] = 0;
    const key: [*:0]const u8 = @ptrCast(&buf);
    return if (std.c.getenv(key)) |v| std.mem.span(v) else null;
}

/// Whether TF_FLASHNEXT_ELASTIC_MEMORY=1.
pub fn enabled() !bool {
    return parseOn(envGet("TF_FLASHNEXT_ELASTIC_MEMORY"));
}

pub fn fromEnv() !Settings {
    return parse(envGet);
}

/// Elastic memory moves bytes between the two arenas, so it needs both elastic: refused at load otherwise.
pub fn requireModes(on: bool, kv_elastic: bool, lines_elastic: bool) !void {
    if (!on) return;
    if (!kv_elastic or !lines_elastic) return error.ElasticMemoryNeedsBothElasticModes;
}

/// The load-time check: whether elastic memory is on, refused unless an EXL3 pack has paged K/V and lines both elastic.
pub fn check(is_exl3: bool, paged: bool) !bool {
    if (!try enabled()) return false;
    const kv = (try @import("../../cuda_kv_elastic.zig").fromEnv()) != null;
    const lines = try @import("../offload/exl3_elastic_lines_env.zig").fromEnv();
    try requireModes(true, kv and paged, lines and is_exl3);
    return true;
}

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};

    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

test "exl3: elastic memory: the flag is exactly 0 or 1" {
    try std.testing.expect(!try parseOn(null));
    try std.testing.expect(!try parseOn(""));
    try std.testing.expect(!try parseOn("0"));
    try std.testing.expect(try parseOn("1"));
    for ([_][]const u8{ "2", "true", "on", "-1" }) |bad| try std.testing.expectError(error.BadFlashNextElasticMemory, parseOn(bad));
}

test "exl3: elastic memory: defaults hold, knobs parse, and unordered watermarks are refused" {
    Fake.pairs = &.{};
    const d = try parse(Fake.get);
    try std.testing.expect(d.policy.valid());
    try std.testing.expectEqual(@as(u64, 250), d.tick_ms);
    try std.testing.expectEqual(@as(u32, 850_000), d.policy.high_ppm);
    try std.testing.expectEqual(@as(u32, 120), d.policy.hold_ticks);
    Fake.pairs = &.{ .{ "TF_FLASHNEXT_ELASTIC_HIGH_PCT", "90" }, .{ "TF_FLASHNEXT_ELASTIC_LOW_PCT", "20" }, .{ "TF_FLASHNEXT_ELASTIC_FILL_PCT", "50" }, .{ "TF_FLASHNEXT_ELASTIC_IDLE_TICKS", "3" }, .{ "TF_FLASHNEXT_ELASTIC_DWELL_TICKS", "0" }, .{ "TF_FLASHNEXT_ELASTIC_HOLD_TICKS", "0" }, .{ "TF_FLASHNEXT_ELASTIC_TICK_MS", "1000" }, .{ "TF_FLASHNEXT_ELASTIC_STEP_MB", "128" } };
    const s = try parse(Fake.get);
    try std.testing.expectEqual(@as(u32, 900_000), s.policy.high_ppm);
    try std.testing.expectEqual(@as(u32, 200_000), s.policy.low_ppm);
    try std.testing.expectEqual(@as(u32, 500_000), s.policy.fill_ppm);
    try std.testing.expectEqual(@as(u32, 3), s.policy.idle_ticks);
    try std.testing.expectEqual(@as(u32, 0), s.policy.dwell_ticks);
    try std.testing.expectEqual(@as(u32, 0), s.policy.hold_ticks);
    try std.testing.expectEqual(@as(u64, 1000), s.tick_ms);
    try std.testing.expectEqual(@as(u64, 128 << 20), s.policy.step_bytes);
    for ([_][2][]const u8{ .{ "TF_FLASHNEXT_ELASTIC_LOW_PCT", "70" }, .{ "TF_FLASHNEXT_ELASTIC_HIGH_PCT", "101" }, .{ "TF_FLASHNEXT_ELASTIC_IDLE_TICKS", "0" }, .{ "TF_FLASHNEXT_ELASTIC_TICK_MS", "0" }, .{ "TF_FLASHNEXT_ELASTIC_STEP_MB", "x" }, .{ "TF_FLASHNEXT_ELASTIC_FILL_PCT", "95" } }) |bad| {
        Fake.pairs = &.{bad};
        try std.testing.expectError(error.BadFlashNextElasticMemory, parse(Fake.get));
    }
    Fake.pairs = &.{};
}

test "exl3: elastic memory: it needs both elastic modes, and off asks for nothing" {
    try requireModes(false, false, false);
    try requireModes(true, true, true);
    try std.testing.expectError(error.ElasticMemoryNeedsBothElasticModes, requireModes(true, true, false));
    try std.testing.expectError(error.ElasticMemoryNeedsBothElasticModes, requireModes(true, false, true));
    try std.testing.expectError(error.ElasticMemoryNeedsBothElasticModes, requireModes(true, false, false));
}
