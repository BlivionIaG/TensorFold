const std = @import("std");

pub const Options = struct {
    overlap: bool = true,
    recent_eviction: bool = false,
    copy_blocks: ?u32 = null,
    staging: usize = 0,
};

pub fn parse(overlap: ?[]const u8, recent: ?[]const u8, blocks: ?[]const u8, staging: ?[]const u8) !Options {
    return .{
        .overlap = try boolean(overlap, true),
        .recent_eviction = try boolean(recent, false),
        .copy_blocks = if (blocks) |text| @intCast(try bounded(text, 1, 1024)) else null,
        .staging = if (staging) |text| try bounded(text, 0, 8) else 0,
    };
}

pub fn fromEnv() !Options {
    return parse(env("TF_FLASHNEXT_EXL3_COPY_OVERLAP"), env("TF_FLASHNEXT_EXL3_RECENT_EVICTION"), env("TF_FLASHNEXT_EXL3_COPY_BLOCKS"), env("TF_FLASHNEXT_EXL3_STAGING_LINES"));
}

fn env(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |value| std.mem.span(value) else null;
}

fn boolean(value: ?[]const u8, default: bool) !bool {
    const text = value orelse return default;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.InvalidOffloadExperiment;
}

fn bounded(text: []const u8, minimum: usize, maximum: usize) !usize {
    if (text.len == 0) return error.InvalidOffloadExperiment;
    for (text) |byte| if (byte < '0' or byte > '9') return error.InvalidOffloadExperiment;
    const value = std.fmt.parseInt(usize, text, 10) catch return error.InvalidOffloadExperiment;
    if (value < minimum or value > maximum) return error.InvalidOffloadExperiment;
    return value;
}

test "exl3: offload experiments preserve defaults and bound resource knobs" {
    try std.testing.expectEqualDeep(Options{}, try parse(null, null, null, null));
    try std.testing.expectEqualDeep(Options{ .overlap = false, .recent_eviction = true, .copy_blocks = 8, .staging = 2 }, try parse("0", "1", "8", "2"));
    for ([_][]const u8{ "", "true", "2", "-1" }) |text| {
        try std.testing.expectError(error.InvalidOffloadExperiment, parse(text, null, null, null));
        try std.testing.expectError(error.InvalidOffloadExperiment, parse(null, text, null, null));
    }
    for ([_][]const u8{ "", "0", "1025", "-1", "+8", "8x", "9999999999999999999999999999" }) |text| {
        try std.testing.expectError(error.InvalidOffloadExperiment, parse(null, null, text, null));
    }
    for ([_][]const u8{ "", "9", "-1", "1x" }) |text| {
        try std.testing.expectError(error.InvalidOffloadExperiment, parse(null, null, null, text));
    }
}

pub fn timingFromEnv() !bool {
    const requested = try boolean(env("TF_FLASHNEXT_EXL3_OFFLOAD_TIMING"), false);
    if (requested and !try boolean(env("TF_FLASHNEXT_CACHE_TELEMETRY"), false)) return error.OffloadTimingNeedsTelemetry;
    if (requested and !try boolean(env("TF_FLASHNEXT_EXL3_LAYER_TELEMETRY"), false)) return error.OffloadTimingNeedsLayerTelemetry;
    return requested;
}
