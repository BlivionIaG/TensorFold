const std = @import("std");
const cuda = @import("cuda");

pub const Sample = @import("../model/exl3_telemetry.zig").LayerTiming;

pub const Timing = struct {
    start: cuda.Event,
    copy_start: cuda.Event,
    copy_end: cuda.Event,
    held_end: cuda.Event,
    joined: cuda.Event,
    end: cuda.Event,
    recorded: bool = false,

    pub fn init(driver: *const cuda.Driver) !Timing {
        var start = try cuda.Event.init(driver, true);
        errdefer start.deinit();
        var copy_start = try cuda.Event.init(driver, true);
        errdefer copy_start.deinit();
        var copy_end = try cuda.Event.init(driver, true);
        errdefer copy_end.deinit();
        var held_end = try cuda.Event.init(driver, true);
        errdefer held_end.deinit();
        var joined = try cuda.Event.init(driver, true);
        errdefer joined.deinit();
        const end = try cuda.Event.init(driver, true);
        return .{ .start = start, .copy_start = copy_start, .copy_end = copy_end, .held_end = held_end, .joined = joined, .end = end };
    }

    pub fn deinit(timing: *Timing) void {
        timing.end.deinit();
        timing.joined.deinit();
        timing.held_end.deinit();
        timing.copy_end.deinit();
        timing.copy_start.deinit();
        timing.start.deinit();
        timing.* = undefined;
    }

    pub fn sampleAfterConsumers(timing: Timing, overlap: bool) !?Sample {
        if (!timing.recorded) return null;
        return .{
            .service_ns = try nanoseconds(try cuda.Event.elapsedMs(timing.start, timing.end)),
            .copy_ns = try nanoseconds(try cuda.Event.elapsedMs(timing.copy_start, timing.copy_end)),
            .exposed_copy_wait_ns = if (overlap) try nanoseconds(try cuda.Event.elapsedMs(timing.held_end, timing.joined)) else try nanoseconds(try cuda.Event.elapsedMs(timing.copy_start, timing.copy_end)),
        };
    }
};

fn nanoseconds(ms: f32) !u64 {
    if (!std.math.isFinite(ms) or ms < 0 or ms >= 0x1p44) return error.InvalidOffloadTiming;
    return @intFromFloat(@as(f64, ms) * 1e6);
}

test "exl3: offload event times reject nonfinite negative and overflow" {
    try std.testing.expectEqual(@as(u64, 1_500_000), try nanoseconds(1.5));
    for ([_]f32{ -1, std.math.inf(f32), std.math.nan(f32), 0x1p44 }) |value| {
        try std.testing.expectError(error.InvalidOffloadTiming, nanoseconds(value));
    }
}
