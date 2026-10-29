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
    last_marker_ns: u64 = 0,

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

    pub fn initialize(timing: *Timing, stream: cuda.Stream) !void {
        inline for (.{ "start", "copy_start", "copy_end", "held_end", "joined", "end" }) |field| {
            try @field(timing, field).recordExternal(stream);
        }
    }

    pub fn initializeAfterConsumers(timing: *Timing, epoch: cuda.Event) !void {
        timing.last_marker_ns = try nanoseconds(try cuda.Event.elapsedMs(epoch, timing.end));
    }

    pub fn sampleAfterConsumers(timing: *Timing, overlap: bool, epoch: cuda.Event) !?Sample {
        if (!timing.recorded) return null;
        const marker = try nanoseconds(try cuda.Event.elapsedMs(epoch, timing.end));
        if (marker == timing.last_marker_ns) return null;
        if (marker < timing.last_marker_ns) return error.InvalidOffloadTiming;
        const result: Sample = .{
            .service_ns = try nanoseconds(try cuda.Event.elapsedMs(timing.start, timing.end)),
            .copy_ns = try nanoseconds(try cuda.Event.elapsedMs(timing.copy_start, timing.copy_end)),
            .exposed_copy_wait_ns = if (overlap) try nanoseconds(try cuda.Event.elapsedMs(timing.held_end, timing.joined)) else try nanoseconds(try cuda.Event.elapsedMs(timing.copy_start, timing.copy_end)),
        };
        timing.last_marker_ns = marker;
        return result;
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

test "exl3: timing owns exactly six events and unwinds every constructor prefix" {
    const Fake = struct {
        var created: usize = 0;
        var destroyed: usize = 0;
        var fail_at: usize = 0;
        var marker_ms: f32 = 1;
        fn create(out: *cuda.abi.Event, flags: c_uint) callconv(.c) c_int {
            if (flags != 0) return cuda.abi.error_not_ready;
            if (created == fail_at) return cuda.abi.error_not_ready;
            created += 1;
            out.* = @ptrFromInt(created);
            return 0;
        }
        fn destroy(_: cuda.abi.Event) callconv(.c) c_int {
            destroyed += 1;
            return 0;
        }
        fn elapsed(out: *f32, start: cuda.abi.Event, end: cuda.abi.Event) callconv(.c) c_int {
            out.* = if (@intFromPtr(start.?) == 1) marker_ms else @as(f32, @floatFromInt(@intFromPtr(end.?) - @intFromPtr(start.?))) / 10;
            return 0;
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuEventCreate = Fake.create;
    driver.api.cuEventDestroy_v2 = Fake.destroy;
    driver.api.cuEventElapsedTime = Fake.elapsed;
    for (0..6) |fail_at| {
        Fake.created = 0;
        Fake.destroyed = 0;
        Fake.fail_at = fail_at;
        try std.testing.expectError(error.NotReady, Timing.init(&driver));
        try std.testing.expectEqual(fail_at, Fake.destroyed);
    }
    Fake.created = 0;
    Fake.destroyed = 0;
    Fake.fail_at = 7;
    var epoch = try cuda.Event.init(&driver, true);
    var timing = try Timing.init(&driver);
    Fake.marker_ms = 1;
    try timing.initializeAfterConsumers(epoch);
    try std.testing.expectEqual(@as(?Sample, null), try timing.sampleAfterConsumers(true, epoch));
    timing.recorded = true;
    try std.testing.expectEqual(@as(?Sample, null), try timing.sampleAfterConsumers(true, epoch));
    Fake.marker_ms = 2;
    const sample = (try timing.sampleAfterConsumers(true, epoch)).?;
    try std.testing.expectEqual(@as(u64, 500_000), sample.service_ns);
    try std.testing.expectEqual(@as(u64, 100_000), sample.copy_ns);
    try std.testing.expectEqual(@as(u64, 100_000), sample.exposed_copy_wait_ns);
    try std.testing.expectEqual(@as(?Sample, null), try timing.sampleAfterConsumers(true, epoch));
    Fake.marker_ms = 1;
    try std.testing.expectError(error.InvalidOffloadTiming, timing.sampleAfterConsumers(true, epoch));
    timing.deinit();
    epoch.deinit();
    try std.testing.expectEqual(@as(usize, 7), Fake.destroyed);
}
