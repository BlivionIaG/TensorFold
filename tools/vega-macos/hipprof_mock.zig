//! A mock HIP runtime for hipprof's check: counts launches and syncs, and keeps a clock of 20 us a recorded event.
const std = @import("std");

const U = usize;
const event_us = 20;
var launches = std.atomic.Value(U).init(0);
var syncs = std.atomic.Value(U).init(0);
var graphs = std.atomic.Value(U).init(0);
var functions = std.atomic.Value(U).init(0);
var events = std.atomic.Value(U).init(0);
var clock = std.atomic.Value(u32).init(0);
var times: [16384]u32 = undefined;

export fn hipmock_launches() U {
    return launches.load(.monotonic);
}

export fn hipmock_syncs() U {
    return syncs.load(.monotonic);
}

export fn hipmock_graphs() U {
    return graphs.load(.monotonic);
}

export fn hipModuleGetFunction(out: *U, _: U, _: [*:0]const u8) callconv(.c) c_int {
    out.* = 0x1000 + 16 * functions.fetchAdd(1, .monotonic);
    return 0;
}

export fn hipModuleLaunchKernel() callconv(.c) c_int {
    _ = launches.fetchAdd(1, .monotonic);
    return 0;
}

export fn hipStreamSynchronize(_: U) callconv(.c) c_int {
    _ = syncs.fetchAdd(1, .monotonic);
    return 0;
}

export fn hipStreamBeginCapture() callconv(.c) c_int {
    return 0;
}

export fn hipStreamEndCapture() callconv(.c) c_int {
    return 0;
}

export fn hipGraphLaunch() callconv(.c) c_int {
    _ = graphs.fetchAdd(1, .monotonic);
    return 0;
}

export fn hipEventCreate(out: *U) callconv(.c) c_int {
    out.* = 1 + events.fetchAdd(1, .monotonic);
    return 0;
}

export fn hipEventRecord(ev: U, _: U) callconv(.c) c_int {
    times[ev] = clock.fetchAdd(event_us, .monotonic);
    return 0;
}

export fn hipEventSynchronize(_: U) callconv(.c) c_int {
    return 0;
}

export fn hipEventElapsedTime(ms: *f32, e0: U, e1: U) callconv(.c) c_int {
    ms.* = @as(f32, @floatFromInt(times[e1] -% times[e0])) / 1000.0;
    return 0;
}

export fn hipMemcpyDtoH(dst: [*]u8, src: [*]const u8, size: U) callconv(.c) c_int {
    @memcpy(dst[0..size], src[0..size]);
    return 0;
}

export fn hipMalloc(out: *U, size: U) callconv(.c) c_int {
    out.* = 0x4000_0000 + size;
    return 0;
}

export fn hipDeviceGetAttribute(out: *c_int, attr: c_int, device: c_int) callconv(.c) c_int {
    out.* = attr * 2 + device;
    return 0;
}
