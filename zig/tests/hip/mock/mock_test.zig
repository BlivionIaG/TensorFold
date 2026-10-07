//! The HIP runtime against the stand-in library: admission, copies, fills, ranges, launches and the GPU table.

const std = @import("std");
const hip = @import("hip");
const libs = @import("mock_libs");

test "the driver admits a library with every entry point, and none without one" {
    var d = try hip.Driver.openPath(libs.full);
    defer d.close();
    try std.testing.expectEqual(@as(c_int, 1), try d.deviceCount());
    try std.testing.expectError(error.MissingSymbol, hip.Driver.openPath(libs.missing));
    try std.testing.expectError(error.DriverUnavailable, hip.Driver.openPath("libamdhip64-absent.so"));
}

test "copies and fills round-trip, and out of range is refused" {
    var d = try hip.Driver.openPath(libs.full);
    defer d.close();
    var ctx = try hip.Context.init(&d, 0);
    defer ctx.deinit();
    var b = try hip.DeviceBuffer.fromHost(&d, "abcdefgh");
    try b.upload(4, "WXYZ");
    var out: [8]u8 = undefined;
    try b.download(0, &out);
    try std.testing.expectEqualStrings("abcdWXYZ", &out);
    try std.testing.expectError(error.Invalid, b.upload(6, "WXYZ"));
    try std.testing.expectError(error.Invalid, b.download(9, out[0..0]));
    try b.fill32(0x01020304, null);
    try b.download(0, &out);
    try std.testing.expectEqualSlices(u8, &[_]u8{ 4, 3, 2, 1, 4, 3, 2, 1 }, &out);
    var other = try hip.DeviceBuffer.alloc(&d, 8);
    try other.copyFrom(0, b.ptr, 8, null);
    try other.fill8(7, null);
    try other.download(0, &out);
    try std.testing.expectEqualSlices(u8, &@as([8]u8, @splat(7)), &out);
    other.free();
    b.free();
    var host = try hip.HostBuffer.allocMapped(&d, 16);
    defer host.free();
    try std.testing.expectEqual(@intFromPtr(host.bytes.ptr), try host.device());
    try std.testing.expectError(error.Invalid, hip.HostBuffer.alloc(&d, 0));
}

test "the GPU's caps come from its gfx name, and a GPU outside the table is refused" {
    var d = try hip.Driver.openPath(libs.full);
    defer d.close();
    var ctx = try hip.Context.init(&d, 0);
    defer ctx.deinit();
    try std.testing.expectEqual(hip.caps.Matrix.wmma11, (try ctx.caps()).matrix);
    var other = try hip.Driver.openPath(libs.unknown);
    defer other.close();
    var unknown = try hip.Context.init(&other, 0);
    defer unknown.deinit();
    try std.testing.expectError(error.Invalid, unknown.caps());
}

test "a launch hands HIP its geometry and packed arguments, and a bad pack or grid never reaches it" {
    var d = try hip.Driver.openPath(libs.full);
    defer d.close();
    const lastLaunch = d.lib.lookup(*const fn () callconv(.c) *const [4]u64, "mockLastLaunch").?;
    const f: hip.Function = .{ .d = &d, .handle = null };
    const s: hip.Stream = .{ .d = &d, .handle = null };
    var args: hip.Args = .{};
    args.add(@as(u64, 0x1122334455667788));
    args.add(@as(u32, 7));
    try hip.launch.launch(f, .{ .grid = .{ .x = 3 }, .block = .{ .x = 64 } }, s, &args);
    try std.testing.expectEqualSlices(u64, &.{ 3, 64, 0x1122334455667788, 7 }, lastLaunch());
    var full: hip.Args = .{};
    for (0..hip.Args.max_args + 1) |_| full.add(@as(u32, 9));
    try std.testing.expectError(error.Invalid, hip.launch.launch(f, .{ .grid = .{}, .block = .{} }, s, &full));
    try std.testing.expectError(error.Invalid, hip.launch.launch(f, .{ .grid = .{ .x = 1 << 31 }, .block = .{ .x = 4 } }, s, &args));
    try std.testing.expectEqual(@as(u64, 3), lastLaunch()[0]);
}
