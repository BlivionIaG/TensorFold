const std = @import("std");
const driver = @import("driver.zig");
const fixtures = @import("hip_fixtures");
const Runtime = @import("runtime.zig").Runtime;
const Context = @import("context.zig").Context;
const Stream = @import("stream.zig").Stream;
const DeviceBuffer = @import("memory.zig").DeviceBuffer;
const HostBuffer = @import("memory.zig").HostBuffer;

test {
    _ = @import("affine.zig");
    _ = @import("args.zig");
    _ = @import("code_object.zig");
    _ = @import("abi.zig");
    _ = @import("runtime.zig");
    _ = @import("launch.zig");
    _ = @import("context.zig").Context.init;
    _ = @import("stream.zig").Stream.init;
    _ = @import("memory.zig").DeviceBuffer.alloc;
    _ = @import("module.zig").Module.load;
    _ = @import("module.zig").Module.loadForArchitecture;
    _ = @import("launch.zig").launch;
}

test "admission-only fixture cannot satisfy complete runtime ABI" {
    try std.testing.expectError(error.MissingSymbol, @import("runtime.zig").Runtime.openPath(fixtures.success));
}

test "real dynamic loading admits the mock HIP ABI" {
    var d = try driver.Driver.openPath(fixtures.success);
    defer d.close();
    try std.testing.expectEqual(@as(c_int, 2), try d.deviceCount());
}

test "real dynamic loading refuses missing symbols and failed initialization" {
    try std.testing.expectError(error.MissingSymbol, driver.Driver.openPath(fixtures.missing));
    try std.testing.expectError(error.HipFailed, driver.Driver.openPath(fixtures.failed));
}

test "the complete mock opens the runtime and a table missing one entry point does not" {
    var r = try Runtime.openPath(fixtures.runtime);
    r.close();
    try std.testing.expectError(error.MissingSymbol, Runtime.openPath(fixtures.runtime_missing));
}

test "copies and fills round-trip through the mock runtime and out-of-range spans are refused" {
    var r = try Runtime.openPath(fixtures.runtime);
    defer r.close();
    var ctx = try Context.init(&r, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&r);
    defer stream.deinit();
    var b = try DeviceBuffer.alloc(&r, 8);
    defer b.free();
    var out: [8]u8 = undefined;
    try b.upload(0, "abcdefgh");
    try b.upload(4, "WXYZ");
    try b.download(0, &out);
    try std.testing.expectEqualStrings("abcdWXYZ", &out);
    try std.testing.expectError(error.Invalid, b.upload(6, "WXYZ"));
    try std.testing.expectError(error.Invalid, b.download(9, out[0..0]));
    try b.fill8(7);
    try b.download(0, &out);
    try std.testing.expectEqualSlices(u8, &@as([8]u8, @splat(7)), &out);
    var src = try HostBuffer.alloc(&r, 8);
    defer src.free();
    var dst = try HostBuffer.alloc(&r, 8);
    defer dst.free();
    @memcpy(src.bytes, "pinned!!");
    try b.uploadAsync(0, src, stream);
    try b.downloadAsync(0, dst, stream);
    try stream.synchronize();
    try std.testing.expectEqualStrings("pinned!!", dst.bytes);
    try b.fill8Async(3, stream);
    try b.downloadAsync(0, dst, stream);
    try stream.synchronize();
    try std.testing.expectEqualSlices(u8, &@as([8]u8, @splat(3)), dst.bytes);
    try std.testing.expectError(error.Invalid, b.uploadAsync(1, src, stream));
}

test "HIP failures through the mock runtime keep their kind" {
    std.testing.log_level = .err;
    var r = try Runtime.openPath(fixtures.runtime);
    defer r.close();
    try std.testing.expectError(error.HipFailed, Context.init(&r, 2));
    try std.testing.expectError(error.OutOfDeviceMemory, DeviceBuffer.alloc(&r, 1 << 41));
    const image align(8) = [_]u8{1};
    var m = try @import("module.zig").Module.load(&r, &image);
    defer m.unload();
    try std.testing.expectError(error.NotFound, m.function("absent"));
}
