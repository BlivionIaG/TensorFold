//! Property ABI comes from the installed HIP headers through a small C adapter.
const std = @import("std");
const runtime = @import("runtime.zig");
const Properties = *const fn (*anyopaque, c_int) callconv(.c) c_int;
extern fn tf_hip_device_arch(Properties, c_int, [*]u8, usize) c_int;

/// The device's gfx name without its feature modes: "gfx900" for a Vega 10 whose runtime reports "gfx900:xnack-" (the code objects are
/// built for the processor with the modes left open), "gfx1151" as reported on RDNA.
pub fn query(r: *runtime.Runtime, ordinal: c_int, out: *[256]u8) runtime.Error![]const u8 {
    if (ordinal < 0) return error.Invalid;
    const properties = r.lib.lookup(Properties, "hipGetDevicePropertiesR0600") orelse return error.MissingSymbol;
    try runtime.check(tf_hip_device_arch(properties, ordinal, out, out.len));
    const len = std.mem.indexOfScalar(u8, out, 0) orelse return error.Invalid;
    if (len == 0) return error.Invalid;
    const end = std.mem.indexOfScalar(u8, out[0..len], ':') orelse len;
    if (end == 0) return error.Invalid;
    return out[0..end];
}
