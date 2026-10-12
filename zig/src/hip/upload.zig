//! The HIP backend's side of a format's `load`: host tensors onto the device, kept for the model to free.

const std = @import("std");
const quant = @import("core").quant;
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");

/// Device memory of one model: every buffer a format uploads is kept in `buffers`, which the model frees.
pub const Upload = struct {
    r: *const runtime.Runtime,
    buffers: *std.ArrayList(memory.DeviceBuffer),
    gpa: std.mem.Allocator,
    /// Pinned bytes each tensor goes up through: HIP would pin a mapped source's pages itself and keep them resident.
    staging: ?*const memory.HostBuffer = null,

    /// What a format's `upload` takes.
    pub fn uploader(u: *const Upload) quant.Uploader {
        return .{ .ctx = @ptrCast(@constCast(u)), .put = put };
    }

    fn put(ctx: *anyopaque, t: quant.Tensor) anyerror!quant.Buf {
        const u: *const Upload = @ptrCast(@alignCast(ctx));
        var b = if (u.staging) |st| try staged(u.r, st.bytes, t.bytes) else try memory.DeviceBuffer.fromHost(u.r, t.bytes);
        errdefer b.free();
        try u.buffers.append(u.gpa, b);
        return .{ .ptr = @intFromPtr(b.ptr), .len = b.len, .dtype = t.dtype, .rank = t.rank, .shape = t.shape };
    }
};

/// `bytes` on the device through `pin`, a piece at a time.
fn staged(r: *const runtime.Runtime, pin: []u8, bytes: []const u8) runtime.Error!memory.DeviceBuffer {
    var b = try memory.DeviceBuffer.alloc(r, bytes.len);
    errdefer b.free();
    var at: usize = 0;
    while (at < bytes.len) : (at += pin.len) {
        const n = @min(pin.len, bytes.len - at);
        @memcpy(pin[0..n], bytes[at..][0..n]);
        try b.upload(at, pin[0..n]);
    }
    return b;
}
