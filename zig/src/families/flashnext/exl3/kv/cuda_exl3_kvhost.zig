//! The host tier of the Flash Next pool (TF_FLASHNEXT_KV_HOST=MiB): canonical K/V pages in pinned host RAM, device frames.
const std = @import("std");
const cuda = @import("cuda");
const kv = @import("core").kv_pool;
const pages = @import("../../cuda_pages.zig");

/// TF_FLASHNEXT_KV_HOST=n: n MiB of pinned host RAM hold the K/V plane of every logical page (unset or 0: no tier).
pub fn hostMiB() !usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_HOST") orelse return 0;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvHost;
}

/// TF_FLASHNEXT_KV_FRAMES=n: at most n device frames of the K/V plane (unset: the pages the plan gave).
pub fn framesCap() !?usize {
    const v = std.c.getenv("TF_FLASHNEXT_KV_FRAMES") orelse return null;
    return std.fmt.parseInt(usize, std.mem.span(v), 10) catch return error.BadFlashNextKvHost;
}

/// The tier of one pool: logical pages, frames, the pinned host store they are canonical in.
pub const Host = struct {
    tier: kv.tier.Tier,
    store: cuda.HostBuffer,
    kv_page: usize, // bytes of one page's K/V plane

    pub fn create(gpa: std.mem.Allocator, d: *const cuda.Driver, logical: usize, frames: usize, kv_page: usize) !*Host {
        const h = try gpa.create(Host);
        errdefer gpa.destroy(h);
        var store = try cuda.HostBuffer.allocMapped(d, try std.math.mul(usize, logical, kv_page));
        errdefer store.free();
        h.* = .{ .tier = undefined, .store = store, .kv_page = kv_page };
        try h.tier.init(gpa, store.bytes, frames, kv_page);
        return h;
    }

    pub fn destroy(h: *Host, gpa: std.mem.Allocator) void {
        h.tier.deinit();
        h.store.free();
        gpa.destroy(h);
    }

    /// Page `id`'s K/V plane from its frame (in `arena`, frames from its start) into the host store; the caller syncs the
    /// stream, then calls `done`. A page with no frame is already canonical.
    pub fn flush(h: *Host, arena: cuda.DeviceBuffer, id: u32, stream: cuda.Stream) !void {
        const tk = h.tier.resident(id) orelse return;
        try arena.downloadAsync(@as(usize, tk.frame) * h.kv_page, h.tier.host_page(id), stream.handle);
    }

    pub fn done(h: *Host, id: u32) void {
        h.tier.sync(id);
    }
};
