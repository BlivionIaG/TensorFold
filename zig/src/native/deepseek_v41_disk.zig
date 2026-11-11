//! DeepSeek V4.1's kept states on disk (--learn-conversations): a Snapshot's image, copies out and in, and its identity.
const std = @import("std");
const api = @import("engine_api");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const Backend = tf.deepseek_v41_backend.Backend;
const rows = api.prompt_cache.rows;
const Allocator = std.mem.Allocator;
const Snapshot = ds.Model.Snapshot;
const layout = ds.kv.cache.pool_layout;
const kv_image = ds.kv.image;

/// Prompt tokens a second the store assumes before a pass of this server has measured one (baselines: 0.9k on the M5).
pub const seed_prefill_rate = 900.0;

/// The store's view of a kept state at `at` tokens: its own words (the Engram history and the taps cursor), the window
/// rings with the compressor carry, the taps ring, then each owner's pool rows and index rows by position.
pub const image: rows.Image = .{ .parts = parts, .gather = gather, .blank = blank, .scatter = scatter };

/// Four history words and five taps cursor words.
const header_bytes = 9 * 4;

fn model(ptr: *anyopaque) *ds.Model {
    const b: *Backend = @ptrCast(@alignCast(ptr));
    return b.model;
}

fn parts(ptr: *anyopaque, at: u32, out: *std.ArrayList(rows.image.Part), a: Allocator) anyerror!void {
    const m = model(ptr);
    try out.append(a, .{ .bytes = header_bytes });
    try out.append(a, .{ .bytes = m.caches[0].plan.mutableBytes() });
    try out.append(a, .{ .bytes = m.taps.length() });
    for (layout.ratios) |ratio| for (layout.strides) |stride| {
        try out.append(a, .{ .bytes = kv_image.rowsAt(at, ratio) * stride, .row = @intCast(stride), .per = ratio });
    };
}

/// The words of a state that are not buffers, in image order.
fn headerOut(s: *const Snapshot) [header_bytes]u8 {
    var buf: [header_bytes]u8 = undefined;
    const words = [9]u32{ s.history[0], s.history[1], s.history[2], s.history[3], s.taps_row, s.taps_n, s.taps_rows_last, s.taps_rows_valid, s.taps_base };
    for (words, 0..) |w, i| std.mem.writeInt(u32, buf[i * 4 ..][0..4], w, .little);
    return buf;
}

fn headerIn(s: *Snapshot, buf: *const [header_bytes]u8) void {
    var w: [9]u32 = undefined;
    for (&w, 0..) |*v, i| v.* = std.mem.readInt(u32, buf[i * 4 ..][0..4], .little);
    s.history = w[0..4].*;
    s.taps_row = w[4];
    s.taps_n = w[5];
    s.taps_rows_last = w[6];
    s.taps_rows_valid = w[7];
    s.taps_base = w[8];
}

/// Move the bytes of each range between `buf` (back to back) and the state's image; the words go through the header.
fn move(s: *Snapshot, ranges: []const rows.image.Range, buf: []u8, into_state: bool) void {
    var hdr = headerOut(s);
    kv_image.move(layout, .{ .header = &hdr, .live = s.live, .taps = s.taps, .storage = s.storage, .capacity = s.capacity, .position = s.position }, ranges, buf, into_state);
    if (into_state) headerIn(s, &hdr);
}

fn gather(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, out: []u8) anyerror!void {
    move(@ptrCast(@alignCast(saved)), ranges, out, false);
}

/// A state for `at` tokens, laid out as the image names it and filled by `scatter`, then restored as any other.
fn blank(ptr: *anyopaque, at: u32) anyerror!api.prompt_cache.Saved {
    const s = try model(ptr).snapshotBlank(at);
    s.position = at;
    return s;
}

fn scatter(_: *anyopaque, saved: api.prompt_cache.Saved, ranges: []const rows.image.Range, in: []const u8) anyerror!void {
    move(@ptrCast(@alignCast(saved)), ranges, @constCast(in), true);
}

/// What kept states read under: this binary, the checkpoint's files, the chunk grid, the OS build and chip, the profile.
pub fn identity(a: Allocator, io: std.Io, dir: []const u8, m: *ds.Model) !u64 {
    var h = std.hash.Wyhash.init(0x6473);
    try @import("sys").file_identity.current(&h, a, io);
    try @import("disk_identity.zig").checkpoint(&h, a, io, dir);
    const plan = m.caches[0].plan;
    for ([_]u64{ plan.layers, m.workspace.plan.rows }) |x| h.update(std.mem.asBytes(&x));
    @import("disk_identity.zig").machine(&h, std.mem.span(m.device.name()));
    @import("disk_identity.zig").profile(&h, .{ .family = "deepseek_v41", .backend = "metal" });
    return h.final();
}
