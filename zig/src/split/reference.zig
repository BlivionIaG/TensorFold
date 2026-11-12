//! The unsplit run's prompt, greedy tokens and the token it predicts at every position, for a split run to compare to.
const std = @import("std");

const c = std.c;
const magic: u32 = 0x5245_4631;

pub const Reference = struct {
    layers: u32,
    hidden: u32,
    prompt: []u32,
    /// The greedy continuation of the prompt, free running.
    want: []u32,
    /// `preds[i]`: the greedy token after the sequence `prompt ++ want[0 .. want.len - 1]` up to position `i`.
    preds: []u32,

    pub fn deinit(r: Reference, a: std.mem.Allocator) void {
        a.free(r.prompt);
        a.free(r.want);
        a.free(r.preds);
    }

    /// The positions' input tokens: the prompt, then every generated token but the last.
    pub fn sequence(r: Reference, a: std.mem.Allocator) ![]u32 {
        const seq = try a.alloc(u32, r.prompt.len + r.want.len - 1);
        @memcpy(seq[0..r.prompt.len], r.prompt);
        @memcpy(seq[r.prompt.len..], r.want[0 .. r.want.len - 1]);
        return seq;
    }
};

pub const Error = error{ Io, BadFile, OutOfMemory };

fn writeFile(path: [:0]const u8, bytes: []const u8) Error!void {
    const fd = c.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(c.mode_t, 0o644));
    if (fd < 0) return error.Io;
    defer _ = c.close(fd);
    var at: usize = 0;
    while (at < bytes.len) {
        const n = c.write(fd, bytes[at..].ptr, bytes.len - at);
        if (n <= 0) return error.Io;
        at += @intCast(n);
    }
}

fn readFile(a: std.mem.Allocator, path: [:0]const u8) Error![]u8 {
    const fd = c.open(path, .{ .ACCMODE = .RDONLY }, @as(c.mode_t, 0));
    if (fd < 0) return error.Io;
    defer _ = c.close(fd);
    var all: std.ArrayList(u8) = .empty;
    errdefer all.deinit(a);
    var buf: [64 * 1024]u8 = undefined;
    while (true) {
        const n = c.read(fd, &buf, buf.len);
        if (n < 0) return error.Io;
        if (n == 0) break;
        try all.appendSlice(a, buf[0..@intCast(n)]);
    }
    return all.toOwnedSlice(a);
}

/// Writes `r` to `path`.
pub fn save(a: std.mem.Allocator, path: [:0]const u8, r: Reference) Error!void {
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(a);
    const head = [_]u32{ magic, r.layers, r.hidden, @intCast(r.prompt.len), @intCast(r.want.len), @intCast(r.preds.len) };
    try bytes.appendSlice(a, std.mem.sliceAsBytes(&head));
    for ([_][]const u32{ r.prompt, r.want, r.preds }) |part| try bytes.appendSlice(a, std.mem.sliceAsBytes(part));
    try writeFile(path, bytes.items);
}

/// Reads what `save` wrote.
pub fn load(a: std.mem.Allocator, path: [:0]const u8) Error!Reference {
    const all = try readFile(a, path);
    defer a.free(all);
    const head_len = 6 * 4;
    if (all.len < head_len) return error.BadFile;
    var head: [6]u32 = undefined;
    @memcpy(std.mem.sliceAsBytes(&head), all[0..head_len]);
    if (head[0] != magic) return error.BadFile;
    const counts = [3]usize{ head[3], head[4], head[5] };
    if (all.len != head_len + 4 * (counts[0] + counts[1] + counts[2]) or counts[1] == 0) return error.BadFile;
    var parts: [3][]u32 = undefined;
    var made: usize = 0;
    errdefer for (parts[0..made]) |p| a.free(p);
    var at: usize = head_len;
    for (&parts, counts) |*p, n| {
        p.* = try a.alloc(u32, n);
        made += 1;
        @memcpy(std.mem.sliceAsBytes(p.*), all[at..][0 .. 4 * n]);
        at += 4 * n;
    }
    return .{ .layers = head[1], .hidden = head[2], .prompt = parts[0], .want = parts[1], .preds = parts[2] };
}

/// The logits the reference run produced at every position, as f32 rows of `vocab` values (a file of its own).
pub fn saveLogits(path: [:0]const u8, rows: []const f32) Error!void {
    return writeFile(path, std.mem.sliceAsBytes(rows));
}

pub fn loadLogits(a: std.mem.Allocator, path: [:0]const u8) Error![]f32 {
    const bytes = try readFile(a, path);
    errdefer a.free(bytes);
    if (bytes.len % 4 != 0) return error.BadFile;
    const out = try a.alloc(f32, bytes.len / 4);
    @memcpy(std.mem.sliceAsBytes(out), bytes);
    a.free(bytes);
    return out;
}

test "a reference survives a file" {
    const a = std.testing.allocator;
    var prompt = [_]u32{ 1, 2, 3 };
    var want = [_]u32{ 4, 5 };
    var preds = [_]u32{ 9, 8, 7, 4 };
    const path = "/tmp/tf-split-reference-test.bin";
    try save(a, path, .{ .layers = 32, .hidden = 2560, .prompt = &prompt, .want = &want, .preds = &preds });
    defer _ = c.unlink(path);
    const r = try load(a, path);
    defer r.deinit(a);
    try std.testing.expectEqual(@as(u32, 2560), r.hidden);
    try std.testing.expectEqualSlices(u32, &want, r.want);
    try std.testing.expectEqualSlices(u32, &preds, r.preds);
    const seq = try r.sequence(a);
    defer a.free(seq);
    try std.testing.expectEqualSlices(u32, &.{ 1, 2, 3, 4 }, seq);
}
