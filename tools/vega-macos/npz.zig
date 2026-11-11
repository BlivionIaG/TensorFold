//! A reader of NumPy `.npz` archives held in memory: a zip (stored or deflated members, zip64 sizes and offsets) of `.npy` files, what
//! `np.savez` / `np.savez_compressed` write. `Npz.array` hands back the named member's decoded `.npy` bytes for `npy.parse`.
const std = @import("std");

pub const Error = error{ NotZip, Truncated, Unsupported, BadMember, NoMember, OutOfMemory };

const eocd_sig = "PK\x05\x06";
const eocd64_sig = "PK\x06\x06";
const locator64_sig = "PK\x06\x07";
const central_sig = "PK\x01\x02";
const local_sig = "PK\x03\x04";
const eocd_len = 22;
const central_len = 46;
const local_len = 30;

pub const Method = enum(u16) { store = 0, deflate = 8, _ };

pub const Entry = struct {
    name: []const u8,
    method: Method,
    compressed_size: u64,
    size: u64,
    crc32: u32,
    local_offset: u64,
};

pub const Npz = struct {
    bytes: []const u8,
    entries: []Entry,

    pub fn deinit(self: *Npz, a: std.mem.Allocator) void {
        a.free(self.entries);
        self.* = undefined;
    }

    /// The entry called `name` or `name.npy` (np.load's lookup), or null.
    pub fn find(self: Npz, name: []const u8) ?Entry {
        for (self.entries) |e| if (std.mem.eql(u8, e.name, name)) return e;
        for (self.entries) |e| {
            if (e.name.len == name.len + 4 and std.mem.startsWith(u8, e.name, name) and std.mem.eql(u8, e.name[name.len..], ".npy")) return e;
        }
        return null;
    }

    /// The decoded bytes of member `name` (a whole `.npy` file), owned by `a`; the CRC-32 is checked.
    pub fn array(self: Npz, a: std.mem.Allocator, name: []const u8) Error![]u8 {
        const e = self.find(name) orelse return error.NoMember;
        const at = std.math.cast(usize, e.local_offset) orelse return error.Truncated;
        if (at > self.bytes.len or self.bytes.len - at < local_len) return error.Truncated;
        const hdr = self.bytes[at..];
        if (!std.mem.eql(u8, hdr[0..4], local_sig)) return error.BadMember;
        const skip = @as(usize, rd(u16, hdr[26..])) + rd(u16, hdr[28..]);
        const start = at + local_len + skip;
        const csize = std.math.cast(usize, e.compressed_size) orelse return error.Truncated;
        if (start > self.bytes.len or self.bytes.len - start < csize) return error.Truncated;
        const raw = self.bytes[start..][0..csize];
        const size = std.math.cast(usize, e.size) orelse return error.Unsupported;
        const out = try a.alloc(u8, size);
        errdefer a.free(out);
        switch (e.method) {
            .store => {
                if (csize != size) return error.BadMember;
                @memcpy(out, raw);
            },
            .deflate => {
                var in: std.Io.Reader = .fixed(raw);
                var d: std.compress.flate.Decompress = .init(&in, .raw, &.{});
                var w: std.Io.Writer = .fixed(out);
                const got = d.reader.streamRemaining(&w) catch return error.BadMember;
                if (got != size) return error.BadMember;
            },
            _ => return error.Unsupported,
        }
        if (std.hash.Crc32.hash(out) != e.crc32) return error.BadMember;
        return out;
    }
};

fn rd(comptime T: type, b: []const u8) T {
    return std.mem.readInt(T, b[0..@sizeOf(T)], .little);
}

/// Index the archive `bytes`; the names in the result borrow from `bytes`.
pub fn open(a: std.mem.Allocator, bytes: []const u8) Error!Npz {
    const tail = @min(bytes.len, eocd_len + 65535);
    const base = bytes.len - tail;
    const at = std.mem.findLast(u8, bytes[base..], eocd_sig) orelse return error.NotZip;
    const eocd = base + at;
    if (bytes.len - eocd < eocd_len) return error.Truncated;
    const e = bytes[eocd..];
    var count: u64 = rd(u16, e[10..]);
    var cd_size: u64 = rd(u32, e[12..]);
    var cd_offset: u64 = rd(u32, e[16..]);
    if (count == 0xffff or cd_size == 0xffffffff or cd_offset == 0xffffffff) {
        if (eocd < 20 or !std.mem.eql(u8, bytes[eocd - 20 ..][0..4], locator64_sig)) return error.Unsupported;
        const r64 = std.math.cast(usize, rd(u64, bytes[eocd - 20 + 8 ..])) orelse return error.Truncated;
        if (r64 > bytes.len or bytes.len - r64 < 56 or !std.mem.eql(u8, bytes[r64..][0..4], eocd64_sig)) return error.Unsupported;
        count = rd(u64, bytes[r64 + 32 ..]);
        cd_size = rd(u64, bytes[r64 + 40 ..]);
        cd_offset = rd(u64, bytes[r64 + 48 ..]);
    }
    var pos = std.math.cast(usize, cd_offset) orelse return error.Truncated;
    const cd_end = std.math.add(usize, pos, std.math.cast(usize, cd_size) orelse return error.Truncated) catch return error.Truncated;
    if (cd_end > bytes.len) return error.Truncated;
    const n = std.math.cast(usize, count) orelse return error.Truncated;
    if (n > cd_size / central_len) return error.Truncated;
    const entries = try a.alloc(Entry, n);
    errdefer a.free(entries);
    for (entries) |*out| {
        if (cd_end - pos < central_len or !std.mem.eql(u8, bytes[pos..][0..4], central_sig)) return error.BadMember;
        const h = bytes[pos..];
        const flags = rd(u16, h[8..]);
        if (flags & 1 != 0) return error.Unsupported; // encrypted
        const name_len = rd(u16, h[28..]);
        const extra_len = rd(u16, h[30..]);
        const comment_len = rd(u16, h[32..]);
        const total = central_len + @as(usize, name_len) + extra_len + comment_len;
        if (cd_end - pos < total) return error.Truncated;
        out.* = .{
            .name = h[central_len..][0..name_len],
            .method = @fromBackingInt(@intCast(rd(u16, h[10..]))),
            .crc32 = rd(u32, h[16..]),
            .compressed_size = rd(u32, h[20..]),
            .size = rd(u32, h[24..]),
            .local_offset = rd(u32, h[42..]),
        };
        // the zip64 extra field carries, in this order, only the values the header holds as 0xffffffff
        var extra = h[central_len + name_len ..][0..extra_len];
        while (extra.len >= 4) {
            const id = rd(u16, extra);
            const len = rd(u16, extra[2..]);
            if (extra.len < 4 + len) return error.BadMember;
            if (id == 1) {
                var f = extra[4..][0..len];
                inline for (.{ &out.size, &out.compressed_size, &out.local_offset }) |field| {
                    if (field.* == 0xffffffff) {
                        if (f.len < 8) return error.BadMember;
                        field.* = rd(u64, f);
                        f = f[8..];
                    }
                }
            }
            extra = extra[4 + len ..];
        }
        pos += total;
    }
    return .{ .bytes = bytes, .entries = entries };
}
