//! Host-side n-gram row placement. Page cache and SSD share one gather.
//! A read is only the requested rows: consecutive entries coalesce when they
//! are the next local row of the same shard, and a run never grows past that.
//! `Disk` is the SSD caller: `pread` of those runs, never a map of the shard.

const std = @import("std");

/// Mapped view (`page_cache`) or exact preads (`ssd`). Both call `gather`.
pub const Mode = enum { page_cache, ssd };

pub const Shard = struct {
    file: u32,
    /// Byte offset of row 0 in that file.
    offset: u64,
    rows: u32,
};

pub const Located = struct { file: u32, offset: u64, bytes: u32 };

pub const Read = struct {
    file: u32,
    offset: u64,
    /// Number of bytes. For a coalesced run this is n_rows * row_bytes.
    bytes: u64,
    /// Destination index of the first gathered row in this read.
    dest_row: u32,
    rows: u32,
};

const Hit = struct { index: u32, local: u64, shard: Shard };

fn rowBytes(words_per_row: u32) u64 {
    return @as(u64, words_per_row) * 2;
}

/// First shard whose prefix range contains `row`. Ids are contiguous in shard order.
fn hit(shards: []const Shard, row: u64) error{ RowOutOfRange, EmptyTable }!Hit {
    if (shards.len == 0) return error.EmptyTable;
    var base: u64 = 0;
    var any = false;
    for (shards, 0..) |shard, index| {
        if (shard.rows != 0) any = true;
        if (row >= base and row - base < @as(u64, shard.rows)) {
            return .{ .index = @intCast(index), .local = row - base, .shard = shard };
        }
        if (@as(u64, shard.rows) > std.math.maxInt(u64) - base) break;
        base += @as(u64, shard.rows);
    }
    if (!any) return error.EmptyTable;
    return error.RowOutOfRange;
}

/// Global row ids run through `shards` in order.
/// Byte offset is `shard.offset + local * words_per_row * 2`.
pub fn locate(shards: []const Shard, words_per_row: u32, row: u64) error{ RowOutOfRange, EmptyTable }!Located {
    const found = try hit(shards, row);
    const bytes = rowBytes(words_per_row);
    return .{
        .file = found.shard.file,
        .offset = found.shard.offset + found.local * bytes,
        .bytes = @intCast(bytes),
    };
}

/// One coalesced run starting at `rows[i]`, or one row when the next entry is not the next local row.
fn nextRun(shards: []const Shard, words_per_row: u32, rows: []const u64, i: usize) error{ RowOutOfRange, EmptyTable }!Read {
    const bytes = rowBytes(words_per_row);
    const here = try hit(shards, rows[i]);
    var local = here.local;
    var count: u32 = 1;
    var j = i + 1;
    while (j < rows.len) {
        if (rows[j - 1] == std.math.maxInt(u64) or rows[j] != rows[j - 1] + 1) break;
        if (count == std.math.maxInt(u32)) break;
        const next = try hit(shards, rows[j]);
        if (next.index != here.index or local == std.math.maxInt(u64) or next.local != local + 1) break;
        local = next.local;
        count += 1;
        j += 1;
    }
    return .{
        .file = here.shard.file,
        .offset = here.shard.offset + here.local * bytes,
        .bytes = @as(u64, count) * bytes,
        .dest_row = @intCast(i),
        .rows = count,
    };
}

const Fill = struct { reads: usize, consumed: usize };

/// Plans a prefix of `rows` that fits in `out`. Does not sort and does not read past a run.
fn fill(shards: []const Shard, words_per_row: u32, rows: []const u64, out: []Read) error{ RowOutOfRange, EmptyTable }!Fill {
    var i: usize = 0;
    var n: usize = 0;
    while (i < rows.len and n < out.len) {
        const run = try nextRun(shards, words_per_row, rows, i);
        out[n] = run;
        n += 1;
        i = @as(usize, run.dest_row) + run.rows;
    }
    return .{ .reads = n, .consumed = i };
}

/// Reads a gather must issue, in request order.
/// Coalesce only when `rows[i + 1] == rows[i] + 1` and both rows sit in the same shard.
/// `PlanTooSmall` when `out` cannot hold every read. No whole-shard read is invented.
pub fn planReads(shards: []const Shard, words_per_row: u32, rows: []const u64, out: []Read) error{ RowOutOfRange, EmptyTable, PlanTooSmall }![]Read {
    const planned = try fill(shards, words_per_row, rows, out);
    if (planned.consumed != rows.len) {
        var i = planned.consumed;
        while (i < rows.len) {
            const run = try nextRun(shards, words_per_row, rows, i);
            i = @as(usize, run.dest_row) + run.rows;
        }
        return error.PlanTooSmall;
    }
    return out[0..planned.reads];
}

fn copyRuns(shards: []const Shard, words_per_row: u32, rows: []const u64, reader: anytype, dest: []i16) !void {
    std.debug.assert(dest.len == rows.len * @as(usize, words_per_row));
    var start: usize = 0;
    while (start < rows.len) {
        var buf: [64]Read = undefined;
        const planned = try fill(shards, words_per_row, rows[start..], &buf);
        std.debug.assert(planned.consumed != 0);
        for (buf[0..planned.reads]) |run| {
            const wpr: usize = words_per_row;
            const at = (start + @as(usize, run.dest_row)) * wpr;
            const raw = std.mem.sliceAsBytes(dest[at..][0 .. @as(usize, run.rows) * wpr]);
            std.debug.assert(raw.len == run.bytes);
            try reader.read(run.file, run.offset, raw);
        }
        start += planned.consumed;
    }
}

const FnReader = struct {
    src: *const fn (file: u32, offset: u64, dest: []u8) anyerror!void,
    fn read(self: @This(), file: u32, offset: u64, dest: []u8) !void {
        return self.src(file, offset, dest);
    }
};

/// Copies each planned span into `dest` (`rows.len * words_per_row` int16s).
/// `src` is the page-cache load of exactly those bytes.
/// At most 64 reads are planned per pass; a coalesced run stays in one pass. No allocation.
pub fn gather(
    shards: []const Shard,
    words_per_row: u32,
    rows: []const u64,
    src: *const fn (file: u32, offset: u64, dest: []u8) anyerror!void,
    dest: []i16,
) !void {
    return copyRuns(shards, words_per_row, rows, FnReader{ .src = src }, dest);
}

/// Same gather, with a reader that carries its own files.
/// `reader.read(file, offset, dest)` is the SSD `pread`. The n-gram is not mapped.
pub fn gatherFrom(
    shards: []const Shard,
    words_per_row: u32,
    rows: []const u64,
    reader: anytype,
    dest: []i16,
) !void {
    return copyRuns(shards, words_per_row, rows, reader, dest);
}

const c = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
    extern "c" fn close(fd: c_int) c_int;
};

fn readExact(fd: c_int, buf: []u8, off: u64) !void {
    var done: usize = 0;
    while (done < buf.len) {
        const n = c.pread(fd, buf.ptr + done, buf.len - done, @intCast(off + done));
        if (n <= 0) return error.ReadFailed;
        done += @intCast(n);
    }
}

/// Open n-gram shards. `fds[i]` is shard file id `i`. `whole == 0` disables the guard.
/// A read whose length equals `whole` is the whole shard and is refused.
pub const Disk = struct {
    fds: []const c_int,
    whole: u64 = 0,

    pub fn openPath(path: [:0]const u8) !c_int {
        const fd = c.open(path.ptr, 0);
        if (fd < 0) return error.OpenFailed;
        return fd;
    }

    pub fn close(self: Disk) void {
        for (self.fds) |fd| _ = c.close(fd);
    }

    pub fn read(self: *const Disk, file: u32, offset: u64, dest: []u8) !void {
        if (self.whole != 0 and dest.len == self.whole) return error.WholeShard;
        if (file >= self.fds.len) return error.MissingFile;
        try readExact(self.fds[file], dest, offset);
    }
};

const hot_slots: usize = 32;
const hot_cap: usize = 128;

/// Direct-mapped cache in front of `Disk`. One n-gram row is 102 bytes at ng5.
/// A repeated row is copied from here. A coalesced run still goes straight to pread.
/// 32 rows is 4 KiB. The 30 GiB table stays on disk.
pub const HotRows = struct {
    disk: *const Disk,
    words: u32,
    misses: u32 = 0,
    tags: [hot_slots]u64 = @splat(std.math.maxInt(u64)),
    stored: [hot_slots][hot_cap]u8 = undefined,

    pub fn read(self: *HotRows, file: u32, offset: u64, dest: []u8) !void {
        const row_bytes: usize = @as(usize, self.words) * 2;
        if (row_bytes == 0 or row_bytes > hot_cap or dest.len != row_bytes) {
            return self.disk.read(file, offset, dest);
        }
        const key = (@as(u64, file) << 40) | (offset & 0x00ff_ffff_ffff);
        const slot = @as(usize, @truncate(key >> 8)) % hot_slots;
        if (self.tags[slot] == key) {
            @memcpy(dest, self.stored[slot][0..row_bytes]);
            return;
        }
        try self.disk.read(file, offset, dest);
        self.misses += 1;
        self.tags[slot] = key;
        @memcpy(self.stored[slot][0..row_bytes], dest);
    }
};

const words: u32 = 51;
const row_b: u64 = 102;

fn twoShards() [2]Shard {
    return .{
        .{ .file = 0, .offset = 100, .rows = 10 },
        .{ .file = 1, .offset = 0, .rows = 10 },
    };
}

test "locate walks shard prefix sums and rejects a row past the table" {
    const shards = twoShards();
    const row0 = try locate(&shards, words, 0);
    try std.testing.expectEqual(@as(u32, 0), row0.file);
    try std.testing.expectEqual(@as(u64, 100), row0.offset);
    try std.testing.expectEqual(@as(u32, 102), row0.bytes);

    const row10 = try locate(&shards, words, 10);
    try std.testing.expectEqual(@as(u32, 1), row10.file);
    try std.testing.expectEqual(@as(u64, 0), row10.offset);
    try std.testing.expectEqual(@as(u32, 102), row10.bytes);

    const row11 = try locate(&shards, words, 11);
    try std.testing.expectEqual(@as(u32, 1), row11.file);
    try std.testing.expectEqual(@as(u64, 102), row11.offset);
    try std.testing.expectEqual(@as(u32, 102), row11.bytes);

    try std.testing.expectError(error.RowOutOfRange, locate(&shards, words, 20));
    try std.testing.expectError(error.EmptyTable, locate(&.{}, words, 0));
    const no_rows = [_]Shard{.{ .file = 0, .offset = 0, .rows = 0 }};
    try std.testing.expectError(error.EmptyTable, locate(&no_rows, words, 0));
}

test "planReads coalesces a forward run and nothing else" {
    const shards = twoShards();
    var buf: [8]Read = undefined;

    const span = try planReads(&shards, words, &.{ 0, 1, 2, 10 }, &buf);
    try std.testing.expectEqual(@as(usize, 2), span.len);
    try std.testing.expectEqual(@as(u32, 0), span[0].file);
    try std.testing.expectEqual(@as(u64, 100), span[0].offset);
    try std.testing.expectEqual(@as(u64, 3 * row_b), span[0].bytes);
    try std.testing.expectEqual(@as(u32, 3), span[0].rows);
    try std.testing.expectEqual(@as(u32, 0), span[0].dest_row);
    try std.testing.expectEqual(@as(u32, 1), span[1].file);
    try std.testing.expectEqual(@as(u64, 0), span[1].offset);
    try std.testing.expectEqual(row_b, span[1].bytes);
    try std.testing.expectEqual(@as(u32, 1), span[1].rows);
    try std.testing.expectEqual(@as(u32, 3), span[1].dest_row);

    const gap = try planReads(&shards, words, &.{ 0, 2 }, &buf);
    try std.testing.expectEqual(@as(usize, 2), gap.len);
    try std.testing.expectEqual(@as(u32, 1), gap[0].rows);
    try std.testing.expectEqual(@as(u32, 1), gap[1].rows);
    try std.testing.expectEqual(@as(u64, 100), gap[0].offset);
    try std.testing.expectEqual(@as(u64, 100 + 2 * row_b), gap[1].offset);

    const back = try planReads(&shards, words, &.{ 5, 4 }, &buf);
    try std.testing.expectEqual(@as(usize, 2), back.len);
    try std.testing.expectEqual(@as(u32, 1), back[0].rows);
    try std.testing.expectEqual(@as(u32, 1), back[1].rows);
    try std.testing.expectEqual(@as(u64, 100 + 5 * row_b), back[0].offset);
    try std.testing.expectEqual(@as(u64, 100 + 4 * row_b), back[1].offset);

    // 9 and 10 are adjacent ids but different shards, including when the bytes abut in one file.
    const edge = try planReads(&shards, words, &.{ 9, 10 }, &buf);
    try std.testing.expectEqual(@as(usize, 2), edge.len);
    const adjacent = [_]Shard{
        .{ .file = 3, .offset = 0, .rows = 10 },
        .{ .file = 3, .offset = 10 * row_b, .rows = 10 },
    };
    const abut = try planReads(&adjacent, words, &.{ 8, 9, 10 }, &buf);
    try std.testing.expectEqual(@as(usize, 2), abut.len);
    try std.testing.expectEqual(@as(u32, 2), abut[0].rows);
    try std.testing.expectEqual(@as(u64, 8 * row_b), abut[0].offset);
    try std.testing.expectEqual(@as(u64, 2 * row_b), abut[0].bytes);
    try std.testing.expectEqual(@as(u32, 1), abut[1].rows);
    try std.testing.expectEqual(@as(u64, 10 * row_b), abut[1].offset);
    try std.testing.expectEqual(@as(u32, 3), abut[1].file);

    var one: [1]Read = undefined;
    try std.testing.expectError(error.PlanTooSmall, planReads(&shards, words, &.{ 0, 2 }, &one));
    try std.testing.expectError(error.RowOutOfRange, planReads(&shards, words, &.{ 0, 20 }, &one));
    try std.testing.expectError(error.RowOutOfRange, planReads(&shards, words, &.{20}, &buf));
    try std.testing.expectError(error.EmptyTable, planReads(&.{}, words, &.{0}, &buf));
    const none = try planReads(&shards, words, &.{}, &buf);
    try std.testing.expectEqual(@as(usize, 0), none.len);
}

/// Byte images for the gather tests. A read of `reject` bytes is the whole shard.
const probe = struct {
    var image: []const []const u8 = &.{};
    var reject: u64 = 0;
    var file: [80]u32 = undefined;
    var offset: [80]u64 = undefined;
    var len: [80]u64 = undefined;
    var n: usize = 0;

    fn reset(files: []const []const u8, whole: u64) void {
        image = files;
        reject = whole;
        n = 0;
    }

    fn read(file_id: u32, at: u64, dest: []u8) anyerror!void {
        if (@as(u64, dest.len) == reject) return error.WholeShard;
        const src = image[file_id];
        const start: usize = @intCast(at);
        if (start + dest.len > src.len) return error.ShortRead;
        @memcpy(dest, src[start..][0..dest.len]);
        file[n] = file_id;
        offset[n] = at;
        len[n] = dest.len;
        n += 1;
    }
};

fn paint(buf: []u8, at: usize, global: u64) void {
    var w: usize = 0;
    while (w < words) : (w += 1) {
        const v: i16 = @intCast(global * 64 + w);
        std.mem.writeInt(i16, buf[at + w * 2 ..][0..2], v, .little);
    }
}

fn expectRows(rows: []const u64, dest: []const i16) !void {
    for (rows, 0..) |global, i| {
        var w: usize = 0;
        while (w < words) : (w += 1) {
            const want: i16 = @intCast(global * 64 + w);
            try std.testing.expectEqual(want, dest[i * words + w]);
        }
    }
}

fn expectLogged(planned: []const Read) !void {
    try std.testing.expectEqual(planned.len, probe.n);
    for (planned, 0..) |run, i| {
        try std.testing.expectEqual(run.file, probe.file[i]);
        try std.testing.expectEqual(run.offset, probe.offset[i]);
        try std.testing.expectEqual(run.bytes, probe.len[i]);
        try std.testing.expect(run.bytes != probe.reject);
        try std.testing.expectEqual(run.bytes, @as(u64, run.rows) * row_b);
    }
}

test "gather copies the requested int16 rows and the same reads planReads issued" {
    const shards = twoShards();
    var image0: [100 + 10 * 102]u8 = undefined;
    var image1: [10 * 102]u8 = undefined;
    @memset(&image0, 0xA5);
    @memset(&image1, 0x5A);
    for (0..10) |r| {
        paint(&image0, 100 + r * 102, r);
        paint(&image1, r * 102, 10 + r);
    }
    const files = [_][]const u8{ &image0, &image1 };
    const whole: u64 = 10 * row_b;
    probe.reset(&files, whole);

    const rows = [_]u64{ 0, 1, 2, 10 };
    var dest: [4 * words]i16 = undefined;
    try gather(&shards, words, &rows, probe.read, &dest);
    try expectRows(&rows, &dest);

    var buf: [8]Read = undefined;
    const planned = try planReads(&shards, words, &rows, &buf);
    try expectLogged(planned);
    try std.testing.expect(probe.len[0] != whole);
    try std.testing.expect(probe.len[1] != whole);

    const gap = [_]u64{ 0, 2 };
    var gap_dest: [2 * words]i16 = undefined;
    probe.reset(&files, whole);
    try gather(&shards, words, &gap, probe.read, &gap_dest);
    try expectRows(&gap, &gap_dest);
    try expectLogged(try planReads(&shards, words, &gap, &buf));

    const back = [_]u64{ 5, 4 };
    var back_dest: [2 * words]i16 = undefined;
    probe.reset(&files, whole);
    try gather(&shards, words, &back, probe.read, &back_dest);
    try expectRows(&back, &back_dest);
    try expectLogged(try planReads(&shards, words, &back, &buf));
}

test "gather plans past 64 reads without splitting a later forward run" {
    const shards = [_]Shard{.{ .file = 0, .offset = 0, .rows = 8 }};
    var image: [8 * 102]u8 = undefined;
    @memset(&image, 0);
    for (0..8) |r| paint(&image, r * 102, r);
    const files = [_][]const u8{&image};
    probe.reset(&files, 8 * row_b);

    var rows: [66]u64 = undefined;
    for (0..64) |i| rows[i] = if (i % 2 == 0) 0 else 2;
    rows[64] = 6;
    rows[65] = 7;
    var dest: [66 * words]i16 = undefined;
    try gather(&shards, words, &rows, probe.read, &dest);
    try expectRows(&rows, &dest);

    var buf: [80]Read = undefined;
    try expectLogged(try planReads(&shards, words, &rows, &buf));
    try std.testing.expectEqual(@as(usize, 65), probe.n);
    try std.testing.expectEqual(@as(u64, 6 * row_b), probe.offset[64]);
    try std.testing.expectEqual(@as(u64, 2 * row_b), probe.len[64]);
}

test "page cache and ssd are the two callers of one gather" {
    try std.testing.expect(Mode.page_cache != Mode.ssd);
    const shards = twoShards();
    var buf: [4]Read = undefined;
    const rows = [_]u64{ 0, 1, 2, 10 };
    const page = try planReads(&shards, words, &rows, &buf);
    var again: [4]Read = undefined;
    const ssd = try planReads(&shards, words, &rows, &again);
    try std.testing.expectEqualSlices(Read, page, ssd);
}

fn zPath(alloc: std.mem.Allocator, path: []const u8) ![:0]u8 {
    return std.fmt.allocPrintSentinel(alloc, "{s}", .{path}, 0);
}

test "ssd pread loads the planned rows and refuses the whole shard" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const alloc = std.testing.allocator;

    var image0: [100 + 10 * 102]u8 = undefined;
    var image1: [10 * 102]u8 = undefined;
    @memset(&image0, 0xA5);
    @memset(&image1, 0x5A);
    for (0..10) |r| {
        paint(&image0, 100 + r * 102, r);
        paint(&image1, r * 102, 10 + r);
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "s0.bin", .data = &image0 });
    try tmp.dir.writeFile(io, .{ .sub_path = "s1.bin", .data = &image1 });
    const p0 = try tmp.dir.realPathFileAlloc(io, "s0.bin", alloc);
    defer alloc.free(p0);
    const p1 = try tmp.dir.realPathFileAlloc(io, "s1.bin", alloc);
    defer alloc.free(p1);
    const z0 = try zPath(alloc, p0);
    defer alloc.free(z0);
    const z1 = try zPath(alloc, p1);
    defer alloc.free(z1);

    const fds = [_]c_int{ try Disk.openPath(z0), try Disk.openPath(z1) };
    var disk = Disk{ .fds = &fds, .whole = 10 * row_b };
    defer disk.close();

    const shards = twoShards();
    const rows = [_]u64{ 0, 1, 2, 10 };
    var dest: [4 * words]i16 = undefined;
    try gatherFrom(&shards, words, &rows, &disk, &dest);
    try expectRows(&rows, &dest);

    var whole: [10 * 102]u8 = undefined;
    try std.testing.expectError(error.WholeShard, disk.read(0, 100, &whole));
    try std.testing.expectError(error.MissingFile, disk.read(2, 0, whole[0..2]));
}

test "a repeated n-gram row is served from the small cache" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const io = std.testing.io;
    const alloc = std.testing.allocator;

    var image: [10 * 102]u8 = undefined;
    @memset(&image, 0);
    for (0..10) |r| paint(&image, r * 102, r);
    try tmp.dir.writeFile(io, .{ .sub_path = "rows.bin", .data = &image });
    const path = try tmp.dir.realPathFileAlloc(io, "rows.bin", alloc);
    defer alloc.free(path);
    const z = try zPath(alloc, path);
    defer alloc.free(z);

    const fds = [_]c_int{try Disk.openPath(z)};
    const disk = Disk{ .fds = &fds, .whole = 10 * row_b };
    defer disk.close();
    var hot = HotRows{ .disk = &disk, .words = words };
    const shards = [_]Shard{.{ .file = 0, .offset = 0, .rows = 10 }};

    var once: [words]i16 = undefined;
    try gatherFrom(&shards, words, &.{4}, &hot, &once);
    try expectRows(&.{4}, &once);
    try std.testing.expectEqual(@as(u32, 1), hot.misses);

    var twice: [words]i16 = undefined;
    try gatherFrom(&shards, words, &.{4}, &hot, &twice);
    try expectRows(&.{4}, &twice);
    try std.testing.expectEqual(@as(u32, 1), hot.misses);

    var other: [words]i16 = undefined;
    try gatherFrom(&shards, words, &.{6}, &hot, &other);
    try expectRows(&.{6}, &other);
    try std.testing.expectEqual(@as(u32, 2), hot.misses);
}
