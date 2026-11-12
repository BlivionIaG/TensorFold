//! Big hand-off buffers on unlinked mmap'd files (`TF_PRODUCE_SPILL_DIR`): page cache the kernel may write back.

const std = @import("std");
const builtin = @import("builtin");

const Alignment = std.mem.Alignment;

pub const Error = error{ BadSpillDir, SpillUnusable };

pub const Spill = struct {
    gpa: std.mem.Allocator,
    /// the directory the files are made under (owned)
    dir: [:0]u8,
    /// allocations of at least this many bytes are mapped from a file
    min_bytes: usize,
    lock: std.atomic.Mutex = .unlocked,
    /// base address -> mapped length, of every live file-backed allocation
    maps: std.AutoHashMapUnmanaged(usize, usize) = .empty,
    serial: std.atomic.Value(u64) = .init(0),
    /// live and highest file-backed bytes
    live: std.atomic.Value(u64) = .init(0),
    peak: std.atomic.Value(u64) = .init(0),

    /// A spill under `dir` (made if missing by the caller); fails when no file can be mapped there.
    pub fn open(gpa: std.mem.Allocator, dir: []const u8, min_bytes: usize) !*Spill {
        if (builtin.os.tag == .windows) return error.SpillUnusable;
        if (dir.len == 0 or dir.len > 900 or std.mem.indexOfScalar(u8, dir, 0) != null) return error.BadSpillDir;
        const s = try gpa.create(Spill);
        errdefer gpa.destroy(s);
        const owned = try gpa.allocSentinel(u8, dir.len, 0);
        @memcpy(owned, dir);
        s.* = .{ .gpa = gpa, .dir = owned, .min_bytes = @max(min_bytes, std.heap.pageSize()) };
        errdefer gpa.free(s.dir);
        const probe = s.map(std.heap.pageSize()) orelse return error.SpillUnusable;
        s.unmap(probe.ptr, probe.len);
        return s;
    }

    pub fn close(s: *Spill) void {
        s.maps.deinit(s.gpa);
        s.gpa.free(s.dir);
        s.gpa.destroy(s);
    }

    pub fn allocator(s: *Spill) std.mem.Allocator {
        return .{ .ptr = s, .vtable = &vtable };
    }

    const vtable: std.mem.Allocator.VTable = .{ .alloc = alloc, .resize = resize, .remap = remap, .free = free };

    fn self(ctx: *anyopaque) *Spill {
        return @ptrCast(@alignCast(ctx));
    }

    fn lockMaps(s: *Spill) void {
        while (!s.lock.tryLock()) std.atomic.spinLoopHint();
    }

    /// The length of the file-backed allocation at `ptr`, or null when the base allocator owns it.
    fn mapped(s: *Spill, ptr: [*]u8) ?usize {
        s.lockMaps();
        defer s.lock.unlock();
        return s.maps.get(@intFromPtr(ptr));
    }

    /// `len` bytes (rounded up to pages) of an unlinked file, mapped shared; null when the file system refuses.
    fn map(s: *Spill, len: usize) ?[]align(std.heap.page_size_min) u8 {
        const page = std.heap.pageSize();
        const size = std.mem.alignForward(usize, len, page);
        var name: [1024]u8 = undefined;
        const n = s.serial.fetchAdd(1, .monotonic);
        const path = std.fmt.bufPrintSentinel(&name, "{s}/export-{d}-{d}.spill", .{ s.dir, std.c.getpid(), n }, 0) catch return null;
        const fd = std.c.open(path.ptr, .{ .ACCMODE = .RDWR, .CREAT = true, .EXCL = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return null;
        defer _ = std.c.close(fd);
        _ = std.c.unlink(path.ptr); // the mapping keeps the pages; nothing is left behind after a crash
        if (std.c.ftruncate(fd, @intCast(size)) != 0) return null;
        const mem = std.posix.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, fd, 0) catch return null;
        s.lockMaps();
        defer s.lock.unlock();
        s.maps.put(s.gpa, @intFromPtr(mem.ptr), size) catch {
            std.posix.munmap(mem);
            return null;
        };
        const now = s.live.fetchAdd(size, .monotonic) + size;
        _ = s.peak.fetchMax(now, .monotonic);
        return mem;
    }

    fn unmap(s: *Spill, ptr: [*]u8, size: usize) void {
        s.lockMaps();
        _ = s.maps.remove(@intFromPtr(ptr));
        s.lock.unlock();
        _ = s.live.fetchSub(size, .monotonic);
        const mem: [*]align(std.heap.page_size_min) u8 = @ptrCast(@alignCast(ptr));
        std.posix.munmap(mem[0..size]);
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: Alignment, ret_addr: usize) ?[*]u8 {
        const s = self(ctx);
        if (len >= s.min_bytes and alignment.toByteUnits() <= std.heap.pageSize()) {
            if (s.map(len)) |mem| return mem.ptr;
        }
        return s.gpa.rawAlloc(len, alignment, ret_addr);
    }

    fn resize(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) bool {
        const s = self(ctx);
        if (s.mapped(memory.ptr) != null) return false; // a file's size is fixed: the caller moves to a new one
        return s.gpa.rawResize(memory, alignment, new_len, ret_addr);
    }

    fn remap(ctx: *anyopaque, memory: []u8, alignment: Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        const s = self(ctx);
        if (s.mapped(memory.ptr) != null) return null;
        return s.gpa.rawRemap(memory, alignment, new_len, ret_addr);
    }

    fn free(ctx: *anyopaque, memory: []u8, alignment: Alignment, ret_addr: usize) void {
        const s = self(ctx);
        if (s.mapped(memory.ptr)) |size| return s.unmap(memory.ptr, size);
        s.gpa.rawFree(memory, alignment, ret_addr);
    }
};

/// `TF_PRODUCE_SPILL_MIB`'s default: buffers from 16 MiB up are file-backed (a 1M export's pieces are far larger).
pub const default_min_mib: u64 = 16;

test "buffers over the threshold are file-backed, small ones stay on the base allocator, and both free cleanly" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const tmp = if (std.c.getenv("TMPDIR")) |t| std.mem.span(t) else "/tmp";
    const s = try Spill.open(std.testing.allocator, tmp, 1 << 20);
    defer s.close();
    const a = s.allocator();
    const small = try a.alloc(u8, 4096);
    defer a.free(small);
    try std.testing.expectEqual(@as(u64, 0), s.live.load(.monotonic));
    const big = try a.alloc(u8, 3 << 20);
    try std.testing.expectEqual(@as(u64, 3 << 20), s.live.load(.monotonic));
    @memset(big, 0xA5);
    try std.testing.expectEqual(@as(u8, 0xA5), big[(3 << 20) - 1]);
    a.free(big);
    try std.testing.expectEqual(@as(u64, 0), s.live.load(.monotonic));
    try std.testing.expectEqual(@as(u64, 3 << 20), s.peak.load(.monotonic));
}

test "an arena over a spill puts its big chunks on files and returns them all at deinit" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    const tmp = if (std.c.getenv("TMPDIR")) |t| std.mem.span(t) else "/tmp";
    const s = try Spill.open(std.testing.allocator, tmp, 1 << 20);
    defer s.close();
    var arena = std.heap.ArenaAllocator.init(s.allocator());
    const x = try arena.allocator().alloc(u8, 5 << 20);
    const y = try arena.allocator().alloc(u8, 100);
    @memset(x, 1);
    @memset(y, 2);
    try std.testing.expect(s.live.load(.monotonic) >= 5 << 20);
    try std.testing.expectEqual(@as(u8, 1), x[x.len - 1]);
    arena.deinit();
    try std.testing.expectEqual(@as(u64, 0), s.live.load(.monotonic));
}

test "a directory that cannot hold files is refused at open" {
    if (builtin.os.tag == .windows) return error.SkipZigTest;
    try std.testing.expectError(error.SpillUnusable, Spill.open(std.testing.allocator, "/nonexistent-spill-dir", 1 << 20));
    try std.testing.expectError(error.BadSpillDir, Spill.open(std.testing.allocator, "", 1 << 20));
}
