//! The split offload's boundary rows: fp32 rows of a host window to and from the activation kind (split/rows.hip).

const std = @import("std");
const Launcher = @import("../launches.zig").Launcher;
const Module = @import("../module.zig").Module;
const Function = @import("../module.zig").Function;
const util = @import("util.zig");
const S = util.S;
const Args = util.Args;
const Error = util.Error;
const dim = util.dim;
const cdiv = util.cdiv;
const invalid = util.invalid;

/// The most threads a consume launch has: one workgroup, a stamped group a lane a pass.
pub const consume_threads = 1024;
/// The fp32 payload values of a 16-byte stamped group (wire.zig's group_data / 4).
pub const group_values = 3;
/// A flat kernel's workgroup.
const flat_threads = 256;

pub const Fns = struct {
    import_row: Function,
    export_row: Function,
    consume: Function,

    pub fn load(m: Module) Error!Fns {
        return .{
            .import_row = try m.function("tf_split_import_row"),
            .export_row = try m.function("tf_split_export_row"),
            .consume = try m.function("tf_split_consume_stamped"),
        };
    }
};

/// Stamped groups a row of `n` values takes: ceil(n / 3).
pub fn groups(n: usize) usize {
    return cdiv(n, group_values);
}

fn elemBytes(kind: c_int) usize {
    return if (kind == 0) 4 else 2;
}

/// Whether a flat row launch takes the shape: `n` values of `kind`, the fp32 row (`window`) 4-byte aligned, the other row aligned to its kind.
pub fn flatFits(window: u64, rows: u64, kind: c_int, n: usize) bool {
    if (n == 0 or n > 0x7fff_ffff or kind < 0 or kind > 2 or window == 0 or rows == 0) return false;
    return window % 4 == 0 and rows % elemBytes(kind) == 0;
}

/// Whether a consume launch takes the shape: the stamped row 16-byte aligned, the status word and dst aligned.
pub fn consumeFits(src: u64, dst: u64, kind: c_int, n: usize, status: u64) bool {
    return flatFits(src, dst, kind, n) and src % 16 == 0 and status != 0 and status % 4 == 0;
}

/// dst (n values of `kind`, 0 fp32, 1 fp16, 2 bf16) = the fp32 row at `src`, rounded to nearest even; src 4-byte aligned.
pub fn importRow(l: *const Launcher, src: u64, dst: u64, kind: c_int, n: usize, s: S) Error!void {
    if (!flatFits(src, dst, kind, n)) return invalid("split import row");
    var a: Args = .{};
    try a.add(src);
    try a.add(dst);
    try a.add(@as(c_int, @intCast(n)));
    try a.add(kind);
    try l.go(l.split_rows.import_row, dim(cdiv(n, flat_threads), 1, 1), dim(flat_threads, 1, 1), 0, s, &a);
}

/// The fp32 row at `dst` = n values of `kind` widened; dst 4-byte aligned.
pub fn exportRow(l: *const Launcher, src: u64, dst: u64, kind: c_int, n: usize, s: S) Error!void {
    if (!flatFits(dst, src, kind, n)) return invalid("split export row");
    var a: Args = .{};
    try a.add(src);
    try a.add(dst);
    try a.add(@as(c_int, @intCast(n)));
    try a.add(kind);
    try l.go(l.split_rows.export_row, dim(cdiv(n, flat_threads), 1, 1), dim(flat_threads, 1, 1), 0, s, &a);
}

/// dst (n values of `kind`) = the stamped row at `src` (`groups(n)` 16-byte groups, 16-byte aligned) once every group's
/// stamp is `seq`: a stale group is read again up to `retries` more times; `status` (a 4-byte aligned word) is set to 0, or
/// to 1 when a group was still stale (dst then holds that group as last read).
pub fn consumeStamped(l: *const Launcher, src: u64, seq: u32, dst: u64, kind: c_int, n: usize, status: u64, retries: u32, s: S) Error!void {
    if (!consumeFits(src, dst, kind, n, status)) return invalid("split consume stamped");
    var a: Args = .{};
    try a.add(src);
    try a.add(seq);
    try a.add(dst);
    try a.add(@as(c_int, @intCast(n)));
    try a.add(kind);
    try a.add(status);
    try a.add(retries);
    try l.go(l.split_rows.consume, dim(1, 1, 1), dim(consume_threads, 1, 1), 0, s, &a);
}

test "the row launches refuse an empty or oversized row, a bad kind and misaligned addresses" {
    try std.testing.expect(flatFits(0x1000, 0x2000, 1, 2560));
    try std.testing.expect(flatFits(0x1004, 0x2002, 2, 1));
    try std.testing.expect(flatFits(0x1004, 0x2004, 0, 3));
    try std.testing.expect(!flatFits(0x1000, 0x2000, 1, 0));
    try std.testing.expect(!flatFits(0x1000, 0x2000, 1, 0x8000_0000));
    try std.testing.expect(!flatFits(0x1000, 0x2000, 3, 8));
    try std.testing.expect(!flatFits(0x1000, 0x2000, -1, 8));
    try std.testing.expect(!flatFits(0x1002, 0x2000, 1, 8)); // the fp32 row off 4 bytes
    try std.testing.expect(!flatFits(0x1000, 0x2001, 1, 8)); // 16-bit values off 2
    try std.testing.expect(!flatFits(0x1000, 0x2002, 0, 8)); // fp32 values off 4
    try std.testing.expect(!flatFits(0, 0x2000, 1, 8));
}

test "the consume launch needs a 16-byte aligned stamped row and an aligned status word" {
    try std.testing.expect(consumeFits(0x1040, 0x2000, 2, 2560, 0x3000));
    try std.testing.expect(!consumeFits(0x1004, 0x2000, 2, 2560, 0x3000));
    try std.testing.expect(!consumeFits(0x1040, 0x2000, 2, 2560, 0x3002));
    try std.testing.expect(!consumeFits(0x1040, 0x2000, 2, 2560, 0));
    try std.testing.expect(!consumeFits(0x1040, 0x2000, 2, 0, 0x3000));
}

/// hipmac's hm_mock_rec (hm_mock.h): what its mock backend logs for a dispatch it executes.
const MockRec = extern struct {
    exec_index: u64,
    queue_id: c_int,
    name: [128]u8,
    kd_va: u64,
    grid: [3]u32,
    block: [3]u32,
    dynamic_lds: u32,
    flags: u32,
    kernarg_size: u32,
    ib_node: u32,
    ib_replay: u64,
    kernarg: [*]const u8,

    fn is(r: MockRec, name: []const u8) bool {
        return std.mem.eql(u8, std.mem.sliceTo(&r.name, 0), name);
    }

    fn word(r: MockRec, comptime T: type, at: usize) T {
        return std.mem.readInt(T, r.kernarg[at..][0..@sizeOf(T)], .little);
    }
};

// The three launches on hipmac's mock backend: the dispatches it logs have these grids and arguments. The mock runs no
// kernel, so this checks plumbing only. It needs HIPMAC_BACKEND=mock, TENSORFOLD_HIP naming hipmac's library, and
// TF_SPLIT_ROWS_HSACO naming a gfx900 code object of split/rows.hip; without all three it skips, and never opens a card.
test "on the mock backend the launches arrive with their grid and arguments" {
    const sys = @import("sys");
    const runtime = @import("../runtime.zig");
    const memory = @import("../memory.zig");
    const Context = @import("../context.zig").Context;
    const Stream = @import("../stream.zig").Stream;
    const backend = sys.getenv("HIPMAC_BACKEND") orelse return error.SkipZigTest;
    if (!std.mem.eql(u8, backend, "mock")) return error.SkipZigTest;
    const object = sys.getenv("TF_SPLIT_ROWS_HSACO") orelse return error.SkipZigTest;
    var rt = runtime.Runtime.open() catch return error.SkipZigTest;
    defer rt.close();
    const Count = *const fn () callconv(.c) usize;
    const Get = *const fn (usize) callconv(.c) *const MockRec;
    const count = rt.lib.lookup(Count, "hm_mock_log_count") orelse return error.SkipZigTest;
    const get = rt.lib.lookup(Get, "hm_mock_log_get") orelse return error.SkipZigTest;

    const a = std.testing.allocator;
    const file = try std.Io.Dir.cwd().readFileAlloc(std.testing.io, object, a, .limited(1 << 24));
    defer a.free(file);
    const image = try a.alignedAlloc(u8, .@"8", file.len);
    defer a.free(image);
    @memcpy(image, file);

    var ctx = try Context.init(&rt, 0);
    defer ctx.deinit();
    var stream = try Stream.init(&rt);
    defer stream.deinit();
    var module = try Module.load(&rt, image);
    defer module.unload();
    var l: Launcher = undefined; // go() reads only the runtime; the launches only their own kernels
    l.r = &rt;
    l.split_rows = try Fns.load(module);

    var window = try memory.HostBuffer.allocMapped(&rt, 1 << 16);
    defer window.free();
    const base = try window.device();
    const row: u64 = base + 64; // after a slot's header, 16-byte aligned
    const n = 2560;
    const dst: u64 = base + 0x8000;
    const status: u64 = base + 0x4000;
    const first = count();
    try importRow(&l, row, dst, 2, n, stream.handle);
    try exportRow(&l, dst, row, 1, n, stream.handle);
    try consumeStamped(&l, row, 77, dst, 2, n, status, 1000, stream.handle);
    try stream.synchronize();
    try std.testing.expectEqual(first + 3, count());

    const imp = get(first).*;
    try std.testing.expect(imp.is("tf_split_import_row"));
    try std.testing.expectEqual([3]u32{ 10, 1, 1 }, imp.grid);
    try std.testing.expectEqual([3]u32{ 256, 1, 1 }, imp.block);
    try std.testing.expectEqual(row, imp.word(u64, 0));
    try std.testing.expectEqual(dst, imp.word(u64, 8));
    try std.testing.expectEqual(@as(u32, n), imp.word(u32, 16));
    try std.testing.expectEqual(@as(u32, 2), imp.word(u32, 20));

    const exp = get(first + 1).*;
    try std.testing.expect(exp.is("tf_split_export_row"));
    try std.testing.expectEqual([3]u32{ 10, 1, 1 }, exp.grid);
    try std.testing.expectEqual(dst, exp.word(u64, 0));
    try std.testing.expectEqual(row, exp.word(u64, 8));
    try std.testing.expectEqual(@as(u32, 1), exp.word(u32, 20));

    const con = get(first + 2).*;
    try std.testing.expect(con.is("tf_split_consume_stamped"));
    try std.testing.expectEqual([3]u32{ 1, 1, 1 }, con.grid);
    try std.testing.expectEqual([3]u32{ consume_threads, 1, 1 }, con.block);
    try std.testing.expectEqual(row, con.word(u64, 0));
    try std.testing.expectEqual(@as(u32, 77), con.word(u32, 8));
    try std.testing.expectEqual(dst, con.word(u64, 16));
    try std.testing.expectEqual(@as(u32, n), con.word(u32, 24));
    try std.testing.expectEqual(@as(u32, 2), con.word(u32, 28));
    try std.testing.expectEqual(status, con.word(u64, 32));
    try std.testing.expectEqual(@as(u32, 1000), con.word(u32, 40));
    try std.testing.expectEqual(first + 3, count());
}
