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

/// Whether a flat row launch takes the shape: `n` values of `kind`, `window` 4-byte aligned, `rows` kind-aligned.
pub fn flatFits(window: u64, rows: u64, kind: c_int, n: usize) bool {
    if (n == 0 or n > 0x7fff_ffff or kind < 0 or kind > 2 or window == 0 or rows == 0) return false;
    return window % 4 == 0 and rows % elemBytes(kind) == 0;
}

/// Whether a consume launch takes the shape: the stamped row 16-byte aligned, the status word and dst aligned.
pub fn consumeFits(src: u64, dst: u64, kind: c_int, n: usize, status: u64) bool {
    return flatFits(src, dst, kind, n) and src % 16 == 0 and status != 0 and status % 4 == 0;
}

/// dst (n values of `kind`, 0 fp32, 1 fp16, 2 bf16) = the fp32 row at `src`, rounded to nearest even.
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

/// dst = the stamped row at `src` (`groups(n)` groups) once every stamp is `seq`; `status` 0, or 1 if one stayed stale.
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
