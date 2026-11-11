//! Number formatting the way Python prints it, so the Zig ports of the Python tools print the same text.
//!
//! Zig's `{d:.N}` rounds the shortest round-trip decimal of a float, Python's `{x:.Nf}` rounds its exact binary value (half to even on an exact tie):
//! 30.25 -> "30.2" and 0.0905 -> "0.090" in Python, "30.3" and "0.091" in Zig. `fixed` goes through the C library's snprintf, which rounds
//! exactly like Python's formatter. `repr` is Python's repr(float) (shortest round-trip digits, exponent form outside 1e-4 <= |x| < 1e16).
const std = @import("std");

extern "c" fn snprintf(buf: [*]u8, size: usize, fmt: [*:0]const u8, ...) c_int;

pub const buf_len = 400;

/// `"%.{prec}f"` (or `"%+.{prec}f"` with `plus`) of `x`, written into `buf`.
pub fn fixed(buf: *[buf_len]u8, x: f64, prec: u8, plus: bool) []const u8 {
    var spec: [8:0]u8 = @splat(0);
    const text = std.fmt.bufPrint(&spec, "%{s}.{d}f", .{ if (plus) "+" else "", prec }) catch unreachable;
    spec[text.len] = 0;
    const n = snprintf(buf, buf.len, &spec, x);
    return buf[0..@min(@as(usize, @intCast(@max(n, 0))), buf.len - 1)];
}

/// Python's round(x, digits) for finite x: the correctly rounded decimal of x's exact value, read back as a float.
pub fn round(x: f64, digits: u8) f64 {
    if (!std.math.isFinite(x)) return x;
    var buf: [buf_len]u8 = undefined;
    return std.fmt.parseFloat(f64, fixed(&buf, x, digits, false)) catch x;
}

/// Python's repr(float): `81.4`, `81.0`, `1e-05`, `1.5e+16`, `inf`, `nan`.
pub fn repr(buf: *[buf_len]u8, x: f64) []const u8 {
    if (std.math.isNan(x)) return "nan";
    if (std.math.isInf(x)) return if (x < 0) "-inf" else "inf";
    var sci: [64]u8 = undefined;
    const s = std.fmt.bufPrint(&sci, "{e}", .{x}) catch unreachable; // shortest digits: "-d.ddde-7"
    var i: usize = 0;
    var neg = false;
    if (s[0] == '-') {
        neg = true;
        i = 1;
    }
    const e_at = std.mem.indexOfScalar(u8, s, 'e').?;
    var digits: [32]u8 = undefined;
    var nd: usize = 0;
    for (s[i..e_at]) |c| if (c != '.') {
        digits[nd] = c;
        nd += 1;
    };
    // trailing zeros of the digit string carry no information
    while (nd > 1 and digits[nd - 1] == '0') nd -= 1;
    const exp = std.fmt.parseInt(i32, s[e_at + 1 ..], 10) catch unreachable;
    const decpt = exp + 1; // the decimal point sits after this many digits
    var w: std.Io.Writer = .fixed(buf);
    if (neg) w.writeByte('-') catch unreachable;
    if (digits[0] == '0') { // zero
        w.writeAll("0.0") catch unreachable;
    } else if (decpt > -4 and decpt <= 16) {
        if (decpt <= 0) {
            w.writeAll("0.") catch unreachable;
            w.splatByteAll('0', @intCast(-decpt)) catch unreachable;
            w.writeAll(digits[0..nd]) catch unreachable;
        } else if (@as(usize, @intCast(decpt)) >= nd) {
            w.writeAll(digits[0..nd]) catch unreachable;
            w.splatByteAll('0', @as(usize, @intCast(decpt)) - nd) catch unreachable;
            w.writeAll(".0") catch unreachable;
        } else {
            const p: usize = @intCast(decpt);
            w.writeAll(digits[0..p]) catch unreachable;
            w.writeByte('.') catch unreachable;
            w.writeAll(digits[p..nd]) catch unreachable;
        }
    } else {
        w.writeByte(digits[0]) catch unreachable;
        if (nd > 1) {
            w.writeByte('.') catch unreachable;
            w.writeAll(digits[1..nd]) catch unreachable;
        }
        const e10 = decpt - 1;
        w.print("e{s}{d:0>2}", .{ if (e10 < 0) "-" else "+", @abs(e10) }) catch unreachable;
    }
    return w.buffered();
}

/// Python's `{x:<w}.<prec>f`-style field: `x` fixed to `prec` places, right-aligned in `width`.
pub fn fieldF(w: *std.Io.Writer, x: f64, width: usize, prec: u8) std.Io.Writer.Error!void {
    var buf: [buf_len]u8 = undefined;
    const t = fixed(&buf, x, prec, false);
    if (t.len < width) try w.splatByteAll(' ', width - t.len);
    try w.writeAll(t);
}

test "fixed rounds the exact binary value like Python" {
    var b: [buf_len]u8 = undefined;
    try std.testing.expectEqualStrings("30.2", fixed(&b, 30.25, 1, false)); // an exact tie goes to even
    try std.testing.expectEqualStrings("0.090", fixed(&b, 90.5 * 1.0 / 1000.0, 3, false));
    try std.testing.expectEqualStrings("6.0", fixed(&b, 6.05, 1, false)); // 6.04999...
    try std.testing.expectEqualStrings("1.2", fixed(&b, 1.25, 1, false));
    try std.testing.expectEqualStrings("+0.57", fixed(&b, 0.575, 2, true)); // 0.57499999999999995559
    try std.testing.expectEqualStrings("-0.00", fixed(&b, -0.001, 2, false));
    try std.testing.expectEqualStrings("3514", fixed(&b, 3514.4, 0, false));
    try std.testing.expectEqualStrings("nan", fixed(&b, std.math.nan(f64), 2, false)[0..3]);
}

test "round is Python's round" {
    try std.testing.expectEqual(@as(f64, 81.4), round(81.43, 1));
    try std.testing.expectEqual(@as(f64, 2.67), round(2.675, 2)); // 2.67499999...
    try std.testing.expectEqual(@as(f64, 0.125), round(0.125, 3));
    try std.testing.expectEqual(@as(f64, 0.12), round(0.125, 2)); // an exact tie goes to even
}

test "repr matches Python's repr(float)" {
    var b: [buf_len]u8 = undefined;
    var tenth: f64 = 0.1; // runtime values: 0.1 + 0.2 folded at comptime would be exactly 0.3
    tenth += 0.2;
    const cases = [_]struct { f64, []const u8 }{
        .{ 81.4, "81.4" },
        .{ 81.0, "81.0" },
        .{ 0.0, "0.0" },
        .{ -0.0, "-0.0" },
        .{ 1e16, "1e+16" },
        .{ 1e15, "1000000000000000.0" },
        .{ 123456789012345678.0, "1.2345678901234568e+17" },
        .{ 0.0001, "0.0001" },
        .{ 0.00001, "1e-05" },
        .{ 1.5e-7, "1.5e-07" },
        .{ tenth, "0.30000000000000004" },
        .{ -2.5, "-2.5" },
        .{ 100.0, "100.0" },
        .{ 1234.5678, "1234.5678" },
        .{ 5e-324, "5e-324" },
        .{ 1.7976931348623157e308, "1.7976931348623157e+308" },
    };
    for (cases) |c| try std.testing.expectEqualStrings(c[1], repr(&b, c[0]));
}
