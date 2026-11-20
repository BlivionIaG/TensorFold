//! `--glp FILE [--glp-alpha F]` on tf-dsv41's served commands (follow, generate): a worker rank loads the same verified
//! vector at the same effective strength as rank 0 before it joins the ranks' agreement (glp_agree.zig).
//! GLP is Matt Suiche's `weightless` format (https://weightle.ss/).

const std = @import("std");
const api = @import("engine_api");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel

pub const Split = struct {
    /// the command's arguments without the GLP flags
    positional: []const []const u8,
    /// the vector with the strength in force; null: none given (every program as unsteered)
    glp: ?api.glp.Glp,
};

pub const Error = error{ BadGlpFlag, BadGlpVector } || std.mem.Allocator.Error;

fn fail(comptime fmt: []const u8, args: anytype) void {
    if (!quiet) std.debug.print(fmt, args);
}

/// A flag's value: `--flag VALUE` or `--flag=VALUE`; null when `x` is not that flag.
fn value(x: []const u8, args: []const []const u8, i: *usize, comptime flag: []const u8) Error!?[]const u8 {
    if (std.mem.eql(u8, x, flag)) {
        i.* += 1;
        if (i.* >= args.len) {
            fail("FAIL {s} needs a value\n", .{flag});
            return error.BadGlpFlag;
        }
        return args[i.*];
    }
    if (std.mem.startsWith(u8, x, flag ++ "=")) return x[flag.len + 1 ..];
    return null;
}

/// `args` split into the positional arguments and the vector the flags name, loaded into `a` (it must outlive the
/// vector). A strength without a vector, a bad strength and a file the reader refuses are refused, with the reason.
pub fn split(a: std.mem.Allocator, io: std.Io, args: []const []const u8) Error!Split {
    var pos: std.ArrayList([]const u8) = .empty;
    var file: ?[]const u8 = null;
    var alpha: ?f32 = null;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const x = args[i];
        if (try value(x, args, &i, "--glp")) |v| {
            file = v;
        } else if (try value(x, args, &i, "--glp-alpha")) |v| {
            const parsed = std.fmt.parseFloat(f64, v) catch {
                fail("FAIL --glp-alpha {s}: not a number\n", .{v});
                return error.BadGlpFlag;
            };
            alpha = api.glp.strength(parsed) orelse {
                fail("FAIL --glp-alpha {s}: the strength is finite, from 0 and a normal float\n", .{v});
                return error.BadGlpFlag;
            };
        } else try pos.append(a, x);
    }
    const path = file orelse {
        if (alpha != null) {
            fail("FAIL --glp-alpha needs --glp FILE\n", .{});
            return error.BadGlpFlag;
        }
        return .{ .positional = pos.items, .glp = null };
    };
    var why: api.glp.Why = .{};
    const g = api.glp.load(a, io, path, alpha, &why) catch |e| {
        if (e == error.OutOfMemory) return e;
        fail("FAIL --glp {s}: {s}\n", .{ path, why.text() });
        return error.BadGlpVector;
    };
    return .{ .positional = pos.items, .glp = g };
}

test "the GLP flags leave the positional arguments and refuse a strength alone" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain = try split(a, std.testing.io, &.{ "pack", "assets" });
    try std.testing.expectEqual(@as(usize, 2), plain.positional.len);
    try std.testing.expect(plain.glp == null);
    try std.testing.expectError(error.BadGlpFlag, split(a, std.testing.io, &.{ "pack", "--glp-alpha", "0.5", "assets" }));
    try std.testing.expectError(error.BadGlpFlag, split(a, std.testing.io, &.{ "pack", "--glp" }));
}
