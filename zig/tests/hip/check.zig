//! Shared helpers for the HIP test runner: the device bundle, pass/fail lines and byte comparison.

const std = @import("std");
const hip = @import("hip");

pub const Gpu = struct {
    d: *const hip.Driver,
    ctx: *const hip.Context,
    gpa: std.mem.Allocator,
    io: std.Io,
    /// The `--policy` flags: `key=value,...`.
    policy: []const u8 = "",
};

/// The policy a run is under: the GPU's defaults, the `--policy` flags, then the old variables and TF_POLICY.
pub fn policyOf(gpu: Gpu) !hip.Policy {
    var notes: hip.Policy.Notes = .{};
    return hip.Policy.resolve(try gpu.ctx.caps(), gpu.policy, .current, &notes);
}

/// Whether the steps inside a case group print their own lines.
pub var verbose = false;

/// A detail line of a case group, printed with `-v`.
pub fn step(comptime fmt: []const u8, args: anytype) void {
    if (verbose) std.debug.print("  " ++ fmt ++ "\n", args);
}

/// The case groups of one command: each prints one PASS or FAIL line, and the summary decides the exit code.
pub const Tally = struct {
    passed: usize = 0,
    failed: usize = 0,
    filter: []const u8 = "",

    pub fn wants(t: Tally, name: []const u8) bool {
        return t.filter.len == 0 or std.mem.indexOf(u8, name, t.filter) != null;
    }

    /// Runs `f(ctx)` as the group `name` when the filter takes it; the group prints its own PASS line.
    pub fn group(t: *Tally, name: []const u8, ctx: anytype, comptime f: anytype) void {
        if (!t.wants(name)) return;
        if (f(ctx)) {
            t.passed += 1;
        } else |e| {
            std.debug.print("FAIL {s}: {t}\n", .{ name, e });
            t.failed += 1;
        }
    }

    pub fn summary(t: Tally, what: []const u8) u8 {
        std.debug.print("{s}: {d} passed, {d} failed\n", .{ what, t.passed, t.failed });
        if (t.passed + t.failed == 0) {
            std.debug.print("FAIL {s}: no case group matches '{s}'\n", .{ what, t.filter });
            return 1;
        }
        return if (t.failed == 0) 0 else 1;
    }
};

pub const Failed = error{TestFailed};

pub fn expect(ok: bool, comptime fmt: []const u8, args: anytype) Failed!void {
    if (ok) return;
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

pub fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}

/// Equal bytes, or the first difference and how many 4-byte words differ.
pub fn sameBytes(what: []const u8, got: []const u8, want: []const u8) Failed!void {
    if (got.len != want.len) {
        std.debug.print("FAIL {s}: {d} bytes, expected {d}\n", .{ what, got.len, want.len });
        return error.TestFailed;
    }
    if (std.mem.eql(u8, got, want)) return;
    const first = std.mem.indexOfDiff(u8, got, want).?;
    var words: usize = 0;
    var i: usize = 0;
    while (i + 4 <= got.len) : (i += 4) {
        if (!std.mem.eql(u8, got[i..][0..4], want[i..][0..4])) words += 1;
    }
    std.debug.print("FAIL {s}: first difference at byte {d}, {d} of {d} words differ\n", .{ what, first, words, got.len / 4 });
    return error.TestFailed;
}

/// Monotonic nanoseconds for host timing.
pub fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

pub fn download(gpu: Gpu, b: hip.DeviceBuffer) ![]u8 {
    const out = try gpu.gpa.alloc(u8, b.len);
    errdefer gpu.gpa.free(out);
    try b.download(0, out);
    return out;
}

pub fn median(xs: []f64) f64 {
    std.mem.sort(f64, xs, {}, std.sort.asc(f64));
    const n = xs.len;
    return if (n % 2 == 1) xs[n / 2] else (xs[n / 2 - 1] + xs[n / 2]) / 2;
}
