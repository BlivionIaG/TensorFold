//! tf-prof-report TABLE.txt [TOKENS]: a hipprof table of a decode run as a per-token breakdown (the port of prof_report.py).
//!
//! Per kernel launch shape: launches a token (count / TOKENS, the tokens the profiled requests decoded; default: the count of the lm_head launch,
//! else the most frequent count), mean time of one launch (timed with a sync after it: kernel plus the runtime's fixed cost), the pipelined time
//! from the replays (rep_us), the weight bytes it reads and the GB/s of both against the card's 409.6 GB/s, and the share of the token's time.
const std = @import("std");
const pf = @import("pyfmt.zig");

pub const Row = struct {
    name: []const u8,
    count: u64,
    grid: []const u8,
    block: []const u8,
    shared: u64,
    m: u64,
    n: u64,
    k: u64,
    sides: u64,
    avg: f64,
    min: f64,
    total: f64,
    rep: f64,
    bytes: f64,
    gap: f64,
    /// the time a launch counts for: the pipelined one where it was replayed
    t: f64 = 0,
};

pub const ParseError = error{ ShortLine, BadNumber };

/// A table line: tab separated, at least the 14 columns of `hipprof_table.formatRow` (the gap column is optional).
pub fn parseLine(line: []const u8) ParseError!Row {
    var f: [15][]const u8 = undefined;
    var n: usize = 0;
    var it = std.mem.splitScalar(u8, line, '\t');
    while (it.next()) |col| {
        if (n == f.len) break;
        f[n] = col;
        n += 1;
    }
    if (n < 14) return error.ShortLine;
    const int = struct {
        fn of(s: []const u8) ParseError!u64 {
            return std.fmt.parseInt(u64, std.mem.trim(u8, s, " "), 10) catch error.BadNumber;
        }
        fn fl(s: []const u8) ParseError!f64 {
            return std.fmt.parseFloat(f64, std.mem.trim(u8, s, " ")) catch error.BadNumber;
        }
    };
    return .{
        .name = f[0],
        .count = try int.of(f[1]),
        .grid = f[2],
        .block = f[3],
        .shared = try int.of(f[4]),
        .m = try int.of(f[5]),
        .n = try int.of(f[6]),
        .k = try int.of(f[7]),
        .sides = try int.of(f[8]),
        .avg = try int.fl(f[9]),
        .min = try int.fl(f[10]),
        .total = try int.fl(f[11]),
        .rep = try int.fl(f[12]),
        .bytes = try int.fl(f[13]),
        .gap = if (n > 14) try int.fl(f[14]) else 0,
    };
}

/// The readable form of a kernel name: `_ZN2tf4rocm<len>NAME...INS0_...Li<N>...` becomes `NAME<N,...>`, anything else is kept.
pub fn short(a: std.mem.Allocator, name: []const u8) ![]u8 {
    const prefix = "_ZN2tf4rocm";
    if (!std.mem.startsWith(u8, name, prefix)) return a.dupe(u8, name);
    var rest = name[prefix.len..];
    var d: usize = 0;
    while (d < rest.len and std.ascii.isDigit(rest[d])) d += 1;
    rest = rest[d..];
    if (std.mem.indexOf(u8, rest, "INS0_")) |cut| rest = rest[0..cut];
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(a);
    try out.appendSlice(a, rest);
    try out.append(a, '<');
    // every `Li<digits>` of the whole name, as re.findall(r"Li(\d+)") finds them
    var first = true;
    var i: usize = 0;
    while (i + 2 < name.len) {
        if (name[i] == 'L' and name[i + 1] == 'i' and std.ascii.isDigit(name[i + 2])) {
            var e = i + 2;
            while (e < name.len and std.ascii.isDigit(name[e])) e += 1;
            if (!first) try out.append(a, ',');
            first = false;
            try out.appendSlice(a, name[i + 2 .. e]);
            i = e;
        } else i += 1;
    }
    try out.append(a, '>');
    return out.toOwnedSlice(a);
}

pub const ReportError = error{ NoDecodeRows, NoLiveRows, BadTokens } || ParseError || std.mem.Allocator.Error || std.Io.Writer.Error;

fn byTimeDesc(_: void, x: Row, y: Row) bool {
    return -(x.t * @as(f64, @floatFromInt(x.count))) < -(y.t * @as(f64, @floatFromInt(y.count)));
}

/// Writes the report of the table `text` to `out`. `tokens` is the second command line argument, when there is one.
pub fn report(a: std.mem.Allocator, text: []const u8, tokens_arg: ?u64, out: *std.Io.Writer) ReportError!void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    const ar = arena.allocator();

    var decode: std.ArrayList(Row) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line0| {
        const line = std.mem.trimEnd(u8, line0, "\r");
        if (line.len == 0 or line[0] == '#') continue;
        const r = try parseLine(line);
        if (r.m == 0 or r.m == 1) try decode.append(ar, r);
    }
    if (decode.items.len == 0) return error.NoDecodeRows;

    var tokens: u64 = undefined;
    if (tokens_arg) |t| {
        tokens = t;
    } else {
        tokens = 0;
        var have_head = false;
        for (decode.items) |r| if (r.n >= 200000) { // the lm_head launch: once a token
            tokens = r.count;
            have_head = true;
            break;
        };
        if (!have_head) { // Counter.most_common(1): the first of the most frequent counts
            var best: usize = 0;
            for (decode.items, 0..) |r, i| {
                var c: usize = 0;
                for (decode.items) |q| c += @intFromBool(q.count == r.count);
                var before: usize = 0;
                for (decode.items[0..i]) |q| before += @intFromBool(q.count == r.count);
                if (before == 0 and c > best) {
                    best = c;
                    tokens = r.count;
                }
            }
        }
    }
    if (tokens == 0) return error.BadTokens;

    var live: std.ArrayList(Row) = .empty;
    for (decode.items) |r| if (r.count >= tokens / 2) {
        var x = r;
        x.t = if (x.rep != 0) x.rep else x.avg; // pipelined where replayed
        try live.append(ar, x);
    };
    if (live.items.len == 0) return error.NoLiveRows;

    const tk: f64 = @floatFromInt(tokens);
    var tot: f64 = 0;
    var launches: f64 = 0;
    var gaps: f64 = 0;
    for (live.items) |r| {
        const c: f64 = @floatFromInt(r.count);
        tot += r.t * c;
        launches += c;
        gaps += r.gap * c;
    }
    tot /= tk;
    launches /= tk;
    gaps /= tk;
    try out.print("{d} decode tokens, ", .{tokens});
    try pf.fieldF(out, launches, 0, 0);
    try out.writeAll(" launches a token; sum of kernel times (replayed, else synced) ");
    try pf.fieldF(out, tot / 1000, 0, 2);
    try out.writeAll(" ms a token, mean gaps between kernels ");
    try pf.fieldF(out, gaps / 1000, 0, 2);
    try out.writeAll(" ms a token (");
    try pf.fieldF(out, gaps / launches, 0, 1);
    try out.writeAll(" us a launch)\n");
    try out.print("{s:<34} {s:>10} {s:>5} {s:>6} {s:>5} {s:>5} {s:>7} {s:>6} {s:>7} {s:>5} {s:>6} {s:>6} {s:>7} {s:>5}\n", .{ "kernel", "grid", "blk", "n", "k", "/tok", "us", "min", "rep_us", "gap", "MB", "GB/s", "ms/tok", "%" });
    std.mem.sort(Row, live.items, {}, byTimeDesc);
    for (live.items) |r| {
        const per = @as(f64, @floatFromInt(r.count)) / tk;
        const us = r.t;
        const ms = us * per / 1000;
        const gbs: f64 = if (r.bytes != 0 and us != 0) r.bytes / us / 1000 else 0;
        const nm = try short(ar, r.name);
        const blk = r.block[0 .. std.mem.indexOfScalar(u8, r.block, ',') orelse r.block.len];
        try out.print("{s:<34} {s:>10} {s:>5} {d:>6} {d:>5}", .{ nm[0..@min(nm.len, 34)], r.grid, blk, r.n, r.k });
        const cols = [_]struct { f64, usize, u8 }{ .{ per, 5, 1 }, .{ us, 7, 1 }, .{ r.min, 6, 1 }, .{ r.rep, 7, 1 }, .{ r.gap, 5, 1 }, .{ r.bytes / 1e6, 6, 1 }, .{ gbs, 6, 0 }, .{ ms, 7, 3 }, .{ 100 * us * per / tot, 5, 1 } };
        for (cols) |c| {
            try out.writeByte(' ');
            try pf.fieldF(out, c[0], c[1], c[2]);
        }
        try out.writeByte('\n');
    }
}

pub fn main(init: std.process.Init) !u8 {
    const a = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    var err_buf: [512]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(init.io, &err_buf);
    const err = &stderr.interface;
    if (args.len < 2 or args.len > 3 or std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h")) {
        try err.writeAll("usage: tf-prof-report TABLE.txt [TOKENS]\n  a hipprof table (HIPPROF_LOG) of a decode run as a per-token breakdown\n");
        try err.flush();
        return if (args.len >= 2 and (std.mem.eql(u8, args[1], "--help") or std.mem.eql(u8, args[1], "-h"))) 0 else 2;
    }
    const tokens: ?u64 = if (args.len == 3) std.fmt.parseInt(u64, args[2], 10) catch {
        try err.print("tf-prof-report: TOKENS must be an integer, got '{s}'\n", .{args[2]});
        try err.flush();
        return 2;
    } else null;
    const text = std.Io.Dir.cwd().readFileAlloc(init.io, args[1], a, .limited(1 << 28)) catch |e| {
        try err.print("tf-prof-report: cannot read {s}: {s}\n", .{ args[1], @errorName(e) });
        try err.flush();
        return 1;
    };
    defer a.free(text);
    var buf: [1 << 14]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(init.io, &buf);
    report(a, text, tokens, &stdout.interface) catch |e| {
        try stdout.interface.flush();
        try err.print("tf-prof-report: {s}: {s}\n", .{ args[1], switch (e) {
            error.NoDecodeRows => "no decode rows (m = 0 or 1) in the table",
            error.NoLiveRows => "no kernel launched at least half as often as the token count",
            error.BadTokens => "the token count is not positive",
            error.ShortLine => "a table line has fewer than 14 columns",
            error.BadNumber => "a table column is not a number",
            else => @errorName(e),
        } });
        try err.flush();
        return 1;
    };
    try stdout.interface.flush();
    return 0;
}

const sample_table =
    "# name\tcount\tgrid\tblock\tshared\tm\tn\tk\tsides\tavg_us\tmin_us\ttotal_ms\trep_avg_us\tweight_bytes\tgap_us\n" ++
    "_ZN2tf4rocm13affine_streamINS0_5AffineELi4ELi64EEEvv\t400\t32,1,1\t256,1,1\t4096\t1\t248320\t2560\t2\t120.00\t100.00\t48.000\t90.50\t318000000\t2.50\n" ++
    "_ZN2tf4rocm10affine_dotINS0_5AffineELi8EEEvv\t800\t64,1,1\t128,1,1\t0\t1\t8192\t2560\t0\t30.25\t28.00\t24.200\t0.00\t11796480\t1.25\n" ++
    "rms_norm_f16\t801\t1,1,1\t1024,1,1\t0\t0\t0\t0\t0\t6.05\t5.00\t4.846\t0.00\t0\t0.75\n" ++
    "prefill_gemm\t40\t8,8,1\t256,1,1\t0\t512\t8192\t2560\t0\t900.00\t800.00\t36.000\t0.00\t0\t0.00\n" ++
    "rare_kernel\t3\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t4.00\t4.00\t0.012\t0.00\t0\t0.00\n";

test "short names" {
    const a = std.testing.allocator;
    const s = try short(a, "_ZN2tf4rocm13affine_streamINS0_5AffineELi4ELi64EEEvv");
    defer a.free(s);
    try std.testing.expectEqualStrings("affine_stream<4,64>", s);
    const p = try short(a, "rms_norm_f16");
    defer a.free(p);
    try std.testing.expectEqualStrings("rms_norm_f16", p);
    const q = try short(a, "_ZN2tf4rocm5thing");
    defer a.free(q);
    try std.testing.expectEqualStrings("thing<>", q);
}

test "a line with a missing gap column and short or malformed lines" {
    const r = try parseLine("k\t3\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t4.0\t4.0\t0.012\t0.0\t0");
    try std.testing.expectEqual(@as(f64, 0), r.gap);
    try std.testing.expectError(error.ShortLine, parseLine("k\t3\t1,1,1"));
    try std.testing.expectError(error.BadNumber, parseLine("k\tx\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t4.0\t4.0\t0.012\t0.0\t0"));
}

test "the report of a table equals prof_report.py's" {
    var aw: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer aw.deinit();
    try report(std.testing.allocator, sample_table, null, &aw.writer);
    try std.testing.expectEqualStrings(@embedFile("prof_report_expected.txt"), aw.written());
}

test "an explicit token count, and the count of the most frequent launch when there is no lm_head row" {
    const a = std.testing.allocator;
    var aw: std.Io.Writer.Allocating = .init(a);
    defer aw.deinit();
    try report(a, sample_table, 400, &aw.writer);
    try std.testing.expect(std.mem.startsWith(u8, aw.written(), "400 decode tokens, "));
    var bw: std.Io.Writer.Allocating = .init(a);
    defer bw.deinit();
    // no n >= 200000: 800 (two launches) wins over 40 / 3
    const no_head = "k1\t800\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t4.0\t4.0\t3.2\t0.0\t0\t0\n" ++
        "k2\t800\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t8.0\t8.0\t6.4\t0.0\t0\t0\n" ++ "k3\t40\t1,1,1\t64,1,1\t0\t1\t16\t16\t0\t8.0\t8.0\t6.4\t0.0\t0\t0\n";
    try report(a, no_head, null, &bw.writer);
    try std.testing.expect(std.mem.startsWith(u8, bw.written(), "800 decode tokens, 2 launches a token;"));
    var cw: std.Io.Writer.Allocating = .init(a);
    defer cw.deinit();
    try std.testing.expectError(error.NoDecodeRows, report(a, "# only prefill\nk\t1\t1,1,1\t1,1,1\t0\t512\t1\t1\t0\t1\t1\t1\t0\t0\t0\n", null, &cw.writer));
}
