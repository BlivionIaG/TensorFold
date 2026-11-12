//! A per-kernel diagnostic armed by TF_DSV41_DIAG_FILE: dispatch counts by pipeline name, skipped or doubled kernels.

const std = @import("std");

const max_pipes = 1024;
const max_rules = 32;
const name_cap = 56;

const Pipe = struct { id: usize = 0, name: [name_cap]u8 = undefined, len: u8 = 0, skip: bool = false, twice: bool = false };
const Rule = struct { text: [name_cap]u8 = undefined, len: u8 = 0, twice: bool = false };

/// The file is re-read at each forward when it changes; lines: skip NAME, double NAME, nobarrier, dump, clear.
pub var armed: bool = false;
/// `nobarrier`: memory barriers are dropped (a serial encoder orders dispatches itself; a concurrent one would race).
pub var no_barrier: bool = false;
var pipes: [max_pipes]Pipe = undefined;
var npipes: usize = 0;
var counts: [max_pipes]u32 = @splat(0);
var groups: [max_pipes]u64 = @splat(0);
var rules: [max_rules]Rule = undefined;
var nrules: usize = 0;
var current: usize = 0;
var path: ?[*:0]const u8 = null;
var last_mtime: i128 = 0;
var checked = false;

/// Remember a pipeline's name so a dispatch can be attributed to it (called when the pipeline is built).
pub fn register(id: usize, name: []const u8) void {
    if (npipes >= max_pipes) return;
    var p = Pipe{ .id = id };
    p.len = @intCast(@min(name.len, name_cap));
    @memcpy(p.name[0..p.len], name[0..p.len]);
    pipes[npipes] = p;
    npipes += 1;
}

pub fn setCurrent(id: usize) void {
    if (armed) current = id;
}

fn find(id: usize) ?usize {
    for (pipes[0..npipes], 0..) |p, i| if (p.id == id) return i;
    return null;
}

/// The dispatch's fate: 0 drop it, 1 issue it once, 2 issue it twice. Counts it either way.
pub fn dispatch(n_groups: usize) u8 {
    const i = find(current) orelse return 1;
    counts[i] += 1;
    groups[i] += n_groups;
    if (pipes[i].skip) return 0;
    return if (pipes[i].twice) 2 else 1;
}

fn applyRules() void {
    for (pipes[0..npipes]) |*p| {
        p.skip = false;
        p.twice = false;
        for (rules[0..nrules]) |r| {
            if (std.mem.indexOf(u8, p.name[0..p.len], r.text[0..r.len]) == null) continue;
            if (r.twice) p.twice = true else p.skip = true;
        }
    }
}

fn dump() void {
    var buf: [16384]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buf);
    var total: u64 = 0;
    for (counts[0..npipes]) |c| total += c;
    w.print("diag dispatches {d}\n", .{total}) catch {};
    for (pipes[0..npipes], 0..) |p, i| {
        if (counts[i] == 0) continue;
        w.print("diag kernel {s} {d} {d}\n", .{ p.name[0..p.len], counts[i], groups[i] }) catch {};
    }
    std.debug.print("{s}", .{w.buffered()});
}

fn load(text: []const u8) void {
    nrules = 0;
    no_barrier = false;
    var want_dump = false;
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    while (lines.next()) |line| {
        var f = std.mem.tokenizeAny(u8, line, " \t");
        const verb = f.next() orelse continue;
        if (std.mem.eql(u8, verb, "dump")) want_dump = true;
        if (std.mem.eql(u8, verb, "clear")) nrules = 0;
        if (std.mem.eql(u8, verb, "nobarrier")) no_barrier = true;
        const arg = f.next() orelse continue;
        const twice = std.mem.eql(u8, verb, "double");
        if (!twice and !std.mem.eql(u8, verb, "skip")) continue;
        if (nrules >= max_rules) continue;
        var r = Rule{ .twice = twice };
        r.len = @intCast(@min(arg.len, name_cap));
        @memcpy(r.text[0..r.len], arg[0..r.len]);
        rules[nrules] = r;
        nrules += 1;
    }
    applyRules();
    if (want_dump) dump();
}

/// At the start of a forward: arm from the environment once, then re-read the file when it changed.
pub fn poll() void {
    if (!checked) {
        checked = true;
        path = std.c.getenv("TF_DSV41_DIAG_FILE");
        armed = path != null;
    }
    const p = path orelse return;
    var st: std.c.Stat = undefined;
    if (std.c.stat(p, &st) != 0) return;
    const mtime: i128 = @as(i128, st.mtime().sec) * 1_000_000_000 + st.mtime().nsec;
    if (mtime != last_mtime) {
        last_mtime = mtime;
        var buf: [4096]u8 = undefined;
        const fd = std.c.open(p, .{ .ACCMODE = .RDONLY });
        if (fd < 0) return;
        defer _ = std.c.close(fd);
        const n = std.c.read(fd, &buf, buf.len);
        if (n > 0) load(buf[0..@intCast(n)]);
    }
    // Counts are per forward: the dump shows the one before this call.
    @memset(&counts, 0);
    @memset(&groups, 0);
}

test "rules match by substring and a dump does not change them" {
    npipes = 0;
    register(1, "tf_exl3_matmul_decode_fma_t2");
    register(2, "dsv41_rope");
    load("skip exl3\ndouble rope\n");
    try std.testing.expect(pipes[0].skip and !pipes[0].twice);
    try std.testing.expect(pipes[1].twice and !pipes[1].skip);
    load("clear\n");
    try std.testing.expect(!pipes[0].skip and !pipes[1].twice);
}
