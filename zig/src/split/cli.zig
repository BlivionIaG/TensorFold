//! `tf-offload probe | plan`: the offload layer's commands. `probe` measures the Mac, the external device and the link
//! and writes the receipt; `plan` reads a model's config.json and a receipt and prints every placement the planner
//! considered, then writes placement.json for the one `auto` picks. Neither command loads weights or runs a model.
const std = @import("std");
const device = @import("device.zig");
const geometry = @import("geometry.zig");
const receipt = @import("receipt.zig");
const placement = @import("placement.zig");
const planner = @import("planner.zig");
const probe = @import("probe.zig");

const Allocator = std.mem.Allocator;
const W = std.Io.Writer;

/// One external backend this binary can open (the Metal build always has `mock`; the HIP build adds `hip`).
pub const Backend = struct {
    name: []const u8,
    open: *const fn (gpa: Allocator, index: c_int) anyerror!device.ExternalDevice,
};

pub const Env = struct {
    io: std.Io,
    a: Allocator,
    out: *W,
    err: *W,
    host: probe.Host,
    backends: []const Backend,
};

pub const usage =
    \\usage: tf-offload probe --device NAME [--index N] [--iterations N] [--mac-decode-gbps X] [--out FILE]
    \\       tf-offload plan --config FILE --receipt FILE [--context N] [--streams N] [--mac-budget-gib G]
    \\                       [--fallback none|metal] [--window-mib N] [--allow-unexecutable] [--out FILE]
    \\probe writes the tensorfold-offload-probe/1 receipt (stdout without --out); plan prints the candidate placements and
    \\writes the tensorfold-split/1 file `auto` chose. Exit codes: 0 ok, 1 refused or invalid, 2 usage.
    \\
;

fn flag(args: []const []const u8, name: []const u8) ?[]const u8 {
    for (args, 0..) |x, i| {
        if (std.mem.eql(u8, x, name)) return if (i + 1 < args.len) args[i + 1] else null;
        if (std.mem.startsWith(u8, x, name) and x.len > name.len and x[name.len] == '=') return x[name.len + 1 ..];
    }
    return null;
}

fn has(args: []const []const u8, name: []const u8) bool {
    for (args) |x| if (std.mem.eql(u8, x, name)) return true;
    return false;
}

fn number(comptime T: type, args: []const []const u8, name: []const u8, default: T) ?T {
    const v = flag(args, name) orelse return default;
    return switch (@typeInfo(T)) {
        .float => std.fmt.parseFloat(T, v) catch null,
        else => std.fmt.parseInt(T, v, 10) catch null,
    };
}

pub fn run(env: Env, args: []const []const u8) !u8 {
    if (args.len >= 1 and std.mem.eql(u8, args[0], "probe")) return probeCmd(env, args[1..]);
    if (args.len >= 1 and std.mem.eql(u8, args[0], "plan")) return planCmd(env, args[1..]);
    try env.err.writeAll(usage);
    return 2;
}

fn refuse(env: Env, comptime fmt: []const u8, a: anytype) !u8 {
    try env.err.print("tf-offload: " ++ fmt ++ "\n", a);
    return 1;
}

fn probeCmd(env: Env, args: []const []const u8) !u8 {
    const name = flag(args, "--device") orelse {
        try env.err.writeAll(usage);
        return 2;
    };
    const index = number(u32, args, "--index", 0) orelse return bad(env, "--index");
    const iterations = number(usize, args, "--iterations", 400) orelse return bad(env, "--iterations");
    const gbps = number(f64, args, "--mac-decode-gbps", 0) orelse return bad(env, "--mac-decode-gbps");
    var backend: ?Backend = null;
    for (env.backends) |b| if (std.mem.eql(u8, b.name, name)) {
        backend = b;
    };
    const b = backend orelse return refuse(env, "this build has no `{s}` device", .{name});
    const dev = b.open(env.a, @intCast(index)) catch |err| return refuse(env, "cannot open the {s} device: {s}", .{ name, @errorName(err) });
    defer dev.deinit();
    const r = probe.run(env.a, env.host, dev, .{ .iterations = iterations, .mac_decode_bps = if (gbps > 0) gbps * 1e9 else null }) catch |err|
        return refuse(env, "the probe failed: {s}", .{@errorName(err)});
    const bytes = try receipt.write(env.a, r);
    if (flag(args, "--out")) |path| {
        std.Io.Dir.cwd().writeFile(env.io, .{ .sub_path = path, .data = bytes }) catch |err| return refuse(env, "cannot write {s}: {s}", .{ path, @errorName(err) });
        try summary(env.out, r);
    } else {
        try env.out.writeAll(bytes);
        try env.out.writeAll("\n");
    }
    return 0;
}

fn bad(env: Env, name: []const u8) !u8 {
    try env.err.print("tf-offload: {s} needs a number\n", .{name});
    return 2;
}

fn us(ns: f64) f64 {
    return ns / 1e3;
}

fn summary(w: *W, r: receipt.Receipt) !void {
    try w.print("{s} over {s} ({s}, {s}): window round trip halves {d:.1} us to the device and {d:.1} us to the host with Metal's wake {d:.1} us\n", .{
        r.mac.name,
        r.device.name,
        r.device.backend,
        r.device.arch,
        us(r.link.hop_to_device_ns.p50),
        us(r.link.hop_to_host_ns.p50),
        us(r.link.metal_wake_ns.p50),
    });
    try w.print("bulk {d:.2} GB/s to the device, {d:.2} GB/s back; Mac decode {d:.0} GB/s ({s}); device {d:.1} GiB, efficiency {d:.2} ({s})\n", .{
        r.link.h2d_bps.value / 1e9,
        r.link.d2h_bps.value / 1e9,
        r.mac.decode_bandwidth_bps.value / 1e9,
        @tagName(r.mac.decode_bandwidth_bps.source),
        @as(f64, @floatFromInt(r.device.vram_bytes.value)) / (1 << 30),
        r.device.decode_efficiency.value,
        @tagName(r.device.decode_efficiency.source),
    });
}

fn readFile(env: Env, path: []const u8) ?[]u8 {
    return std.Io.Dir.cwd().readFileAlloc(env.io, path, env.a, .limited(16 << 20)) catch null;
}

fn planCmd(env: Env, args: []const []const u8) !u8 {
    const config_path = flag(args, "--config") orelse {
        try env.err.writeAll(usage);
        return 2;
    };
    const receipt_path = flag(args, "--receipt") orelse {
        try env.err.writeAll(usage);
        return 2;
    };
    const context = number(u32, args, "--context", 8192) orelse return bad(env, "--context");
    const streams = number(u32, args, "--streams", 1) orelse return bad(env, "--streams");
    const window = number(u32, args, "--window-mib", 64) orelse return bad(env, "--window-mib");
    const budget_gib = number(f64, args, "--mac-budget-gib", 0) orelse return bad(env, "--mac-budget-gib");
    const fallback: placement.Fallback = if (flag(args, "--fallback")) |v|
        (if (std.mem.eql(u8, v, "metal")) .metal else if (std.mem.eql(u8, v, "none")) .none else return bad(env, "--fallback none|metal"))
    else
        .none;
    const config_bytes = readFile(env, config_path) orelse return refuse(env, "cannot read {s}", .{config_path});
    const receipt_bytes = readFile(env, receipt_path) orelse return refuse(env, "cannot read {s}", .{receipt_path});
    var diag: geometry.Diag = .{};
    const model = geometry.fromConfigJsonDiag(env.a, config_bytes, &diag) catch |err|
        return refuse(env, "{s}: {s} ({s})", .{ config_path, @errorName(err), diag.field });
    var rdiag: receipt.Diag = .{};
    const parsed = receipt.parseDiag(env.a, receipt_bytes, &rdiag) catch |err|
        return refuse(env, "{s}: {s} at {s}", .{ receipt_path, @errorName(err), rdiag.path() });
    const r = parsed.value;
    const req: planner.Request = .{
        .context = context,
        .streams = streams,
        .mac_budget_override = if (budget_gib > 0) @intFromFloat(budget_gib * (1 << 30)) else null,
    };
    const candidates = try planner.plan(env.a, model, req, planner.Mac.fromReceipt(r), planner.External.fromReceipt(r), planner.Link.fromReceipt(r));
    try env.out.print("{s}: {d} layers, weights {d:.2} GiB; context {d}, {d} stream(s); Mac budget {d:.1} GiB, {s} {d:.1} GiB\n", .{
        model.name,
        model.layers.len,
        @as(f64, @floatFromInt(model.weights())) / (1 << 30),
        context,
        streams,
        @as(f64, @floatFromInt(candidates.mac_budget)) / (1 << 30),
        r.device.name,
        @as(f64, @floatFromInt(candidates.vram_usable)) / (1 << 30),
    });
    if (r.device.backend.len != 0 and std.mem.eql(u8, r.device.backend, "mock"))
        try env.out.writeAll("note: this receipt is from the mock device; its link numbers say nothing about a card\n");
    for (candidates.items) |p| {
        if (!p.fits) {
            try env.out.print("  {s:<24} does not fit ({s})\n", .{ @tagName(p.mode), p.note });
            continue;
        }
        try env.out.print("  {s:<24} {d:>6.1} tok/s  {s}{s}\n", .{ @tagName(p.mode), p.tok_s, p.note, if (p.executable) "" else "  [not executable in this build]" });
    }
    const chosen = planner.choose(candidates, .{ .allow_unexecutable = has(args, "--allow-unexecutable") }) catch {
        const s = candidates.shortfall();
        try env.err.print("tf-offload: nothing fits: Mac needs {d:.1} of {d:.1} GiB, the device {d:.1} of {d:.1} GiB, both together {d:.1} of {d:.1} GiB\n", .{
            gib(s.mac.needed),      gib(s.mac.available),
            gib(s.vram.needed),     gib(s.vram.available),
            gib(s.combined.needed), gib(s.combined.available),
        });
        if (s.unexecutable_fits) try env.err.writeAll("tf-offload: a placement exists that this build cannot execute yet (--allow-unexecutable lists it)\n");
        return 1;
    };
    try env.out.print("auto: {s}, {d:.1} tok/s\n", .{ @tagName(chosen.mode), chosen.tok_s });
    const file = chosen.toPlacement(.{ .backend = r.device.backend, .index = 0, .window_mib = window }, req, fallback) catch
        return refuse(env, "fallback metal needs a Mac that holds the whole model", .{});
    placement.validate(file, model.layers.len) catch |err| return refuse(env, "internal: the plan does not validate: {s}", .{@errorName(err)});
    const bytes = try placement.write(env.a, file);
    if (flag(args, "--out")) |path| {
        std.Io.Dir.cwd().writeFile(env.io, .{ .sub_path = path, .data = bytes }) catch |err| return refuse(env, "cannot write {s}: {s}", .{ path, @errorName(err) });
    } else {
        try env.out.writeAll(bytes);
        try env.out.writeAll("\n");
    }
    return 0;
}

fn gib(b: u64) f64 {
    return @as(f64, @floatFromInt(b)) / (1 << 30);
}

/// The entry both binaries share: argv after the program name, stdout and stderr.
pub fn main(init: std.process.Init, argv: []const [:0]const u8, host: probe.Host, backends: []const Backend) !u8 {
    const a = init.arena.allocator();
    var out_buf: [64 << 10]u8 = undefined;
    var err_buf: [4 << 10]u8 = undefined;
    var out = std.Io.File.stdout().writerStreaming(init.io, &out_buf);
    var err = std.Io.File.stderr().writerStreaming(init.io, &err_buf);
    const args = try a.alloc([]const u8, argv.len);
    for (args, argv) |*x, y| x.* = y;
    const code = try run(.{ .io = init.io, .a = a, .out = &out.interface, .err = &err.interface, .host = host, .backends = backends }, args);
    try out.interface.flush();
    try err.interface.flush();
    return code;
}

test "plan prints the candidates for a config and a receipt and writes a valid placement" {
    const fixtures = @import("fixtures.zig");
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const sample = try receipt.write(a, receipt.sample);
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "config.json", .data = fixtures.d4_json });
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "receipt.json", .data = sample });
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    var dummy: u8 = 0;
    const host: probe.Host = .{ .ptr = &dummy, .info = undefined, .wake = undefined, .wrap = undefined };
    const env: Env = .{ .io = std.testing.io, .a = a, .out = &out.writer, .err = &err.writer, .host = host, .backends = &.{} };
    // relative paths resolve against the current directory: run through the tmp dir's absolute path
    const base = try tmp.dir.realPathFileAlloc(std.testing.io, ".", a);
    const cfg = try std.fs.path.join(a, &.{ base, "config.json" });
    const rcp = try std.fs.path.join(a, &.{ base, "receipt.json" });
    const outp = try std.fs.path.join(a, &.{ base, "placement.json" });
    const code = try run(env, &.{ "plan", "--config", cfg, "--receipt", rcp, "--context", "8192", "--streams", "2", "--out", outp });
    try std.testing.expectEqual(@as(u8, 0), code);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "all_metal") != null);
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "auto: ") != null);
    const written = try tmp.dir.readFileAlloc(std.testing.io, "placement.json", a, .limited(1 << 20));
    const parsed = try placement.parse(a, written);
    try placement.validate(parsed.value, 32);
}

test "plan refuses a config it cannot read, and a missing flag is usage" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var out: std.Io.Writer.Allocating = .init(a);
    var err: std.Io.Writer.Allocating = .init(a);
    var dummy: u8 = 0;
    const host: probe.Host = .{ .ptr = &dummy, .info = undefined, .wake = undefined, .wrap = undefined };
    const env: Env = .{ .io = std.testing.io, .a = a, .out = &out.writer, .err = &err.writer, .host = host, .backends = &.{} };
    try std.testing.expectEqual(@as(u8, 1), try run(env, &.{ "plan", "--config", "/nonexistent.json", "--receipt", "/nonexistent2.json" }));
    try std.testing.expectEqual(@as(u8, 2), try run(env, &.{"plan"}));
    try std.testing.expectEqual(@as(u8, 2), try run(env, &.{"frobnicate"}));
    try std.testing.expectEqual(@as(u8, 1), try run(env, &.{ "probe", "--device", "nope" }));
}
