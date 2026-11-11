//! The command line of the Zig gates: `--name value`, `--name=value`, flags and repeatable options, like the argparse of the Python tools they replace.
//! No positional arguments. `-h` / `--help` prints the tool's usage text (exit 0), any error prints `prog: error: ...` (exit 2).
const std = @import("std");

pub const Opt = struct {
    name: []const u8,
    /// false: a flag (`--logits`)
    value: bool = true,
    /// may be given several times (`--env K=V`)
    repeat: bool = false,
    /// the allowed values when not empty
    choices: []const []const u8 = &.{},
};

pub const Pair = struct { name: []const u8, value: []const u8 };

pub const Args = struct {
    pairs: []const Pair,

    pub fn get(self: Args, name: []const u8) ?[]const u8 {
        var i = self.pairs.len;
        while (i > 0) {
            i -= 1;
            if (std.mem.eql(u8, self.pairs[i].name, name)) return self.pairs[i].value;
        }
        return null;
    }

    pub fn flag(self: Args, name: []const u8) bool {
        return self.get(name) != null;
    }

    /// Every value given for `name`, in order.
    pub fn all(self: Args, a: std.mem.Allocator, name: []const u8) ![]const []const u8 {
        var out: std.ArrayList([]const u8) = .empty;
        for (self.pairs) |p| if (std.mem.eql(u8, p.name, name)) try out.append(a, p.value);
        return out.toOwnedSlice(a);
    }

    pub fn str(self: Args, name: []const u8, default: []const u8) []const u8 {
        return self.get(name) orelse default;
    }

    pub fn int(self: Args, comptime T: type, name: []const u8, default: T) !T {
        const v = self.get(name) orelse return default;
        return std.fmt.parseInt(T, v, 10) catch return error.BadInteger;
    }
};

pub const Parsed = union(enum) {
    ok: Args,
    /// the process exit status: 0 after --help, 2 after an error (already printed)
    exit: u8,
};

pub fn parse(a: std.mem.Allocator, argv: []const []const u8, prog: []const u8, opts: []const Opt, usage: []const u8, out: *std.Io.Writer, err: *std.Io.Writer) !Parsed {
    var pairs: std.ArrayList(Pair) = .empty;
    var i: usize = 0;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try out.writeAll(usage);
            try out.flush();
            return .{ .exit = 0 };
        }
        const eq = std.mem.indexOfScalar(u8, arg, '=');
        const name = if (std.mem.startsWith(u8, arg, "--")) (if (eq) |e| arg[0..e] else arg) else "";
        const opt = for (opts) |o| {
            if (std.mem.eql(u8, o.name, name)) break o;
        } else return fail(err, prog, usage, "unrecognized arguments: {s}", .{arg});
        var value: []const u8 = "";
        if (opt.value) {
            if (eq) |e| {
                value = arg[e + 1 ..];
            } else {
                i += 1;
                if (i == argv.len or (std.mem.startsWith(u8, argv[i], "--"))) return fail(err, prog, usage, "argument {s}: expected one argument", .{name});
                value = argv[i];
            }
            if (opt.choices.len > 0) {
                const ok = for (opt.choices) |c| {
                    if (std.mem.eql(u8, c, value)) break true;
                } else false;
                if (!ok) return fail(err, prog, usage, "argument {s}: invalid choice: '{s}'", .{ name, value });
            }
        } else if (eq != null) return fail(err, prog, usage, "argument {s}: ignored explicit argument", .{name});
        if (!opt.repeat) {
            var replaced = false;
            for (pairs.items) |*p| if (std.mem.eql(u8, p.name, opt.name)) {
                p.value = value; // the last one wins, like argparse
                replaced = true;
            };
            if (replaced) continue;
        }
        try pairs.append(a, .{ .name = opt.name, .value = value });
    }
    return .{ .ok = .{ .pairs = try pairs.toOwnedSlice(a) } };
}

fn fail(err: *std.Io.Writer, prog: []const u8, usage: []const u8, comptime fmt: []const u8, args: anytype) !Parsed {
    const first = std.mem.indexOfScalar(u8, usage, '\n') orelse usage.len;
    try err.print("{s}\n{s}: error: ", .{ usage[0..first], prog });
    try err.print(fmt ++ "\n", args);
    try err.flush();
    return .{ .exit = 2 };
}

test "options, equals form, flags, repeats and the last value winning" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opts = [_]Opt{ .{ .name = "--port" }, .{ .name = "--logits", .value = false }, .{ .name = "--env", .repeat = true }, .{ .name = "--prof", .choices = &.{ "each", "graph" } } };
    var sink: std.Io.Writer.Allocating = .init(a);
    const p = try parse(a, &.{ "--port", "1", "--env", "A=1", "--logits", "--env=B=2", "--port=18100", "--prof", "each" }, "t", &opts, "usage: t\n", &sink.writer, &sink.writer);
    const args = p.ok;
    try std.testing.expectEqualStrings("18100", args.get("--port").?);
    try std.testing.expect(args.flag("--logits"));
    try std.testing.expect(!args.flag("--nothing"));
    try std.testing.expectEqual(@as(u16, 18100), try args.int(u16, "--port", 0));
    const envs = try args.all(a, "--env");
    try std.testing.expectEqual(@as(usize, 2), envs.len);
    try std.testing.expectEqualStrings("B=2", envs[1]);
    try std.testing.expectEqualStrings("each", args.get("--prof").?);
}

test "errors and help" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const opts = [_]Opt{ .{ .name = "--port" }, .{ .name = "--prof", .choices = &.{ "each", "graph" } } };
    var sink: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 2), (try parse(a, &.{"--nope"}, "t", &opts, "usage: t\n", &sink.writer, &sink.writer)).exit);
    try std.testing.expect(std.mem.indexOf(u8, sink.written(), "t: error: unrecognized arguments: --nope") != null);
    try std.testing.expectEqual(@as(u8, 2), (try parse(a, &.{"--port"}, "t", &opts, "usage: t\n", &sink.writer, &sink.writer)).exit);
    try std.testing.expectEqual(@as(u8, 2), (try parse(a, &.{ "--prof", "events" }, "t", &opts, "usage: t\n", &sink.writer, &sink.writer)).exit);
    try std.testing.expectEqual(@as(u8, 2), (try parse(a, &.{"plain"}, "t", &opts, "usage: t\n", &sink.writer, &sink.writer)).exit);
    var help: std.Io.Writer.Allocating = .init(a);
    try std.testing.expectEqual(@as(u8, 0), (try parse(a, &.{"--help"}, "t", &opts, "usage: t\n", &help.writer, &help.writer)).exit);
    try std.testing.expectEqualStrings("usage: t\n", help.written());
}
