//! tf-moe-gate: the greedy gate of a running `tensorfold-native serve` (Qwen3.6-35B-A3B) against tools/vega-macos/moe_ref.py's mlx_lm references
//! (the port of moe_gate.py; the references themselves stay Python, they call mlx_lm).
//!
//!     tf-moe-gate --log SERVER.LOG [--port 18099] [--ref ~/.cache/tensorfold-vega/ref-moe.json] [--tag NAME] [--out DIR]
//!
//! One chat request a prompt (temperature 0, thinking off, max_tokens = the reference's length); the server log's `done <id> ... sha=` is the sha of the
//! generated token ids, which must be the reference's. Prints per prompt the first diverging character when it differs. Also writes the replies' text to
//! DIR/TAG.json so two runs (cache on / TF_EXPERT_COPIES=0) can be compared with --against OTHER.json (texts and shas equal).
//! Exit status 0 when every sha matches (or, with --against, when the two runs are identical).
const std = @import("std");
const cli = @import("cli.zig");
const lib = @import("gate_lib.zig");
const pf = @import("pyfmt.zig");
const pyjson = @import("pyjson.zig");
const Value = std.json.Value;

pub const usage =
    \\usage: tf-moe-gate --log LOG [--port PORT] [--ref REF] [--tag TAG] [--out OUT] [--against AGAINST] [--rounds ROUNDS]
    \\
    \\greedy gate of a running tensorfold-native serve against the mlx_lm references of moe_ref.py
    \\
    \\options:
    \\  --log LOG          the server's log (its `done <id> ... sha=` lines); required
    \\  --port PORT        the server's port (default 18099)
    \\  --ref REF          the references (default ~/.cache/tensorfold-vega/ref-moe.json)
    \\  --tag TAG          name of this run's DIR/TAG.json (default run)
    \\  --out OUT          directory of DIR/TAG.json (default ~/.cache/tensorfold-vega/bench)
    \\  --against AGAINST  a previous run's JSON: compare with it instead of the mlx references
    \\  --rounds ROUNDS    repeat the prompt set (the cache warms across repeats; every repeat must match)
    \\
;

const opts = [_]cli.Opt{
    .{ .name = "--log" }, .{ .name = "--port" }, .{ .name = "--ref" }, .{ .name = "--tag" }, .{ .name = "--out" }, .{ .name = "--against" }, .{ .name = "--rounds" },
};

/// A sha the way the Python prints it: the 12 digits or `None`.
fn shaText(s: ?[]const u8) []const u8 {
    return s orelse "None";
}

fn field(obj: Value, key: []const u8) ?Value {
    if (obj != .object) return null;
    return obj.object.get(key);
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var stderr = std.Io.File.stderr().writerStreaming(io, &err_buf);
    const out = &stdout.interface;
    const err = &stderr.interface;

    const parsed = try cli.parse(a, argv[1..], "tf-moe-gate", &opts, usage, out, err);
    const args = switch (parsed) {
        .ok => |p| p,
        .exit => |code| return code,
    };
    const log_path = args.get("--log") orelse {
        try err.writeAll("usage: tf-moe-gate --log LOG [--port PORT] ...\ntf-moe-gate: error: the following arguments are required: --log\n");
        try err.flush();
        return 2;
    };
    const port = args.int(u16, "--port", 18099) catch {
        try err.writeAll("tf-moe-gate: error: argument --port: invalid int value\n");
        try err.flush();
        return 2;
    };
    const rounds = args.int(u32, "--rounds", 1) catch {
        try err.writeAll("tf-moe-gate: error: argument --rounds: invalid int value\n");
        try err.flush();
        return 2;
    };
    const ref_path = if (args.get("--ref")) |r| r else try lib.expandUser(a, init.environ_map, "~/.cache/tensorfold-vega/ref-moe.json");
    const out_dir = if (args.get("--out")) |r| r else try lib.expandUser(a, init.environ_map, "~/.cache/tensorfold-vega/bench");
    const tag = args.str("--tag", "run");

    return gate(a, io, out, err, .{ .log = log_path, .port = port, .ref = ref_path, .out_dir = out_dir, .tag = tag, .against = args.get("--against"), .rounds = rounds }) catch |e| {
        try err.print("tf-moe-gate: {s}\n", .{@errorName(e)});
        try err.flush();
        return 1;
    };
}

pub const Options = struct { log: []const u8, port: u16, ref: []const u8, out_dir: []const u8, tag: []const u8, against: ?[]const u8, rounds: u32 };

fn loadJson(a: std.mem.Allocator, io: std.Io, path: []const u8) !Value {
    return std.json.parseFromSliceLeaky(Value, a, try lib.readFile(a, io, path), .{});
}

fn gate(a: std.mem.Allocator, io: std.Io, out: *std.Io.Writer, err: *std.Io.Writer, o: Options) !u8 {
    const ref = try loadJson(a, io, o.ref);
    if (ref != .array) {
        try err.print("tf-moe-gate: {s} is not a JSON list of references\n", .{o.ref});
        try err.flush();
        return 1;
    }
    const other: ?Value = if (o.against) |p| try loadJson(a, io, p) else null;

    var runs = std.json.Array.init(a);
    var ok = true;
    for (0..o.rounds) |rnd| {
        for (ref.array.items, 0..) |r, i| {
            const t0 = lib.wallSeconds(io);
            const question = (field(r, "question") orelse return error.ReferenceWithoutQuestion).string;
            const tokens = (field(r, "tokens") orelse return error.ReferenceWithoutTokens).array;
            const body = try lib.chatBody(a, question, @intCast(tokens.items.len), false);
            const reply = try lib.chat(a, io, o.port, body, null);
            const text = reply.content;
            const sha = try shaOf(a, io, o.log, reply.id);
            var want: ?[]const u8 = null; // null: the previous run had no sha either (Python's None)
            var want_text: ?[]const u8 = null;
            if (other) |ov| {
                if (ov != .array or runs.items.len >= ov.array.items.len) return error.AgainstRunTooShort;
                const prev = ov.array.items[runs.items.len];
                const w = field(prev, "sha") orelse return error.AgainstRunWithoutSha;
                want = if (w == .string) w.string else null;
                const wt = field(prev, "text") orelse return error.AgainstRunWithoutText;
                want_text = wt.string;
            } else want = (field(r, "sha") orelse return error.ReferenceWithoutSha).string;
            const sha_equal = if (sha) |s| (if (want) |w| std.mem.eql(u8, s, w) else false) else want == null;
            const same = sha_equal and (want_text == null or std.mem.eql(u8, want_text.?, text));
            ok = ok and same;
            var tb: [pf.buf_len]u8 = undefined;
            try out.print("round {d} prompt {d}: {d} tokens {s} s sha {s} want {s} {s}", .{ rnd, i, reply.completion_tokens, pf.fixed(&tb, lib.wallSeconds(io) - t0, 1, false), shaText(sha), shaText(want), if (same) "ok" else "FAIL" });
            if (!same) try out.print("  DIVERGES at char {d}", .{lib.diverge(text, want_text orelse (field(r, "text") orelse return error.ReferenceWithoutText).string)});
            try out.writeByte('\n');
            try out.flush();
            var obj: std.json.ObjectMap = .empty;
            try obj.put(a, "question", .{ .string = question });
            try obj.put(a, "sha", if (sha) |s| .{ .string = s } else .null);
            try obj.put(a, "text", .{ .string = text });
            try runs.append(.{ .object = obj });
        }
    }
    try std.Io.Dir.cwd().createDirPath(io, o.out_dir);
    const path = try std.fmt.allocPrint(a, "{s}/{s}.json", .{ o.out_dir, o.tag });
    var w: std.Io.Writer.Allocating = .init(a);
    try pyjson.dumps(&w.writer, .{ .array = runs });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = w.written() });
    try out.writeAll(if (ok) "PASS\n" else "FAIL\n");
    try out.flush();
    return if (ok) 0 else 1;
}

/// The Python's sha_of: up to 20 reads of the log, 0.25 s apart, for the sha of the reply's `done` line.
fn shaOf(a: std.mem.Allocator, io: std.Io, log_path: []const u8, id: []const u8) !?[]const u8 {
    const tmp = std.heap.smp_allocator;
    for (0..20) |_| {
        const log = try lib.readFile(tmp, io, log_path);
        defer tmp.free(log);
        if (lib.shaOfLog(log, id, false)) |s| return try a.dupe(u8, s);
        lib.sleepSeconds(io, 0.25);
    }
    return null;
}

test "a sha that is not there prints as None and never matches a reference" {
    try std.testing.expectEqualStrings("None", shaText(null));
    try std.testing.expectEqualStrings("0123456789ab", shaText("0123456789ab"));
}
