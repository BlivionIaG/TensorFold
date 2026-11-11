//! gate_check VEGA_GATE MOE_GATE FAKE DIR: runs tf-vega-gate and tf-moe-gate as processes against gate_fake (a stand-in for tensorfold-native and tf-qwen35-test)
//! and checks their output, exit status and results files, passing and failing. Exit status 0 when every check holds. Files land in DIR; no card is involved.
const std = @import("std");
const fake = @import("gate_fake.zig");
const npy = @import("npy.zig");

var failures: usize = 0;

fn check(ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (ok) return;
    failures += 1;
    std.debug.print("gate_check: FAIL " ++ fmt ++ "\n", args);
}

fn contains(label: []const u8, hay: []const u8, needle: []const u8) void {
    check(std.mem.indexOf(u8, hay, needle) != null, "{s}: expected '{s}' in:\n{s}", .{ label, needle, hay });
}

const Ran = struct { code: u8, stdout: []const u8, stderr: []const u8 };

const Ctx = struct {
    a: std.mem.Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    dir: []const u8,
    next_port: u16,

    fn port(c: *Ctx) u16 {
        c.next_port += 1;
        return c.next_port;
    }

    fn path(c: *Ctx, comptime fmt: []const u8, args: anytype) ![]u8 {
        const rel = try std.fmt.allocPrint(c.a, fmt, args);
        return std.fmt.allocPrint(c.a, "{s}/{s}", .{ c.dir, rel });
    }

    fn run(c: *Ctx, argv: []const []const u8, env: []const [2][]const u8) !Ran {
        var e = try c.base_env.clone(c.a);
        for (env) |kv| try e.put(kv[0], kv[1]);
        const r = try std.process.run(c.a, c.io, .{ .argv = argv, .environ_map = &e });
        return .{ .code = switch (r.term) {
            .exited => |code| code,
            else => 255,
        }, .stdout = r.stdout, .stderr = r.stderr };
    }

    fn write(c: *Ctx, p: []const u8, data: []const u8) !void {
        try std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = p, .data = data });
    }

    fn lastLine(c: *Ctx, p: []const u8) ![]const u8 {
        const text = try std.Io.Dir.cwd().readFileAlloc(c.io, p, c.a, .limited(1 << 24));
        const t = std.mem.trimEnd(u8, text, "\n");
        return t[(std.mem.lastIndexOfScalar(u8, t, '\n') orelse return t) + 1 ..];
    }
};

fn refDir(c: *Ctx, name: []const u8) ![]u8 {
    const a = c.a;
    const dir = try c.path("{s}", .{name});
    try std.Io.Dir.cwd().createDirPath(c.io, dir);
    // ref.json: one entry carries its own sha, the other only its tokens (the gate hashes them)
    const nothink = try fake.tokens(a, false, 8, 0);
    const think = try fake.tokens(a, true, 96, 0);
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.writeAll("{\"nothink\": {\"tokens\": [");
    for (nothink, 0..) |t, i| try w.writer.print("{s}{d}", .{ if (i > 0) ", " else "", t });
    try w.writer.print("]}}, \"think96\": {{\"tokens\": [0], \"sha\": \"{s}\"}}}}", .{try fake.shaOf(a, think)});
    try c.write(try std.fmt.allocPrint(a, "{s}/ref.json", .{dir}), w.written());
    try c.write(try std.fmt.allocPrint(a, "{s}/ids.npy", .{dir}), try npy.encode(a, "<i8", &.{3}, std.mem.sliceAsBytes(&[_]i64{ 1, 2, 3 })));
    try c.write(try std.fmt.allocPrint(a, "{s}/mlx_logits.npy", .{dir}), try fake.logitsFile(a, false));
    return dir;
}

fn vegaGate(c: *Ctx, gate: []const u8, fake_exe: []const u8, refs: []const u8, tag: []const u8, extra: []const []const u8, env: []const [2][]const u8) !Ran {
    var argv: std.ArrayList([]const u8) = .empty;
    try argv.appendSlice(c.a, &.{ gate, "--bin", fake_exe, "--test-bin", fake_exe, "--model", "fake-model", "--ref", refs, "--out", try c.path("out", .{}), "--tag", tag, "--port", try std.fmt.allocPrint(c.a, "{d}", .{c.port()}), "--lib", "/nonexistent/libamdhip64.dylib", "--backend", "mock" });
    try argv.appendSlice(c.a, extra);
    return c.run(argv.items, env);
}

fn vegaScenarios(c: *Ctx, gate: []const u8, fake_exe: []const u8) !void {
    const refs = try refDir(c, "ref");
    const results = try c.path("out/results.jsonl", .{});
    const want_nothink = try fake.shaOf(c.a, try fake.tokens(c.a, false, 8, 0));
    const want_think = try fake.shaOf(c.a, try fake.tokens(c.a, true, 96, 0));

    // 1. everything matches: shas, speed, logits
    {
        const r = try vegaGate(c, gate, fake_exe, refs, "pass", &.{ "--speed", "2", "--story-tokens", "4", "--logits", "--policy", "stream=reference", "--env", "TF_X=1" }, &.{});
        check(r.code == 0, "vega pass: exit {d}\n{s}\n{s}", .{ r.code, r.stdout, r.stderr });
        contains("vega pass", r.stdout, try std.fmt.allocPrint(c.a, "PASS nothink: 8 tokens identical to mlx_lm (sha {s})\n", .{want_nothink}));
        contains("vega pass", r.stdout, try std.fmt.allocPrint(c.a, "PASS think96: 96 tokens identical to mlx_lm (sha {s})\n", .{want_think}));
        contains("vega pass", r.stdout, "speed run 0: 4 tokens, wall ");
        contains("vega pass", r.stdout, ", tensorfold {'engine': 'fake', 'tokens_per_second': 83.5, 'seconds': 1.25, 'drafts': False, 'widths': []}\n");
        contains("vega pass", r.stdout, "decode tok/s (median of 2): 84.5\n");
        contains("vega pass", r.stdout, "rows 4  cosine min 1.000000 mean 1.000000  argmax equal 4/4  mean|diff| 0.0000 \n"); // the trailing space is the Python print's
        contains("vega pass", r.stdout, "TF_POLICY': 'stream=reference'");
        contains("vega pass", r.stdout, "'TF_X': '1'");
        contains("vega pass", r.stdout, "server up in ");
        check(std.mem.endsWith(u8, r.stdout, "GATE PASS\n"), "vega pass: ends with GATE PASS:\n{s}", .{r.stdout});
        const line = try c.lastLine(results);
        check(std.mem.startsWith(u8, line, "{\"tag\": \"pass\", \"policy\": \"stream=reference\", \"env\": [\"TF_X=1\"], \"ok\": true, \"time\": \""), "results.jsonl head: {s}", .{line});
        contains("results.jsonl", line, try std.fmt.allocPrint(c.a, "\"nothink\": {{\"sha\": \"{s}\", \"want\": \"{s}\", \"tokens\": 8}}", .{ want_nothink, want_nothink }));
        contains("results.jsonl", line, "\"decode_tok_s\": 84.5, \"logits_seconds\": ");
        contains("results.jsonl", line, "\"logits\": \"rows 4  cosine min 1.000000 mean 1.000000  argmax equal 4/4  mean|diff| 0.0000\"}");
        contains("results.jsonl", line, "\"speed\": [{\"completion_tokens\": 4, \"wall\": ");
        contains("results.jsonl", line, "\"tensorfold\": {\"engine\": \"fake\", \"tokens_per_second\": 83.5, \"seconds\": 1.25, \"drafts\": false, \"widths\": []}}");
    }
    // 2. the server generates other tokens: both sha checks fail, the gate fails
    {
        const r = try vegaGate(c, gate, fake_exe, refs, "variant", &.{ "--speed", "0" }, &.{.{ "FAKE_VARIANT", "1" }});
        check(r.code == 1, "vega variant: exit {d}", .{r.code});
        contains("vega variant", r.stdout, "FAIL nothink: token sha ");
        contains("vega variant", r.stdout, " != mlx_lm ");
        contains("vega variant", r.stdout, "FAIL think96: token sha ");
        check(std.mem.endsWith(u8, r.stdout, "GATE FAIL\n"), "vega variant: ends with GATE FAIL", .{});
        contains("results.jsonl", try c.lastLine(results), "\"ok\": false");
    }
    // 3. a server that dies on start: its log is printed, exit 1, no results line
    {
        const before = try c.lastLine(results);
        const r = try vegaGate(c, gate, fake_exe, refs, "dead", &.{"--speed=0"}, &.{.{ "FAKE_EXIT", "1" }});
        check(r.code == 1, "vega dead: exit {d}", .{r.code});
        contains("vega dead", r.stdout, "boom: fake server dies on start");
        contains("vega dead", r.stderr, "server exited");
        check(std.mem.eql(u8, before, try c.lastLine(results)), "vega dead: results.jsonl changed", .{});
    }
    // 4. logits that differ: the cosine / argmax gate fails (token checks skipped)
    {
        const r = try vegaGate(c, gate, fake_exe, refs, "flip", &.{ "--no-gate", "--speed", "0", "--logits" }, &.{.{ "FAKE_FLIP", "1" }});
        check(r.code == 1, "vega flip: exit {d}", .{r.code});
        contains("vega flip", r.stdout, "argmax equal 0/4");
        contains("vega flip", r.stdout, "FAIL logits cosine/argmax gate\n");
        check(std.mem.indexOf(u8, r.stdout, "PASS nothink") == null, "vega flip: --no-gate still checked tokens", .{});
    }
    // 5. the logits program fails: its output tail is reported
    {
        const r = try vegaGate(c, gate, fake_exe, refs, "logitsfail", &.{ "--no-gate", "--speed", "0", "--logits" }, &.{.{ "FAKE_LOGITS_FAIL", "1" }});
        check(r.code == 1, "vega logits fail: exit {d}", .{r.code});
        contains("vega logits fail", r.stdout, "FAIL logits run failed: fake logits written");
    }
    // 6. usage errors are exit 2, --help is exit 0
    {
        const r = try c.run(&.{ gate, "--prof", "bogus" }, &.{});
        check(r.code == 2, "vega bad choice: exit {d}", .{r.code});
        contains("vega bad choice", r.stderr, "invalid choice: 'bogus'");
        const h = try c.run(&.{ gate, "--help" }, &.{});
        check(h.code == 0, "vega help: exit {d}", .{h.code});
        contains("vega help", h.stdout, "usage: tf-vega-gate");
    }
}

fn startFake(c: *Ctx, fake_exe: []const u8, port: u16, log: []const u8, variant: []const u8) !std.process.Child {
    var e = try c.base_env.clone(c.a);
    try e.put("FAKE_VARIANT", variant);
    const f = try std.Io.Dir.cwd().createFile(c.io, log, .{});
    defer f.close(c.io);
    const child = try std.process.spawn(c.io, .{ .argv = &.{ fake_exe, "serve", "m", "--port", try std.fmt.allocPrint(c.a, "{d}", .{port}) }, .environ_map = &e, .stdin = .ignore, .stdout = .{ .file = f }, .stderr = .{ .file = f } });
    var waited: usize = 0;
    while (waited < 200) : (waited += 1) {
        const text = std.Io.Dir.cwd().readFileAlloc(c.io, log, c.a, .limited(1 << 20)) catch "";
        if (std.mem.indexOf(u8, text, "serving ") != null) return child;
        c.io.sleep(.fromMilliseconds(50), .awake) catch {};
    }
    return error.FakeDidNotStart;
}

fn stopFake(c: *Ctx, child: *std.process.Child) void {
    std.posix.kill(child.id.?, .INT) catch {};
    _ = child.wait(c.io) catch {};
}

const prompts = [_][]const u8{ "What is 2+2?", "Caf\u{e9} r\u{e9}sum\u{e9}: say \u{20ac}5", "A long one" };
const lengths = [_]usize{ 5, 12, 7 };

fn moeRefs(c: *Ctx) ![]u8 {
    const a = c.a;
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.writeByte('[');
    for (prompts, lengths, 0..) |q, n, i| {
        const ids = try fake.tokens(a, false, n, 0);
        if (i > 0) try w.writer.writeAll(", ");
        try w.writer.writeAll("{\"question\": ");
        try std.json.Stringify.value(q, .{}, &w.writer);
        try w.writer.writeAll(", \"tokens\": [");
        for (ids, 0..) |t, k| try w.writer.print("{s}{d}", .{ if (k > 0) ", " else "", t });
        try w.writer.print("], \"sha\": \"{s}\", \"text\": ", .{try fake.shaOf(a, ids)});
        try std.json.Stringify.value(try fake.textOf(a, q, 0), .{}, &w.writer);
        try w.writer.writeByte('}');
    }
    try w.writer.writeByte(']');
    const p = try c.path("ref-moe.json", .{});
    try c.write(p, w.written());
    return p;
}

fn moeScenarios(c: *Ctx, gate: []const u8, fake_exe: []const u8) !void {
    const refs = try moeRefs(c);
    const out = try c.path("moe-out", .{});
    // a server that generates the references' tokens
    {
        const port = c.port();
        const log = try c.path("moe-ok.log", .{});
        var srv = try startFake(c, fake_exe, port, log, "0");
        defer stopFake(c, &srv);
        const ps = try std.fmt.allocPrint(c.a, "{d}", .{port});
        const r = try c.run(&.{ gate, "--log", log, "--port", ps, "--ref", refs, "--tag", "a", "--out", out }, &.{});
        check(r.code == 0, "moe pass: exit {d}\n{s}\n{s}", .{ r.code, r.stdout, r.stderr });
        const sha0 = try fake.shaOf(c.a, try fake.tokens(c.a, false, 5, 0));
        contains("moe pass", r.stdout, try std.fmt.allocPrint(c.a, "round 0 prompt 0: 5 tokens ", .{}));
        contains("moe pass", r.stdout, try std.fmt.allocPrint(c.a, " s sha {s} want {s} ok\n", .{ sha0, sha0 }));
        contains("moe pass", r.stdout, "round 0 prompt 2: 7 tokens ");
        check(std.mem.endsWith(u8, r.stdout, "PASS\n"), "moe pass: ends with PASS:\n{s}", .{r.stdout});
        // DIR/TAG.json: the texts, written like json.dumps
        const a_json = try std.Io.Dir.cwd().readFileAlloc(c.io, try c.path("moe-out/a.json", .{}), c.a, .limited(1 << 20));
        contains("moe a.json", a_json, "[{\"question\": \"What is 2+2?\", \"sha\": \"");
        contains("moe a.json", a_json, "\"question\": \"Caf\\u00e9 r\\u00e9sum\\u00e9: say \\u20ac5\"");
        contains("moe a.json", a_json, " v0 \\u00e9\\u20ac\"}");
        // --against the run just made: identical, with two rounds the third run has no counterpart
        const same = try c.run(&.{ gate, "--log", log, "--port", ps, "--ref", refs, "--tag", "b", "--out", out, "--against", try c.path("moe-out/a.json", .{}) }, &.{});
        check(same.code == 0, "moe against: exit {d}\n{s}", .{ same.code, same.stdout });
        const short = try c.run(&.{ gate, "--log", log, "--port", ps, "--ref", refs, "--tag", "c", "--out", out, "--against", try c.path("moe-out/a.json", .{}), "--rounds", "2" }, &.{});
        check(short.code == 1, "moe against short: exit {d}", .{short.code});
        contains("moe against short", short.stderr, "AgainstRunTooShort");
    }
    // a server that generates other tokens: FAIL, and the first diverging character is counted in characters, not bytes
    {
        const port = c.port();
        const log = try c.path("moe-bad.log", .{});
        var srv = try startFake(c, fake_exe, port, log, "1");
        defer stopFake(c, &srv);
        const ps = try std.fmt.allocPrint(c.a, "{d}", .{port});
        const r = try c.run(&.{ gate, "--log", log, "--port", ps, "--ref", refs, "--tag", "bad", "--out", out }, &.{});
        check(r.code == 1, "moe variant: exit {d}", .{r.code});
        // the text is "answer to " + the prompt + " v" + the variant digit: the variants differ at that digit. The second prompt is 19 characters in 24 bytes
        // (two accents and a euro sign), so counting bytes would give another number.
        const at = [_]usize{ 10 + 12 + 2, 10 + 19 + 2, 10 + 10 + 2 };
        for (at, 0..) |want, i| {
            contains("moe variant", r.stdout, try std.fmt.allocPrint(c.a, "round 0 prompt {d}: {d} tokens ", .{ i, lengths[i] }));
            contains("moe variant", r.stdout, try std.fmt.allocPrint(c.a, "FAIL  DIVERGES at char {d}\n", .{want}));
        }
        check(std.mem.endsWith(u8, r.stdout, "FAIL\n"), "moe variant: ends with FAIL", .{});
        // against the variant-0 run it is not identical either
        const ag = try c.run(&.{ gate, "--log", log, "--port", ps, "--ref", refs, "--tag", "bad2", "--out", out, "--against", try c.path("moe-out/a.json", .{}) }, &.{});
        check(ag.code == 1, "moe variant against: exit {d}", .{ag.code});
    }
    // a usage error
    {
        const r = try c.run(&.{ gate, "--port", "1" }, &.{});
        check(r.code == 2, "moe missing --log: exit {d}", .{r.code});
        contains("moe missing --log", r.stderr, "required: --log");
    }
}

pub fn main(init: std.process.Init) !u8 {
    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) {
        std.debug.print("usage: gate_check VEGA_GATE MOE_GATE FAKE DIR\n", .{});
        return 2;
    }
    var c: Ctx = .{ .a = arena.allocator(), .io = init.io, .base_env = init.environ_map, .dir = args[4], .next_port = 20000 + @as(u16, @intCast(@mod(std.c.getpid(), 20000))) };
    try std.Io.Dir.cwd().createDirPath(c.io, c.dir);
    try vegaScenarios(&c, args[1], args[3]);
    if (!std.mem.eql(u8, args[2], "-")) try moeScenarios(&c, args[2], args[3]); // "-": this tree has no tf-moe-gate
    if (failures != 0) {
        std.debug.print("gate_check: {d} checks failed\n", .{failures});
        return 1;
    }
    try std.Io.File.stdout().writeStreamingAll(init.io, "gate_check: all checks passed\n");
    return 0;
}
