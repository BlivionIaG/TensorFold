//! tf-vega-gate: the gfx900 gate and decode benchmark: serve Qwen3.5-4B MLX 4-bit through a HIP runtime and compare it with mlx_lm (the port of vega_gate.py).
//!
//!     tf-vega-gate [--lib hipmac/build/libamdhip64.dylib] [--tag NAME] [--speed N] [--logits] [--prof each|events|graph]
//!
//! One server (--parallel 1, greedy, no drafts) answers
//!   nothink   "What is the capital of France? Answer in one short sentence."  8 tokens, token sha must be the mlx_lm one
//!   think96   the same with thinking on, 96 tokens, sha must be the mlx_lm one
//!   speed     N runs of a 128-token story: TensorFold's own decode tokens/s (and prefill) per run
//! then, with --logits, `tf-qwen35-test logits --decode` over 87 ids (the decode path row by row) against mlx_lm's float32 logits: cosine per row, argmax
//! equality (computed here, the same figures as tools/vega-macos/cosine.py prints). With --prof the server runs behind tools/vega-macos/hipprof.zig
//! (`zig build tf-hipprof`): `each` (graphs off, every kernel launched and synced, timed; the table of the decode run is written to the output directory),
//! `events` (graphs off, GPU timestamps) or `graph` (graphs on, every graph replay timed); `tf-prof-report` reads the table.
//! The mlx_lm references come from `uvx --from mlx-lm python tools/vega-macos/mlx_ref.py` (cached in ~/.cache/tensorfold-vega/ref).
//! Env TF_POLICY / --policy and --env K=V reach the engine, which is how a kernel variant is switched on (see docs/recipes/vega-macos.md).
//! Exit status 0 when every check passed. Results are also appended as JSON lines to OUT/results.jsonl.
const std = @import("std");
const cli = @import("cli.zig");
const lib = @import("gate_lib.zig");
const npy = @import("npy.zig");
const pf = @import("pyfmt.zig");
const pyjson = @import("pyjson.zig");
const Value = std.json.Value;
const Sha256 = std.crypto.hash.sha2.Sha256;

const question = "What is the capital of France? Answer in one short sentence.";
const story = "Write a long story about a dragon who learns to code.";

pub const usage =
    \\usage: tf-vega-gate [--lib LIB] [--backend BACKEND] [--bin BIN] [--test-bin TEST_BIN] [--model MODEL] [--ref REF] [--out OUT] [--tag TAG]
    \\                    [--port PORT] [--context CONTEXT] [--speed SPEED] [--story-tokens STORY_TOKENS] [--logits] [--prof {each,events,graph}]
    \\                    [--rep REP] [--rep-match REP_MATCH] [--policy POLICY] [--env ENV] [--no-gate]
    \\
    \\The gfx900 gate and decode benchmark: serve Qwen3.5-4B MLX 4-bit through a HIP runtime and compare it with mlx_lm.
    \\
    \\options:
    \\  --lib LIB          libamdhip64.dylib of hipmac (default $TENSORFOLD_HIP, else $UT4G_USBGPU/hipmac/build/libamdhip64.dylib)
    \\  --backend BACKEND  HIPMAC_BACKEND (ut4g card, mock offline; default ut4g)
    \\  --bin BIN          tensorfold-native (default <tree>/zig-out/native-hip/bin/tensorfold-native)
    \\  --test-bin BIN     tf-qwen35-test (default <tree>/zig-out/bin/tf-qwen35-test)
    \\  --model MODEL      the MLX checkpoint (default ~/models/Qwen3.5-4B-MLX-4bit)
    \\  --ref REF          the mlx_lm references (default ~/.cache/tensorfold-vega/ref)
    \\  --out OUT          output directory (default ~/.cache/tensorfold-vega/bench)
    \\  --tag TAG          name of this run in OUT (default run)
    \\  --port PORT        server port (default 18099)
    \\  --context CONTEXT  server context (default 4096)
    \\  --speed SPEED      decode speed runs, 0 skips (default 3)
    \\  --story-tokens N   tokens of a speed run (default 128)
    \\  --logits           also compare decode-path logits with mlx_lm
    \\  --prof MODE        run behind hipprof: each, events or graph
    \\  --rep REP          hipprof: replays of each matching launch (pipelined time; default 16)
    \\  --rep-match STR    hipprof: kernel names to replay ('' all: state is garbage afterwards, the timing is not; default affine_)
    \\  --policy POLICY    TF_POLICY, e.g. stream=reference
    \\  --env K=V          K=V for the engine (repeatable)
    \\  --no-gate          skip the token checks (speed / profile only)
    \\
;

const opts = [_]cli.Opt{
    .{ .name = "--lib" },                     .{ .name = "--backend" },                                          .{ .name = "--bin" },
    .{ .name = "--test-bin" },                .{ .name = "--model" },                                            .{ .name = "--ref" },
    .{ .name = "--out" },                     .{ .name = "--tag" },                                              .{ .name = "--port" },
    .{ .name = "--context" },                 .{ .name = "--speed" },                                            .{ .name = "--story-tokens" },
    .{ .name = "--logits", .value = false },  .{ .name = "--prof", .choices = &.{ "each", "events", "graph" } }, .{ .name = "--rep" },
    .{ .name = "--rep-match" },               .{ .name = "--policy" },                                           .{ .name = "--env", .repeat = true },
    .{ .name = "--no-gate", .value = false },
};

pub const Prof = enum { each, events, graph };

pub const Config = struct {
    lib: []const u8,
    backend: []const u8,
    bin: []const u8,
    test_bin: []const u8,
    model: []const u8,
    ref: []const u8,
    out: []const u8,
    tag: []const u8,
    port: u16,
    context: u32,
    speed: u32,
    story_tokens: u32,
    logits: bool,
    prof: ?Prof,
    rep: u32,
    rep_match: []const u8,
    policy: []const u8,
    env: []const []const u8,
    no_gate: bool,
    /// the TensorFold tree (`zig build tf-hipprof` runs there)
    root: []const u8,
};

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(init.arena.allocator());

    var out_buf: [4096]u8 = undefined;
    var err_buf: [4096]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var stderr = std.Io.File.stderr().writerStreaming(io, &err_buf);
    const out = &stdout.interface;
    const err = &stderr.interface;

    const parsed = try cli.parse(a, argv[1..], "tf-vega-gate", &opts, usage, out, err);
    const args = switch (parsed) {
        .ok => |p| p,
        .exit => |code| return code,
    };
    const cfg = configure(a, io, init.environ_map, args) catch |e| {
        try err.print("tf-vega-gate: error: {s}\n", .{switch (e) {
            error.BadInteger => "an integer option is not an integer (or does not fit)",
            error.BadEnv => "--env wants K=V",
            else => @errorName(e),
        }});
        try err.flush();
        return 2;
    };
    return run(a, io, init.environ_map, cfg, out, err) catch |e| {
        try out.flush();
        try err.print("tf-vega-gate: {s}\n", .{@errorName(e)});
        try err.flush();
        return 1;
    };
}

fn configure(a: std.mem.Allocator, io: std.Io, env: *const std.process.Environ.Map, args: cli.Args) !Config {
    const home = env.get("HOME") orelse "";
    const cache = try std.fmt.allocPrint(a, "{s}/.cache/tensorfold-vega", .{home});
    const hip_env = env.get("TENSORFOLD_HIP");
    const ut4g = env.get("UT4G_USBGPU") orelse try std.fmt.allocPrint(a, "{s}/Projects/Infra/ut4g-usbgpu", .{home});
    const default_lib = if (hip_env != null and hip_env.?.len > 0) hip_env.? else try std.fmt.allocPrint(a, "{s}/hipmac/build/libamdhip64.dylib", .{ut4g});

    // the tree: <tree>/zig-out/bin/tf-vega-gate, else the working directory
    var exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    var root: []const u8 = try std.process.currentPathAlloc(io, a);
    if (std.process.executableDirPath(io, &exe_dir_buf)) |n| {
        const dir = exe_dir_buf[0..n];
        const tail = "/zig-out/bin";
        if (std.mem.endsWith(u8, dir, tail)) root = try a.dupe(u8, dir[0 .. dir.len - tail.len]);
    } else |_| {}

    const prof: ?Prof = if (args.get("--prof")) |p| std.meta.stringToEnum(Prof, p) else null;
    const envs = try args.all(a, "--env");
    for (envs) |kv| if (std.mem.indexOfScalar(u8, kv, '=') == null) return error.BadEnv;
    return .{
        .lib = args.str("--lib", default_lib),
        .backend = args.str("--backend", "ut4g"),
        .bin = args.get("--bin") orelse try std.fmt.allocPrint(a, "{s}/zig-out/native-hip/bin/tensorfold-native", .{root}),
        .test_bin = args.get("--test-bin") orelse try std.fmt.allocPrint(a, "{s}/zig-out/bin/tf-qwen35-test", .{root}),
        .model = args.get("--model") orelse try std.fmt.allocPrint(a, "{s}/models/Qwen3.5-4B-MLX-4bit", .{home}),
        .ref = args.get("--ref") orelse try std.fmt.allocPrint(a, "{s}/ref", .{cache}),
        .out = args.get("--out") orelse try std.fmt.allocPrint(a, "{s}/bench", .{cache}),
        .tag = args.str("--tag", "run"),
        .port = try args.int(u16, "--port", 18099),
        .context = try args.int(u32, "--context", 4096),
        .speed = try args.int(u32, "--speed", 3),
        .story_tokens = try args.int(u32, "--story-tokens", 128),
        .logits = args.flag("--logits"),
        .prof = prof,
        .rep = try args.int(u32, "--rep", 16),
        .rep_match = args.str("--rep-match", "affine_"),
        .policy = args.str("--policy", ""),
        .env = envs,
        .no_gate = args.flag("--no-gate"),
        .root = root,
    };
}

/// The engine's environment: ours, HIPMAC_BACKEND, TENSORFOLD_HIP (the runtime or the shim), TF_POLICY when a policy or a graph-less profile asks, then --env.
fn engineEnv(a: std.mem.Allocator, base: *const std.process.Environ.Map, cfg: Config, shim: ?[]const u8) !std.process.Environ.Map {
    var e = try base.clone(a);
    try e.put("HIPMAC_BACKEND", cfg.backend);
    try e.put("TENSORFOLD_HIP", shim orelse cfg.lib);
    const graphless = cfg.prof != null and cfg.prof.? != .graph;
    if (cfg.policy.len > 0 or graphless) {
        try e.put("TF_POLICY", if (cfg.policy.len > 0 and graphless) try std.fmt.allocPrint(a, "{s},graphs=off", .{cfg.policy}) else if (graphless) "graphs=off" else cfg.policy);
    }
    for (cfg.env) |kv| {
        const eq = std.mem.indexOfScalar(u8, kv, '=').?;
        try e.put(kv[0..eq], kv[eq + 1 ..]);
    }
    return e;
}

fn say(w: *std.Io.Writer, comptime fmt: []const u8, args: anytype) !void {
    try w.print(fmt, args);
    try w.flush();
}

/// Python's round(seconds, digits) as a JSON float.
fn rounded(x: f64, digits: u8) Value {
    return .{ .float = pf.round(x, digits) };
}

fn reprFloat(w: *std.Io.Writer, x: f64) !void {
    var b: [pf.buf_len]u8 = undefined;
    try w.writeAll(pf.repr(&b, x));
}

const Gate = struct {
    a: std.mem.Allocator,
    io: std.Io,
    cfg: Config,
    out: *std.Io.Writer,
    results: std.json.ObjectMap = .empty,

    fn fail(g: *Gate, comptime fmt: []const u8, args: anytype) !void {
        try g.results.put(g.a, "ok", .{ .bool = false });
        try say(g.out, "FAIL " ++ fmt ++ "\n", args);
    }

    fn ok(g: *const Gate) bool {
        return g.results.get("ok").?.bool;
    }
};

fn run(a: std.mem.Allocator, io: std.Io, base_env: *const std.process.Environ.Map, cfg: Config, out: *std.Io.Writer, err: *std.Io.Writer) !u8 {
    try std.Io.Dir.cwd().createDirPath(io, cfg.out);
    const ref_path = try std.fmt.allocPrint(a, "{s}/ref.json", .{cfg.ref});
    const ref = try std.json.parseFromSliceLeaky(Value, a, try lib.readFile(a, io, ref_path), .{});

    var g: Gate = .{ .a = a, .io = io, .cfg = cfg, .out = out };
    var tb: [32]u8 = undefined;
    try g.results.put(a, "tag", .{ .string = cfg.tag });
    try g.results.put(a, "policy", .{ .string = cfg.policy });
    var env_list = std.json.Array.init(a);
    for (cfg.env) |kv| try env_list.append(.{ .string = kv });
    try g.results.put(a, "env", .{ .array = env_list });
    try g.results.put(a, "ok", .{ .bool = true });
    try g.results.put(a, "time", .{ .string = try a.dupe(u8, lib.localTime(&tb)) });

    // ---- the server ----
    var env = try engineEnv(a, base_env, cfg, null);
    if (cfg.prof) |mode| {
        const shim = try buildShim(a, io, base_env, cfg);
        env = try engineEnv(a, base_env, cfg, shim);
        try env.put("HIPPROF_REAL", cfg.lib);
        try env.put("HIPPROF_REP", if (mode == .each) try std.fmt.allocPrint(a, "{d}", .{cfg.rep}) else "0");
        try env.put("HIPPROF_REP_MATCH", cfg.rep_match);
        try env.put("HIPPROF_START", "1");
        try env.put("HIPPROF_LOG", try std.fmt.allocPrint(a, "{s}/{s}.prof.txt", .{ cfg.out, cfg.tag }));
        if (mode == .graph) try env.put("HIPPROF_GRAPH", "1");
        if (mode == .events) try env.put("HIPPROF_EVENTS", "1");
    }
    const log_path = try std.fmt.allocPrint(a, "{s}/{s}.server.log", .{ cfg.out, cfg.tag });
    const port_s = try std.fmt.allocPrint(a, "{d}", .{cfg.port});
    const ctx_s = try std.fmt.allocPrint(a, "{d}", .{cfg.context});
    const cmd = [_][]const u8{ cfg.bin, "serve", cfg.model, "--parallel", "1", "--context", ctx_s, "--port", port_s, "--no-thinking", "--temperature", "0", "--no-drafts" };
    try out.writeAll("$");
    for (cmd) |c| try out.print(" {s}", .{c});
    try out.writeAll("   env: {");
    var first = true;
    for (env.keys(), env.values()) |k, v| {
        if (!(std.mem.startsWith(u8, k, "HIP") or std.mem.startsWith(u8, k, "TF_"))) continue;
        if (!first) try out.writeAll(", ");
        first = false;
        try pyjson.reprString(out, k);
        try out.writeAll(": ");
        try pyjson.reprString(out, v);
    }
    try say(out, "}}\n", .{});

    var srv = try lib.Server.start(io, &cmd, &env, log_path);
    defer srv.stop();
    serverPhase(&g, &srv, ref, log_path) catch |e| switch (e) {
        error.ServerExited, error.ServerDidNotStart => {
            try out.flush();
            try err.print("{s}\n", .{if (e == error.ServerExited) "server exited" else "server did not start"});
            try err.flush();
            return 1;
        },
        else => return e,
    };
    srv.stop();

    if (cfg.logits and cfg.prof == null) try logitsPhase(&g, base_env);

    // OUT/results.jsonl
    var line: std.Io.Writer.Allocating = .init(a);
    try pyjson.dumps(&line.writer, .{ .object = g.results });
    try line.writer.writeByte('\n');
    const results_path = try std.fmt.allocPrint(a, "{s}/results.jsonl", .{cfg.out});
    const old = lib.readFile(a, io, results_path) catch |e| switch (e) {
        error.FileNotFound => try a.dupe(u8, ""),
        else => return e,
    };
    const all = try std.mem.concat(a, u8, &.{ old, line.written() });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = results_path, .data = all });
    try say(out, "GATE {s}\n", .{if (g.ok()) "PASS" else "FAIL"});
    return if (g.ok()) 0 else 1;
}

fn buildShim(a: std.mem.Allocator, io: std.Io, base_env: *const std.process.Environ.Map, cfg: Config) ![]const u8 {
    var argv: std.ArrayList([]const u8) = .empty;
    var it = std.mem.tokenizeAny(u8, base_env.get("TF_ZIG") orelse "zig", " \t\n");
    while (it.next()) |w| try argv.append(a, w);
    try argv.appendSlice(a, &.{ "build", "tf-hipprof" });
    var child = try std.process.spawn(io, .{ .argv = argv.items, .cwd = .{ .path = cfg.root } }); // incremental
    const term = try child.wait(io);
    if (!term.success()) return error.ShimBuildFailed;
    return std.fmt.allocPrint(a, "{s}/zig-out/lib/libhipprof.dylib", .{cfg.root});
}

fn sha12(a: std.mem.Allocator, tokens: []const Value) ![]const u8 {
    var joined: std.Io.Writer.Allocating = .init(a);
    for (tokens, 0..) |t, i| {
        if (i > 0) try joined.writer.writeByte(',');
        switch (t) {
            .integer => |n| try joined.writer.print("{d}", .{n}),
            .number_string => |s| try joined.writer.writeAll(s),
            else => return error.BadReferenceTokens,
        }
    }
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(joined.written(), &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return a.dupe(u8, hex[0..12]);
}

/// The sha the gate wants for `name`: the reference's own when it has one, else the sha256 of its comma-joined token ids.
fn wantSha(a: std.mem.Allocator, ref: Value, name: []const u8) ![]const u8 {
    const entry = (if (ref == .object) ref.object.get(name) else null) orelse return error.ReferenceMissing;
    if (entry != .object) return error.ReferenceMissing;
    if (entry.object.get("sha")) |s| if (s == .string and s.string.len > 0) return s.string;
    const tokens = entry.object.get("tokens") orelse return error.ReferenceWithoutTokens;
    if (tokens != .array) return error.ReferenceWithoutTokens;
    return sha12(a, tokens.array.items);
}

fn serverPhase(g: *Gate, srv: *lib.Server, ref: Value, log_path: []const u8) !void {
    const a = g.a;
    const io = g.io;
    const cfg = g.cfg;
    const out = g.out;

    const t0 = lib.wallSeconds(io);
    while (true) {
        const log = try lib.readFile(a, io, log_path);
        if (std.mem.indexOf(u8, log, "serving ") != null) break;
        if (srv.exited()) {
            try out.print("{s}\n", .{log}); // print(open(log).read())
            try out.flush();
            return error.ServerExited;
        }
        if (lib.wallSeconds(io) - t0 > 300) return error.ServerDidNotStart;
        lib.sleepSeconds(io, 0.5);
    }
    const load = rounded(lib.wallSeconds(io) - t0, 1);
    try g.results.put(a, "load_seconds", load);
    try out.writeAll("server up in ");
    try reprFloat(out, load.float);
    try say(out, " s\n", .{});

    if (!cfg.no_gate) {
        const checks = [_]struct { []const u8, bool, i64 }{ .{ "nothink", false, 8 }, .{ "think96", true, 96 } };
        for (checks) |c| {
            const reply = try lib.chat(a, io, cfg.port, try lib.chatBody(a, question, c[2], c[1]), null);
            lib.sleepSeconds(io, 0.3);
            const log = try lib.readFile(a, io, log_path);
            const sha = lib.shaOfLog(log, reply.id, true);
            const want = try wantSha(a, ref, c[0]);
            var entry: std.json.ObjectMap = .empty;
            try entry.put(a, "sha", if (sha) |s| .{ .string = s } else .null);
            try entry.put(a, "want", .{ .string = want });
            try entry.put(a, "tokens", .{ .integer = reply.completion_tokens });
            try g.results.put(a, c[0], .{ .object = entry });
            if (sha == null or !std.mem.eql(u8, sha.?, want)) {
                try g.fail("{s}: token sha {s} != mlx_lm {s}", .{ c[0], sha orelse "None", want });
            } else try say(out, "PASS {s}: {d} tokens identical to mlx_lm (sha {s})\n", .{ c[0], reply.completion_tokens, sha.? });
        }
    }

    if (cfg.speed > 0) {
        if (cfg.prof != null) {
            lib.sleepSeconds(io, 0.5);
            srv.signal(.USR1); // the table starts here
        }
        var runs = std.json.Array.init(a);
        var tps: std.ArrayList(f64) = .empty;
        for (0..cfg.speed) |i| {
            const t1 = lib.wallSeconds(io);
            const reply = try lib.chat(a, io, cfg.port, try lib.chatBody(a, story, cfg.story_tokens, false), null);
            const wall = lib.wallSeconds(io) - t1;
            const tf: Value = reply.tensorfold orelse .{ .object = .empty };
            var entry: std.json.ObjectMap = .empty;
            try entry.put(a, "completion_tokens", .{ .integer = reply.completion_tokens });
            try entry.put(a, "wall", rounded(wall, 3));
            try entry.put(a, "tensorfold", tf);
            try runs.append(.{ .object = entry });
            var wb: [pf.buf_len]u8 = undefined;
            try out.print("speed run {d}: {d} tokens, wall {s} s, tensorfold ", .{ i, reply.completion_tokens, pf.fixed(&wb, wall, 2, false) });
            try pyjson.repr(out, tf);
            try out.writeByte('\n');
            try out.flush();
            if (tf == .object) if (tf.object.get("tokens_per_second")) |v| {
                const x: f64 = switch (v) {
                    .float => |f| f,
                    .integer => |n| @floatFromInt(n),
                    else => 0,
                };
                if (x != 0) try tps.append(a, x); // Python: truthy
            };
        }
        try g.results.put(a, "speed", .{ .array = runs });
        if (tps.items.len > 0) {
            std.mem.sort(f64, tps.items, {}, std.sort.asc(f64));
            const med = rounded(tps.items[tps.items.len / 2], 2);
            try g.results.put(a, "decode_tok_s", med);
            try out.print("decode tok/s (median of {d}): ", .{tps.items.len});
            try reprFloat(out, med.float);
            try say(out, "\n", .{});
        }
        if (cfg.prof != null) {
            srv.signal(.USR2);
            lib.sleepSeconds(io, 1.0);
        }
    }
}

fn logitsPhase(g: *Gate, base_env: *const std.process.Environ.Map) !void {
    const a = g.a;
    const io = g.io;
    const cfg = g.cfg;
    const ids = try std.fmt.allocPrint(a, "{s}/ids.npy", .{cfg.ref});
    const ours_path = try std.fmt.allocPrint(a, "{s}/{s}.logits.npy", .{ cfg.out, cfg.tag });
    var env = try engineEnv(a, base_env, cfg, null);
    const t2 = lib.wallSeconds(io);
    const p = try std.process.run(a, io, .{ .argv = &.{ cfg.test_bin, "logits", cfg.model, ids, ours_path, "--decode" }, .environ_map = &env });
    try g.results.put(a, "logits_seconds", rounded(lib.wallSeconds(io) - t2, 1));
    if (!p.term.success()) {
        const both = try std.mem.concat(a, u8, &.{ p.stdout, p.stderr });
        try g.fail("logits run failed: {s}", .{both[both.len -| 500..]});
        return;
    }
    var line_buf: [256]u8 = undefined;
    const theirs_path = try std.fmt.allocPrint(a, "{s}/mlx_logits.npy", .{cfg.ref});
    const verdict: ?lib.Cosine = compare(a, io, ours_path, theirs_path) catch |e| blk: {
        try say(g.out, "{s}\n", .{@errorName(e)});
        break :blk null;
    };
    const line: []const u8 = if (verdict) |v| try a.dupe(u8, v.line(&line_buf)) else "";
    try say(g.out, "{s} \n", .{line}); // print(stdout, stderr): the empty stderr leaves a space
    try g.results.put(a, "logits", .{ .string = line });
    if (verdict == null or !verdict.?.passes()) try g.fail("logits cosine/argmax gate", .{});
}

fn compare(a: std.mem.Allocator, io: std.Io, ours_path: []const u8, theirs_path: []const u8) !lib.Cosine {
    const ours = try npy.parse(try lib.readFile(a, io, ours_path));
    const theirs = try npy.parse(try lib.readFile(a, io, theirs_path));
    return lib.cosine(ours, theirs);
}

test "the reference's own sha wins, else the first 12 hex digits of the sha256 of the comma-joined ids" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ref = try std.json.parseFromSliceLeaky(Value, a,
        \\{"nothink": {"tokens": [1, 2, 3]}, "think96": {"tokens": [5], "sha": "abcdefabcdef"}, "empty": {"tokens": [4], "sha": ""}}
    , .{});
    // python3 -c "import hashlib;print(hashlib.sha256(b'1,2,3').hexdigest()[:12])"
    try std.testing.expectEqualStrings("8a6ae1512200", try wantSha(a, ref, "nothink"));
    try std.testing.expectEqualStrings("abcdefabcdef", try wantSha(a, ref, "think96"));
    try std.testing.expectEqualStrings("4b227777d4dd", try wantSha(a, ref, "empty")); // sha256(b'4')
    try std.testing.expectError(error.ReferenceMissing, wantSha(a, ref, "other"));
}

test "TF_POLICY is the policy, graphs=off for a profile without graphs, or both" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var base = std.process.Environ.Map.init(a);
    try base.put("HIPFOO", "1");
    var cfg: Config = .{ .lib = "/l", .backend = "ut4g", .bin = "", .test_bin = "", .model = "", .ref = "", .out = "", .tag = "t", .port = 1, .context = 1, .speed = 0, .story_tokens = 1, .logits = false, .prof = null, .rep = 16, .rep_match = "", .policy = "", .env = &.{"X=a=b"}, .no_gate = false, .root = "" };
    var e = try engineEnv(a, &base, cfg, null);
    try std.testing.expectEqualStrings("ut4g", e.get("HIPMAC_BACKEND").?);
    try std.testing.expectEqualStrings("/l", e.get("TENSORFOLD_HIP").?);
    try std.testing.expectEqualStrings("a=b", e.get("X").?);
    try std.testing.expect(e.get("TF_POLICY") == null);
    cfg.prof = .each;
    e = try engineEnv(a, &base, cfg, "/shim");
    try std.testing.expectEqualStrings("graphs=off", e.get("TF_POLICY").?);
    try std.testing.expectEqualStrings("/shim", e.get("TENSORFOLD_HIP").?);
    cfg.policy = "stream=reference";
    e = try engineEnv(a, &base, cfg, null);
    try std.testing.expectEqualStrings("stream=reference,graphs=off", e.get("TF_POLICY").?);
    cfg.prof = .graph;
    e = try engineEnv(a, &base, cfg, null);
    try std.testing.expectEqualStrings("stream=reference", e.get("TF_POLICY").?);
}
