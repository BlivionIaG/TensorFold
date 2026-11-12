//! What `tf-vega-gate` and `tf-moe-gate` share: one chat request to a local `tensorfold-native serve`, the `done <id> ... sha=` line of its log, the server
//! process (started with its log, signalled, stopped), the comparison of two logits files and the Python-style clock and text helpers.
const std = @import("std");
const npy = @import("npy.zig");
const pf = @import("pyfmt.zig");
const Value = std.json.Value;

pub const Io = std.Io;

// ---------------------------------------------------------------------------------------------------- clock, files

/// time.time(): seconds since the epoch.
pub fn wallSeconds(io: Io) f64 {
    return @as(f64, @floatFromInt(Io.Clock.real.now(io).toNanoseconds())) / 1e9;
}

pub fn sleepSeconds(io: Io, s: f64) void {
    io.sleep(.fromNanoseconds(@intFromFloat(s * 1e9)), .awake) catch {};
}

extern "c" fn time(t: ?*i64) i64;
extern "c" fn localtime_r(t: *const i64, out: *Tm) ?*Tm;
extern "c" fn strftime(buf: [*]u8, max: usize, fmt: [*:0]const u8, tm: *const Tm) usize;
const Tm = extern struct { sec: c_int, min: c_int, hour: c_int, mday: c_int, mon: c_int, year: c_int, wday: c_int, yday: c_int, isdst: c_int, gmtoff: c_long, zone: ?[*:0]const u8 };

/// time.strftime("%Y-%m-%d %H:%M:%S"): the local time.
pub fn localTime(buf: *[32]u8) []const u8 {
    const now = time(null);
    var tm: Tm = undefined;
    if (localtime_r(&now, &tm) == null) return "";
    return buf[0..strftime(buf, buf.len, "%Y-%m-%d %H:%M:%S", &tm)];
}

pub fn readFile(a: std.mem.Allocator, io: Io, path: []const u8) ![]u8 {
    return std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 30));
}

/// `~/rest` with the home directory expanded (os.path.expanduser), anything else unchanged.
pub fn expandUser(a: std.mem.Allocator, env: *const std.process.Environ.Map, path: []const u8) ![]u8 {
    if (std.mem.eql(u8, path, "~") or std.mem.startsWith(u8, path, "~/")) {
        if (env.get("HOME")) |home| return std.fmt.allocPrint(a, "{s}{s}", .{ home, path[1..] });
    }
    return a.dupe(u8, path);
}

// ---------------------------------------------------------------------------------------------------- the chat request

/// The request body of every gate: one user message, greedy, thinking switched as asked, written like json.dumps.
pub fn chatBody(a: std.mem.Allocator, prompt: []const u8, max_tokens: i64, thinking: bool) ![]u8 {
    var w: Io.Writer.Allocating = .init(a);
    errdefer w.deinit();
    try w.writer.writeAll("{\"messages\": [{\"role\": \"user\", \"content\": ");
    try @import("pyjson.zig").dumpsString(&w.writer, prompt);
    try w.writer.print("}}], \"max_tokens\": {d}, \"temperature\": 0, \"chat_template_kwargs\": {{\"enable_thinking\": {s}}}}}", .{ max_tokens, if (thinking) "true" else "false" });
    return w.toOwnedSlice();
}

pub const Reply = struct {
    id: []const u8,
    completion_tokens: i64,
    /// choices[0].message.content, "" when it is null or missing
    content: []const u8,
    /// the server's own `tensorfold` object, null when it sent none
    tensorfold: ?Value,
};

pub const ChatError = error{ HttpStatus, BadReply } || std.http.Client.FetchError || std.mem.Allocator.Error || std.json.ParseError(std.json.Scanner);

/// POST `body` to `http://127.0.0.1:PORT/v1/chat/completions`. The strings of the reply belong to `a` (an arena).
pub fn chat(a: std.mem.Allocator, io: Io, port: u16, body: []const u8, status_out: ?*u10) ChatError!Reply {
    var client: std.http.Client = .{ .allocator = a, .io = io };
    defer client.deinit();
    const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
    var out: Io.Writer.Allocating = .init(a);
    const res = try client.fetch(.{
        .location = .{ .url = url },
        .method = .POST,
        .payload = body,
        .headers = .{ .content_type = .{ .override = "application/json" } },
        .keep_alive = false,
        .response_writer = &out.writer,
    });
    if (status_out) |s| s.* = @backingInt(res.status);
    if (res.status != .ok) return error.HttpStatus;
    return parseReply(a, out.written());
}

pub fn parseReply(a: std.mem.Allocator, text: []const u8) ChatError!Reply {
    const root = try std.json.parseFromSliceLeaky(Value, a, text, .{});
    if (root != .object) return error.BadReply;
    const id = root.object.get("id") orelse return error.BadReply;
    const usage = root.object.get("usage") orelse return error.BadReply;
    if (id != .string or usage != .object) return error.BadReply;
    const n = usage.object.get("completion_tokens") orelse return error.BadReply;
    if (n != .integer) return error.BadReply;
    var content: []const u8 = "";
    if (root.object.get("choices")) |choices| if (choices == .array and choices.array.items.len > 0) {
        const c0 = choices.array.items[0];
        if (c0 == .object) if (c0.object.get("message")) |msg| if (msg == .object) if (msg.object.get("content")) |c| if (c == .string) {
            content = c.string;
        };
    };
    return .{ .id = id.string, .completion_tokens = n.integer, .content = content, .tensorfold = root.object.get("tensorfold") };
}

// ---------------------------------------------------------------------------------------------------- the server log

/// The 12 hex digits after the first `sha=` of `line` that has 12 of them (re.search(r"sha=([0-9a-f]{12})")).
pub fn shaIn(line: []const u8) ?[]const u8 {
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, line, from, "sha=")) |at| {
        const s = at + 4;
        if (s + 12 <= line.len) {
            const hex = line[s .. s + 12];
            const all_hex = for (hex) |c| {
                if (!std.ascii.isDigit(c) and !(c >= 'a' and c <= 'f')) break false;
            } else true;
            if (all_hex) return hex;
        }
        from = at + 1;
    }
    return null;
}

/// How `sha_of` of the two Python gates reads a log: the first line that has `done <id>`; `stop_at_first` (vega_gate): its sha or null even when that line has
/// none; otherwise (moe_gate) a line without a sha is skipped and the scan goes on.
pub fn shaOfLog(log: []const u8, id: []const u8, stop_at_first: bool) ?[]const u8 {
    var lines = std.mem.splitScalar(u8, log, '\n');
    while (lines.next()) |line| {
        // "done " + id in line: the id follows a `done ` somewhere on the line
        var from: usize = 0;
        const hit = while (std.mem.indexOfPos(u8, line, from, "done ")) |p| {
            if (std.mem.startsWith(u8, line[p + 5 ..], id)) break true;
            from = p + 1;
        } else false;
        if (!hit) continue;
        if (shaIn(line)) |s| return s;
        if (stop_at_first) return null;
    }
    return null;
}

// ---------------------------------------------------------------------------------------------------- the server process

pub const Server = struct {
    io: Io,
    pid: std.posix.pid_t,
    log_path: []const u8,
    reaped: bool = false,

    /// Starts `argv` with `env` and stdout + stderr into the log file (created / truncated), stdin closed.
    pub fn start(io: Io, argv: []const []const u8, env: *const std.process.Environ.Map, log_path: []const u8) !Server {
        const log = try std.Io.Dir.cwd().createFile(io, log_path, .{ .truncate = true });
        defer log.close(io);
        const child = try std.process.spawn(io, .{ .argv = argv, .environ_map = env, .stdin = .ignore, .stdout = .{ .file = log }, .stderr = .{ .file = log } });
        return .{ .io = io, .pid = child.id.?, .log_path = log_path };
    }

    /// Popen.poll() is not None: the child has exited (and is reaped).
    pub fn exited(self: *Server) bool {
        if (self.reaped) return true;
        var status: c_int = 0;
        const rc = std.c.waitpid(self.pid, &status, std.c.W.NOHANG);
        if (rc == self.pid or (rc < 0 and std.posix.errno(rc) == .CHILD)) {
            self.reaped = true;
            return true;
        }
        return false;
    }

    pub fn signal(self: *Server, sig: std.posix.SIG) void {
        std.posix.kill(self.pid, sig) catch {};
    }

    /// SIGINT, up to 10 s for it to exit, else SIGKILL and wait.
    pub fn stop(self: *Server) void {
        if (self.reaped) return;
        self.signal(.INT);
        var waited: u32 = 0;
        while (waited < 10_000) : (waited += 50) {
            if (self.exited()) return;
            sleepSeconds(self.io, 0.05);
        }
        self.signal(.KILL);
        var status: c_int = 0;
        while (true) {
            const rc = std.c.waitpid(self.pid, &status, 0);
            if (rc >= 0 or std.posix.errno(rc) != .INTR) break;
        }
        self.reaped = true;
    }
};

// ---------------------------------------------------------------------------------------------------- logits comparison

pub const Cosine = struct {
    rows: usize,
    min: f64,
    mean: f64,
    argmax_equal: usize,
    mean_abs_diff: f64,

    /// cosine.py's exit status: every row's cosine above 0.999 and the argmax of every row equal.
    pub fn passes(self: Cosine) bool {
        return self.min > 0.999 and self.argmax_equal == self.rows;
    }

    /// cosine.py's one line.
    pub fn line(self: Cosine, buf: []u8) []const u8 {
        var w: Io.Writer = .fixed(buf);
        var b: [4][pf.buf_len]u8 = undefined;
        w.print("rows {d}  cosine min {s} mean {s}  argmax equal {d}/{d}  mean|diff| {s}", .{ self.rows, pf.fixed(&b[0], self.min, 6, false), pf.fixed(&b[1], self.mean, 6, false), self.argmax_equal, self.rows, pf.fixed(&b[2], self.mean_abs_diff, 4, false) }) catch unreachable;
        return w.buffered();
    }
};

pub const CosineError = error{ NotMatrix, Empty };

fn argmax(row: usize, arr: npy.Array, cols: usize, v: usize) usize {
    var best: usize = 0;
    var best_val = arr.at(row * cols);
    for (1..v) |j| {
        const x = arr.at(row * cols + j);
        if (x > best_val) { // numpy's argmax: the first maximum
            best = j;
            best_val = x;
        }
    }
    return best;
}

/// cosine.py over two [rows, vocab] float arrays: the rows and columns both have in common, per-row cosine in float64.
pub fn cosine(ours: npy.Array, theirs: npy.Array) CosineError!Cosine {
    if (ours.ndim != 2 or theirs.ndim != 2) return error.NotMatrix;
    const n = @min(ours.shape[0], theirs.shape[0]);
    const v = @min(ours.shape[1], theirs.shape[1]);
    if (n == 0 or v == 0) return error.Empty;
    var min: f64 = std.math.inf(f64);
    var sum_cos: f64 = 0;
    var equal: usize = 0;
    var sum_abs: f64 = 0;
    for (0..n) |i| {
        var dot: f64 = 0;
        var na: f64 = 0;
        var nb: f64 = 0;
        for (0..v) |j| {
            const x = ours.at(i * ours.shape[1] + j);
            const y = theirs.at(i * theirs.shape[1] + j);
            dot += x * y;
            na += x * x;
            nb += y * y;
            sum_abs += @abs(x - y);
        }
        const c = dot / (@sqrt(na) * @sqrt(nb));
        if (c < min or std.math.isNan(c)) min = c;
        sum_cos += c;
        if (argmax(i, ours, ours.shape[1], v) == argmax(i, theirs, theirs.shape[1], v)) equal += 1;
    }
    return .{ .rows = n, .min = min, .mean = sum_cos / @as(f64, @floatFromInt(n)), .argmax_equal = equal, .mean_abs_diff = sum_abs / @as(f64, @floatFromInt(n * v)) };
}

// ---------------------------------------------------------------------------------------------------- text

/// The index, in code points, of the first difference of `x` and `y` (the shorter one's length when one is a prefix of the other); the Python gates count characters.
pub fn diverge(x: []const u8, y: []const u8) usize {
    var a = std.unicode.Utf8View.initUnchecked(x).iterator();
    var b = std.unicode.Utf8View.initUnchecked(y).iterator();
    var i: usize = 0;
    while (true) : (i += 1) {
        const p = a.nextCodepointSlice() orelse return i;
        const q = b.nextCodepointSlice() orelse return i;
        if (!std.mem.eql(u8, p, q)) return i;
    }
}

test "the sha of a done line" {
    try std.testing.expectEqualStrings("eb415dcbaf6e", shaIn("[tensorfold] done chatcmpl-5d prompt=25 tokens=8 sha=eb415dcbaf6e finish=stop").?);
    try std.testing.expect(shaIn("done x sha=XYZ") == null);
    try std.testing.expect(shaIn("sha=abc") == null);
    try std.testing.expectEqualStrings("0123456789ab", shaIn("sha=short sha=0123456789abcdef").?);
}

test "a log is scanned like the Python gates scan it" {
    const log =
        \\[tensorfold] serving m at http://127.0.0.1:18099/v1
        \\[tensorfold] done chatcmpl-aa prompt=25 tokens=8 sha=111111111111 finish=stop
        \\[tensorfold] done chatcmpl-bb prompt=25 tokens=8 finish=stop
        \\[tensorfold] done chatcmpl-bb prompt=25 tokens=8 sha=222222222222 finish=stop
        \\
    ;
    try std.testing.expectEqualStrings("111111111111", shaOfLog(log, "chatcmpl-aa", true).?);
    try std.testing.expect(shaOfLog(log, "chatcmpl-bb", true) == null); // vega_gate: the first line decides
    try std.testing.expectEqualStrings("222222222222", shaOfLog(log, "chatcmpl-bb", false).?); // moe_gate: skips the line without one
    try std.testing.expect(shaOfLog(log, "chatcmpl-cc", false) == null);
}

test "divergence counts characters" {
    try std.testing.expectEqual(@as(usize, 2), diverge("abc", "abd"));
    try std.testing.expectEqual(@as(usize, 2), diverge("ab", "abc"));
    try std.testing.expectEqual(@as(usize, 1), diverge("é€x", "éZx")); // 1, not the byte offset 2
    try std.testing.expectEqual(@as(usize, 3), diverge("é€😀", "é€😀"));
}

test "the request body is json.dumps of the Python gates' dict" {
    const a = std.testing.allocator;
    const b = try chatBody(a, "What is \"it\"? é", 8, false);
    defer a.free(b);
    try std.testing.expectEqualStrings(
        \\{"messages": [{"role": "user", "content": "What is \"it\"? \u00e9"}], "max_tokens": 8, "temperature": 0, "chat_template_kwargs": {"enable_thinking": false}}
    , b);
}

test "reply parsing" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try parseReply(a, "{\"id\": \"chatcmpl-1\", \"choices\": [{\"message\": {\"content\": null}}], \"usage\": {\"completion_tokens\": 8}, \"tensorfold\": {\"tokens_per_second\": 81.5}}");
    try std.testing.expectEqualStrings("chatcmpl-1", r.id);
    try std.testing.expectEqual(@as(i64, 8), r.completion_tokens);
    try std.testing.expectEqualStrings("", r.content);
    try std.testing.expect(r.tensorfold != null);
    try std.testing.expectError(error.BadReply, parseReply(a, "{\"id\": \"x\"}"));
    try std.testing.expectError(error.BadReply, parseReply(a, "[1]"));
}

test "cosine of two logits files" {
    const a = std.testing.allocator;
    const x = [_]f32{ 1, 2, 3, 0, 4, 4, 1, 0 };
    const y = [_]f32{ 1, 2, 3, 9, 4.5, 4, 1, 0 };
    const bx = try npy.encode(a, "<f4", &.{ 2, 4 }, std.mem.sliceAsBytes(&x));
    defer a.free(bx);
    const by = try npy.encode(a, "<f4", &.{ 2, 4 }, std.mem.sliceAsBytes(&y));
    defer a.free(by);
    const c = try cosine(try npy.parse(bx), try npy.parse(by));
    try std.testing.expectEqual(@as(usize, 2), c.rows);
    // row 0: dot 14, |x| sqrt 14, |y| sqrt 95; row 1: dot 4*4.5 + 4*4 + 1 = 35, |x| sqrt 33, |y| sqrt 37.25
    const c0 = 14.0 / (@sqrt(14.0) * @sqrt(95.0));
    const c1 = 35.0 / (@sqrt(33.0) * @sqrt(37.25));
    try std.testing.expectApproxEqAbs(@min(c0, c1), c.min, 1e-12);
    try std.testing.expectApproxEqAbs((c0 + c1) / 2.0, c.mean, 1e-12);
    try std.testing.expectEqual(@as(usize, 1), c.argmax_equal); // row 0 argmax 2 vs 3
    try std.testing.expectApproxEqAbs(@as(f64, (9.0 + 0.5) / 8.0), c.mean_abs_diff, 1e-12);
    try std.testing.expect(!c.passes());
    var buf: [200]u8 = undefined;
    try std.testing.expect(std.mem.startsWith(u8, c.line(&buf), "rows 2  cosine min "));
}

test "cosine uses the rows and columns both files have" {
    const a = std.testing.allocator;
    const x = [_]f32{ 1, 2, 3, 9, 1, 2, 3, 9, 7, 7, 7, 7 }; // 3 rows x 4
    const y = [_]f32{ 1, 2, 3, 1, 2, 3 }; // 2 rows x 3
    const bx = try npy.encode(a, "<f4", &.{ 3, 4 }, std.mem.sliceAsBytes(&x));
    defer a.free(bx);
    const by = try npy.encode(a, "<f4", &.{ 2, 3 }, std.mem.sliceAsBytes(&y));
    defer a.free(by);
    const c = try cosine(try npy.parse(bx), try npy.parse(by));
    try std.testing.expectEqual(@as(usize, 2), c.rows);
    try std.testing.expectApproxEqAbs(@as(f64, 1), c.min, 1e-12);
    try std.testing.expect(c.passes());
    try std.testing.expectEqual(@as(usize, 2), c.argmax_equal);
}
