//! hipprof_check MODE SHIM MOCK DIR: drives a hipprof shim over a mock runtime and checks the table it writes.
//! Modes: each, events, graph, start, exit, after-exit. Files land in DIR; exit code 0 when every check holds.
const std = @import("std");

const U = usize;
const Fn = *const fn (U, U, U, U, U, U, U, U, U, U, U) callconv(.c) U;
const Counter = *const fn () callconv(.c) U;

extern "c" fn dlopen(path: [*:0]const u8, mode: c_int) ?*anyopaque;
extern "c" fn dlsym(handle: ?*anyopaque, symbol: [*:0]const u8) ?*anyopaque;
extern "c" fn dlerror() ?[*:0]const u8;
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern "c" fn raise(sig: c_int) c_int;
extern "c" fn usleep(us: c_uint) c_int;
extern "c" fn unlink(path: [*:0]const u8) c_int;
extern "c" fn fopen(path: [*:0]const u8, mode: [*:0]const u8) ?*std.c.FILE;
extern "c" fn fread(buf: [*]u8, size: usize, count: usize, f: *std.c.FILE) usize;

const usr1 = @backingInt(std.c.SIG.USR1);
const usr2 = @backingInt(std.c.SIG.USR2);
const fields = 15;
const max_rows = 32;

var shim: ?*anyopaque = null;
var mock: ?*anyopaque = null;
var file_buf: [1 << 16]u8 = undefined;
var log_path: [:0]const u8 = "";

fn fail(comptime fmt: []const u8, args: anytype) error{Check} {
    std.debug.print("hipprof_check: " ++ fmt ++ "\n", args);
    return error.Check;
}

fn open(path: [*:0]const u8, flags: c_int) !?*anyopaque {
    return dlopen(path, flags) orelse return fail("cannot open {s}: {s}", .{ path, dlerror() orelse "?" });
}

fn shimFn(name: [*:0]const u8) !Fn {
    return @ptrCast(@alignCast(dlsym(shim, name) orelse return fail("the shim does not export {s}", .{name})));
}

fn mockCount(name: [*:0]const u8) !U {
    const f: Counter = @ptrCast(@alignCast(dlsym(mock, name) orelse return fail("the mock has no {s}", .{name})));
    return f();
}

fn setEnv(name: [*:0]const u8, value: [*:0]const u8) void {
    _ = setenv(name, value, 1);
}

/// The device side of the fake engine: function handles by name and parameter blocks the shim decodes.
const Engine = struct {
    row: U = 0,
    stream_fn: U = 0,
    norm: U = 0,
    affine: [80]u8 = @splat(0),
    sides: [192]u8 = @splat(0),
    x: [2560]u16 = @splat(0x3c00),
    plain: [2]?*anyopaque = undefined,
    side: [2]?*anyopaque = undefined,
    get_function: Fn = undefined,
    launch_fn: Fn = undefined,

    fn putI32(buf: []u8, off: usize, v: i32) void {
        std.mem.writeInt(i32, buf[off..][0..4], v, .little);
    }

    fn setup(self: *Engine) !void {
        self.get_function = try shimFn("hipModuleGetFunction");
        self.launch_fn = try shimFn("hipModuleLaunchKernel");
        self.x[7] = 0x7c00;
        std.mem.writeInt(u64, self.affine[0..8], @intFromPtr(&self.x), .little);
        putI32(&self.affine, 56, 1);
        putI32(&self.affine, 60, 4096);
        putI32(&self.affine, 64, 2560);
        putI32(&self.sides, 164, 2);
        putI32(&self.sides, 128, 1024);
        putI32(&self.sides, 132, 512);
        self.plain = .{ &self.affine, null };
        self.side = .{ &self.affine, &self.sides };
        self.row = try self.function("affine_dot2_row");
        self.stream_fn = try self.function("affine_dot2_stream");
        self.norm = try self.function("rmsnorm");
    }

    fn function(self: *Engine, name: [*:0]const u8) !U {
        var handle: U = 0;
        const r = self.get_function(@intFromPtr(&handle), 0, @intFromPtr(name), 0, 0, 0, 0, 0, 0, 0, 0);
        if (@as(u32, @truncate(r)) != 0 or handle == 0) return fail("hipModuleGetFunction({s}) failed", .{name});
        return handle;
    }

    fn launch(self: *Engine, f: U, grid: U, params: ?*anyopaque) void {
        _ = self.launch_fn(f, grid, 1, 1, 64, 1, 1, 0, 7, @intFromPtr(params), 0);
    }

    fn launchRow(self: *Engine) void {
        self.launch(self.row, 32, &self.plain);
    }

    fn launchStream(self: *Engine) void {
        self.launch(self.stream_fn, 8, &self.side);
    }

    fn launchNorm(self: *Engine) void {
        self.launch(self.norm, 1, null);
    }
};

const Parsed = struct { f: [fields][]const u8 };
var rows: [max_rows]Parsed = undefined;

/// The table file as rows; null while it is missing, cut short or still being written.
fn readTable() ?usize {
    const f = fopen(log_path, "r") orelse return null;
    defer _ = std.c.fclose(f);
    const len = fread(&file_buf, 1, file_buf.len, f);
    const text = file_buf[0..len];
    if (!std.mem.startsWith(u8, text, "# name\tcount\t") or text.len == 0 or text[text.len - 1] != '\n') return null;
    var n: usize = 0;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#') continue;
        if (n == max_rows) return null;
        var cols = std.mem.splitScalar(u8, line, '\t');
        var i: usize = 0;
        while (cols.next()) |col| : (i += 1) {
            if (i == fields) return null;
            rows[n].f[i] = col;
        }
        if (i != fields) return null;
        n += 1;
    }
    return n;
}

fn find(n: usize, name: []const u8) ?*const Parsed {
    for (rows[0..n]) |*r| if (std.mem.eql(u8, r.f[0], name)) return r;
    return null;
}

/// Columns 1..14 of a row against `want`: exact text, or "*" for any (times of a real clock are not checked).
fn expectRow(n: usize, name: []const u8, want: [fields - 1][]const u8) !void {
    const r = find(n, name) orelse return fail("no row {s}", .{name});
    for (want, 1..) |w, i| {
        if (std.mem.eql(u8, w, "*")) continue;
        if (!std.mem.eql(u8, w, r.f[i])) return fail("{s} column {d}: {s}, want {s}", .{ name, i, r.f[i], w });
    }
}

fn expectCount(what: []const u8, got: U, want: U) !void {
    if (got != want) return fail("{s}: {d}, want {d}", .{ what, got, want });
}

/// SIGUSR2 and then the table, until `ready` holds for it (the shim's watcher thread writes it).
fn dumpUntil(comptime ready: fn (usize) bool) !usize {
    _ = unlink(log_path);
    _ = raise(usr2);
    for (0..500) |_| {
        if (readTable()) |n| if (ready(n)) return n;
        _ = usleep(10_000);
    }
    return fail("the table never became ready", .{});
}

fn any(_: usize) bool {
    return true;
}

fn nonEmpty(n: usize) bool {
    return n > 0;
}

fn modeEach(e: *Engine) !void {
    const malloc: Fn = try shimFn("hipMalloc");
    var ptr: U = 0;
    _ = malloc(@intFromPtr(&ptr), 4096, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    try expectCount("hipMalloc forwarded", ptr, 0x4000_0000 + 4096);
    const attr: Fn = try shimFn("hipDeviceGetAttribute");
    var value: c_int = 0;
    _ = attr(@intFromPtr(&value), 63, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    try expectCount("hipDeviceGetAttribute forwarded", @intCast(value), 127);
    e.launchRow();
    e.launchStream();
    e.launchRow();
    e.launchStream();
    e.launchRow();
    for (0..4) |_| e.launchNorm();
    const begin = try shimFn("hipStreamBeginCapture");
    const end = try shimFn("hipStreamEndCapture");
    _ = begin(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    for (0..5) |_| e.launchRow();
    _ = end(7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    e.launchNorm();
    _ = (try shimFn("hipGraphLaunch"))(1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    const n = try dumpUntil(nonEmpty);
    try expectCount("rows", n, 3);
    try expectRow(n, "affine_dot2_row", .{ "3", "32,1,1", "64,1,1", "0", "1", "4096", "2560", "0", "*", "*", "*", "*", "5898240", "0.00" });
    try expectRow(n, "affine_dot2_stream", .{ "2", "8,1,1", "64,1,1", "0", "1", "1536", "2560", "2", "*", "*", "*", "*", "2211840", "0.00" });
    try expectRow(n, "rmsnorm", .{ "5", "1,1,1", "64,1,1", "0", "0", "0", "0", "0", "*", "*", "*", "0.00", "0", "0.00" });
    try expectCount("mock launches (timed, replayed twice, captured)", try mockCount("hipmock_launches"), 9 + 6 + 5 + 5);
    try expectCount("graph launches passed through", try mockCount("hipmock_graphs"), 1);
}

fn modeEvents(e: *Engine) !void {
    e.launchRow();
    e.launchStream();
    e.launchRow();
    e.launchStream();
    e.launchRow();
    const n = try dumpUntil(nonEmpty);
    try expectCount("rows", n, 2);
    try expectRow(n, "affine_dot2_row", .{ "3", "32,1,1", "64,1,1", "0", "1", "4096", "2560", "0", "20.00", "20.00", "0.060", "0.00", "5898240", "20.00" });
    try expectRow(n, "affine_dot2_stream", .{ "2", "8,1,1", "64,1,1", "0", "1", "1536", "2560", "2", "20.00", "20.00", "0.040", "0.00", "2211840", "20.00" });
    try expectCount("mock launches (no replay in events mode)", try mockCount("hipmock_launches"), 5);
    try expectCount("mock syncs (events mode never syncs)", try mockCount("hipmock_syncs"), 0);
}

fn modeGraph(e: *Engine) !void {
    const graph = try shimFn("hipGraphLaunch");
    _ = graph(1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    _ = graph(1, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    e.launchNorm();
    const n = try dumpUntil(nonEmpty);
    try expectCount("rows", n, 2);
    try expectRow(n, "GRAPH_LAUNCH", .{ "2", "0,0,0", "0,0,0", "0", "0", "0", "0", "0", "*", "*", "*", "0.00", "0", "0.00" });
    try expectCount("mock graph launches", try mockCount("hipmock_graphs"), 2);
}

fn modeStart(e: *Engine) !void {
    e.launchRow();
    e.launchRow();
    const empty = try dumpUntil(any);
    try expectCount("rows before the first SIGUSR1", empty, 0);
    try expectCount("syncs before the first SIGUSR1", try mockCount("hipmock_syncs"), 0);
    _ = raise(usr1);
    // the watcher thread starts the timing; a probe launch tells when it has
    for (0..500) |_| {
        e.launch(e.norm, 1, null);
        if ((try mockCount("hipmock_syncs")) > 0) break;
        _ = usleep(10_000);
    } else return fail("SIGUSR1 never started the timing", .{});
    e.launchRow();
    e.launchRow();
    const n = try dumpUntil(nonEmpty);
    try expectRow(n, "affine_dot2_row", .{ "2", "32,1,1", "64,1,1", "0", "1", "4096", "2560", "0", "*", "*", "*", "0.00", "5898240", "0.00" });
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 5) return fail("usage: hipprof_check MODE SHIM MOCK DIR", .{});
    const mode = std.mem.sliceTo(args[1], 0);
    const dir = std.mem.sliceTo(args[4], 0);
    const after_exit = std.mem.eql(u8, mode, "after-exit"); // reads what the "exit" run wrote in the same DIR
    const name = if (after_exit) "exit" else mode;
    log_path = try std.fmt.allocPrintSentinel(init.arena.allocator(), "{s}/{s}.table.txt", .{ dir, name }, 0);
    if (after_exit) {
        const n = readTable() orelse return fail("the process that exited left no complete table", .{});
        try expectCount("rows", n, 1);
        try expectRow(n, "affine_dot2_row", .{ "1", "32,1,1", "64,1,1", "0", "1", "4096", "2560", "0", "*", "*", "*", "0.00", "5898240", "0.00" });
        _ = std.c.write(1, "ok\n", 3);
        return;
    }
    const trace_path = try std.fmt.allocPrintSentinel(init.arena.allocator(), "{s}/{s}.trace.txt", .{ dir, mode }, 0);
    _ = unlink(trace_path);
    _ = unlink(log_path);
    setEnv("HIPPROF_REAL", args[3]);
    setEnv("HIPPROF_LOG", log_path);
    if (std.mem.eql(u8, mode, "each")) {
        setEnv("HIPPROF_REP", "2");
        setEnv("HIPPROF_TRACE", trace_path);
        setEnv("HIPPROF_TRACE_N", "3");
        setEnv("HIPPROF_XCHK", "1");
    } else if (std.mem.eql(u8, mode, "events")) {
        setEnv("HIPPROF_EVENTS", "1");
    } else if (std.mem.eql(u8, mode, "graph")) {
        setEnv("HIPPROF_GRAPH", "1");
    } else if (std.mem.eql(u8, mode, "start")) {
        setEnv("HIPPROF_START", "1");
        setEnv("HIPPROF_REP", "0");
    } else if (std.mem.eql(u8, mode, "exit")) {
        setEnv("HIPPROF_REP", "0"); // no dump is asked for: the one at process exit writes the table
    } else return fail("unknown mode {s}", .{mode});
    mock = try open(args[3], 2);
    shim = try open(args[2], 2);
    var engine: Engine = .{};
    try engine.setup();
    if (std.mem.eql(u8, mode, "each")) try modeEach(&engine) else if (std.mem.eql(u8, mode, "events")) try modeEvents(&engine) else if (std.mem.eql(u8, mode, "graph")) try modeGraph(&engine) else if (std.mem.eql(u8, mode, "start")) try modeStart(&engine) else engine.launchRow();
    if (std.mem.eql(u8, mode, "each")) try checkTrace(trace_path);
    _ = std.c.write(1, "ok\n", 3);
}

fn checkTrace(path: [:0]const u8) !void {
    const f = fopen(path, "r") orelse return fail("no trace file {s}", .{path});
    defer _ = std.c.fclose(f);
    const len = fread(&file_buf, 1, file_buf.len, f);
    const want = "affine_dot2_row g=32,1,1 b=64,1,1 m=1 n=4096 k=2560\n" ++
        "affine_dot2_stream g=8,1,1 b=64,1,1 m=1 n=1536 k=2560\n" ++
        "affine_dot2_row g=32,1,1 b=64,1,1 m=1 n=4096 k=2560\n";
    if (!std.mem.eql(u8, file_buf[0..len], want)) return fail("trace file:\n{s}", .{file_buf[0..len]});
}
