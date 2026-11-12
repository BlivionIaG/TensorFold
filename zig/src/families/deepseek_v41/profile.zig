//! TF_DSV41_PROFILE=<file>: eager windows' calls timed on the GPU plus host wall, summed by name, written as JSON.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const run = @import("run.zig");

pub const Profile = struct {
    gpa: std.mem.Allocator,
    d: *const cuda.Driver,
    path: []const u8,
    events: std.ArrayList(cuda.Event) = .empty,
    /// this window's calls: name and the index of its start event
    open: std.ArrayList(struct { name: []const u8, at: usize, layer: u32 }) = .empty,
    /// the layer position within the open window (calls begin .layer at each layer after the first)
    layer: u32 = 0,
    used: usize = 0,
    totals: std.StringArrayHashMapUnmanaged(Total) = .empty,
    windows: u64 = 0,
    wall_ns: u64 = 0,
    gpu_ns: u64 = 0,
    t0: u64 = 0,
    /// the open window is a prefill segment (beginPrefill): its times go to `pf`
    prefill: bool = false,
    pf: Totals = .{},
    /// the prefill window that just ended: its calls summed by name, until `prefillRows` writes its line
    cur: std.StringArrayHashMapUnmanaged(Total) = .empty,
    cur_wall_ns: u64 = 0,
    cur_gpu_ns: u64 = 0,
    /// one JSON line a prefill window (TF_DSV41_PROFILE's `<file>.windows.jsonl`): rows before it, its rows, times, calls
    lines: std.ArrayList(u8) = .empty,
    lines_n: u64 = 0,

    pub const Totals = struct {
        calls: std.StringArrayHashMapUnmanaged(Total) = .empty,
        windows: u64 = 0,
        wall_ns: u64 = 0,
        gpu_ns: u64 = 0,
        /// prompt rows the encoder / full segments covered, and the decoder replays'
        rows: u64 = 0,
        tail_rows: u64 = 0,
    };

    pub const Total = struct { calls: u64 = 0, ns: u64 = 0 };

    /// The runner's view of `p` (Runner.prof).
    pub fn timer(p: *Profile) run.Timer {
        return .{ .ptr = p, .vtable = &timer_vtable };
    }

    const timer_vtable: run.Timer.VTable = .{
        .begin = struct {
            fn f(ptr: *anyopaque) void {
                self(ptr).begin();
            }
        }.f,
        .beginPrefill = struct {
            fn f(ptr: *anyopaque) void {
                self(ptr).beginPrefill();
            }
        }.f,
        .prefillRows = struct {
            fn f(ptr: *anyopaque, rows: u64, tail: bool) void {
                self(ptr).prefillRows(rows, tail);
            }
        }.f,
        .before = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream, c: *const calls.Call) anyerror!void {
                return self(ptr).before(s, c);
            }
        }.f,
        .after = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream) anyerror!void {
                return self(ptr).after(s);
            }
        }.f,
        .end = struct {
            fn f(ptr: *anyopaque, s: cuda.Stream) anyerror!void {
                return self(ptr).end(s);
            }
        }.f,
        .close = struct {
            fn f(ptr: *anyopaque, io: std.Io) void {
                self(ptr).close(io);
            }
        }.f,
    };

    fn self(ptr: *anyopaque) *Profile {
        return @ptrCast(@alignCast(ptr));
    }

    pub fn fromEnv(gpa: std.mem.Allocator, d: *const cuda.Driver) !?*Profile {
        const p = std.c.getenv("TF_DSV41_PROFILE") orelse return null;
        const x = try gpa.create(Profile);
        x.* = .{ .gpa = gpa, .d = d, .path = std.mem.span(p) };
        return x;
    }

    fn event(p: *Profile) !*cuda.Event {
        if (p.used == p.events.items.len) try p.events.append(p.gpa, try cuda.Event.init(p.d, true));
        p.used += 1;
        return &p.events.items[p.used - 1];
    }

    pub fn begin(p: *Profile) void {
        p.prefill = false;
        p.used = 0;
        p.open.clearRetainingCapacity();
        p.t0 = nowNs();
    }

    /// A prefill segment's window (its end adds to the prefill totals).
    pub fn beginPrefill(p: *Profile) void {
        p.begin();
        p.prefill = true;
        p.cur.clearRetainingCapacity();
    }

    /// The rows the last prefill window covered (`tail`: a CED decoder replay's): its line joins the windows file.
    pub fn prefillRows(p: *Profile, rows: u64, tail: bool) void {
        p.windowLine(rows, tail) catch |e| std.log.err("profile: window line not written ({t})", .{e});
        if (tail) p.pf.tail_rows += rows else p.pf.rows += rows;
    }

    /// One JSON line for the window `end` just summed: {"i", "at" (rows before it), "rows", "tail", wall/gpu ms, "calls": {name: [n, ms]}}.
    fn windowLine(p: *Profile, rows: u64, tail: bool) !void {
        if (p.cur.count() == 0) return;
        var w: std.Io.Writer.Allocating = .init(p.gpa);
        defer w.deinit();
        const at = if (tail) p.pf.tail_rows else p.pf.rows;
        try w.writer.print("{{\"i\": {d}, \"at\": {d}, \"rows\": {d}, \"tail\": {}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.lines_n, at, rows, tail, @as(f64, @floatFromInt(p.cur_wall_ns)) / 1e6, @as(f64, @floatFromInt(p.cur_gpu_ns)) / 1e6 });
        for (p.cur.keys(), p.cur.values(), 0..) |k, v, i| try w.writer.print("{s}\"{s}\": [{d}, {d:.3}]", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 });
        try w.writer.writeAll("}}\n");
        try p.lines.appendSlice(p.gpa, w.written());
        p.cur.clearRetainingCapacity();
        p.lines_n += 1;
        if (p.lines_n % 4 == 0) p.flushLines();
    }

    /// `<file>.windows.jsonl`: every line so far (rewritten whole; the engine may be killed before `close`).
    fn flushLines(p: *Profile) void {
        const path = std.fmt.allocPrint(p.gpa, "{s}.windows.jsonl", .{p.path}) catch return;
        defer p.gpa.free(path);
        const io = std.Io.Threaded.global_single_threaded.io();
        std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = p.lines.items }) catch |e| std.log.err("profile: {s} not written ({t})", .{ path, e });
    }

    /// Before a call: its start event on the stream.
    pub fn before(p: *Profile, s: cuda.Stream, c: *const calls.Call) !void {
        const at = p.used;
        if (c.begin == .window) p.layer = 0 else if (c.begin == .layer) p.layer += 1;
        try (try p.event()).record(s);
        try p.open.append(p.gpa, .{ .name = c.name, .at = at, .layer = p.layer });
    }

    /// After a call: its end event.
    pub fn after(p: *Profile, s: cuda.Stream) !void {
        try (try p.event()).record(s);
    }

    /// The window's end: wait for the stream, add every call's time and the window's wall time.
    pub fn end(p: *Profile, s: cuda.Stream) !void {
        try s.synchronize();
        const wall = nowNs() -| p.t0;
        const map = if (p.prefill) &p.pf.calls else &p.totals;
        if (p.prefill) {
            p.pf.wall_ns += wall;
            p.pf.windows += 1;
        } else {
            p.wall_ns += wall;
            p.windows += 1;
        }
        if (p.prefill) {
            p.cur_wall_ns = wall;
            p.cur_gpu_ns = 0;
        }
        for (p.open.items) |o| {
            const ms = try p.events.items[o.at].elapsedMs(p.events.items[o.at + 1]);
            const ns: u64 = @intFromFloat(@max(0.0, ms) * 1e6);
            const g = try map.getOrPut(p.gpa, o.name);
            if (!g.found_existing) {
                g.key_ptr.* = try p.gpa.dupe(u8, o.name);
                g.value_ptr.* = .{};
            }
            g.value_ptr.calls += 1;
            g.value_ptr.ns += ns;
            if (p.prefill) {
                p.pf.gpu_ns += ns;
                p.cur_gpu_ns += ns;
                const c = try p.cur.getOrPut(p.gpa, g.key_ptr.*);
                if (!c.found_existing) c.value_ptr.* = .{};
                c.value_ptr.calls += 1;
                c.value_ptr.ns += ns;
            } else p.gpu_ns += ns;
            var kb: [24]u8 = undefined;
            const lk = std.fmt.bufPrint(&kb, "layer.{d:0>2}", .{o.layer}) catch continue;
            const lg = try map.getOrPut(p.gpa, lk);
            if (!lg.found_existing) {
                lg.key_ptr.* = try p.gpa.dupe(u8, lk);
                lg.value_ptr.* = .{};
            }
            lg.value_ptr.calls += 1;
            lg.value_ptr.ns += ns;
            if (p.prefill) {
                const c = try p.cur.getOrPut(p.gpa, lg.key_ptr.*);
                if (!c.found_existing) c.value_ptr.* = .{};
                c.value_ptr.calls += 1;
                c.value_ptr.ns += ns;
            }
        }
        p.prefill = false;
    }

    /// Writes the totals (JSON) and frees everything.
    pub fn close(p: *Profile, io: std.Io) void {
        var w: std.Io.Writer.Allocating = .init(p.gpa);
        defer w.deinit();
        w.writer.print("{{\"windows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.windows, @as(f64, @floatFromInt(p.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.gpu_ns)) / 1e6 }) catch {};
        for (p.totals.keys(), p.totals.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.print("}}, \"prefill\": {{\"segments\": {d}, \"rows\": {d}, \"tail_rows\": {d}, \"wall_ms\": {d:.3}, \"gpu_ms\": {d:.3}, \"calls\": {{", .{ p.pf.windows, p.pf.rows, p.pf.tail_rows, @as(f64, @floatFromInt(p.pf.wall_ns)) / 1e6, @as(f64, @floatFromInt(p.pf.gpu_ns)) / 1e6 }) catch {};
        for (p.pf.calls.keys(), p.pf.calls.values(), 0..) |k, v, i| w.writer.print("{s}\"{s}\": {{\"calls\": {d}, \"ms\": {d:.4}}}", .{ if (i == 0) "" else ", ", k, v.calls, @as(f64, @floatFromInt(v.ns)) / 1e6 }) catch {};
        w.writer.print("}}}}}}\n", .{}) catch {};
        if (std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p.path, .data = w.written() })) |_| {
            std.log.info("profile: {s} written ({d} windows, {d} prefill segments)", .{ p.path, p.windows, p.pf.windows });
        } else |e| std.log.err("profile: {s} not written ({t})", .{ p.path, e });
        p.flushLines();
        p.lines.deinit(p.gpa);
        p.cur.deinit(p.gpa);
        for (p.events.items) |*e| e.deinit();
        p.events.deinit(p.gpa);
        p.open.deinit(p.gpa);
        for (p.totals.keys()) |k| p.gpa.free(k);
        p.totals.deinit(p.gpa);
        for (p.pf.calls.keys()) |k| p.gpa.free(k);
        p.pf.calls.deinit(p.gpa);
        p.gpa.destroy(p);
    }
};

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}
