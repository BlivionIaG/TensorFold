//! `rows <model dir> <ids.npy> [n] [streams] [keep]`: the window invariant layer by layer. Stream j holds the ids cut j
//! tokens short; its last n tokens run as n one-row rounds alone and, with every stream's window, as one shared round
//! that keeps only its first `keep` rows (a rejected draft's commit), the rest then one row at a time. Every layer's
//! residual rows and the final rows are compared byte for byte, and the first difference is named.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen35");
const ids_file = @import("ids_file.zig");

const Engine = qwen35.engine.Engine;
const win = qwen35.window;

/// Every layer's residual rows of one run: (layers + 1, n, hidden) bytes, the last slot the final rows.
const Sink = struct {
    e: *Engine,
    bytes: []u8,
    n: usize,
    /// The row the next capture lands at.
    row: usize = 0,

    fn width(s: *const Sink) usize {
        return s.e.model().spec.hidden * s.e.model().act.size();
    }

    fn put(s: *Sink, slot: usize, x: hip.ops.Tensor, rows: usize) !void {
        try s.e.stream.synchronize();
        const w = s.width();
        const at = (slot * s.n + s.row) * w;
        try s.e.driver.check(s.e.driver.api.hipMemcpyDtoH(s.bytes[at..].ptr, x.ptr, rows * w), "download");
    }

    fn layer(ctx: *anyopaque, index: usize, x: hip.ops.Tensor, rows: usize) anyerror!void {
        const s: *Sink = @ptrCast(@alignCast(ctx));
        try s.put(index, x, rows);
    }
};

/// One round of every stream's `tokens` (stream j from slot `pos[j]` of `caches[j]`), captured into `sink` from its
/// current row, then kept.
fn round(e: *Engine, caches: []const *qwen35.state.Caches, pos: []const usize, tokens: []const []const u32, keep: usize, sink: *Sink, snaps: []win.Snapshot) !void {
    const m = e.model();
    const layers = m.spec.n_layers;
    e.rounds.reset();
    var total: usize = 0;
    for (tokens) |t| total += t.len;
    var host: [128]i32 = undefined;
    var at: usize = 0;
    for (tokens, pos) |t, p| {
        for (t, 0..) |id, i| {
            host[at + i] = @intCast(id);
            host[total + at + i] = @intCast(p + i);
        }
        at += t.len;
    }
    var dev = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(host[0 .. 2 * total]));
    defer dev.free();
    const o: hip.ops.Ops = .{ .lib = &e.lib, .stream = e.stream.handle, .arena = &e.rounds };
    var wins: [8]win.Window = undefined;
    at = 0;
    for (caches, pos, tokens, 0..) |c, p, t, j| {
        wins[j] = .{ .caches = c, .pos = p, .rows = t.len, .at32 = dev.ptr + (total + at) * 4, .snaps = snaps[j * layers ..][0..layers] };
        at += t.len;
    }
    const trace: qwen35.forward.Trace = .{ .ctx = sink, .layer = Sink.layer };
    const out = try win.forward(o, m, wins[0..caches.len], dev.ptr, trace);
    try sink.put(layers, out, total);
    for (wins[0..caches.len], tokens) |w, t| try win.commit(o, m, w, @min(keep, t.len));
    try e.stream.synchronize();
}

/// Streams of `n` rows each, the first `keep` of each kept.
pub const Case = struct { streams: usize, n: usize, keep: usize };

/// Where a case's two runs first differ.
pub const Diff = struct { layer: usize, kind: []const u8, stream: usize, row: usize, column: usize };

/// Runs one case over prompts cut from `ids`; null when every layer's rows and the final rows are equal.
pub fn runCase(gpa: std.mem.Allocator, e: *Engine, ids: []const u32, c: Case) !?Diff {
    const n = c.n;
    const streams = c.streams;
    if (n < 1 or n > 16) return error.BadWindow;
    if (c.keep < 1 or c.keep > n) return error.BadKeep;
    if (streams < 1 or streams > 8 or ids.len <= n + streams or streams * n > 64) return error.PromptTooShort;
    const layers = e.model().spec.n_layers;
    const snaps = try gpa.alloc(win.Snapshot, streams * layers);
    defer gpa.free(snaps);
    const rows = streams * n;
    var serial: Sink = .{ .e = e, .bytes = undefined, .n = rows };
    var shared: Sink = .{ .e = e, .bytes = undefined, .n = rows };
    serial.bytes = try gpa.alloc(u8, (layers + 1) * rows * serial.width());
    defer gpa.free(serial.bytes);
    shared.bytes = try gpa.alloc(u8, (layers + 1) * rows * shared.width());
    defer gpa.free(shared.bytes);

    // stream j: the ids short by j, its prompt all but its last n tokens
    var alone: [8]qwen35.state.Caches = undefined;
    var together: [8]qwen35.state.Caches = undefined;
    var caches: [8]*qwen35.state.Caches = undefined;
    var pos: [8]usize = undefined;
    var tails: [8][]const u32 = undefined;
    var made: usize = 0;
    defer for (alone[0..made], together[0..made]) |*a1, *t1| {
        a1.deinit(gpa);
        t1.deinit(gpa);
    };
    for (0..streams) |j| {
        const end = ids.len - j;
        const prompt = ids[0 .. end - n];
        tails[j] = ids[end - n .. end];
        pos[j] = prompt.len;
        const req: qwen35.draw.Request = .{ .sampling = null, .position = prompt.len };
        alone[j] = try e.newCaches(ids.len + 8);
        together[j] = e.newCaches(ids.len + 8) catch |err| {
            alone[j].deinit(gpa);
            return err;
        };
        made += 1;
        _ = try e.prefill(&alone[j], prompt, 0, null, req, null);
        _ = try e.prefill(&together[j], prompt, 0, null, req, null);
        caches[j] = &together[j];
    }
    for (0..streams) |j| for (0..n) |i| {
        serial.row = j * n + i;
        const one = [1]*qwen35.state.Caches{&alone[j]};
        try round(e, &one, &.{pos[j] + i}, &.{tails[j][i..][0..1]}, 1, &serial, snaps);
    };
    try round(e, caches[0..streams], pos[0..streams], tails[0..streams], c.keep, &shared, snaps);
    // the rows past `keep` again, one at a time from the kept state, over the shared round's captures
    for (c.keep..n) |i| for (0..streams) |j| {
        shared.row = j * n + i;
        const one = [1]*qwen35.state.Caches{&together[j]};
        try round(e, &one, &.{pos[j] + i}, &.{tails[j][i..][0..1]}, 1, &shared, snaps);
    };

    const w = serial.width();
    for (0..layers + 1) |slot| for (0..rows) |r| {
        const at = (slot * rows + r) * w;
        if (std.mem.indexOfDiff(u8, serial.bytes[at..][0..w], shared.bytes[at..][0..w])) |col| {
            const kind = if (slot == layers) "final" else if (e.model().spec.full(slot)) "full attention" else "linear attention";
            return .{ .layer = slot, .kind = kind, .stream = r / n, .row = r % n, .column = col / e.model().act.size() };
        }
    };
    return null;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 2) return error.MissingArgument;
    const n = if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 3;
    const streams = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else 1;
    const keep = if (args.len > 4) try std.fmt.parseInt(usize, args[4], 10) else n;
    const ids = try ids_file.load(gpa, io, args[1]);
    defer gpa.free(ids);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = ids.len + 64, .batch_rows = 128, .graphs = false });
    defer e.deinit();
    if (try runCase(gpa, e, ids, .{ .streams = streams, .n = n, .keep = keep })) |d| {
        std.debug.print("FAIL layer {d} ({s}), stream {d} row {d} of {d}: first difference at column {d}\n", .{ d.layer, d.kind, d.stream, d.row, n, d.column });
        return error.Mismatch;
    }
    std.debug.print("PASS {d} streams of {d} rows, {d} kept: every layer's residuals and the final rows equal each stream one row at a time\n", .{ streams, n, keep });
}
