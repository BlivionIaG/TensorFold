//! TF_DSV41_HANDOFF_CHECK: land a hand-off on the loaded model with no network, then measure it against a local prefill.
//!
//! `roundtrip` runs the Metal lane's own prefill, exports that slot through the library's Metal-to-neutral direction as a
//! hand-off file, lands the file in a blank Snapshot exactly as a pull does, and compares the continuation with the original
//! (the loss is the fp8/fp4 requantisation of a dequantised row, and the carry). Any other value names a hand-off file (the
//! library's blob: tokens, protocol 2 manifest, parts; a synthetic one or a dump of a producer's export): it lands the same
//! way and is compared with a local prefill of the same prompt. The full prompt comes from TF_DSV41_HANDOFF_TOKENS (decimal
//! ids, white space or commas): the file covers its first `upto` tokens and at least one more must follow. A remote prefix
//! is outside the exactness contract: agreement is reported, not required. TF_DSV41_HANDOFF_DUMP=<path> keeps the round trip's
//! file, which TF_DSV41_HANDOFF_FILE then serves as the remote source of a later (or this) server.
const std = @import("std");
const tf = @import("tensorfold");
const ds = tf.deepseek_v41;
const ho = tf.deepseek_v41_handoff;
const remote = @import("deepseek_v41_remote.zig");

const vocab = 129280;
const log = std.log.scoped(.handoff_check);

fn envText(name: [*:0]const u8) ?[]const u8 {
    return if (std.c.getenv(name)) |v| std.mem.span(v) else null;
}

fn argmax(row: []const f32) u32 {
    var best: u32 = 0;
    for (row, 0..) |v, i| if (v > row[best]) {
        best = @intCast(i);
    };
    return best;
}

/// ‖a - b‖ / ‖a‖ over a vocabulary row.
fn nrms(a: []const f32, b: []const f32) f64 {
    var num: f64 = 0;
    var den: f64 = 0;
    for (a, b) |x, y| {
        const d: f64 = @as(f64, x) - y;
        num += d * d;
        den += @as(f64, x) * x;
    }
    return @sqrt(num / @max(den, 1e-30));
}

fn readTokens(a: std.mem.Allocator, io: std.Io, path: []const u8) ![]u32 {
    const text = try std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20));
    defer a.free(text);
    var out: std.ArrayList(u32) = .empty;
    errdefer out.deinit(a);
    var it = std.mem.tokenizeAny(u8, text, " \t\r\n,[]");
    while (it.next()) |w| try out.append(a, try std.fmt.parseInt(u32, w, 10));
    return out.toOwnedSlice(a);
}

/// `ids` through slot 0 in workspace-sized chunks; the last chunk's final row (copied) when `last`.
fn feed(m: *ds.Model, a: std.mem.Allocator, ids: []const u32, last: bool) !?[]f32 {
    const rows = m.workspace.plan.rows;
    var at: usize = 0;
    while (at < ids.len) {
        const end = @min(at + rows, ids.len);
        if (last and end == ids.len) return try a.dupe(f32, try m.forward(ids[at..end], .last));
        _ = try m.forward(ids[at..end], .none);
        at = end;
    }
    return null;
}

/// `steps` greedy tokens after `first_logits`; `forced` (a reference stream) is fed instead of the run's own picks. Returns the
/// logits of every step (steps + 1 rows, the first being `first_logits`) and the run's own picks.
fn decode(m: *ds.Model, a: std.mem.Allocator, first_logits: []const f32, forced: ?[]const u32, steps: usize, picks: []u32) ![]f32 {
    const rows = try a.alloc(f32, (steps + 1) * vocab);
    @memcpy(rows[0..vocab], first_logits);
    for (0..steps) |k| {
        picks[k] = argmax(rows[k * vocab ..][0..vocab]);
        const feed_id = [1]u32{if (forced) |f| f[k] else picks[k]};
        @memcpy(rows[(k + 1) * vocab ..][0..vocab], try m.forward(&feed_id, .last));
    }
    return rows;
}

fn compare(name: []const u8, want: []const f32, got: []const f32, steps: usize) void {
    var agree: usize = 0;
    var worst: f64 = 0;
    var total: f64 = 0;
    for (0..steps + 1) |k| {
        const r = want[k * vocab ..][0..vocab];
        const g = got[k * vocab ..][0..vocab];
        agree += @intFromBool(argmax(r) == argmax(g));
        const e = nrms(r, g);
        worst = @max(worst, e);
        total += e;
    }
    log.info("{s}: top-1 agrees on {d}/{d} steps, NRMS mean {d:.5} worst {d:.5}", .{ name, agree, steps + 1, total / @as(f64, @floatFromInt(steps + 1)), worst });
}

fn streamAgree(want: []const u32, got: []const u32) usize {
    var n: usize = 0;
    while (n < want.len and want[n] == got[n]) n += 1;
    return n;
}

/// Runs the check named by `spec` against slot 0 of `m`; every failure is logged and returned.
pub fn run(gpa: std.mem.Allocator, io: std.Io, m: *ds.Model, spec: []const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const steps: usize = if (envText("TF_DSV41_HANDOFF_STEPS")) |s| try std.fmt.parseInt(usize, s, 10) else 16;
    const round = std.mem.eql(u8, spec, "roundtrip");
    const file: ?ho.blob.Blob = if (round) null else try ho.blob.Blob.parse(a, try std.Io.Dir.cwd().readFileAlloc(io, spec, a, .limited(8 << 30)));
    var prompt: []u32 = if (envText("TF_DSV41_HANDOFF_TOKENS")) |p| try readTokens(a, io, p) else blk: {
        const n: usize = if (file) |f| f.tokens.len + 64 else 1500;
        const t = try a.alloc(u32, n);
        for (t, 0..) |*id, i| id.* = @intCast(1200 + (i * 37) % 9000);
        break :blk t;
    };
    if (file) |f| {
        if (prompt.len <= f.tokens.len or !std.mem.eql(u32, prompt[0..f.tokens.len], f.tokens)) {
            if (envText("TF_DSV41_HANDOFF_TOKENS") != null) return error.PromptDoesNotExtendTheFile;
            const joined = try a.alloc(u32, f.tokens.len + 64);
            @memcpy(joined[0..f.tokens.len], f.tokens);
            for (joined[f.tokens.len..], 0..) |*id, i| id.* = @intCast(1200 + (i * 37) % 9000);
            prompt = joined;
        }
    }
    const upto: u32 = if (file) |f| @intCast(f.tokens.len) else @intCast(@max(1, (prompt.len - 1) / 128 * 128));
    if (prompt.len < upto + 1 or prompt.len + steps + 1 > m.caches[0].plan.capacity) return error.PromptTooShortOrLong;
    log.info("{d} prompt tokens, prefix {d}, {d} steps, {s}", .{ prompt.len, upto, steps, if (round) "Metal round trip" else spec });

    // the local prefill, with the native slot at `upto` kept for the round trip's export
    try m.reset();
    const t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    _ = try feed(m, a, prompt[0..upto], false);
    const native = try m.snapshotSave(0);
    defer m.snapshotDrop(native);
    const first = (try feed(m, a, prompt[upto..], true)).?;
    const t1 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const ref_picks = try a.alloc(u32, steps);
    const ref = try decode(m, a, first, null, steps, ref_picks);
    log.info("local prefill of {d} tokens: {d:.2} s", .{ prompt.len, @as(f64, @floatFromInt(t1 - t0)) / 1e9 });

    // the control: the native snapshot restored into the slot continues exactly as the local run did
    try m.reset();
    try m.snapshotRestore(0, native);
    const control = try decode(m, a, (try feed(m, a, prompt[upto..], true)).?, ref_picks, steps, try a.alloc(u32, steps));
    compare("control (the native snapshot restored)", ref, control, steps);

    // the hand-off: a file (or the round trip's own export) landed in a blank Snapshot as a pull lands it
    const snap = try m.snapshotBlank();
    defer m.snapshotDrop(snap);
    const bytes = if (file != null) null else try exportNative(a, m, native, prompt[0..upto]);
    if (bytes) |b| if (envText("TF_DSV41_HANDOFF_DUMP")) |path| {
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = b });
        log.info("wrote the round trip's hand-off file ({d} bytes, {d} tokens) to {s}", .{ b.len, upto, path });
    };
    const landed = if (file) |f| f else try ho.blob.Blob.parse(a, bytes.?);
    var landing: ho.landing.Landing = undefined;
    const t2 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    try landing.init(gpa, upto, .{ .@"packed" = remote.packedOf(m, snap) });
    defer landing.deinit();
    try landed.feed(landing.decoderSink());
    _ = try landing.complete();
    try m.snapshotSeal(snap, prompt, upto);
    const t3 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    log.info("landed {d} tokens ({d} ring parts dropped) in {d:.1} ms", .{ upto, landing.dropped, @as(f64, @floatFromInt(t3 - t2)) / 1e6 });

    try m.reset();
    try m.snapshotRestore(0, snap);
    const tail_t0 = std.Io.Timestamp.now(io, .awake).toNanoseconds();
    const got_first = (try feed(m, a, prompt[upto..], true)).?;
    log.info("local tail of {d} tokens after the restore: {d:.1} ms", .{ prompt.len - upto, @as(f64, @floatFromInt(std.Io.Timestamp.now(io, .awake).toNanoseconds() - tail_t0)) / 1e6 });
    const got = try decode(m, a, got_first, ref_picks, steps, try a.alloc(u32, steps));
    compare("hand-off, the reference's tokens fed", ref, got, steps);

    // free-running greedy on the imported prefix
    try m.reset();
    try m.snapshotRestore(0, snap);
    const own_picks = try a.alloc(u32, steps);
    _ = try decode(m, a, (try feed(m, a, prompt[upto..], true)).?, null, steps, own_picks);
    log.info("hand-off, free-running: {d}/{d} tokens equal the local run's, ids {any}", .{ streamAgree(ref_picks, own_picks), steps, own_picks });
    log.info("local run's ids {any}", .{ref_picks});
    try m.reset();
}

/// The native slot as a hand-off file: the Metal lane's own state through the library's Metal-to-neutral export.
fn exportNative(a: std.mem.Allocator, m: *ds.Model, native: *const ds.Model.Snapshot, tokens: []const u32) ![]u8 {
    const view = remote.packedOf(m, @constCast(native));
    const parts = try ho.export_metal.build(a, .{ .upto = tokens.len }, view);
    return ho.blob.write(a, "dsv41-roundtrip", tokens, parts);
}
