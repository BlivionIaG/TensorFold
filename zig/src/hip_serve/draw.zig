//! Tokens from logits rows on the device: argmax per greedy row, top_k + MARGIN candidates per sampled row.

const std = @import("std");
const hip = @import("hip");
const lanes = @import("lanes");
const sample = @import("sample.zig");

/// Most candidates a row downloads; wider requests (top_k 0, or past this) download the row and draw on the host.
pub const max_candidates = 512;

/// One row's draw: its sampling (null is greedy), the position that keys it, the grammar's allowed bits, a host copy.
pub const Request = struct {
    sampling: ?lanes.Sampling,
    position: u64,
    /// Structured output: only tokens whose bit is set (empty: every token).
    mask: []const u32 = &.{},
    /// The row's logits copied here (the prompt's last row, kept for a masked first draw).
    keep: ?[]u16 = null,
    /// The checkpoint's multiplicative penalty over prompt and reply tokens, applied once per seen token in FP32.
    repetition: f32 = 1,
    /// A checkpoint-forced final EOS, after its ordinary score processors.
    forced: ?u32 = null,
    history: []const u32 = &.{},
    /// OpenAI penalties count reply tokens only; history may also contain prompt tokens.
    reply: []const u32 = &.{},
    penalties: lanes.sampling.Penalties = .{},
};

/// Whether a row is drawn on the host from the whole row.
fn whole(r: Request) bool {
    return r.mask.len > 0 or r.keep != null or r.repetition != 1 or r.penalties.active() or r.forced != null;
}

fn greedy(s: ?lanes.Sampling) bool {
    return if (s) |x| x.temperature <= 0.0 else true;
}

/// Candidates a sampled row asks for, or 0 when it takes the whole row.
fn candidates(s: lanes.Sampling, vocab: usize) usize {
    if (s.top_k == 0) return 0;
    const n = @min(vocab, @as(usize, s.top_k) + sample.MARGIN);
    return if (n > max_candidates) 0 else n;
}

fn view(comptime T: type, bytes: []u8) []T {
    return std.mem.bytesAsSlice(T, @as([]align(@alignOf(T)) u8, @alignCast(bytes)));
}

/// Slots of the host mirror the argmax launch stores its picks to: slot 0 for rounds the host waits for at once, the others in turn for
/// the rounds queued ahead (the host reads a round's slot long before the ring comes round to it again).
pub const mirror_slots = 8;

/// What a round's rows ask the device for: whether any row is a plain greedy draw, and the widest row's candidate count.
pub const Asks = struct { greedy: bool, widest: usize };

/// What `reqs` ask the device for, each row's candidate count (0 for a greedy, a whole or a padding row) in `ks`.
pub fn asksOf(vocab: usize, reqs: []const Request, ks: []i32) Asks {
    var a: Asks = .{ .greedy = false, .widest = 0 };
    for (reqs, ks[0..reqs.len]) |r, *k| {
        k.* = 0;
        if (whole(r)) continue;
        if (greedy(r.sampling)) {
            a.greedy = true;
            continue;
        }
        const n = candidates(r.sampling.?, vocab);
        k.* = @intCast(n);
        a.widest = @max(a.widest, n);
    }
    return a;
}

/// What the drawing launches of a round did, which `draw` then reads: the argmax ran (`argmax`) and stored its picks to mirror slot
/// `mirror` (null: the picks are at `argmaxAt`, downloaded), the top-k launch ran with `stride` candidates a row over `rows` rows (0: it did not).
pub const Done = struct { argmax: bool = false, mirror: ?usize = null, stride: u16 = 0, rows: usize = 0 };

/// The drawing launches of one round's forward: the argmax (where its picks go on the device and, through the plan word at `mirror_at`,
/// to the host) and the top-k over `rows` rows with `stride` candidates a row, its counts, ids and values.
pub const Launches = struct {
    argmax: bool = true,
    argmax_at: u64 = 0,
    mirror_at: u64 = 0,
    rows: usize = 0,
    stride: usize = 0,
    ks: u64 = 0,
    ids: u64 = 0,
    vals: u64 = 0,
};

pub const Drawer = struct {
    gpa: std.mem.Allocator,
    d: *const hip.Runtime,
    dtype: sample.Dtype,
    vocab: usize,
    rows: usize,
    /// Per row: the candidate count, the argmax, then (rows, stride) candidate ids and value patterns.
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    row: hip.HostBuffer,
    /// The picks of argmax launches as the host reads them: `mirror_slots` slots of `rows` words, mapped into the device.
    mirror: hip.HostBuffer,
    mirror_dev: u64,

    /// Device bytes a drawer of `rows` rows holds: each row's argmax and its sampling candidates.
    pub fn deviceBytes(rows: usize) usize {
        return rows * (8 + max_candidates * 6);
    }

    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, dtype: sample.Dtype, vocab: usize, rows: usize) !Drawer {
        const bytes = deviceBytes(rows);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        var dev = try hip.DeviceBuffer.alloc(d, bytes);
        errdefer dev.free();
        var row = try hip.HostBuffer.alloc(d, vocab * 2);
        errdefer row.free();
        var mirror = try hip.HostBuffer.allocMapped(d, mirror_slots * rows * 4);
        errdefer mirror.free();
        const mirror_dev = try mirror.device();
        @memset(mirror.bytes, 0);
        return .{ .gpa = gpa, .d = d, .dtype = dtype, .vocab = vocab, .rows = rows, .host = host, .dev = dev, .row = row, .mirror = mirror, .mirror_dev = mirror_dev };
    }

    pub fn deinit(w: *Drawer) void {
        w.mirror.free();
        w.row.free();
        w.dev.free();
        w.host.free();
    }

    /// Offsets into the scratch of a round of `rows` rows of `stride` candidates each, as byte counts: argmax, ids, values (ks come first,
    /// as a plan's words for a graph and as an upload for the eager top-k). The candidates are contiguous: one copy brings them down.
    fn layout(w: *const Drawer, rows: usize, stride: usize) struct { arg: usize, ids: usize, vals: usize, total: usize } {
        const arg = w.rows * 4;
        const ids = arg + w.rows * 4;
        const vals = ids + rows * stride * 4;
        return .{ .arg = arg, .ids = ids, .vals = vals, .total = vals + rows * stride * 2 };
    }

    /// Where the argmax of a logits block's first rows goes on the device (a graph can hold the launch).
    pub fn argmaxAt(w: *const Drawer) u64 {
        return w.dev.base() + w.layout(0, 0).arg;
    }

    /// The host address mirror slot `slot` starts at, as an argmax launch stores to it (a device address, the host pointer's).
    pub fn mirrorAddr(w: *const Drawer, slot: usize) u64 {
        return w.mirror_dev + 4 * slot * w.rows;
    }

    /// Row `r`'s pick in mirror slot `slot`; valid once the fence behind the launch that stored it has been waited on.
    pub fn mirrored(w: *const Drawer, slot: usize, r: usize) i32 {
        const words = view(i32, w.mirror.bytes);
        return @atomicLoad(i32, &words[slot * w.rows + r], .acquire);
    }

    /// What `reqs` ask the device for, each row's candidate count (0 for a greedy, a whole or a padding row) in `ks`.
    pub fn asks(w: *const Drawer, reqs: []const Request, ks: []i32) Asks {
        return asksOf(w.vocab, reqs, ks);
    }

    /// The drawing launches of a graph's forward over `rows` rows with `stride` candidates a row (0: no top-k); the argmax stores its picks
    /// to `out`, and through the plan word at `mirror_at` to the host; the candidate counts are the plan words at `ks`.
    pub fn launches(w: *const Drawer, rows: usize, stride: usize, argmax: bool, out: u64, mirror_at: u64, ks: u64) Launches {
        const at = w.layout(rows, stride);
        const base = w.dev.base();
        return .{ .argmax = argmax, .argmax_at = out, .mirror_at = mirror_at, .rows = rows, .stride = stride, .ks = ks, .ids = base + at.ids, .vals = base + at.vals };
    }

    /// `out[r]` the token of row r of `logits` per `reqs[r]`, stream synchronized; `argmaxed`: the greedy rows are drawn already.
    pub fn draw(w: *Drawer, o: hip.ops.Ops, stream: hip.Stream, logits: hip.ops.Tensor, reqs: []const Request, out: []u32, argmaxed: bool) !void {
        return w.drawWith(o, stream, logits, reqs, out, .{ .argmax = argmaxed });
    }

    /// `draw` after a graph's own drawing launches: `done` names what they drew (the greedy rows to the device, or also the host).
    pub fn drawWith(w: *Drawer, o: hip.ops.Ops, stream: hip.Stream, logits: hip.ops.Tensor, reqs: []const Request, out: []u32, done: Done) !void {
        const rows = reqs.len;
        if (rows > w.rows) return error.WindowTooWide;
        const ks = w.host.slice(i32)[0..rows];
        const a = w.asks(reqs, ks);
        // a round's top-k launch had its stride's room for each of its rows; an eager one takes the widest row's
        if (done.stride > 0 and (a.widest > done.stride or done.rows < rows or done.rows > w.rows)) return error.Invalid;
        const stride: usize = if (done.stride > 0) done.stride else a.widest;
        const at = w.layout(if (done.stride > 0) done.rows else rows, stride);
        const base = w.dev.base();
        if (a.greedy and !done.argmax) try o.argmaxRows(logits, rows, w.vocab, base + at.arg);
        if (a.widest > 0 and done.stride == 0) {
            try hip.raw.upload(w.dev, 0, std.mem.sliceAsBytes(ks), stream.handle);
            try o.topkRows(logits, rows, w.vocab, base, stride, base + at.ids, base + at.vals);
        }
        const host = w.host.bytes;
        const from_mirror = a.greedy and done.mirror != null;
        if (a.greedy and !from_mirror) try hip.raw.download(w.dev, at.arg, host[at.arg..][0 .. rows * 4], stream.handle);
        if (a.widest > 0) try hip.raw.download(w.dev, at.ids, host[at.ids..at.total], stream.handle);
        try stream.synchronize();
        const arg = view(i32, host[at.arg..][0 .. rows * 4]);
        for (reqs, 0..) |r, i| {
            if (whole(r)) {
                out[i] = try w.drawWhole(stream, logits, i, r);
                continue;
            }
            if (greedy(r.sampling)) {
                out[i] = @intCast(if (done.mirror) |slot| w.mirrored(slot, i) else arg[i]);
                continue;
            }
            const s = r.sampling.?;
            const n: usize = @intCast(ks[i]);
            if (n == 0) {
                const bytes = w.vocab * 2;
                try hip.runtime.check(w.d.api.hipMemcpyDtoHAsync(w.row.bytes.ptr, @ptrFromInt(logits.ptr + i * bytes), bytes, stream.handle));
                try stream.synchronize();
                out[i] = try sample.draw(w.gpa, w.row.slice(u16)[0..w.vocab], w.dtype, s, r.position);
                continue;
            }
            const ids = view(i32, host[at.ids + i * stride * 4 ..][0 .. n * 4]);
            const vals = view(u16, host[at.vals + i * stride * 2 ..][0 .. n * 2]);
            var values: [max_candidates]f64 = undefined;
            var wide: [max_candidates]u64 = undefined;
            for (0..n) |j| {
                values[j] = sample.widen(vals, w.dtype, j);
                wide[j] = @intCast(ids[j]);
            }
            out[i] = @intCast(try lanes.sampling.choose(w.gpa, values[0..n], wide[0..n], r.position, s));
        }
    }

    /// Row `i` downloaded whole: kept if asked, masked if asked, then drawn as an unmasked whole row is.
    fn drawWhole(w: *Drawer, stream: hip.Stream, logits: hip.ops.Tensor, i: usize, r: Request) !u32 {
        const bytes = w.vocab * 2;
        try hip.runtime.check(w.d.api.hipMemcpyDtoHAsync(w.row.bytes.ptr, @ptrFromInt(logits.ptr + i * bytes), bytes, stream.handle));
        try stream.synchronize();
        const row = w.row.slice(u16)[0..w.vocab];
        if (r.keep) |k| @memcpy(k, row);
        return w.drawHost(row, r);
    }

    /// A host row's token per `r`: its mask applied in place, then greedy or the keyed draw.
    pub fn drawHost(w: *const Drawer, row: []u16, r: Request) !u32 {
        if (r.forced) |id| {
            if (id >= row.len) return error.BadForcedToken;
            if (r.mask.len > 0 and (id / 32 >= r.mask.len or (r.mask[id / 32] >> @intCast(id % 32)) & 1 == 0)) return error.NoAllowedToken;
            return id;
        }
        if (r.mask.len > 0) sample.mask(row, w.dtype, r.mask);
        if (r.repetition == 1 and !r.penalties.active()) return sample.draw(w.gpa, row, w.dtype, r.sampling, r.position);
        const values = try w.gpa.alloc(f32, row.len);
        defer w.gpa.free(values);
        for (values, 0..) |*v, i| v.* = sample.widen(row, w.dtype, i);
        try process(w.gpa, values, r);
        return sample.drawWide(w.gpa, values, r.sampling, r.position);
    }
};

/// HF repetition processing widens raw scores before FP32 multiply/divide; it must not round back to BF16.
pub fn process(a: std.mem.Allocator, values: []f32, r: Request) !void {
    if (!(r.repetition > 0) or !std.math.isFinite(r.repetition)) return error.BadRepetitionPenalty;
    const counts = try a.alloc(u32, values.len);
    defer a.free(counts);
    @memset(counts, 0);
    if (r.repetition != 1) {
        for (r.history) |id| {
            if (id >= values.len) return error.BadHistoryToken;
            counts[id] = 1;
        }
        for (values, counts) |*v, seen| if (seen != 0) {
            v.* = if (v.* < 0) v.* * r.repetition else v.* / r.repetition;
        };
    }
    if (r.penalties.active()) {
        @memset(counts, 0);
        for (r.reply) |id| {
            if (id >= values.len) return error.BadHistoryToken;
            counts[id] += 1;
        }
        for (values, counts) |*v, count| if (count != 0) {
            v.* -= r.penalties.of(count);
        };
    }
}

test "repetition keeps FP32 ranking and counts prompt tokens once, OpenAI penalties count only reply tokens" {
    var values = [_]f32{ 1.5, 1.3671875, -2, 4 };
    try process(std.testing.allocator, &values, .{ .sampling = null, .position = 0, .repetition = 1.1, .history = &.{ 0, 0, 2 }, .reply = &.{ 3, 3 }, .penalties = .{ .presence = 1, .frequency = 0.5 } });
    try std.testing.expectEqual(@as(f32, 1.5) / @as(f32, 1.1), values[0]);
    try std.testing.expectEqual(@as(f32, -2) * @as(f32, 1.1), values[2]);
    try std.testing.expectEqual(@as(f32, 2), values[3]);
    try std.testing.expectEqual(@as(u32, 1), try sample.drawWide(std.testing.allocator, values[0..2], null, 0));
    try std.testing.expectError(error.BadHistoryToken, process(std.testing.allocator, &values, .{ .sampling = null, .position = 0, .repetition = 1.1, .history = &.{4} }));
}

test "a round asks the device for what its rows need: argmax for greedy rows, candidates for sampled ones, nothing for whole rows" {
    const sampled: lanes.Sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 20 };
    var ks: [5]i32 = @splat(-1);
    const reqs = [_]Request{
        .{ .sampling = null, .position = 4 },
        .{ .sampling = sampled, .position = 4 },
        .{ .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 0 }, .position = 4 },
        .{ .sampling = sampled, .position = 4, .mask = &.{1} },
        .{ .sampling = .{ .seed = 1, .temperature = 0.0 }, .position = 4 },
    };
    const all = asksOf(1000, &reqs, &ks);
    try std.testing.expect(all.greedy);
    // top_k 20 and the margin, the same for every sampled row that has candidates
    try std.testing.expectEqual(@as(usize, 20 + sample.MARGIN), all.widest);
    try std.testing.expectEqualSlices(i32, &.{ 0, 28, 0, 0, 0 }, &ks);
    const only = asksOf(1000, reqs[1..2], &ks);
    try std.testing.expect(!only.greedy);
    try std.testing.expectEqual(@as(usize, 28), only.widest);
    // no row asks for candidates when the row is whole or the request is wider than a round downloads
    const wide = asksOf(1000, &.{.{ .sampling = .{ .seed = 1, .temperature = 0.7, .top_k = 600 }, .position = 4 }}, &ks);
    try std.testing.expect(!wide.greedy and wide.widest == 0);
    try std.testing.expectEqual(@as(i32, 0), ks[0]);
}
