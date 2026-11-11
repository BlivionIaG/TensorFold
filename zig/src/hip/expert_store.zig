//! A MoE model's routed experts in two tiers on one HIP device: every expert in a host tier (pinned host memory the device reads in place), the
//! hot ones also in cache lines of device memory. The routed GEMVs take a pointer a (projection, expert) from a table (kernels/hip/tiles/plan.hpp,
//! caps `expert_ptrs`), so a miss is read over the link where the expert lives and a hit from device memory, with no stall and no change of
//! arithmetic: the bytes are the checkpoint's, and an expert computes the same wherever it sits.
//!
//! One blob holds an expert's six tensors in kernel order: the fused (gate | up) projection's words, scales and biases, then the down
//! projection's. Table entry (layer, e) is the device address of expert e's blob, in a cache line or in the host tier; the shared expert
//! (id `experts`) has a line of its own that never leaves.
//!
//! Between rounds, on the host (rules of core/expert_offload.zig, the CUDA EXL3 host cache's):
//!   * `beginRound`: copies that have landed make their experts resident (their entries now name the line);
//!   * `endRound`: the round's router picks (a trace the kernels wrote to host memory) go through `schedule`; the eviction's entry points back at
//!     the host tier first, then the missed expert's copy is queued on a copy stream (SDMA) and runs under the next round's compute.
//!   * `send` puts the table up before any forward reads it, as a node of the round's graph or before a prompt span. A table the device holds is
//!     never newer than the host's, and a line is never named after its eviction: a forward always sees entries whose bytes are intact.
//! Counters (`Stats`) report hits and misses an access (a router pick of a row), copies started and landed.
//! Architecture: plain pointer loads; nothing here names a GPU, and the budget is whatever device memory is left.

const std = @import("std");
const core = @import("core");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const stream_mod = @import("stream.zig");
const raw = @import("raw.zig");

const policy = core.expert_offload;
const abi = @import("abi.zig");
const Runtime = runtime.Runtime;
const DeviceBuffer = memory.DeviceBuffer;
const HostBuffer = memory.HostBuffer;
const Stream = stream_mod.Stream;
const Event = stream_mod.Event;

pub const Error = runtime.Error || std.mem.Allocator.Error || error{ BadShape, NotSized };

/// The six tensors of one expert in blob order: fused words, scales, biases, then down's.
pub const parts = 6;

/// What a model offers the store: its layers, the routed experts a layer holds in the host tier, and the bytes of each part of one expert.
pub const Shape = struct {
    layers: usize,
    /// Routed experts a layer; the shared expert is one more, always resident.
    experts: usize,
    /// Router picks a row (the routed ones plus the shared expert's): the trace holds this many ids a row.
    picks: usize,
    /// The most rows a round traces.
    trace_rows: usize,
    part_bytes: [parts]usize,

    pub fn blobBytes(s: Shape) usize {
        var n: usize = 0;
        for (s.part_bytes) |b| n += b;
        return n;
    }

    /// Where the down projection starts in a blob.
    pub fn downOffset(s: Shape) usize {
        return s.part_bytes[0] + s.part_bytes[1] + s.part_bytes[2];
    }

    fn check(s: Shape) Error!void {
        if (s.layers == 0 or s.experts == 0 or s.picks == 0 or s.trace_rows == 0) return error.BadShape;
        // every part starts where a 16-byte load can (the kernels load chunks of codes that wide)
        for (s.part_bytes) |b| if (b == 0 or b % 16 != 0) return error.BadShape;
        if (s.layers * s.experts >= policy.empty_id) return error.BadShape;
    }
};

/// What a layer's views need: its table, where its projections sit in a blob, its trace, and an address inside its host tier.
pub const Ref = struct { ptab: u64, fused_off: u64, down_off: u64, trace: u64, anchor: u64 };

pub const Stats = struct {
    rounds: u64 = 0,
    accesses: u64 = 0,
    hits: u64 = 0,
    copies_started: u64 = 0,
    copies_landed: u64 = 0,
    bytes_copied: u64 = 0,
    /// Distinct experts a round named, summed over rounds and layers.
    unique: u64 = 0,
};

/// Device memory is allocated in pieces of this many bytes at most: one physically contiguous gigabytes-long block may not exist.
const chunk_bytes: usize = 512 << 20;

const Flight = struct { event: Event, first: usize, count: usize };

/// The cache lines: device memory in chunks, the policy's bookkeeping and the copies in flight.
const Pool = struct {
    chunks: []DeviceBuffer,
    per_chunk: usize,
    vram: policy.Vram,
    slots: []policy.Id,
    line: []policy.Line,
    stamp: []u32,
    where: []i32,
    scratch_unique: []policy.Id,
    ids: []policy.Id,
    tally: []u16,
    decisions: []policy.Decision,
    copies: []policy.Copy,
    stream: Stream,
    /// Copies of the batches in flight, oldest first, and the batches.
    landing: std.ArrayList(policy.Copy),
    flights: std.ArrayList(Flight),
    spare: std.ArrayList(Event),
    copy_bytes: u64,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    r: *const Runtime,
    shape: Shape,
    /// Per layer: every routed expert's blob, pinned host memory mapped into the device.
    host: []HostBuffer,
    host_dev: []u64,
    /// The shared experts, one blob a layer.
    shared: DeviceBuffer,
    /// (layers, experts + 1) device addresses: the device's copy and the pinned words it is sent from.
    ptab_dev: DeviceBuffer,
    ptab_host: HostBuffer,
    /// The round's picks as the kernels write them: (layers, trace_rows, picks) int32, host memory the device writes.
    trace: HostBuffer,
    trace_dev: u64,
    pool: ?Pool = null,
    counts: []u32,
    now: u32 = 0,
    stats: Stats = .{},
    layers_added: usize = 0,

    pub fn init(gpa: std.mem.Allocator, r: *const Runtime, shape: Shape) Error!Store {
        try shape.check();
        const blob = shape.blobBytes();
        const host = try gpa.alloc(HostBuffer, shape.layers);
        errdefer gpa.free(host);
        const host_dev = try gpa.alloc(u64, shape.layers);
        errdefer gpa.free(host_dev);
        var made: usize = 0;
        errdefer for (host[0..made]) |*h| h.free();
        for (host, host_dev) |*h, *d| {
            h.* = try HostBuffer.allocMapped(r, shape.experts * blob);
            made += 1;
            d.* = try h.device();
        }
        var shared = try DeviceBuffer.alloc(r, shape.layers * blob);
        errdefer shared.free();
        const entries = shape.layers * (shape.experts + 1);
        var ptab_dev = try DeviceBuffer.alloc(r, entries * 8);
        errdefer ptab_dev.free();
        var ptab_host = try HostBuffer.allocMapped(r, entries * 8);
        errdefer ptab_host.free();
        var trace = try HostBuffer.allocMapped(r, shape.layers * shape.trace_rows * shape.picks * 4);
        errdefer trace.free();
        @memset(trace.bytes, 0);
        const counts = try gpa.alloc(u32, shape.layers * shape.experts);
        errdefer gpa.free(counts);
        @memset(counts, 0);
        const words = ptab_host.slice(u64);
        for (0..shape.layers) |l| {
            const row = words[l * (shape.experts + 1) ..][0 .. shape.experts + 1];
            for (row[0..shape.experts], 0..) |*w, e| w.* = host_dev[l] + e * blob;
            row[shape.experts] = shared.base() + l * blob;
        }
        return .{
            .gpa = gpa,
            .r = r,
            .shape = shape,
            .host = host,
            .host_dev = host_dev,
            .shared = shared,
            .ptab_dev = ptab_dev,
            .ptab_host = ptab_host,
            .trace = trace,
            .trace_dev = try trace.device(),
            .counts = counts,
        };
    }

    pub fn deinit(s: *Store) void {
        s.drain();
        if (s.pool) |*p| {
            for (p.flights.items) |*f| f.event.deinit();
            for (p.spare.items) |*e| e.deinit();
            p.flights.deinit(s.gpa);
            p.spare.deinit(s.gpa);
            p.landing.deinit(s.gpa);
            p.stream.deinit();
            for (p.chunks) |*c| c.free();
            s.gpa.free(p.chunks);
            s.gpa.free(p.slots);
            s.gpa.free(p.line);
            s.gpa.free(p.stamp);
            s.gpa.free(p.where);
            s.gpa.free(p.scratch_unique);
            s.gpa.free(p.ids);
            s.gpa.free(p.tally);
            s.gpa.free(p.decisions);
            s.gpa.free(p.copies);
        }
        s.gpa.free(s.counts);
        s.trace.free();
        s.ptab_host.free();
        s.ptab_dev.free();
        s.shared.free();
        for (s.host) |*h| h.free();
        s.gpa.free(s.host_dev);
        s.gpa.free(s.host);
        s.* = undefined;
    }

    /// Waits for the copies in flight (before their sources or lines are freed).
    pub fn drain(s: *Store) void {
        if (s.pool) |*p| p.stream.synchronize() catch {};
    }

    pub fn blobBytes(s: *const Store) usize {
        return s.shape.blobBytes();
    }

    /// Host bytes of the tier and device bytes of what is always resident (the shared experts and the table).
    pub fn hostBytes(shape: Shape) usize {
        return shape.layers * shape.experts * shape.blobBytes();
    }

    pub fn residentBytes(shape: Shape) usize {
        return shape.layers * shape.blobBytes() + shape.layers * (shape.experts + 1) * 8;
    }

    /// Layer `layer`'s experts as a stack: `tensors` are the six parts, each (experts + 1, ...) with the shared expert last. The routed ones go
    /// to the host tier, the shared one to its line.
    pub fn addLayer(s: *Store, layer: usize, tensors: [parts][]const u8) Error!void {
        if (layer >= s.shape.layers) return error.BadShape;
        const e = s.shape.experts;
        const blob = s.shape.blobBytes();
        for (tensors, s.shape.part_bytes) |t, b| if (t.len != b * (e + 1)) return error.BadShape;
        const dst = s.host[layer].bytes;
        for (0..e) |x| {
            var at = x * blob;
            for (tensors, s.shape.part_bytes) |t, b| {
                @memcpy(dst[at..][0..b], t[x * b ..][0..b]);
                at += b;
            }
        }
        const one = try s.gpa.alloc(u8, blob);
        defer s.gpa.free(one);
        var at: usize = 0;
        for (tensors, s.shape.part_bytes) |t, b| {
            @memcpy(one[at..][0..b], t[e * b ..][0..b]);
            at += b;
        }
        try s.shared.upload(layer * blob, one);
        s.layers_added += 1;
    }

    /// The addresses layer `layer`'s views carry.
    pub fn ref(s: *const Store, layer: usize) Ref {
        const entries = s.shape.experts + 1;
        const per_layer = s.shape.trace_rows * s.shape.picks * 4;
        return .{
            .ptab = s.ptab_dev.base() + layer * entries * 8,
            .fused_off = 0,
            .down_off = s.shape.downOffset(),
            .trace = s.trace_dev + layer * per_layer,
            .anchor = s.host_dev[layer],
        };
    }

    fn entry(s: *Store, key: policy.Id) *u64 {
        const e = s.shape.experts;
        const l = key / e;
        const x = key % e;
        return &s.ptab_host.slice(u64)[l * (e + 1) + x];
    }

    fn hostAddr(s: *const Store, key: policy.Id) u64 {
        const e = s.shape.experts;
        return s.host_dev[key / e] + (key % e) * s.shape.blobBytes();
    }

    fn lineAddr(s: *const Store, p: *const Pool, slot: usize) u64 {
        return p.chunks[slot / p.per_chunk].base() + (slot % p.per_chunk) * s.shape.blobBytes();
    }

    /// Cache lines `lines` of device memory (as many as `bytes` hold when 0), copying at most `copy_experts` experts a round.
    pub fn attachPool(s: *Store, lines: usize, copy_experts: usize) Error!void {
        std.debug.assert(s.pool == null);
        const blob = s.shape.blobBytes();
        const keys = s.shape.layers * s.shape.experts;
        const per_chunk = @max(1, chunk_bytes / blob);
        const n_chunks = (lines + per_chunk - 1) / per_chunk;
        const chunks = try s.gpa.alloc(DeviceBuffer, n_chunks);
        errdefer s.gpa.free(chunks);
        var made: usize = 0;
        errdefer for (chunks[0..made]) |*c| c.free();
        for (chunks, 0..) |*c, i| {
            c.* = try DeviceBuffer.alloc(s.r, @min(per_chunk, lines - i * per_chunk) * blob);
            made += 1;
        }
        const slots = try s.gpa.alloc(policy.Id, lines);
        errdefer s.gpa.free(slots);
        const line = try s.gpa.alloc(policy.Line, lines);
        errdefer s.gpa.free(line);
        const stamp = try s.gpa.alloc(u32, lines);
        errdefer s.gpa.free(stamp);
        const where = try s.gpa.alloc(i32, keys);
        errdefer s.gpa.free(where);
        const picks_max = s.shape.layers * s.shape.trace_rows * (s.shape.picks - 1);
        const unique = try s.gpa.alloc(policy.Id, picks_max);
        errdefer s.gpa.free(unique);
        const tally = try s.gpa.alloc(u16, keys);
        errdefer s.gpa.free(tally);
        @memset(tally, 0);
        const decisions = try s.gpa.alloc(policy.Decision, picks_max);
        errdefer s.gpa.free(decisions);
        const ids = try s.gpa.alloc(policy.Id, picks_max);
        errdefer s.gpa.free(ids);
        const copies = try s.gpa.alloc(policy.Copy, @max(lines, 1));
        errdefer s.gpa.free(copies);
        var stream = try Stream.init(s.r);
        errdefer stream.deinit();
        s.pool = .{
            .chunks = chunks,
            .per_chunk = per_chunk,
            .vram = policy.Vram.init(slots, line, stamp, where),
            .slots = slots,
            .line = line,
            .stamp = stamp,
            .where = where,
            .scratch_unique = unique,
            .ids = ids,
            .tally = tally,
            .decisions = decisions,
            .copies = copies,
            .stream = stream,
            .landing = .empty,
            .flights = .empty,
            .spare = .empty,
            .copy_bytes = @as(u64, copy_experts) * blob,
        };
    }

    pub fn lineCount(s: *const Store) usize {
        return if (s.pool) |p| p.slots.len else 0;
    }

    /// The table up to the device, on `stream`: first in a round's graph and before a prompt span, so no forward reads entries an eviction has made stale.
    pub fn send(s: *Store, stream: abi.Stream) Error!void {
        try raw.upload(s.ptab_dev, 0, s.ptab_host.bytes, stream);
    }

    /// Experts that have landed become resident: their entries name the line from the next `send`.
    pub fn beginRound(s: *Store) Error!void {
        const p = if (s.pool) |*p| p else return;
        while (p.flights.items.len > 0) {
            const f = p.flights.items[0];
            if (!try f.event.done()) break;
            for (p.landing.items[f.first..][0..f.count]) |c| {
                // a line whose expert was staged again before it landed is not this copy's
                policy.reap(&p.vram, c.id, s.now) catch continue;
                s.entry(c.id).* = s.lineAddr(p, c.slot);
                s.stats.copies_landed += 1;
            }
            try p.spare.append(s.gpa, f.event);
            _ = p.flights.orderedRemove(0);
            if (p.flights.items.len == 0) p.landing.clearRetainingCapacity() else {
                // the landing list is a queue: drop what the front batch held
                const drop = f.count;
                std.mem.copyForwards(policy.Copy, p.landing.items[0 .. p.landing.items.len - drop], p.landing.items[drop..]);
                p.landing.shrinkRetainingCapacity(p.landing.items.len - drop);
                for (p.flights.items) |*g| g.first -= drop;
            }
        }
    }

    /// The round over its first `rows` rows is finished: its picks decide what the next round's copies move.
    pub fn endRound(s: *Store, rows: usize) Error!void {
        const p = if (s.pool) |*p| p else return;
        if (rows == 0 or rows > s.shape.trace_rows) return error.BadShape;
        s.now +%= 1;
        s.stats.rounds += 1;
        const e = s.shape.experts;
        const per_layer = s.shape.trace_rows * s.shape.picks;
        const words = s.trace.slice(i32);
        var n: usize = 0;
        for (0..s.shape.layers) |l| {
            const row = words[l * per_layer ..][0 .. rows * s.shape.picks];
            for (0..rows) |r| for (row[r * s.shape.picks ..][0 .. s.shape.picks - 1]) |id| {
                if (id < 0 or id >= e) continue;
                p.ids[n] = @intCast(l * e + @as(usize, @intCast(id)));
                n += 1;
            };
        }
        const ids = p.ids[0..n];
        policy.observe(s.counts, ids) catch unreachable;
        const decisions = p.decisions[0..n];
        const started = policy.schedule(&p.vram, ids, .{ .slots = @intCast(p.slots.len), .expert_bytes = s.shape.blobBytes(), .copy_bytes = p.copy_bytes }, s.now, decisions, .{ .unique = p.scratch_unique, .tally = p.tally }, p.copies) catch return error.BadShape;
        for (decisions) |d| {
            s.stats.accesses += 1;
            if (d == .hit) s.stats.hits += 1;
        }
        try s.launch(p, started);
    }

    /// Queues the copies `schedule` or a seed started: each eviction's entry names the host tier first, then the bytes move.
    fn launch(s: *Store, p: *Pool, started: usize) Error!void {
        if (started == 0) return;
        const blob = s.shape.blobBytes();
        const first = p.landing.items.len;
        for (p.copies[0..started]) |c| {
            if (c.evicted) |old| s.entry(old).* = s.hostAddr(old);
            // the expert itself stays on the host tier until its copy lands
            const from = s.hostViewOf(c.id);
            try p.chunks[c.slot / p.per_chunk].uploadAsync((c.slot % p.per_chunk) * blob, from, p.stream);
            try p.landing.append(s.gpa, c);
        }
        var event = if (p.spare.pop()) |ev| ev else try Event.init(s.r, false);
        errdefer event.deinit();
        try event.record(p.stream);
        try p.flights.append(s.gpa, .{ .event = event, .first = first, .count = started });
        s.stats.copies_started += started;
        s.stats.bytes_copied += started * blob;
    }

    fn hostViewOf(s: *const Store, key: policy.Id) HostBuffer {
        const e = s.shape.experts;
        const blob = s.shape.blobBytes();
        return s.host[key / e].view((key % e) * blob, (key % e + 1) * blob);
    }

    /// Fills the empty lines with the hottest experts of `counts` (a cell per (layer, expert)): the working set a prompt showed.
    pub fn seed(s: *Store, counts: []const u32) Error!usize {
        const p = if (s.pool) |*p| p else return 0;
        if (counts.len != s.counts.len) return error.BadShape;
        const order = try s.gpa.alloc(policy.Id, counts.len);
        defer s.gpa.free(order);
        var total: usize = 0;
        while (!p.vram.full()) {
            const started = policy.seedHottest(&p.vram, counts, order, p.copies) catch return error.BadShape;
            if (started == 0) break;
            try s.launch(p, started);
            total += started;
        }
        return total;
    }

    /// Fills every empty line with experts in key order, so a card is not left idle while the counts warm up.
    pub fn fill(s: *Store) Error!usize {
        const p = if (s.pool) |*p| p else return 0;
        var total: usize = 0;
        var key: policy.Id = 0;
        const keys: policy.Id = @intCast(s.counts.len);
        while (key < keys and !p.vram.full()) {
            var started: usize = 0;
            while (key < keys and !p.vram.full() and started < p.copies.len) : (key += 1) {
                if (p.vram.where[key] >= 0) continue;
                p.copies[started] = policy.stage(&p.vram, key) catch break;
                started += 1;
            }
            try s.launch(p, started);
            total += started;
        }
        return total;
    }

    /// Hit rate of the accesses so far and the misses a round, for a log line.
    pub fn summary(s: *const Store) struct { hit_rate: f64, misses_per_round: f64 } {
        const a: f64 = @floatFromInt(@max(s.stats.accesses, 1));
        const r: f64 = @floatFromInt(@max(s.stats.rounds, 1));
        return .{
            .hit_rate = @as(f64, @floatFromInt(s.stats.hits)) / a,
            .misses_per_round = @as(f64, @floatFromInt(s.stats.accesses - s.stats.hits)) / r,
        };
    }
};

fn testTensors(gpa: std.mem.Allocator, shape: Shape, seed: u8) ![parts][]u8 {
    var out: [parts][]u8 = undefined;
    for (&out, shape.part_bytes, 0..) |*t, b, i| {
        t.* = try gpa.alloc(u8, b * (shape.experts + 1));
        for (t.*, 0..) |*x, j| x.* = @truncate(j *% 31 +% i *% 7 +% seed);
    }
    return out;
}

fn blobOf(shape: Shape, tensors: [parts][]u8, expert: usize, out: []u8) void {
    var at: usize = 0;
    for (tensors, shape.part_bytes) |t, b| {
        @memcpy(out[at..][0..b], t[expert * b ..][0..b]);
        at += b;
    }
}

fn setPicks(s: *Store, layer: usize, row: usize, ids: []const i32) void {
    const at = (layer * s.shape.trace_rows + row) * s.shape.picks;
    @memcpy(s.trace.slice(i32)[at..][0..ids.len], ids);
}

test "the host tier holds every expert's blob, the shared one is resident, and a landed copy flips the entry" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 2, .experts = 4, .picks = 3, .trace_rows = 2, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    const blob = shape.blobBytes();
    try std.testing.expectEqual(@as(usize, 160), blob);
    try std.testing.expectEqual(@as(usize, 96), shape.downOffset());
    var s = try Store.init(gpa, &r, shape);
    defer s.deinit();
    const t0 = try testTensors(gpa, shape, 1);
    const t1 = try testTensors(gpa, shape, 2);
    defer for (t0 ++ t1) |t| gpa.free(t);
    try s.addLayer(0, t0);
    try s.addLayer(1, t1);
    // an expert's blob in the host tier is its six parts back to back; the shared one is in its line
    var want: [160]u8 = undefined;
    blobOf(shape, t1, 2, &want);
    try std.testing.expectEqualSlices(u8, &want, s.host[1].bytes[2 * blob ..][0..blob]);
    blobOf(shape, t1, 4, &want);
    var got: [160]u8 = undefined;
    try s.shared.download(1 * blob, &got);
    try std.testing.expectEqualSlices(u8, &want, &got);
    const words = s.ptab_host.slice(u64);
    try std.testing.expectEqual(s.host_dev[1] + 3 * blob, words[1 * 5 + 3]);
    try std.testing.expectEqual(s.shared.base() + blob, words[1 * 5 + 4]);
    try std.testing.expectError(error.BadShape, s.addLayer(2, t0));

    try s.attachPool(3, 2);
    var stream = try Stream.init(&r);
    defer stream.deinit();
    // round 1: layer 0 names 1 twice, 2 and 3; layer 1 names 0. Two copies fit the budget: the hottest (key 1) and the lowest id of the rest
    setPicks(&s, 0, 0, &.{ 1, 2, 4 });
    setPicks(&s, 0, 1, &.{ 1, 3, 4 });
    setPicks(&s, 1, 0, &.{ 0, 0, 4 });
    setPicks(&s, 1, 1, &.{ 0, 0, 4 });
    try s.endRound(2);
    try std.testing.expectEqual(@as(u64, 2), s.stats.copies_started);
    try std.testing.expectEqual(@as(u64, 0), s.stats.hits);
    // nothing is resident until the copy lands and the next round begins: the entry still names the host tier
    try std.testing.expectEqual(s.host_dev[0] + 1 * blob, s.ptab_host.slice(u64)[0 * 5 + 1]);
    s.drain();
    try s.beginRound();
    try std.testing.expectEqual(@as(u64, 2), s.stats.copies_landed);
    const line = s.ptab_host.slice(u64)[0 * 5 + 1];
    try std.testing.expect(line != s.host_dev[0] + 1 * blob);
    blobOf(shape, t0, 1, &want);
    var dev: [160]u8 = undefined;
    try s.pool.?.chunks[0].download(line - s.pool.?.chunks[0].base(), &dev);
    try std.testing.expectEqualSlices(u8, &want, &dev);
    try s.send(stream.handle);
    try stream.synchronize();
    var table: [10]u64 = undefined;
    try s.ptab_dev.download(0, std.mem.sliceAsBytes(table[0..5]));
    try std.testing.expectEqual(line, table[1]);
    // round 2: the same picks are hits now for the resident ones, and a new expert evicts the older line
    try s.endRound(2);
    try std.testing.expect(s.stats.hits >= 2);
    const sum = s.summary();
    try std.testing.expect(sum.hit_rate > 0 and sum.hit_rate < 1);
    s.drain();
    try s.beginRound();
    try std.testing.expectEqual(s.pool.?.slots.len, @as(usize, 3));
    // round 3: a full pool names its victim's host address again before the copy runs; the touched residents stay
    try std.testing.expect(policy.onDevice(s.pool.?.vram, 2));
    setPicks(&s, 0, 0, &.{ 3, 3, 4 });
    setPicks(&s, 0, 1, &.{ 3, 3, 4 });
    setPicks(&s, 1, 0, &.{ 0, 0, 4 });
    setPicks(&s, 1, 1, &.{ 0, 0, 4 });
    try s.endRound(2);
    try std.testing.expectEqual(s.host_dev[0] + 2 * blob, s.ptab_host.slice(u64)[2]);
    try std.testing.expect(policy.onDevice(s.pool.?.vram, 4));
    try std.testing.expect(!policy.onDevice(s.pool.?.vram, 2));
    s.drain();
    try s.beginRound();
    try std.testing.expect(policy.onDevice(s.pool.?.vram, 3));
    try std.testing.expect(s.ptab_host.slice(u64)[3] != s.host_dev[0] + 3 * blob);
}
