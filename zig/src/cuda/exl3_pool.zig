//! Device lines for one layer's EXL3 routed-expert offload on upstream's `exl3_experts.Experts`.

const std = @import("std");
const abi = @import("abi.zig");
const memory = @import("memory.zig");
const stream_ = @import("stream.zig");
pub const dma = @import("exl3_dma.zig");

const Driver = @import("driver.zig").Driver;
const Error = @import("driver.zig").Error;
const DeviceBuffer = memory.DeviceBuffer;
const HostBuffer = memory.HostBuffer;
const Stream = stream_.Stream;
const Event = stream_.Event;

fn align256(n: usize) usize {
    return (n + 255) / 256 * 256;
}

pub const Spec = struct {
    /// Table length E (513: 512 routed experts and the shared expert as id 512).
    experts: usize,
    /// Largest trellis bytes of any expert, per projection (gate, up, down): a line fits every expert of the layer.
    bytes: [3]usize,
    lines: usize,
    /// Pinned staging lines (a ring); 0 when every copy is `direct`.
    staging: usize = 8,
    /// Keep the table entries in a host shadow and upload the whole table once per stage (see `Lines.flush`).
    batch: bool = false,
};

/// Byte offsets of gate, up and down inside a line, each 256-aligned.
pub const Layout = struct {
    off: [3]usize,
    line_bytes: usize,

    pub fn init(bytes: [3]usize) Layout {
        var l: Layout = .{ .off = undefined, .line_bytes = 0 };
        for (bytes, &l.off) |b, *o| {
            o.* = l.line_bytes;
            l.line_bytes += align256(b);
        }
        return l;
    }
};

/// Where the six tables sit inside one allocation: three int64 [E] pointer tables, then three int32 [E] width tables.
pub const Tables = struct {
    ptr: [3]usize,
    k2: [3]usize,
    bytes: usize,

    pub fn init(experts: usize) Tables {
        var t: Tables = .{ .ptr = undefined, .k2 = undefined, .bytes = 0 };
        for (&t.ptr) |*o| {
            o.* = t.bytes;
            t.bytes += align256(experts * 8);
        }
        for (&t.k2) |*o| {
            o.* = t.bytes;
            t.bytes += align256(experts * 4);
        }
        return t;
    }
};

pub const Stats = struct {
    copies: u64 = 0,
    bytes: u64 = 0,
    staged_bytes: u64 = 0, // bytes the host memcpy'd into pinned staging
    stage_ns: u64 = 0, // host time in those memcpys
    staging_waits: u64 = 0, // a staging slot was still being copied from (the host blocked)
    release_waits: u64 = 0, // a copy waited on the device for a pass that read its line
    gate_waits: u64 = 0, // a pass waited on the device for a line copy
};

const release_ring = 64;
const up_ring = 8;

pub const Lines = struct {
    d: *const Driver,
    spec: Spec,
    layout: Layout,
    tlay: Tables,
    mem: DeviceBuffer,
    tables: DeviceBuffer,
    shared: ?DeviceBuffer = null,
    pinned_bank: ?[]const u8 = null,
    staging: ?HostBuffer,
    stage_ev: []Event,
    stage_busy: []bool,
    stage_next: usize = 0,
    copy: Stream,
    line_ev: []Event,
    line_pending: []bool,
    /// Epoch of the last compute pass that read the line (0: never read).
    line_epoch: []u64,
    rel_ev: [release_ring]Event,
    epoch: u64 = 0, // last epoch released
    done_epoch: u64 = 0, // highest epoch known finished
    stats: Stats = .{},
    gpa: std.mem.Allocator,
    /// `Spec.batch`: entries go to a host shadow and the whole table uploads once per stage, not 18 memsets per expert.
    batch: bool = false,
    deferred: bool = false,
    shadow: []u8 = &.{},
    shadow_dirty: bool = false,
    up_slots: ?HostBuffer = null,
    up_ev: [up_ring]Event = undefined,
    up_made: bool = false,
    up_next: usize = 0,
    up_used: [up_ring]bool = @splat(false),
    pend: []u32 = &.{},
    n_pend: usize = 0,

    pub fn init(gpa: std.mem.Allocator, d: *const Driver, spec: Spec) !Lines {
        if (spec.lines == 0 or spec.experts == 0) return error.Invalid;
        const layout = Layout.init(spec.bytes);
        const tlay = Tables.init(spec.experts);
        var mem = try DeviceBuffer.alloc(d, spec.lines * layout.line_bytes);
        errdefer mem.free();
        var tables = try DeviceBuffer.alloc(d, tlay.bytes);
        errdefer tables.free();
        try tables.fill8(0, null); // every expert absent; synchronous, before any stream uses the tables
        var staging: ?HostBuffer = if (spec.staging > 0) try HostBuffer.alloc(d, spec.staging * layout.line_bytes) else null;
        errdefer if (staging) |*h| h.free();
        var copy = try Stream.init(d, true);
        errdefer copy.deinit();
        const stage_ev = try makeEvents(gpa, d, spec.staging);
        errdefer freeEvents(gpa, stage_ev);
        const stage_busy = try gpa.alloc(bool, spec.staging);
        errdefer gpa.free(stage_busy);
        @memset(stage_busy, false);
        const line_ev = try makeEvents(gpa, d, spec.lines);
        errdefer freeEvents(gpa, line_ev);
        const line_pending = try gpa.alloc(bool, spec.lines);
        errdefer gpa.free(line_pending);
        @memset(line_pending, false);
        const line_epoch = try gpa.alloc(u64, spec.lines);
        errdefer gpa.free(line_epoch);
        @memset(line_epoch, 0);
        var rel: [release_ring]Event = undefined;
        var made: usize = 0;
        errdefer for (rel[0..made]) |*e| e.deinit();
        for (&rel) |*e| {
            e.* = try Event.init(d, false);
            made += 1;
        }
        var out: Lines = .{
            .d = d,
            .spec = spec,
            .layout = layout,
            .tlay = tlay,
            .mem = mem,
            .tables = tables,
            .staging = staging,
            .stage_ev = stage_ev,
            .stage_busy = stage_busy,
            .copy = copy,
            .line_ev = line_ev,
            .line_pending = line_pending,
            .line_epoch = line_epoch,
            .rel_ev = rel,
            .gpa = gpa,
        };
        if (spec.batch) {
            out.batch = true;
            out.shadow = try gpa.alloc(u8, tlay.bytes);
            errdefer gpa.free(out.shadow);
            @memset(out.shadow, 0);
            out.pend = try gpa.alloc(u32, spec.lines);
            errdefer gpa.free(out.pend);
            out.up_slots = try HostBuffer.alloc(d, up_ring * tlay.bytes);
            errdefer if (out.up_slots) |*h| h.free();
            for (&out.up_ev) |*e| e.* = try Event.init(d, false);
            out.up_made = true;
        }
        return out;
    }

    fn makeEvents(gpa: std.mem.Allocator, d: *const Driver, n: usize) ![]Event {
        const evs = try gpa.alloc(Event, n);
        var made: usize = 0;
        errdefer {
            for (evs[0..made]) |*e| e.deinit();
            gpa.free(evs);
        }
        for (evs) |*e| {
            e.* = try Event.init(d, false);
            made += 1;
        }
        return evs;
    }

    fn freeEvents(gpa: std.mem.Allocator, evs: []Event) void {
        for (evs) |*e| e.deinit();
        gpa.free(evs);
    }

    pub fn deinit(l: *Lines) void {
        l.copy.synchronize() catch {};
        if (l.up_made) {
            for (&l.up_ev) |*e| e.deinit();
            if (l.up_slots) |*h| h.free();
            l.gpa.free(l.shadow);
            l.gpa.free(l.pend);
        }
        for (&l.rel_ev) |*e| e.deinit();
        freeEvents(l.gpa, l.line_ev);
        freeEvents(l.gpa, l.stage_ev);
        l.gpa.free(l.stage_busy);
        l.gpa.free(l.line_pending);
        l.gpa.free(l.line_epoch);
        l.copy.deinit();
        if (l.staging) |*h| h.free();
        if (l.shared) |*b| b.free();
        l.tables.free();
        l.mem.free();
        l.* = undefined;
    }

    /// Device bytes: the lines, the tables and the resident shared expert.
    pub fn deviceBytes(l: Lines) usize {
        return l.mem.len + l.tables.len + if (l.shared) |b| b.len else 0;
    }

    pub fn table(l: Lines, which: usize) u64 {
        return l.tables.ptr + l.tlay.ptr[which];
    }

    pub fn widths(l: Lines, which: usize) u64 {
        return l.tables.ptr + l.tlay.k2[which];
    }

    /// Device address of projection `which` (0 gate, 1 up, 2 down) of line `line`.
    pub fn at(l: Lines, line: usize, which: usize) u64 {
        return l.mem.ptr + line * l.layout.line_bytes + l.layout.off[which];
    }

    /// Uploads the never-offloaded expert `id` (the shared expert) once into its own region (it may be wider than a
    pub fn setResident(l: *Lines, id: usize, src: [3][]const u8, k2: [3]i32) !void {
        if (id >= l.spec.experts or l.shared != null) return error.Invalid;
        const lay = Layout.init(.{ src[0].len, src[1].len, src[2].len });
        var b = try DeviceBuffer.alloc(l.d, lay.line_bytes);
        errdefer b.free();
        for (src, lay.off) |s, o| try b.upload(o, s);
        l.shared = b;
        for (0..3) |w| try l.writeEntry(id, b.ptr + lay.off[w], k2[w], w);
        try l.copy.synchronize();
    }

    fn memset32(l: *Lines, addr: u64, value: u32, st: Stream) !void {
        try l.d.check(l.d.api.cuMemsetD32Async(addr, value, 1, st.handle), "cuMemsetD32Async");
    }

    /// One expert's projection `which`: its pointer (two 32-bit words) and width, on the copy stream.
    fn writeEntry(l: *Lines, id: usize, ptr: u64, k2: i32, which: usize) !void {
        const st = l.copy;
        if (l.batch) {
            std.mem.writeInt(u64, l.shadow[l.tlay.ptr[which] + id * 8 ..][0..8], ptr, .little);
            std.mem.writeInt(i32, l.shadow[l.tlay.k2[which] + id * 4 ..][0..4], k2, .little);
            if (l.deferred) {
                l.shadow_dirty = true;
                return;
            }
        }
        const p = l.table(which) + id * 8;
        try l.memset32(p, @truncate(ptr), st);
        try l.memset32(p + 4, @truncate(ptr >> 32), st);
        try l.memset32(l.widths(which) + id * 4, @bitCast(k2), st);
    }

    /// The copy stream waits until the last compute pass that read `line` has finished (no wait if known done).
    fn afterReaders(l: *Lines, line: usize) !void {
        const e = l.line_epoch[line];
        if (e == 0 or e <= l.done_epoch) return;
        const slot = e % release_ring;
        const ev = l.rel_ev[slot];
        if (try ev.done()) {
            // the slot holds epoch e or a newer one; either way e has finished
            l.done_epoch = @max(l.done_epoch, e);
            return;
        }
        try l.copy.wait(ev); // a newer epoch in the slot is a conservative wait
        l.stats.release_waits += 1;
    }

    /// Makes `id` resident in `line`: the copy waits for the line's readers, then H2D (staged or direct), then `id`'s
    pub fn load(l: *Lines, line: usize, id: usize, src: [3][]const u8, k2: [3]i32, evicted: ?usize, direct: bool, io: std.Io) !void {
        if (line >= l.spec.lines or id >= l.spec.experts) return error.Invalid;
        for (src, l.spec.bytes) |s, cap| if (s.len > cap) return error.Invalid;
        try l.afterReaders(line);
        const dst = l.mem.ptr + line * l.layout.line_bytes;
        var moved: usize = 0;
        if (direct or l.staging == null) {
            moved = try dma.directLines(l.copy, dst, l.layout.line_bytes, l.layout.off, src, l.pinned_bank);
        } else {
            const slot = l.stage_next;
            l.stage_next = (l.stage_next + 1) % l.spec.staging;
            if (l.stage_busy[slot]) {
                if (!try l.stage_ev[slot].done()) l.stats.staging_waits += 1;
                try l.stage_ev[slot].synchronize();
            }
            const buf = l.staging.?.bytes[slot * l.layout.line_bytes ..][0..l.layout.line_bytes];
            const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
            for (src, l.layout.off) |s, o| @memcpy(buf[o..][0..s.len], s);
            l.stats.stage_ns += @intCast(std.Io.Clock.awake.now(io).toNanoseconds() - t0);
            // one copy for the line: up to the end of down's bytes (alignment padding rides along)
            const n = l.layout.off[2] + src[2].len;
            try l.d.check(l.d.api.cuMemcpyHtoDAsync_v2(dst, buf.ptr, n, l.copy.handle), "cuMemcpyHtoDAsync");
            try l.stage_ev[slot].record(l.copy);
            l.stage_busy[slot] = true;
            l.stats.staged_bytes += src[0].len + src[1].len + src[2].len;
            moved = n;
        }
        l.deferred = l.batch;
        defer l.deferred = false;
        if (evicted) |old| for (0..3) |w| try l.writeEntry(old, 0, 0, w);
        for (0..3) |w| try l.writeEntry(id, dst + l.layout.off[w], k2[w], w);
        if (l.batch) {
            l.pend[l.n_pend] = @intCast(line);
            l.n_pend += 1;
        } else try l.line_ev[line].record(l.copy);
        l.line_pending[line] = true;
        l.stats.copies += 1;
        l.stats.bytes += moved;
    }

    /// Batch mode: one copy of the whole table after a stage's loads, then the loaded lines' events; no-op otherwise.
    pub fn flush(l: *Lines) !void {
        if (!l.batch) return;
        if (l.shadow_dirty) {
            const slot = l.up_next;
            l.up_next = (l.up_next + 1) % up_ring;
            if (l.up_used[slot]) try l.up_ev[slot].synchronize();
            const buf = l.up_slots.?.bytes[slot * l.tlay.bytes ..][0..l.tlay.bytes];
            @memcpy(buf, l.shadow);
            try l.d.check(l.d.api.cuMemcpyHtoDAsync_v2(l.tables.ptr, buf.ptr, l.tlay.bytes, l.copy.handle), "cuMemcpyHtoDAsync");
            try l.up_ev[slot].record(l.copy);
            l.up_used[slot] = true;
            l.shadow_dirty = false;
        }
        for (l.pend[0..l.n_pend]) |line| try l.line_ev[line].record(l.copy);
        l.n_pend = 0;
    }

    /// True once the line's last copy finished (or it never had one); clears the pending flag.
    pub fn landed(l: *Lines, line: usize) !bool {
        if (!l.line_pending[line]) return true;
        if (!try l.line_ev[line].done()) return false;
        l.line_pending[line] = false;
        return true;
    }

    /// `compute` waits on the device for the line's copy, if one may still be running. The host does not block.
    pub fn gate(l: *Lines, compute: Stream, line: usize) !void {
        if (try l.landed(line)) return;
        try compute.wait(l.line_ev[line]);
        l.stats.gate_waits += 1;
    }

    /// After a compute pass that read `lines`: a release event on `compute`; a later copy into any of those lines
    pub fn release(l: *Lines, compute: Stream, lines: []const u32) !void {
        l.epoch += 1;
        const slot = l.epoch % release_ring;
        try l.rel_ev[slot].record(compute);
        for (lines) |i| l.line_epoch[i] = l.epoch;
    }

    /// Blocks until every queued copy and table write finished.
    pub fn drain(l: *Lines) !void {
        try l.copy.synchronize();
        for (l.line_pending) |*p| p.* = false;
        @memset(l.stage_busy, false);
    }
};

test {
    _ = dma;
}

test "a line packs gate, up and down 256-aligned; the tables sit in one allocation" {
    // Flash Next layer 0: 3-bit experts, K=2560 N=640 -> 160 x 40 tiles of 48 words
    const tb = 160 * 40 * 48 * 2;
    const l = Layout.init(.{ tb, tb, tb });
    try std.testing.expectEqual([3]usize{ 0, tb, 2 * tb }, l.off);
    try std.testing.expectEqual(@as(usize, 3 * tb), l.line_bytes);
    const odd = Layout.init(.{ 100, 300, 5 });
    try std.testing.expectEqual([3]usize{ 0, 256, 768 }, odd.off);
    try std.testing.expectEqual(@as(usize, 1024), odd.line_bytes);
    const t = Tables.init(513);
    try std.testing.expectEqual([3]usize{ 0, 4352, 8704 }, t.ptr);
    try std.testing.expectEqual(@as(usize, 13056), t.k2[0]);
    try std.testing.expectEqual(@as(usize, 13056 + 3 * 2304), t.bytes);
    for (t.ptr ++ t.k2) |o| try std.testing.expectEqual(@as(usize, 0), o % 256);
}
