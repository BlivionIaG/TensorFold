//! A layer's stage of the expert offload on the host: the picks of a forward decide where the device reads experts.
const std = @import("std");
const policy = @import("expert_offload.zig");
const window = @import("expert_window.zig");
const stats_mod = @import("expert_stats.zig");

pub const Shape = window.Shape;
pub const Window = window.Window;
pub const parts = window.parts;
pub const Stats = stats_mod.Stats;

pub const Error = error{ BadShape, NotSized, StageFailed, OutOfMemory };

/// A layer's stage: its `rows` rows of picks are in the trace; `prefill` marks a prompt span, which admits nothing.
pub const Job = struct { layer: u32, rows: u32, prefill: bool };

/// What a stage asks of the device: where a cache line is, and the scratch that prompt-span experts are copied into.
pub const Ports = struct {
    ctx: *anyopaque,
    line_addr: *const fn (ctx: *anyopaque, slot: usize) u64,
    scratch_base: u64,
    scratch_copy: *const fn (ctx: *anyopaque, at: usize, window_off: usize, len: usize) error{CopyFailed}!void,
    scratch_wait: *const fn (ctx: *anyopaque) error{CopyFailed}!void,
};

/// The cache lines' bookkeeping and the policy's working sets; the lines' memory belongs to the device.
pub const Pool = struct {
    vram: policy.Vram,
    slots: []policy.Id,
    line: []policy.Line,
    stamp: []u32,
    flags: []bool,
    where: []i32,
    scratch_unique: []policy.Id,
    ids: []policy.Id,
    tally: []u16,
    decisions: []policy.Decision,
    copies: []policy.Copy,
    /// Admissions the round's remaining stages may still make.
    budget: usize,
    copy_experts: usize,

    /// `lines` cache lines; `protect_pct` percent of them an SLRU's protected segment (0: plain LRU).
    pub fn init(gpa: std.mem.Allocator, shape: Shape, lines: usize, copy_experts: usize, protect_pct: u8) Error!Pool {
        const keys = shape.layers * shape.experts;
        const picks_max = shape.trace_rows * (shape.picks - 1);
        const slots = try gpa.alloc(policy.Id, lines);
        errdefer gpa.free(slots);
        const line = try gpa.alloc(policy.Line, lines);
        errdefer gpa.free(line);
        const stamp = try gpa.alloc(u32, lines);
        errdefer gpa.free(stamp);
        const flags = try gpa.alloc(bool, lines);
        errdefer gpa.free(flags);
        const where = try gpa.alloc(i32, keys);
        errdefer gpa.free(where);
        const unique = try gpa.alloc(policy.Id, picks_max);
        errdefer gpa.free(unique);
        const tally = try gpa.alloc(u16, keys);
        errdefer gpa.free(tally);
        @memset(tally, 0);
        const decisions = try gpa.alloc(policy.Decision, picks_max);
        errdefer gpa.free(decisions);
        const ids = try gpa.alloc(policy.Id, picks_max);
        errdefer gpa.free(ids);
        const copies = try gpa.alloc(policy.Copy, @min(@max(copy_experts, 1), picks_max));
        errdefer gpa.free(copies);
        var vram = policy.Vram.init(slots, line, stamp, where);
        if (protect_pct > 0) vram.enableSegments(flags, lines * @as(usize, protect_pct) / 100);
        return .{
            .vram = vram,
            .slots = slots,
            .line = line,
            .stamp = stamp,
            .flags = flags,
            .where = where,
            .scratch_unique = unique,
            .ids = ids,
            .tally = tally,
            .decisions = decisions,
            .copies = copies,
            .budget = copy_experts,
            .copy_experts = copy_experts,
        };
    }

    pub fn deinit(p: *Pool, gpa: std.mem.Allocator) void {
        inline for (.{ p.slots, p.line, p.stamp, p.flags, p.where, p.scratch_unique, p.ids, p.tally, p.decisions, p.copies }) |s| gpa.free(s);
    }
};

pub const Tier = struct {
    gpa: std.mem.Allocator,
    shape: Shape,
    win: Window,
    /// Per layer: every routed expert's blob in pageable host memory (null until the layer is added).
    host: []?[]u8,
    /// Slot states: 0 free, 1 holds a layer's expert this stage.
    used: []u8,
    cursor: [2]usize = .{ 0, 0 },
    pool: ?Pool = null,
    now: u32 = 0,
    stats: Stats = .{},
    /// The host's scratch for one stage: the layer's experts seen.
    mark: []u32,
    mark_serial: u32 = 0,

    /// A tier over the window's `bytes` (device address `dev`); its table starts with every expert at the first slot.
    pub fn init(gpa: std.mem.Allocator, shape: Shape, bytes: []u8, dev: u64, shared_base: u64) Error!Tier {
        try shape.check();
        const w = Window.init(shape, bytes, dev);
        if (w.nslots < Window.leastSlots(shape) or w.nslots % 2 != 0) return error.BadShape;
        const host = try gpa.alloc(?[]u8, shape.layers);
        errdefer gpa.free(host);
        @memset(host, null);
        const used = try gpa.alloc(u8, w.nslots);
        errdefer gpa.free(used);
        @memset(used, 0);
        const mark = try gpa.alloc(u32, shape.experts);
        errdefer gpa.free(mark);
        @memset(mark, 0);
        w.initTable(shape, shared_base);
        return .{ .gpa = gpa, .shape = shape, .win = w, .host = host, .used = used, .mark = mark };
    }

    pub fn deinit(t: *Tier) void {
        if (t.pool) |*p| p.deinit(t.gpa);
        t.gpa.free(t.mark);
        t.gpa.free(t.used);
        for (t.host) |h| if (h) |bytes| std.heap.page_allocator.free(bytes);
        t.gpa.free(t.host);
        t.* = undefined;
    }

    /// Host bytes of the tier, and the device bytes of what is always resident (the shared experts).
    pub fn hostBytes(shape: Shape) usize {
        return shape.layers * shape.experts * shape.blobBytes();
    }

    pub fn residentBytes(shape: Shape) usize {
        return shape.layers * shape.blobBytes();
    }

    /// Layer `layer`'s routed experts into the host tier; `tensors` are the six parts of every expert, shared last.
    pub fn addLayer(t: *Tier, layer: usize, tensors: [parts][]const u8) Error!void {
        if (layer >= t.shape.layers) return error.BadShape;
        const e = t.shape.experts;
        const blob = t.shape.blobBytes();
        for (tensors, t.shape.part_bytes) |part, b| if (part.len != b * (e + 1)) return error.BadShape;
        if (t.host[layer] == null) t.host[layer] = try std.heap.page_allocator.alloc(u8, e * blob);
        const dst = t.host[layer].?;
        for (0..e) |x| packExpert(t.shape, tensors, x, dst[x * blob ..][0..blob]);
    }

    /// The shared expert's blob of a layer's stacked parts: the last expert of each.
    pub fn sharedBlob(shape: Shape, tensors: [parts][]const u8, out: []u8) void {
        packExpert(shape, tensors, shape.experts, out[0..shape.blobBytes()]);
    }

    fn packExpert(shape: Shape, tensors: [parts][]const u8, x: usize, out: []u8) void {
        var at: usize = 0;
        for (tensors, shape.part_bytes) |part, b| {
            @memcpy(out[at..][0..b], part[x * b ..][0..b]);
            at += b;
        }
    }

    /// Keeps a pool of `lines` cache lines, admitting at most `copy_experts` misses a round.
    pub fn attachPool(t: *Tier, lines: usize, copy_experts: usize, protect_pct: u8) Error!void {
        std.debug.assert(t.pool == null);
        t.pool = try Pool.init(t.gpa, t.shape, lines, copy_experts, protect_pct);
    }

    pub fn lineCount(t: *const Tier) usize {
        return if (t.pool) |p| p.slots.len else 0;
    }

    /// Decides the layer's experts from the trace and fills its table row and window slots; `ports` is the device's.
    pub fn stage(t: *Tier, job: Job, ports: Ports) error{StageFailed}!void {
        const shape = t.shape;
        const e = shape.experts;
        const picks = shape.picks;
        const layer: usize = job.layer;
        const blob = shape.blobBytes();
        const words = t.win.traceWords();
        const base: policy.Id = @intCast(layer * e);
        var no_ids: [0]policy.Id = .{};
        var n: usize = 0;
        const ids: []policy.Id = if (t.pool) |p| p.ids else &no_ids;
        const decide = t.pool != null and !job.prefill;
        t.mark_serial +%= 1;
        if (t.mark_serial == 0) {
            @memset(t.mark, 0);
            t.mark_serial = 1;
        }
        var uniq: [4096]u32 = undefined;
        var nu: usize = 0;
        if (job.rows > shape.trace_rows) return error.StageFailed;
        for (0..job.rows) |r| for (words[r * picks ..][0 .. picks - 1]) |id| {
            if (id < 0 or id >= e) continue;
            const x: u32 = @intCast(id);
            if (decide) {
                ids[n] = base + x;
                n += 1;
            }
            if (t.mark[x] != t.mark_serial) {
                t.mark[x] = t.mark_serial;
                if (nu >= uniq.len) return error.StageFailed;
                uniq[nu] = x;
                nu += 1;
            }
        };
        var started: usize = 0;
        if (decide) {
            const p = &t.pool.?;
            t.now +%= 1;
            const budget: policy.Policy = .{ .slots = @intCast(p.slots.len), .expert_bytes = blob, .copy_bytes = @as(u64, p.budget) * blob };
            started = policy.schedule(&p.vram, ids[0..n], budget, t.now, p.decisions[0..n], .{ .unique = p.scratch_unique, .tally = p.tally }, p.copies) catch
                return error.StageFailed;
            p.budget -= started;
            for (p.decisions[0..n]) |d| {
                t.stats.accesses += 1;
                if (d == .hit) t.stats.hits += 1;
            }
        }
        const half = layer & 1;
        const per_half = t.win.nslots / 2;
        for (t.used[half * per_half ..][0..per_half]) |*u| {
            if (u.* == 1) u.* = 0;
        }
        const table = t.win.tableWords()[layer * shape.entries() * 2 ..][0 .. e * 2];
        const host = t.host[layer] orelse return error.StageFailed;
        const prefill_pool = job.prefill and t.pool != null;
        var staged: usize = 0;
        for (uniq[0..nu]) |x| {
            const key: policy.Id = base + x;
            if (t.pool) |*p| if (policy.onDevice(p.vram, key)) {
                table[2 * x] = ports.line_addr(ports.ctx, @intCast(p.vram.where[key]));
                table[2 * x + 1] = 0;
                continue;
            };
            const w = t.takeSlot(half, per_half) orelse return error.StageFailed;
            @memcpy(t.win.slotBytes(shape, w), host[x * blob ..][0..blob]);
            t.stats.window_bytes += blob;
            table[2 * x + 1] = 0;
            if (prefill_pool) {
                const at = staged * blob;
                ports.scratch_copy(ports.ctx, at, t.win.slotOffset(shape, w), blob) catch return error.StageFailed;
                table[2 * x] = ports.scratch_base + at;
                staged += 1;
                continue;
            }
            table[2 * x] = t.win.slotAddr(shape, w);
        }
        if (staged > 0) {
            ports.scratch_wait(ports.ctx) catch return error.StageFailed;
            t.stats.scratch_bytes += staged * blob;
        }
        if (started > 0) {
            const p = &t.pool.?;
            for (p.copies[0..started]) |c| {
                table[2 * (c.id - base) + 1] = ports.line_addr(ports.ctx, c.slot);
                policy.reap(&p.vram, c.id, t.now) catch return error.StageFailed;
            }
            t.stats.admissions += started;
        }
    }

    fn takeSlot(t: *Tier, half: usize, per_half: usize) ?usize {
        const lo = half * per_half;
        var at = t.cursor[half];
        for (0..per_half) |_| {
            if (at >= per_half) at = 0;
            if (t.used[lo + at] == 0) {
                t.used[lo + at] = 1;
                t.cursor[half] = at + 1;
                return lo + at;
            }
            at += 1;
        }
        return null;
    }

    /// The round over its first `rows` rows is finished: its time joins the counters and the admission budget renews.
    pub fn endRound(t: *Tier, rows: usize, ns: u64) void {
        t.stats.rounds += 1;
        t.stats.tokens += rows;
        t.stats.round_ns += ns;
        if (t.pool) |*p| p.budget = p.copy_experts;
    }
};
