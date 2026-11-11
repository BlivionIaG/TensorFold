//! A MoE model's routed experts in three tiers on one HIP device whose host memory the card can see only through a small DMA window
//! (hipmac's TinyGPU dext maps about 1.96 GiB of sysmem per connection in total: docs/HIPMAC.md):
//!   * the host tier: every expert's blob in ordinary pageable memory (17 GiB for Qwen3.6-35B-A3B 4-bit), which the card cannot read;
//!   * the window: a pinned, device-mapped region of about a GiB and a half that the host fills with the experts a layer needs and the card
//!     reads in place over the link (an expert served from here is a miss);
//!   * cache lines of device memory: the hot experts, admitted by SDMA copies from the window (a hit reads VRAM).
//! The routed GEMVs take a pointer a (projection, expert) from a table (kernels/hip/tiles/plan.hpp, caps `expert_ptrs`), which lives in the
//! window too: the card learns where an expert sits one layer at a time, so no expert is read before its bytes are in place and the
//! arithmetic is the checkpoint's wherever an expert sits.
//!
//! One blob holds an expert's six tensors in kernel order: the fused (gate | up) projection's words, scales and biases, then down's. Table
//! entry (layer, e) is the device address of expert e's blob, in a cache line or in a window slot; the shared expert (id `experts`) has a
//! line of its own that never leaves.
//!
//! A layer's stage, in the stream (the decode rounds and prompt spans of an offloaded model run eager, not as a graph: stream value waits
//! cannot be captured):
//!   router, pick, plan -> `tf_expert_trace` (the picks to the window) -> write `req` = n -> wait `ack` >= n -> the expert GEMVs
//! and on the host a worker thread spins on `req`, reads the layer's picks, lets the policy of core/expert_offload.zig (the CUDA EXL3 host
//! cache's rules) decide hits and admissions, copies the blobs of the experts that are not in a cache line into window slots, writes their
//! table entries, then stores `ack`. Two halves of the window alternate by layer, so a layer's slots are never rewritten while its GEMVs run.
//!   * `beginRound`: copies that landed make their experts resident (the next layer's table names the line);
//!   * `endRound`: the admissions a round decided are queued on a copy stream (SDMA from the window), under the next round's compute.
//! Counters (`Stats`) report hits and misses an access (a router pick of a row), copies started and landed, window bytes and stage time.
//! Architecture: plain pointer loads; nothing here names a GPU, and the budget is whatever device memory is left.

const std = @import("std");
const core = @import("core");
const platform = @import("sys");
const runtime = @import("runtime.zig");
const memory = @import("memory.zig");
const stream_mod = @import("stream.zig");
const moe = @import("moe.zig");

const policy = core.expert_offload;
const abi = @import("abi.zig");
const Runtime = runtime.Runtime;
const DeviceBuffer = memory.DeviceBuffer;
const HostBuffer = memory.HostBuffer;
const Stream = stream_mod.Stream;
const Event = stream_mod.Event;

extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
const c_setenv = setenv;

pub const Error = runtime.Error || std.mem.Allocator.Error || error{ BadShape, NotSized, NoValueOps, StageFailed };

/// The six tensors of one expert in blob order: fused words, scales, biases, then down's.
pub const parts = 6;

/// What a model offers the store: its layers, the routed experts a layer holds in the host tier, and the bytes of each part of one expert.
pub const Shape = struct {
    layers: usize,
    /// Routed experts a layer; the shared expert is one more, always resident.
    experts: usize,
    /// Router picks a row (the routed ones plus the shared expert's): the trace holds this many ids a row.
    picks: usize,
    /// The most rows a forward stages (a round's bucket, a prompt span).
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

/// What a layer's views need: its table, where its projections sit in a blob, the stage hook, and an address the stub tensors name.
pub const Ref = struct { ptab: u64, fused_off: u64, down_off: u64, anchor: u64, hook: moe.Hook };

pub const Stats = struct {
    rounds: u64 = 0,
    accesses: u64 = 0,
    hits: u64 = 0,
    copies_started: u64 = 0,
    copies_landed: u64 = 0,
    bytes_copied: u64 = 0,
    /// Rows decoded, and the wall nanoseconds of the rounds (forward, readback and draw included).
    tokens: u64 = 0,
    round_ns: u64 = 0,
    /// Layer stages (decode and prompt), the bytes the host copied into the window, and the nanoseconds the worker spent from a request to its ack.
    stages: u64 = 0,
    window_bytes: u64 = 0,
    stage_ns: u64 = 0,
    /// Nanoseconds of the stages' waits for the card's request (the card computing; not stage work).
    idle_ns: u64 = 0,
};

/// An `interval` log line every this many rounds.
const report_every: u64 = 256;

/// Device memory is allocated in pieces of this many bytes at most: one physically contiguous gigabytes-long block may not exist.
const chunk_bytes: usize = 512 << 20;

/// Layer stages that may be posted and not yet served (a forward posts one per layer; a prompt's spans queue several forwards).
const ring: u32 = 4096;

/// hipStreamWaitValue32's greater-or-equal compare.
const wait_gte: c_uint = 0;

const WaitValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint, u32) callconv(.c) abi.Result;
const WriteValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint) callconv(.c) abi.Result;

const Job = struct { layer: u32, rows: u32, prefill: bool };

/// An admission: expert `key` is copied from window slot `win` to cache line `line`.
const Landing = struct { key: policy.Id, line: u32, win: u32 };

const Flight = struct { event: Event, items: []Landing };

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
    /// Admissions the worker decided in this round's stages; `endRound` queues them.
    pending: []Landing,
    pending_n: usize = 0,
    flights: std.ArrayList(Flight),
    spare: std.ArrayList(Event),
    /// Copies the round's remaining stages may still start.
    budget: usize,
    copy_experts: usize,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    r: *const Runtime,
    shape: Shape,
    /// Per layer: every routed expert's blob, pageable host memory (null until the layer is added).
    host: []?[]u8,
    /// The shared experts, one blob a layer.
    shared: DeviceBuffer,
    /// The window: control words, the table, the picks of the stage in flight, then the blob slots.
    win: HostBuffer,
    win_dev: u64,
    ptab_off: usize,
    trace_off: usize,
    slots_off: usize,
    nslots: usize,
    /// Slot states: 0 free, 1 holds a layer's expert this stage, 2 is the source of an admission in flight.
    used: []u8,
    cursor: [2]usize = .{ 0, 0 },
    wait32: WaitValue32,
    write32: WriteValue32,
    pool: ?Pool = null,
    now: u32 = 0,
    stats: Stats = .{},
    since: Stats = .{},
    layers_added: usize = 0,
    /// The host's scratch for one stage: the layer's experts seen, and each one's window slot.
    mark: []u32,
    mark_serial: u32 = 0,
    slot_of: []i32,
    jobs: []Job,
    /// Jobs posted (main thread only), published to the worker, and served.
    posted: u32 = 0,
    published: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,

    const ctl_bytes: usize = 4096;
    const req_at: usize = 0;
    const ack_at: usize = 64;

    /// Bytes of the window's head (control, table, trace) for `shape`.
    fn headBytes(shape: Shape) usize {
        const ptab = std.mem.alignForward(usize, shape.layers * (shape.experts + 1) * 8, 4096);
        const trace = std.mem.alignForward(usize, shape.trace_rows * shape.picks * 4, 4096);
        return ctl_bytes + ptab + trace;
    }

    /// `window_bytes` of pinned memory for the window (slots rounded down to an even count of at least two halves that each hold a layer's
    /// every expert); less when the runtime refuses that much, down to the least the halves need.
    pub fn init(gpa: std.mem.Allocator, r: *const Runtime, shape: Shape, window_bytes: usize) Error!Store {
        try shape.check();
        const blob = shape.blobBytes();
        const lib: *@TypeOf(r.lib) = @constCast(&r.lib);
        const wait32 = lib.lookup(WaitValue32, "hipStreamWaitValue32") orelse return error.NoValueOps;
        const write32 = lib.lookup(WriteValue32, "hipStreamWriteValue32") orelse return error.NoValueOps;
        // hipmac bounds a value wait from its submission (10 s by default), and a forward queues all its layers' waits at once: a prompt span whose
        // earlier layers take longer than that would be released early. The host worker answers every wait, so none is left unbounded for long.
        _ = c_setenv("HIPMAC_WAIT_VALUE_TIMEOUT_MS", "0", 0);
        const head = headBytes(shape);
        const least = 2 * (shape.experts + 8);
        var want = window_bytes;
        var win: HostBuffer = undefined;
        var nslots: usize = 0;
        while (true) {
            nslots = (want -| head) / blob / 2 * 2;
            if (nslots < least) return error.BadShape;
            if (HostBuffer.allocMapped(r, head + nslots * blob)) |w| {
                win = w;
                break;
            } else |err| {
                if (want / 4 * 3 < head + least * blob) return err;
                want = want / 4 * 3;
            }
        }
        errdefer win.free();
        @memset(win.bytes[0..head], 0);
        const host = try gpa.alloc(?[]u8, shape.layers);
        errdefer gpa.free(host);
        @memset(host, null);
        var shared = try DeviceBuffer.alloc(r, shape.layers * blob);
        errdefer shared.free();
        const used = try gpa.alloc(u8, nslots);
        errdefer gpa.free(used);
        @memset(used, 0);
        const mark = try gpa.alloc(u32, shape.experts);
        errdefer gpa.free(mark);
        @memset(mark, 0);
        const slot_of = try gpa.alloc(i32, shape.experts);
        errdefer gpa.free(slot_of);
        const jobs = try gpa.alloc(Job, ring);
        errdefer gpa.free(jobs);
        const win_dev = try win.device();
        var s: Store = .{
            .gpa = gpa,
            .r = r,
            .shape = shape,
            .host = host,
            .shared = shared,
            .win = win,
            .win_dev = win_dev,
            .ptab_off = ctl_bytes,
            .trace_off = ctl_bytes + std.mem.alignForward(usize, shape.layers * (shape.experts + 1) * 8, 4096),
            .slots_off = head,
            .nslots = nslots,
            .used = used,
            .wait32 = wait32,
            .write32 = write32,
            .mark = mark,
            .slot_of = slot_of,
            .jobs = jobs,
        };
        const words = s.tableWords();
        for (0..shape.layers) |l| {
            const row = words[l * (shape.experts + 1) ..][0 .. shape.experts + 1];
            @memset(row[0..shape.experts], win_dev + head);
            row[shape.experts] = shared.base() + l * blob;
        }
        return s;
    }

    pub fn deinit(s: *Store) void {
        s.report("total");
        s.stopWorker();
        s.drain();
        if (s.pool) |*p| {
            for (p.flights.items) |*f| {
                f.event.deinit();
                s.gpa.free(f.items);
            }
            for (p.spare.items) |*e| e.deinit();
            p.flights.deinit(s.gpa);
            p.spare.deinit(s.gpa);
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
            s.gpa.free(p.pending);
        }
        s.gpa.free(s.jobs);
        s.gpa.free(s.slot_of);
        s.gpa.free(s.mark);
        s.gpa.free(s.used);
        s.shared.free();
        s.win.free();
        for (s.host) |h| if (h) |bytes| std.heap.page_allocator.free(bytes);
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

    /// Host bytes of the tier and device bytes of what is always resident (the shared experts).
    pub fn hostBytes(shape: Shape) usize {
        return shape.layers * shape.experts * shape.blobBytes();
    }

    pub fn residentBytes(shape: Shape) usize {
        return shape.layers * shape.blobBytes();
    }

    /// The window's pinned bytes, slots included.
    pub fn windowBytes(s: *const Store) usize {
        return s.win.bytes.len;
    }

    /// Layer `layer`'s experts as a stack: `tensors` are the six parts, each (experts + 1, ...) with the shared expert last. The routed ones go
    /// to the host tier, the shared one to its line.
    pub fn addLayer(s: *Store, layer: usize, tensors: [parts][]const u8) Error!void {
        if (layer >= s.shape.layers) return error.BadShape;
        const e = s.shape.experts;
        const blob = s.shape.blobBytes();
        for (tensors, s.shape.part_bytes) |t, b| if (t.len != b * (e + 1)) return error.BadShape;
        if (s.host[layer] == null) s.host[layer] = try std.heap.page_allocator.alloc(u8, e * blob);
        const dst = s.host[layer].?;
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
    pub fn ref(s: *Store, layer: usize) Ref {
        const entries = s.shape.experts + 1;
        return .{
            .ptab = s.win_dev + s.ptab_off + layer * entries * 8,
            .fused_off = 0,
            .down_off = s.shape.downOffset(),
            .anchor = s.win_dev,
            .hook = .{ .ctx = s, .post = post, .layer = @intCast(layer), .trace = s.win_dev + s.trace_off },
        };
    }

    fn tableWords(s: *const Store) []u64 {
        const bytes = s.win.bytes[s.ptab_off..s.trace_off];
        return @alignCast(std.mem.bytesAsSlice(u64, bytes[0 .. bytes.len / 8 * 8]));
    }

    fn traceWords(s: *const Store) []i32 {
        const bytes = s.win.bytes[s.trace_off..s.slots_off];
        return @alignCast(std.mem.bytesAsSlice(i32, bytes[0 .. bytes.len / 4 * 4]));
    }

    fn word(s: *const Store, at: usize) *u32 {
        return @ptrCast(@alignCast(s.win.bytes[at..].ptr));
    }

    fn slotBytes(s: *const Store, slot: usize) []u8 {
        const blob = s.shape.blobBytes();
        return s.win.bytes[s.slots_off + slot * blob ..][0..blob];
    }

    fn slotAddr(s: *const Store, slot: usize) u64 {
        return s.win_dev + s.slots_off + slot * s.shape.blobBytes();
    }

    fn lineAddr(s: *const Store, p: *const Pool, slot: usize) u64 {
        return p.chunks[slot / p.per_chunk].base() + (slot % p.per_chunk) * s.shape.blobBytes();
    }

    /// Cache lines `lines` of device memory, admitting at most `copy_experts` experts a round.
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
        const picks_max = s.shape.trace_rows * (s.shape.picks - 1);
        const unique = try s.gpa.alloc(policy.Id, picks_max);
        errdefer s.gpa.free(unique);
        const tally = try s.gpa.alloc(u16, keys);
        errdefer s.gpa.free(tally);
        @memset(tally, 0);
        const decisions = try s.gpa.alloc(policy.Decision, picks_max);
        errdefer s.gpa.free(decisions);
        const ids = try s.gpa.alloc(policy.Id, picks_max);
        errdefer s.gpa.free(ids);
        const copies = try s.gpa.alloc(policy.Copy, @max(copy_experts, 1));
        errdefer s.gpa.free(copies);
        const pending = try s.gpa.alloc(Landing, @max(copy_experts, 1));
        errdefer s.gpa.free(pending);
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
            .pending = pending,
            .flights = .empty,
            .spare = .empty,
            .budget = copy_experts,
            .copy_experts = copy_experts,
        };
    }

    pub fn lineCount(s: *const Store) usize {
        return if (s.pool) |p| p.slots.len else 0;
    }

    /// Starts the worker that serves the layers' stages; the first offloaded forward needs it.
    pub fn start(s: *Store) Error!void {
        if (s.worker != null) return;
        s.worker = std.Thread.spawn(.{}, workerMain, .{s}) catch return error.OutOfMemory;
    }

    fn stopWorker(s: *Store) void {
        const w = s.worker orelse return;
        s.stop.store(true, .release);
        w.join();
        s.worker = null;
    }

    /// Waits until the worker has served every posted stage: no policy or window state moves under the caller after this.
    fn quiesce(s: *Store) void {
        while (s.done.load(.acquire) != s.posted) std.atomic.spinLoopHint();
    }

    /// A layer's stage in the stream, after its picks were traced: the card asks (`req`), then waits for the host's answer (`ack`).
    fn post(ctx: *anyopaque, stream: abi.Stream, layer: u32, rows: usize, prefill: bool) moe.Error!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        // a dry run measuring scratch launches nothing and waits for no one
        if (stream == @import("launches/util.zig").counting) return;
        if (rows == 0 or rows > s.shape.trace_rows or s.worker == null) return error.BadShape;
        while (s.posted -% s.done.load(.acquire) >= ring - 1) std.atomic.spinLoopHint();
        s.posted +%= 1;
        s.jobs[s.posted % ring] = .{ .layer = layer, .rows = @intCast(rows), .prefill = prefill };
        s.published.store(s.posted, .release);
        try runtime.check(s.write32(stream, s.win.bytes[req_at..].ptr, s.posted, 0));
        try runtime.check(s.wait32(stream, s.win.bytes[ack_at..].ptr, s.posted, wait_gte, 0xffff_ffff));
    }

    fn workerMain(s: *Store) void {
        const req: *volatile u32 = @ptrCast(@alignCast(s.win.bytes[req_at..].ptr));
        var waited: u64 = nowNs();
        while (!s.stop.load(.acquire)) {
            const n = s.done.load(.monotonic) +% 1;
            if (s.published.load(.acquire) -% n >= 0x8000_0000) {
                // nothing posted: the engine is between forwards
                napNs(50_000);
                waited = nowNs();
                continue;
            }
            // the card asks once it has traced the picks of stage n
            if (@atomicLoad(u32, @as(*const u32, @volatileCast(req)), .acquire) -% n >= 0x8000_0000) {
                std.atomic.spinLoopHint();
                continue;
            }
            const began = nowNs();
            s.stats.idle_ns += began - waited;
            s.serve(s.jobs[n % ring]);
            const ended = nowNs();
            s.stats.stage_ns += ended - began;
            s.stats.stages += 1;
            s.done.store(n, .release);
            waited = nowNs();
        }
    }

    /// Stage `job`: decide, fill the window, write the table row, answer.
    fn serve(s: *Store, job: Job) void {
        s.stage(job) catch s.failed.store(true, .release);
        @atomicStore(u32, s.word(ack_at), s.done.load(.monotonic) +% 1, .release);
    }

    fn stage(s: *Store, job: Job) error{StageFailed}!void {
        const e = s.shape.experts;
        const picks = s.shape.picks;
        const layer: usize = job.layer;
        const blob = s.shape.blobBytes();
        const words = s.traceWords();
        const base: policy.Id = @intCast(layer * e);
        var no_ids: [0]policy.Id = .{};
        var n: usize = 0;
        const ids: []policy.Id = if (s.pool) |p| p.ids else &no_ids;
        const have_pool = s.pool != null;
        s.mark_serial +%= 1;
        if (s.mark_serial == 0) {
            @memset(s.mark, 0);
            s.mark_serial = 1;
        }
        // this layer's distinct experts, and (with a cache) every pick for the policy
        var uniq: [4096]u32 = undefined;
        var nu: usize = 0;
        if (job.rows > s.shape.trace_rows) return error.StageFailed;
        for (0..job.rows) |r| for (words[r * picks ..][0 .. picks - 1]) |id| {
            if (id < 0 or id >= e) continue;
            const x: u32 = @intCast(id);
            if (have_pool and !job.prefill) {
                ids[n] = base + x;
                n += 1;
            }
            if (s.mark[x] != s.mark_serial) {
                s.mark[x] = s.mark_serial;
                if (nu >= uniq.len) return error.StageFailed;
                uniq[nu] = x;
                nu += 1;
            }
        };
        var started: usize = 0;
        if (have_pool and !job.prefill) {
            const p = &s.pool.?;
            s.now +%= 1;
            started = policy.schedule(&p.vram, ids[0..n], .{ .slots = @intCast(p.slots.len), .expert_bytes = blob, .copy_bytes = @as(u64, p.budget) * blob }, s.now, p.decisions[0..n], .{ .unique = p.scratch_unique, .tally = p.tally }, p.copies) catch return error.StageFailed;
            p.budget -= started;
            for (p.decisions[0..n]) |d| {
                s.stats.accesses += 1;
                if (d == .hit) s.stats.hits += 1;
            }
        }
        // the layer's half of the window is free again: its previous layer's GEMVs have finished (they precede this layer's trace in the stream)
        const half = layer & 1;
        const per_half = s.nslots / 2;
        for (s.used[half * per_half ..][0..per_half]) |*u| {
            if (u.* == 1) u.* = 0;
        }
        const table = s.tableWords()[layer * (e + 1) ..][0..e];
        const host = s.host[layer] orelse return error.StageFailed;
        for (uniq[0..nu]) |x| {
            const key: policy.Id = base + x;
            if (have_pool and policy.onDevice(s.pool.?.vram, key)) {
                const p = &s.pool.?;
                table[x] = s.lineAddr(p, @intCast(p.vram.where[key]));
                s.slot_of[x] = -1;
                continue;
            }
            const w = s.takeSlot(half, per_half) orelse return error.StageFailed;
            @memcpy(s.slotBytes(w), host[x * blob ..][0..blob]);
            s.stats.window_bytes += blob;
            table[x] = s.slotAddr(w);
            s.slot_of[x] = @intCast(w);
        }
        if (started > 0) {
            const p = &s.pool.?;
            for (p.copies[0..started]) |c| {
                const x = c.id - base;
                const w = s.slot_of[x];
                if (w < 0 or p.pending_n >= p.pending.len) return error.StageFailed;
                s.used[@intCast(w)] = 2;
                p.pending[p.pending_n] = .{ .key = c.id, .line = c.slot, .win = @intCast(w) };
                p.pending_n += 1;
            }
        }
    }

    fn takeSlot(s: *Store, half: usize, per_half: usize) ?usize {
        const lo = half * per_half;
        var at = s.cursor[half];
        for (0..per_half) |_| {
            if (at >= per_half) at = 0;
            if (s.used[lo + at] == 0) {
                s.used[lo + at] = 1;
                s.cursor[half] = at + 1;
                return lo + at;
            }
            at += 1;
        }
        return null;
    }

    /// Experts that have landed become resident: the next stage's table names their lines.
    pub fn beginRound(s: *Store) Error!void {
        s.quiesce();
        const p = if (s.pool) |*p| p else return;
        while (p.flights.items.len > 0) {
            const f = p.flights.items[0];
            if (!try f.event.done()) break;
            for (f.items) |c| {
                // a line whose expert was staged again before it landed is not this copy's
                policy.reap(&p.vram, c.key, s.now) catch {};
                s.used[c.win] = 0;
                s.stats.copies_landed += 1;
            }
            s.gpa.free(f.items);
            try p.spare.append(s.gpa, f.event);
            _ = p.flights.orderedRemove(0);
        }
    }

    /// The round over its first `rows` rows is finished: the admissions its stages decided are queued on the copy stream.
    pub fn endRound(s: *Store, rows: usize, ns: u64) Error!void {
        s.quiesce();
        if (s.failed.load(.acquire)) return error.StageFailed;
        s.stats.rounds += 1;
        s.stats.tokens += rows;
        s.stats.round_ns += ns;
        if (s.stats.rounds % report_every == 0) s.report("interval");
        const p = if (s.pool) |*p| p else return;
        defer {
            p.pending_n = 0;
            p.budget = p.copy_experts;
        }
        const n = p.pending_n;
        if (n == 0) return;
        const blob = s.shape.blobBytes();
        const items = try s.gpa.dupe(Landing, p.pending[0..n]);
        errdefer s.gpa.free(items);
        for (items) |c| {
            const from = s.win.view(s.slots_off + c.win * blob, s.slots_off + (c.win + 1) * blob);
            try p.chunks[c.line / p.per_chunk].uploadAsync((c.line % p.per_chunk) * blob, from, p.stream);
        }
        var event = if (p.spare.pop()) |ev| ev else try Event.init(s.r, false);
        errdefer event.deinit();
        try event.record(p.stream);
        try p.flights.append(s.gpa, .{ .event = event, .items = items });
        s.stats.copies_started += n;
        s.stats.bytes_copied += n * blob;
    }

    /// One log line for the rounds since the last (or since the start): hit rate, misses and window traffic a token, the stage time.
    /// A miss is a router pick read over the link from a window slot; the shared expert and the dense layers are not counted.
    pub fn report(s: *Store, label: []const u8) void {
        const d = Stats{
            .rounds = s.stats.rounds - s.since.rounds,
            .accesses = s.stats.accesses - s.since.accesses,
            .hits = s.stats.hits - s.since.hits,
            .copies_started = s.stats.copies_started - s.since.copies_started,
            .copies_landed = s.stats.copies_landed - s.since.copies_landed,
            .bytes_copied = s.stats.bytes_copied - s.since.bytes_copied,
            .tokens = s.stats.tokens - s.since.tokens,
            .round_ns = s.stats.round_ns - s.since.round_ns,
            .stages = s.stats.stages - s.since.stages,
            .window_bytes = s.stats.window_bytes - s.since.window_bytes,
            .stage_ns = s.stats.stage_ns - s.since.stage_ns,
            .idle_ns = s.stats.idle_ns - s.since.idle_ns,
        };
        s.since = s.stats;
        if (d.rounds == 0) return;
        const a: f64 = @floatFromInt(@max(d.accesses, 1));
        const t: f64 = @floatFromInt(@max(d.tokens, 1));
        const r: f64 = @floatFromInt(d.rounds);
        const miss: f64 = @floatFromInt(d.accesses - d.hits);
        const mib: f64 = 1024.0 * 1024.0;
        const blob: f64 = @floatFromInt(s.shape.blobBytes());
        std.log.info(
            "experts {s}: {d} rounds {d} tokens, hit {d:.1}% miss {d:.1}/token ({d:.1} MiB/token over the link), window fill {d:.1} MiB/token, copies {d:.1}/round ({d:.1} MiB/round, {d} landed), stage {d:.2} ms/token (card compute+wait {d:.2}), round {d:.2} ms ({d:.2} ms/token), {d} lines",
            .{ label, d.rounds, d.tokens, 100.0 * @as(f64, @floatFromInt(d.hits)) / a, miss / t, miss / t * blob / mib, @as(f64, @floatFromInt(d.window_bytes)) / t / mib, @as(f64, @floatFromInt(d.copies_started)) / r, @as(f64, @floatFromInt(d.bytes_copied)) / r / mib, d.copies_landed, @as(f64, @floatFromInt(d.stage_ns)) / t / 1e6, @as(f64, @floatFromInt(d.idle_ns)) / t / 1e6, @as(f64, @floatFromInt(d.round_ns)) / r / 1e6, @as(f64, @floatFromInt(d.round_ns)) / t / 1e6, s.lineCount() },
        );
    }
};

fn nowNs() u64 {
    var ts: platform.c.timespec = undefined;
    _ = platform.c.clock_gettime(platform.c.CLOCK.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

fn napNs(ns: u64) void {
    var ts: std.c.timespec = .{ .sec = 0, .nsec = @intCast(ns) };
    _ = std.c.nanosleep(&ts, null);
}

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

fn setPicks(s: *Store, row: usize, ids: []const i32) void {
    @memcpy(s.traceWords()[row * s.shape.picks ..][0..ids.len], ids);
}

test "stages fill window slots, admissions land in lines, and the table names where each expert sits" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 2, .experts = 4, .picks = 3, .trace_rows = 2, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    const blob = shape.blobBytes();
    try std.testing.expectEqual(@as(usize, 160), blob);
    try std.testing.expectEqual(@as(usize, 96), shape.downOffset());
    var s = try Store.init(gpa, &r, shape, 1 << 20);
    defer s.deinit();
    try std.testing.expect(s.nslots >= 2 * (shape.experts + 8) and s.nslots % 2 == 0);
    const t0 = try testTensors(gpa, shape, 1);
    const t1 = try testTensors(gpa, shape, 2);
    defer for (t0 ++ t1) |t| gpa.free(t);
    try s.addLayer(0, t0);
    try s.addLayer(1, t1);
    // an expert's blob in the host tier is its six parts back to back; the shared one is in its line
    var want: [160]u8 = undefined;
    blobOf(shape, t1, 2, &want);
    try std.testing.expectEqualSlices(u8, &want, s.host[1].?[2 * blob ..][0..blob]);
    blobOf(shape, t1, 4, &want);
    var got: [160]u8 = undefined;
    try s.shared.download(1 * blob, &got);
    try std.testing.expectEqualSlices(u8, &want, &got);
    try std.testing.expectEqual(s.shared.base() + blob, s.tableWords()[1 * 5 + 4]);
    try std.testing.expectError(error.BadShape, s.addLayer(2, t0));

    try s.attachPool(3, 2);
    // stage 1 (layer 0, decode): rows name 1 twice, 2 and 3. Two admissions fit the budget: the hottest (key 1) and the lowest id of the rest
    setPicks(&s, 0, &.{ 1, 2, 4 });
    setPicks(&s, 1, &.{ 1, 3, 4 });
    try s.stage(.{ .layer = 0, .rows = 2, .prefill = false });
    try std.testing.expectEqual(@as(u64, 0), s.stats.hits);
    try std.testing.expectEqual(@as(usize, 2), s.pool.?.pending_n);
    const slot1 = s.tableWords()[1];
    try std.testing.expect(slot1 >= s.win_dev + s.slots_off and slot1 < s.win_dev + s.win.bytes.len);
    blobOf(shape, t0, 1, &want);
    try std.testing.expectEqualSlices(u8, &want, s.slotBytes((slot1 - s.win_dev - s.slots_off) / blob));
    // the card reads misses where they sit: all three experts have window slots; the two admitted ones are pinned as copy sources
    var pinned: usize = 0;
    for (s.used) |u| pinned += @intFromBool(u == 2);
    try std.testing.expectEqual(@as(usize, 2), pinned);
    // the round ends: the admissions are queued, land, and become resident
    s.posted = s.done.load(.acquire);
    try s.endRound(2, 1000);
    try std.testing.expectEqual(@as(u64, 2), s.stats.copies_started);
    s.drain();
    try s.beginRound();
    try std.testing.expectEqual(@as(u64, 2), s.stats.copies_landed);
    for (s.used) |u| try std.testing.expect(u != 2);
    try std.testing.expect(policy.onDevice(s.pool.?.vram, 1));
    // stage 2 (layer 0 again): the resident expert is a hit and its entry names its line, with the bytes the copy moved
    setPicks(&s, 0, &.{ 1, 2, 4 });
    setPicks(&s, 1, &.{ 1, 2, 4 });
    try s.stage(.{ .layer = 0, .rows = 2, .prefill = false });
    try std.testing.expect(s.stats.hits >= 2);
    const line = s.tableWords()[1];
    try std.testing.expect(line < s.win_dev + s.slots_off or line >= s.win_dev + s.win.bytes.len);
    blobOf(shape, t0, 1, &want);
    var dev: [160]u8 = undefined;
    try s.pool.?.chunks[0].download(line - s.pool.?.chunks[0].base(), &dev);
    try std.testing.expectEqualSlices(u8, &want, &dev);
    // a prompt span stages every named expert, in its layer's half, and admits nothing
    const before = s.pool.?.pending_n;
    setPicks(&s, 0, &.{ 0, 3, 4 });
    setPicks(&s, 1, &.{ 2, 3, 4 });
    try s.stage(.{ .layer = 1, .rows = 2, .prefill = true });
    try std.testing.expectEqual(before, s.pool.?.pending_n);
    const half = s.nslots / 2;
    const e0 = s.tableWords()[1 * 5 + 0];
    const w0 = (e0 - s.win_dev - s.slots_off) / blob;
    try std.testing.expect(w0 >= half);
    blobOf(shape, t1, 0, &want);
    try std.testing.expectEqualSlices(u8, &want, s.slotBytes(w0));
}
