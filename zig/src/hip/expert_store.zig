//! A MoE model's routed experts in three tiers on one HIP device whose host memory the card can see only through a small DMA window
//! (hipmac's TinyGPU dext maps about 1.96 GiB of sysmem per connection in total: docs/HIPMAC.md):
//!   * the host tier: every expert's blob in ordinary pageable memory (17 GiB for Qwen3.6-35B-A3B 4-bit), which the card cannot read;
//!   * the window: a pinned, device-mapped region of about a GiB and a half that the host fills with the experts a layer needs and the card
//!     reads in place over the link (an expert served from here is a miss);
//!   * cache lines of device memory: the hot experts, admitted by SDMA copies from the window (a hit reads VRAM).
//! The routed GEMVs take a pointer a (projection, expert) from a table (kernels/hip/tiles/plan.hpp, caps `expert_ptrs`): the card learns where an
//! expert sits one layer at a time, so no expert is read before its bytes are in place and the arithmetic is the checkpoint's wherever an expert sits.
//!
//! One blob holds an expert's six tensors in kernel order: the fused (gate | up) projection's words, scales and biases, then down's. Table
//! entry (layer, e) is the device address of expert e's blob, in a cache line or in a window slot; the shared expert (id `experts`) has a
//! line of its own that never leaves.
//!
//! A layer's stage, in the stream (the decode rounds and prompt spans of an offloaded model run eager, not as a graph: stream value waits
//! cannot be captured), is a *job*, numbered in posting order. A prompt span's stage (and a decode forward of more than `split_rows` rows) keeps the
//! plain handshake:
//!   router, pick, plan -> `tf_expert_trace` (the picks to the window) -> write `req` = n -> wait `ack` >= n -> copy the table row to the card's
//!   and on the host a worker thread spins on `req`, reads the layer's picks, lets the policy of core/expert_offload.zig (the CUDA EXL3 host
//!   cache's rules) decide hits and admissions, copies the blobs of the experts that are not in a cache line into window slots, writes their
//!   table entries, then stores `ack`. Two halves of the window alternate by layer, so a layer's slots are never rewritten while its GEMVs run.
//! A decode stage is split by what the card finds out for itself (docs/LATENCY.md M1): `tf_expert_check` copies the picks to the layer's own trace
//! slot and looks each one up in `resident`, the table of which experts sit in a cache line (a word a key, kept by the worker through
//! `Cache.resident`). All resident: it writes the layer's table row in device memory and stores the ack itself, so the stream's wait is satisfied
//! at once and the host is not asked (a *hit* job); the worker replays it later, touch only, from the trace slot. Otherwise it asks (`req`), the worker
//! serves the job as above, and `tf_expert_rows` after the wait moves the host's table row into the device's (a *miss* job). The worker serves jobs
//! in order, so before a miss it replays the hits that came before it and the pool is in the same state as if every layer had asked.
//! The miss is answered as soon as the experts' places are decided (M3): the window slots are filled after the ack and flag each slot when its blob is in;
//! a GEMV block whose expert sits in a window slot waits for that flag before it reads a byte (plan.hpp `expert_blob_ready`).
//! The ack word is host memory, or (M6) a signal word in device memory that the host stores through BAR0 and the command processor polls locally.
//!   * `beginRound`: the worker has answered every job posted so far;
//!   * `endRound`: the same again, and the round's admission budget is renewed.
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
const Ops = @import("ops/ops.zig").Ops;
const ItemSplit = @import("launches/moe.zig").ItemSplit;

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

    /// Bytes of one layer's table in either memory: a 64-byte header (plan.hpp), then an entry of two u64 an expert, the shared one last.
    fn rowStride(s: Shape) usize {
        return std.mem.alignForward(usize, header_bytes + (s.experts + 1) * 16, 64);
    }
};

/// What a layer's views need: its table, where its projections sit in a blob, the stage hook, and an address the stub tensors name.
pub const Ref = struct { ptab: u64, fused_off: u64, down_off: u64, anchor: u64, hook: moe.Hook };

pub const Stats = struct {
    rounds: u64 = 0,
    accesses: u64 = 0,
    hits: u64 = 0,
    /// Misses given a cache line (their GEMVs write what they read there).
    admissions: u64 = 0,
    /// Bytes of prompt-span experts the copy engine moved window -> scratch.
    scratch_bytes: u64 = 0,
    /// Rows decoded, and the wall nanoseconds of the rounds (forward, readback and draw included).
    tokens: u64 = 0,
    round_ns: u64 = 0,
    /// Layer stages the card asked for (decode and prompt), the bytes the host copied into the window, and the nanoseconds the worker spent from a request to its ack.
    stages: u64 = 0,
    window_bytes: u64 = 0,
    stage_ns: u64 = 0,
    /// Nanoseconds of the stages' waits for the card's request (the card computing; not stage work).
    idle_ns: u64 = 0,
    /// Decode stages the card answered by itself (every pick resident), and the nanoseconds the worker spent replaying them.
    hit_stages: u64 = 0,
    replay_ns: u64 = 0,
    /// Nanoseconds the worker spent filling window slots after it had answered (the card reads them as they flag).
    fill_ns: u64 = 0,
};

/// An `interval` log line every this many rounds.
const report_every: u64 = 256;

/// Device memory is allocated in pieces of this many bytes at most: one physically contiguous gigabytes-long block may not exist.
const chunk_bytes: usize = 512 << 20;

/// Layer stages that may be posted and not yet served (a forward posts one per layer; a prompt's spans queue several forwards).
const ring: u32 = 4096;

/// The most rows of a decode stage that the card checks itself: each layer keeps a trace slot of this many rows. Past it a stage keeps the plain handshake.
pub const split_rows: usize = 16;

/// Job slots of TF_EXPERT_VERIFY's snapshot ring (a slot is 64 bytes and 64 a pick of `split_rows` rows).
const vring: u32 = 256;

/// One of the two arrays the hit-first check writes: an entry (expert, first pair, pairs) of three i32 for the most items a stage of `split_rows` rows plans
/// (at most 9/8 of its pairs for each, bounded by twice them).
fn itemBytes(shape: Shape) usize {
    return std.mem.alignForward(usize, 2 * split_rows * shape.picks * 12, 256);
}

/// One of the arrays the chunked staging writes: the plan's entries for the most items a prompt span plans (its rows' next power of two times the picks, as `moe.capacity` has it).
fn chunkItemBytes(shape: Shape) usize {
    const rows = std.math.ceilPowerOfTwo(usize, shape.trace_rows) catch shape.trace_rows;
    const pairs = rows * shape.picks;
    return std.mem.alignForward(usize, (@min(pairs, shape.experts + 1) + pairs / 8 + 8) * 12, 256);
}

/// TF_EXPERT_PREFILL_CHUNKS: the chunks a prompt span's layer is staged in (2 to `max_chunks`), 0 when unset or out of range.
fn chunksWanted() usize {
    const v = platform.getenv("TF_EXPERT_PREFILL_CHUNKS") orelse return 0;
    const k = std.fmt.parseInt(usize, v, 10) catch return 0;
    return if (k >= 2 and k <= max_chunks) k else 0;
}

/// The chunk of the `i`-th of `total` experts a stage copies: 1 to `k`, each chunk the same number of experts give or take one.
fn chunkOf(i: usize, total: usize, k: usize) usize {
    return 1 + i * k / total;
}

/// The header of each layer's table (plan.hpp): flags address, first slot, bytes of the slots, bytes of a slot.
const header_bytes: usize = 64;

/// hipStreamWaitValue32's greater-or-equal compare.
const wait_gte: c_uint = 0;

const WaitValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint, u32) callconv(.c) abi.Result;
const WriteValue32 = *const fn (abi.Stream, ?*anyopaque, u32, c_uint) callconv(.c) abi.Result;

/// hipmac's VRAM signal words (docs/LATENCY.md C3; hipmacSignalAlloc/Store/Free of the runtime, absent from a runtime without the ut4g backend).
const Signal = extern struct { gpu_va: u64, bar0_off: u64, nwords: u32, stride: u32 };
const SignalAlloc = *const fn (u32, *Signal) callconv(.c) abi.Result;
const SignalStore = *const fn (*const Signal, u32, u32) callconv(.c) abi.Result;
const SignalFree = *const fn (*Signal) callconv(.c) abi.Result;

const SignalWord = struct { s: Signal, store: SignalStore, free: SignalFree };

/// Where the host's answer to a request is stored and the stream's wait polls it: a host word in the window, or a signal word in device memory.
const Ack = struct {
    host: *u32,
    host_dev: u64,
    /// The window's first byte: a prompt span's chunk words are host words after the control ones when the ack is not a signal word.
    base: [*]u8,
    signal: ?SignalWord = null,

    /// The address the stream waits on and the check kernel stores to.
    fn deviceWord(a: *const Ack) u64 {
        return if (a.signal) |g| g.s.gpu_va else a.host_dev;
    }

    fn waitPtr(a: *const Ack) ?*anyopaque {
        return a.waitPtrAt(0);
    }

    /// Word `idx` (0 the stage's ack, c the arrival of chunk c of a prompt span's stage).
    fn waitPtrAt(a: *const Ack, idx: usize) ?*anyopaque {
        if (a.signal) |g| return @ptrFromInt(g.s.gpu_va + idx * g.s.stride);
        return if (idx == 0) a.host else @ptrCast(a.base + chunk_word_at + idx * 64);
    }

    /// The host answers job `n`: everything it stored before is visible to the card once its wait is released.
    fn publish(a: *const Ack, n: u32) bool {
        return a.publishAt(0, n);
    }

    fn publishAt(a: *const Ack, idx: usize, n: u32) bool {
        if (a.signal) |g| return g.store(&g.s, @intCast(idx), n) == 0;
        @atomicStore(u32, @as(*u32, @ptrCast(@alignCast(a.waitPtrAt(idx)))), n, .release);
        return true;
    }
};

/// A prompt span's layer may be staged in this many chunks at most (words after `ack_at`'s in the window, or in the signal block).
pub const max_chunks: usize = 8;
/// The first chunk word of the host ack (chunk c at this + 64 c; the control words are below).
const chunk_word_at: usize = 192;

/// A window slot to fill after the answer: the slot, and the expert whose blob goes there.
const Fill = struct { slot: u32, expert: u32 };

const Job = struct {
    layer: u32,
    rows: u32,
    prefill: bool,
    /// The span's stage is answered in chunks (`Chunked` of the hook): the first answer names each expert's chunk, each chunk then has its own word.
    chunked: bool = false,
    /// The card checked this stage itself (its picks are in the layer's trace slot, and it may have passed the stage without asking).
    split: bool = false,
};

/// The cache lines: device memory in chunks and the policy's bookkeeping.
const Pool = struct {
    chunks: []DeviceBuffer,
    per_chunk: usize,
    cache: policy.Cache,
    line_addr: []u64,
    /// A prompt span's staging: one layer's every routed expert in device memory (SDMA copies from the window, which a stage waits for before it answers).
    scratch: DeviceBuffer,
    copy: Stream,
    /// Chunked staging: one event a chunk (index c - 1), recorded on `copy` after the chunk's uploads.
    events: [max_chunks]?Event = @splat(null),
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    r: *const Runtime,
    shape: Shape,
    /// Per layer: every routed expert's blob, pageable host memory (null until the layer is added).
    host: []?[]u8,
    /// The shared experts, one blob a layer.
    shared: DeviceBuffer,
    /// The window: control words, the host's tables, the prompt trace, the residency table, the slot flags, the decode traces, then the blob slots.
    win: HostBuffer,
    win_dev: u64,
    ptab_off: usize,
    trace_off: usize,
    resident_off: usize,
    dtrace_off: usize,
    flags_off: usize,
    /// A prompt span's chunk of each expert of the layer being staged (u32 an expert, written before the first answer).
    cmap_off: usize,
    slots_off: usize,
    nslots: usize,
    /// The tables in device memory the GEMVs read (layers x row stride), then a word a layer for the check's verdict; null: the plain handshake only.
    rows_dev: ?DeviceBuffer,
    /// The card checks decode stages itself (`rows_dev` is there).
    split: bool,
    /// Window slots flag when filled and a stage is answered before they are (the header of the tables names the flags).
    early: bool,
    /// Hit-first (M2): the card's check splits the plan's items into two arrays in `items_dev` (TF_EXPERT_HITFIRST=1), and the stream runs the hit items' GEMVs before its wait.
    hitfirst: bool = false,
    items_dev: ?DeviceBuffer = null,
    /// TF_EXPERT_VERIFY=1 (with the split, without hit-first): after each checked stage's table row is final the stream runs `tf_expert_snap`, which records per pick what the
    /// GEMVs will read (the card's residency word, the table entry, the blob's ends at its address) in the host-mapped ring `vsnap`; the worker compares a hit stage's record with its own state.
    verify: bool = false,
    vsnap: ?HostBuffer = null,
    vsnap_dev: u64 = 0,
    vfy: struct { picks: u64 = 0, bad_resident: u64 = 0, bad_rows: u64 = 0, bad_data: u64 = 0, late: u64 = 0, miss_picks: u64 = 0, miss_bad_rows: u64 = 0, logged: u32 = 0 } = .{},
    /// Chunked staging of a prompt span (F1, TF_EXPERT_PREFILL_CHUNKS=K, 2 to `max_chunks`): the plan's items in `nchunks` = K + 1 arrays of `chunk_stride` bytes (array 0 the
    /// experts already in a line), in `chunk_items`; 0 chunks: a span's stage answers once, after every copy.
    chunk_items: ?DeviceBuffer = null,
    nchunks: usize = 0,
    chunk_stride: usize = 0,
    ack: Ack,
    /// Slot states: 0 free, 1 holds a layer's expert this stage, 2 is the source of an admission in flight.
    used: []u8,
    cursor: [2]usize = .{ 0, 0 },
    wait32: WaitValue32,
    write32: WriteValue32,
    pool: ?Pool = null,
    span: Span = .{},
    stats: Stats = .{},
    since: Stats = .{},
    layers_added: usize = 0,
    /// The host's scratch for one stage: the layer's experts seen, and each one's window slot.
    mark: []u32,
    mark_serial: u32 = 0,
    slot_of: []i32,
    fills: []Fill,
    /// The host-tier blobs of the layer whose stage was answered before its slots were filled.
    fill_layer_host: ?[]u8 = null,
    nfills: usize = 0,
    jobs: []Job,
    /// Jobs posted (main thread only), published to the worker, and served.
    posted: u32 = 0,
    published: std.atomic.Value(u32) = .init(0),
    done: std.atomic.Value(u32) = .init(0),
    stop: std.atomic.Value(bool) = .init(false),
    failed: std.atomic.Value(bool) = .init(false),
    worker: ?std.Thread = null,

    const ctl_bytes: usize = 4096;
    /// The last job the card asked for; the last it passed by itself; the host ack word (when the ack is not a signal word).
    const req_at: usize = 0;
    const ack_at: usize = 64;
    const seen_at: usize = 128;

    /// The window's head layout for `shape` with room for `nslots` slot flags: where each part starts.
    const Layout = struct { ptab: usize, trace: usize, resident: usize, dtrace: usize, flags: usize, cmap: usize, slots: usize };

    fn layoutOf(shape: Shape, nslots: usize) Layout {
        const page = 4096;
        const ptab = ctl_bytes;
        const trace = ptab + std.mem.alignForward(usize, shape.layers * shape.rowStride(), page);
        const resident = trace + std.mem.alignForward(usize, shape.trace_rows * shape.picks * 4, page);
        const dtrace = resident + std.mem.alignForward(usize, shape.layers * shape.experts * 8, page);
        const flags = dtrace + std.mem.alignForward(usize, shape.layers * split_rows * shape.picks * 4, page);
        const cmap = flags + std.mem.alignForward(usize, nslots * 4, page);
        return .{ .ptab = ptab, .trace = trace, .resident = resident, .dtrace = dtrace, .flags = flags, .cmap = cmap, .slots = cmap + std.mem.alignForward(usize, (shape.experts + 1) * 4, page) };
    }

    /// Bytes of the window's head (control, tables, traces, residency, flags) for `shape`, with the flags of up to `nslots` slots.
    fn headBytes(shape: Shape, nslots: usize) usize {
        return layoutOf(shape, nslots).slots;
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
        const least = 2 * (shape.experts + 8);
        var want = window_bytes;
        var win: HostBuffer = undefined;
        var nslots: usize = 0;
        var layout: Layout = undefined;
        while (true) {
            // the flags are sized for the slots before the flags are subtracted: at most one page more than needed
            const first = (want -| headBytes(shape, 0)) / blob / 2 * 2;
            layout = layoutOf(shape, first);
            nslots = (want -| layout.slots) / blob / 2 * 2;
            if (nslots < least) return error.BadShape;
            if (HostBuffer.allocMapped(r, layout.slots + nslots * blob)) |w| {
                win = w;
                break;
            } else |err| {
                if (want / 4 * 3 < headBytes(shape, 0) + least * blob) return err;
                want = want / 4 * 3;
            }
        }
        errdefer win.free();
        @memset(win.bytes[0..layout.slots], 0);
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
        const fills = try gpa.alloc(Fill, shape.experts);
        errdefer gpa.free(fills);
        const jobs = try gpa.alloc(Job, ring);
        errdefer gpa.free(jobs);
        const win_dev = try win.device();
        const stride = shape.rowStride();
        // the card checks decode stages itself (M1) only with TF_EXPERT_SPLIT=1: default is the plain handshake every layer (pre-L9 behaviour), until the card gates pass
        const want_split = envOn("TF_EXPERT_SPLIT");
        var rows_dev: ?DeviceBuffer = null;
        if (want_split) {
            if (DeviceBuffer.alloc(r, shape.layers * stride + shape.layers * 64)) |b| {
                rows_dev = b;
            } else |_| std.log.warn("experts: no device memory for the layers' tables: every stage keeps the plain handshake", .{});
        }
        errdefer if (rows_dev) |*b| b.free();
        var items_dev: ?DeviceBuffer = null;
        if (rows_dev != null and envOn("TF_EXPERT_HITFIRST")) {
            if (DeviceBuffer.alloc(r, 2 * itemBytes(shape))) |b| {
                items_dev = b;
            } else |_| std.log.warn("experts: no device memory for the hit-first item arrays: stages run the plain way", .{});
        }
        errdefer if (items_dev) |*b| b.free();
        // chunked staging of prompt spans (F1): off unless TF_EXPERT_PREFILL_CHUNKS names a chunk count
        const chunk_k: usize = if (rows_dev != null) chunksWanted() else 0;
        var chunk_items: ?DeviceBuffer = null;
        if (chunk_k > 0) {
            if (DeviceBuffer.alloc(r, (chunk_k + 1) * chunkItemBytes(shape))) |b| {
                chunk_items = b;
            } else |_| std.log.warn("experts: no device memory for the chunked staging's item arrays: prompt spans answer once", .{});
        }
        errdefer if (chunk_items) |*b| b.free();
        var vsnap: ?HostBuffer = null;
        var vsnap_dev: u64 = 0;
        if (rows_dev != null and items_dev == null and envOn("TF_EXPERT_VERIFY")) {
            if (HostBuffer.allocMapped(r, vring * (64 + split_rows * shape.picks * 64))) |b| {
                if (b.device()) |d| {
                    @memset(b.bytes, 0);
                    vsnap = b;
                    vsnap_dev = d;
                } else |_| {
                    var f = b;
                    f.free();
                }
            } else |_| {}
        }
        errdefer if (vsnap) |*b| b.free();
        var ack: Ack = .{ .host = @ptrCast(@alignCast(win.bytes[ack_at..].ptr)), .host_dev = win_dev + ack_at, .base = win.bytes.ptr };
        // the ack is a host word in the window; a VRAM signal word (docs/LATENCY.md C3) only with TF_EXPERT_ACK=vram, which needs hipmac 07b7e35 or later (its first
        // signal pages hung the card); validated on the card for the release leg, so opt-in until it is measured on the 35B
        const ack_vram = if (platform.getenv("TF_EXPERT_ACK")) |mode| std.mem.eql(u8, mode, "vram") else false;
        if (ack_vram) ack.signal = openSignal(r, if (chunk_items != null) 1 + max_chunks else 1);
        errdefer if (ack.signal) |*g| {
            _ = g.free(&g.s);
        };
        var s: Store = .{
            .gpa = gpa,
            .r = r,
            .shape = shape,
            .host = host,
            .shared = shared,
            .win = win,
            .win_dev = win_dev,
            .ptab_off = layout.ptab,
            .trace_off = layout.trace,
            .resident_off = layout.resident,
            .dtrace_off = layout.dtrace,
            .flags_off = layout.flags,
            .cmap_off = layout.cmap,
            .slots_off = layout.slots,
            .nslots = nslots,
            .rows_dev = rows_dev,
            .split = rows_dev != null,
            .early = envOn("TF_EXPERT_FLAGS"),
            .hitfirst = items_dev != null,
            .items_dev = items_dev,
            .verify = vsnap != null,
            .vsnap = vsnap,
            .vsnap_dev = vsnap_dev,
            .chunk_items = chunk_items,
            .nchunks = if (chunk_items != null) chunk_k + 1 else 0,
            .chunk_stride = chunkItemBytes(shape),
            .ack = ack,
            .used = used,
            .wait32 = wait32,
            .write32 = write32,
            .mark = mark,
            .slot_of = slot_of,
            .fills = fills,
            .jobs = jobs,
        };
        for (0..shape.layers) |l| {
            const head = s.hostHeader(l);
            head[0] = if (s.early) win_dev + layout.flags else 0;
            head[1] = win_dev + layout.slots;
            head[2] = nslots * blob;
            head[3] = blob;
            const row = s.hostRow(l);
            for (0..shape.experts) |x| row[2 * x ..][0..2].* = .{ win_dev + layout.slots, 0 };
            row[2 * shape.experts ..][0..2].* = .{ shared.base() + l * blob, 0 };
        }
        // the tables in device memory start as the host's (the shared experts' entries never change), the verdict words at zero
        if (s.rows_dev) |*b| {
            const image = try gpa.alloc(u8, b.len);
            defer gpa.free(image);
            @memset(image, 0);
            @memcpy(image[0 .. shape.layers * stride], s.win.bytes[layout.ptab..][0 .. shape.layers * stride]);
            try b.upload(0, image);
        }
        return s;
    }

    /// Whether the environment variable `name` is set to 1.
    fn envOn(name: [:0]const u8) bool {
        const v = platform.getenv(name) orelse return false;
        return std.mem.eql(u8, v, "1");
    }

    fn openSignal(r: *const Runtime, nwords: u32) ?SignalWord {
        const lib: *@TypeOf(r.lib) = @constCast(&r.lib);
        const alloc = lib.lookup(SignalAlloc, "hipmacSignalAlloc");
        const store = lib.lookup(SignalStore, "hipmacSignalStore");
        const free = lib.lookup(SignalFree, "hipmacSignalFree");
        if (alloc == null or store == null or free == null) {
            std.log.info("experts: the runtime has no signal words: the ack stays in host memory", .{});
            return null;
        }
        var sig: Signal = undefined;
        if (alloc.?(nwords, &sig) != 0) {
            std.log.warn("experts: no signal word could be allocated: the ack stays in host memory", .{});
            return null;
        }
        return .{ .s = sig, .store = store.?, .free = free.? };
    }

    pub fn deinit(s: *Store) void {
        s.report("total");
        s.stopWorker();
        if (s.pool) |*p| {
            p.copy.deinit();
            for (&p.events) |*ev| if (ev.*) |*x| x.deinit();
            p.scratch.free();
            for (p.chunks) |*c| c.free();
            s.gpa.free(p.chunks);
            s.gpa.free(p.line_addr);
            p.cache.deinit();
        }
        if (s.ack.signal) |*g| _ = g.free(&g.s);
        if (s.rows_dev) |*b| b.free();
        if (s.items_dev) |*b| b.free();
        if (s.chunk_items) |*b| b.free();
        if (s.vsnap) |*b| b.free();
        s.gpa.free(s.jobs);
        s.gpa.free(s.fills);
        s.gpa.free(s.slot_of);
        s.gpa.free(s.mark);
        s.gpa.free(s.used);
        s.shared.free();
        s.win.free();
        for (s.host) |h| if (h) |bytes| std.heap.page_allocator.free(bytes);
        s.gpa.free(s.host);
        s.* = undefined;
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

    /// The addresses layer `layer`'s views carry: its table is the one in device memory when the card checks stages itself, else the host's.
    pub fn ref(s: *Store, layer: usize) Ref {
        const stride = s.shape.rowStride();
        const table = if (s.rows_dev) |b| b.base() + layer * stride else s.win_dev + s.ptab_off + layer * stride;
        return .{
            .ptab = table + header_bytes,
            .fused_off = 0,
            .down_off = s.shape.downOffset(),
            .anchor = s.win_dev,
            .hook = .{ .ctx = s, .post = post, .layer = @intCast(layer), .trace = s.win_dev + s.trace_off, .first = if (s.items_dev) |b| .{
                .hit_items = b.base(),
                .miss_items = b.base() + itemBytes(s.shape),
                .begin = begin,
                .finish = finish,
                .off = hitFirstOff,
            } else null, .chunked = if (s.chunk_items) |b| .{
                .items = b.base(),
                .stride = s.chunk_stride,
                .chunks = s.nchunks,
                .begin = chunkBegin,
                .wait = chunkWait,
            } else null },
        };
    }

    /// Layer `layer`'s table in the window: entries of two u64 an expert (the blob's address, the admission line or 0), the shared expert's last.
    fn hostRow(s: *const Store, layer: usize) []u64 {
        const at = s.ptab_off + layer * s.shape.rowStride() + header_bytes;
        const bytes = s.win.bytes[at..][0 .. (s.shape.experts + 1) * 16];
        return @alignCast(std.mem.bytesAsSlice(u64, bytes));
    }

    fn hostHeader(s: *const Store, layer: usize) []u64 {
        const at = s.ptab_off + layer * s.shape.rowStride();
        return @alignCast(std.mem.bytesAsSlice(u64, s.win.bytes[at..][0..header_bytes]));
    }

    /// The prompt trace: a forward's rows' picks (the plain handshake).
    fn traceWords(s: *const Store) []i32 {
        const bytes = s.win.bytes[s.trace_off..s.resident_off];
        return @alignCast(std.mem.bytesAsSlice(i32, bytes[0 .. bytes.len / 4 * 4]));
    }

    /// Layer `layer`'s decode trace slot: the picks of the stage the card checked, until the host has served or replayed it.
    fn decodeTrace(s: *const Store, layer: usize) []i32 {
        const n = split_rows * s.shape.picks;
        return @alignCast(std.mem.bytesAsSlice(i32, s.win.bytes[s.dtrace_off + layer * n * 4 ..][0 .. n * 4]));
    }

    /// The residency table: a word a key (layer * experts + expert), the address of the cache line the expert sits in, or 0.
    fn residentWords(s: *const Store) []u64 {
        const n = s.shape.layers * s.shape.experts;
        return @alignCast(std.mem.bytesAsSlice(u64, s.win.bytes[s.resident_off..][0 .. n * 8]));
    }

    /// A flag a window slot: 0 until the host has copied the slot's blob.
    fn flagWords(s: *const Store) []u32 {
        return @alignCast(std.mem.bytesAsSlice(u32, s.win.bytes[s.flags_off..][0 .. s.nslots * 4]));
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

    /// `total` cache lines of device memory, a prompt span's scratch (one layer's every expert) among them; decode admits at most `copy_experts` misses a round;
    /// `protect_pct` percent of the lines are an SLRU's protected segment (0: plain LRU).
    pub fn attachPool(s: *Store, total: usize, copy_experts: usize, protect_pct: u8) Error!void {
        std.debug.assert(s.pool == null);
        const blob = s.shape.blobBytes();
        const keys = s.shape.layers * s.shape.experts;
        if (total <= 2 * s.shape.experts) return error.NotSized;
        const lines = total - s.shape.experts;
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
        const line_addr = try s.gpa.alloc(u64, lines);
        errdefer s.gpa.free(line_addr);
        for (line_addr, 0..) |*a, i| a.* = chunks[i / per_chunk].base() + (i % per_chunk) * blob;
        var scratch = try DeviceBuffer.alloc(s.r, s.shape.experts * blob);
        errdefer scratch.free();
        var copy = try Stream.init(s.r);
        errdefer copy.deinit();
        var events: [max_chunks]?Event = @splat(null);
        errdefer for (&events) |*ev| if (ev.*) |*x| x.deinit();
        if (s.nchunks > 1) for (events[0 .. s.nchunks - 1]) |*ev| {
            ev.* = try Event.init(s.r, false);
        };
        const cache = try policy.Cache.init(s.gpa, .{
            .keys = keys,
            .lines = lines,
            .picks_max = s.shape.trace_rows * (s.shape.picks - 1),
            .copy_experts = copy_experts,
            .expert_bytes = blob,
            .protect_pct = protect_pct,
        }, s.residentWords(), line_addr);
        s.pool = .{ .chunks = chunks, .per_chunk = per_chunk, .cache = cache, .line_addr = line_addr, .scratch = scratch, .copy = copy, .events = events };
    }

    pub fn lineCount(s: *const Store) usize {
        return if (s.pool) |p| p.cache.slots.len else 0;
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

    /// Posts job `n` for the stage the stream is about to run: the worker may see it from here. Returns its number.
    fn announce(s: *Store, job: Job) u32 {
        while (s.posted -% s.done.load(.acquire) >= ring - 1) std.atomic.spinLoopHint();
        s.posted +%= 1;
        s.jobs[s.posted % ring] = job;
        s.published.store(s.posted, .release);
        return s.posted;
    }

    /// A layer's stage in the stream, after its picks were chosen. The plain handshake: the picks go to the prompt trace, the card asks (`req`), then waits for the
    /// host's answer (`ack`). A decode stage the card checks itself: `tf_expert_check`, the wait, `tf_expert_rows`.
    fn post(ctx: *anyopaque, o: Ops, layer: u32, picks: u64, pairs: usize, rows: usize, prefill: bool) moe.Error!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        // a dry run measuring scratch launches nothing and waits for no one
        if (o.stream == @import("launches/util.zig").counting) return;
        if (rows == 0 or rows > s.shape.trace_rows or s.worker == null) return error.BadShape;
        const checked = s.split and !prefill and rows <= split_rows;
        const n = s.announce(.{ .layer = layer, .rows = @intCast(rows), .prefill = prefill, .split = checked });
        const e = s.shape.experts;
        const stride = s.shape.rowStride();
        const picks_ptr: [*]const i32 = @ptrFromInt(picks);
        if (!s.split) {
            try o.expertTrace(picks, pairs, s.win_dev + s.trace_off);
            try runtime.check(s.write32(o.stream, s.win.bytes[req_at..].ptr, n, 0));
            try runtime.check(s.wait32(o.stream, s.ack.waitPtr(), n, wait_gte, 0xffff_ffff));
            return;
        }
        const table = s.rows_dev.?.base() + layer * stride + header_bytes;
        const need = s.rows_dev.?.base() + s.shape.layers * stride + layer * 64;
        const host_table = s.win_dev + s.ptab_off + layer * stride + header_bytes;
        if (checked) {
            try s.queueCheck(o, layer, picks, pairs, n, .{});
            try runtime.check(s.wait32(o.stream, s.ack.waitPtr(), n, wait_gte, 0xffff_ffff));
            try o.l.tf_expert_rows(picks_ptr, @intCast(pairs), @intCast(e), host_table, table, need, false, o.stream);
            if (s.verify) try o.l.tf_expert_snap(picks_ptr, @intCast(pairs), @intCast(e), s.win_dev + s.resident_off + layer * e * 8, table, s.vsnap_dev + @as(u64, n % vring) * (64 + split_rows * s.shape.picks * 64), n, @intCast(s.shape.blobBytes()), o.stream);
            return;
        }
        try o.expertTrace(picks, pairs, s.win_dev + s.trace_off);
        try runtime.check(s.write32(o.stream, s.win.bytes[req_at..].ptr, n, 0));
        try runtime.check(s.wait32(o.stream, s.ack.waitPtr(), n, wait_gte, 0xffff_ffff));
        try o.l.tf_expert_rows(picks_ptr, @intCast(pairs), @intCast(e), host_table, table, need, true, o.stream);
    }

    /// The check kernel of job `n` for `layer`, with the item split when `split` has one.
    fn queueCheck(s: *Store, o: Ops, layer: u32, picks: u64, pairs: usize, n: u32, split: ItemSplit) moe.Error!void {
        const e = s.shape.experts;
        const stride = s.shape.rowStride();
        const table = s.rows_dev.?.base() + layer * stride + header_bytes;
        const need = s.rows_dev.?.base() + s.shape.layers * stride + layer * 64;
        const resident = s.win_dev + s.resident_off + layer * e * 8;
        const trace = s.win_dev + s.dtrace_off + layer * split_rows * s.shape.picks * 4;
        try o.l.tf_expert_check(@ptrFromInt(picks), @intCast(pairs), @intCast(e), resident, table, trace, need, s.ack.deviceWord(), s.win_dev + seen_at, s.win_dev + req_at, n, split, o.stream);
    }

    /// Hit-first, first half (moe.HitFirst.begin): the job is announced and the check queued, with the plan's items split in two.
    fn begin(ctx: *anyopaque, o: Ops, layer: u32, picks: u64, pairs: usize, rows: usize, items: u64, count: usize) moe.Error!bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        if (o.stream == @import("launches/util.zig").counting) return false;
        const b = s.items_dev orelse return false;
        if (!s.hitfirst or !s.split or o.prefill or rows == 0 or rows > split_rows or rows > s.shape.trace_rows or s.worker == null) return false;
        if (count * 12 > itemBytes(s.shape)) return false;
        const n = s.announce(.{ .layer = layer, .rows = @intCast(rows), .prefill = false, .split = true });
        try s.queueCheck(o, layer, picks, pairs, n, .{ .items = items, .count = @intCast(count), .hit = b.base(), .miss = b.base() + itemBytes(s.shape) });
        return true;
    }

    /// Hit-first, second half: the stream's wait for the host's answer to the job `begin` announced, and the table row copy.
    fn finish(ctx: *anyopaque, o: Ops, layer: u32, picks: u64, pairs: usize) moe.Error!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        const n = s.posted;
        const stride = s.shape.rowStride();
        const table = s.rows_dev.?.base() + layer * stride + header_bytes;
        const need = s.rows_dev.?.base() + s.shape.layers * stride + layer * 64;
        const host_table = s.win_dev + s.ptab_off + layer * stride + header_bytes;
        try runtime.check(s.wait32(o.stream, s.ack.waitPtr(), n, wait_gte, 0xffff_ffff));
        try o.l.tf_expert_rows(@ptrFromInt(picks), @intCast(pairs), @intCast(s.shape.experts), host_table, table, need, false, o.stream);
    }

    /// The fused tile did not take the shape: hit-first is off from here (the stage that found out finishes the plain way).
    fn hitFirstOff(ctx: *anyopaque) void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        s.hitfirst = false;
    }

    /// Chunked staging, first half (moe.Chunked.begin): the job is announced, and the stream queued the trace and the request, the wait for the first answer, the table row copy
    /// and the split of the plan's items by the chunk each expert arrives in.
    fn chunkBegin(ctx: *anyopaque, o: Ops, layer: u32, picks: u64, pairs: usize, rows: usize, items: u64, count: usize) moe.Error!bool {
        const s: *Store = @ptrCast(@alignCast(ctx));
        if (o.stream == @import("launches/util.zig").counting) return false;
        const b = s.chunk_items orelse return false;
        if (s.pool == null or !o.prefill or rows == 0 or rows > s.shape.trace_rows or s.worker == null) return false;
        if (count * 12 > s.chunk_stride) return false;
        const n = s.announce(.{ .layer = layer, .rows = @intCast(rows), .prefill = true, .chunked = true });
        const e = s.shape.experts;
        const stride = s.shape.rowStride();
        const table = s.rows_dev.?.base() + layer * stride + header_bytes;
        const need = s.rows_dev.?.base() + s.shape.layers * stride + layer * 64;
        const host_table = s.win_dev + s.ptab_off + layer * stride + header_bytes;
        try o.expertTrace(picks, pairs, s.win_dev + s.trace_off);
        try runtime.check(s.write32(o.stream, s.win.bytes[req_at..].ptr, n, 0));
        try runtime.check(s.wait32(o.stream, s.ack.waitPtr(), n, wait_gte, 0xffff_ffff));
        try o.l.tf_expert_rows(@ptrFromInt(picks), @intCast(pairs), @intCast(e), host_table, table, need, true, o.stream);
        try o.l.tf_expert_chunks(@ptrFromInt(items), @intCast(count), @intCast(e), s.win_dev + s.cmap_off, @intCast(s.nchunks), @intCast(s.chunk_stride / 4), b.base(), o.stream);
        return true;
    }

    /// Chunked staging, second half: the stream's wait for chunk `c` of the job `chunkBegin` announced to have landed.
    fn chunkWait(ctx: *anyopaque, o: Ops, c: usize) moe.Error!void {
        const s: *Store = @ptrCast(@alignCast(ctx));
        try runtime.check(s.wait32(o.stream, s.ack.waitPtrAt(c), s.posted, wait_gte, 0xffff_ffff));
    }

    fn chunkWords(s: *Store) []u32 {
        return @as([*]u32, @ptrCast(@alignCast(s.win.bytes[s.cmap_off..].ptr)))[0 .. s.shape.experts + 1];
    }

    fn workerMain(s: *Store) void {
        // busy CPUs must not delay the answer to a request: the card is waiting for it
        _ = platform.c.pthread_set_qos_class_self_np(.USER_INTERACTIVE, 0);
        const req: *const u32 = @ptrCast(@alignCast(s.win.bytes[req_at..].ptr));
        const seen: *const u32 = @ptrCast(@alignCast(s.win.bytes[seen_at..].ptr));
        var waited: u64 = nowNs();
        while (!s.stop.load(.acquire)) {
            const n = s.done.load(.monotonic) +% 1;
            if (!policy.reached(s.published.load(.acquire), n)) {
                // nothing posted: the engine is between forwards
                napNs(50_000);
                waited = nowNs();
                continue;
            }
            // the card either asks once it has traced the picks of stage n, or passes it by itself
            const how = policy.classify(n, @atomicLoad(u32, seen, .acquire), @atomicLoad(u32, req, .acquire));
            if (how == .pending) {
                std.atomic.spinLoopHint();
                continue;
            }
            const began = nowNs();
            const job = s.jobs[n % ring];
            if (how == .hit) {
                s.replay(job, n);
                const ended = nowNs();
                s.stats.hit_stages += 1;
                s.stats.replay_ns += ended - began;
                s.done.store(n, .release);
                waited = nowNs();
                continue;
            }
            s.stats.idle_ns += began - waited;
            if (job.prefill and job.layer == 0) s.span = .{ .began = began, .bytes = s.stats.scratch_bytes };
            s.serve(job, n);
            const ended = nowNs();
            s.stats.stage_ns += ended - began;
            s.stats.stages += 1;
            if (job.prefill) {
                s.span.stage_ns += ended - began;
                if (job.layer + 1 == s.shape.layers) s.reportSpan(job.rows, ended);
            }
            if (s.verify and job.split) s.verifyMiss(job, n);
            s.done.store(n, .release);
            waited = nowNs();
        }
    }

    /// Stage `job`: decide, write the table row, answer, and (answered early) fill the window slots.
    fn serve(s: *Store, job: Job, n: u32) void {
        if (job.chunked) return s.serveChunked(job, n);
        // a prompt span's stage copies before it answers (its scratch copies read the slots)
        const early = s.early and !job.prefill;
        s.stage(job, early) catch s.failed.store(true, .release);
        if (!s.ack.publish(n)) s.failed.store(true, .release);
        if (early) {
            const began = nowNs();
            s.fill();
            s.stats.fill_ns += nowNs() - began;
        }
    }

    /// A prompt span's stage in chunks: whatever happens, every word the card waits for is released.
    fn serveChunked(s: *Store, job: Job, n: u32) void {
        s.stageChunked(job, n) catch {
            s.failed.store(true, .release);
            _ = s.ack.publish(n);
            for (1..s.nchunks) |c| _ = s.ack.publishAt(c, n);
        };
    }

    /// A prompt span's stage in chunks (F1): the experts the cache holds are chunk 0; each other expert gets its place in the scratch and its chunk (the table row and the
    /// chunk map), and the first answer (`ack`) goes out at once, before any copy; then the blobs go window slot -> scratch chunk by chunk on the copy stream, and the word of
    /// chunk c is stored when its uploads have landed, which is when the span's GEMMs for those experts may run.
    fn stageChunked(s: *Store, job: Job, n: u32) error{StageFailed}!void {
        const p = &(s.pool orelse return error.StageFailed);
        const e = s.shape.experts;
        const picks = s.shape.picks;
        const layer: usize = job.layer;
        const blob = s.shape.blobBytes();
        const k = s.nchunks - 1;
        const words = s.traceWords();
        const base: policy.Id = @intCast(layer * e);
        s.mark_serial +%= 1;
        if (s.mark_serial == 0) {
            @memset(s.mark, 0);
            s.mark_serial = 1;
        }
        var uniq: [4096]u32 = undefined;
        var nu: usize = 0;
        if (job.rows > s.shape.trace_rows) return error.StageFailed;
        for (0..job.rows) |r| for (words[r * picks ..][0 .. picks - 1]) |id| {
            if (id < 0 or id >= e) continue;
            const x: u32 = @intCast(id);
            if (s.mark[x] != s.mark_serial) {
                s.mark[x] = s.mark_serial;
                if (nu >= uniq.len) return error.StageFailed;
                uniq[nu] = x;
                nu += 1;
            }
        };
        // the layer's half of the window is free again (its previous layer's GEMMs have finished, which the request of this one is ordered after)
        const half = layer & 1;
        const per_half = s.nslots / 2;
        for (s.used[half * per_half ..][0..per_half]) |*u| {
            if (u.* == 1) u.* = 0;
        }
        const table = s.hostRow(layer);
        const host = s.host[layer] orelse return error.StageFailed;
        const cmap = s.chunkWords();
        var staged: usize = 0;
        for (uniq[0..nu]) |x| {
            if (p.cache.address(base + x) == null) staged += 1;
        }
        if (staged > s.fills.len) return error.StageFailed;
        var i: usize = 0;
        for (uniq[0..nu]) |x| {
            if (p.cache.address(base + x)) |line| {
                table[2 * x] = line;
                table[2 * x + 1] = 0;
                cmap[x] = 0;
                continue;
            }
            const w = s.takeSlot(half, per_half) orelse return error.StageFailed;
            s.fills[i] = .{ .slot = @intCast(w), .expert = x };
            table[2 * x] = p.scratch.base() + i * blob;
            table[2 * x + 1] = 0;
            cmap[x] = @intCast(chunkOf(i, staged, k));
            s.stats.window_bytes += blob;
            i += 1;
        }
        if (!s.ack.publish(n)) return error.StageFailed;
        // chunk by chunk; a chunk's word goes out once its event has passed, which is checked as the next chunk's copies are queued
        var pending: usize = 0;
        i = 0;
        for (1..k + 1) |c| {
            while (i < staged and chunkOf(i, staged, k) == c) : (i += 1) {
                const f = s.fills[i];
                @memcpy(s.slotBytes(f.slot), host[f.expert * blob ..][0..blob]);
                p.scratch.uploadAsync(i * blob, s.win.view(s.slots_off + f.slot * blob, s.slots_off + (f.slot + 1) * blob), p.copy) catch return error.StageFailed;
                if (pending != 0 and (p.events[pending - 1].?.done() catch return error.StageFailed)) {
                    if (!s.ack.publishAt(pending, n)) return error.StageFailed;
                    pending = 0;
                }
            }
            p.events[c - 1].?.record(p.copy) catch return error.StageFailed;
            if (pending != 0) {
                p.events[pending - 1].?.synchronize() catch return error.StageFailed;
                if (!s.ack.publishAt(pending, n)) return error.StageFailed;
            }
            pending = c;
        }
        if (pending != 0) {
            p.events[pending - 1].?.synchronize() catch return error.StageFailed;
            if (!s.ack.publishAt(pending, n)) return error.StageFailed;
        }
        s.stats.scratch_bytes += staged * blob;
    }

    /// A hit job, late: the card passed the stage by itself, so the pool only needs the touches of its picks (from the stage's trace slot).
    fn replay(s: *Store, job: Job, n_job: u32) void {
        if (!job.split) return s.failed.store(true, .release);
        const p = &(s.pool orelse return s.failed.store(true, .release));
        const e = s.shape.experts;
        const picks = s.shape.picks;
        const base: policy.Id = @intCast(@as(usize, job.layer) * e);
        const words = s.decodeTrace(job.layer);
        var n: usize = 0;
        for (0..job.rows) |r| for (words[r * picks ..][0 .. picks - 1]) |id| {
            if (id < 0 or id >= e) continue;
            p.cache.ids[n] = base + @as(policy.Id, @intCast(id));
            n += 1;
        };
        if (s.verify) s.verifyHit(job, n_job, p);
        p.cache.replay(p.cache.ids[0..n]) catch return s.failed.store(true, .release);
        s.stats.accesses += n;
        s.stats.hits += n;
    }

    /// TF_EXPERT_VERIFY: what the card's GEMVs read for a stage it passed. `tf_expert_snap` ran after the stage's table row was final and left, per pick, the residency word it
    /// loaded, the VRAM table entry, and the ends of the blob at the entry's address, in the host-mapped ring slot of job `n`. Compared with the host's state, which is the state
    /// the card saw (the host only changes the pool at the next miss, after this replay): the card's residency word (a stale one is the card reading host memory late), the entry
    /// (must be the line, with no admission line) and the blob's ends (must be the host tier's).
    fn verifyHit(s: *Store, job: Job, n: u32, p: *Pool) void {
        const slot = s.snapSlot(n) orelse return;
        const layer: usize = job.layer;
        const e = s.shape.experts;
        const blob = s.shape.blobBytes();
        const host = s.host[layer] orelse return;
        const pairs = std.mem.readInt(u32, slot[4..8], .little);
        for (0..pairs) |i| {
            const rec = slot[64 + 64 * i ..][0..64];
            const id: i64 = @bitCast(std.mem.readInt(u64, rec[0..8], .little));
            if (id < 0 or id >= @as(i64, @intCast(e))) continue;
            const x: usize = @intCast(id);
            const line = p.cache.resident[layer * e + x];
            const seen = std.mem.readInt(u64, rec[8..16], .little);
            const e0 = std.mem.readInt(u64, rec[16..24], .little);
            const e1 = std.mem.readInt(u64, rec[24..32], .little);
            const head_ok = std.mem.eql(u8, rec[32..48], host[x * blob ..][0..16]);
            const tail_ok = std.mem.eql(u8, rec[48..64], host[x * blob + blob - 16 ..][0..16]);
            s.vfy.picks += 1;
            const stale = seen != line;
            const bad_row = e0 != line or e1 != 0;
            const bad_data = !head_ok or !tail_ok;
            s.vfy.bad_resident += @intFromBool(stale);
            s.vfy.bad_rows += @intFromBool(bad_row);
            s.vfy.bad_data += @intFromBool(bad_data);
            if ((stale or bad_row or bad_data) and s.vfy.logged < 32) {
                s.vfy.logged += 1;
                std.log.err("experts verify: job {d} layer {d} expert {d}: host line 0x{x}, card's residency word 0x{x}, table entry 0x{x} 0x{x}, head {s}, tail {s}", .{ n, layer, x, line, seen, e0, e1, if (head_ok) "ok" else "WRONG", if (tail_ok) "ok" else "WRONG" });
            }
        }
    }

    /// TF_EXPERT_VERIFY, a stage the card asked about: after the host's answer the card's table entries for the stage's picks (the rows kernel's copy of the host's row, read
    /// over the link) must be the host's row's.
    fn verifyMiss(s: *Store, job: Job, n: u32) void {
        const slot = s.snapSlot(n) orelse return;
        const e = s.shape.experts;
        const row = s.hostRow(job.layer);
        const pairs = std.mem.readInt(u32, slot[4..8], .little);
        for (0..pairs) |i| {
            const rec = slot[64 + 64 * i ..][0..64];
            const id: i64 = @bitCast(std.mem.readInt(u64, rec[0..8], .little));
            if (id < 0 or id >= @as(i64, @intCast(e))) continue;
            const x: usize = @intCast(id);
            const e0 = std.mem.readInt(u64, rec[16..24], .little);
            const e1 = std.mem.readInt(u64, rec[24..32], .little);
            s.vfy.miss_picks += 1;
            if (e0 != row[2 * x] or e1 != row[2 * x + 1]) {
                s.vfy.miss_bad_rows += 1;
                if (s.vfy.logged < 32) {
                    s.vfy.logged += 1;
                    std.log.err("experts verify: miss job {d} layer {d} expert {d}: host row 0x{x} 0x{x}, card's table entry 0x{x} 0x{x}", .{ n, job.layer, x, row[2 * x], row[2 * x + 1], e0, e1 });
                }
            }
        }
    }

    /// The snapshot slot of job `n` once the card has written it (null: it did not within 200 ms, or the ring has wrapped over it).
    fn snapSlot(s: *Store, n: u32) ?[]const u8 {
        const snap = s.vsnap orelse return null;
        const slot_bytes = 64 + split_rows * s.shape.picks * 64;
        const slot = snap.bytes[(n % vring) * slot_bytes ..][0..slot_bytes];
        const tag: *const u32 = @ptrCast(@alignCast(slot.ptr));
        // the card writes the slot right after its wait on the answer is released
        const deadline = nowNs() + 200 * std.time.ns_per_ms;
        while (@atomicLoad(u32, tag, .acquire) != n) {
            if (nowNs() > deadline) break;
            std.atomic.spinLoopHint();
        }
        if (@atomicLoad(u32, tag, .acquire) != n) {
            s.vfy.late += 1;
            return null;
        }
        return slot;
    }

    /// One log line for a prompt span: its wall time from the first stage to the last answer, the time in stages, the experts copied to the scratch.
    fn reportSpan(s: *Store, rows: u32, ended: u64) void {
        const wall: f64 = @floatFromInt(ended - s.span.began);
        const staged_ns: f64 = @floatFromInt(s.span.stage_ns);
        const mib: f64 = @as(f64, @floatFromInt(s.stats.scratch_bytes - s.span.bytes)) / (1 << 20);
        std.log.info("experts span: {d} rows, {d:.2} s first stage to last answer, {d:.2} s in stages, {d:.0} MiB copied to the scratch ({d:.2} GB/s of the stages' time)", .{ rows, wall / 1e9, staged_ns / 1e9, mib, mib * 1.048576e6 / staged_ns });
    }

    /// A prompt span's counters.
    const Span = struct { began: u64 = 0, stage_ns: u64 = 0, bytes: u64 = 0 };

    /// The window slots a stage took and has not filled: the blobs go in ascending expert order (the order the plan's items, and so the GEMV blocks, ask for them),
    /// each flagged when it is in.
    fn fill(s: *Store) void {
        const n = s.nfills;
        s.nfills = 0;
        const list = s.fills[0..n];
        std.mem.sort(Fill, list, {}, struct {
            fn less(_: void, a: Fill, b: Fill) bool {
                return a.expert < b.expert;
            }
        }.less);
        const blob = s.shape.blobBytes();
        const flags = s.flagWords();
        for (list) |f| {
            const layer_host = s.fill_layer_host orelse return s.failed.store(true, .release);
            @memcpy(s.slotBytes(f.slot), layer_host[f.expert * blob ..][0..blob]);
            @atomicStore(u32, &flags[f.slot], 1, .release);
        }
        s.fill_layer_host = null;
    }

    fn stage(s: *Store, job: Job, early: bool) error{StageFailed}!void {
        const e = s.shape.experts;
        const picks = s.shape.picks;
        const layer: usize = job.layer;
        const blob = s.shape.blobBytes();
        const words = if (job.split) s.decodeTrace(layer) else s.traceWords();
        const base: policy.Id = @intCast(layer * e);
        const cache: ?*policy.Cache = if (s.pool) |*p| &p.cache else null;
        const decode_cache: ?*policy.Cache = if (job.prefill) null else cache;
        var n: usize = 0;
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
            if (decode_cache) |c| {
                c.ids[n] = base + x;
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
        if (decode_cache) |c| {
            const d = c.decide(n) catch return error.StageFailed;
            started = d.started;
            s.stats.accesses += n;
            s.stats.hits += d.hits;
        }
        // the layer's half of the window is free again: its previous layer's GEMVs have finished (they precede this layer's trace in the stream)
        const half = layer & 1;
        const per_half = s.nslots / 2;
        for (s.used[half * per_half ..][0..per_half]) |*u| {
            if (u.* == 1) u.* = 0;
        }
        const table = s.hostRow(layer);
        const host = s.host[layer] orelse return error.StageFailed;
        const flags = s.flagWords();
        s.nfills = 0;
        // a prompt span's experts are copied (SDMA, from their window slots) into the scratch and read there by the span's GEMMs: the tiles read a
        // window slot in place far below the link's speed, a copy runs at it
        const prefill_pool: ?*Pool = if (job.prefill and s.pool != null) &s.pool.? else null;
        var staged: usize = 0;
        for (uniq[0..nu]) |x| {
            const key: policy.Id = base + x;
            if (cache) |c| if (c.address(key)) |line| {
                table[2 * x] = line;
                table[2 * x + 1] = 0;
                continue;
            };
            const w = s.takeSlot(half, per_half) orelse return error.StageFailed;
            table[2 * x + 1] = 0;
            if (early) {
                // answered before the copy: the card waits for the flag of this slot
                @atomicStore(u32, &flags[w], 0, .release);
                s.fills[s.nfills] = .{ .slot = @intCast(w), .expert = x };
                s.nfills += 1;
                s.stats.window_bytes += blob;
                table[2 * x] = s.slotAddr(w);
                continue;
            }
            @memcpy(s.slotBytes(w), host[x * blob ..][0..blob]);
            @atomicStore(u32, &flags[w], 1, .release);
            s.stats.window_bytes += blob;
            if (prefill_pool) |p| {
                const at = staged * blob;
                p.scratch.uploadAsync(at, s.win.view(s.slots_off + w * blob, s.slots_off + (w + 1) * blob), p.copy) catch return error.StageFailed;
                table[2 * x] = p.scratch.base() + at;
                staged += 1;
                continue;
            }
            table[2 * x] = s.slotAddr(w);
        }
        if (early) s.fill_layer_host = host;
        if (staged > 0) {
            prefill_pool.?.copy.synchronize() catch return error.StageFailed;
            s.stats.scratch_bytes += staged * blob;
        }
        // read-through admission: the GEMVs of this layer that read an admitted miss in its window slot store what they read into the line the
        // policy gave it (table entry word 1); the line is resident for the next stage, which the stream orders after those GEMVs
        if (started > 0) {
            const c = decode_cache.?;
            for (c.copies[0..started]) |cp| table[2 * (cp.id - base) + 1] = c.line_addr[cp.slot];
            c.commit(started) catch return error.StageFailed;
            s.stats.admissions += started;
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

    /// A round's first stage may be served: the worker has answered every stage posted so far.
    pub fn beginRound(s: *Store) Error!void {
        s.quiesce();
    }

    /// The round over its first `rows` rows is finished: its time joins the counters and the admission budget is renewed.
    pub fn endRound(s: *Store, rows: usize, ns: u64) Error!void {
        s.quiesce();
        if (s.failed.load(.acquire)) return error.StageFailed;
        s.stats.rounds += 1;
        s.stats.tokens += rows;
        s.stats.round_ns += ns;
        if (s.stats.rounds % report_every == 0) s.report("interval");
        if (s.pool) |*p| p.cache.beginRound();
    }

    /// One log line for the rounds since the last (or since the start): hit rate, misses and window traffic a token, the stage time.
    /// A miss is a router pick read over the link from a window slot; the shared expert and the dense layers are not counted.
    pub fn report(s: *Store, label: []const u8) void {
        const d = Stats{
            .rounds = s.stats.rounds - s.since.rounds,
            .accesses = s.stats.accesses - s.since.accesses,
            .hits = s.stats.hits - s.since.hits,
            .admissions = s.stats.admissions - s.since.admissions,
            .tokens = s.stats.tokens - s.since.tokens,
            .round_ns = s.stats.round_ns - s.since.round_ns,
            .stages = s.stats.stages - s.since.stages,
            .window_bytes = s.stats.window_bytes - s.since.window_bytes,
            .stage_ns = s.stats.stage_ns - s.since.stage_ns,
            .idle_ns = s.stats.idle_ns - s.since.idle_ns,
            .hit_stages = s.stats.hit_stages - s.since.hit_stages,
            .replay_ns = s.stats.replay_ns - s.since.replay_ns,
            .fill_ns = s.stats.fill_ns - s.since.fill_ns,
        };
        s.since = s.stats;
        if (d.rounds == 0) return;
        const a: f64 = @floatFromInt(@max(d.accesses, 1));
        const t: f64 = @floatFromInt(@max(d.tokens, 1));
        const r: f64 = @floatFromInt(d.rounds);
        const miss: f64 = @floatFromInt(d.accesses - d.hits);
        const mib: f64 = 1024.0 * 1024.0;
        const blob: f64 = @floatFromInt(s.shape.blobBytes());
        const asked: f64 = @floatFromInt(d.stages + d.hit_stages);
        std.log.info(
            "experts {s}: {d} rounds {d} tokens, hit {d:.1}% miss {d:.1}/token ({d:.1} MiB/token over the link), window fill {d:.1} MiB/token, admissions {d:.1}/round, stage {d:.2} ms/token (card compute+wait {d:.2}; fill after the answer {d:.2}), card passed {d:.1}% of the stages itself ({d:.2} ms/token replaying), round {d:.2} ms ({d:.2} ms/token), {d} lines",
            .{ label, d.rounds, d.tokens, 100.0 * @as(f64, @floatFromInt(d.hits)) / a, miss / t, miss / t * blob / mib, @as(f64, @floatFromInt(d.window_bytes)) / t / mib, @as(f64, @floatFromInt(d.admissions)) / r, @as(f64, @floatFromInt(d.stage_ns)) / t / 1e6, @as(f64, @floatFromInt(d.idle_ns)) / t / 1e6, @as(f64, @floatFromInt(d.fill_ns)) / t / 1e6, 100.0 * @as(f64, @floatFromInt(d.hit_stages)) / @max(asked, 1), @as(f64, @floatFromInt(d.replay_ns)) / t / 1e6, @as(f64, @floatFromInt(d.round_ns)) / r / 1e6, @as(f64, @floatFromInt(d.round_ns)) / t / 1e6, s.lineCount() },
        );
        if (s.verify) std.log.info("experts verify: {d} hit picks checked, {d} with a residency word the card read differing from the host's, {d} with a wrong table entry, {d} with wrong blob ends; {d} miss picks checked, {d} whose table entry is not the host row's; {d} stages whose snapshot was missing", .{ s.vfy.picks, s.vfy.bad_resident, s.vfy.bad_rows, s.vfy.bad_data, s.vfy.miss_picks, s.vfy.miss_bad_rows, s.vfy.late });
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

/// A store with the card-side split and the slot flags on (both are opt-in in production: TF_EXPERT_SPLIT=1, TF_EXPERT_FLAGS=1); the environment is back to off after.
fn initOn(gpa: std.mem.Allocator, r: *const Runtime, shape: Shape) Error!Store {
    _ = c_setenv("TF_EXPERT_SPLIT", "1", 1);
    _ = c_setenv("TF_EXPERT_FLAGS", "1", 1);
    defer {
        _ = c_setenv("TF_EXPERT_SPLIT", "0", 1);
        _ = c_setenv("TF_EXPERT_FLAGS", "0", 1);
    }
    return Store.init(gpa, r, shape, 1 << 20);
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

test "stages fill window slots, admitted misses name their lines (the GEMVs write them), and the table names where each expert sits" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 2, .experts = 4, .picks = 3, .trace_rows = 2, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    const blob = shape.blobBytes();
    try std.testing.expectEqual(@as(usize, 160), blob);
    try std.testing.expectEqual(@as(usize, 96), shape.downOffset());
    var s = try initOn(gpa, &r, shape);
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
    try std.testing.expectEqual(s.shared.base() + blob, s.hostRow(1)[4 * 2]);
    try std.testing.expectEqual(@as(u64, 0), s.hostRow(1)[4 * 2 + 1]);
    try std.testing.expectError(error.BadShape, s.addLayer(2, t0));
    // each layer's table has its header before the entries (the slot flags, the first slot, the slots' bytes, a slot's bytes)
    const head = s.hostHeader(1);
    try std.testing.expectEqual(s.win_dev + s.slots_off, head[1]);
    try std.testing.expectEqual(s.nslots * blob, head[2]);
    try std.testing.expectEqual(blob, head[3]);

    // 12 lines of device memory: 4 of them (one layer's every expert) the prompt scratch
    try s.attachPool(12, 2, 0);
    try std.testing.expectEqual(@as(usize, 8), s.lineCount());
    // stage 1 (layer 0, decode): rows name 1 twice, 2 and 3. Two admissions fit the budget: the hottest (key 1) and the lowest id of the rest
    setPicks(&s, 0, &.{ 1, 2, 4 });
    setPicks(&s, 1, &.{ 1, 3, 4 });
    try s.stage(.{ .layer = 0, .rows = 2, .prefill = false }, false);
    try std.testing.expectEqual(@as(u64, 0), s.stats.hits);
    try std.testing.expectEqual(@as(u64, 2), s.stats.admissions);
    // the card reads every miss where it sits: all three experts have window slots with their bytes
    const slot1 = s.hostRow(0)[2];
    try std.testing.expect(slot1 >= s.win_dev + s.slots_off and slot1 < s.win_dev + s.win.bytes.len);
    blobOf(shape, t0, 1, &want);
    try std.testing.expectEqualSlices(u8, &want, s.slotBytes((slot1 - s.win_dev - s.slots_off) / blob));
    try std.testing.expectEqual(@as(u32, 1), s.flagWords()[(slot1 - s.win_dev - s.slots_off) / blob]);
    // the two admitted ones also name a line (their GEMVs store what they read there); the third does not
    var named: usize = 0;
    for (1..4) |x| named += @intFromBool(s.hostRow(0)[2 * x + 1] != 0);
    try std.testing.expectEqual(@as(usize, 2), named);
    const line_of_1 = s.hostRow(0)[3];
    try std.testing.expect(line_of_1 != 0 and (line_of_1 < s.win_dev + s.slots_off or line_of_1 >= s.win_dev + s.win.bytes.len));
    // the stores, as the stream tile makes them
    for (1..4) |x| {
        const dst = s.hostRow(0)[2 * x + 1];
        if (dst == 0) continue;
        const src = s.hostRow(0)[2 * x];
        const chunk = &s.pool.?.chunks[0];
        try chunk.upload(dst - chunk.base(), s.slotBytes((src - s.win_dev - s.slots_off) / blob));
    }
    try s.endRound(2, 1000);
    try std.testing.expect(policy.onDevice(s.pool.?.cache.vram, 1));
    // the residency table the card checks names the line of each admitted expert, and nothing else in this layer
    try std.testing.expectEqual(line_of_1, s.residentWords()[1]);
    try std.testing.expectEqual(@as(u64, 0), s.residentWords()[0]);
    // stage 2 (layer 0 again): the resident expert is a hit and its entry names its line, with the bytes the GEMVs stored
    setPicks(&s, 0, &.{ 1, 2, 4 });
    setPicks(&s, 1, &.{ 1, 2, 4 });
    try s.stage(.{ .layer = 0, .rows = 2, .prefill = false }, false);
    try std.testing.expect(s.stats.hits >= 2);
    const line = s.hostRow(0)[2];
    try std.testing.expect(line < s.win_dev + s.slots_off or line >= s.win_dev + s.win.bytes.len);
    try std.testing.expectEqual(@as(u64, 0), s.hostRow(0)[3]);
    blobOf(shape, t0, 1, &want);
    var dev: [160]u8 = undefined;
    try s.pool.?.chunks[0].download(line - s.pool.?.chunks[0].base(), &dev);
    try std.testing.expectEqualSlices(u8, &want, &dev);
    // a prompt span copies every named expert (the ones not resident) into the scratch, by the copy engine, before it answers, and admits nothing
    const before = s.stats.admissions;
    setPicks(&s, 0, &.{ 0, 3, 4 });
    setPicks(&s, 1, &.{ 2, 3, 4 });
    try s.stage(.{ .layer = 1, .rows = 2, .prefill = true }, false);
    try std.testing.expectEqual(before, s.stats.admissions);
    const e0 = s.hostRow(1)[0];
    const scratch = &s.pool.?.scratch;
    try std.testing.expect(e0 >= scratch.base() and e0 < scratch.base() + scratch.len);
    blobOf(shape, t1, 0, &want);
    try scratch.download(e0 - scratch.base(), &dev);
    try std.testing.expectEqualSlices(u8, &want, &dev);
    try std.testing.expectEqual(@as(u64, 3 * blob), s.stats.scratch_bytes);
}

/// The card's side of a decode stage, on the window's own words as tf_expert_check makes them: the picks into the layer's trace slot, each looked up in the
/// residency table, then either `seen` and the ack (a hit: nothing asks the host) or `req` and a wait for the ack (a miss). Returns whether the card passed it.
fn cardChecks(s: *Store, layer: usize, rows: usize, ids: []const i32) bool {
    const n = s.announce(.{ .layer = @intCast(layer), .rows = @intCast(rows), .prefill = false, .split = true });
    @memcpy(s.decodeTrace(layer)[0..ids.len], ids);
    var all = true;
    for (ids) |id| {
        if (id < 0 or id >= s.shape.experts) continue;
        if (@atomicLoad(u64, &s.residentWords()[layer * s.shape.experts + @as(usize, @intCast(id))], .acquire) == 0) all = false;
    }
    if (all) {
        @atomicStore(u32, s.word(Store.seen_at), n, .release);
        _ = s.ack.publish(n);
        return true;
    }
    @atomicStore(u32, s.word(Store.req_at), n, .release);
    while (!policy.reached(@atomicLoad(u32, s.word(Store.ack_at), .acquire), n)) std.atomic.spinLoopHint();
    return false;
}

test "the worker answers a stage the card could not check and replays one the card passed, in order, and the pool ends as if every stage had asked" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 3, .experts = 8, .picks = 3, .trace_rows = 4, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    const blob = shape.blobBytes();
    var s = try initOn(gpa, &r, shape);
    defer s.deinit();
    // the host answers into the window's own word here, whatever the runtime offers
    if (s.ack.signal) |*g| {
        _ = g.free(&g.s);
        s.ack.signal = null;
    }
    const t0 = try testTensors(gpa, shape, 1);
    const t1 = try testTensors(gpa, shape, 2);
    defer for (t0 ++ t1) |t| gpa.free(t);
    try s.addLayer(0, t0);
    try s.addLayer(1, t1);
    try s.addLayer(2, t0);
    try s.attachPool(24, 4, 0);
    try s.start();
    try s.beginRound();
    // round 1: nothing is resident, the card asks for layer 0 and layer 1
    try std.testing.expect(!cardChecks(&s, 0, 1, &.{ 1, 2, 8 }));
    // answered: the entries name window slots, and the slots fill after (flags rise, bytes are the blobs')
    const row0 = s.hostRow(0);
    for ([_]usize{ 1, 2 }) |x| {
        const at = (row0[2 * x] - s.win_dev - s.slots_off) / blob;
        while (@atomicLoad(u32, &s.flagWords()[at], .acquire) == 0) std.atomic.spinLoopHint();
        var want: [160]u8 = undefined;
        blobOf(shape, t0, x, &want);
        try std.testing.expectEqualSlices(u8, &want, s.slotBytes(at));
    }
    try std.testing.expect(!cardChecks(&s, 1, 1, &.{ 3, 4, 8 }));
    try std.testing.expect(!cardChecks(&s, 2, 1, &.{ 5, 6, 8 }));
    try s.endRound(1, 1000);
    try std.testing.expectEqual(@as(u64, 3), s.stats.stages);
    try std.testing.expectEqual(@as(u64, 0), s.stats.hit_stages);
    // both admitted pairs are resident, with the lines' addresses
    try std.testing.expect(s.residentWords()[1] != 0 and s.residentWords()[2] != 0);
    try std.testing.expect(s.residentWords()[8 + 3] != 0 and s.residentWords()[8 + 4] != 0);
    // round 2: layer 0 and layer 1 name resident experts only: the card passes both, the worker replays them
    try s.beginRound();
    try std.testing.expect(cardChecks(&s, 0, 1, &.{ 2, 1, 8 }));
    try std.testing.expect(cardChecks(&s, 1, 1, &.{ 3, 4, 8 }));
    // layer 2 asks (expert 7 is new) after the two hits: the worker replays them first, then serves it
    try std.testing.expect(!cardChecks(&s, 2, 1, &.{ 5, 7, 8 }));
    try s.endRound(1, 1000);
    try std.testing.expectEqual(@as(u64, 2), s.stats.hit_stages);
    try std.testing.expectEqual(@as(u64, 4), s.stats.stages);
    try std.testing.expect(!s.failed.load(.acquire));
    // the two late touches happened before the third stage chose anything: the pool has the same stamps as the stage-per-layer run
    var twin = try initOn(gpa, &r, shape);
    defer twin.deinit();
    try twin.addLayer(0, t0);
    try twin.addLayer(1, t1);
    try twin.addLayer(2, t0);
    try twin.attachPool(24, 4, 0);
    const stages = [_]struct { layer: usize, ids: [3]i32 }{
        .{ .layer = 0, .ids = .{ 1, 2, 8 } },
        .{ .layer = 1, .ids = .{ 3, 4, 8 } },
        .{ .layer = 2, .ids = .{ 5, 6, 8 } },
        .{ .layer = 0, .ids = .{ 2, 1, 8 } },
        .{ .layer = 1, .ids = .{ 3, 4, 8 } },
        .{ .layer = 2, .ids = .{ 5, 7, 8 } },
    };
    for (stages, 0..) |st, i| {
        if (i == 0 or i == 3) twin.pool.?.cache.beginRound();
        @memcpy(twin.traceWords()[0..3], &st.ids);
        try twin.stage(.{ .layer = @intCast(st.layer), .rows = 1, .prefill = false }, false);
    }
    const a = &s.pool.?.cache;
    const b = &twin.pool.?.cache;
    try std.testing.expectEqual(b.now, a.now);
    try std.testing.expectEqualSlices(u32, b.stamp, a.stamp);
    try std.testing.expectEqualSlices(i32, b.where, a.where);
}

test "a prompt span's stage in chunks: the first answer names each expert's chunk and place, then each chunk's word follows its copies, in order" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 2, .experts = 8, .picks = 4, .trace_rows = 4, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    const blob = shape.blobBytes();
    _ = c_setenv("TF_EXPERT_PREFILL_CHUNKS", "2", 1);
    const made = initOn(gpa, &r, shape);
    _ = c_setenv("TF_EXPERT_PREFILL_CHUNKS", "0", 1);
    var s = try made;
    defer s.deinit();
    try std.testing.expectEqual(@as(usize, 3), s.nchunks);
    if (s.ack.signal) |*g| {
        _ = g.free(&g.s);
        s.ack.signal = null;
    }
    const t0 = try testTensors(gpa, shape, 1);
    const t1 = try testTensors(gpa, shape, 2);
    defer for (t0 ++ t1) |t| gpa.free(t);
    try s.addLayer(0, t0);
    try s.addLayer(1, t1);
    try s.attachPool(24, 4, 0);
    try s.start();
    try s.beginRound();
    // layer 1's span: two rows name experts 1 2 3 and 4 5 2 (the shared one's id, 8, last in each row); nothing is in a line, so five experts are staged in two chunks (3 and 2)
    setPicks(&s, 0, &.{ 1, 2, 3, 8 });
    setPicks(&s, 1, &.{ 4, 5, 2, 8 });
    const n = s.announce(.{ .layer = 1, .rows = 2, .prefill = true, .chunked = true });
    @atomicStore(u32, s.word(Store.req_at), n, .release);
    while (!policy.reached(@atomicLoad(u32, s.word(Store.ack_at), .acquire), n)) std.atomic.spinLoopHint();
    // the first answer names where each expert will be and which chunk it comes in
    const cmap = s.chunkWords();
    try std.testing.expectEqualSlices(u32, &.{ 1, 1, 1, 2, 2 }, cmap[1..6]);
    const scratch = &s.pool.?.scratch;
    const row = s.hostRow(1);
    for ([_]usize{ 1, 2, 3, 4, 5 }, 0..) |x, i| try std.testing.expectEqual(scratch.base() + i * blob, row[2 * x]);
    // the words come in chunk order, and the bytes of a chunk are in the scratch once its word is
    var dev: [160]u8 = undefined;
    var want: [160]u8 = undefined;
    for ([_]usize{ 1, 2 }) |c| {
        while (!policy.reached(@atomicLoad(u32, s.word(chunk_word_at + c * 64), .acquire), n)) std.atomic.spinLoopHint();
        for ([_]usize{ 1, 2, 3, 4, 5 }, 0..) |x, i| {
            if (cmap[x] != c) continue;
            try scratch.download(i * blob, &dev);
            blobOf(shape, t1, x, &want);
            try std.testing.expectEqualSlices(u8, &want, &dev);
        }
    }
    try s.endRound(1, 1000);
    try std.testing.expect(!s.failed.load(.acquire));
    try std.testing.expectEqual(@as(u64, 5 * blob), s.stats.scratch_bytes);
}

test "a chunked stage that fails still releases every word the card waits for" {
    const gpa = std.testing.allocator;
    var r = try Runtime.open();
    defer r.close();
    var ctx = try @import("context.zig").Context.init(&r, 0);
    defer ctx.deinit();
    const shape: Shape = .{ .layers = 2, .experts = 8, .picks = 4, .trace_rows = 4, .part_bytes = .{ 64, 16, 16, 32, 16, 16 } };
    _ = c_setenv("TF_EXPERT_PREFILL_CHUNKS", "3", 1);
    const made = initOn(gpa, &r, shape);
    _ = c_setenv("TF_EXPERT_PREFILL_CHUNKS", "0", 1);
    var s = try made;
    defer s.deinit();
    if (s.ack.signal) |*g| {
        _ = g.free(&g.s);
        s.ack.signal = null;
    }
    // no layer was added: the stage cannot find the host tier, and says so by failing
    try s.attachPool(24, 4, 0);
    setPicks(&s, 0, &.{ 1, 2, 3, 8 });
    s.serveChunked(.{ .layer = 0, .rows = 1, .prefill = true, .chunked = true }, 7);
    try std.testing.expect(s.failed.load(.acquire));
    for (0..s.nchunks) |c| try std.testing.expectEqual(@as(u32, 7), @atomicLoad(u32, @as(*u32, @ptrCast(@alignCast(s.ack.waitPtrAt(c)))), .acquire));
}
