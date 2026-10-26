//! Expert-major prefill for an EXL3 Flash Next checkpoint whose routed experts live in host memory (the offload pool).
//!
//! The batched pass (cuda_batch.zig `chunk`) takes up to `Engine.chunk` rows through every layer, and its MoE copies the
//! experts a 256-row block picked into the layer's pool lines: a long prompt copies every expert of every layer about
//! once per block. Here a prompt goes through in DEEP chunks of `rows` rows (thousands): every layer runs the pass's own
//! blocks (HC read-outs, DeltaNet / attention, write-backs, router and top-k) on sub-blocks of the engine's chunk size
//! over device rows that persist for the whole deep chunk, then the layer's routed experts run EXPERT-MAJOR over all
//! `rows` rows: each expert a pair picked is staged into a pool line once and computed on every pair that picked it
//! (flashnext_exl3_prefill.cu, 64-member tiles), in passes of at most half the layer's lines, so a pass's copies run on
//! the copy engine while the previous pass computes. A deep chunk thus copies each expert of a layer at most once.
//!
//! Bits: every block but the routed experts is the pass's own, row for row (its kernels are row-invariant), so the
//! streams stay those of the batched pass until the experts' fp32 sums, which run in another order (64-member tiles
//! over all of K instead of runSlots' split-K partials): per pair the output moves by fp32 rounding, measured at about
//! 4e-5 of a layer's peak (tf-cuda-test exl3-prefill prompt). The logits change accordingly (bf16 noise).
//!
//! The hook: Engine.forward calls `forward` first; it declines (false) for anything but a plain prompt on one rank of an
//! EXL3 checkpoint with pooled or resident experts, no media, no window keep, no logits for every row.
//! TF_FLASHNEXT_EXL3_PREFILL=0 turns it off; _ROWS sets the deep chunk (default 8192), _MIN the shortest prompt
//! that takes it (default 512 rows), _TILES=0 keeps the dense projections on the reference kernels, _RING=0 stages
//! each layer into its own lines, _KERNEL=slots runs the experts through runSlots (the batched pass's bits).

const std = @import("std");
const cuda = @import("cuda");
const batch = @import("cuda_batch.zig");
const wt = @import("cuda_prompt_wt.zig");
const load = @import("cuda_exl3_load.zig");
const pool_ = @import("cuda_exl3_pool.zig");
const pk = @import("cuda_exl3_prefill_k.zig");
const prompt = @import("cuda_prompt.zig");
const ring_ = @import("cuda_exl3_prefill_ring.zig");
const history_ = @import("cuda_exl3_prefill_history.zig");
const tp_ = @import("cuda_exl3_tp.zig");
const window_ = @import("exl3_prefill_window.zig");

const dims = wt.dims;
const wide = wt.wide;
const streams = wt.streams;
const slots = wt.slots;
const eps = wt.eps;
const vocab = wt.vocab;
const out_sk = wt.out_sk;
const ple_layer = wt.ple_layer;
const ple_tail_n = wt.ple_tail_n;
const table_n = wt.experts_n + 1;
const routed_n_ = wt.experts_n;
const shared_id = wt.experts_n;
const att_rows = batch.att_rows;
const routed_guess: usize = 400;
const pss_n = batch.pss_n;

pub const Options = struct {
    enabled: bool = true,
    /// Rows a deep chunk holds (whole attention blocks).
    rows: usize = 8192,
    /// The deep chunk the server reserves at load (reserveFor): its buffers come out of the pools' share of the card
    /// (0.16 GiB a 1024 rows), so the server takes fewer than `rows` unless TF_FLASHNEXT_EXL3_PREFILL_ROWS says.
    serve_rows: usize = 4096,
    /// The shortest prompt that takes this path.
    min_rows: usize = 512,
    /// New experts a pass stages at most, as a fraction of the layer's lines (0.5: double buffering).
    pass_lines_half: bool = true,
    /// Stage the next layer's first pass while this layer's blocks compute.
    prefetch: bool = true,
    /// Run each pass through Experts.runSlots (256-row blocks over masked picks) instead of the 64-member prefill
    /// kernel: the batched pass's bits exactly (a check of everything but the kernel; slow).
    slots_kernel: bool = false,
    /// Every decoder layer's pool lines as one ring the copy engine streams the coming layers' experts into
    /// (cuda_exl3_prefill_ring.zig); false: each layer stages into its own lines, in passes of half of them.
    ring: bool = true,
    /// Experts a ring pass computes at most.
    ring_pass: usize = 32,
    /// Layers ahead the ring prefetches (the guess: the hottest experts, as many as the current layer used).
    ring_ahead: usize = 1,
    /// Dense EXL3 projections of a deep chunk's sub-blocks on #806's prompt tiles (Dense.prompt, .tiles) instead of
    /// the reference 128-row blocks: 1.4-2x faster at 128+ rows, a different fp32 order (bf16-level changes; the
    /// same float64 error, tf-cuda-test exl3-prefill dense).
    tiles: bool = true,
    /// After a ring prefill, refill every pool with its layer's hottest experts of the prefills (measured: decode hit
    /// +5%, ms/token unchanged, prefill +0.17 s; off).
    reseed: bool = false,
    reduced_branch: bool = false,
    two_chunk: bool = false,
};

pub fn optionsFromEnv() Options {
    var o: Options = .{};
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL")) |v| o.enabled = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_ROWS")) |v| {
        o.rows = std.fmt.parseInt(usize, std.mem.span(v), 10) catch o.rows;
        o.serve_rows = o.rows;
    }
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_REDUCED_BRANCH")) |v| o.reduced_branch = std.mem.eql(u8, std.mem.span(v), "1");
    o.two_chunk = window_.enabled(if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_TWO_CHUNK")) |value| std.mem.span(value) else null);
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_RING")) |v| o.ring = !std.mem.eql(u8, std.mem.span(v), "0");
    if (std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_RING_AHEAD")) |v| o.ring_ahead = std.fmt.parseInt(usize, std.mem.span(v), 10) catch o.ring_ahead;
    return o;
}

/// The options every call reads (a test may change them between calls).
pub var options: ?Options = null;

fn opts() Options {
    if (options == null) options = optionsFromEnv();
    return options.?;
}

/// What the last deep forward did (read by tf-cuda-test exl3-prefill engine).
pub const Stats = struct {
    chunks: u64 = 0,
    rows: u64 = 0,
    passes: u64 = 0,
    copies: u64 = 0,
    prefetched: u64 = 0,
    members: u64 = 0,
    host_wait_ns: u64 = 0, // the picks' D2H and its wait, every layer
    plan_ns: u64 = 0, // host plan of the passes (members, items, pool rounds)
    reduces: u64 = 0, // expert parallel: the ranks' all-reduces (one a layer a deep chunk)
    reduce_ns: u64 = 0, // their combine + all-reduce + place, measured only with TF_FLASHNEXT_PROFILE=1 (a wait each)
    head_rows: u64 = 0, // MTP head rows run expert-major (`head`)
};
pub var stats: Stats = .{};

/// Device bytes of a deep chunk's own rows: streams, sums, gates, mixed rows, slot outputs, picks, n-gram rows, and
/// the experts' compact scratch and member lists.
pub fn deviceBytes(rows: usize) usize {
    return Layout.of(rows, 1).total;
}

/// The same at `world` ranks (expert parallel adds the ranks' sum, rows x 2560 fp32).
pub fn deviceBytesAt(rows: usize, world: usize) usize {
    return Layout.of(rows, world).total;
}

const Layout = struct {
    h: usize,
    pss: usize,
    inj: usize,
    mixed: usize,
    xsm: usize,
    y: usize,
    wts: usize,
    picks: usize,
    ple_emb: usize,
    ple_xs: usize,
    ids: usize,
    pos: usize,
    members: usize,
    items: usize,
    kscratch: usize,
    /// Expert parallel only: a rank's combined branch, then the ranks' sum [rows, D] fp32 (0 bytes at world 1).
    sum: usize,
    total: usize,
    member_cap: usize,

    fn of(rows: usize, world: usize) Layout {
        var l: Layout = undefined;
        var at: usize = 0;
        const take = struct {
            fn f(a: *usize, n: usize) usize {
                const r = a.*;
                a.* = r + (n + 255) / 256 * 256;
                return r;
            }
        }.f;
        const pairs = rows * slots;
        l.member_cap = 2 * rows;
        l.h = take(&at, rows * wide * 2);
        l.pss = take(&at, rows * pss_n * 4);
        l.inj = take(&at, rows * streams * 2);
        l.mixed = take(&at, rows * dims * 2);
        l.xsm = take(&at, rows * (dims / 32) * 4);
        l.y = take(&at, pairs * dims * 4);
        l.wts = take(&at, pairs * 4);
        l.picks = take(&at, pairs * 4);
        l.ple_emb = take(&at, rows * dims * 2);
        l.ple_xs = take(&at, rows * (dims / 32) * 4);
        l.ids = take(&at, rows * 4);
        l.pos = take(&at, (rows / att_rows + 1) * 16);
        l.members = take(&at, pairs * 4);
        l.items = take(&at, 3 * pk.maxItems(pairs, table_n) * 4);
        l.kscratch = take(&at, pk.scratchBytes(l.member_cap, dims, wt.moe_w));
        l.sum = take(&at, if (world > 1) rows * dims * 4 else 0);
        l.total = at;
        return l;
    }
};

/// Host side: pinned uploads (ids, positions, n-gram rows, members and items) and the picks' download.
const HostLayout = struct {
    ids: usize,
    pos: usize,
    pos_stride: usize,
    emb: usize,
    xs: usize,
    picks: usize,
    members: usize,
    items: usize,
    total: usize,

    fn of(rows: usize) HostLayout {
        var l: HostLayout = undefined;
        var at: usize = 0;
        const take = struct {
            fn f(a: *usize, n: usize) usize {
                const r = a.*;
                a.* = r + (n + 63) / 64 * 64;
                return r;
            }
        }.f;
        const pairs = rows * slots;
        l.ids = take(&at, rows * 4);
        l.pos_stride = (rows / att_rows + 1) * 16;
        l.pos = take(&at, l.pos_stride * (wt.layers_n + 1));
        l.emb = take(&at, rows * dims * 2);
        l.xs = take(&at, rows * (dims / 32) * 4);
        l.picks = take(&at, pairs * 4);
        l.members = take(&at, pairs * 4);
        l.items = take(&at, 3 * pk.maxItems(pairs, table_n) * 4);
        l.total = at;
        return l;
    }
};

/// One pass of a layer: its experts (the pool round's routed ids), its members and items.
const Pass = struct { id_lo: u32, id_hi: u32, m_lo: u32, m_n: u32, i_lo: u32, i_n: u32 };

/// Sub-blocks a deep chunk holds at most (chunk's view array).
const max_sub = 64;

const Persistent = struct {
    h: usize,
    pss: usize,
    inj: usize,
    y: usize,
    wts: usize,
    total: usize,

    fn of(rows: usize) Persistent {
        const h = rows * wide * 2;
        const pss = std.mem.alignForward(usize, rows * pss_n * 4, 256);
        const inj = std.mem.alignForward(usize, rows * streams * 2, 256);
        const y = rows * slots * dims * 4;
        const wts = std.mem.alignForward(usize, rows * slots * 4, 256);
        return .{ .h = 0, .pss = h, .inj = h + pss, .y = h + pss + inj, .wts = h + pss + inj + y, .total = h + pss + inj + y + wts };
    }
};

const Deep = struct {
    gpa: std.mem.Allocator,
    owner: *const anyopaque,
    rows: usize,
    world: usize = 1,
    lay: Layout,
    hl: HostLayout,
    dev: cuda.DeviceBuffer,
    sum_dev: ?cuda.DeviceBuffer = null,
    host: cuda.HostBuffer,
    kern: pk.Kernels,
    aux: cuda.Stream,
    /// Expert parallel: the ranks' all-reduces run on `red`, one a sub-block, so the next layer's sub-block b only waits
    /// for its own rows' sum (`red_ev[b]`) and the others' reduce under its compute; `red_n` sub-blocks are in flight.
    red: ?cuda.Stream = null,
    red_ev: [max_sub]cuda.Event = undefined,
    red_go: cuda.Event = undefined,
    red_n: usize = 0,
    reduced_branch: bool = false,
    unit_weights: ?cuda.DeviceBuffer = null,
    /// The layer's pass plan: expert ids in pass order, the passes, member counts and starts.
    order: [table_n]u16 = undefined,
    passes: [table_n + 1]Pass = undefined,
    n_passes: usize = 0,
    cnt: [table_n]u32 = undefined,
    start: [table_n]u32 = undefined,
    /// Experts the previous layer's prefetch staged for each layer (held at its MoE).
    fake: [48 * slots]i32 = undefined,
    /// The prefill line ring over the pools (made at the first prefill, once the pools exist).
    ring: ?*ring_.Ring = null,
    ring_failed: bool = false,
    /// Experts each layer used in the current chunk (the next layers' prefetch size).
    used: [wt.layers_n]usize = @splat(routed_guess),
    history: [wt.layers_n]history_.Slot = @splat(.{}),
    history_shape: ?history_.Shape = null,
    second: ?*Deep = null,
    shared: ?*const Deep = null,
    window_graph_reservation: u64 = 0,

    fn dptr(dp: *const Deep, off: usize) u64 {
        if (dp.shared) |source| {
            const persistent = Persistent.of(dp.rows);
            inline for (.{ "h", "pss", "inj", "y", "wts" }) |field| {
                if (off == @field(dp.lay, field)) return dp.dev.ptr + @field(persistent, field);
            }
            if (off == dp.lay.sum) if (dp.sum_dev) |buffer| return buffer.ptr;
            return source.dptr(off);
        }
        return dp.dev.ptr + off;
    }
    fn hbytes(dp: *const Deep, off: usize, n: usize) []u8 {
        return dp.host.bytes[off..][0..n];
    }
};

var deep: ?*Deep = null;

pub const RingTelemetry = struct { owner: u64, copies: u64, dma_requested_bytes: u64, unused_copies: u64, hit_picks: u64, miss_picks: u64, hit_expert_requests: u64, miss_expert_requests: u64, attribution: ?@import("cuda_exl3_prefill_attribution.zig").Totals };

pub fn ringTelemetry(owner: *const anyopaque) ?RingTelemetry {
    const buffers = deep orelse return null;
    if (buffers.owner != owner) return null;
    const ring = buffers.ring orelse return null;
    return .{ .owner = ring.telemetry_id, .copies = ring.stats.copies, .dma_requested_bytes = ring.stats.dma_requested_bytes, .unused_copies = ring.stats.wasted, .hit_picks = ring.stats.hit_picks, .miss_picks = ring.stats.miss_picks, .hit_expert_requests = ring.stats.hit_expert_requests, .miss_expert_requests = ring.stats.miss_expert_requests, .attribution = if (ring.attribution) |state| state.totals else null };
}

/// Frees the deep chunk's buffers (the engine's deinit, or a test).
pub fn release() void {
    if (deep) |dp| {
        @import("cuda_exl3_graph.zig").clearPrefill();
        if (dp.second) |second| {
            dp.aux.synchronize() catch @panic("two-chunk aux retirement synchronization failed");
            dp.dev.d.synchronize() catch @panic("two-chunk compute retirement synchronization failed");
            second.host.free();
            if (dp.red != null) {
                for (&second.red_ev) |*event| event.deinit();
                second.red_go.deinit();
            }
            if (second.sum_dev) |*buffer| buffer.free();
            second.dev.free();
            dp.gpa.destroy(second);
        }
        if (dp.ring) |r| r.deinit();
        if (dp.red) |*rs| {
            rs.synchronize() catch {};
            for (&dp.red_ev) |*ev| ev.deinit();
            dp.red_go.deinit();
            rs.deinit();
        }
        if (dp.unit_weights) |*buffer| buffer.free();
        dp.kern.unload();
        dp.aux.deinit();
        dp.host.free();
        dp.dev.free();
        dp.gpa.destroy(dp);
        deep = null;
    }
    if (agree_buf) |*b| b.free();
    agree_buf = null;
}

/// The server's sizing hook (cuda_lanes_load.zig, before the pools take the free memory): the deep chunk's buffers for
/// an EXL3 checkpoint with offloaded experts on one rank, sized for min(TF_FLASHNEXT_EXL3_PREFILL_ROWS, context).
/// A failure leaves the batched pass in charge of prompts.
pub fn reserveFor(e: *prompt.Engine, context: usize, tp: usize) void {
    const o = opts();
    if (!o.enabled or (tp != 1 and tp != 2) or e.n_layers != wt.layers_n) return;
    for (e.layers[0..e.n_layers]) |*l| {
        const x = l.exl3 orelse return;
        if (x.bank == null) return;
        // expert parallel: every layer's experts carry this world's rank
        if ((if (x.tp) |r| r.world else 1) != tp) return;
    }
    const rows = @min(o.serve_rows, (context + att_rows - 1) / att_rows * att_rows);
    if (rows < o.min_rows) return;
    reserveAt(e.gpa, e.driver, e, rows, tp) catch |err| std.log.warn("flashnext exl3 prefill: deep chunks of {d} rows refused ({t}); prompts take the batched pass", .{ rows, err });
}

/// Rows a serving lane prefills a step (cuda_lanes.zig `step`): a deep chunk when this engine holds deep buffers,
/// else the engine's chunk; at most `draft_rows` (the lane's draft keeps that many rows of streams).
pub fn stepRows(e: *const prompt.Engine, draft_rows: ?usize) usize {
    var n = e.chunk;
    if (deep) |dp| if (dp.owner == @as(*const anyopaque, e) and opts().enabled) {
        n = @max(n, stepCapacity(dp.rows, opts().two_chunk and dp.second != null));
    };
    if (draft_rows) |r| n = @min(n, @max(r, 1));
    return n;
}

fn stepCapacity(rows: usize, paired: bool) usize {
    return if (paired and rows <= window_.max_rows) rows * 2 else rows;
}

/// Rows of streams a lane's MTP draft keeps (cuda_lanes.zig, cuda_lanes_load.zig): a whole step's.
pub fn draftRows(e: *const prompt.Engine) usize {
    var n = stepRows(e, null);
    if (deep) |dp| if (dp.owner == @as(*const anyopaque, e)) {
        const o = opts();
        if (o.two_chunk and o.enabled and o.ring and !o.reduced_branch and dp.world == 2) {
            n = @max(n, stepCapacity(dp.rows, true));
        }
    };
    return n;
}

/// The engine's deinit: frees the deep chunk's buffers if `owner` made them.
pub fn releaseFor(owner: *const anyopaque) void {
    if (deep) |dp| if (dp.owner == owner) release();
}

/// Allocates the deep chunk's buffers for `rows` rows now (before the pool takes the free memory). A no-op if they
/// exist and are at least that large.
pub fn reserve(gpa: std.mem.Allocator, d: *const cuda.Driver, owner: *const anyopaque, rows_in: usize) !void {
    return reserveAt(gpa, d, owner, rows_in, 1);
}

/// `reserve` for a rank of `world` (expert parallel at 2: the ranks' sum, and per-slot outputs zeroed once so a slot
/// another rank computed only ever holds finite values for the write-back's zero weights).
pub fn reserveAt(gpa: std.mem.Allocator, d: *const cuda.Driver, owner: *const anyopaque, rows_in: usize, world: usize) !void {
    const rows = rows_in / att_rows * att_rows;
    if (rows == 0 or (world != 1 and world != 2)) return error.Invalid;
    if (deep) |dp| {
        if (dp.owner == owner and dp.rows >= rows and dp.world == world) return;
        release();
    }
    const dp = try gpa.create(Deep);
    errdefer gpa.destroy(dp);
    dp.* = .{ .gpa = gpa, .owner = owner, .rows = rows, .world = world, .lay = Layout.of(rows, world), .hl = HostLayout.of(rows), .dev = undefined, .host = undefined, .kern = undefined, .aux = undefined };
    dp.dev = try cuda.DeviceBuffer.alloc(d, dp.lay.total);
    errdefer dp.dev.free();
    dp.host = try cuda.HostBuffer.alloc(d, dp.hl.total);
    errdefer dp.host.free();
    dp.kern = try pk.Kernels.load(d);
    errdefer dp.kern.unload();
    dp.aux = try cuda.Stream.init(d, true);
    errdefer dp.aux.deinit();
    if (world > 1) {
        try d.check(d.api.cuMemsetD8Async(dp.dptr(dp.lay.y), 0, rows * slots * dims * 4, dp.aux.handle), "cuMemsetD8Async");
        try dp.aux.synchronize();
        var rs = try cuda.Stream.init(d, true);
        errdefer rs.deinit();
        var made: usize = 0;
        errdefer for (dp.red_ev[0..made]) |*ev| ev.deinit();
        for (&dp.red_ev) |*ev| {
            ev.* = try cuda.Event.init(d, false);
            made += 1;
        }
        dp.red_go = try cuda.Event.init(d, false);
        dp.red = rs;
    }
    deep = dp;
    std.log.info("flashnext exl3 prefill: deep chunks of {d} rows, {d:.2} GiB on the device, {d:.1} MiB pinned (world {d})", .{ rows, @as(f64, @floatFromInt(dp.lay.total)) / (1 << 30), @as(f64, @floatFromInt(dp.hl.total)) / (1 << 20), world });
}

pub fn reserveTwoChunk(d: *const cuda.Driver, free: u64, protected: u64, graph_reservation: u64) !u64 {
    const o = opts();
    if (!o.two_chunk or !o.enabled or !o.ring or o.reduced_branch) return 0;
    const dp = deep orelse return 0;
    if (dp.dev.d != d or dp.world != 2 or dp.rows > window_.max_rows) return 0;
    if (dp.second != null) return 0;
    const graph_required = @import("cuda_exl3_graph_memory.zig").twoChunkReservationBytes();
    if (@import("cuda_exl3_graph.zig").enabled() and (graph_required == 0 or graph_reservation < graph_required)) {
        std.log.info("flashnext exl3 prefill: two-chunk window declined, incremental graph reservation unavailable (required={d} supplied={d})", .{ graph_required, graph_reservation });
        return 0;
    }
    const persistent = Persistent.of(dp.rows);
    const sum_bytes = if (dp.world > 1) try std.math.mul(usize, dp.rows, dims * 4) else 0;
    const device_bytes = try std.math.add(usize, persistent.total, sum_bytes);
    if (!window_.admits(free, protected, device_bytes)) {
        std.log.info("flashnext exl3 prefill: two-chunk window declined, extra_device={d} full_layout={d} extra_pinned=0 protected={d} free={d}", .{ device_bytes, dp.lay.total, protected, free });
        return 0;
    }
    const second = try dp.gpa.create(Deep);
    second.* = .{ .gpa = dp.gpa, .owner = dp.owner, .rows = dp.rows, .world = dp.world, .lay = dp.lay, .hl = dp.hl, .dev = undefined, .host = undefined, .kern = dp.kern, .aux = dp.aux, .red = dp.red, .red_ev = undefined, .red_go = undefined, .shared = dp };
    var device_allocated = false;
    var host_allocated = false;
    var made_events: usize = 0;
    var reduce_event_allocated = false;
    errdefer {
        if (reduce_event_allocated) second.red_go.deinit();
        for (second.red_ev[0..made_events]) |*event| event.deinit();
        if (host_allocated) second.host.free();
        if (second.sum_dev) |*buffer| buffer.free();
        if (device_allocated) second.dev.free();
        dp.gpa.destroy(second);
    }
    second.dev = try cuda.DeviceBuffer.alloc(d, persistent.total);
    device_allocated = true;
    if (sum_bytes != 0) {
        second.sum_dev = try cuda.DeviceBuffer.alloc(d, sum_bytes);
    }
    second.host = try cuda.HostBuffer.alloc(d, second.hl.total);
    host_allocated = true;
    if (dp.red != null) {
        for (&second.red_ev) |*event| {
            event.* = try cuda.Event.init(d, false);
            made_events += 1;
        }
        second.red_go = try cuda.Event.init(d, false);
        reduce_event_allocated = true;
    } else {
        second.red_ev = dp.red_ev;
        second.red_go = dp.red_go;
    }
    try d.check(d.api.cuMemsetD8Async(second.dptr(second.lay.y), 0, dp.rows * slots * dims * 4, dp.aux.handle), "cuMemsetD8Async");
    try dp.aux.synchronize();
    dp.second = second;
    dp.window_graph_reservation = graph_reservation;
    std.log.info("flashnext exl3 prefill: experimental two-chunk window, chunk_rows={d} hidden_bytes={d} full_layout={d} extra_device={d} extra_pinned={d}", .{ dp.rows, 2 * dp.rows * wide * 2, dp.lay.total, device_bytes, second.hl.total });
    return device_bytes;
}

pub fn twoChunkAllocated() bool {
    return if (deep) |buffers| buffers.second != null else false;
}

fn applies(e: *prompt.Engine, seq: *prompt.Seq, ids: []const u32, hooks: batch.Hooks, o: Options) bool {
    if (!o.enabled or ids.len < o.min_rows) return false;
    if (hooks.keep != null or hooks.logits == .all) return false;
    if (seq.media != null or e.n_layers != wt.layers_n) return false;
    // expert parallel (cuda_exl3_tp.zig) at two ranks over the engine's collective; the head whole or split by vocabulary
    const world = worldOf(e);
    if (e.head_q.n != vocab and !(world == 2 and e.head_q.n == vocab / 2)) return false;
    for (e.layers[0..e.n_layers]) |*l| {
        const x = l.exl3 orelse return false;
        if ((if (x.tp) |r| r.world else 1) != world or x.needsPool()) return false;
        if (x.layer.act_mode != 1 and x.layer.act_mode != 0) return false;
    }
    return e.chunk >= att_rows;
}

fn worldOf(e: *const prompt.Engine) usize {
    return if (e.collective != null) 2 else 1;
}

/// Expert parallel: both ranks take the deep path or neither, with the same deep chunk (the smaller rank's): each
/// rank's room ([rank 0's, rank 1's] summed over the collective). 0: this prompt takes the batched pass on both.
fn agree(e: *prompt.Engine, mine: usize) !usize {
    const c = e.collective orelse return mine;
    const st = e.stream.*;
    const rank = e.layers[0].exl3.?.tp.?.rank;
    if (agree_buf == null) agree_buf = try cuda.DeviceBuffer.alloc(e.driver, 256);
    const b = agree_buf.?;
    var v: [2]f32 = .{ 0, 0 };
    v[rank] = @floatFromInt(mine);
    try st.synchronize();
    try b.upload(0, std.mem.asBytes(&v));
    try c.sum(st, b.ptr, 2);
    try c.finish(st);
    try b.download(0, std.mem.asBytes(&v));
    if (v[0] < 1 or v[1] < 1) return 0;
    return @intFromFloat(@min(v[0], v[1]));
}
var agree_buf: ?cuda.DeviceBuffer = null;
var tp_logged = false;

fn now(io: std.Io) i96 {
    return std.Io.Clock.awake.now(io).toNanoseconds();
}

/// The hook (Engine.forward): runs `ids` from `seq.fed` and returns true, or returns false having done nothing.
pub fn forward(e: *prompt.Engine, comptime Tri: type, tri: Tri, seq: *prompt.Seq, ids: []const u32, hooks: batch.Hooks) !bool {
    const o = opts();
    if (!applies(e, seq, ids, hooks, o)) return false;
    try @import("cuda_exl3_dplan.zig").toHost(); // the pools' residency on the host before planning a prompt
    if (ids.len == 0 or seq.fed + ids.len > seq.cap) return error.UnexpectedTensor;
    const s = try e.reserve(@min(ids.len, e.chunk));
    const sub = s.cap / att_rows * att_rows;
    if (sub == 0) return false;
    // the deep chunk: as large as asked (whole attention blocks), no larger than the prompt needs; its sub-blocks are
    // the pass's chunks, the last one shorter
    const rows = @max(att_rows, @min(o.rows, (ids.len + att_rows - 1) / att_rows * att_rows) / att_rows * att_rows);
    // the buffers reserved at load (reserveFor) are kept whatever their size; a first prompt without them allocates
    const world = worldOf(e);
    const have = if (deep) |d| d.owner == @as(*const anyopaque, e) and d.world == world else false;
    var room: usize = if (have) deep.?.rows else 0;
    if (!have) {
        if (reserveAt(e.gpa, e.driver, e, rows, world)) |_| {
            room = deep.?.rows;
        } else |err| std.log.warn("flashnext exl3 prefill: no room for deep chunks of {d} rows ({t}); the batched pass takes the prompt", .{ rows, err });
    }
    // both ranks the same way (a collective a rank skips would hang the other)
    room = try agree(e, room);
    if (room == 0) return false;
    if (world == 2 and !tp_logged) std.log.info("flashnext exl3 prefill: expert-parallel deep chunks of up to {d} rows (rank {d}: its own experts, one all-reduce a layer)", .{ room, e.layers[0].exl3.?.tp.?.rank });
    tp_logged = world == 2 or tp_logged;
    const dp = deep.?;
    try selectWriteback(dp, o.reduced_branch and world == 2);
    const scores = try e.reserveScores(seq.cap, s.cap);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = tri,
        .k = &e.batch_k,
        .s = s,
        .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = e.layers[0..e.n_layers], .mixer = e.mixer, .head = e.head_q, .ngram = &e.ngram, .collective = e.collective },
        .seq = seq.view(scores),
        .hooks = hooks,
    };
    // equal deep chunks (a short last chunk would pay a whole chunk's expert copies for a few rows)
    const cap_rows = @min(rows, room);
    const n_chunks = (ids.len + cap_rows - 1) / cap_rows;
    const deep_rows = @min(cap_rows, ((ids.len + n_chunks - 1) / n_chunks + att_rows - 1) / att_rows * att_rows);
    const rg = if (o.ring) try ringOf(dp, e) else null;
    var window_room: usize = 0;
    if (o.two_chunk) {
        var supported = dp.second != null and rg != null and world == 2 and !dp.reduced_branch and seq.kv8 == null and deep_rows <= window_.max_rows and ids.len > deep_rows;
        if (@import("cuda_exl3_graph.zig").enabled()) {
            const required = @import("cuda_exl3_graph_memory.zig").twoChunkReservationBytes();
            if (required == 0 or dp.window_graph_reservation < required) supported = false;
        }
        for (e.layers[0..e.n_layers]) |*layer| if (@import("cuda_exl3_heads.zig").isSplit(layer)) {
            supported = false;
        };
        window_room = try agree(e, if (supported) deep_rows else 0);
    }
    if (o.tiles and !o.slots_kernel) {
        try warmTiles(e);
        tiles_on = true;
    }
    defer tiles_on = false;
    if (rg) |r| try r.begin(p.stream);
    defer if (rg) |r| r.end(p.stream, o.reseed) catch |err| std.log.err("flashnext exl3 prefill: ring end: {t}", .{err});
    var at: usize = 0;
    while (at < ids.len) {
        if (window_room != 0) {
            const plan_ = try window_.Plan.init(ids.len - at, window_room, att_rows);
            try twoChunk(Tri, dp, &p, s, ids[at..][0..plan_.rows], seq.fed + at, sub, hooks, at, at + plan_.rows == ids.len, seq.logits, o, rg.?, plan_);
            at += plan_.rows;
            continue;
        }
        const n = @min(deep_rows, ids.len - at);
        const last = at + n == ids.len;
        const lg: batch.Logits = if (last) hooks.logits else .none;
        const st: u64 = if (hooks.streams != 0) hooks.streams + at * wide * 2 else 0;
        try chunk(Tri, dp, &p, s, ids[at..][0..n], seq.fed + at, sub, lg, st, if (last and hooks.logits != .none) seq.logits else null, o, rg);
        at += n;
    }
    if (hooks.logits == .none) try p.stream.synchronize();
    seq.fed += ids.len;
    return true;
}

/// A sub-block's view of the pass's scratch: the per-row buffers a layer keeps across its sub-blocks point into the
/// deep chunk's rows [r0, r0 + n); everything else is the pass's own (reused by every sub-block).
fn view(base: *const batch.Scratch, dp: *const Deep, r0: usize) batch.Scratch {
    var v = base.*;
    const l = dp.lay;
    v.h = dp.dptr(l.h) + r0 * wide * 2;
    v.pss = dp.dptr(l.pss) + r0 * pss_n * 4;
    v.inj = dp.dptr(l.inj) + r0 * streams * 2;
    v.mixed = dp.dptr(l.mixed) + r0 * dims * 2;
    v.xsm = dp.dptr(l.xsm) + r0 * (dims / 32) * 4;
    v.y = dp.dptr(l.y) + r0 * slots * dims * 4;
    v.wts = dp.dptr(l.wts) + r0 * slots * 4;
    v.picks = dp.dptr(l.picks) + r0 * slots * 4;
    if (dp.world > 1) v.moe_f32 = dp.dptr(l.sum) + r0 * dims * 4;
    v.ple_emb = dp.dptr(l.ple_emb) + r0 * dims * 2;
    v.ple_xs = dp.dptr(l.ple_xs) + r0 * (dims / 32) * 4;
    v.ids = dp.dptr(l.ids) + r0 * 4;
    v.pos = dp.dptr(l.pos) + (r0 / att_rows) * 16;
    return v;
}

fn selectWriteback(dp: *Deep, reduced_branch: bool) !void {
    if (reduced_branch and dp.unit_weights == null) {
        const weights: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(dp.hl.ids, dp.rows * 4)));
        @memset(weights, 1);
        dp.unit_weights = try cuda.DeviceBuffer.fromHost(dp.dev.d, std.mem.sliceAsBytes(weights));
    }
    if (dp.reduced_branch != reduced_branch) @import("cuda_exl3_graph.zig").clearPrefill();
    dp.reduced_branch = reduced_branch;
}

fn pendingWriteback(p: anytype, dp: *const Deep, layer: ?usize, r0: usize) void {
    p.pending = layer;
    p.pending_f32 = layer != null and dp.reduced_branch;
    p.pending_f32_weights = if (p.pending_f32) dp.unit_weights.?.ptr + r0 * 4 else null;
}

fn compactLast(p: anytype, reduced_branch: bool, row: usize) !void {
    if (row == 0) return;
    const s = p.s;
    try p.copy(s.h, s.h + row * wide * 2, wide * 2);
    try p.copy(s.pss, s.pss + row * pss_n * 4, pss_n * 4);
    try p.copy(s.inj, s.inj + row * streams * 2, streams * 2);
    if (reduced_branch) {
        try p.copy(s.moe_f32, s.moe_f32 + row * dims * 4, dims * 4);
    } else {
        try p.copy(s.y, s.y + row * slots * dims * 4, slots * dims * 4);
        try p.copy(s.wts, s.wts + row * slots * 4, slots * 4);
    }
}

const Gather = struct {
    ngram: *const @import("cuda_ple.zig").Ngram,
    history: []i64,
    ids: []const u32,
    emb: []u16,
    xs: []f32,
    failed: ?anyerror = null,
    ns: i96 = 0,
    io: std.Io,

    fn main(g: *Gather) void {
        const t0 = now(g.io);
        defer g.ns = now(g.io) - t0;
        g.ngram.rows(g.history, g.ids, g.emb, g.xs) catch |err| {
            g.failed = err;
        };
    }
};

fn upload(st: cuda.Stream, dst: u64, bytes: []const u8) !void {
    try st.d.check(st.d.api.cuMemcpyHtoDAsync_v2(dst, bytes.ptr, bytes.len, st.handle), "cuMemcpyHtoDAsync");
}

fn windowStage(dp: *Deep, p: anytype, ids: []const u32, pos0: usize) !void {
    const n = ids.len;
    try p.stream.synchronize();
    const id32 = std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.ids, n * 4));
    for (id32, ids) |*destination, id| destination.* = @intCast(id);
    const blocks = (n + att_rows - 1) / att_rows;
    const positions = std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.pos, blocks * 16));
    @memset(positions, 0);
    for (0..blocks) |block| positions[4 * block] = @intCast(pos0 + block * att_rows);
    try upload(p.stream, dp.dptr(dp.lay.ids), dp.hbytes(dp.hl.ids, n * 4));
    try upload(p.stream, dp.dptr(dp.lay.pos), dp.hbytes(dp.hl.pos, blocks * 16));
    if (p.m.table_dense) {
        try p.tri.embedDense(dp.dptr(dp.lay.ids), p.m.table_w, dp.dptr(dp.lay.h), dims, streams, n);
    } else {
        try p.tri.embed(dp.dptr(dp.lay.ids), p.m.table_w, p.m.table_s, p.m.table_b, dp.dptr(dp.lay.h), dims, streams, n);
    }
}

fn windowInputs(dp: *Deep, p: anytype, ids: []const u32, pos0: usize, layer: usize) !void {
    const blocks = (ids.len + att_rows - 1) / att_rows;
    const positions_offset = dp.hl.pos + (layer + 1) * dp.hl.pos_stride;
    const positions: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(positions_offset, blocks * 16)));
    @memset(positions, 0);
    for (0..blocks) |block| positions[4 * block] = @intCast(pos0 + block * att_rows);
    try upload(p.stream, dp.dptr(dp.lay.pos), dp.hbytes(positions_offset, blocks * 16));
    if (layer == ple_layer) {
        const embedding: []u16 = @alignCast(std.mem.bytesAsSlice(u16, dp.hbytes(dp.hl.emb, ids.len * dims * 2)));
        const sums: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(dp.hl.xs, ids.len * (dims / 32) * 4)));
        try p.m.ngram.rows(p.seq.history[0..], ids, embedding, sums);
        try upload(p.stream, dp.dptr(dp.lay.ple_emb), std.mem.sliceAsBytes(embedding));
        try upload(p.stream, dp.dptr(dp.lay.ple_xs), std.mem.sliceAsBytes(sums));
    }
}

fn windowDense(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, layer: usize, n: usize, pos0: usize, sub: usize, waits: []const cuda.Event) !void {
    const graph = @import("cuda_exl3_graph.zig");
    const lw = &p.m.layers[layer];
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_sub) |block| views[block] = view(base, dp, block * sub);
    defer p.s = base;
    p.s = base;
    p.rows = n;
    p.pos0 = pos0;
    const capture = try graph.prefillBegin(p, lw.linear, layer, n, sub, pos0, dp.dev.ptr, base, tiles_on, waits, dp.dptr(dp.lay.pos));
    errdefer graph.prefillAbort(p, capture);
    if (capture != .replayed) for (0..n_sub) |block| {
        const at = block * sub;
        const count = @min(sub, n - at);
        p.s = &views[block];
        p.rows = count;
        p.pos0 = pos0 + at;
        pendingWriteback(p, dp, if (layer == 0) null else layer - 1, at);
        if (block < waits.len) switch (capture) {
            .eager => try p.stream.wait(waits[block]),
            .capturing => try p.stream.waitExternal(waits[block]),
            .replayed => unreachable,
        };
        try layerBlock(Tri, p, lw, layer, count);
    };
    try graph.prefillEnd(p, capture);
    dp.red_n = 0;
}

fn windowFinish(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, n: usize, sub: usize, logits: batch.Logits, streams_out: u64, logits_host: ?[]u8) !void {
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_sub) |block| views[block] = view(base, dp, block * sub);
    defer p.s = base;
    stats.chunks += 1;
    stats.rows += n;
    const last_block = n_sub - 1;
    const last_rows = n - last_block * sub;
    if (streams_out != 0) {
        for (0..n_sub) |block| {
            const count = @min(sub, n - block * sub);
            p.s = &views[block];
            p.rows = count;
            pendingWriteback(p, dp, wt.layers_n - 1, block * sub);
            try p.writeback(0, count);
            try p.copy(streams_out + block * sub * wide * 2, views[block].h, count * wide * 2);
        }
    }
    if (logits == .none) return;
    p.s = &views[last_block];
    p.rows = last_rows;
    const scratch = p.s;
    const row = last_rows - 1;
    if (streams_out != 0) {
        if (row > 0) {
            try p.copy(scratch.h, scratch.h + row * wide * 2, wide * 2);
            try p.copy(scratch.pss, scratch.pss + row * pss_n * 4, pss_n * 4);
        }
    } else {
        try compactLast(p, dp.reduced_branch, row);
        pendingWriteback(p, dp, wt.layers_n - 1, last_block * sub);
        try p.writeback(0, 1);
    }
    try p.readout(p.m.mixer, 0, 1);
    if (p.m.head.n == vocab) {
        try p.lane(scratch.mixed, scratch.xsm, p.m.head, scratch.logits, scratch.dummy, 1, 1);
    } else {
        const collective = p.m.collective orelse return error.TensorParallelUnavailable;
        try p.lane(scratch.mixed, scratch.xsm, p.m.head, scratch.head_local, scratch.dummy, 1, 1);
        try collective.gather(p.stream, scratch.head_local, scratch.logits, vocab / 2);
    }
    if (logits_host) |destination| {
        if (p.m.collective) |collective| try collective.finish(p.stream);
        try p.stream.synchronize();
        try p.stream.d.check(p.stream.d.api.cuMemcpyDtoH_v2(destination.ptr, scratch.logits, vocab * 2), "cuMemcpyDtoH");
    }
}

fn twoChunk(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, ids: []const u32, pos0: usize, sub: usize, hooks: batch.Hooks, output_at: usize, last: bool, logits_host: []u8, o: Options, ring: *ring_.Ring, plan_: window_.Plan) !void {
    const buffers = [_]*Deep{ dp, dp.second.? };
    for (plan_.segments[0..plan_.count]) |segment| {
        if ((segment.rows + sub - 1) / sub > max_sub) return error.UnexpectedTensor;
    }
    try p.stream.synchronize();
    try ring.newChunk();
    history_.prepare(&dp.history, &dp.history_shape, .{ .sub = sub, .lo = ring.own_lo, .hi = ring.own_hi });
    for (0..@min(o.ring_ahead + 1, wt.layers_n)) |layer| _ = try ring.prefetch(layer, @min(routed_n_, dp.used[layer]), keepFree(ring, o), .initial_prefetch);
    for (plan_.segments[0..plan_.count], 0..) |segment, index| {
        try windowStage(buffers[index], p, ids[segment.offset..][0..segment.rows], pos0 + segment.offset);
    }
    var schedule = plan_.iterator(p.m.layers.len);
    while (schedule.next()) |step| {
        const active = buffers[step.chunk];
        const segment = plan_.segments[step.chunk];
        try windowInputs(active, p, ids[segment.offset..][0..segment.rows], pos0 + segment.offset, step.layer);
        try windowDense(Tri, active, p, base, step.layer, segment.rows, pos0 + segment.offset, sub, active.red_ev[0..active.red_n]);
        if (active != dp) {
            active.used = dp.used;
            active.history = dp.history;
        }
        const layer = &p.m.layers[step.layer];
        try ringMoe(Tri, active, ring, p, layer, step.layer, segment.rows, o);
        if (active != dp) {
            dp.used = active.used;
            dp.history = active.history;
        }
        try rankSum(active, p, layer.exl3.?, segment.rows, sub);
        if (step.layer + 1 == p.m.layers.len) {
            for (active.red_ev[0..active.red_n]) |event| try p.stream.wait(event);
            active.red_n = 0;
        }
    }
    for (plan_.segments[0..plan_.count], 0..) |segment, index| {
        const final = last and index + 1 == plan_.count;
        const output = if (hooks.streams == 0) 0 else hooks.streams + (output_at + segment.offset) * wide * 2;
        try windowFinish(Tri, buffers[index], p, base, segment.rows, sub, if (final) hooks.logits else .none, output, if (final and hooks.logits != .none) logits_host else null);
    }
}

fn chunk(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), base: *batch.Scratch, ids: []const u32, pos0: usize, sub: usize, logits: batch.Logits, streams_out: u64, logits_host: ?[]u8, o: Options, rg: ?*ring_.Ring) !void {
    const n = ids.len;
    const st = p.stream;
    const hl = dp.hl;
    // the previous chunk's uploads read the pinned rows: they landed before its head (a synchronize), or below
    try st.synchronize();
    if (rg) |r| {
        try r.newChunk();
        history_.prepare(&dp.history, &dp.history_shape, .{ .sub = sub, .lo = r.own_lo, .hi = r.own_hi });
        // the first layers' experts start streaming now, under the embedding and layer 0's blocks
        for (0..@min(o.ring_ahead + 1, wt.layers_n)) |l| _ = try r.prefetch(l, @min(routed_n_, dp.used[l]), keepFree(r, o), .initial_prefetch);
    }
    const id32 = std.mem.bytesAsSlice(i32, dp.hbytes(hl.ids, n * 4));
    for (id32, ids) |*x, v| x.* = @intCast(v);
    const blocks = (n + att_rows - 1) / att_rows;
    const pos = std.mem.bytesAsSlice(i32, dp.hbytes(hl.pos, blocks * 16));
    @memset(pos, 0);
    for (0..blocks) |b| pos[4 * b] = @intCast(pos0 + b * att_rows);
    try upload(st, dp.dptr(dp.lay.ids), dp.hbytes(hl.ids, n * 4));
    try upload(st, dp.dptr(dp.lay.pos), dp.hbytes(hl.pos, blocks * 16));
    const emb: []u16 = @alignCast(std.mem.bytesAsSlice(u16, dp.hbytes(hl.emb, n * dims * 2)));
    const xs: []f32 = @alignCast(std.mem.bytesAsSlice(f32, dp.hbytes(hl.xs, n * (dims / 32) * 4)));
    var gather: Gather = .{ .ngram = p.m.ngram, .history = p.seq.history[0..], .ids = ids, .emb = emb, .xs = xs, .io = p.io };
    var thread: ?std.Thread = try std.Thread.spawn(.{}, Gather.main, .{&gather});
    defer if (thread) |th| th.join();
    const t0 = now(p.io);
    defer if (profile()) {
        st.synchronize() catch {};
        std.debug.print("exl3 prefill: {d} rows at {d} in {d:.1} ms (n-gram gather {d:.1} ms on the host)\n", .{ n, pos0, @as(f64, @floatFromInt(now(p.io) - t0)) / 1e6, @as(f64, @floatFromInt(gather.ns)) / 1e6 });
    };
    p.rows = n;
    p.pos0 = pos0;
    pendingWriteback(p, dp, null, 0);
    if (p.m.table_dense) {
        try p.tri.embedDense(dp.dptr(dp.lay.ids), p.m.table_w, dp.dptr(dp.lay.h), dims, streams, n);
    } else {
        try p.tri.embed(dp.dptr(dp.lay.ids), p.m.table_w, p.m.table_s, p.m.table_b, dp.dptr(dp.lay.h), dims, streams, n);
    }
    const n_sub = (n + sub - 1) / sub;
    var views: [max_sub]batch.Scratch = undefined;
    if (n_sub > views.len) return error.UnexpectedTensor;
    for (0..n_sub) |b| views[b] = view(base, dp, b * sub);
    defer p.s = base;
    for (p.m.layers, 0..) |*lw, layer| {
        if (layer == ple_layer) {
            if (thread) |th| th.join();
            thread = null;
            if (gather.failed) |err| return err;
            try upload(st, dp.dptr(dp.lay.ple_emb), std.mem.sliceAsBytes(emb));
            try upload(st, dp.dptr(dp.lay.ple_xs), std.mem.sliceAsBytes(xs));
        }
        const pg = try @import("cuda_exl3_graph.zig").prefillBegin(p, lw.linear, layer, n, sub, pos0, dp.dev.ptr, base, tiles_on, dp.red_ev[0..dp.red_n], dp.dptr(dp.lay.pos));
        errdefer @import("cuda_exl3_graph.zig").prefillAbort(p, pg);
        if (pg != .replayed) for (0..n_sub) |b| {
            const m = @min(sub, n - b * sub);
            p.s = &views[b];
            p.rows = m;
            p.pos0 = pos0 + b * sub;
            pendingWriteback(p, dp, if (layer > 0) layer - 1 else null, b * sub);
            if (b < dp.red_n) switch (pg) {
                .eager => try st.wait(dp.red_ev[b]),
                .capturing => try st.waitExternal(dp.red_ev[b]),
                .replayed => unreachable,
            };
            try layerBlock(Tri, p, lw, layer, m);
        };
        try @import("cuda_exl3_graph.zig").prefillEnd(p, pg);
        dp.red_n = 0;
        if (rg) |r| try ringMoe(Tri, dp, r, p, lw, layer, n, o) else try moe(Tri, dp, p, lw, layer, n, o);
        try rankSum(dp, p, lw.exl3.?, n, sub);
    }
    for (dp.red_ev[0..dp.red_n]) |ev| try st.wait(ev);
    dp.red_n = 0;
    stats.chunks += 1;
    stats.rows += n;
    // the last layer's experts written back where the caller keeps the streams, then the head on the last row
    const last_b = n_sub - 1;
    const m_last = n - last_b * sub;
    if (streams_out != 0) {
        for (0..n_sub) |b| {
            const m = @min(sub, n - b * sub);
            p.s = &views[b];
            p.rows = m;
            pendingWriteback(p, dp, wt.layers_n - 1, b * sub);
            try p.writeback(0, m);
            try p.copy(streams_out + b * sub * wide * 2, views[b].h, m * wide * 2);
        }
    }
    if (logits == .none) return;
    p.s = &views[last_b];
    p.rows = m_last;
    const s = p.s;
    const r = m_last - 1;
    if (streams_out != 0) {
        if (r > 0) {
            try p.copy(s.h, s.h + r * wide * 2, wide * 2);
            try p.copy(s.pss, s.pss + r * pss_n * 4, pss_n * 4);
        }
    } else {
        try compactLast(p, dp.reduced_branch, r);
        pendingWriteback(p, dp, wt.layers_n - 1, last_b * sub);
        try p.writeback(0, 1);
    }
    try p.readout(p.m.mixer, 0, 1);
    if (p.m.head.n == vocab) {
        try p.lane(s.mixed, s.xsm, p.m.head, s.logits, s.dummy, 1, 1);
    } else {
        // the head split by vocabulary (expert parallel): each rank's half, gathered in vocabulary order (projectHead)
        const c = p.m.collective orelse return error.TensorParallelUnavailable;
        try p.lane(s.mixed, s.xsm, p.m.head, s.head_local, s.dummy, 1, 1);
        try c.gather(st, s.head_local, s.logits, vocab / 2);
    }
    if (logits_host) |lh| {
        if (p.m.collective) |c| try c.finish(st);
        try st.synchronize();
        try st.d.check(st.d.api.cuMemcpyDtoH_v2(lh.ptr, s.logits, vocab * 2), "cuMemcpyDtoH");
    }
}

fn profile() bool {
    const v = std.c.getenv("TF_FLASHNEXT_PROFILE") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// One sub-block of one layer, as Pass.chunk runs it, up to the router's picks (the experts come after every
/// sub-block, expert-major).
fn layerBlock(comptime Tri: type, p: *batch.Pass(Tri), lw: *const @import("cuda_prompt_load.zig").Layer, layer: usize, m: usize) !void {
    const s = p.s;
    if (layer == ple_layer) {
        if (p.pending != null) try p.writeback(0, m);
        // Pass.pleBlock for a prompt's rows (no lanes)
        try p.lane(s.ple_emb, s.ple_xs, lw.key, s.ple_keys, s.dummy, m, 1);
        try p.lane(s.ple_emb, s.ple_xs, lw.val, s.dummy, s.ple_part, m, 4);
        try p.sumSlices(s.ple_part, s.ple_vals, true, m * dims, 4);
        try p.tri.pleGate(s.ple_keys, s.ple_vals, s.h, lw.nk.ptr, lw.nq.ptr, s.ple_gated, s.ple_pss, eps, m, dims, streams);
        try p.tri.pleConv(s.ple_gated, s.ple_pss, lw.nc.ptr, p.seq.tail.ptr, lw.ple_cw.ptr, s.h, s.h, s.ple_nrow, eps, m, dims, streams, 4, 3);
        try p.shiftWindow(p.seq.tail.ptr, s.ple_nrow, wide, m, ple_tail_n, wide);
    }
    try p.writeback(0, m);
    try p.readout(lw.attn_hc, 0, m);
    if (lw.linear) {
        try p.gdnBlock(lw, wt.linearIndex(layer), .{});
    } else {
        try p.attnBlock(lw, wt.attnIndex(layer));
    }
    try p.tri.hcWriteback(s.h, s.h, s.pss, s.part, "*fp32", s.inj, s.dummy, "*bf16", s.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
    try p.readout(lw.mlp_hc, 0, m);
    try p.tri.routerMany(s.mixed, lw.router.ptr, s.router_l, dims, m);
    try p.tri.topkMany(s.router_l, s.picks, s.wts, m);
}

/// Lines a pass may fill: half the layer's lines (the other half holds the previous pass, still computing).
fn passLines(pool: *const pool_.Pool, o: Options) usize {
    const lines = pool.opt.lines;
    return @max(1, if (o.pass_lines_half) lines / 2 else lines);
}

/// The layer's experts over all `n` rows of the deep chunk, expert-major.
fn moe(comptime Tri: type, dp: *Deep, p: *batch.Pass(Tri), lw: *const @import("cuda_prompt_load.zig").Layer, layer: usize, n: usize, o: Options) !void {
    const x = lw.exl3.?;
    const st = p.stream;
    const pairs = n * slots;
    const hl = dp.hl;
    const t0 = now(p.io);
    // the picks of every row on the host: the layer's only wait
    const host_picks: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.picks, pairs * 4)));
    try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(host_picks.ptr, dp.dptr(dp.lay.picks), pairs * 4, st.handle), "cuMemcpyDtoHAsync");
    try st.synchronize();
    maskOwned(host_picks, x.tp);
    const t1 = now(p.io);
    try plan(dp, x, host_picks, o);
    // members and items of every pass in one upload (the pinned bytes stay put until the next layer's wait)
    const members: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.members, pairs * 4)));
    const items: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.items, 3 * pk.maxItems(pairs, table_n) * 4)));
    const counts = build(dp, host_picks, members, items);
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(members[0..counts.members]));
    try upload(st, dp.dptr(dp.lay.items), std.mem.sliceAsBytes(items[0 .. 3 * counts.items]));
    const t2 = now(p.io);
    stats.host_wait_ns += @intCast(t1 - t0);
    stats.plan_ns += @intCast(t2 - t1);
    stats.members += counts.members;
    const pool = x.pool;
    var fake_rows: usize = 0;
    for (dp.passes[0..dp.n_passes]) |ps| {
        var r: ?*const pool_.Round = null;
        if (pool) |pl| {
            const ids_ = dp.order[ps.id_lo..ps.id_hi];
            fake_rows = fakePicks(dp, ids_);
            if (fake_rows > 0) {
                r = try pl.planUnobserved(dp.fake[0 .. fake_rows * slots], fake_rows);
                if (r.?.n_passes != 1) return error.Exl3PrefillPass;
                try pl.stage(r.?, 0, st);
                stats.copies += r.?.misses_();
            }
        }
        const l = if (pool) |pl| pl.tables() else x.layer;
        if (o.slots_kernel) {
            try slotsPass(dp, x, st, l, host_picks, ps, n);
            if (r) |rr| try pool.?.release(rr, 0, st);
            stats.passes += 1;
            continue;
        }
        const dv: pk.Device = .{ .members = dp.dptr(dp.lay.members) + ps.m_lo * 4, .items = dp.dptr(dp.lay.items) + ps.i_lo * 12, .scratch = dp.dptr(dp.lay.kscratch) };
        try pk.run(&dp.kern, st, l, dp.dptr(dp.lay.mixed), dims, dp.dptr(dp.lay.picks), slots, ps.m_n, ps.i_n, dv, dp.dptr(dp.lay.y));
        if (r) |rr| try pool.?.release(rr, 0, st);
        stats.passes += 1;
    }
    // the next layer's first pass, staged on the copy engine while this layer's experts and the next blocks run
    if (o.prefetch and layer + 1 < p.m.layers.len) {
        if (p.m.layers[layer + 1].exl3) |nx| if (nx.pool) |npl| try prefetch(dp, npl, x.tp, o);
    }
    p.pending = layer;
}

/// Expert parallel: another rank's picks become -1 on the host (this rank plans, copies and computes its own pairs).
fn maskOwned(picks: []i32, r: ?tp_.Rank) void {
    const rk = r orelse return;
    for (picks) |*pk_| if (pk_.* >= 0 and !rk.owns(@intCast(pk_.*))) {
        pk_.* = -1;
    };
}

/// Expert parallel: after a layer's passes, each row's branch over this rank's slots (on the compute stream), then on
/// the reduce stream one all-reduce of [rows, 2560] fp32 a sub-block (both ranks issue the same calls with the same
/// rows in the same order) and its sum placed where the write-back reads it; the next layer's sub-block b waits for
/// red_ev[b] only. TF_FLASHNEXT_EXL3_PREFILL_OVERLAP=0: one all-reduce of all rows on the compute stream.
fn rankSum(dp: *Deep, p: anytype, x: *const load.Experts, n: usize, sub: usize) !void {
    const r = x.tp orelse return;
    if (r.world == 1) return;
    const c = p.m.collective orelse return error.TensorParallelUnavailable;
    const st = p.stream;
    const sum = dp.dptr(dp.lay.sum);
    const y = dp.dptr(dp.lay.y);
    const wts = dp.dptr(dp.lay.wts);
    const t0 = if (profile()) now(p.io) else 0;
    try pk.tpCombine(&dp.kern, st, y, wts, dp.dptr(dp.lay.picks), sum, n, slots, dims, r.lo, r.hi);
    if (!overlap() or dp.red == null) {
        try c.sum(st, sum, n * dims);
        if (!dp.reduced_branch) try pk.tpPlace(&dp.kern, st, sum, y, wts, n, slots, dims);
    } else {
        const rs = dp.red.?;
        try dp.red_go.record(st);
        try rs.wait(dp.red_go);
        const n_sub = (n + sub - 1) / sub;
        for (0..n_sub) |b| {
            const r0 = b * sub;
            const m = @min(sub, n - r0);
            try c.sum(rs, sum + r0 * dims * 4, m * dims);
            if (!dp.reduced_branch) try pk.tpPlace(&dp.kern, rs, sum + r0 * dims * 4, y + r0 * slots * dims * 4, wts + r0 * slots * 4, m, slots, dims);
            try dp.red_ev[b].record(rs);
        }
        dp.red_n = n_sub;
    }
    if (profile()) {
        if (dp.red) |rs| try c.finish(rs);
        try c.finish(st);
        stats.reduce_ns += @intCast(now(p.io) - t0);
    }
    stats.reduces += 1;
}

fn overlap() bool {
    const v = std.c.getenv("TF_FLASHNEXT_EXL3_PREFILL_OVERLAP") orelse return true;
    return !std.mem.eql(u8, std.mem.span(v), "0");
}

/// Picks for a pool round that holds `ids` (routed, < 512): rows of 11 slots, -1 after the last.
fn fakePicks(dp: *Deep, ids: []const u16) usize {
    var routed: usize = 0;
    for (ids) |id| routed += @intFromBool(id != shared_id);
    if (routed == 0) return 0;
    const rows = (routed + slots - 1) / slots;
    std.debug.assert(rows * slots <= dp.fake.len);
    @memset(dp.fake[0 .. rows * slots], -1);
    var k: usize = 0;
    for (ids) |id| {
        if (id == shared_id) continue;
        dp.fake[k] = id;
        k += 1;
    }
    return rows;
}

/// Stages up to a pass's worth of the next layer's experts that its pool does not hold (lowest ids first): a deep
/// chunk uses nearly every expert, so any of them is a copy its MoE would make anyway.
fn prefetch(dp: *Deep, pool: *pool_.Pool, own: ?tp_.Rank, o: Options) !void {
    const want = passLines(pool, o);
    var ids: [table_n]u16 = undefined;
    var k: usize = 0;
    for (0..wt.experts_n) |id| {
        if (k == want) break;
        if (own) |rk| if (!rk.owns(id)) continue;
        if (pool_.offload.lineOf(pool.v, pool.opt.layer, @intCast(id)) != null) continue;
        ids[k] = @intCast(id);
        k += 1;
    }
    const rows = fakePicks(dp, ids[0..k]);
    if (rows == 0) return;
    const r = try pool.plan(dp.fake[0 .. rows * slots], rows);
    if (r.n_passes != 1) return error.Exl3PrefillPass;
    // the copies wait for nothing on the compute stream: the aux stream takes the gates and the release
    try pool.stage(r, 0, dp.aux);
    try pool.release(r, 0, dp.aux);
    stats.prefetched += r.misses_();
    stats.copies += r.misses_();
}

/// The layer's passes: pass 0 the shared expert and every expert the pool holds (no copy), then the others in id
/// order, at most `passLines` new experts and `member_cap` members a pass. Fills order, passes, cnt.
fn plan(dp: *Deep, x: *const load.Experts, picks: []const i32, o: Options) !void {
    @memset(&dp.cnt, 0);
    for (picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    const cap = dp.lay.member_cap;
    var k: usize = 0;
    dp.n_passes = 0;
    var held: [table_n]bool = @splat(false);
    held[shared_id] = true;
    const per: usize = if (x.pool) |pl| passLines(pl, o) else table_n;
    if (x.pool) |pl| {
        for (0..wt.experts_n) |id| held[id] = pool_.offload.lineOf(pl.v, pl.opt.layer, @intCast(id)) != null;
    } else {
        @memset(&held, true);
    }
    // pass 0 (and more if the held experts' members exceed the cap): held experts, the shared one first
    var lo: usize = 0;
    var members: usize = 0;
    const order_ids = struct {
        fn close(d: *Deep, lo_: *usize, k_: usize, members_: *usize) void {
            if (k_ == lo_.*) return;
            d.passes[d.n_passes] = .{ .id_lo = @intCast(lo_.*), .id_hi = @intCast(k_), .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
            d.n_passes += 1;
            lo_.* = k_;
            members_.* = 0;
        }
    };
    if (dp.cnt[shared_id] > 0) {
        dp.order[k] = shared_id;
        k += 1;
        members += dp.cnt[shared_id];
    }
    for (0..wt.experts_n) |id| {
        if (!held[id] or dp.cnt[id] == 0) continue;
        if (members + dp.cnt[id] > cap) order_ids.close(dp, &lo, k, &members);
        dp.order[k] = @intCast(id);
        k += 1;
        members += dp.cnt[id];
    }
    order_ids.close(dp, &lo, k, &members);
    // the misses, `per` new experts a pass
    var in_pass: usize = 0;
    for (0..wt.experts_n) |id| {
        if (held[id] or dp.cnt[id] == 0) continue;
        if (in_pass == per or members + dp.cnt[id] > cap) {
            order_ids.close(dp, &lo, k, &members);
            in_pass = 0;
        }
        dp.order[k] = @intCast(id);
        k += 1;
        members += dp.cnt[id];
        in_pass += 1;
    }
    order_ids.close(dp, &lo, k, &members);
}

/// Members (pairs grouped by expert in pass order, pairs ascending) and items (first relative to the pass's first
/// member) of every pass; fills each pass's ranges.
fn build(dp: *Deep, picks: []const i32, members: []i32, items: []i32) struct { members: usize, items: usize } {
    var m_at: u32 = 0;
    var i_at: u32 = 0;
    @memset(&dp.start, 0);
    for (dp.passes[0..dp.n_passes]) |*ps| {
        ps.m_lo = m_at;
        ps.i_lo = i_at;
        for (dp.order[ps.id_lo..ps.id_hi]) |id| {
            const c = dp.cnt[id];
            dp.start[id] = m_at;
            var off: u32 = 0;
            while (off < c) : (off += pk.tile) {
                items[3 * i_at] = id;
                items[3 * i_at + 1] = @intCast(m_at - ps.m_lo + off);
                items[3 * i_at + 2] = @intCast(@min(@as(u32, pk.tile), c - off));
                i_at += 1;
            }
            m_at += c;
        }
        ps.m_n = m_at - ps.m_lo;
        ps.i_n = i_at - ps.i_lo;
    }
    for (picks, 0..) |pk_, i| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        const id: usize = @intCast(pk_);
        members[dp.start[id]] = @intCast(i);
        dp.start[id] += 1;
    }
    return .{ .members = m_at, .items = i_at };
}

/// One pass through Experts.runSlots: the deep chunk's picks with every expert outside the pass masked to -1, in the
/// batched pass's 256-row blocks (a check: these are the batched pass's kernels and bits).
fn slotsPass(dp: *Deep, x: *load.Experts, st: cuda.Stream, l: cuda.exl3_experts.Layer, picks: []const i32, ps: Pass, n: usize) !void {
    var in_pass: [table_n]bool = @splat(false);
    for (dp.order[ps.id_lo..ps.id_hi]) |id| in_pass[id] = true;
    const pairs = n * slots;
    // the members' pinned region holds the masked picks (members are not used in this mode); the previous pass's
    // upload must have read it
    try st.synchronize();
    const masked: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(dp.hl.members, pairs * 4)));
    for (picks[0..pairs], masked) |pk_, *o| o.* = if (pk_ >= 0 and pk_ < table_n and in_pass[@intCast(pk_)]) pk_ else -1;
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(masked));
    const ex = try x.ops.grouped();
    const block: usize = @import("cuda_exl3_pass.zig").moe_rows;
    const bytes = try cuda.exl3_experts.scratchBytes(l, block, slots);
    const base = try x.ops.scratch(st, &x.ops.moe, bytes);
    var r0: usize = 0;
    while (r0 < n) : (r0 += block) {
        const rows = @min(block, n - r0);
        var sc = try cuda.exl3_experts.scratch(base, l, block, slots);
        sc.y = dp.dptr(dp.lay.y) + r0 * slots * dims * 4;
        try ex.runSlots(st, l, dp.dptr(dp.lay.mixed) + r0 * dims * 2, dims, .bf16, dp.dptr(dp.lay.members) + r0 * slots * 4, rows, sc);
    }
}

// ---------------------------------------------------------------------------------------------------- the MTP head

/// The hook in cuda_mtp_ops.zig `forwardHead` (a lane's prompt chunk through the MTP head, after the main pass kept
/// its streams): the head's one layer EXPERT-MAJOR over the whole chunk. forwardHead runs the layer in 256-row pieces,
/// each piece's MoE copying nearly every expert of the head's pool (about 1.7 s per 8192 rows of copies); here every
/// piece runs up to the router's picks over the deep chunk's rows (the attention reads the head's cache in piece
/// order, as there), then the experts once over all rows (`moe`: each expert staged once), then each piece's
/// write-back. The head is one layer, so nothing a piece reads depends on another piece's experts: the same maths as
/// forwardHead but the experts' fp32 sum order (the 64-member kernel). Declines (false) unless the engine holds deep
/// buffers of at least `next.len` rows and the head's experts are pooled on this rank (whole: the MTP head is rank 0's
/// under expert parallel), for a prompt chunk (no scores) of at least the minimum rows without media.
pub fn head(comptime Tri: type, o: anytype, d: *@import("cuda_mtp.zig").Draft, next: []const u32, rows: u64, scores: bool) !bool {
    const op = opts();
    const n = next.len;
    // A paired window hands the head up to two deep chunks of rows: it runs them as two slices (headSplit).
    if (!scores and deep != null and n > deep.?.rows) {
        const first = headSplit(n, deep.?.rows, op.min_rows) orelse return false;
        if (!try headSlice(Tri, o, d, next[0..first], rows, false)) return false;
        if (!try headSlice(Tri, o, d, next[first..], rows + first * wide * 2, false)) return error.UnexpectedTensor;
        return true;
    }
    return headSlice(Tri, o, d, next, rows, scores);
}

/// Where a head call of `n` rows over a paired window splits into two deep-sized slices, or null when it cannot.
fn headSplit(n: usize, deep_rows: usize, min_rows: usize) ?usize {
    if (n <= deep_rows or n > 2 * deep_rows) return null;
    const first = ((n + 1) / 2 + att_rows - 1) / att_rows * att_rows;
    if (first >= n or first > deep_rows or n - first < min_rows) return null;
    return first;
}

test "exl3: a paired window's head splits into two deep slices that cannot decline after the first runs" {
    try std.testing.expectEqual(@as(?usize, 4096), headSplit(8192, 4096, 512));
    try std.testing.expectEqual(@as(?usize, null), headSplit(4096, 4096, 512));
    try std.testing.expectEqual(@as(?usize, null), headSplit(8193, 4096, 512));
    var n: usize = 4097;
    while (n <= 8192) : (n += 61) {
        const first = headSplit(n, 4096, 512) orelse continue;
        try std.testing.expect(first <= 4096 and first % att_rows == 0 and n - first <= first and n - first >= 512);
    }
}

fn headSlice(comptime Tri: type, o: anytype, d: *@import("cuda_mtp.zig").Draft, next: []const u32, rows: u64, scores: bool) !bool {
    const op = opts();
    const n = next.len;
    if (!op.enabled or !headEnabled() or scores or n < op.min_rows or d.media != null) return false;
    const e = o.e;
    const dp = deep orelse return false;
    if (dp.owner != @as(*const anyopaque, e) or dp.rows < n) return false;
    const h = o.h;
    const x = h.layer.exl3 orelse return false;
    if (x.tp != null or x.pool == null or (x.layer.act_mode != 1 and x.layer.act_mode != 0)) return false;
    if (d.len + n > d.cap) return error.PromptTooLong;
    try @import("cuda_exl3_dplan.zig").toHost(); // the head's pool may be device-planned (cuda_exl3_chain.zig)
    const mtp = @import("cuda_mtp.zig");
    const tri = o.tri;
    const want = @max(@as(usize, 32), 2 * @min(n, mtp.piece_rows));
    const s = try e.reserve(want);
    const piece = @max(@as(usize, 1), @min(mtp.piece_rows, s.cap / 2)) / att_rows * att_rows;
    if (piece == 0) return false;
    const n_pieces = (n + piece - 1) / piece;
    if (n_pieces > max_sub) return false;
    const score_rows = try e.reserveScores(d.cap, piece);
    var p: batch.Pass(Tri) = .{
        .gpa = e.gpa,
        .io = e.io,
        .stream = e.stream.*,
        .tri = tri,
        .k = &e.batch_k,
        .s = s,
        .m = .{ .table_w = e.table.w.ptr, .table_s = e.table.s.ptr, .table_b = e.table.b.ptr, .table_dense = e.table.dense, .layers = (&h.layer)[0..1], .mixer = h.mixer, .head = h.draft, .ngram = &e.ngram },
        .seq = .{ .media = null, .conv = &.{}, .state = &.{}, .kc = (&d.kc)[0..1], .vc = (&d.vc)[0..1], .ikc = (&d.ikc)[0..1], .pooled = (&d.pooled)[0..1], .tail = d.kc, .history = &d.unused_history, .cap = d.cap, .scores = score_rows },
        .hooks = .{ .logits = .none },
    };
    var views: [max_sub]batch.Scratch = undefined;
    for (0..n_pieces) |b| views[b] = view(s, dp, b * piece);
    defer p.s = s;
    if (op.tiles and !op.slots_kernel) {
        try warmTiles(e);
        tiles_on = true;
    }
    defer tiles_on = false;
    const lw = &h.layer;
    for (0..n_pieces) |b| {
        const at = b * piece;
        const m = @min(piece, n - at);
        p.s = &views[b];
        const v = p.s;
        p.rows = m;
        p.pos0 = d.len + at;
        p.pending = null;
        try p.stage(next[at..][0..m]);
        // forwardHead's piece up to the router (the embedding branch, the hidden branch, attention over the head's cache)
        if (e.table.dense) {
            try tri.embedDense(v.ids, e.table.w.ptr, v.ple_keys, dims, streams, m);
        } else {
            try tri.embed(v.ids, e.table.w.ptr, e.table.s.ptr, e.table.b.ptr, v.ple_keys, dims, streams, m);
        }
        try tri.rmsnorm(v.ple_keys, h.norm_e.ptr, v.mixed, v.xsm, eps, wide, m, dims, dims);
        try p.lane(v.mixed, v.xsm, h.fc_e, v.dummy, v.part, m, mtp.fc_sk);
        try p.sumSlices(v.part, v.branch, true, m * dims, mtp.fc_sk);
        try tri.rmsnorm(rows + at * wide * 2, h.norm_h.ptr, v.norm, v.gq, eps, wide, m, wide, wide);
        try p.lane(v.norm, v.gq, h.fc_h, v.dummy, v.part, streams * m, mtp.fc_sk);
        try p.sumSlices(v.part, v.ple_gated, true, streams * m * dims, mtp.fc_sk);
        try tri.addStreams(v.branch, v.ple_gated, v.h, m, dims, streams);
        try p.writeback(0, m);
        try p.readout(lw.attn_hc, 0, m);
        try p.attnBlock(lw, 0);
        try tri.hcWriteback(v.h, v.h, v.pss, v.part, "*fp32", v.inj, v.dummy, "*bf16", v.dummy, "*bf16", m * dims, m, dims, streams, 4, 1, 1, out_sk);
        try p.readout(lw.mlp_hc, 0, m);
        try tri.routerMany(v.mixed, lw.router.ptr, v.router_l, dims, m);
        try tri.topkMany(v.router_l, v.picks, v.wts, m);
    }
    // the head's experts once over the chunk (its own pool, passes of half its lines)
    try moe(Tri, dp, &p, lw, 0, n, op);
    for (0..n_pieces) |b| {
        const m = @min(piece, n - b * piece);
        p.s = &views[b];
        p.rows = m;
        p.pending = 0;
        try p.writeback(0, m);
    }
    const last = &views[n_pieces - 1];
    try p.copy(d.out.ptr, last.h + (n - 1 - (n_pieces - 1) * piece) * wide * 2, wide * 2);
    d.len += n;
    stats.head_rows += n;
    if (!head_logged) std.log.info("flashnext exl3 prefill: the MTP head runs expert-major over deep chunks ({d} rows first)", .{n});
    head_logged = true;
    return true;
}
var head_logged = false;

fn headEnabled() bool {
    return true;
}

// ---------------------------------------------------------------------------------------------------- the ring

/// The ring over the engine's decoder pools, made once; null (and the per-layer passes) if they cannot form one.
fn ringOf(dp: *Deep, e: *prompt.Engine) !?*ring_.Ring {
    if (dp.ring) |r| return r;
    if (dp.ring_failed) return null;
    var pools: [wt.layers_n]*pool_.Pool = undefined;
    for (e.layers[0..e.n_layers], 0..) |*l, i| {
        const x = l.exl3 orelse {
            dp.ring_failed = true;
            return null;
        };
        pools[i] = x.pool orelse {
            dp.ring_failed = true;
            return null;
        };
    }
    dp.ring = ring_.Ring.init(dp.gpa, e.driver, &pools, e.telemetry.enabled) catch |err| {
        std.log.warn("flashnext exl3 prefill: no line ring ({t}); each layer stages into its own lines", .{err});
        dp.ring_failed = true;
        return null;
    };
    if (e.layers[0].exl3.?.tp) |rk| {
        dp.ring.?.own_lo = rk.lo;
        dp.ring.?.own_hi = @min(rk.hi, routed_n_);
    }
    std.log.info("flashnext exl3 prefill: a ring of {d} expert slots over the {d} layers' pool lines", .{ dp.ring.?.slots(), wt.layers_n });
    return dp.ring;
}

/// Slots a prefetch leaves free: a layer's worth of misses and two passes.
fn keepFree(r: *const ring_.Ring, o: Options) usize {
    return @min(r.slots() / 2, routed_n_ + 2 * o.ring_pass);
}

/// The layer's experts over all `n` rows of the deep chunk, through the ring: the experts the ring already holds
/// first (in load order), then the misses (hottest first), `ring_pass` a pass, each pass's copies queued a pass ahead.
fn ringMoe(comptime Tri: type, dp: *Deep, r: *ring_.Ring, p: *batch.Pass(Tri), lw: *const @import("cuda_prompt_load.zig").Layer, layer: usize, n: usize, o: Options) !void {
    const x = lw.exl3.?;
    const pool = x.pool.?;
    const st = p.stream;
    const pairs = n * slots;
    const hl = dp.hl;
    const t0 = now(p.io);
    const host_picks: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.picks, pairs * 4)));
    try st.d.check(st.d.api.cuMemcpyDtoHAsync_v2(host_picks.ptr, dp.dptr(dp.lay.picks), pairs * 4, st.handle), "cuMemcpyDtoHAsync");
    try st.synchronize();
    maskOwned(host_picks, x.tp);
    const t1 = now(p.io);
    @memset(&dp.cnt, 0);
    for (host_picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    r.drop(layer, dp.cnt[0..routed_n_]);
    // pass order: shared, then the held experts by load batch, then the misses hottest first
    var held: [routed_n_]u16 = undefined;
    var n_held: usize = 0;
    var miss: [routed_n_]u16 = undefined;
    var n_miss: usize = 0;
    var used: usize = 0;
    for (0..routed_n_) |id| {
        if (dp.cnt[id] == 0) continue;
        used += 1;
        r.counts[layer][id] +|= dp.cnt[id];
        const present = r.isLoaded(layer, id);
        r.stats.observeDemand(present, dp.cnt[id]);
        if (present) {
            held[n_held] = @intCast(id);
            n_held += 1;
        } else {
            miss[n_miss] = @intCast(id);
            n_miss += 1;
        }
    }
    dp.used[layer] = used;
    const ByBatch = struct {
        r: *const ring_.Ring,
        l: usize,
        fn lt(c: @This(), a: u16, b: u16) bool {
            return c.r.batch_of[c.l][a] < c.r.batch_of[c.l][b];
        }
    };
    std.mem.sort(u16, held[0..n_held], ByBatch{ .r = r, .l = layer }, ByBatch.lt);
    const ByCount = struct {
        c: *const [table_n]u32,
        fn gt(c: @This(), a: u16, b: u16) bool {
            if (c.c[a] != c.c[b]) return c.c[a] > c.c[b];
            return a < b;
        }
    };
    std.mem.sort(u16, miss[0..n_miss], ByCount{ .c = &dp.cnt }, ByCount.gt);
    // passes: at most ring_pass experts and member_cap members each; the shared expert opens pass 0
    var k: usize = 0;
    var lo: usize = 0;
    var members: usize = 0;
    var in_pass: usize = 0;
    dp.n_passes = 0;
    const cap = dp.lay.member_cap;
    const close = struct {
        fn f(d: *Deep, lo_: *usize, k_: usize, members_: *usize, in_pass_: *usize) void {
            if (k_ == lo_.*) return;
            d.passes[d.n_passes] = .{ .id_lo = @intCast(lo_.*), .id_hi = @intCast(k_), .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
            d.n_passes += 1;
            lo_.* = k_;
            members_.* = 0;
            in_pass_.* = 0;
        }
    }.f;
    if (dp.cnt[shared_id] > 0) {
        dp.order[k] = shared_id;
        k += 1;
        members += dp.cnt[shared_id];
    }
    for ([2][]const u16{ held[0..n_held], miss[0..n_miss] }) |list| for (list) |id| {
        if (in_pass == o.ring_pass or members + dp.cnt[id] > cap) close(dp, &lo, k, &members, &in_pass);
        dp.order[k] = id;
        k += 1;
        members += dp.cnt[id];
        in_pass += 1;
    };
    close(dp, &lo, k, &members, &in_pass);
    const staging0 = now(p.io);
    const copies0 = r.stats.copies;
    for (0..@min(2, dp.n_passes)) |j| try loadPass(dp, r, layer, j, false);
    const staging_ns = now(p.io) - staging0;
    const members_h: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.members, pairs * 4)));
    const items_h: []i32 = @alignCast(std.mem.bytesAsSlice(i32, dp.hbytes(hl.items, 3 * pk.maxItems(pairs, table_n) * 4)));
    const counts = build(dp, host_picks, members_h, items_h);
    try upload(st, dp.dptr(dp.lay.members), std.mem.sliceAsBytes(members_h[0..counts.members]));
    try upload(st, dp.dptr(dp.lay.items), std.mem.sliceAsBytes(items_h[0 .. 3 * counts.items]));
    const t2 = now(p.io);
    stats.host_wait_ns += @intCast(t1 - t0);
    stats.plan_ns += @intCast(t2 - t1 - staging_ns);
    stats.members += counts.members;
    const l = pool.tables();
    // the first two passes' copies, then each pass: the next pass's copies, the gate, the kernel, the release
    for (dp.passes[0..dp.n_passes], 0..) |ps, j| {
        if (j + 2 < dp.n_passes) try loadPass(dp, r, layer, j + 2, true);
        try loadPass(dp, r, layer, j, false);
        const ids_ = dp.order[ps.id_lo..ps.id_hi];
        try r.gate(st, layer, ids_);
        if (o.slots_kernel) {
            try slotsPass(dp, x, st, l, host_picks, ps, n);
        } else {
            const dv: pk.Device = .{ .members = dp.dptr(dp.lay.members) + ps.m_lo * 4, .items = dp.dptr(dp.lay.items) + ps.i_lo * 12, .scratch = dp.dptr(dp.lay.kscratch) };
            try pk.run(&dp.kern, st, l, dp.dptr(dp.lay.mixed), dims, dp.dptr(dp.lay.picks), slots, ps.m_n, ps.i_n, dv, dp.dptr(dp.lay.y));
        }
        try r.release(st, layer, ids_);
        stats.passes += 1;
    }
    stats.copies += r.stats.copies - copies0;
    dp.history[layer] = .{ .rows = n, .used = used };
    const c1 = r.stats.copies;
    for (1..o.ring_ahead + 1) |ahead| {
        const nl = layer + ahead;
        if (nl >= wt.layers_n) break;
        _ = try r.prefetch(nl, history_.want(dp.history[nl], n, used, routed_n_), keepFree(r, o), .cross_layer_prefetch);
    }
    stats.prefetched += r.stats.copies - c1;
    stats.copies += r.stats.copies - c1;
    p.pending = layer;
}

/// Queues the copies of pass `j`'s routed experts the ring does not hold; `best_effort`: stop quietly when full.
fn loadPass(dp: *Deep, r: *ring_.Ring, layer: usize, j: usize, best_effort: bool) !void {
    const ps = dp.passes[j];
    for (dp.order[ps.id_lo..ps.id_hi]) |id| {
        if (id == shared_id) continue;
        if (try r.load(layer, id, .demand)) continue;
        if (best_effort) return;
        return error.Exl3PrefillRingFull;
    }
    try r.flush();
}

// ---------------------------------------------------------------------------------------------------- dense tiles

/// Set while a deep prefill runs with Options.tiles: cuda_exl3_pass.part sends prompt rows to Dense.prompt on tiles.
var tiles_on: bool = false;
/// Rows below which a projection keeps the reference kernels (tiles lose at a few rows).
const tiles_min_rows: usize = 64;

/// The hook in cuda_exl3_pass.part: true when it ran the part on the tiles (bf16 or fp32 out).
pub fn tilesPart(de: *cuda.exl3.Dense, s: cuda.Stream, w: cuda.exl3.Weight, x: u64, m: usize, y: u64, f32_out: bool) !bool {
    if (!tiles_on or m < tiles_min_rows) return false;
    _ = cuda.exl3.Dense.candidateWeight(w) catch return false;
    if (de.candidate_k == null or de.candidate_k.?.k_max < w.k) return false;
    de.implementation = .tiles;
    defer de.implementation = .reference;
    try de.prompt(s, w, x, .bf16, m, y, if (f32_out) .f32 else .bf16);
    return true;
}

/// Loads the tiles once with the widest input any EXL3 projection has (no allocation inside a pass).
fn warmTiles(e: *prompt.Engine) !void {
    const x = e.layers[0].exl3 orelse return;
    const de = try x.ops.linear();
    const k_max = @max(x.ops.max_k, wt.dims * 3);
    if (de.candidate_k != null and de.candidate_k.?.k_max >= k_max) return;
    if (!cuda.kernels.available) return error.BuiltWithoutKernels;
    var mod = try cuda.Module.load(de.d, cuda.kernels.exl3_candidate);
    errdefer mod.unload();
    const k = try cuda.exl3_candidate.Kernels.resolve(de.d, mod, k_max);
    if (de.candidate_k) |*old| old.deinit();
    if (de.candidate_mod) |*old| old.unload();
    de.candidate_mod = mod;
    de.candidate_k = k;
}

test "exl3: reduced prefill writeback preserves slot rounding sub-blocks and tail selection" {
    try @import("cuda_exl3_prefill_writeback_test.zig").run(Deep, Layout.of, view, pendingWriteback, compactLast, selectWriteback);
    try std.testing.expect(!(Options{}).reduced_branch);
}

test "exl3: two-chunk scheduling and reservation contracts" {
    _ = window_;
    try std.testing.expect(!(Options{}).two_chunk);
    try std.testing.expectEqual(@as(usize, 4096), stepCapacity(4096, false));
    try std.testing.expectEqual(@as(usize, 8192), stepCapacity(4096, true));
    try std.testing.expectEqual(@as(usize, 8192), stepCapacity(8192, true));
}

test "exl3: two-chunk arenas retain hidden and deferred HC operands without aliasing" {
    const rows = 4096;
    const layout = Layout.of(rows, 2);
    const persistent = Persistent.of(rows);
    try std.testing.expectEqual(@as(usize, 727399424), layout.total);
    const host_layout = HostLayout.of(rows);
    try std.testing.expectEqual(@as(usize, 160 << 20), 2 * rows * wide * 2);
    var driver: cuda.Driver = undefined;
    var first: Deep = undefined;
    first.world = 2;
    first.rows = rows;
    first.shared = null;
    first.lay = layout;
    first.dev = .{ .d = &driver, .ptr = 0x100000, .len = layout.total };
    first.sum_dev = null;
    var second = first;
    second.dev.ptr += layout.total;
    second.dev.len = persistent.total;
    second.rows = rows;
    second.shared = &first;
    var base: batch.Scratch = undefined;
    base.moe_f32 = 0;
    const fields = .{ "h", "pss", "inj", "mixed", "xsm", "y", "wts", "picks", "ple_emb", "ple_xs", "ids", "pos", "members", "items", "kscratch", "sum" };
    inline for (fields, 0..) |field, index| {
        const start = @field(layout, field);
        const end = if (index + 1 < fields.len) @field(layout, fields[index + 1]) else layout.total;
        try std.testing.expectEqual(@as(usize, 0), start % 256);
        try std.testing.expect(start < end and end <= layout.total);
        try std.testing.expect(first.dev.ptr + end <= second.dev.ptr);
    }
    for ([_]usize{ 0, 1024, 3072 }) |offset| {
        const a = view(&base, &first, offset);
        const b = view(&base, &second, offset);
        inline for (.{ "h", "pss", "inj", "y", "wts" }) |field| {
            const second_start = second.dev.ptr + @field(persistent, field);
            try std.testing.expect(@field(b, field) >= second_start and @field(b, field) < second.dev.ptr + persistent.total);
            try std.testing.expect(@field(a, field) < second.dev.ptr);
        }
        inline for (.{ "mixed", "xsm", "picks", "ple_emb", "ple_xs", "ids", "pos", "moe_f32" }) |field| {
            try std.testing.expectEqual(@field(a, field), @field(b, field));
        }
    }
    inline for (.{ "members", "items", "kscratch", "sum" }) |field| try std.testing.expectEqual(first.dptr(@field(layout, field)), second.dptr(@field(layout, field)));
    try std.testing.expectEqual(@as(usize, 546127872), persistent.total);
    try std.testing.expect(persistent.total < layout.total);
    try std.testing.expect(host_layout.total > rows * dims * 2);
    try std.testing.expect(!window_.admits(layout.total - 1, 0, layout.total));
    try std.testing.expect(window_.admits(layout.total, 0, layout.total));
}

test "exl3: two-chunk memory refusal performs no CUDA allocation" {
    const saved_deep = deep;
    const saved_options = options;
    const graph = @import("cuda_exl3_graph.zig");
    const saved_force = graph.force;
    defer {
        deep = saved_deep;
        options = saved_options;
        graph.force = saved_force;
    }
    graph.force = .off;
    options = .{ .two_chunk = true };
    var driver: cuda.Driver = undefined;
    var buffers: Deep = undefined;
    buffers.dev = .{ .d = &driver, .ptr = 0x100000, .len = 1 };
    buffers.world = 2;
    buffers.rows = 4096;
    buffers.lay = Layout.of(4096, 2);
    buffers.second = null;
    deep = &buffers;
    try std.testing.expectEqual(@as(u64, 0), try reserveTwoChunk(&driver, Persistent.of(4096).total - 1, 0, 0));
    try std.testing.expect(buffers.second == null);
}

test "exl3: two-chunk PLE input staging advances mutable history only in chronological PLE segments" {
    const Fixture = struct {
        calls: usize = 0,
        before: [2][2]i64 = undefined,

        fn sync(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        fn copy(_: cuda.abi.DevicePtr, _: ?*const anyopaque, _: usize, _: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }

        pub fn rows(self: *@This(), history: []i64, ids: []const u32, embedding: []u16, sums: []f32) !void {
            self.before[self.calls] = history[0..2].*;
            self.calls += 1;
            for (ids) |id| {
                history[0] = history[1];
                history[1] = id;
            }
            @memset(embedding, 0);
            @memset(sums, 0);
        }
    };
    var driver: cuda.Driver = undefined;
    driver.api.cuStreamSynchronize = Fixture.sync;
    driver.api.cuMemcpyHtoDAsync_v2 = Fixture.copy;
    const host_layout = HostLayout.of(256);
    const host = try std.testing.allocator.alignedAlloc(u8, .@"16", host_layout.total);
    defer std.testing.allocator.free(host);
    var buffers: Deep = undefined;
    buffers.rows = 256;
    buffers.shared = null;
    buffers.lay = Layout.of(256, 2);
    buffers.hl = host_layout;
    buffers.host = .{ .d = &driver, .bytes = host };
    buffers.dev = .{ .d = &driver, .ptr = 0x100000, .len = buffers.lay.total };
    var fixture: Fixture = .{};
    var history: [2]i64 = .{ -2, -1 };
    var pass = .{ .stream = cuda.Stream{ .d = &driver, .handle = null }, .m = .{ .ngram = &fixture }, .seq = .{ .history = &history } };
    var ids: [512]u32 = undefined;
    for (&ids, 1..) |*id, value| id.* = @intCast(value);
    for (0..2) |index| try windowInputs(&buffers, &pass, ids[index * 256 ..][0..256], index * 256, 0);
    try std.testing.expectEqual(@as(usize, 0), fixture.calls);
    try std.testing.expectEqualSlices(i64, &.{ -2, -1 }, &history);
    for (0..2) |index| try windowInputs(&buffers, &pass, ids[index * 256 ..][0..256], index * 256, ple_layer);
    try std.testing.expectEqual(@as(usize, 2), fixture.calls);
    try std.testing.expectEqualSlices(i64, &.{ -2, -1 }, &fixture.before[0]);
    try std.testing.expectEqualSlices(i64, &.{ 255, 256 }, &fixture.before[1]);
    try std.testing.expectEqualSlices(i64, &.{ 511, 512 }, &history);
}

test "exl3: prefill passes keep each expert's pairs together, items relative to their pass" {
    var dp: Deep = undefined;
    // pairs 0..8: experts 5, 512, 2 / 2, 512, 7 / 5, 512, 5 (-1 and out-of-table ids skipped)
    const picks = [_]i32{ 5, 512, 2, 2, 512, 7, 5, 512, 5, -1, 600 };
    @memset(&dp.cnt, 0);
    for (picks) |pk_| {
        if (pk_ < 0 or pk_ >= table_n) continue;
        dp.cnt[@intCast(pk_)] += 1;
    }
    // pass 0: the shared expert and 5; pass 1: 2 and 7
    dp.order[0] = shared_id;
    dp.order[1] = 5;
    dp.order[2] = 2;
    dp.order[3] = 7;
    dp.passes[0] = .{ .id_lo = 0, .id_hi = 2, .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
    dp.passes[1] = .{ .id_lo = 2, .id_hi = 4, .m_lo = 0, .m_n = 0, .i_lo = 0, .i_n = 0 };
    dp.n_passes = 2;
    var members: [16]i32 = undefined;
    var items: [3 * 16]i32 = undefined;
    const c = build(&dp, &picks, &members, &items);
    try std.testing.expectEqual(@as(usize, 9), c.members);
    try std.testing.expectEqual(@as(usize, 4), c.items);
    try std.testing.expectEqualSlices(i32, &.{ 1, 4, 7, 0, 6, 8, 2, 3, 5 }, members[0..9]);
    try std.testing.expectEqual(Pass{ .id_lo = 0, .id_hi = 2, .m_lo = 0, .m_n = 6, .i_lo = 0, .i_n = 2 }, dp.passes[0]);
    try std.testing.expectEqual(Pass{ .id_lo = 2, .id_hi = 4, .m_lo = 6, .m_n = 3, .i_lo = 2, .i_n = 2 }, dp.passes[1]);
    // (expert, first within its pass, count)
    try std.testing.expectEqualSlices(i32, &.{ 512, 0, 3, 5, 3, 3, 2, 0, 2, 7, 2, 1 }, items[0..12]);
}

test "exl3: expert-parallel prefill masks the other rank's picks and keeps the shared expert on rank 1" {
    const r0 = try tp_.Rank.init(2, 0);
    const r1 = try tp_.Rank.init(2, 1);
    const picks = [_]i32{ 5, 300, 512, -1, 255, 256, 511, 0 };
    var a = picks;
    maskOwned(&a, r0);
    try std.testing.expectEqualSlices(i32, &.{ 5, -1, -1, -1, 255, -1, -1, 0 }, &a);
    var b = picks;
    maskOwned(&b, r1);
    try std.testing.expectEqualSlices(i32, &.{ -1, 300, 512, -1, -1, 256, 511, -1 }, &b);
    var c = picks;
    maskOwned(&c, null);
    try std.testing.expectEqualSlices(i32, &picks, &c);
    // every pick lands on exactly one rank
    for (picks, a, b) |p, x, y| if (p >= 0) try std.testing.expect((x == p) != (y == p));
    // the sum buffer exists only at world 2
    try std.testing.expectEqual(deviceBytesAt(4096, 1), deviceBytes(4096));
    try std.testing.expect(deviceBytesAt(4096, 2) >= deviceBytes(4096) + 4096 * dims * 4);
}

test "exl3: prefetch history follows deep engine ownership reuse replacement and release" {
    const Fixture = struct {
        fn unload(_: cuda.abi.Module) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn stream(_: cuda.abi.Stream) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn freeHost(_: ?*anyopaque) callconv(.c) cuda.abi.Result {
            return cuda.abi.success;
        }
        fn refuse(_: *cuda.abi.DevicePtr, _: usize) callconv(.c) cuda.abi.Result {
            return cuda.abi.error_not_ready;
        }
        fn install(driver: *const cuda.Driver, owner: *const anyopaque) !*Deep {
            const buffers = try std.testing.allocator.create(Deep);
            buffers.* = .{
                .gpa = std.testing.allocator,
                .owner = owner,
                .rows = 4096,
                .lay = undefined,
                .hl = undefined,
                .dev = .{ .d = driver, .ptr = 0, .len = 0 },
                .host = .{ .d = driver, .bytes = @constCast(&.{}) },
                .kern = undefined,
                .aux = .{ .d = driver, .handle = null },
            };
            buffers.kern.mod = .{ .d = driver, .handle = null };
            deep = buffers;
            return buffers;
        }
    };
    try std.testing.expect(deep == null);
    defer release();
    var driver: cuda.Driver = undefined;
    driver.api.cuModuleUnload = Fixture.unload;
    driver.api.cuStreamSynchronize = Fixture.stream;
    driver.api.cuStreamDestroy_v2 = Fixture.stream;
    driver.api.cuMemFreeHost = Fixture.freeHost;
    driver.api.cuMemAlloc_v2 = Fixture.refuse;
    var owners: [2]u8 = .{ 0, 0 };
    const first = try Fixture.install(&driver, &owners[0]);
    first.history[3] = .{ .rows = 4096, .used = 80 };
    first.history_shape = .{ .sub = 256, .lo = 0, .hi = 256 };
    releaseFor(&owners[1]);
    try std.testing.expect(deep.? == first);
    try reserveAt(std.testing.allocator, &driver, &owners[0], 2048, 1);
    try std.testing.expectEqual(@as(usize, 96), history_.want(deep.?.history[3], 4096, 400, routed_n_));
    try std.testing.expectError(error.NotReady, reserveAt(std.testing.allocator, &driver, &owners[1], 4096, 1));
    try std.testing.expect(deep == null);
    const second = try Fixture.install(&driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(second.history[3], 4096, 400, routed_n_));
    try std.testing.expect(second.history_shape == null);
    second.history[3] = .{ .rows = 4096, .used = 80 };
    try std.testing.expectError(error.NotReady, reserveAt(std.testing.allocator, &driver, &owners[1], 4096, 2));
    try std.testing.expect(deep == null);
    const third = try Fixture.install(&driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(third.history[3], 4096, 400, routed_n_));
    third.history[3] = .{ .rows = 4096, .used = 80 };
    releaseFor(&owners[1]);
    try std.testing.expect(deep == null);
    const fourth = try Fixture.install(&driver, &owners[1]);
    try std.testing.expectEqual(@as(usize, 416), history_.want(fourth.history[3], 4096, 400, routed_n_));
}
