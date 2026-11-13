//! Host-side hand-off producer: a finished prefill's engine snapshot and pool pages become a protocol-2 export.
const std = @import("std");
const fabric = @import("fabric");
const g = @import("handoff/geometry.zig");
const plan = @import("handoff/plan.zig");
const ex = @import("handoff/export.zig");
const p2 = @import("handoff/p2.zig");
const lsplit = @import("handoff/split.zig");
const producer = fabric.producer;

pub const Error = ex.Error || error{ BadSnapshot, NotAHandoffPrompt, NotASplitExport };

/// One bounded role of the engine's snapshot: its name and this slot's bytes (stale drafter rows are zeroed in place).
pub const Piece = struct { role: []const u8, bytes: []u8 };

/// A KV source's comp and index pages of the slot, logical page 0 first, back to back (a replicated pool's view).
pub const Pages = struct { layer: u32, comp: []const u8, index: []const u8 };

pub const State = struct {
    /// the position the slot stands at: the prompt's tokens before its last
    pos: u64,
    /// the drafter's key rows the snapshot holds valid: positions [pos - ds_rows, pos)
    ds_rows: u64,
    index_format: ex.IndexFormat = .fp8,
    pieces: []const Piece,
    pages: []const Pages,
    /// CED replay: layers from here hold `replay_rows` replayed rows; the window's older ones are another prompt's
    decoder_from: u32 = none,
    replay_rows: u32 = 0,
    /// KV_SPLIT: this producer is shard `shard` of `shards`.
    shard: u32 = 0,
    shards: u32 = 1,
    /// a layer-split state (0: whole): only layers [0, cut) are valid, no drafter rings ride and pages above are absent
    cut: u32 = 0,

    /// `decoder_from` of a state with no replayed layers
    pub const none: u32 = std.math.maxInt(u32);
};

/// A state read back from the engine, held until `Source.release`.
pub const Captured = struct { ctx: *anyopaque, state: State };

/// A prompt's pool pages read back while it prefills: `pages` are full-size buffers `land` fills in logical page order.
pub const Feed = struct {
    ptr: *anyopaque,
    pages: []const Pages,
    /// Engine thread, slot at `at` tokens: queues the whole pages not queued yet (all if `final`); the job, or null.
    queue: *const fn (ptr: *anyopaque, at: u64, final: bool) anyerror!?*anyopaque,
    /// Worker thread: waits for the queued read `job` and returns the pages now valid in `pages` (below that page).
    land: *const fn (ptr: *anyopaque, job: *anyopaque) anyerror!u64,
    close: *const fn (ptr: *anyopaque) void,
};

/// What a streamed export is planned from before its prompt has prefilled.
pub const Plan = struct { ds_rows: u32, index_format: ex.IndexFormat, shard: u32 = 0, shards: u32 = 1 };

/// Where a state comes from: the engine's active slot at `at` tokens (a device read), or a test's synthetic one.
pub const Source = struct {
    ptr: *anyopaque,
    take: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator, at: u64) anyerror!Captured,
    release: *const fn (ctx: *anyopaque) void,
    /// Streaming (null: this source exports at the cut only): the plan and page feed for a prompt cut at `at` tokens.
    open_feed: ?*const fn (ptr: *anyopaque, gpa: std.mem.Allocator, at: u64, plan: *Plan) anyerror!Feed = null,
    /// `take` without the pool pages (a feed read them): the snapshot, rings and carries at the cut.
    take_state: ?*const fn (ptr: *anyopaque, gpa: std.mem.Allocator, at: u64, feed: Feed) anyerror!Captured = null,
    /// The same at `at` tokens for a layer-split export of layers [0, cut): its pages and drafter rows are not read.
    take_cut: ?*const fn (ptr: *anyopaque, gpa: std.mem.Allocator, at: u64, cut: u32) anyerror!Captured = null,
};

/// An export and everything it reads: the converted parts live in `arena` until `deinit`.
pub const Built = struct {
    arena: std.heap.ArenaAllocator,
    exp: producer.Export,
    /// the parts' bytes: what a table counts against its cap
    bytes: u64,
    /// the parts in plan order, `exp.layers` over them (a streamed export converts them one by one)
    parts: []ex.Part = &.{},
    /// a layer-split export's rows and state still arriving (null: a finished export)
    early: ?*Early = null,

    /// Frees the arena and the frames a responder planned into `exp` with `gpa`; the caller destroys no `Built` itself.
    pub fn deinit(b: *Built, gpa: std.mem.Allocator) void {
        if (b.exp.frames.len > 0) gpa.free(b.exp.frames);
        if (b.early) |e| {
            e.stage.deinit();
            e.latch.deinit();
            gpa.destroy(e);
        }
        b.arena.deinit();
        gpa.destroy(b);
    }
};

fn piece(pieces: []const Piece, layer: u32, part: []const u8) ?[]u8 {
    var buf: [48]u8 = undefined;
    const role = std.fmt.bufPrint(&buf, "s.L{d}.{s}", .{ layer, part }) catch return null;
    for (pieces) |p| if (std.mem.eql(u8, p.role, role)) return p.bytes;
    return null;
}

/// A layer's ring (values [rows][576], scales [rows][8]); null when the snapshot holds none for it.
fn ringOf(pieces: []const Piece, layer: u32, kind: []const u8, rows: *u32) Error!?ex.Ring {
    var vb: [8]u8 = undefined;
    var sb: [8]u8 = undefined;
    const v = piece(pieces, layer, std.fmt.bufPrint(&vb, "{s}.v", .{kind}) catch unreachable) orelse return null;
    const s = piece(pieces, layer, std.fmt.bufPrint(&sb, "{s}.s", .{kind}) catch unreachable) orelse return null;
    if (v.len == 0 or v.len % g.cuda_values_bytes != 0 or v.len / g.cuda_values_bytes != s.len / g.cuda_scale_bytes or s.len % g.cuda_scale_bytes != 0) return error.BadSnapshot;
    const n: u32 = @intCast(v.len / g.cuda_values_bytes);
    if (rows.* != 0 and rows.* != n) return error.BadSnapshot; // every ring of a kind has the engine's one row count
    rows.* = n;
    return .{ .values = v, .scales = s };
}

/// Zero the ring slots of positions [from, to): rows the snapshot never held, which would carry another prompt's.
fn clearRows(r: ex.Ring, rows: u32, from: u64, to: u64) void {
    for (from..to) |pos| {
        const slot = pos % rows;
        @memset(@constCast(r.values[slot * g.cuda_values_bytes ..][0..g.cuda_values_bytes]), 0);
        @memset(@constCast(r.scales[slot * g.cuda_scale_bytes ..][0..g.cuda_scale_bytes]), 0);
    }
}

/// A replayed layer's ring holds the last `replay_rows` positions only: the window's older row is zeroed, not stale.
fn clearUnreplayed(st: State, swa: []const ?ex.Ring, rows: u32) Error!void {
    if (st.replay_rows == 0 or rows == 0) return;
    const window: u64 = @min(st.pos, g.release.window);
    const from = st.pos - window;
    const to = st.pos - @min(st.pos, st.replay_rows);
    for (swa[@min(st.decoder_from, swa.len)..]) |maybe| clearRows(maybe orelse return error.MissingRing, rows, from, @max(from, to));
}

/// The sources' pool pages by rank: a split shard knows its own rank's bytes only (the other ranks' stay empty).
pub fn sourcesOf(arena: std.mem.Allocator, st: State) Error![]ex.SourceState {
    const geo = g.release;
    const split = st.shards > 1;
    if (st.shards == 0 or st.shard >= st.shards or st.shards > plan.max_world) return error.BadShape;
    const sources = try arena.alloc(ex.SourceState, geo.sources.len);
    for (geo.sources, sources) |s, *x| {
        const pg = for (st.pages) |p| {
            if (p.layer == s.layer) break p;
        } else if (st.cut != 0 and s.layer >= st.cut) Pages{ .layer = s.layer, .comp = &.{}, .index = &.{} } else return error.MissingSource;
        const comp = try arena.alloc([]const u8, st.shards);
        const index = try arena.alloc([]const u8, st.shards);
        @memset(comp, &.{});
        @memset(index, &.{});
        comp[st.shard] = pg.comp;
        index[st.shard] = pg.index;
        x.* = .{ .layer = s.layer, .comp = .{ .split = split, .ranks = comp }, .index = .{ .split = split, .ranks = index } };
    }
    return sources;
}

/// The library's view of `st`: slices over the snapshot's bytes, no copy; `ds_rows` rides (0: no drafter rings).
pub fn cudaOf(arena: std.mem.Allocator, st: State, ds_rows: *u32) Error!ex.Cuda {
    const geo = g.release;
    var out: ex.Cuda = .{ .index_format = st.index_format, .sources = &.{}, .swa = &.{}, .carries = &.{} };
    ds_rows.* = 0;
    if (st.shard != 0) { // the other shards carry pool pages only: no ring, carry or drafter row
        out.sources = try sourcesOf(arena, st);
        return out;
    }
    var ring_rows: u32 = 0;
    const swa = try arena.alloc(?ex.Ring, geo.layers);
    for (swa, 0..) |*r, l| r.* = try ringOf(st.pieces, @intCast(l), "swa", &ring_rows);
    if (ring_rows != 0) out.rings.rows = ring_rows;
    try clearUnreplayed(st, swa, ring_rows);
    out.swa = swa;
    const carries = try arena.alloc([]const u8, geo.sources.len);
    for (geo.sources, carries) |s, *c| {
        const above = st.cut != 0 and s.layer >= st.cut; // a layer-split state has no carry past the cut: none is read
        c.* = if (s.ratio == 2 and !above) (piece(st.pieces, s.layer, "carry") orelse return error.MissingCarry) else &.{};
    }
    out.carries = carries;
    out.sources = try sourcesOf(arena, st);
    // the drafter's three rings ride whole or not at all; a short run of valid rows is zero-filled below it
    if (st.cut != 0) return out; // a layer-split state has none
    var drows: u32 = 0;
    const drafter = try arena.alloc(ex.Ring, geo.mtp_layers);
    var found: u32 = 0;
    for (drafter, 0..) |*r, i| {
        r.* = (try ringOf(st.pieces, geo.layers + @as(u32, @intCast(i)), "ring", &drows)) orelse break;
        found += 1;
    }
    ds_rows.* = 0;
    if (found == geo.mtp_layers) {
        out.drafter = drafter;
        out.rings.drafter_rows = drows;
        const want: u32 = @intCast(@min(st.pos, geo.window));
        if (want > drows) return error.BadSnapshot;
        ds_rows.* = want;
        const valid = @min(st.ds_rows, want);
        for (drafter) |r| clearRows(r, drows, st.pos - want, st.pos - valid);
    }
    return out;
}

/// The export of `st` under handoff `id` for `tokens` (the prompt before its last token): parts, layers, sources.
pub fn build(gpa: std.mem.Allocator, id: [16]u8, tokens: []const u32, st: State) Error!*Built {
    if (tokens.len == 0 or st.pos != tokens.len) return error.NotAHandoffPrompt;
    const b = try gpa.create(Built);
    errdefer gpa.destroy(b);
    b.arena = .init(gpa);
    errdefer b.arena.deinit();
    b.early = null;
    const a = b.arena.allocator();
    var ds: u32 = 0;
    const cuda = try cudaOf(a, st, &ds);
    if (st.cut != 0) return error.NotAHandoffPrompt; // a layer-split state goes through `Early`, into its early buffers
    const shape: plan.Shape = .{ .upto = st.pos, .ds_rows = ds, .shard = st.shard, .shards = st.shards, .split = st.shards > 1 };
    const parts = try ex.build(a, shape, cuda);
    try lay(b, id, tokens, parts);
    return b;
}

fn lay(b: *Built, id: [16]u8, tokens: []const u32, parts: []ex.Part) Error!void {
    const a = b.arena.allocator();
    var bytes: u64 = 0;
    for (parts) |p| bytes += p.bytes.len;
    const laid = try p2.layers(a, parts);
    b.exp = .{ .handoff = id, .tokens = try a.dupe(u32, tokens), .first_token = 0, .tokens_per_row = 1, .layers = laid.layers, .sources = laid.sources };
    b.bytes = bytes;
    b.parts = parts;
}

/// A streamed export's shell: the parts of `shape` allocated and laid out, none converted yet.
pub fn shell(gpa: std.mem.Allocator, id: [16]u8, tokens: []const u32, shape: plan.Shape) Error!*Built {
    if (tokens.len == 0 or shape.upto != tokens.len) return error.NotAHandoffPrompt;
    const b = try gpa.create(Built);
    errdefer gpa.destroy(b);
    b.arena = .init(gpa);
    errdefer b.arena.deinit();
    b.early = null;
    try lay(b, id, tokens, try ex.alloc(b.arena.allocator(), shape));
    return b;
}

/// The bytes of the parts a hand-off at `upto` tokens holds (what `build` allocates), for the table's cap.
pub fn estimate(upto: u64) u64 {
    const geo = g.release;
    const window: u64 = @min(upto, geo.window);
    var n: u64 = (geo.layers + geo.mtp_layers) * window * geo.head_dim * 2;
    for (geo.sources) |s| {
        n += upto / s.ratio * (geo.head_dim + geo.index_dim) * 2;
        if (s.ratio == 2) n += 2 * geo.head_dim * 4;
    }
    return n;
}

/// A layer-split export's live parts: boundary rows still being produced and state parts landed at the cut.
pub const Early = struct {
    /// the boundary stream (`lsplit.row_bytes` a row), published chunk by chunk
    stage: fabric.row_stage.Stage,
    /// the state parts' latch: published once they are in `buffers`
    latch: fabric.row_stage.Stage,
    cut: u32,
    upto: u64,
    /// the state parts' bytes in manifest order (the export's sources read these), owned by the `Built`'s arena
    buffers: []const []u8,

    /// The producer gave up: every reader of the rows or the state is released with the producer's failure.
    pub fn fail(e: *Early) void {
        e.stage.fail();
        e.latch.fail();
    }
};

/// The early export of a split hand-off of `tokens` cut at `cut`; boundary rows are held `window` at a time.
pub fn early(gpa: std.mem.Allocator, id: [16]u8, tokens: []const u32, cut: u32, window: u64, wait_ns: u64) !*Built {
    const b = try gpa.create(Built);
    errdefer gpa.destroy(b);
    b.arena = .init(gpa);
    errdefer b.arena.deinit();
    const a = b.arena.allocator();
    const upto: u64 = tokens.len;
    const layers = try lsplit.describe(a, upto, cut);
    const sources = try a.alloc(fabric.frames.Source, layers.len);
    const e = try gpa.create(Early);
    errdefer gpa.destroy(e);
    e.stage = try fabric.row_stage.Stage.initRing(gpa, upto, lsplit.row_bytes, window, 0);
    errdefer e.stage.deinit();
    e.latch = try fabric.row_stage.Stage.init(gpa, 1, 0);
    errdefer e.latch.deinit();
    sources[0] = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 2, .rows = &.{}, .stage = &e.stage, .wait_ns = wait_ns };
    var count: usize = 0;
    for (layers[1..]) |l| count += l.parts.len;
    const buffers = try a.alloc([]u8, count);
    var at: usize = 0;
    var state_bytes: u64 = 0;
    for (layers[1..], sources[1..]) |l, *src| {
        const pieces = try a.alloc([]const u8, l.parts.len);
        for (l.parts, pieces) |p, *slot| {
            buffers[at] = try a.alloc(u8, @intCast(p.bytes()));
            state_bytes += buffers[at].len;
            slot.* = buffers[at];
            at += 1;
        }
        src.* = .{ .bytes = &.{}, .shape = &.{}, .block_dim = 0, .elem = 1, .rows = &.{}, .parts = pieces, .latch = &e.latch, .wait_ns = wait_ns };
    }
    e.cut = cut;
    e.upto = upto;
    e.buffers = buffers;
    b.exp = .{ .handoff = id, .tokens = try a.dupe(u32, tokens), .first_token = 0, .tokens_per_row = 1, .layers = layers, .sources = sources };
    b.bytes = e.stage.cap * lsplit.row_bytes + state_bytes;
    b.early = e;
    return b;
}

/// The state of layers [0, cut) converted into the early export's buffers, then released to the readers.
pub fn land(gpa: std.mem.Allocator, b: *Built, st: State) (Error || fabric.row_stage.Stage.Error)!void {
    const e = b.early orelse return error.NotASplitExport;
    if (st.cut != e.cut or st.pos != e.upto or st.shard != 0 or st.shards != 1) return error.NotAHandoffPrompt;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    var ds: u32 = 0;
    const cuda = try cudaOf(a, st, &ds);
    _ = try ex.buildInto(a, lsplit.stateShape(st.pos, st.cut), cuda, e.buffers);
    try e.latch.publish(1);
}
