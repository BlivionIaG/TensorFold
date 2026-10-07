//! Startup memory plan: the window one request can use, the bytes the prompt cache may hold and the pages of the pool.

const std = @import("std");
const config = @import("../../../weights/config.zig");
const fwd = @import("../forward/forward.zig");
const pages = @import("../forward/pages.zig");

pub const gib: usize = 1 << 30;

/// Memory kept free in a pool: max(4 GiB, a tenth of the GPU's).
pub fn reserve(total: usize) usize {
    return @max(4 * gib, total / 10);
}

/// Bytes of one page of the pool: the keys and values of 64 positions in every attention layer.
pub fn pageBytes(s: config.Spec, act_bytes: usize) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    return full * 2 * s.kv_heads * s.head_dim * act_bytes * pages.tokens;
}

/// One stream's (or one snapshot's) linear state: the conv window and the recurrent state of every linear layer.
pub fn linearBytes(s: config.Spec) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    const conv = (s.keyWidth() * 2 + s.valueWidth()) * (s.conv - 1) * 4;
    const recurrent = s.value_heads * s.value_dim * s.key_dim * 4;
    return (s.n_layers - full) * (conv + recurrent);
}

/// Bytes a copy of one stream's caches at `tokens` positions holds: keys and values of the attention layers plus the linear layers' state.
pub fn stateBytes(s: config.Spec, act_bytes: usize, tokens: usize) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    return full * 2 * s.kv_heads * s.head_dim * act_bytes * tokens + linearBytes(s);
}

/// The engine's device scratch for streams of `capacity` positions and shared forwards of `rows` rows.
pub const Scratch = struct {
    /// A window's scratch, kept from its verify to its keep.
    rounds: usize,
    /// A prompt's residual and final rows, plus one SPAN step's temporaries.
    prompts: usize,
    /// The token ids and row positions a forward reads.
    ids: usize,

    pub fn of(s: config.Spec, capacity: usize, rows: usize) Scratch {
        // causal_at's scores over a whole cache a query, then a forward's activations, plans and expert products
        const per_row = s.heads * capacity * 4 + 64 * s.hidden * 4 + (s.top_k + 1) * (3 * @max(s.moe_width, 1) + s.hidden) * 4 * 2;
        // a window of several rows keeps every linear layer's conv and DeltaNet state after each row
        var linear: usize = 0;
        for (0..s.n_layers) |i| linear += @intFromBool(!s.full(i));
        const snapshot = linear * ((s.conv - 1) * (2 * s.keyWidth() + s.valueWidth()) + s.value_heads * s.value_dim * s.key_dim) * 4;
        return .{
            .rounds = (256 << 20) + rows * (per_row * 2 + snapshot),
            .prompts = (768 << 20) + 2 * capacity * s.hidden * 2 + fwd.SPAN * per_row / 4,
            .ids = @max(capacity, 2 * rows) * 4,
        };
    }

    pub fn total(c: Scratch) usize {
        return c.rounds + c.prompts + c.ids;
    }
};

pub const Plan = struct {
    /// Prompt plus reply tokens one request may use.
    window: usize,
    /// Positions the engine's buffers hold: the window and a verify's rows.
    capacity: usize,
    /// Bytes the prefix tree may hold together: its pages and its linear snapshots.
    cache_budget: usize,
    /// The GPU's memory the weights and the runtime hold, its size, the reserve, the scratch.
    weights: usize,
    total: usize,
    reserve: usize,
    scratch: usize,
};

pub const Input = struct {
    spec: config.Spec,
    act_bytes: usize,
    /// Streams served at once, each with the pages of a whole window; the prefix tree gets one window's worth beside them.
    streams: usize,
    /// Rows of a shared forward, and the extra positions a verify writes past a reply.
    rows: usize,
    slack: usize,
    /// The window asked for (the model's, or --context).
    target: usize,
    free: usize,
    total: usize,
};

/// What the pool of pages and the prefix tree hold, from the bytes the tree may use.
pub const Pool = struct {
    /// Pages of the pool: every stream's whole window, the scratch rows and the tree's pages.
    pages: usize,
    /// Pages the tree may hold.
    cache_pages: usize,
    /// Linear snapshots the tree may hold.
    snaps: usize,
};

/// The pool for `budget` bytes of prefix tree and at most `slots` snapshots, which may take up to half of it.
pub fn pool(s: config.Spec, act_bytes: usize, streams: usize, rows: usize, capacity: usize, budget: usize, slots: usize) Pool {
    const snap = @max(linearBytes(s), 1);
    const snaps = @min(slots, budget / 2 / snap);
    const page = @max(pageBytes(s, act_bytes), 1);
    const cache_pages = (budget - snaps * snap) / page;
    return .{ .pages = streams * pages.pagesFor(capacity) + pages.pagesFor(rows) + cache_pages, .cache_pages = cache_pages, .snaps = snaps };
}

/// Bytes the streams hold at a window of `capacity`: their linear state and a page for each 64 positions.
fn streamBytes(s: config.Spec, act_bytes: usize, streams: usize, rows: usize, capacity: usize) usize {
    return streams * (linearBytes(s) + pages.pagesFor(capacity) * pageBytes(s, act_bytes)) + pages.pagesFor(rows) * pageBytes(s, act_bytes);
}

/// The largest window at most `target` for which the scratch, every lane's caches and a window's worth of prefix tree fit; 0 if none.
pub fn plan(in: Input) Plan {
    const s = in.spec;
    const keep = reserve(in.total);
    const room = in.free -| keep;
    var lo: usize = 0;
    var hi: usize = in.target;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        const cap = mid + in.slack;
        if (Scratch.of(s, cap, in.rows).total() + streamBytes(s, in.act_bytes, in.streams + 1, in.rows, cap) <= room) lo = mid else hi = mid - 1;
    }
    const cap = lo + in.slack;
    const scratch = Scratch.of(s, cap, in.rows).total();
    return .{
        .window = lo,
        .capacity = cap,
        .cache_budget = room -| scratch -| streamBytes(s, in.act_bytes, in.streams, in.rows, cap),
        .weights = in.total -| in.free,
        .total = in.total,
        .reserve = keep,
        .scratch = scratch,
    };
}
