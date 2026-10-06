//! Startup memory plan (Python's serving/memory.plan): the window one request can use and the bytes the prompt cache
//! may hold, from the GPU's free memory after the weights, a reserve, the engine's scratch and every lane's caches.

const std = @import("std");
const config = @import("config.zig");
const fwd = @import("forward.zig");

pub const gib: usize = 1 << 30;

/// Memory kept free in a pool: max(4 GiB, a tenth of the GPU's).
pub fn reserve(total: usize) usize {
    return @max(4 * gib, total / 10);
}

/// One stream's caches at `tokens` positions: keys and values of the attention layers plus the linear layers' state.
pub fn stateBytes(s: config.Spec, act_bytes: usize, tokens: usize) usize {
    var full: usize = 0;
    for (0..s.n_layers) |i| full += @intFromBool(s.full(i));
    const linear = s.n_layers - full;
    const keys = full * 2 * s.kv_heads * s.head_dim * act_bytes * tokens;
    const conv = (s.keyWidth() * 2 + s.valueWidth()) * (s.conv - 1) * 4;
    const recurrent = s.value_heads * s.value_dim * s.key_dim * 4;
    return keys + linear * (conv + recurrent);
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
    /// Bytes the kept prompt copies may hold together.
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
    /// Streams served at once; one more copy of a prompt is kept beside them.
    streams: usize,
    /// Rows of a shared forward, and the extra positions a verify writes past a reply.
    rows: usize,
    slack: usize,
    /// The window asked for (the model's, or --context).
    target: usize,
    free: usize,
    total: usize,
};

/// The largest window at most `target` for which the scratch, every lane's caches and one kept copy fit; the window
/// is 0 when not even one token does.
pub fn plan(in: Input) Plan {
    const s = in.spec;
    const keep = reserve(in.total);
    const room = in.free -| keep;
    const copies = in.streams + 1;
    var lo: usize = 0;
    var hi: usize = in.target;
    while (lo < hi) {
        const mid = lo + (hi - lo + 1) / 2;
        const cap = mid + in.slack;
        if (Scratch.of(s, cap, in.rows).total() + copies * stateBytes(s, in.act_bytes, cap) <= room) lo = mid else hi = mid - 1;
    }
    const cap = lo + in.slack;
    const scratch = Scratch.of(s, cap, in.rows).total();
    return .{
        .window = lo,
        .capacity = cap,
        .cache_budget = room -| scratch -| in.streams * stateBytes(s, in.act_bytes, cap),
        .weights = in.total -| in.free,
        .total = in.total,
        .reserve = keep,
        .scratch = scratch,
    };
}
