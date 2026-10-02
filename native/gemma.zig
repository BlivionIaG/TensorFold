const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const ops = @import("gemma_ops.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const kv = @import("kv_buffer.zig");

const OlderCache = struct {
    keys: A = mx.empty,
    values: A = mx.empty,
    position: i32 = 0,
    fn deinit(c: *OlderCache) void {
        mx.free(c.keys);
        mx.free(c.values);
        c.* = .{};
    }
};
const RecentCache = struct {
    keys: A = mx.empty,
    values: A = mx.empty,
    position: i32 = 0,
    backing_bytes: [2]u64 = @splat(0),
    fn deinit(c: *RecentCache) void {
        mx.free(c.keys);
        mx.free(c.values);
        c.* = .{};
    }
};
pub const CacheStorage = struct {
    keys: kv.Buffer = .{},
    values: kv.Buffer = .{},
    ring: [2]OlderCache = @splat(.{}),
    ring_count: usize = 0,
    recent: [2]RecentCache = @splat(.{}),
    recent_count: usize = 0,
    pipelined: bool = false,
    backing_bytes: [2]u64 = @splat(0),
    donors: [2]usize = @splat(0),
    fn deinit(c: *CacheStorage) void {
        c.keys.deinit();
        c.values.deinit();
        for (&c.ring) |*older| older.deinit();
        for (&c.recent) |*recent| recent.deinit();
        c.* = .{};
    }
};

pub const Cache = struct {
    keys: A = mx.empty,
    values: A = mx.empty,
    storage: CacheStorage = .{},
    pub fn clone(c: Cache) !Cache {
        var out = Cache{};
        errdefer out.deinit();
        inline for (.{ "keys", "values" }, 0..) |field, i| {
            const value = @field(c, field);
            if (value.ctx != null) @field(out, field) = try mx.retain(value);
            const capacity = @field(c.storage, field).current;
            out.storage.backing_bytes[i] = @max(c.storage.backing_bytes[i], bytes(if (capacity.ctx != null) capacity else value));
        }
        return out;
    }
    pub fn deinit(c: *Cache) void {
        mx.free(c.keys);
        mx.free(c.values);
        c.storage.deinit();
        c.* = .{};
    }
    pub fn nbytes(c: Cache) u64 {
        var total: u64 = 0;
        inline for (.{ "keys", "values" }, 0..) |field, i| {
            const buffer = @field(c.storage, field);
            total +|= if (buffer.current.ctx != null) bytes(buffer.current) else @max(bytes(@field(c, field)), c.storage.backing_bytes[i]);
            total +|= bytes(buffer.spare) +| bytes(buffer.recent);
        }
        for (c.storage.ring) |older| total +|= bytes(older.keys) +| bytes(older.values);
        for (c.storage.recent) |recent| total +|= @max(bytes(recent.keys), recent.backing_bytes[0]) +| @max(bytes(recent.values), recent.backing_bytes[1]);
        return total;
    }
    fn bytes(a: A) u64 {
        return if (a.ctx == null) 0 else mx.c.mlx_array_nbytes(a);
    }
    pub fn observe(c: Cache) !void {
        if (!kv.track_reuse) return;
        inline for (.{ "keys", "values" }, 0..) |field, i| {
            const buffer = @field(c.storage, field);
            try kv.observe(.{ .capacity = if (buffer.current.ctx != null) buffer.current else @field(c, field), .donor = c.storage.donors[i] });
        }
    }
    pub fn attention(c: Cache, comptime field: []const u8) A {
        const buffer = @field(c.storage, field);
        return if (buffer.current.ctx != null) buffer.current else @field(c, field);
    }
    fn dropSpares(c: *Cache) void {
        for (&c.storage.ring) |*older| older.deinit();
        for (&c.storage.recent) |*recent| recent.deinit();
        c.storage.ring_count = 0;
        c.storage.recent_count = 0;
        inline for (.{ "keys", "values" }) |field| {
            const buffer = &@field(c.storage, field);
            mx.free(buffer.spare);
            mx.free(buffer.recent);
            buffer.spare = mx.empty;
            buffer.recent = mx.empty;
        }
    }
    fn pipeline(c: *Cache) !void {
        if (c.storage.pipelined) return;
        const keys = &c.storage.keys;
        const values = &c.storage.values;
        if (c.storage.ring_count != 0 or c.storage.recent_count != 0 or (keys.spare.ctx == null) != (values.spare.ctx == null)) return error.InvalidCacheState;
        if (keys.current.ctx == null or values.current.ctx == null or keys.offset != values.offset or keys.axis != 2 or values.axis != 2) return error.InvalidCacheState;
        if (keys.spare.ctx != null) {
            if (keys.spare_end != values.spare_end or keys.spare_end < 0 or keys.spare_end >= keys.offset or keys.recent.ctx == null or values.recent.ctx == null) return error.InvalidCacheState;
            if (mx.dim(keys.recent, 2) != keys.offset - keys.spare_end or mx.dim(values.recent, 2) != values.offset - values.spare_end) return error.InvalidCacheState;
            c.storage.ring[0] = .{ .keys = keys.spare, .values = values.spare, .position = keys.spare_end };
            c.storage.ring_count = 1;
            c.storage.recent[0] = .{ .keys = keys.recent, .values = values.recent, .position = keys.spare_end, .backing_bytes = .{ bytes(keys.recent), bytes(values.recent) } };
            c.storage.recent_count = 1;
            keys.spare = mx.empty;
            values.spare = mx.empty;
        } else {
            mx.free(keys.recent);
            mx.free(values.recent);
        }
        keys.recent = mx.empty;
        values.recent = mx.empty;
        c.storage.pipelined = true;
    }
    fn prepareBuffered(c: *Cache, s: *mx.Scope, indices: *RingIndices, added: Cache, position: i32, backing_bytes: [2]u64, ring: bool) !Cache {
        var next = Cache{};
        errdefer next.deinit();
        next.storage.pipelined = !ring;
        if (!ring) inline for (.{ "keys", "values" }) |field| {
            const buffer = @field(c.storage, field);
            if (buffer.current.ctx != null and (buffer.offset != position or buffer.axis != 2)) return error.InvalidCacheState;
        };
        var donor = OlderCache{};
        defer donor.deinit();
        var since = position;
        if (c.storage.ring_count == 2) {
            const index: usize = if (c.storage.ring[0].position <= c.storage.ring[1].position) 0 else 1;
            donor = c.storage.ring[index];
            c.storage.ring[index] = c.storage.ring[1];
            c.storage.ring[1] = .{};
            c.storage.ring_count = 1;
            since = donor.position;
        }
        if (since < 0 or since > position or c.storage.recent_count > c.storage.recent.len) return error.InvalidCacheState;
        for (c.storage.ring[0..c.storage.ring_count]) |older| {
            const slot = &next.storage.ring[next.storage.ring_count];
            slot.position = older.position;
            slot.keys = try mx.retain(older.keys);
            slot.values = try mx.retain(older.values);
            next.storage.ring_count += 1;
        }
        if (c.keys.ctx != null) {
            const slot = &next.storage.ring[next.storage.ring_count];
            slot.position = position;
            slot.keys = try mx.retain(if (ring) c.keys else c.attention("keys"));
            slot.values = try mx.retain(if (ring) c.values else c.attention("values"));
            next.storage.ring_count += 1;
        }
        var oldest = position + mx.dim(added.keys, 2);
        for (next.storage.ring[0..next.storage.ring_count]) |older| oldest = @min(oldest, older.position);
        for (c.storage.recent[0..c.storage.recent_count]) |recent| {
            if (recent.position + mx.dim(recent.keys, 2) <= oldest) continue;
            if (next.storage.recent_count == next.storage.recent.len) return error.InvalidCacheState;
            const slot = &next.storage.recent[next.storage.recent_count];
            slot.position = recent.position;
            slot.backing_bytes = recent.backing_bytes;
            slot.keys = try mx.retain(recent.keys);
            slot.values = try mx.retain(recent.values);
            next.storage.recent_count += 1;
        }
        if (oldest < position + mx.dim(added.keys, 2)) {
            if (next.storage.recent_count == next.storage.recent.len) return error.InvalidCacheState;
            const slot = &next.storage.recent[next.storage.recent_count];
            slot.position = position;
            slot.backing_bytes = backing_bytes;
            slot.keys = try mx.retain(added.keys);
            slot.values = try mx.retain(added.values);
            next.storage.recent_count += 1;
        }
        inline for (.{ "keys", "values" }, 0..) |field, j| {
            const rows = @field(added, field);
            var parts: [3]A = undefined;
            var count: usize = 0;
            var cursor = since;
            for (c.storage.recent[0..c.storage.recent_count]) |recent| {
                const part = @field(recent, field);
                const end = recent.position + mx.dim(part, 2);
                if (end <= since) continue;
                const skip = @max(0, since - recent.position);
                if (recent.position + skip != cursor or end > position) return error.InvalidCacheState;
                parts[count] = if (skip > 0) try s.slice(part, 2, skip, mx.dim(part, 2)) else part;
                count += 1;
                cursor = end;
            }
            if (cursor != position) return error.InvalidCacheState;
            parts[count] = rows;
            count += 1;
            const fill = if (count == 1) rows else try s.cat(parts[0..count], 2);
            const current = if (ring) @field(c.*, field) else c.attention(field);
            var target = if (@field(donor, field).ctx != null) @field(donor, field) else current;
            const end = position + mx.dim(rows, 2);
            const grown = !ring and (target.ctx == null or mx.dim(target, 2) < end);
            if (ring and target.ctx == null) target = try s.zeros(&.{ 1, 8, 1152, 256 }, mx.bf16);
            if (grown) {
                const zeros = try s.zeros(&.{ 1, 2, std.mem.alignForward(i32, end, 2048) - since, 512 }, mx.bf16);
                target = if (since == 0) zeros else try s.cat(&.{ try s.slice(target, 2, 0, since), zeros }, 2);
            }
            if (kv.track_reuse and @field(donor, field).ctx != null and !grown) {
                try mx.eval(target);
                next.storage.donors[j] = kv.address(target);
            }
            const capacity = if (ring) try ringWrite(s, indices, target, fill, since) else try ringPut(s, target, fill, try indices.get(s, since));
            @field(next, field) = try mx.retain(if (ring) capacity else try s.slice(capacity, 2, 0, end));
            if (!ring) @field(next.storage, field) = .{ .current = try mx.retain(capacity), .offset = end, .axis = 2 };
        }
        return next;
    }
};
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    taps: A = mx.empty,
    records: [30]Cache = @splat(.{}),
    record_bytes: [30][2]u64 = @splat(@splat(0)),
    staged: [30]Cache = @splat(.{}),
    staged_ready: bool = false,
    stage_indices: RingIndices = .{},
    position: i32,
    generation: u64,
    rows: usize,
    pub fn deinit(p: *Pass) void {
        for (&p.staged) |*cache| cache.deinit();
        p.scope.deinit();
    }
};
const Front = struct {
    kernels: mx.Kernels,
    geometry: ops.Geometry,
    qkv: [3]A,
    q_norm: A,
    k_norm: A,
    inverse: A,
    eps: A,
    group: i32,

    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        const p: *Front = @ptrCast(@alignCast(raw.?));
        p.kernels.deinit();
        mx.allocator.destroy(p);
    }
    fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *Front = @ptrCast(@alignCast(raw.?));
        return p.graph(out, ins) catch -1;
    }
    fn graph(p: *Front, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
        var s = mx.Scope{};
        defer s.deinit();
        var args: [2]A = undefined;
        for (&args, 0..) |*arg, j| {
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_vector_array_get(&value, ins, j);
            arg.* = try s.result(rc, value);
        }
        const values = try ops.qkv(&p.kernels, &s, p.geometry, args[0], p.qkv, p.q_norm, p.k_norm, p.inverse, args[1], p.eps, p.group);
        return mx.c.mlx_vector_array_set_data(out, &values, values.len);
    }
};
const Back = struct {
    kernels: mx.Kernels,
    o: [3]A,
    gate_up: [3]A,
    down: [3]A,
    router: [3]A,
    expert_gate: [3]A,
    expert_up: [3]A,
    expert_down: [3]A,
    attention_norms: [4]A,
    moe_norms: [5]A,
    expert_scale: A,
    eps: A,

    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        const p: *Back = @ptrCast(@alignCast(raw.?));
        p.kernels.deinit();
        mx.allocator.destroy(p);
    }
    fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *Back = @ptrCast(@alignCast(raw.?));
        return p.graph(out, ins) catch -1;
    }
    fn graph(p: *Back, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
        var s = mx.Scope{};
        defer s.deinit();
        var args: [2]A = undefined;
        for (&args, 0..) |*arg, i| {
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_vector_array_get(&value, ins, i);
            arg.* = try s.result(rc, value);
        }
        const values = try p.apply(&s, args[0], args[1]);
        return mx.c.mlx_vector_array_set_data(out, &values, values.len);
    }
    fn apply(p: *Back, s: *mx.Scope, attended: A, h: A) ![2]A {
        const rows = mx.dim(h, 0);
        const out = try projectRows(&p.kernels, s, try s.reshape(attended, &.{ rows, -1 }), p.o);
        const tail = try p.kernels.run(s, src.gemma_attn_tail, &.{ h, out, p.attention_norms[0], p.attention_norms[1], p.attention_norms[2], p.attention_norms[3], p.eps }, &.{ ti("D", 2816), ti("T", 256) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } } });
        const gu = try projectRows(&p.kernels, s, tail[1], p.gate_up);
        const activated = try @import("prefill_ops.zig").uncompiled(s, .geglu, &.{ try s.slice(gu, 1, 0, 2112), try s.slice(gu, 1, 2112, 4224) });
        const dense = try projectRows(&p.kernels, s, activated, p.down);
        const routes = try ops.route(&p.kernels, s, try ops.router(&p.kernels, s, tail[3], p.router, group(p.router, 8)), p.expert_scale, 8);
        const expert_act = try ops.gateUp(&p.kernels, s, tail[2], routes[0], 8, p.expert_gate, p.expert_up, group(p.expert_gate, 4));
        const expert = try ops.down(&p.kernels, s, expert_act, routes[0], routes[1], 8, p.expert_down, group(p.expert_down, 4));
        const end = try p.kernels.run(s, src.gemma_moe_tail, &.{ tail[0], dense, expert, p.moe_norms[0], p.moe_norms[1], p.moe_norms[2], p.moe_norms[3], p.moe_norms[4], p.eps }, &.{ ti("D", 2816), ti("T", 256) }, .{ 256 * rows, 1, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, 2816 } }, .{ .shape = &.{ rows, 2816 } } });
        return end[0..2].*;
    }
    fn group(weights: [3]A, bits: i32) i32 {
        return @divExact(mx.dim(weights[0], -1) * @divExact(32, bits), mx.dim(weights[1], -1));
    }
};

fn projectRows(kernels: *mx.Kernels, s: *mx.Scope, x: A, weights: [3]A) !A {
    const rows = mx.dim(x, 0);
    const n = mx.dim(weights[0], 0);
    const width = mx.dim(x, 1);
    if (rows < 1 or rows > Model.max_shared_rows or @mod(n, 8) != 0 or @mod(width, 64) != 0) return error.InvalidGemmaProjection;
    if (rows > Model.max_decode_rows) {
        var parts: [Model.max_shared_rows / Model.max_decode_rows]A = undefined;
        var count: usize = 0;
        var first: i32 = 0;
        while (first < rows) {
            const end = @min(rows, first + Model.max_decode_rows);
            parts[count] = try projectRows(kernels, s, try s.slice(x, 0, first, end), weights);
            count += 1;
            first = end;
        }
        return s.cat(parts[0..count], 0);
    }
    const gs = @divExact(width, mx.dim(weights[1], -1));
    return (try kernels.run(s, src.nemotron_rows_qmv, &.{ x, weights[0], weights[1], weights[2] }, &.{ ti("K", width), ti("N", n), ti("GS", gs), ti("RPS", 4) }, .{ 32 * rows, @divExact(n, 4), 1 }, .{ 32 * rows, if (rows <= 8) 2 else 1, 1 }, &.{.{ .shape = &.{ rows, n } }}))[0];
}

pub const Model = struct {
    round_owner: @import("decode_round.zig").Owner = .{},
    weights: cp.Store,
    kernels: mx.Kernels,
    activations: @import("prefill_ops.zig").Ops = .{},
    prefills: @import("gemma_prefill.zig").Ops = .{},
    fronts: [30]mx.c.mlx_closure = @splat(.{ .ctx = null }),
    backs: [30]mx.c.mlx_closure = @splat(.{ .ctx = null }),
    attentions: [max_shared_streams][2]ops.Attention = @splat(@splat(.{})),
    cache: [30]Cache = @splat(.{}),
    position: i32 = 0,
    generation: u64 = 0,
    draft: ?@import("dflash.zig").Draft = null,
    has_mtp: bool = false,
    pub const vocab = 262144;
    pub const max_shared_rows = 64;
    pub const max_shared_streams = 16;
    pub const max_decode_rows = 16;
    pub fn forwardStreams(m: *Model, streams: []const @import("gemma_shared.zig").Stream) !@import("gemma_shared.zig").Pass {
        return @import("gemma_shared.zig").forward(m, streams);
    }
    pub fn eos(id: i32) bool {
        return id == 1 or id == 106 or id == 50;
    }
    pub fn init(io: std.Io, dir: []const u8) !Model {
        var m = Model{ .weights = cp.Store.init(64), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer parsed.deinit();
        try config(parsed.value);
        try @import("schema.zig").checkCheckpoint(.gemma, io, dir);
        try m.weights.load(io, dir, "language_model.");
        try @import("schema.zig").validateConfig(.gemma, &m.weights.arrays, false, parsed.value);
        var entries = m.weights.arrays.iterator();
        while (entries.next()) |entry| {
            if (mx.dtype(entry.value_ptr.*) != mx.c.MLX_UINT32) continue;
            const spec = (try @import("quantization.zig").resolve(parsed.value, entry.key_ptr.*)) orelse return error.UnsupportedQuantization;
            const bits: i32 = if (std.mem.endsWith(u8, entry.key_ptr.*, ".router.proj.weight")) 8 else 4;
            if (spec.bits != bits) return error.UnsupportedQuantization;
        }
        m.weights.group = (try @import("quantization.zig").resolve(parsed.value, "model.embed_tokens")).?.group_size;
        var s = mx.Scope{};
        defer s.deinit();
        var local: [128]f32 = undefined;
        for (&local, 0..) |*v, i| v.* = @floatCast(@exp2(-@as(f64, @floatFromInt(i)) / 128 * @log2(@as(f64, 10000))));
        try m.weights.put("inv_local", try s.data(&local, &.{128}, mx.f32t));
        var exponents: [64]f32 = undefined;
        for (&exponents, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i * 2)) / 512;
        const powers = try s.binary(mx.c.mlx_power, try s.scalar(1000000), try s.data(&exponents, &.{64}, mx.f32t));
        const inverse = try s.binary(mx.c.mlx_divide, try s.scalar(1), powers);
        try m.weights.put("inv_global", try s.cat(&.{ inverse, try s.zeros(&.{192}, mx.f32t) }, 0));
        try m.weights.put("freq_global", try s.cat(&.{ powers, try s.binary(mx.c.mlx_add, try s.zeros(&.{192}, mx.f32t), try s.scalar(std.math.inf(f32))) }, 0));
        try m.weights.put("eps", try s.scalar(1e-6));
        for (0..30) |i| {
            try m.stack(&s, i, "qkv", if (sliding(i)) &.{ "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj" } else &.{ "self_attn.q_proj", "self_attn.k_proj" });
            try m.stack(&s, i, "gate_up", &.{ "mlp.gate_proj", "mlp.up_proj" });
            const root = try s.cast(try s.scalar(@floatCast(1.0 / @sqrt(@as(f64, 2816)))), mx.bf16);
            const scale = try s.binary(mx.c.mlx_multiply, try m.weight(i, "router.scale"), root);
            try m.weights.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.router_norm", .{i}), scale);
        }
        const prepared = try mx.allocator.alloc(A, m.weights.arrays.count());
        defer mx.allocator.free(prepared);
        var tensors = m.weights.arrays.valueIterator();
        for (prepared) |*tensor| tensor.* = tensors.next().?.*;
        // Compile copies unevaluated captured graphs into every shape specialization.
        try mx.evalMany(prepared, false);
        return m;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        if (m.draft) |*d| d.deinit();
        for (m.fronts) |fun| if (fun.ctx != null) {
            _ = mx.c.mlx_closure_free(fun);
        };
        for (m.backs) |fun| if (fun.ctx != null) {
            _ = mx.c.mlx_closure_free(fun);
        };
        for (&m.attentions) |*pair| for (pair) |*attention| attention.deinit();
        m.prefills.deinit();
        m.activations.deinit();
        m.kernels.deinit();
        m.weights.deinit();
    }
    pub fn reset(m: *Model) void {
        for (&m.cache) |*cache| cache.deinit();
        if (m.draft) |*d| d.reset();
        m.position = 0;
        m.generation +%= 1;
    }
    pub fn loadDraft(m: *Model, io: std.Io, dir: []const u8) !void {
        return m.loadDraftBits(io, dir, 8);
    }
    pub fn loadDraftBits(m: *Model, io: std.Io, dir: []const u8, bits: i32) !void {
        if (m.position != 0) return error.DraftRequiresEmptyCache;
        const d = try @import("dflash.zig").Draft.init(io, dir, 2816, vocab, 30, bits);
        if (m.draft) |*old| old.deinit();
        m.draft = d;
        m.has_mtp = true;
    }
    pub fn draftAbsorbsOnCommit(_: *Model) bool {
        return true;
    }
    pub fn maxDrafts(m: *Model) usize {
        return if (m.draft) |d| @intCast(d.parsed.value.dflash_config.block_size - 1) else 0;
    }
    pub fn propose(m: *Model, _: A, anchor: i32, output: []i32, _: @import("sampling.zig").Sampling) !void {
        output[0] = anchor;
        if (output.len <= 1) return;
        const d = if (m.draft) |*value| value else return error.MissingDraft;
        if (d.position != m.position or output.len - 1 > m.maxDrafts()) return error.InvalidDraftBlock;
        var s = mx.Scope{};
        defer s.deinit();
        const cfg = d.parsed.value.dflash_config;
        var ids: [128]i32 = @splat(cfg.mask_token_id);
        ids[0] = anchor;
        const embeddings = try s.reshape(try s.binary(mx.c.mlx_multiply, try m.weights.embed(&s, "model.embed_tokens", ids[0..output.len]), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)) * cfg.input_embedding_scale)), mx.bf16)), &.{ 1, @intCast(output.len), 2816 });
        const hidden = try d.forward(&s, embeddings);
        var logits = try s.binary(mx.c.mlx_multiply, try m.weights.linear(&m.kernels, &s, "model.embed_tokens", hidden, false), try s.cast(try s.scalar(cfg.output_multiplier), mx.bf16));
        if (cfg.final_logit_softcapping orelse d.parsed.value.final_logit_softcapping) |cap| if (cap > 0) {
            logits = try @import("prefill_ops.zig").uncompiled(&s, .softcap, &.{ logits, try s.scalar(cap) });
        };
        var selected = mx.c.mlx_array_new();
        const rc = mx.c.mlx_argmax_axis(&selected, logits, -1, false, mx.stream);
        selected = try s.cast(try s.result(rc, selected), mx.c.MLX_INT32);
        try mx.eval(selected);
        @memcpy(output[1..], mx.c.mlx_array_data_int32(selected)[0 .. output.len - 1]);
    }
    pub fn sliding(i: usize) bool {
        return i % 6 != 5;
    }
    pub fn geometry(i: usize) ops.Geometry {
        const local = sliding(i);
        return .{ .heads = 16, .kv_heads = if (local) 8 else 2, .head_dim = if (local) 256 else 512, .values_are_keys = !local };
    }
    pub fn weight(m: *Model, i: usize, suffix: []const u8) !A {
        var buf: [256]u8 = undefined;
        return m.weights.get(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, suffix }));
    }
    pub fn triple(m: *Model, i: usize, suffix: []const u8) ![3]A {
        var buf: [256]u8 = undefined;
        return m.weights.triple(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}", .{ i, suffix }));
    }
    fn group(m: *Model, i: usize, suffix: []const u8, bits: i32) !i32 {
        const t = try m.triple(i, suffix);
        return @divExact(mx.dim(t[0], -1) * @divExact(32, bits), mx.dim(t[1], -1));
    }
    fn stack(m: *Model, s: *mx.Scope, i: usize, name: []const u8, members: []const []const u8) !void {
        for ([_][]const u8{ "weight", "scales", "biases" }) |suffix| {
            var arrays: [3]A = undefined;
            var buf: [256]u8 = undefined;
            for (members, 0..) |member, j| arrays[j] = try m.weight(i, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ member, suffix }));
            const value = try s.cat(arrays[0..members.len], 0);
            try mx.eval(value);
            try m.weights.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.{s}", .{ i, name, suffix }), value);
            var offset: i32 = 0;
            for (members, 0..) |member, j| {
                const end = offset + mx.dim(arrays[j], 0);
                try m.weights.put(try std.fmt.bufPrint(&buf, "model.layers.{d}.{s}.{s}", .{ i, member, suffix }), try s.slice(value, 0, offset, end));
                offset = end;
            }
        }
    }
    pub fn project(m: *Model, s: *mx.Scope, x: A, weights: [3]A) !A {
        return projectRows(&m.kernels, s, x, weights);
    }
    pub fn front(m: *Model, s: *mx.Scope, i: usize, normed: A, positions: A) ![3]A {
        if (m.fronts[i].ctx == null) {
            const value = Front{
                .kernels = mx.Kernels.init(),
                .geometry = geometry(i),
                .qkv = try m.triple(i, "qkv"),
                .q_norm = try m.weight(i, "self_attn.q_norm.weight"),
                .k_norm = try m.weight(i, "self_attn.k_norm.weight"),
                .inverse = try m.weights.get(if (sliding(i)) "inv_local" else "inv_global"),
                .eps = try m.weights.get("eps"),
                .group = try m.group(i, "qkv", 4),
            };
            const payload = try mx.allocator.create(Front);
            payload.* = value;
            const fun = mx.c.mlx_closure_new_func_payload(Front.callback, payload, Front.destroy);
            if (fun.ctx == null) {
                Front.destroy(payload);
                return error.MlxFailure;
            }
            defer _ = mx.c.mlx_closure_free(fun);
            try mx.check(mx.c.mlx_compile(&m.fronts[i], fun, false));
        }
        const arguments = [_]A{ normed, positions };
        var result: [3]A = undefined;
        try m.kernels.call(s, m.fronts[i], &arguments, &result);
        return result;
    }
    pub fn back(m: *Model, s: *mx.Scope, i: usize, attended: A, h: A) ![2]A {
        if (m.backs[i].ctx == null) {
            const value = Back{
                .kernels = mx.Kernels.init(),
                .o = try m.triple(i, "self_attn.o_proj"),
                .gate_up = try m.triple(i, "gate_up"),
                .down = try m.triple(i, "mlp.down_proj"),
                .router = try m.triple(i, "router.proj"),
                .expert_gate = try m.triple(i, "experts.switch_glu.gate_proj"),
                .expert_up = try m.triple(i, "experts.switch_glu.up_proj"),
                .expert_down = try m.triple(i, "experts.switch_glu.down_proj"),
                .attention_norms = .{ try m.weight(i, "post_attention_layernorm.weight"), try m.weight(i, "pre_feedforward_layernorm.weight"), try m.weight(i, "pre_feedforward_layernorm_2.weight"), try m.weight(i, "router_norm") },
                .moe_norms = .{ try m.weight(i, "post_feedforward_layernorm_1.weight"), try m.weight(i, "post_feedforward_layernorm_2.weight"), try m.weight(i, "post_feedforward_layernorm.weight"), try m.weight(i, "layer_scalar"), if (i < 29) try m.weight(i + 1, "input_layernorm.weight") else try m.weights.get("model.norm.weight") },
                .expert_scale = try m.weight(i, "router.per_expert_scale"),
                .eps = try m.weights.get("eps"),
            };
            const payload = try mx.allocator.create(Back);
            payload.* = value;
            const fun = mx.c.mlx_closure_new_func_payload(Back.callback, payload, Back.destroy);
            if (fun.ctx == null) {
                Back.destroy(payload);
                return error.MlxFailure;
            }
            defer _ = mx.c.mlx_closure_free(fun);
            try mx.check(mx.c.mlx_compile(&m.backs[i], fun, false));
        }
        const arguments = [_]A{ attended, h };
        var result: [2]A = undefined;
        try m.kernels.call(s, m.backs[i], &arguments, &result);
        return result;
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        var p = try m.forwardQueued(tokens);
        errdefer p.deinit();
        try mx.eval(p.logits);
        return p;
    }

    pub fn forwardQueued(m: *Model, tokens: []const i32) !Pass {
        return m.forwardState(&m.cache, m.position, m.generation, tokens);
    }

    pub fn forwardState(m: *Model, cache: []Cache, position: i32, generation: u64, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > max_decode_rows or position > 262144 - tokens.len) return error.ContextLimitExceeded;
        for (tokens) |token| if (token < 0 or token >= vocab) return error.InvalidToken;
        var scope = mx.Scope{};
        defer scope.deinit();
        return m.forwardInput(cache, position, generation, try scope.ints(tokens), tokens.len);
    }

    pub fn forwardAfter(m: *Model, previous: *Pass, sampled: A) !Pass {
        if (previous.rows != 1 or !previous.staged_ready or previous.position != m.position or previous.generation != m.generation) return error.InvalidPreview;
        if (sampled.ctx == null or mx.dtype(sampled) != mx.c.MLX_UINT32 or !std.mem.eql(i32, mx.shape(sampled), &.{1})) return error.InvalidSamplingShape;
        if (previous.position < 0 or previous.position >= 262143) return error.ContextLimitExceeded;
        if (kv.enabled) for (&previous.staged, 0..) |*cache, layer| if (!sliding(layer)) try cache.pipeline();
        // The staged cache owns these donors; the committed logical cache is retained.
        for (&m.cache) |*cache| cache.dropSpares();
        return m.forwardInput(&previous.staged, previous.position + 1, previous.generation +% 1, sampled, 1);
    }

    fn forwardInput(m: *Model, cache: []Cache, position: i32, generation: u64, input: A, count: usize) !Pass {
        var p = Pass{ .position = position, .generation = generation, .rows = count };
        errdefer p.deinit();
        const s = &p.scope;
        const rows: i32 = @intCast(count);
        var positions: [max_decode_rows]i32 = undefined;
        for (0..count) |i| positions[i] = position + @as(i32, @intCast(i));
        const at = try ops.paddedInts(s, positions[0..count]);
        const attention_rows = [_]ops.Rows{ try ops.Rows.init(s, positions[0..count], 0, 0, 512), try ops.Rows.init(s, positions[0..count], 1024, 1152, 256) };
        var h = try s.binary(mx.c.mlx_multiply, try m.weights.embedArray(s, "model.embed_tokens", input), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)))), mx.bf16));
        var normed = try s.rms(h, try m.weight(0, "input_layernorm.weight"));
        var carried = [_]A{ mx.empty, mx.empty };
        defer for (carried) |array| mx.free(array);
        var taps: [32]A = undefined;
        var tap_count: usize = 0;
        for (0..30) |i| {
            var layer_scope = mx.Scope{};
            defer layer_scope.deinit();
            const layer_s = &layer_scope;
            const local = sliding(i);
            const g = geometry(i);
            const qkv = try m.front(layer_s, i, normed, at);
            const keys = if (cache[i].keys.ctx != null) cache[i].attention("keys") else try layer_s.zeros(&.{ 1, g.kv_heads, if (local) 1152 else 1, g.head_dim }, mx.bf16);
            const values = if (cache[i].values.ctx != null) cache[i].attention("values") else try layer_s.zeros(mx.shape(keys), mx.bf16);
            p.records[i] = .{ .keys = try s.reshape(qkv[1], &.{ 1, g.kv_heads, rows, g.head_dim }), .values = try s.reshape(qkv[2], &.{ 1, g.kv_heads, rows, g.head_dim }) };
            try stageCacheLayer(cache, &p, i);
            const attended = try m.attentions[0][@intFromBool(local)].apply(&m.kernels, layer_s, qkv[0], keys, values, qkv[1], qkv[2], attention_rows[@intFromBool(local)], 1);
            const end = try m.back(layer_s, i, attended, h);
            const next_h = try mx.retain(end[0]);
            const next_normed = mx.retain(end[1]) catch |err| {
                mx.free(next_h);
                return err;
            };
            for (carried) |array| mx.free(array);
            carried = .{ next_h, next_normed };
            h = next_h;
            normed = next_normed;
            if (m.draft) |d| for (d.parsed.value.dflash_config.target_layer_ids) |id| if (id == i) {
                taps[tap_count] = try s.own(try mx.retain(h));
                tap_count += 1;
            };
            if ((i + 1) % 8 == 0) {
                var pending: [17]A = undefined;
                pending[0] = normed;
                for (p.staged[i - 7 .. i + 1], 0..) |staged, j| {
                    pending[1 + 2 * j] = staged.keys;
                    pending[2 + 2 * j] = staged.values;
                }
                try mx.evalMany(&pending, true);
            }
        }
        p.hidden = try s.own(try mx.retain(normed));
        if (tap_count > 0) p.taps = try s.cat(taps[0..tap_count], -1);
        p.logits = try m.activations.call(s, .softcap, &.{ try m.project(s, normed, try m.weights.triple("model.embed_tokens")), try s.scalar(30) });
        var writes: [60]A = undefined;
        for (p.staged, 0..) |staged, i| {
            writes[2 * i] = staged.keys;
            writes[2 * i + 1] = staged.values;
        }
        p.logits = try cacheDependency(s, p.logits, &writes);
        return p;
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        return @import("gemma_prefill.zig").forward(m, tokens);
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        if (p.position != m.position or p.generation != m.generation or keep == 0 or keep > p.rows) return error.InvalidCommit;
        var s = mx.Scope{};
        defer s.deinit();
        const staged = p.staged_ready;
        if (staged) try mx.eval(p.logits);
        var next = try acceptCache(&s, &m.cache, p, keep);
        errdefer for (&next) |*cache| cache.deinit();
        var arrays: [60]A = undefined;
        for (next, 0..) |cache, i| {
            arrays[2 * i] = cache.keys;
            arrays[2 * i + 1] = cache.values;
        }
        if (!staged or keep < p.rows) try mx.evalMany(&arrays, false);
        for (next) |cache| try cache.observe();
        if (m.draft) |*d| {
            if (d.position != m.position or p.taps.ctx == null) return error.InvalidDraftContext;
            try d.absorb(try s.slice(p.taps, 0, 0, @intCast(keep)));
        }
        for (&m.cache) |*cache| cache.deinit();
        m.cache = next;
        m.position += @intCast(keep);
        m.generation +%= 1;
    }
    pub fn stageCache(current: []Cache, p: *Pass) !void {
        if (p.staged_ready) return error.InvalidCacheState;
        for (0..30) |layer| try stageCacheLayer(current, p, layer);
    }
    pub fn stageCacheLayer(current: []Cache, p: *Pass, layer: usize) !void {
        if (current.len != 30 or layer >= 30 or p.staged_ready or p.staged[layer].keys.ctx != null or p.staged[layer].values.ctx != null) return error.InvalidCacheState;
        var scope = mx.Scope{};
        defer scope.deinit();
        const first = p.stage_indices.count;
        errdefer p.stage_indices.count = first;
        var staged = try prepareCacheLayer(&scope, &p.stage_indices, &current[layer], p, layer, p.rows);
        errdefer staged.deinit();
        for (p.stage_indices.arrays[first..p.stage_indices.count]) |*array| array.* = try p.scope.own(try mx.retain(array.*));
        p.staged[layer] = staged;
        if (layer == 29) p.staged_ready = true;
    }
    pub fn cacheDependency(s: *mx.Scope, value: A, arrays: []const A) !A {
        const inputs = [_]A{value};
        const ins = mx.c.mlx_vector_array_new_data(&inputs, inputs.len);
        defer _ = mx.c.mlx_vector_array_free(ins);
        const dependencies = mx.c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
        defer _ = mx.c.mlx_vector_array_free(dependencies);
        var outputs = mx.c.mlx_vector_array_new();
        defer _ = mx.c.mlx_vector_array_free(outputs);
        try mx.check(mx.c.mlx_depends(&outputs, ins, dependencies));
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_vector_array_get(&out, outputs, 0);
        return s.result(rc, out);
    }
    pub fn acceptCache(s: *mx.Scope, current: []Cache, p: *Pass, keep: usize) ![30]Cache {
        if (current.len != 30 or keep == 0 or keep > p.rows) return error.InvalidCommit;
        // A long prefill can overwrite kept rows in its staged ring more than once.
        if (!p.staged_ready or (p.rows > max_decode_rows and keep < p.rows)) return prepareCache(s, current, p, keep);
        var next = p.staged;
        p.staged = @splat(.{});
        p.staged_ready = false;
        errdefer for (&next) |*cache| cache.deinit();
        if (keep == p.rows) return next;
        const accepted: i32 = @intCast(keep);
        const end = p.position + accepted;
        var indices = RingIndices{};
        for (&next, current, 0..) |*cache, old, layer| {
            inline for (.{ "keys", "values" }) |field| {
                const value = if (sliding(layer)) blk: {
                    break :blk try restoreRing(s, &indices, @field(cache.*, field), @field(old, field), end, @intCast(p.rows - keep));
                } else blk: {
                    const buffer = &@field(cache.storage, field);
                    if (buffer.current.ctx != null) {
                        buffer.offset = end;
                        if (buffer.recent.ctx != null) {
                            const recent = try mx.retain(try s.slice(buffer.recent, 2, 0, accepted));
                            mx.free(buffer.recent);
                            buffer.recent = recent;
                        }
                        break :blk try s.slice(buffer.current, 2, 0, end);
                    }
                    break :blk try s.slice(@field(cache.*, field), 2, 0, end);
                };
                const owned = try mx.retain(value);
                mx.free(@field(cache.*, field));
                @field(cache, field) = owned;
            }
            for (cache.storage.recent[0..cache.storage.recent_count]) |*recent| {
                const count = end - recent.position;
                if (count <= 0) return error.InvalidCacheState;
                if (count >= mx.dim(recent.keys, 2)) continue;
                inline for (.{ "keys", "values" }) |field| {
                    const value = try mx.retain(try s.slice(@field(recent.*, field), 2, 0, count));
                    mx.free(@field(recent.*, field));
                    @field(recent, field) = value;
                }
            }
        }
        return next;
    }
    pub fn prepareCache(s: *mx.Scope, current_cache: []Cache, p: *const Pass, keep: usize) ![30]Cache {
        if (current_cache.len != 30 or keep == 0 or keep > p.rows) return error.InvalidCommit;
        var next: [30]Cache = @splat(.{});
        errdefer for (&next) |*cache| cache.deinit();
        var indices = RingIndices{};
        for (current_cache, &next, 0..) |*old, *target, i| target.* = try prepareCacheLayer(s, &indices, old, p, i, keep);
        return next;
    }
    fn prepareCacheLayer(s: *mx.Scope, indices: *RingIndices, old: *Cache, p: *const Pass, layer: usize, keep: usize) !Cache {
        const record = p.records[layer];
        const added = if (keep == p.rows) record else Cache{ .keys = try s.slice(record.keys, 2, 0, @intCast(keep)), .values = try s.slice(record.values, 2, 0, @intCast(keep)) };
        if (kv.enabled and sliding(layer) and p.rows <= max_decode_rows) {
            const backing_bytes = [2]u64{ @max(p.record_bytes[layer][0], Cache.bytes(record.keys)), @max(p.record_bytes[layer][1], Cache.bytes(record.values)) };
            return old.prepareBuffered(s, indices, added, p.position, backing_bytes, true);
        }
        if (kv.enabled and !sliding(layer) and old.storage.pipelined) {
            const backing_bytes = [2]u64{ @max(p.record_bytes[layer][0], Cache.bytes(record.keys)), @max(p.record_bytes[layer][1], Cache.bytes(record.values)) };
            return old.prepareBuffered(s, indices, added, p.position, backing_bytes, false);
        }
        var target = Cache{};
        errdefer target.deinit();
        if (kv.enabled and !sliding(layer)) {
            inline for (.{ "keys", "values" }, 0..) |field, j| {
                const buffer = &@field(old.storage, field);
                const write = try buffer.append(s, @field(old.*, field), @field(added, field), 2);
                @field(target, field) = try mx.retain(write.view);
                @field(target.storage, field) = try buffer.finish(s, write, @intCast(keep));
                target.storage.donors[j] = write.donor;
            }
            return target;
        }
        inline for (.{ "keys", "values" }) |field| {
            const rows = @field(added, field);
            const current = @field(old.*, field);
            const value = if (sliding(layer)) blk: {
                const buffer = if (current.ctx != null) current else try s.zeros(&.{ 1, 8, 1152, 256 }, mx.bf16);
                break :blk try ringWrite(s, indices, buffer, rows, p.position);
            } else if (current.ctx == null) rows else try s.cat(&.{ current, rows }, 2);
            @field(target, field) = try mx.retain(value);
        }
        return target;
    }
    pub fn checkExact(m: *Model, prefix_count: usize) !void {
        defer m.reset();
        const tracking = kv.track_reuse;
        const reused_before = kv.reused;
        kv.track_reuse = true;
        defer kv.track_reuse = tracking;
        {
            m.reset();
            var stale = try m.forward(&.{1});
            defer stale.deinit();
            m.reset();
            try std.testing.expectError(error.InvalidCommit, m.commit(&stale, 1));
        }
        var serial: [4]A = undefined;
        var saved: [30]Cache = @splat(.{});
        defer for (&saved) |*cache| cache.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        for (0..2) |run| {
            m.reset();
            var offset: usize = 0;
            while (offset < prefix_count) {
                var tokens: [16]i32 = undefined;
                const count = @min(16, prefix_count - offset);
                for (tokens[0..count], 0..) |*token, j| token.* = @intCast(1000 + (offset + j) % 2000);
                var p = try m.forward(tokens[0..count]);
                defer p.deinit();
                try m.commit(&p, count);
                offset += count;
            }
            if (run == 0) {
                for ([_]i32{ 23, 41, 59, 83 }, 0..) |token, j| {
                    var p = try m.forward(&.{token});
                    defer p.deinit();
                    serial[j] = try scope.own(try mx.retain(p.logits));
                    try m.commit(&p, 1);
                }
                for (m.cache, &saved) |cache, *copy| {
                    copy.keys = try mx.retain(cache.keys);
                    copy.values = try mx.retain(cache.values);
                }
            } else {
                var p = try m.forward(&.{ 23, 41, 59, 83, 97, 101 });
                defer p.deinit();
                for (serial, 0..) |expected, j| try equal(&scope, expected, try scope.slice(p.logits, 0, @intCast(j), @intCast(j + 1)));
                try m.commit(&p, 3);
                try std.testing.expectError(error.InvalidCommit, m.commit(&p, 1));
                var next = try m.forward(&.{83});
                defer next.deinit();
                try equal(&scope, serial[3], next.logits);
                try m.commit(&next, 1);
                for (m.cache, saved) |actual, expected| {
                    try equal(&scope, actual.keys, expected.keys);
                    try equal(&scope, actual.values, expected.values);
                }
            }
        }
        if (kv.enabled) try std.testing.expect(kv.reused > reused_before);
        std.debug.print("PASS: Gemma serial/chain logits, partial commit and all 60 cache arrays at prefix {d}.\n", .{prefix_count});
    }
};
const RingIndices = struct {
    offsets: [60]i32 = undefined,
    arrays: [60]A = undefined,
    count: usize = 0,
    fn get(indices: *RingIndices, s: *mx.Scope, offset: i32) !A {
        for (indices.offsets[0..indices.count], indices.arrays[0..indices.count]) |cached, array| if (cached == offset) return array;
        if (indices.count == indices.arrays.len) return error.InvalidCacheState;
        const array = try s.ints(&.{offset});
        indices.offsets[indices.count] = offset;
        indices.arrays[indices.count] = array;
        indices.count += 1;
        return array;
    }
};
fn ringWrite(s: *mx.Scope, indices: *RingIndices, buffer: A, rows: A, position: i32) !A {
    const total = mx.dim(rows, 2);
    const skipped = @max(0, total - 1152);
    const count = total - skipped;
    const slot = @mod(position + skipped, 1152);
    if (skipped == 0 and count <= 1152 - slot) return ringPut(s, buffer, rows, try indices.get(s, slot));
    var local = mx.Scope{};
    defer local.deinit();
    const retained = if (skipped > 0) try local.slice(rows, 2, skipped, total) else rows;
    const first = @min(count, 1152 - slot);
    const front = try ringPut(&local, buffer, if (first < count) try local.slice(retained, 2, 0, first) else retained, try indices.get(s, slot));
    const result = if (first < count) try ringPut(&local, front, try local.slice(retained, 2, first, count), try indices.get(s, 0)) else front;
    return s.own(try mx.retain(result));
}
fn ringPut(s: *mx.Scope, buffer: A, rows: A, index: A) !A {
    var out = mx.c.mlx_array_new();
    const axis: i32 = 2;
    const rc = mx.c.mlx_slice_update_dynamic(&out, buffer, rows, index, &axis, 1, mx.stream);
    return s.result(rc, out);
}
fn restoreRing(s: *mx.Scope, indices: *RingIndices, buffer: A, previous: A, position: i32, count: i32) !A {
    var local = mx.Scope{};
    defer local.deinit();
    const slot = @mod(position, 1152);
    const first = @min(count, 1152 - slot);
    const rows = if (previous.ctx != null) try local.slice(previous, 2, slot, slot + first) else try local.zeros(&.{ 1, 8, first, 256 }, mx.bf16);
    const front = try ringPut(&local, buffer, rows, try indices.get(s, slot));
    const result = if (first < count) blk: {
        const rest = if (previous.ctx != null) try local.slice(previous, 2, 0, count - first) else try local.zeros(&.{ 1, 8, count - first, 256 }, mx.bf16);
        break :blk try ringPut(&local, front, rest, try indices.get(s, 0));
    } else front;
    return s.own(try mx.retain(result));
}
fn config(root: std.json.Value) !void {
    if (root != .object) return error.InvalidModelConfig;
    const text = root.object.get("text_config") orelse return error.UnsupportedModel;
    if (text != .object) return error.InvalidModelConfig;
    inline for (.{ .{ "rms_norm_eps", 1e-6 }, .{ "final_logit_softcapping", 30.0 } }) |field| try number(text, field[0], field[1]);
    try string(text, "dtype", "bfloat16");
    try string(text, "hidden_activation", "gelu_pytorch_tanh");
    for ([_][]const u8{ "attention_bias", "use_double_wide_mlp" }) |field| {
        const value = text.object.get(field) orelse return error.InvalidModelConfig;
        if (value != .bool or value.bool) return error.UnsupportedModelGeometry;
    }
    const rope = text.object.get("rope_parameters") orelse return error.InvalidModelConfig;
    if (rope != .object) return error.InvalidModelConfig;
    const global = rope.object.get("full_attention") orelse return error.InvalidModelConfig;
    const local = rope.object.get("sliding_attention") orelse return error.InvalidModelConfig;
    try string(global, "rope_type", "proportional");
    try number(global, "rope_theta", 1000000);
    try number(global, "partial_rotary_factor", 0.25);
    try string(local, "rope_type", "default");
    try number(local, "rope_theta", 10000);
    inline for (.{ .{ "hidden_size", 2816 }, .{ "num_hidden_layers", 30 }, .{ "vocab_size", 262144 }, .{ "num_attention_heads", 16 }, .{ "num_key_value_heads", 8 }, .{ "num_global_key_value_heads", 2 }, .{ "head_dim", 256 }, .{ "global_head_dim", 512 }, .{ "intermediate_size", 2112 }, .{ "moe_intermediate_size", 704 }, .{ "num_experts", 128 }, .{ "top_k_experts", 8 }, .{ "sliding_window", 1024 }, .{ "num_kv_shared_layers", 0 }, .{ "hidden_size_per_layer_input", 0 } }) |field| {
        const value = text.object.get(field[0]) orelse return error.InvalidModelConfig;
        if (value != .integer or value.integer != field[1]) return error.UnsupportedModelGeometry;
    }
    for ([_][]const u8{ "tie_word_embeddings", "enable_moe_block", "attention_k_eq_v" }) |field| {
        const value = text.object.get(field) orelse return error.InvalidModelConfig;
        if (value != .bool or !value.bool) return error.UnsupportedModelGeometry;
    }
    const layers = text.object.get("layer_types") orelse return error.InvalidModelConfig;
    if (layers != .array or layers.array.items.len != 30) return error.UnsupportedModelGeometry;
    for (layers.array.items, 0..) |layer, i| if (layer != .string or !std.mem.eql(u8, layer.string, if (Model.sliding(i)) "sliding_attention" else "full_attention")) return error.UnsupportedModelGeometry;
}
fn number(object: std.json.Value, key: []const u8, expected: f64) !void {
    if (object != .object) return error.InvalidModelConfig;
    const value = object.object.get(key) orelse return error.InvalidModelConfig;
    const actual: f64 = switch (value) {
        .integer => |n| @floatFromInt(n),
        .float => |n| n,
        else => return error.InvalidModelConfig,
    };
    if (actual != expected) return error.UnsupportedModelGeometry;
}
fn string(object: std.json.Value, key: []const u8, expected: []const u8) !void {
    if (object != .object) return error.InvalidModelConfig;
    const value = object.object.get(key) orelse return error.InvalidModelConfig;
    if (value != .string or !std.mem.eql(u8, value.string, expected)) return error.UnsupportedModelGeometry;
}

pub fn checkModel(io: std.Io, dir: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var model = try Model.init(io, dir);
    defer model.deinit();
    var p = try model.forward(&.{ 1, 2, 3, 4 });
    defer p.deinit();
    const logits = try p.scope.cast(p.logits, mx.f32t);
    const path = try mx.allocator.dupeSentinel(u8, output, 0);
    defer mx.allocator.free(path);
    try mx.saveArray(path, logits);
    try model.commit(&p, 2);
    var continuation = try model.forward(&.{ 3, 4 });
    defer continuation.deinit();
    try equal(&continuation.scope, try continuation.scope.slice(p.logits, 0, 2, 4), continuation.logits);
    for (p.records, continuation.records) |full, partial| {
        try equal(&continuation.scope, try continuation.scope.slice(full.keys, 2, 2, 4), partial.keys);
        try equal(&continuation.scope, try continuation.scope.slice(full.values, 2, 2, 4), partial.values);
    }
    std.debug.print("PASS: Gemma partial commit, continuation and every layer's keys/values match the complete chain.\n", .{});
}
fn equal(s: *mx.Scope, a: A, b: A) !void {
    const x = try s.cast(try s.contiguous(a), mx.f32t);
    const y = try s.cast(try s.contiguous(b), mx.f32t);
    try mx.evalMany(&.{ x, y }, false);
    const count = mx.c.mlx_array_size(x);
    if (count != mx.c.mlx_array_size(y) or !std.mem.eql(u8, std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(x)[0..count]), std.mem.sliceAsBytes(mx.c.mlx_array_data_float32(y)[0..count]))) return error.GemmaExactnessMismatch;
}

pub fn checkDraft(io: std.Io, dir: []const u8, drafter: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var m = try Model.init(io, dir);
    defer m.deinit();
    try m.loadDraft(io, drafter);
    try std.Io.Dir.cwd().createDirPath(io, output);
    var buf: [256]u8 = undefined;
    for ([_]usize{ 3, 5, 16, 1, 16 }, [_]usize{ 3, 15, 1, 7, 15 }, 0..) |count, budget, step| {
        var tokens: [16]i32 = undefined;
        const attempted = @min(16, count + 2);
        for (tokens[0..attempted], 0..) |*token, j| token.* = 1000 + m.position + @as(i32, @intCast(j));
        var pass = try m.forward(tokens[0..attempted]);
        defer pass.deinit();
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "target-{d}", .{step}), try pass.scope.slice(pass.logits, 0, 0, @intCast(count)));
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "taps-{d}", .{step}), try pass.scope.slice(pass.taps, 0, 0, @intCast(count)));
        try m.commit(&pass, count);
        try std.testing.expectError(error.InvalidCommit, m.commit(&pass, 1));
        var proposed: [16]i32 = undefined;
        try m.propose(mx.empty, 42, proposed[0 .. budget + 1], .{});
        try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "proposal-{d}", .{step}), try pass.scope.ints(proposed[0 .. budget + 1]));
        for (m.draft.?.cache, 0..) |cache, i| {
            try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, i }), cache.keys);
            try saveDraft(&pass.scope, output, try std.fmt.bufPrint(&buf, "values-{d}-{d}", .{ step, i }), cache.values);
        }
    }
    for ([_]f64{ 0, 0.8 }) |temperature| {
        const settings = @import("sampling.zig").Sampling{ .temperature = temperature, .seed = 1234, .metal = true };
        m.reset();
        var serial = try @import("serial_generation.zig").generate(io, &m, &.{ 1000, 1001, 1002, 1003 }, 12, settings, 0, null);
        defer serial.deinit();
        for ([_]usize{ 1, 3, 15 }) |budget| {
            m.reset();
            var drafted = try @import("serial_generation.zig").generate(io, &m, &.{ 1000, 1001, 1002, 1003 }, 12, settings, budget, null);
            defer drafted.deinit();
            try std.testing.expectEqualSlices(u32, serial.tokens.items, drafted.tokens.items);
        }
    }
    std.debug.print("PASS: Gemma DFlash partial target commits and greedy/seeded generation at budgets 1, 3 and 15.\n", .{});
}
fn saveDraft(s: *mx.Scope, dir: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ dir, name }, 0);
    defer mx.allocator.free(path);
    const out = try s.cast(value, mx.f32t);
    try mx.eval(out);
    try mx.saveArray(path, out);
}
