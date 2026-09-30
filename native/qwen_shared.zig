const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("model.zig");
const lanes = @import("lanes.zig");
const grouped = @import("qwen_streams.zig");
const src = @import("kernel_sources.zig");
const kv = @import("kv_buffer.zig");
const round = @import("decode_round.zig");
const A = mx.Array;
pub const State = @import("request_state.zig").State(model.Model);
pub const Stream = struct { state: *State, tokens: []const i32, parents: []const i32 };
const Entry = struct { state: *State, first: i32, pass: model.Pass };
const Group = struct { layout: grouped.Layout, first: usize, len: usize };
const Gdn = struct { vals: [5]A = @splat(mx.empty), qkv: A = mx.empty };

// The model and states must remain at stable addresses until deinit; do not copy.
pub const Pass = struct {
    scope: mx.Scope = .{},
    model: *model.Model,
    ticket: round.Ticket,
    entries: []Entry,
    groups: []Group,
    gdn: [64]Gdn = @splat(.{}),
    logits: A = mx.empty,
    hidden: A = mx.empty,
    count: usize,

    pub fn view(p: *Pass, index: usize) !*const model.Pass {
        try p.ticket.expect(.forwarded);
        if (index >= p.entries.len) return error.InvalidStreams;
        return &p.entries[index].pass;
    }

    pub fn deinit(p: *Pass) void {
        if (!p.ticket.active()) return;
        p.scope.deinit();
        for (p.entries) |*entry| {
            entry.pass.deinit();
            entry.state.borrowed = false;
        }
        mx.allocator.free(p.entries);
        mx.allocator.free(p.groups);
        p.ticket.release();
    }

    pub fn commit(p: *Pass, paths: []const []const i32) !void {
        try p.ticket.expect(.forwarded);
        errdefer p.ticket.owner.stage = .failed;
        if (paths.len != p.entries.len) return error.InvalidCommit;
        for (p.entries) |entry| if (!entry.state.borrowed or entry.state.position != entry.pass.start) return error.InvalidCommit;
        const plans = try mx.allocator.alloc(grouped.Commit, p.groups.len);
        defer mx.allocator.free(plans);
        for (p.groups, plans) |*group, *plan| plan.* = try grouped.Commit.init(&group.layout, paths[group.first..][0..group.len]);
        const next = try mx.allocator.alloc([64]model.Cache, p.entries.len);
        defer mx.allocator.free(next);
        @memset(next, @splat(.{}));
        defer for (next) |*cache| for (cache) |*c| c.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        const s = &scope;
        for (0..64) |layer| {
            if (layer % 4 == 3) {
                for (p.entries, paths, next) |*entry, path, *cache| {
                    if (path.len == 0) continue;
                    const rec = &entry.pass.records[layer];
                    const old = &entry.state.cache[layer];
                    var consecutive = true;
                    for (path, 0..) |row, j| if (row != j) {
                        consecutive = false;
                    };
                    if (kv.enabled and consecutive and rec.key_write.capacity.ctx != null) {
                        const end = entry.pass.start + @as(i32, @intCast(path.len));
                        cache[layer].a = try mx.retain(try s.slice(rec.key_write.capacity, 2, 0, end));
                        cache[layer].b = try mx.retain(try s.slice(rec.value_write.capacity, 2, 0, end));
                        cache[layer].keys = try old.keys.finish(s, rec.key_write, @intCast(path.len));
                        cache[layer].values = try old.values.finish(s, rec.value_write, @intCast(path.len));
                    } else {
                        const ids = try s.ints(path);
                        var keys = try s.take(rec.values[0], ids, 2);
                        var vals = try s.take(rec.values[1], ids, 2);
                        if (kv.enabled) {
                            const kw = try old.keys.append(s, old.a, keys, 2);
                            const vw = try old.values.append(s, old.b, vals, 2);
                            keys = kw.view;
                            vals = vw.view;
                            cache[layer].keys = try old.keys.finish(s, kw, @intCast(path.len));
                            cache[layer].values = try old.values.finish(s, vw, @intCast(path.len));
                        } else if (old.a.ctx != null) {
                            keys = try s.cat(&.{ old.a, keys }, 2);
                            vals = try s.cat(&.{ old.b, vals }, 2);
                        }
                        cache[layer].a = try mx.retain(keys);
                        cache[layer].b = try mx.retain(vals);
                    }
                }
            } else {
                for (p.groups, plans) |*group, *plan| {
                    var states: [8]A = undefined;
                    var convs: [8]A = undefined;
                    for (p.entries[group.first..][0..group.len], 0..) |entry, st| {
                        states[st] = entry.pass.records[layer].values[5];
                        convs[st] = try s.slice(entry.pass.records[layer].values[6], 1, 0, 3);
                    }
                    const first = p.entries[group.first].first;
                    const last = first + @as(i32, @intCast(group.layout.rows));
                    var vals: [5]A = undefined;
                    for (p.gdn[layer].vals, &vals) |value, *out| out.* = try s.slice(value, 1, first, last);
                    const replayed = try grouped.replay(&p.model.kernels, s, plan, vals, states[0..group.len]);
                    const tails = try grouped.tails(&p.model.kernels, s, plan, try s.slice(p.gdn[layer].qkv, 1, first, last), convs[0..group.len]);
                    for (0..group.len) |st| {
                        const index = group.first + st;
                        if (paths[index].len == 0) continue;
                        next[index][layer].a = try mx.retain(tails[st]);
                        next[index][layer].b = try mx.retain(replayed[st]);
                    }
                }
            }
        }
        var arrays: std.ArrayList(A) = .empty;
        defer arrays.deinit(mx.allocator);
        for (next, paths) |cache, path| if (path.len > 0) for (cache) |c| {
            try arrays.appendSlice(mx.allocator, &.{ c.a, c.b });
        };
        if (arrays.items.len > 0) try mx.evalMany(arrays.items, false);
        for (p.entries, paths, next) |entry, path, *cache| {
            if (path.len == 0) continue;
            for (entry.state.cache, cache) |*old, *replacement| {
                old.deinit();
                old.* = replacement.*;
                replacement.* = .{};
            }
            entry.state.position += @intCast(path.len);
        }
        try p.ticket.advance(.forwarded, .settled);
    }
};

pub fn forward(m: *model.Model, streams: []const Stream) !Pass {
    if (m.round_owner.stage != .idle) return error.ModelRoundActive;
    if (streams.len == 0 or streams.len > 64) return error.InvalidStreams;
    var rows: usize = 0;
    for (streams, 0..) |stream, index| {
        if (stream.state.borrowed) return error.RequestRoundActive;
        if (stream.state.cache.len != 64 or stream.tokens.len != stream.parents.len or stream.tokens.len > 128 - rows) return error.InvalidStreams;
        for (streams[0..index]) |other| if (other.state == stream.state) return error.DuplicateStream;
        for (stream.tokens) |token| if (token < 0 or token >= 248320) return error.InvalidToken;
        const tree = try lanes.Tree.init(stream.parents);
        _ = std.math.add(i32, stream.state.position, @intCast(stream.tokens.len)) catch return error.InvalidStreams;
        _ = std.math.add(i32, std.math.add(i32, stream.state.position, stream.state.rope_delta) catch return error.InvalidStreams, tree.max_depth) catch return error.InvalidStreams;
        if (stream.state.position < 0) return error.InvalidStreams;
        for (stream.state.cache, 0..) |cache, layer| {
            if (cache.a.ctx == null or cache.b.ctx == null) {
                if (cache.a.ctx != null or cache.b.ctx != null or stream.state.position != 0) return error.InvalidCacheState;
                continue;
            }
            const attention_layer = layer % 4 == 3;
            const a_shape: []const i32 = if (attention_layer) &.{ 1, 4, stream.state.position, 256 } else &.{ 1, 3, 10240 };
            const b_shape: []const i32 = if (attention_layer) a_shape else &.{ 1, 48, 128, 128 };
            if (!std.mem.eql(i32, mx.shape(cache.a), a_shape) or !std.mem.eql(i32, mx.shape(cache.b), b_shape) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != (if (attention_layer) mx.bf16 else mx.f32t)) return error.InvalidCacheState;
        }
        rows += stream.tokens.len;
    }
    const entries = try mx.allocator.alloc(Entry, streams.len);
    errdefer mx.allocator.free(entries);
    const groups = try mx.allocator.alloc(Group, (streams.len + 7) / 8);
    errdefer mx.allocator.free(groups);
    var first: i32 = 0;
    for (streams, entries) |stream, *entry| {
        entry.* = .{ .state = stream.state, .first = first, .pass = .{ .count = stream.tokens.len, .start = stream.state.position } };
        @memcpy(entry.pass.parents[0..stream.parents.len], stream.parents);
        first += @intCast(stream.tokens.len);
    }
    for (groups, 0..) |*group, i| {
        const base = i * 8;
        const count = @min(8, streams.len - base);
        var parents: [8][]const i32 = undefined;
        var starts: [8]i32 = undefined;
        for (streams[base..][0..count], 0..) |stream, j| {
            parents[j] = stream.parents;
            starts[j] = stream.state.position;
        }
        group.* = .{ .layout = try grouped.Layout.init(parents[0..count], starts[0..count]), .first = base, .len = count };
    }
    const ticket = try m.round_owner.begin();
    for (streams) |stream| stream.state.borrowed = true;
    var p = Pass{ .model = m, .ticket = ticket, .entries = entries, .groups = groups, .count = rows };
    // Ownership of allocations transfers to p only on success.
    errdefer {
        p.scope.deinit();
        for (streams) |stream| stream.state.borrowed = false;
        ticket.release();
    }
    const s = &p.scope;
    var tokens: [128]i32 = undefined;
    var positions: [128]i32 = undefined;
    for (streams, entries) |stream, entry| {
        const t = try lanes.Tree.init(stream.parents);
        const offset: usize = @intCast(entry.first);
        @memcpy(tokens[offset..][0..stream.tokens.len], stream.tokens);
        for (0..stream.tokens.len) |j| positions[offset + j] = stream.state.position + stream.state.rope_delta + t.depths[j];
    }
    var h = try m.weights.embedArray(s, try s.ints(tokens[0..rows]));
    const pos = try s.ints(positions[0..rows]);
    var pending: ?A = null;
    for (0..64) |layer| {
        const norm = try lanes.norm(&m.kernels, s, h, pending, try m.weight(layer, "input_layernorm.weight"));
        h = norm.h;
        const r = if (layer % 4 == 3) try attention(&p, layer, norm.x, pos) else try recurrence(&p, layer, norm.x);
        const post = try lanes.norm(&m.kernels, s, h, r, try m.weight(layer, "post_attention_layernorm.weight"));
        h = post.h;
        const act = try lanes.mlp(&m.kernels, s, try m.project(s, layer, "mlp.gate_proj", post.x), try m.project(s, layer, "mlp.up_proj", post.x));
        pending = try m.project(s, layer, "mlp.down_proj", act);
        for ([_]usize{ 5, 19, 33, 47, 61 }, 0..) |tap_layer, j| if (layer == tap_layer) {
            const tap = try s.binary(mx.c.mlx_add, h, pending.?);
            for (p.entries) |*entry| entry.pass.taps[j] = try s.slice(tap, 1, entry.first, entry.first + @as(i32, @intCast(entry.pass.count)));
        };
        if (layer == 0 or (layer + 1) % 4 == 0) try mx.evalMany(&.{ h, pending.? }, true);
    }
    const norm = try lanes.norm(&m.kernels, s, h, pending, try m.weights.get("model.norm.weight"));
    p.hidden = norm.x.x;
    p.logits = try (try m.weights.linear("lm_head")).apply(&m.kernels, s, norm.x);
    for (p.entries) |*entry| {
        const end = entry.first + @as(i32, @intCast(entry.pass.count));
        entry.pass.logits = try s.slice(p.logits, 1, entry.first, end);
        entry.pass.hidden = try s.slice(p.hidden, 1, entry.first, end);
    }
    try mx.eval(p.logits);
    for (p.entries) |*entry| try model.Model.observeBuffers(&entry.pass);
    try ticket.advance(.bound, .forwarded);
    return p;
}

fn recurrence(p: *Pass, layer: usize, x: lanes.Act) !A {
    const m = p.model;
    const s = &p.scope;
    const r: i32 = @intCast(p.count);
    const mp = @divTrunc(r + 15, 16) * 16;
    const qkv = try m.project(s, layer, "linear_attn.in_proj_qkv", x);
    const z = try m.project(s, layer, "linear_attn.in_proj_z", x);
    const b = try m.project(s, layer, "linear_attn.in_proj_b", x);
    const a = try m.project(s, layer, "linear_attn.in_proj_a", x);
    const cw = try s.reshape(try m.weight(layer, "linear_attn.conv1d.weight"), &.{ 10240, 4 });
    var pieces: [8][5]A = undefined;
    var ys: [8]A = undefined;
    for (p.groups, 0..) |*group, gi| {
        var cs: [8]A = undefined;
        var st: [8]A = undefined;
        const entries = p.entries[group.first..][0..group.len];
        const first = entries[0].first;
        const end = first + @as(i32, @intCast(group.layout.rows));
        for (entries, 0..) |entry, j| {
            const cache = entry.state.cache[layer];
            cs[j] = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 1, 3, 10240 }, mx.bf16);
            st[j] = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 1, 48, 128, 128 }, mx.f32t);
        }
        pieces[gi] = try grouped.pre(&m.kernels, s, &group.layout, try s.slice(qkv, 1, first, end), cs[0..group.len], cw, try s.slice(a, 1, first, end), try s.slice(b, 1, first, end), try m.weight(layer, "linear_attn.A_log"), try m.weight(layer, "linear_attn.dt_bias"));
        ys[gi] = try grouped.recurrence(&m.kernels, s, &group.layout, pieces[gi], st[0..group.len]);
        for (entries, 0..) |*entry, j| {
            const rec = &entry.pass.records[layer];
            const local = group.layout.firsts[j];
            const last = local + @as(i32, @intCast(entry.pass.count));
            for (pieces[gi], 0..) |value, v| rec.values[v] = try s.slice(value, 1, local, last);
            rec.values[5] = try s.own(try mx.retain(st[j]));
            rec.values[6] = try s.cat(&.{ cs[j], try s.slice(qkv, 1, entry.first, entry.first + @as(i32, @intCast(entry.pass.count))) }, 1);
        }
    }
    for (0..5) |v| {
        var values: [8]A = undefined;
        for (0..p.groups.len) |gi| values[gi] = pieces[gi][v];
        p.gdn[layer].vals[v] = try s.cat(values[0..p.groups.len], 1);
    }
    p.gdn[layer].qkv = qkv;
    const y = try s.cat(ys[0..p.groups.len], 1);
    const post = try m.kernels.run(s, src.lane_glue_gdn_post, &.{ y, z, try m.weight(layer, "linear_attn.norm.weight"), try s.scalar(1e-6), try s.ints(&.{ r, mp }) }, &.{ mx.ti("NV", 48), mx.ti("DV", 128) }, .{ 32, 48, mp }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ 1, r, 6144 } }, .{ .shape = &.{ 96, mp }, .dtype = mx.f32t } });
    return m.project(s, layer, "linear_attn.out_proj", .{ .x = post[0], .sums = post[1] });
}

fn attention(p: *Pass, layer: usize, x: lanes.Act, pos: A) !A {
    const m = p.model;
    const s = &p.scope;
    const r: i32 = @intCast(p.count);
    const qg = try s.reshape(try m.project(s, layer, "self_attn.q_proj", x), &.{ 1, r, 24, 512 });
    var q = try s.rms(try s.slice(qg, 3, 0, 256), try m.weight(layer, "self_attn.q_norm.weight"));
    const gate = try s.reshape(try s.slice(qg, 3, 256, 512), &.{ 1, r, 6144 });
    var key = try s.rms(try s.reshape(try m.project(s, layer, "self_attn.k_proj", x), &.{ 1, r, 4, 256 }), try m.weight(layer, "self_attn.k_norm.weight"));
    const value = try s.transpose(try s.reshape(try m.project(s, layer, "self_attn.v_proj", x), &.{ 1, r, 4, 256 }), &.{ 0, 2, 1, 3 });
    q = try s.transpose(try s.rope(try s.transpose(q, &.{ 1, 2, 0, 3 }), pos, 64), &.{ 2, 1, 0, 3 });
    key = try s.transpose(try s.rope(try s.transpose(key, &.{ 1, 2, 0, 3 }), pos, 64), &.{ 2, 1, 0, 3 });
    var outputs: [64]A = undefined;
    var output_count: usize = 0;
    for (p.groups) |*group| {
        const entries = p.entries[group.first..][0..group.len];
        var keys: [8]A = undefined;
        var values: [8]A = undefined;
        for (entries, 0..) |*entry, st| {
            const end = entry.first + @as(i32, @intCast(entry.pass.count));
            const rec = &entry.pass.records[layer];
            const cache = &entry.state.cache[layer];
            rec.values[0] = try s.slice(key, 2, entry.first, end);
            rec.values[1] = try s.slice(value, 2, entry.first, end);
            keys[st] = rec.values[0];
            values[st] = rec.values[1];
            if (kv.enabled) {
                rec.key_write = try cache.keys.append(s, cache.a, keys[st], 2);
                rec.value_write = try cache.values.append(s, cache.b, values[st], 2);
                keys[st] = rec.key_write.capacity;
                values[st] = rec.value_write.capacity;
            } else if (cache.a.ctx != null) {
                keys[st] = try s.cat(&.{ cache.a, keys[st] }, 2);
                values[st] = try s.cat(&.{ cache.b, values[st] }, 2);
            }
            if (!kv.enabled) {
                keys[st] = try s.contiguous(keys[st]);
                values[st] = try s.contiguous(values[st]);
            }
        }
        if (mx.tensor_units) {
            const first = entries[0].first;
            outputs[output_count] = try grouped.attention(&m.kernels, s, &group.layout, try s.slice(q, 2, first, first + @as(i32, @intCast(group.layout.rows))), keys[0..group.len], values[0..group.len]);
            output_count += 1;
        } else {
            for (entries, 0..) |entry, st| {
                const tree = try lanes.Tree.init(entry.pass.parents[0..entry.pass.count]);
                outputs[output_count] = try lanes.attentionCapacity(&m.kernels, s, try s.slice(q, 2, entry.first, entry.first + @as(i32, @intCast(entry.pass.count))), keys[st], values[st], &tree, entry.pass.start + @as(i32, @intCast(entry.pass.count)));
                output_count += 1;
            }
        }
    }
    const out = try s.reshape(try s.transpose(try s.cat(outputs[0..output_count], 2), &.{ 0, 2, 1, 3 }), &.{ 1, r, 6144 });
    return m.project(s, layer, "self_attn.o_proj", .{ .x = try s.binary(mx.c.mlx_multiply, out, try s.unary(mx.c.mlx_sigmoid, gate)) });
}

test "shared rounds reject aliased requests and invalid geometry before execution" {
    var m: model.Model = undefined;
    m.round_owner = .{};
    var state = try State.init(&m);
    defer state.deinit();
    const stream = Stream{ .state = &state, .tokens = &.{123}, .parents = &.{-1} };
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{}));
    try std.testing.expectError(error.DuplicateStream, forward(&m, &.{ stream, stream }));
    state.borrowed = true;
    try std.testing.expectError(error.RequestRoundActive, forward(&m, &.{stream}));
    try std.testing.expectError(error.RequestRoundActive, state.clone());
    state.borrowed = false;
    state.position = std.math.maxInt(i32);
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{stream}));
    state.position = -1;
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{stream}));
    state.position = 0;
    state.position = 1;
    try std.testing.expectError(error.InvalidCacheState, forward(&m, &.{stream}));
    state.position = 0;
    try std.testing.expectError(error.InvalidToken, forward(&m, &.{.{ .state = &state, .tokens = &.{248320}, .parents = &.{-1} }}));
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{.{ .state = &state, .tokens = &.{ 123, 124 }, .parents = &.{-1} }}));
    try std.testing.expectError(error.InvalidTree, forward(&m, &.{.{ .state = &state, .tokens = &.{ 123, 124 }, .parents = &.{ -1, 1 } }}));
    const ticket = try m.round_owner.begin();
    defer ticket.release();
    try std.testing.expectError(error.ModelRoundActive, forward(&m, &.{stream}));
}
