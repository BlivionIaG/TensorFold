const std = @import("std");
const mx = @import("mlx.zig");
const flash = @import("flash.zig");
const src = @import("kernel_sources.zig");
const kv = @import("kv_buffer.zig");
const round = @import("decode_round.zig");
const A = mx.Array;
const ti = mx.ti;
const max_rows = flash.Model.max_shared_rows;
pub const State = @import("request_state.zig").State(flash.Model);
pub const Stream = struct { state: *State, tokens: []const i32, parents: []const i32 };
const Entry = struct { state: *State, first: i32, pass: flash.Pass, staged: [48]flash.Cache = @splat(.{}) };
const AttentionStream = struct { cache: *flash.Cache, record: *flash.Cache, record_scope: *mx.Scope, first: i32, count: usize };

const gdn_specs = [_]src.Spec{ src.q4_gdn_step_multi1, src.q4_gdn_step_multi2, src.q4_gdn_step_multi3, src.q4_gdn_step_multi4, src.q4_gdn_step_multi5, src.q4_gdn_step_multi6, src.q4_gdn_step_multi7, src.q4_gdn_step_multi8 };
const attention_specs = [_]src.Spec{ src.q4_attn_parts_multi1, src.q4_attn_parts_multi2, src.q4_attn_parts_multi3, src.q4_attn_parts_multi4, src.q4_attn_parts_multi5, src.q4_attn_parts_multi6, src.q4_attn_parts_multi7, src.q4_attn_parts_multi8 };
const score_specs = [_]src.Spec{ src.q4_idx_scores_multi1, src.q4_idx_scores_multi2, src.q4_idx_scores_multi3, src.q4_idx_scores_multi4, src.q4_idx_scores_multi5, src.q4_idx_scores_multi6, src.q4_idx_scores_multi7, src.q4_idx_scores_multi8 };

// The model and request states remain at stable addresses until deinit.
pub const Pass = struct {
    scope: mx.Scope = .{},
    model: *flash.Model,
    ticket: round.Ticket,
    entries: []Entry,
    logits: A = mx.empty,
    hidden: A = mx.empty,
    count: usize,

    pub fn view(p: *Pass, index: usize) !*const flash.Pass {
        try p.ticket.expect(.forwarded);
        if (index >= p.entries.len) return error.InvalidStreams;
        return &p.entries[index].pass;
    }

    pub fn deinit(p: *Pass) void {
        if (!p.ticket.active()) return;
        for (p.entries) |*entry| {
            for (&entry.staged) |*cache| cache.deinit();
            entry.pass.deinit();
            entry.state.borrowed = false;
        }
        p.scope.deinit();
        mx.allocator.free(p.entries);
        p.ticket.release();
    }

    pub fn commit(p: *Pass, paths: []const []const i32) !void {
        try p.ticket.expect(.forwarded);
        errdefer p.ticket.owner.stage = .failed;
        if (paths.len != p.entries.len) return error.InvalidCommit;
        for (p.entries, paths) |entry, path| {
            if (!entry.state.borrowed or entry.state.position != entry.pass.start) return error.InvalidCommit;
            try validatePath(entry.pass.count, path);
        }
        const next = try mx.allocator.alloc([48]flash.Cache, p.entries.len);
        defer mx.allocator.free(next);
        @memset(next, @splat(.{}));
        defer for (next) |*cache| for (cache) |*c| c.deinit();
        var partial = false;
        for (paths) |path| if (path.len > 0) {
            try mx.eval(p.logits);
            break;
        };
        for (p.entries, paths, next) |*entry, path, *cache| {
            if (path.len == entry.pass.count) {
                cache.* = entry.staged;
                entry.staged = @splat(.{});
            } else if (path.len != 0) {
                cache.* = try p.model.committedCache(&entry.pass, entry.state.cache, path.len);
                partial = true;
            }
        }
        if (partial) {
            var arrays: [max_rows * 48 * 6]A = undefined;
            var count: usize = 0;
            for (next) |cache| for (cache) |c| inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
                const value = @field(c, field);
                if (value.ctx != null) {
                    arrays[count] = value;
                    count += 1;
                }
            };
            if (count > 0) try mx.evalMany(arrays[0..count], false);
        }
        for (p.entries) |*entry| try flash.Model.observeBuffers(&entry.pass);
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

fn validatePath(rows: usize, path: []const i32) !void {
    if (path.len > rows) return error.InvalidCommit;
    for (path, 0..) |row, i| if (row != i) return error.InvalidCommit;
}

fn validate(m: *flash.Model, streams: []const Stream) !usize {
    if (m.round_owner.stage != .idle) return error.ModelRoundActive;
    if (streams.len == 0 or streams.len > max_rows) return error.InvalidStreams;
    var rows: usize = 0;
    for (streams, 0..) |stream, index| {
        if (stream.state.borrowed) return error.RequestRoundActive;
        if (stream.state.cache.len != 48 or stream.tokens.len == 0 or stream.tokens.len != stream.parents.len or stream.tokens.len > 16 or stream.tokens.len > max_rows - rows) return error.InvalidStreams;
        for (streams[0..index]) |other| if (other.state == stream.state) return error.DuplicateStream;
        for (stream.tokens) |token| if (token < 0 or token >= flash.Model.vocab) return error.InvalidToken;
        for (stream.parents, 0..) |parent, i| if (parent != @as(i32, @intCast(i)) - 1) return error.InvalidTree;
        if (stream.state.position < 0) return error.InvalidStreams;
        const end = std.math.add(i32, stream.state.position, @intCast(stream.tokens.len)) catch return error.InvalidStreams;
        if (end > std.math.maxInt(i32) - 2047) return error.InvalidStreams;
        for (stream.state.cache, 0..) |cache, layer| {
            if (cache.offset != stream.state.position) return error.InvalidCacheState;
            if (cache.a.ctx == null or cache.b.ctx == null) {
                if (cache.a.ctx != null or cache.b.ctx != null or stream.state.position != 0) return error.InvalidCacheState;
                continue;
            }
            const attention_layer = layer % 4 == 3;
            const a_shape: []const i32 = if (attention_layer) &.{ 1, 2, stream.state.position, 256 } else &.{ 3, 10240 };
            const b_shape: []const i32 = if (attention_layer) a_shape else &.{ 48, 128, 128 };
            const a_dims = mx.shape(cache.a);
            const b_dims = mx.shape(cache.b);
            if (!std.mem.eql(i32, a_dims, a_shape) or !std.mem.eql(i32, b_dims, b_shape) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != (if (attention_layer) mx.bf16 else mx.f32t)) return error.InvalidCacheState;
            if (attention_layer) {
                if (cache.raw.ctx == null or !std.mem.eql(i32, mx.shape(cache.raw), &.{ stream.state.position, 128 }) or mx.dtype(cache.raw) != mx.bf16) return error.InvalidCacheState;
                if (cache.pooled.ctx != null) {
                    const pooled = mx.shape(cache.pooled);
                    if (pooled.len != 2 or pooled[1] != 128 or pooled[0] > @divTrunc(stream.state.position, 4) or mx.dtype(cache.pooled) != mx.bf16) return error.InvalidCacheState;
                }
            }
            if (layer == 1) {
                if (cache.ple.ctx == null or !std.mem.eql(i32, mx.shape(cache.ple), &.{ 9, 10240 }) or mx.dtype(cache.ple) != mx.bf16) return error.InvalidCacheState;
                if (m.gpuTokensEnabled() and (cache.token_history.ctx == null or !std.mem.eql(i32, mx.shape(cache.token_history), &.{2}) or mx.dtype(cache.token_history) != mx.i32t)) return error.InvalidCacheState;
            }
        }
        rows += stream.tokens.len;
    }
    return rows;
}

pub fn forward(m: *flash.Model, streams: []const Stream) !Pass {
    const rows = try validate(m, streams);
    const entries = try mx.allocator.alloc(Entry, streams.len);
    errdefer mx.allocator.free(entries);
    var first: i32 = 0;
    var tokens: [max_rows]i32 = undefined;
    var positions: [max_rows]i32 = undefined;
    for (streams, entries) |stream, *entry| {
        entry.* = .{ .state = stream.state, .first = first, .pass = .{ .count = stream.tokens.len, .start = stream.state.position } };
        @memcpy(entry.pass.tokens[0..stream.tokens.len], stream.tokens);
        const at: usize = @intCast(first);
        @memcpy(tokens[at..][0..stream.tokens.len], stream.tokens);
        for (0..stream.tokens.len) |j| positions[at + j] = stream.state.position + @as(i32, @intCast(j));
        first += @intCast(stream.tokens.len);
    }
    const ticket = try m.round_owner.begin();
    errdefer ticket.release();
    for (streams) |stream| stream.state.borrowed = true;
    errdefer {
        for (streams) |stream| stream.state.borrowed = false;
    }
    var p = Pass{ .model = m, .ticket = ticket, .entries = entries, .count = rows };
    errdefer {
        for (entries) |*entry| {
            for (&entry.staged) |*cache| cache.deinit();
            entry.pass.deinit();
        }
        p.scope.deinit();
    }
    const pos = try p.scope.ints(positions[0..rows]);
    var hn: [2]A = @splat(mx.empty);
    defer for (hn) |value| mx.free(value);
    {
        var scope = mx.Scope{};
        defer scope.deinit();
        const h = try m.embedResidual(&scope, try scope.ints(tokens[0..rows]));
        hn = try retainPair(try m.hcNorm(&scope, h, null, mx.empty));
    }
    var base_buf: [256]u8 = undefined;
    var buf: [256]u8 = undefined;
    for (0..48) |layer| {
        var scope = mx.Scope{};
        defer scope.deinit();
        const s = &scope;
        const input = if (layer == 1) try m.hcNorm(s, try ple(&p, s, hn[0]), null, mx.empty) else hn;
        const base = try std.fmt.bufPrint(&base_buf, "model.layers.{d}", .{layer});
        const mix = try m.hcProject(s, try std.fmt.bufPrint(&buf, "{s}.attn_hyper_connection", .{base}), input[0], input[1], true);
        const branch = if (layer % 4 == 3) try attention(&p, s, layer, try std.fmt.bufPrint(&buf, "{s}.self_attn", .{base}), mix[0], pos) else try recurrence(&p, s, layer, try std.fmt.bufPrint(&buf, "{s}.linear_attn", .{base}), mix[0]);
        const post = try m.hcNorm(s, input[0], branch, mix[1]);
        const mm = try m.hcProject(s, try std.fmt.bufPrint(&buf, "{s}.mlp_hyper_connection", .{base}), post[0], post[1], true);
        const next = try retainPair((try m.moe(s, try std.fmt.bufPrint(&buf, "{s}.mlp", .{base}), post[0], mm[0], mm[1]))[0..2].*);
        for (hn) |value| mx.free(value);
        hn = next;
        var pending: [1 + max_rows * 6]A = undefined;
        pending[0] = hn[0];
        var pending_count: usize = 1;
        for (entries) |*entry| {
            entry.staged[layer] = try m.committedCacheLayer(s, &entry.pass, entry.state.cache[layer], layer, entry.pass.count);
            inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
                const value = @field(entry.staged[layer], field);
                if (value.ctx != null) {
                    pending[pending_count] = value;
                    pending_count += 1;
                }
            }
        }
        try mx.evalMany(pending[0..pending_count], true);
    }
    const s = &p.scope;
    p.hidden = try s.own(try mx.retain(hn[0]));
    p.logits = try m.headWithNorm(s, hn);
    var writes: [max_rows * 48 * 6]A = undefined;
    var write_count: usize = 0;
    for (entries) |*entry| {
        for (entry.staged) |cache| inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
            const value = @field(cache, field);
            if (value.ctx != null) {
                writes[write_count] = value;
                write_count += 1;
            }
        };
    }
    p.logits = try cacheDependency(s, p.logits, writes[0..write_count]);
    for (entries) |*entry| {
        const end = entry.first + @as(i32, @intCast(entry.pass.count));
        entry.pass.hidden = try s.slice(p.hidden, 0, entry.first, end);
        entry.pass.logits = try s.slice(p.logits, 0, entry.first, end);
    }
    try ticket.advance(.bound, .forwarded);
    return p;
}

fn retainPair(values: [2]A) ![2]A {
    const first = try mx.retain(values[0]);
    errdefer mx.free(first);
    return .{ first, try mx.retain(values[1]) };
}

fn cacheDependency(s: *mx.Scope, value: A, arrays: []const A) !A {
    const input = [_]A{value};
    const ins = mx.c.mlx_vector_array_new_data(&input, input.len);
    defer _ = mx.c.mlx_vector_array_free(ins);
    const dependencies = mx.c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = mx.c.mlx_vector_array_free(dependencies);
    var outputs = mx.c.mlx_vector_array_new();
    defer _ = mx.c.mlx_vector_array_free(outputs);
    try mx.check(mx.c.mlx_depends(&outputs, ins, dependencies));
    var output = mx.c.mlx_array_new();
    const rc = mx.c.mlx_vector_array_get(&output, outputs, 0);
    return s.result(rc, output);
}

fn recurrence(p: *Pass, s: *mx.Scope, layer: usize, base: []const u8, x: A) !A {
    const m = p.model;
    const projected = try m.projectStack(s, base, &.{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" }, x);
    const cw = try s.reshape(try m.f(base, "conv1d.weight"), &.{ 10240, 4 });
    const alog = try m.f(base, "A_log");
    const dt = try m.f(base, "dt_bias");
    const nw = try m.f(base, "norm.weight");
    const eps = try m.weights.get("decode.eps");
    var outputs: [(max_rows + 7) / 8]A = undefined;
    var groups: usize = 0;
    var at: usize = 0;
    while (at < p.entries.len) : (at += 8) {
        const entries = p.entries[at..@min(at + 8, p.entries.len)];
        const n = entries.len;
        const first = entries[0].first;
        var starts: [9]i32 = undefined;
        starts[0] = 0;
        var inputs: [23]A = undefined;
        for (entries, 0..) |entry, j| {
            const cache = entry.state.cache[layer];
            inputs[1 + j] = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 3, 10240 }, mx.bf16);
            inputs[1 + n + j] = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 48, 128, 128 }, mx.f32t);
            starts[j + 1] = starts[j] + @as(i32, @intCast(entry.pass.count));
        }
        const rows = starts[n];
        inputs[0] = try s.slice(projected, 0, first, first + rows);
        inputs[1 + 2 * n ..][0..6].* = .{ cw, alog, dt, nw, eps, try s.ints(starts[0 .. n + 1]) };
        const out = try m.kernels.run(s, gdn_specs[n - 1], inputs[0 .. 7 + 2 * n], &.{ ti("NK", 16), ti("NV", 48), ti("DK", 128), ti("DV", 128), ti("TAPS", 4), ti("HAS_STATE", 1) }, .{ 48 * 1024, @intCast(n), 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = &.{ rows, 6144 } }, .{ .shape = &.{ rows, 3, 10240 } }, .{ .shape = &.{ rows, 48, 128, 128 }, .dtype = mx.f32t } });
        outputs[groups] = out[0];
        groups += 1;
        for (entries, 0..) |*entry, j| {
            entry.pass.records[layer].a = try entry.pass.scope.slice(out[1], 0, starts[j], starts[j + 1]);
            entry.pass.records[layer].b = try entry.pass.scope.slice(out[2], 0, starts[j], starts[j + 1]);
        }
    }
    return m.lin(s, base, "out_proj", if (groups == 1) outputs[0] else try s.cat(outputs[0..groups], 0));
}

fn attention(p: *Pass, s: *mx.Scope, layer: usize, base: []const u8, x: A, pos: A) !A {
    var streams: [max_rows]AttentionStream = undefined;
    for (p.entries, streams[0..p.entries.len]) |*entry, *stream| stream.* = .{ .cache = &entry.state.cache[layer], .record = &entry.pass.records[layer], .record_scope = &entry.pass.scope, .first = entry.first, .count = entry.pass.count };
    return attentionStreams(p.model, s, base, x, pos, streams[0..p.entries.len]);
}

fn attentionStreams(m: *flash.Model, s: *mx.Scope, base: []const u8, x: A, pos: A, streams: []const AttentionStream) !A {
    const r = mx.dim(x, 0);
    const projected = try m.projectStack(s, base, &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" }, x);
    const eps = try m.weights.get("decode.eps");
    const log_base = try m.weights.get("decode.log_base");
    const prep = try m.kernels.run(s, src.q4_attn_prep, &.{ projected, pos, try m.scale(s, base, "q_norm.weight"), try m.scale(s, base, "k_norm.weight"), try m.scale(s, base, "indexer.q_layernorm.weight"), eps, log_base }, &.{ ti("NQ", 24), ti("NKV", 2), ti("HD", 256), ti("RD", 64), ti("PW", 13952), ti("NI", 4), ti("IHD", 128) }, .{ 256, 30, r }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 24, 256 } }, .{ .shape = &.{ r, 2, 256 } }, .{ .shape = &.{ r, 4, 128 } } });
    const new_keys = try s.transpose(try s.reshape(prep[1], &.{ 1, r, 2, 256 }), &.{ 0, 2, 1, 3 });
    const new_values = try s.transpose(try s.reshape(try s.slice(projected, 1, 12800, 13312), &.{ 1, r, 2, 256 }), &.{ 0, 2, 1, 3 });
    const new_raw = try s.slice(projected, 1, 13824, 13952);
    var counts: [max_rows]i32 = undefined;
    var complete: [max_rows]i32 = undefined;
    var ends: [max_rows]i32 = undefined;
    var sparse: [max_rows]i32 = undefined;
    for (streams) |entry| {
        const cache = entry.cache;
        const rec = entry.record;
        const rs = entry.record_scope;
        const rows: i32 = @intCast(entry.count);
        const last = entry.first + rows;
        var keys = try rs.slice(new_keys, 2, entry.first, last);
        var values = try rs.slice(new_values, 2, entry.first, last);
        var raw = try rs.slice(new_raw, 0, entry.first, last);
        if (kv.enabled) {
            rec.key_write = try cache.keys.append(rs, cache.a, keys, 2);
            rec.value_write = try cache.values.append(rs, cache.b, values, 2);
            rec.index_write = try cache.index_keys.append(rs, cache.raw, raw, 0);
            keys = rec.key_write.view;
            values = rec.value_write.view;
            raw = rec.index_write.view;
        } else if (cache.a.ctx != null) {
            keys = try rs.cat(&.{ cache.a, keys }, 2);
            values = try rs.cat(&.{ cache.b, values }, 2);
            raw = try rs.cat(&.{ cache.raw, raw }, 0);
        }
        rec.a = keys;
        rec.b = values;
        rec.raw = raw;
        rec.offset = cache.offset + rows;
        rec.pooled = if (cache.pooled.ctx != null) try rs.own(try mx.retain(cache.pooled)) else mx.empty;
        const first: usize = @intCast(entry.first);
        for (0..entry.count) |j| {
            const row = first + j;
            ends[row] = cache.offset + @as(i32, @intCast(j)) + 1;
            complete[row] = @divTrunc(ends[row], 4);
            sparse[row] = @intFromBool(complete[row] > 512);
            counts[row] = if (sparse[row] != 0) 2048 + @mod(ends[row], 4) else ends[row];
        }
        const blocks = @divTrunc(rec.offset, 4);
        if (blocks > 512) {
            const done = if (cache.pooled.ctx != null) mx.dim(cache.pooled, 0) else 0;
            if (blocks > done) {
                const fresh = (try m.kernels.run(s, src.q4_idx_pool, &.{ raw, try s.ints(&.{done}), try m.scale(s, base, "indexer.k_layernorm.weight"), eps, log_base }, &.{ ti("DI", 128), ti("RD", 64) }, .{ 128, blocks - done, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ blocks - done, 128 } }}))[0];
                rec.pooled = if (done > 0) try rs.cat(&.{ cache.pooled, fresh }, 0) else try rs.own(try mx.retain(fresh));
            }
        }
    }
    var outputs: [(max_rows + 7) / 8]A = undefined;
    var groups: usize = 0;
    var at: usize = 0;
    while (at < streams.len) : (at += 8) {
        const entries = streams[at..@min(at + 8, streams.len)];
        const n = entries.len;
        const first = entries[0].first;
        const last = entries[n - 1].first + @as(i32, @intCast(entries[n - 1].count));
        const rows = last - first;
        const begin: usize = @intCast(first);
        const end: usize = @intCast(last);
        var srow: [max_rows]i32 = undefined;
        var caps: [8]i32 = undefined;
        var inputs: [23]A = undefined;
        var pooled: [8]A = undefined;
        var filler = mx.empty;
        var blocks: i32 = 0;
        for (entries, 0..) |entry, j| {
            const rec = entry.record;
            inputs[1 + j] = if (kv.enabled) rec.key_write.capacity else rec.a;
            inputs[1 + n + j] = if (kv.enabled) rec.value_write.capacity else rec.b;
            caps[j] = mx.dim(inputs[1 + j], 2);
            const start: usize = @intCast(entry.first - first);
            @memset(srow[start..][0..entry.count], @as(i32, @intCast(j)));
            pooled[j] = rec.pooled;
            if (rec.pooled.ctx != null) {
                filler = rec.pooled;
                blocks = @max(blocks, mx.dim(rec.pooled, 0));
            }
        }
        const row_stream = try s.ints(srow[0..@intCast(rows)]);
        var ids = try s.zeros(&.{ @max(rows, 8), 1 }, mx.i32t);
        if (filler.ctx != null) {
            var score_inputs: [12]A = undefined;
            score_inputs[0] = try s.slice(prep[2], 0, first, last);
            for (pooled[0..n], 0..) |value, j| score_inputs[1 + j] = if (value.ctx != null) value else try s.slice(filler, 0, 0, 1);
            const ca = try s.ints(complete[begin..end]);
            score_inputs[1 + n ..][0..3].* = .{ ca, row_stream, try s.ints(&.{blocks}) };
            const block_group: i32 = if (rows == 1 or blocks < 4096) 1 else if (blocks < 8192) 2 else if (blocks < 16384) 4 else 8;
            const scores = (try m.kernels.run(s, score_specs[n - 1], score_inputs[0 .. n + 4], &.{ ti("HI", 4), ti("DI", 128), ti("TOP", 512), ti("BB", block_group), ti("RB", 8) }, .{ @divTrunc(blocks + 8 * block_group - 1, 8 * block_group) * 256, @divTrunc(rows + 7, 8), 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, blocks }, .dtype = mx.f32t }}))[0];
            ids = (try m.kernels.run(s, src.q4_idx_select, &.{ scores, ca, try s.ints(ends[begin..end]) }, &.{ ti("TOP", 512), ti("KW", 2051) }, .{ 1024 * rows, 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{ rows, 2051 }, .dtype = mx.i32t }}))[0];
        }
        inputs[0] = try s.slice(prep[0], 0, first, last);
        inputs[1 + 2 * n ..][0..6].* = .{ ids, try s.ints(counts[begin..end]), try s.ints(sparse[begin..end]), try m.weights.get("decode.attention_scale"), row_stream, try s.ints(caps[0..n]) };
        const partial = try m.kernels.run(s, attention_specs[n - 1], inputs[0 .. 7 + 2 * n], &.{ ti("H", 24), ti("KVH", 2), ti("D", 256), ti("P", 16) }, .{ 256 * 24, rows, 16 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ rows, 24, 16, 256 }, .dtype = mx.f32t }, .{ .shape = &.{ rows, 24, 16, 2 }, .dtype = mx.f32t } });
        outputs[groups] = (try m.kernels.run(s, src.q4_attn_merge_gate, &.{ partial[0], partial[1], try s.slice(projected, 0, first, last) }, &.{ ti("H", 24), ti("D", 256), ti("P", 16), ti("PW", 13952) }, .{ 256, 24, rows }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, 6144 } }}))[0];
        groups += 1;
    }
    return m.lin(s, base, "o_proj", if (groups == 1) outputs[0] else try s.cat(outputs[0..groups], 0));
}

pub fn draftStep(m: *flash.Model, s: *mx.Scope, hidden: A, tokens: A, caches: []const *flash.Cache) !A {
    if (caches.len == 0 or caches.len > 8) return error.InvalidDraftRows;
    if (hidden.ctx == null or tokens.ctx == null) return error.InvalidDraftRows;
    const rows: i32 = @intCast(caches.len);
    if (!std.mem.eql(i32, mx.shape(hidden), &.{ rows, 10240 }) or !std.mem.eql(i32, mx.shape(tokens), &.{rows}) or mx.dtype(hidden) != mx.bf16 or (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32)) return error.InvalidDraftRows;
    var records: [8]flash.Cache = @splat(.{});
    var streams: [8]AttentionStream = undefined;
    var positions: [8]i32 = undefined;
    for (caches, 0..) |cache, i| {
        for (caches[0..i]) |other| if (other == cache) return error.DuplicateStream;
        if (cache.offset < 0 or cache.offset > std.math.maxInt(i32) - 2048) return error.InvalidCacheState;
        if (cache.a.ctx == null or cache.b.ctx == null or cache.raw.ctx == null) {
            if (cache.offset != 0 or cache.a.ctx != null or cache.b.ctx != null or cache.raw.ctx != null) return error.InvalidCacheState;
        } else {
            if (!std.mem.eql(i32, mx.shape(cache.a), &.{ 1, 2, cache.offset, 256 }) or !std.mem.eql(i32, mx.shape(cache.b), mx.shape(cache.a)) or !std.mem.eql(i32, mx.shape(cache.raw), &.{ cache.offset, 128 })) return error.InvalidCacheState;
        }
        streams[i] = .{ .cache = cache, .record = &records[i], .record_scope = s, .first = @intCast(i), .count = 1 };
        positions[i] = cache.offset;
    }
    const hn = try m.hcNorm(s, try m.draftInput(s, hidden, tokens), null, mx.empty);
    const mix = try m.hcProject(s, "mtp.layers.0.attn_hyper_connection", hn[0], hn[1], true);
    const branch = try attentionStreams(m, s, "mtp.layers.0.self_attn", mix[0], try s.ints(positions[0..caches.len]), streams[0..caches.len]);
    const post = try m.hcNorm(s, hn[0], branch, mix[1]);
    const mm = try m.hcProject(s, "mtp.layers.0.mlp_hyper_connection", post[0], post[1], true);
    const out = try m.moe(s, "mtp.layers.0.mlp", post[0], mm[0], mm[1]);
    var next: [8]flash.Cache = @splat(.{});
    defer for (&next) |*cache| cache.deinit();
    for (caches, 0..) |cache, i| {
        next[i] = try records[i].clone();
        next[i].keys = try cache.keys.finish(s, records[i].key_write, 1);
        next[i].values = try cache.values.finish(s, records[i].value_write, 1);
        next[i].index_keys = try cache.index_keys.finish(s, records[i].index_write, 1);
    }
    for (caches, next[0..caches.len]) |cache, *replacement| {
        cache.deinit();
        cache.* = replacement.*;
        replacement.* = .{};
    }
    return out[0];
}

pub fn absorbDraft(m: *flash.Model, s: *mx.Scope, hidden: A, tokens: A, lengths: []const usize, caches: []const *flash.Cache) !void {
    if (caches.len > 8 or caches.len != lengths.len) return error.InvalidDraftRows;
    var count: usize = 0;
    for (caches, lengths, 0..) |cache, n, j| {
        if (n > 16 or n > max_rows - count) return error.InvalidDraftRows;
        for (caches[0..j]) |other| if (other == cache) return error.DuplicateStream;
        if (cache.offset < 0 or cache.offset > std.math.maxInt(i32) - 2047 - @as(i32, @intCast(n))) return error.InvalidCacheState;
        if (cache.a.ctx == null or cache.b.ctx == null or cache.raw.ctx == null) {
            if (cache.offset != 0 or cache.a.ctx != null or cache.b.ctx != null or cache.raw.ctx != null or cache.pooled.ctx != null) return error.InvalidCacheState;
        } else {
            if (!std.mem.eql(i32, mx.shape(cache.a), &.{ 1, 2, cache.offset, 256 }) or !std.mem.eql(i32, mx.shape(cache.b), mx.shape(cache.a)) or !std.mem.eql(i32, mx.shape(cache.raw), &.{ cache.offset, 128 }) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16 or mx.dtype(cache.raw) != mx.bf16) return error.InvalidCacheState;
            if (cache.pooled.ctx != null and (mx.shape(cache.pooled).len != 2 or mx.dim(cache.pooled, 1) != 128 or mx.dim(cache.pooled, 0) > @divTrunc(cache.offset, 4) or mx.dtype(cache.pooled) != mx.bf16)) return error.InvalidCacheState;
        }
        if (kv.enabled) inline for (.{ "keys", "values", "index_keys" }, .{ @as(usize, 2), 2, 0 }) |field, axis| {
            const buffer = @field(cache, field);
            if (buffer.current.ctx != null and (buffer.offset != cache.offset or buffer.axis != axis or mx.shape(buffer.current).len <= axis or mx.dim(buffer.current, @intCast(axis)) < cache.offset)) return error.InvalidCacheOffset;
            if (buffer.spare.ctx != null and (buffer.spare_end < 0 or buffer.spare_end > cache.offset or mx.shape(buffer.spare).len <= axis or mx.dim(buffer.spare, @intCast(axis)) < buffer.spare_end)) return error.InvalidCacheOffset;
            if (buffer.spare.ctx != null and buffer.spare_end < cache.offset and (buffer.recent.ctx == null or mx.shape(buffer.recent).len <= axis or mx.dim(buffer.recent, @intCast(axis)) != cache.offset - buffer.spare_end)) return error.InvalidCacheOffset;
        };
        count += n;
    }
    if (count == 0) return;
    const rows: i32 = @intCast(count);
    if (hidden.ctx == null or tokens.ctx == null or !std.mem.eql(i32, mx.shape(hidden), &.{ rows, 10240 }) or !std.mem.eql(i32, mx.shape(tokens), &.{rows}) or mx.dtype(hidden) != mx.bf16 or (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32)) return error.InvalidDraftRows;
    const hn = try m.hcNorm(s, try m.draftInput(s, hidden, tokens), null, mx.empty);
    const mix = try m.hcProject(s, "mtp.layers.0.attn_hyper_connection", hn[0], hn[1], true);
    const base = "mtp.layers.0.self_attn";
    const projected = try m.projectCacheStack(s, base, mix[0]);
    var positions: [max_rows]i32 = undefined;
    var at: usize = 0;
    for (caches, lengths) |cache, n| {
        for (positions[at..][0..n], 0..) |*position, j| position.* = cache.offset + @as(i32, @intCast(j));
        at += n;
    }
    const weight = try m.scale(s, base, "k_norm.weight");
    const eps = try m.weights.get("decode.eps");
    const log_base = try m.weights.get("decode.log_base");
    // NQ/NI=0 preserves the production key norm/RoPE while omitting unused query heads.
    const prepared = try m.kernels.run(s, src.q4_attn_prep, &.{ projected, try s.ints(positions[0..count]), weight, weight, weight, eps, log_base }, &.{ ti("NQ", 0), ti("NKV", 2), ti("HD", 256), ti("RD", 64), ti("PW", 1152), ti("NI", 0), ti("IHD", 128) }, .{ 256, 2, rows }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{1} }, .{ .shape = &.{ rows, 2, 256 } }, .{ .shape = &.{1} } });
    var next: [8]flash.Cache = @splat(.{});
    defer for (&next) |*cache| cache.deinit();
    var first: i32 = 0;
    for (caches, lengths, 0..) |cache, n, j| {
        if (n == 0) continue;
        const length: i32 = @intCast(n);
        const end = first + length;
        var rec = flash.Cache{ .offset = cache.offset + length };
        var keys = try s.transpose(try s.reshape(try s.slice(prepared[1], 0, first, end), &.{ 1, length, 2, 256 }), &.{ 0, 2, 1, 3 });
        const local = try s.slice(projected, 0, first, end);
        var values = try s.transpose(try s.reshape(try s.slice(local, 1, 512, 1024), &.{ 1, length, 2, 256 }), &.{ 0, 2, 1, 3 });
        var raw = try s.slice(local, 1, 1024, 1152);
        if (kv.enabled) {
            rec.key_write = try cache.keys.append(s, cache.a, keys, 2);
            rec.value_write = try cache.values.append(s, cache.b, values, 2);
            rec.index_write = try cache.index_keys.append(s, cache.raw, raw, 0);
            keys = rec.key_write.view;
            values = rec.value_write.view;
            raw = rec.index_write.view;
        } else if (cache.a.ctx != null) {
            keys = try s.cat(&.{ cache.a, keys }, 2);
            values = try s.cat(&.{ cache.b, values }, 2);
            raw = try s.cat(&.{ cache.raw, raw }, 0);
        }
        rec.a = keys;
        rec.b = values;
        rec.raw = raw;
        rec.pooled = cache.pooled;
        const blocks = @divTrunc(rec.offset, 4);
        if (blocks > 512) {
            const done = if (cache.pooled.ctx != null) mx.dim(cache.pooled, 0) else 0;
            if (blocks > done) {
                const fresh = (try m.kernels.run(s, src.q4_idx_pool, &.{ raw, try s.ints(&.{done}), try m.scale(s, base, "indexer.k_layernorm.weight"), eps, log_base }, &.{ ti("DI", 128), ti("RD", 64) }, .{ 128, blocks - done, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ blocks - done, 128 } }}))[0];
                rec.pooled = if (done > 0) try s.cat(&.{ cache.pooled, fresh }, 0) else fresh;
            }
        }
        next[j] = try rec.clone();
        next[j].keys = try cache.keys.finish(s, rec.key_write, length);
        next[j].values = try cache.values.finish(s, rec.value_write, length);
        next[j].index_keys = try cache.index_keys.finish(s, rec.index_write, length);
        first = end;
    }
    for (caches, lengths, next[0..caches.len]) |cache, n, *replacement| {
        if (n == 0) continue;
        cache.deinit();
        cache.* = replacement.*;
        replacement.* = .{};
    }
}

fn ple(p: *Pass, s: *mx.Scope, h: A) !A {
    const m = p.model;
    var ids: [max_rows * 16]i64 = undefined;
    var gpu_ids: [max_rows]A = undefined;
    for (p.entries, 0..) |*entry, index| {
        const cache = entry.state.cache[1];
        const rec = &entry.pass.records[1];
        const tokens = entry.pass.tokens[0..entry.pass.count];
        if (m.gpuTokensEnabled()) {
            const previous = if (cache.token_history.ctx != null) cache.token_history else try s.ints(&.{ 248044, 248044 });
            rec.token_history = try entry.pass.scope.cat(&.{ previous, try s.ints(tokens) }, 0);
            gpu_ids[index] = try m.ngram.idsArray(s, rec.token_history);
        } else {
            var history = cache.history;
            const first: usize = @intCast(entry.first);
            for (tokens, 0..) |token, row| {
                @memcpy(ids[(first + row) * 16 ..][0..16], &m.ngram.ids(history, token));
                history = .{ history[1], token };
            }
            rec.history = history;
        }
    }
    const emb = if (m.gpuTokensEnabled()) try m.ple_tables.?.resident.?.gather(&m.kernels, s, try s.cat(gpu_ids[0..p.entries.len], 0)) else try s.reshape(try m.ple_tables.?.gather(s, ids[0 .. p.count * 16]), &.{ @as(i32, @intCast(p.count)), 2560 });
    const out = try m.pleGate(s, h, emb);
    var outputs: [max_rows]A = undefined;
    for (p.entries, 0..) |*entry, index| {
        const end = entry.first + @as(i32, @intCast(entry.pass.count));
        outputs[index] = try m.pleConv(s, try s.slice(h, 0, entry.first, end), try s.slice(out[0], 0, entry.first, end), try s.slice(out[1], 0, entry.first, end), entry.state.cache[1], &entry.pass.records[1]);
        entry.pass.records[1].ple = try entry.pass.scope.own(try mx.retain(entry.pass.records[1].ple));
    }
    return if (p.entries.len == 1) outputs[0] else s.cat(outputs[0..p.entries.len], 0);
}

fn equalArray(actual: A, expected: A) !void {
    try std.testing.expectEqual(expected.ctx == null, actual.ctx == null);
    if (actual.ctx == null) return;
    var s = mx.Scope{};
    defer s.deinit();
    try @import("sampling_checks.zig").equal(&s, actual, expected);
}

fn equalState(actual: *const State, expected: *const State) !void {
    try std.testing.expectEqual(expected.position, actual.position);
    for (actual.cache, expected.cache, 0..) |a, b, layer| {
        errdefer std.debug.print("Flash shared cache mismatch at layer {d}\n", .{layer});
        try std.testing.expectEqual(b.offset, a.offset);
        try std.testing.expectEqualSlices(i32, &b.history, &a.history);
        inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| try equalArray(@field(a, field), @field(b, field));
    }
}

pub fn checkAttention(m: *flash.Model, cases: anytype) !void {
    if (cases.len == 0) return error.EmptyFixtures;
    const base = "model.layers.3.self_attn";
    for ([_]usize{ 3, 9 }) |n| {
        var scope = mx.Scope{};
        defer scope.deinit();
        const s = &scope;
        var caches: [9]flash.Cache = @splat(.{});
        defer for (&caches) |*cache| cache.deinit();
        var references: [9]flash.Cache = @splat(.{});
        defer for (&references) |*cache| cache.deinit();
        var records: [9]flash.Cache = @splat(.{});
        var streams: [9]AttentionStream = undefined;
        var inputs: [9]A = undefined;
        var positions: [max_rows]i32 = undefined;
        var first: i32 = 0;
        for (0..n) |i| {
            const case = cases[i % cases.len];
            const rows: i32 = if (n == 3) (if (i == 0) 8 else 4) else if (i < 7) 2 else 1;
            const old = flash.Cache{ .offset = case.past, .a = try m.weights.field(case.key, "a"), .b = try m.weights.field(case.key, "b"), .raw = try m.weights.field(case.key, "raw"), .pooled = if (case.pooled > 0) try m.weights.field(case.key, "pooled") else mx.empty };
            caches[i] = try old.clone();
            references[i] = try old.clone();
            inputs[i] = try s.slice(try m.weights.field(case.key, "x"), 0, 0, rows);
            streams[i] = .{ .cache = &caches[i], .record = &records[i], .record_scope = s, .first = first, .count = @intCast(rows) };
            for (0..@intCast(rows)) |j| positions[@as(usize, @intCast(first)) + j] = case.past + @as(i32, @intCast(j));
            first += rows;
        }
        const actual = try attentionStreams(m, s, base, try s.cat(inputs[0..n], 0), try s.ints(positions[0..@intCast(first)]), streams[0..n]);
        for (streams[0..n], 0..) |stream, i| {
            var rec = flash.Cache{};
            const expected = try m.attention(s, base, inputs[i], &references[i], &rec);
            try equalArray(try s.slice(actual, 0, stream.first, stream.first + @as(i32, @intCast(stream.count))), expected);
            inline for (.{ "a", "b", "raw", "pooled" }) |field| try equalArray(@field(records[i], field), @field(rec, field));
            const end = references[i].offset + 1;
            const shared_prefix = try records[i].attentionPrefix(s, end);
            const isolated_prefix = try rec.attentionPrefix(s, end);
            inline for (.{ "a", "b", "raw", "pooled" }) |field| try equalArray(@field(shared_prefix, field), @field(isolated_prefix, field));
        }
    }
    std.debug.print("PASS: Flash shared sparse attention, mixed thresholds, delayed pools and rollback at 3/9 streams.\n", .{});
}

pub fn checkDraft(m: *flash.Model) !void {
    if (!m.mtp) return error.MissingDraftHead;
    try checkCompiledMoE(m, "mtp.layers.0.mlp");
    var scope = mx.Scope{};
    defer scope.deinit();
    const s = &scope;
    var caches: [8]flash.Cache = @splat(.{});
    defer for (&caches) |*cache| cache.deinit();
    var reference: [8]flash.Cache = @splat(.{});
    defer for (&reference) |*cache| cache.deinit();
    var pointers: [8]*flash.Cache = undefined;
    var tokens: [8]i32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(1230 + i);
    const emb = try m.weights.embedArray(s, "model.embed_tokens", try s.ints(&tokens));
    var h = try s.cat(&.{ emb, emb, emb, emb }, -1);
    for (&caches, &reference, &pointers, 0..) |*cache, *ref, *pointer, i| {
        pointer.* = cache;
        const warm = i % 3;
        if (warm != 0) {
            const row = try s.slice(h, 0, @intCast(i), @intCast(i + 1));
            const hs = [_]A{ row, row };
            const ids = [_]i32{ tokens[i], tokens[i] + 17 };
            _ = try m.draftStepArray(s, try s.cat(hs[0..warm], 0), try s.ints(ids[0..warm]), cache, false);
        }
        ref.* = try cache.clone();
    }
    for (0..3) |step| {
        for (&tokens, 0..) |*token, i| token.* = @intCast(1300 + step * 37 + i);
        var expected: [8]A = undefined;
        for (&reference, 0..) |*cache, i| expected[i] = try m.draftStepArray(s, try s.slice(h, 0, @intCast(i), @intCast(i + 1)), try s.ints(tokens[i..][0..1]), cache, false);
        h = try m.draftStepStreams(s, h, try s.ints(&tokens), &pointers);
        const head = try m.draftHead(s, h);
        for (caches, reference, 0..) |actual, ref, i| {
            try equalArray(try s.slice(h, 0, @intCast(i), @intCast(i + 1)), expected[i]);
            try equalArray(try s.slice(head, 0, @intCast(i), @intCast(i + 1)), try m.draftHead(s, expected[i]));
            try std.testing.expectEqual(ref.offset, actual.offset);
            inline for (.{ "a", "b", "raw", "pooled" }) |field| try equalArray(@field(actual, field), @field(ref, field));
        }
    }
    try checkDraftAbsorption(m);
    std.debug.print("PASS: Flash shared 8-stream MTP hidden states, heads and independent caches through three chained steps.\n", .{});
}

fn seedBuffer(s: *mx.Scope, buffer: *kv.Buffer, value: A, axis: usize) !A {
    const n = mx.dim(value, @intCast(axis));
    const first = if (n > 1) n - 1 else n;
    const write = try buffer.append(s, mx.empty, try s.slice(value, axis, 0, first), axis);
    buffer.* = try buffer.finish(s, write, first);
    if (first == n) return write.view;
    const last = try buffer.append(s, write.view, try s.slice(value, axis, first, n), axis);
    const next = try buffer.finish(s, last, 1);
    buffer.deinit();
    buffer.* = next;
    return last.view;
}

fn seedDraftCache(m: *flash.Model, index: usize, past: i32, blocks: i32) !flash.Cache {
    var cache = flash.Cache{ .offset = past };
    errdefer cache.deinit();
    if (past == 0) return cache;
    var scope = mx.Scope{};
    defer scope.deinit();
    const values = try mx.allocator.alloc(f32, @as(usize, @intCast(past)) * 512);
    defer mx.allocator.free(values);
    inline for (.{ "a", "b", "raw" }, .{ "keys", "values", "index_keys" }, 0..) |field, buffer_field, part| {
        const count: usize = @as(usize, @intCast(past)) * (if (part == 2) @as(usize, 128) else 512);
        for (values[0..count], 0..) |*value, p| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((p * 13 + index * 7 + part * 3) % 97)) - 48)) / 64;
        const shape: []const i32 = if (part == 2) &.{ past, 128 } else &.{ 1, 2, past, 256 };
        var value = try scope.cast(try scope.data(values.ptr, shape, mx.f32t), mx.bf16);
        if (kv.enabled) value = try seedBuffer(&scope, &@field(cache, buffer_field), value, if (part == 2) 0 else 2);
        @field(cache, field) = try mx.retain(value);
    }
    if (blocks > 0) cache.pooled = try mx.retain((try m.kernels.run(&scope, src.q4_idx_pool, &.{ cache.raw, try scope.ints(&.{0}), try m.scale(&scope, "mtp.layers.0.self_attn", "indexer.k_layernorm.weight"), try scope.scalar(1e-6), try scope.scalar(@log2(@as(f32, 10000000))) }, &.{ ti("DI", 128), ti("RD", 64) }, .{ 128, blocks, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ blocks, 128 } }}))[0]);
    try evalDraftCaches(&.{&cache});
    return cache;
}

fn evalDraftCaches(caches: []const *flash.Cache) !void {
    var arrays: [8 * 15]A = undefined;
    var n: usize = 0;
    for (caches) |cache| {
        inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
            const value = @field(cache, field);
            if (value.ctx != null) {
                arrays[n] = value;
                n += 1;
            }
        }
        inline for (.{ "keys", "values", "index_keys" }) |field| inline for (.{ "current", "spare", "recent" }) |part| {
            const value = @field(@field(cache, field), part);
            if (value.ctx != null) {
                arrays[n] = value;
                n += 1;
            }
        };
    }
    if (n > 0) try mx.evalMany(arrays[0..n], false);
}

fn equalDraftCache(s: *mx.Scope, actual: flash.Cache, expected: flash.Cache) !void {
    try std.testing.expectEqual(expected.offset, actual.offset);
    try std.testing.expectEqualSlices(i32, &expected.history, &actual.history);
    inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
        const a = @field(actual, field);
        const b = @field(expected, field);
        try std.testing.expectEqual(b.ctx == null, a.ctx == null);
        if (a.ctx != null) {
            try std.testing.expectEqualSlices(i32, mx.shape(b), mx.shape(a));
            try @import("variant_checks.zig").equalBits(s, b, a);
        }
    }
}

fn checkDraftAbsorption(m: *flash.Model) !void {
    const previous_buffers = kv.enabled;
    defer kv.enabled = previous_buffers;
    for ([_]bool{ false, true }) |buffered| {
        kv.enabled = buffered;
        var scope = mx.Scope{};
        defer scope.deinit();
        var seeds: [8]flash.Cache = @splat(.{});
        defer for (&seeds) |*cache| cache.deinit();
        var actual: [8]flash.Cache = @splat(.{});
        defer for (&actual) |*cache| cache.deinit();
        var expected: [8]flash.Cache = @splat(.{});
        defer for (&expected) |*cache| cache.deinit();
        var pointers: [8]*flash.Cache = undefined;
        const pasts = [_]i32{ 0, 1, 3, 2044, 2051, 2052, 2063, 2055 };
        const pooled = [_]i32{ 0, 0, 0, 0, 0, 513, 500, 513 };
        for (&seeds, &actual, &expected, &pointers, pasts, pooled, 0..) |*seed, *cache, *reference, *pointer, past, blocks, j| {
            seed.* = try seedDraftCache(m, j, past, blocks);
            cache.* = try seed.clone();
            reference.* = try seed.clone();
            pointer.* = cache;
        }
        var tokens: [16]i32 = undefined;
        for (&tokens, 0..) |*token, j| token.* = @intCast(1300 + 37 * j);
        const e = try m.weights.embedArray(&scope, "model.embed_tokens", try scope.ints(&tokens));
        const hidden = try scope.cat(&.{ e, e, e, e }, -1);
        try @import("neural_draft.zig").checkAbsorbStreams(m, hidden, &tokens, &seeds);
        const lengths = [_]usize{ 0, 1, 2, 3, 4, 1, 2, 3 };
        var at: usize = 0;
        for (&expected, lengths) |*cache, n| {
            for (0..n) |j| {
                const row: i32 = @intCast(at + j);
                _ = try m.draftStepArray(&scope, try scope.slice(hidden, 0, row, row + 1), try scope.ints(tokens[at + j ..][0..1]), cache, false);
            }
            at += n;
        }
        try m.absorbDraftStreams(&scope, hidden, try scope.ints(&tokens), &lengths, &pointers);
        try evalDraftCaches(&pointers);
        for (actual, expected, 0..) |cache, reference, j| {
            errdefer std.debug.print("Flash MTP absorption cache mismatch: buffered={}, stream={d}\n", .{ buffered, j });
            try equalDraftCache(&scope, cache, reference);
            var original = try seedDraftCache(m, j, pasts[j], pooled[j]);
            defer original.deinit();
            try equalDraftCache(&scope, seeds[j], original);
        }
        const input = try scope.slice(hidden, 0, 0, 8);
        const continuation = try m.draftStepStreams(&scope, input, try scope.ints(tokens[0..8]), &pointers);
        const head = try m.draftHead(&scope, continuation);
        for (&expected, actual, 0..) |*cache, appended, j| {
            const row: i32 = @intCast(j);
            const isolated = try m.draftStepArray(&scope, try scope.slice(input, 0, row, row + 1), try scope.ints(tokens[j..][0..1]), cache, false);
            try equalArray(try scope.slice(continuation, 0, row, row + 1), isolated);
            try equalArray(try scope.slice(head, 0, row, row + 1), try m.draftHead(&scope, isolated));
            try equalDraftCache(&scope, appended, cache.*);
        }
        var wide_tokens: [64]i32 = undefined;
        for (&wide_tokens, 0..) |*token, j| token.* = @intCast(1900 + 17 * j);
        const wide_ids = try scope.ints(&wide_tokens);
        const wide_hidden = try m.embedResidual(&scope, wide_ids);
        for (&expected, 0..) |*cache, j| {
            const first: i32 = @intCast(j * 8);
            _ = try m.draftStepArray(&scope, try scope.slice(wide_hidden, 0, first, first + 8), try scope.slice(wide_ids, 0, first, first + 8), cache, false);
        }
        try m.absorbDraftStreams(&scope, wide_hidden, wide_ids, &@as([8]usize, @splat(8)), &pointers);
        try evalDraftCaches(&pointers);
        for (actual, expected) |cache, reference| try equalDraftCache(&scope, cache, reference);
        const unchanged = actual[0].a.ctx;
        try std.testing.expectError(error.DuplicateStream, m.absorbDraftStreams(&scope, input, try scope.ints(tokens[0..8]), &.{ 4, 4 }, &.{ &actual[0], &actual[0] }));
        try std.testing.expectError(error.InvalidDraftRows, m.absorbDraftStreams(&scope, hidden, try scope.ints(&tokens), &.{ 17, 0 }, pointers[0..2]));
        try std.testing.expectError(error.InvalidDraftRows, m.absorbDraftStreams(&scope, wide_hidden, wide_ids, &.{ 16, 16, 16, 16, 1 }, pointers[0..5]));
        try std.testing.expectEqual(unchanged, actual[0].a.ctx);
        try m.absorbDraftStreams(&scope, mx.empty, mx.empty, &.{}, &.{});
    }
    std.debug.print("PASS: Flash MTP cache-only absorption through64 rows, eight ragged segments, sparse threshold/delayed pooling, retained snapshots and exact continuation with/without capacity buffers.\n", .{});
}

fn checkCompiledMoE(m: *flash.Model, base: []const u8) !void {
    if (!mx.tensor_units or m.weights.flash_drafts == null) return;
    const exact = @import("variant_checks.zig").equalBits;
    var fixture = mx.Scope{};
    defer fixture.deinit();
    var tokens: [128]i32 = undefined;
    for (&tokens, 0..) |*token, i| token.* = @intCast(1000 + i * 17);
    const x = try m.weights.embedArray(&fixture, "model.embed_tokens", try fixture.ints(&tokens));
    const h = try fixture.cat(&.{ x, x, x, x }, -1);
    const inject = try fixture.binary(mx.c.mlx_add, try fixture.zeros(&.{ 128, 4 }, mx.bf16), try fixture.cast(try fixture.scalar(0.125), mx.bf16));
    try mx.evalMany(&.{ h, x, inject }, false);
    var compiled: flash.CompiledMoE = .{};
    defer compiled.deinit();
    {
        var source = mx.Scope{};
        defer source.deinit();
        var inputs = try m.moeInputs(&source, base);
        inputs.router = try source.reshape(inputs.router, mx.shape(inputs.router));
        inline for (.{ "gate", "up", "shared_gate", "shared_up", "down", "shared_down" }) |field| {
            for (&@field(inputs, field).arrays) |*array| {
                array.* = try source.reshape(array.*, mx.shape(array.*));
            }
        }
        compiled = try flash.CompiledMoE.init(inputs);
    }
    var resident: u64 = 0;
    var workspace: u64 = 0;
    var closure: ?*anyopaque = null;
    const cases = [_]i32{ 64, 1, 2, 3, 4, 8, 16, 17, 31, 32, 33, 48, 63, 1 };
    for (0..2) |repetition| for (cases, 0..) |rows, i| {
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        try mx.check(mx.c.mlx_clear_cache());
        const before = try @import("memory_runtime.zig").activeBytes();
        if (repetition == 0 and i == 0) try mx.check(mx.c.mlx_reset_peak_memory());
        {
            var s = mx.Scope{};
            defer s.deinit();
            var source = mx.Scope{};
            const actual = blk: {
                defer source.deinit();
                const first: i32 = @intCast(repetition * 64);
                break :blk try compiled.call(&m.kernels, &s, try source.slice(h, 0, first, first + rows), try source.slice(x, 0, first, first + rows), try source.slice(inject, 0, first, first + rows));
            };
            const first: i32 = @intCast(repetition * 64);
            const hs = try s.slice(h, 0, first, first + rows);
            const xs = try s.slice(x, 0, first, first + rows);
            const injections = try s.slice(inject, 0, first, first + rows);
            const expected = try m.moeDirect(&s, base, hs, xs, injections);
            const cached = try m.moe(&s, base, hs, xs, injections);
            for (actual, expected, cached) |got, want, reused| {
                try std.testing.expectEqual(want.ctx == null, got.ctx == null);
                try std.testing.expectEqual(want.ctx == null, reused.ctx == null);
                if (want.ctx == null) continue;
                try std.testing.expectEqualSlices(i32, mx.shape(want), mx.shape(got));
                try std.testing.expectEqual(mx.dtype(want), mx.dtype(got));
                try exact(&s, want, got);
                try exact(&s, want, reused);
            }
            const current = m.moe_plans.get(base).?.closure.ctx.?;
            if (closure) |previous| try std.testing.expectEqual(previous, current) else closure = current;
            if (repetition == 0 and i == 0) {
                var peak: usize = 0;
                try mx.check(mx.c.mlx_get_peak_memory(&peak));
                workspace = peak -| before;
            }
        }
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        const after = try @import("memory_runtime.zig").activeBytes();
        if (repetition == 0 and i == 0) resident = after else if (after > resident +| workspace) return error.CompiledWeightsDuplicated;
    };
    var s = mx.Scope{};
    defer s.deinit();
    const hs = try s.slice(h, 0, 0, 4);
    const xs = try s.slice(x, 0, 0, 4);
    const injections = try s.slice(inject, 0, 0, 4);
    try std.testing.expectError(error.InvalidLaneWidth, compiled.call(&m.kernels, &s, hs, try s.zeros(&.{ 65, 2560 }, mx.bf16), injections));
    try std.testing.expectError(error.InvalidLaneWidth, m.moe(&s, base, hs, try s.zeros(&.{ 0, 2560 }, mx.bf16), injections));
    const generation = mx.gpu_generation;
    defer mx.gpu_generation = generation;
    for ([_]u32{ 13, 15, 18 }) |other| if (other != generation) {
        mx.gpu_generation = other;
        try std.testing.expectError(error.InvalidCompiledGeneration, compiled.call(&m.kernels, &s, hs, xs, injections));
        const actual = try m.moe(&s, base, hs, xs, injections);
        const expected = try m.moeDirect(&s, base, hs, xs, injections);
        for (actual[0..2], expected[0..2]) |got, want| try exact(&s, want, got);
        try std.testing.expectEqual(generation, m.moe_plans.get(base).?.generation);
    };
    mx.gpu_generation = generation;
    const tensor_units = mx.tensor_units;
    defer mx.tensor_units = tensor_units;
    mx.tensor_units = false;
    const actual = try m.moe(&s, base, hs, xs, injections);
    const expected = try m.moeDirect(&s, base, hs, xs, injections);
    for (actual[0..2], expected[0..2]) |got, want| try exact(&s, want, got);
    try std.testing.expect(m.moe_plans.count() <= 49);
    std.debug.print("PASS: Flash compiled {s} matches the direct graph across thirteen row shapes and changed inputs, released source handles, cached closure reuse, bounded residency and generation/SIMD bypass.\n", .{base});
}

fn checkProjectionStacks(m: *flash.Model) !void {
    for ([_]usize{ 1, 8, 16, 32, 64 }) |rows| {
        var scope = mx.Scope{};
        defer scope.deinit();
        const s = &scope;
        var tokens: [max_rows]i32 = undefined;
        for (tokens[0..rows], 0..) |*token, i| token.* = if (i % 3 == 0) 0 else if (i % 3 == 1) flash.Model.vocab - 1 else @intCast(123 + 17 * i);
        const ids = try s.ints(tokens[0..rows]);
        const x = try m.weights.embedArray(s, "model.embed_tokens", ids);
        const embedding = try m.embedResidual(s, ids);
        try equalArray(embedding, try s.cat(&.{ x, x, x, x }, -1));
        try equalArray(embedding, try m.embedResidual(s, try s.cast(ids, mx.c.MLX_UINT32)));
        inline for (.{
            .{ "model.layers.0.linear_attn", .{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" } },
            .{ "model.layers.3.self_attn", .{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" } },
        }) |group| {
            const names = [_][]const u8{ group[1][0], group[1][1], group[1][2], group[1][3] };
            const actual = try m.projectStack(s, group[0], &names, x);
            var independent: [max_rows]A = undefined;
            for (0..rows) |row| independent[row] = try m.projectStack(s, group[0], &names, try s.slice(x, 0, @intCast(row), @intCast(row + 1)));
            try equalArray(actual, try s.cat(independent[0..rows], 0));
            var parts: [4]A = undefined;
            var first: i32 = 0;
            var name: [256]u8 = undefined;
            for (names, &parts) |member, *part| {
                const weight = try m.weights.affine(try std.fmt.bufPrint(&name, "{s}.{s}", .{ group[0], member }));
                const end = first + (try weight.geometry(2)).n;
                part.* = try s.slice(actual, 1, first, end);
                first = end;
            }
            try std.testing.expectEqual(mx.dim(actual, 1), first);
            try equalArray(actual, try s.cat(&parts, -1));
        }
    }
    {
        var s = mx.Scope{};
        defer s.deinit();
        try std.testing.expectError(error.InvalidToken, m.embedResidual(&s, mx.empty));
        try std.testing.expectError(error.InvalidToken, m.embedResidual(&s, try s.zeros(&.{1}, mx.f32t)));
        try std.testing.expectError(error.InvalidToken, m.embedResidual(&s, try s.zeros(&.{ 1, 1 }, mx.i32t)));
        try std.testing.expectError(error.InvalidLaneWidth, m.embedResidual(&s, try s.zeros(&.{max_rows + 1}, mx.i32t)));
    }
    std.debug.print("PASS: Flash fused embeddings and stacked projections exact against independent paths at 1/8/16/32/64 rows.\n", .{});
}

fn checkShortPrefill(m: *flash.Model) !void {
    const previous_buffers = kv.enabled;
    defer kv.enabled = previous_buffers;
    const exact = @import("variant_checks.zig").equalBits;
    for ([_]bool{ false, true }) |buffered| {
        kv.enabled = buffered;
        var state = try State.init(m);
        defer state.deinit();
        var reference = try State.init(m);
        defer reference.deinit();
        for ([_]usize{ 1, 7, 8, 16, 1 }, 0..) |rows, step| {
            const keep = if (step == 1) 3 else if (step == 2) 0 else rows;
            var tokens: [16]i32 = undefined;
            for (tokens[0..rows], 0..) |*token, j| token.* = @intCast(2300 + step * 37 + j);
            var snapshot = try state.clone();
            defer snapshot.deinit();
            var expected_snapshot = try reference.clone();
            defer expected_snapshot.deinit();
            var saved = mx.Scope{};
            defer saved.deinit();
            var hidden: A = undefined;
            var logits: A = undefined;
            {
                reference.swap(m);
                defer reference.swap(m);
                var pass = try m.forward(tokens[0..rows]);
                defer pass.deinit();
                hidden = try saved.own(try mx.retain(pass.hidden));
                logits = try saved.own(try mx.retain(try pass.scope.slice(pass.logits, 0, @intCast(rows - 1), @intCast(rows))));
                if (keep != 0) try m.commit(&pass, keep);
            }
            {
                state.swap(m);
                defer state.swap(m);
                const position = m.position;
                var pass = if (step == 4) try m.forward(tokens[0..rows]) else try m.prefill(tokens[0..rows]);
                defer pass.deinit();
                try std.testing.expectEqual(position, m.position);
                try std.testing.expectEqual(rows, pass.count);
                try std.testing.expectEqualSlices(i32, &.{ @as(i32, @intCast(rows)), 10240 }, mx.shape(pass.hidden));
                try std.testing.expectEqualSlices(i32, &.{ 1, flash.Model.vocab }, mx.shape(pass.logits));
                try exact(&saved, hidden, pass.hidden);
                try exact(&saved, logits, pass.logits);
                if (keep != 0) try m.commit(&pass, keep) else try std.testing.expectError(error.InvalidCommit, m.commit(&pass, 0));
            }
            try std.testing.expectEqual(reference.position, state.position);
            for (state.cache, reference.cache) |actual, expected| try equalDraftCache(&saved, actual, expected);
            for (snapshot.cache, expected_snapshot.cache) |actual, expected| try equalDraftCache(&saved, actual, expected);
            if (keep == 0) for (state.cache, snapshot.cache) |actual, expected| try equalDraftCache(&saved, actual, expected);
        }
    }
    std.debug.print("PASS: Flash short prefill keeps full hidden rows and exact last logits at1/7/8/16 rows, partial/cancel/full cache settlement, retained snapshots and continuation with/without capacity buffers.\n", .{});
}

pub fn check(m: *flash.Model) !void {
    try checkShortPrefill(m);
    try checkCompiledMoE(m, "model.layers.0.mlp");
    try checkProjectionStacks(m);
    var states: [9]State = undefined;
    var reference: [9]State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized], reference[0..initialized]) |*state, *ref| {
        state.deinit();
        ref.deinit();
    };
    for (&states, &reference) |*state, *ref| {
        state.* = try State.init(m);
        ref.* = State.init(m) catch |err| {
            state.deinit();
            return err;
        };
        initialized += 1;
    }
    for ([_]usize{ 2, 8, 9, 9, 4, 8, 9, 4 }, 0..) |n, trial| {
        var expected = mx.Scope{};
        defer expected.deinit();
        var tokens: [9][16]i32 = undefined;
        var parents: [9][16]i32 = undefined;
        var paths: [9][]const i32 = undefined;
        var streams: [9]Stream = undefined;
        var logits: [9]A = undefined;
        var hidden: [9]A = undefined;
        var positions: [9]i32 = undefined;
        var committed: [9][48][2]?*anyopaque = undefined;
        const prefix = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
        for (0..n) |i| {
            const rows: usize = switch (trial) {
                0 => if (i == 0) 3 else 2,
                1 => 2,
                2 => if (i < 7) 2 else 1,
                4, 5 => 8,
                6 => if (i < 7) 8 else 4,
                7 => 16,
                else => 1,
            };
            const keep: usize = if (trial == 0 and i == 0) 1 else if (trial == 1 and i % 3 == 0) 0 else if (trial == 2 and i == 4) 1 else if (trial == 6 and i == 3) 3 else rows;
            for (0..rows) |j| {
                tokens[i][j] = @intCast(1000 + trial * 97 + i * 7 + j);
                parents[i][j] = @as(i32, @intCast(j)) - 1;
            }
            paths[i] = prefix[0..keep];
            streams[i] = .{ .state = &states[i], .tokens = tokens[i][0..rows], .parents = parents[i][0..rows] };
            positions[i] = states[i].position;
            for (states[i].cache, &committed[i]) |cache, *handles| handles.* = .{ cache.a.ctx, cache.b.ctx };
            reference[i].swap(m);
            defer reference[i].swap(m);
            var pass = try m.forward(tokens[i][0..rows]);
            defer pass.deinit();
            logits[i] = try expected.own(try mx.retain(pass.logits));
            hidden[i] = try expected.own(try mx.retain(pass.hidden));
            if (keep != 0) try m.commit(&pass, keep);
        }
        var shared = try forward(m, streams[0..n]);
        defer shared.deinit();
        var staged: [9][48][2]?*anyopaque = undefined;
        for (0..n) |i| {
            errdefer std.debug.print("Flash shared mismatch: trial {d}, stream {d}\n", .{ trial, i });
            const view = try shared.view(i);
            try equalArray(view.hidden, hidden[i]);
            try equalArray(view.logits, logits[i]);
            try std.testing.expectEqual(positions[i], states[i].position);
            for (states[i].cache, committed[i], shared.entries[i].staged, &staged[i]) |cache, handles, candidate, *prepared| {
                try std.testing.expectEqual(handles[0], cache.a.ctx);
                try std.testing.expectEqual(handles[1], cache.b.ctx);
                try std.testing.expectEqual(positions[i] + @as(i32, @intCast(view.count)), candidate.offset);
                prepared.* = .{ candidate.a.ctx, candidate.b.ctx };
            }
        }
        try shared.commit(paths[0..n]);
        for (paths[0..n], 0..) |path, i| {
            if (path.len == streams[i].tokens.len) for (states[i].cache, staged[i], shared.entries[i].staged) |cache, handles, candidate| {
                try std.testing.expectEqual(handles[0], cache.a.ctx);
                try std.testing.expectEqual(handles[1], cache.b.ctx);
                try std.testing.expect(candidate.a.ctx == null and candidate.b.ctx == null);
            };
            if (path.len == 0) for (states[i].cache, committed[i]) |cache, handles| {
                try std.testing.expectEqual(handles[0], cache.a.ctx);
                try std.testing.expectEqual(handles[1], cache.b.ctx);
            };
        }
        for (states[0..n], reference[0..n]) |*state, *ref| try equalState(state, ref);
        try std.testing.expectError(error.InvalidRoundStage, shared.commit(paths[0..n]));
        try std.testing.expectError(error.InvalidRoundStage, shared.view(0));
    }
    {
        const streams = [_]Stream{
            .{ .state = &states[0], .tokens = &.{ 1800, 1801 }, .parents = &.{ -1, 0 } },
            .{ .state = &states[1], .tokens = &.{1802}, .parents = &.{-1} },
        };
        for ([_]bool{ false, true }) |evaluate| {
            var cancelled = try forward(m, &streams);
            errdefer cancelled.deinit();
            if (evaluate) try mx.eval(cancelled.logits);
            cancelled.deinit();
            for (states[0..2], reference[0..2]) |*state, *ref| try equalState(state, ref);
        }
        var expected = mx.Scope{};
        defer expected.deinit();
        var logits: [2]A = undefined;
        for (streams, 0..) |stream, i| {
            reference[i].swap(m);
            defer reference[i].swap(m);
            var isolated = try m.forward(stream.tokens);
            defer isolated.deinit();
            logits[i] = try expected.own(try mx.retain(isolated.logits));
            try m.commit(&isolated, 1);
        }
        var resumed = try forward(m, &streams);
        defer resumed.deinit();
        for (0..streams.len) |i| try equalArray((try resumed.view(i)).logits, logits[i]);
        try resumed.commit(&.{ &.{0}, &.{0} });
        for (states[0..2], reference[0..2]) |*state, *ref| try equalState(state, ref);
    }
    std.debug.print("PASS: Flash shared 2/4/8/9-stream rounds through64 rows: exact logits, hidden states, caches, partial/zero commits and cancellation.\n", .{});
}

test "Flash shared rounds require distinct states and bounded consecutive windows" {
    var m: flash.Model = undefined;
    m.round_owner = .{};
    var state = try State.init(&m);
    defer state.deinit();
    const stream = Stream{ .state = &state, .tokens = &.{123}, .parents = &.{-1} };
    try std.testing.expectError(error.InvalidStreams, validate(&m, &.{}));
    try std.testing.expectError(error.DuplicateStream, validate(&m, &.{ stream, stream }));
    try std.testing.expectError(error.InvalidTree, validate(&m, &.{.{ .state = &state, .tokens = &.{ 123, 124, 125 }, .parents = &.{ -1, 0, 0 } }}));
    try std.testing.expectError(error.InvalidToken, validate(&m, &.{.{ .state = &state, .tokens = &.{248320}, .parents = &.{-1} }}));
    try std.testing.expectError(error.InvalidStreams, validate(&m, &.{.{ .state = &state, .tokens = &.{}, .parents = &.{} }}));
    const ids: [17]i32 = @splat(123);
    const parents: [17]i32 = @splat(-1);
    try std.testing.expectError(error.InvalidStreams, validate(&m, &.{.{ .state = &state, .tokens = &ids, .parents = &parents }}));
    {
        var extra: [5]State = undefined;
        var made: usize = 0;
        defer for (extra[0..made]) |*entry| entry.deinit();
        var windows: [5]Stream = undefined;
        const row_ids: [16]i32 = @splat(123);
        const chain = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
        for (&extra, &windows) |*entry, *window| {
            entry.* = try State.init(&m);
            made += 1;
            window.* = .{ .state = entry, .tokens = &row_ids, .parents = &chain };
        }
        try std.testing.expectEqual(@as(usize, 64), try validate(&m, windows[0..4]));
        windows[4].tokens = row_ids[0..1];
        windows[4].parents = chain[0..1];
        try std.testing.expectError(error.InvalidStreams, validate(&m, &windows));
    }
    state.borrowed = true;
    try std.testing.expectError(error.RequestRoundActive, validate(&m, &.{stream}));
    state.borrowed = false;
    state.position = -1;
    try std.testing.expectError(error.InvalidStreams, validate(&m, &.{stream}));
    state.position = std.math.maxInt(i32);
    try std.testing.expectError(error.InvalidStreams, validate(&m, &.{stream}));
    state.position = 1;
    try std.testing.expectError(error.InvalidCacheState, validate(&m, &.{stream}));
    state.position = 0;
    const ticket = try m.round_owner.begin();
    defer ticket.release();
    try std.testing.expectError(error.ModelRoundActive, validate(&m, &.{stream}));
}

test "Flash shared commits accept only complete or partial prefixes" {
    try validatePath(4, &.{});
    try validatePath(4, &.{0});
    try validatePath(4, &.{ 0, 1, 2, 3 });
    try std.testing.expectError(error.InvalidCommit, validatePath(4, &.{1}));
    try std.testing.expectError(error.InvalidCommit, validatePath(4, &.{ 0, 2 }));
    try std.testing.expectError(error.InvalidCommit, validatePath(4, &.{ 0, 1, 1 }));
    try std.testing.expectError(error.InvalidCommit, validatePath(4, &.{ 0, 1, 2, 3, 4 }));
}
