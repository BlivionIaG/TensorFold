const std = @import("std");
const mx = @import("mlx.zig");
const model = @import("nemotron.zig");
const src = @import("kernel_sources.zig");
const round = @import("decode_round.zig");
const A = mx.Array;
pub const State = @import("request_state.zig").State(model.Model);
pub const Stream = struct { state: *State, tokens: []const i32, parents: []const i32 };
pub const ArrayStream = struct { state: *State, count: usize, parents: []const i32 };
const Entry = struct { state: *State, first: i32, pass: model.Pass };
pub const max_shared_rows = model.Model.max_shared_rows;
const max_streams = model.Model.max_shared_streams;

// The model and states must remain at stable addresses until deinit; do not copy.
pub const Pass = struct {
    scope: mx.Scope = .{},
    model: *model.Model,
    ticket: round.Ticket,
    entries: []Entry,
    logits: A = mx.empty,
    hidden: A = mx.empty,
    count: usize,
    segments: A = mx.empty,
    starts: A = mx.empty,
    slots: A = mx.empty,
    dimensions: A = mx.empty,

    pub fn view(p: *Pass, index: usize) !*const model.Pass {
        try p.ticket.expect(.forwarded);
        if (index >= p.entries.len) return error.InvalidStreams;
        return &p.entries[index].pass;
    }

    pub fn deinit(p: *Pass) void {
        if (!p.ticket.active()) return;
        for (p.entries) |*entry| {
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
            try validatePath(path, entry.pass.count);
        }
        const next = try mx.allocator.alloc([52]model.Cache, p.entries.len);
        defer mx.allocator.free(next);
        @memset(next, @splat(.{}));
        defer for (next) |*cache| for (cache) |*c| c.deinit();
        var arrays: std.ArrayList(A) = .empty;
        defer arrays.deinit(mx.allocator);
        var recurrent_ready: [52]bool = @splat(false);
        for (p.entries, paths, next) |*entry, path, *cache| {
            if (path.len == 0) continue;
            try model.Model.observeBuffers(&entry.pass);
            cache.* = try p.model.prepareCommit(&entry.pass, entry.state.cache, entry.pass.start, path.len);
            for (p.model.kinds, cache, 0..) |kind, c, layer| if (kind != 'E') {
                if (c.recurrent.ssm.ctx != null) {
                    // Every stream keeps a row of the same forward's recurrent pool.
                    if (!recurrent_ready[layer]) try arrays.appendSlice(mx.allocator, &.{ c.recurrent.conv, c.recurrent.ssm });
                    recurrent_ready[layer] = true;
                } else try arrays.appendSlice(mx.allocator, &.{ c.a, c.b });
            };
        }
        if (arrays.items.len > 0) try mx.evalMany(arrays.items, true);
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

fn validatePath(path: []const i32, count: usize) !void {
    if (path.len > count) return error.InvalidCommit;
    for (path, 0..) |row, index| if (row != @as(i32, @intCast(index))) return error.InvalidCommit;
}

fn validate(m: *const model.Model, streams: []const Stream) !usize {
    if (streams.len > max_streams) return error.InvalidStreams;
    var metadata: [max_streams]ArrayStream = undefined;
    for (streams, 0..) |stream, i| metadata[i] = .{ .state = stream.state, .count = stream.tokens.len, .parents = stream.parents };
    const rows = try validateArrays(m, metadata[0..streams.len]);
    for (streams) |stream| for (stream.tokens) |token| {
        if (token < 0 or token >= model.Model.vocab) return error.InvalidToken;
    };
    return rows;
}

fn validateArrays(m: *const model.Model, streams: []const ArrayStream) !usize {
    if (m.round_owner.stage != .idle) return error.ModelRoundActive;
    if (streams.len == 0 or streams.len > max_streams) return error.InvalidStreams;
    var rows: usize = 0;
    for (streams, 0..) |stream, index| {
        if (stream.state.borrowed) return error.RequestRoundActive;
        if (stream.state.cache.len != 52 or stream.count == 0 or stream.count > 16 or stream.count != stream.parents.len or stream.count > max_shared_rows - rows) return error.InvalidStreams;
        for (streams[0..index]) |other| if (other.state == stream.state) return error.DuplicateStream;
        for (stream.parents, 0..) |parent, row| if (parent != @as(i32, @intCast(row)) - 1) return error.InvalidTree;
        if (stream.state.position < 0) return error.InvalidStreams;
        _ = std.math.add(i32, stream.state.position, @intCast(stream.count)) catch return error.InvalidStreams;
        for (m.kinds, stream.state.cache) |kind, cache| {
            if (kind == 'E') continue;
            if (kind != 'M' and kind != '*') return error.InvalidLayerKind;
            if (cache.a.ctx == null or cache.b.ctx == null) {
                if (cache.a.ctx != null or cache.b.ctx != null or stream.state.position != 0) return error.InvalidCacheState;
                continue;
            }
            const attention_layer = kind == '*';
            const a_shape: []const i32 = if (attention_layer) &.{ 1, 2, stream.state.position, 128 } else &.{ 1, 3, 6144 };
            const b_shape: []const i32 = if (attention_layer) a_shape else &.{ 1, 64, 64, 128 };
            if (!std.mem.eql(i32, mx.shape(cache.a), a_shape) or !std.mem.eql(i32, mx.shape(cache.b), b_shape) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != (if (attention_layer) mx.bf16 else mx.f32t)) return error.InvalidCacheState;
            const held = cache.recurrent;
            if (held.conv.ctx != null or held.ssm.ctx != null) {
                if (attention_layer or held.conv.ctx == null or held.ssm.ctx == null or held.owner == 0 or held.epoch == 0) return error.InvalidCacheState;
                const shape = mx.shape(held.conv);
                if (shape.len != 3 or shape[0] < 1 or held.row < 0 or held.row >= shape[0] or shape[1] != 3 or shape[2] != 6144 or !std.mem.eql(i32, mx.shape(held.ssm), &.{ shape[0], 64, 64, 128 }) or mx.dtype(held.conv) != mx.bf16 or mx.dtype(held.ssm) != mx.f32t) return error.InvalidCacheState;
            }
        }
        rows += stream.count;
    }
    return rows;
}

pub fn forward(m: *model.Model, streams: []const Stream) !Pass {
    const rows = try validate(m, streams);
    var metadata: [max_streams]ArrayStream = undefined;
    var tokens: [max_shared_rows]i32 = undefined;
    var first: usize = 0;
    for (streams, 0..) |stream, i| {
        metadata[i] = .{ .state = stream.state, .count = stream.tokens.len, .parents = stream.parents };
        @memcpy(tokens[first..][0..stream.tokens.len], stream.tokens);
        first += stream.tokens.len;
    }
    var scope = mx.Scope{};
    defer scope.deinit();
    return forwardInput(m, metadata[0..streams.len], rows, try scope.ints(tokens[0..rows]));
}

// Internal GPU-token entry point; proposal IDs come from the validated sampler.
pub fn forwardArray(m: *model.Model, streams: []const ArrayStream, tokens: A) !Pass {
    const rows = try validateArrays(m, streams);
    if (tokens.ctx == null or (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32)) return error.InvalidToken;
    if (!std.mem.eql(i32, mx.shape(tokens), &.{@intCast(rows)})) return error.InvalidStreams;
    return forwardInput(m, streams, rows, tokens);
}

fn forwardInput(m: *model.Model, streams: []const ArrayStream, rows: usize, tokens: A) !Pass {
    const entries = try mx.allocator.alloc(Entry, streams.len);
    errdefer mx.allocator.free(entries);
    var segments: [max_shared_rows]i32 = @splat(0);
    var starts: [max_shared_rows]i32 = @splat(0);
    var slots: [max_shared_rows]i32 = @splat(0);
    var first: usize = 0;
    for (streams, entries, 0..) |stream, *entry, index| {
        entry.* = .{ .state = stream.state, .first = @intCast(first), .pass = .{ .start = stream.state.position, .count = stream.count } };
        @memset(segments[first..][0..stream.count], @intCast(index));
        starts[index] = @intCast(first);
        slots[index] = @intCast(index);
        first += stream.count;
    }
    const ticket = try m.round_owner.begin();
    for (streams) |stream| stream.state.borrowed = true;
    var p = Pass{ .model = m, .ticket = ticket, .entries = entries, .count = rows };
    errdefer {
        for (entries) |*entry| {
            entry.pass.deinit();
            entry.state.borrowed = false;
        }
        p.scope.deinit();
        ticket.release();
    }
    const s = &p.scope;
    p.segments = try s.ints(segments[0..@max(rows, 8)]);
    p.starts = try s.ints(starts[0..@max(streams.len, 8)]);
    p.slots = try s.ints(slots[0..@max(streams.len, 8)]);
    p.dimensions = try s.ints(&.{ @intCast(rows), 0, 0, 0, 0, 0, 0, 0 });
    var h = try m.weights.embedArray(s, "backbone.embeddings", tokens);
    var x = try m.norm(s, h, "backbone.layers.0.norm");
    var sums: ?A = null;
    var carried: [3]A = @splat(mx.empty);
    defer for (carried) |value| mx.free(value);
    var base_buf: [256]u8 = undefined;
    var next_buf: [256]u8 = undefined;
    for (m.kinds, 0..) |kind, layer| {
        var layer_scope = mx.Scope{};
        defer layer_scope.deinit();
        const base = try std.fmt.bufPrint(&base_buf, "backbone.layers.{d}.mixer", .{layer});
        const next = if (layer + 1 == 52) "backbone.norm_f" else try std.fmt.bufPrint(&next_buf, "backbone.layers.{d}.norm", .{layer + 1});
        const nw = try m.weights.field(next, "weight");
        const both = switch (kind) {
            'M' => if (entries.len == 1) blk: {
                const out = try m.blockSums(&layer_scope, layer, x, h, entries[0].state.cache[layer], sums);
                const conv = try s.own(try mx.retain(out[2]));
                const ssm = try s.own(try mx.retain(out[3]));
                entries[0].pass.records[layer] = .{ .a = conv, .b = ssm, .recurrent = .{ .conv = conv, .ssm = ssm, .owner = @intFromPtr(ticket.owner), .epoch = ticket.epoch, .row = 0 } };
                break :blk out;
            } else try mamba(&p, &layer_scope, layer, x, h, sums),
            '*' => try m.addNormSums(s, h, try attention(&p, base, layer, x, sums), nw),
            'E' => try m.blockSums(&layer_scope, layer, x, h, .{}, sums),
            else => return error.InvalidLayerKind,
        };
        try mx.replace(&carried[0], both[0]);
        try mx.replace(&carried[1], both[1]);
        if (both[4].ctx != null) try mx.replace(&carried[2], both[4]);
        h = carried[0];
        x = carried[1];
        sums = if (both[4].ctx != null) carried[2] else null;
        if ((layer + 1) % 8 == 0) try mx.evalMany(&.{x}, true);
    }
    p.hidden = try s.own(try mx.retain(x));
    p.logits = try m.headSums(s, x, sums);
    for (entries) |*entry| {
        const end = entry.first + @as(i32, @intCast(entry.pass.count));
        entry.pass.logits = try s.slice(p.logits, 0, entry.first, end);
        entry.pass.hidden = try s.slice(p.hidden, 0, entry.first, end);
    }
    try ticket.advance(.bound, .forwarded);
    return p;
}

fn mamba(p: *Pass, s: *mx.Scope, layer: usize, x: A, h: A, sums: ?A) ![5]A {
    const m = p.model;
    var refs: [max_shared_rows]model.RecurrentRows = undefined;
    for (p.entries, 0..) |entry, index| refs[index] = entry.state.cache[layer].recurrent;
    var conv_in: A = undefined;
    var ssm_in: A = undefined;
    var slots = p.slots;
    if (commonPool(refs[0..p.entries.len])) |pool| {
        conv_in = pool.conv;
        ssm_in = pool.ssm;
        slots = try s.ints(pool.slots[0..@max(p.entries.len, 8)]);
    } else {
        var conv_states: [max_shared_rows]A = undefined;
        var ssm_states: [max_shared_rows]A = undefined;
        for (p.entries, 0..) |entry, index| {
            const cache = entry.state.cache[layer];
            conv_states[index] = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 1, 3, 6144 }, mx.bf16);
            ssm_states[index] = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 1, 64, 64, 128 }, mx.f32t);
        }
        conv_in = try s.cat(conv_states[0..p.entries.len], 0);
        ssm_in = try s.cat(ssm_states[0..p.entries.len], 0);
    }
    const out = try m.blockSharedSums(s, layer, x, h, .{ .conv = conv_in, .ssm = ssm_in, .segments = p.segments, .starts = p.starts, .slots = slots, .dimensions = p.dimensions }, sums);
    const conv = try p.scope.own(try mx.retain(out[2]));
    const ssm = try p.scope.own(try mx.retain(out[3]));
    for (p.entries) |*entry| {
        entry.pass.records[layer] = .{ .recurrent = .{ .conv = conv, .ssm = ssm, .owner = @intFromPtr(p.ticket.owner), .epoch = p.ticket.epoch, .row = entry.first } };
    }
    return out;
}

const Pool = struct { conv: A, ssm: A, slots: [max_shared_rows]i32 = @splat(0) };

fn commonPool(refs: []const model.RecurrentRows) ?Pool {
    if (refs.len == 0 or refs.len > max_shared_rows) return null;
    const first = refs[0];
    if (first.conv.ctx == null or first.ssm.ctx == null or first.owner == 0 or first.epoch == 0) return null;
    var out = Pool{ .conv = first.conv, .ssm = first.ssm };
    for (refs, 0..) |ref, index| {
        if (ref.conv.ctx == null or ref.ssm.ctx == null or ref.owner != first.owner or ref.epoch != first.epoch or ref.row < 0) return null;
        out.slots[index] = ref.row;
    }
    return out;
}

fn attention(p: *Pass, base: []const u8, layer: usize, x: A, sums: ?A) !A {
    const m = p.model;
    const s = &p.scope;
    const projected = try m.qkv(s, base, x, sums);
    const q = projected[0];
    const keys = projected[1];
    const values = projected[2];
    var outputs: [max_shared_rows]A = undefined;
    for (p.entries, 0..) |*entry, index| {
        const end = entry.first + @as(i32, @intCast(entry.pass.count));
        outputs[index] = try m.attend(s, try s.slice(q, 2, entry.first, end), try s.slice(keys, 2, entry.first, end), try s.slice(values, 2, entry.first, end), &entry.state.cache[layer], &entry.pass.records[layer]);
    }
    return m.project(s, base, "o_proj", try s.cat(outputs[0..p.entries.len], 0));
}

fn equalCache(expected: State, actual: State) !void {
    try std.testing.expectEqual(expected.position, actual.position);
    for (expected.cache, actual.cache, 0..) |before, after, layer| {
        var scope = mx.Scope{};
        defer scope.deinit();
        errdefer std.debug.print("Nemotron shared cache mismatch at layer {d}\n", .{layer});
        if (before.a.ctx == null) {
            try std.testing.expect(after.a.ctx == null and after.b.ctx == null);
        } else {
            try @import("variant_checks.zig").equalBits(&scope, before.a, after.a);
            try @import("variant_checks.zig").equalBits(&scope, before.b, after.b);
        }
    }
}

fn expectPool(m: *const model.Model, states: []const State, slots: ?[]const i32) !void {
    for (m.kinds, 0..) |kind, layer| if (kind == 'M') {
        var refs: [max_shared_rows]model.RecurrentRows = undefined;
        for (states, 0..) |state, index| {
            const cache = state.cache[layer];
            refs[index] = cache.recurrent;
            if (cache.recurrent.ssm.ctx != null) {
                try std.testing.expectEqual(mx.c.mlx_array_nbytes(cache.recurrent.conv) + mx.c.mlx_array_nbytes(cache.recurrent.ssm), cache.nbytes());
            }
        }
        const pool = commonPool(refs[0..states.len]);
        if (slots) |expected| {
            try std.testing.expect(pool != null);
            try std.testing.expectEqualSlices(i32, expected, pool.?.slots[0..states.len]);
        } else try std.testing.expect(pool == null);
    };
}

fn exercise(m: *model.Model, states: []State, counts: []const usize, keeps: []const usize) !void {
    const references = try mx.allocator.alloc(State, states.len);
    defer mx.allocator.free(references);
    var cloned: usize = 0;
    defer for (references[0..cloned]) |*state| state.deinit();
    for (states, references) |*state, *reference| {
        reference.* = try state.clone();
        cloned += 1;
    }
    const Expected = struct { hidden: A = mx.empty, logits: A = mx.empty, start: i32 = 0 };
    const expected = try mx.allocator.alloc(Expected, states.len);
    defer mx.allocator.free(expected);
    @memset(expected, .{});
    defer for (expected) |value| {
        mx.free(value.hidden);
        mx.free(value.logits);
    };
    var ids: [max_shared_rows]i32 = undefined;
    const parents = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    const committed = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    var streams: [max_shared_rows]Stream = undefined;
    var paths: [max_shared_rows][]const i32 = undefined;
    var first: usize = 0;
    try checkProgress("isolated", counts);
    for (states, references, counts, keeps, expected, 0..) |*state, *reference, count, keep, *value, index| {
        for (ids[first..][0..count], 0..) |*id, row| id.* = @intCast(103 + row + index * 19);
        streams[index] = .{ .state = state, .tokens = ids[first..][0..count], .parents = parents[0..count] };
        paths[index] = committed[0..keep];
        first += count;
        reference.swap(m);
        defer reference.swap(m);
        var single = try m.forward(streams[index].tokens);
        defer single.deinit();
        value.hidden = try mx.retain(single.hidden);
        value.logits = try mx.retain(single.logits);
        value.start = single.start;
        if (keep > 0) try m.commit(&single, keep);
    }
    try checkProgress("shared", counts);
    var pass = try forward(m, streams[0..states.len]);
    defer pass.deinit();
    try mx.eval(pass.logits);
    try std.testing.expectEqual(first, pass.count);
    try std.testing.expectEqual(@as(i32, 0), m.position);
    try std.testing.expectError(error.ModelRoundActive, forward(m, streams[0..states.len]));
    try std.testing.expectError(error.RequestRoundActive, states[0].clone());
    for (expected, 0..) |single, index| {
        var scope = mx.Scope{};
        defer scope.deinit();
        errdefer std.debug.print("Nemotron shared forward mismatch at stream {d}, start {d}, rows {d}\n", .{ index, single.start, counts[index] });
        const view = try pass.view(index);
        try @import("variant_checks.zig").equalBits(&scope, single.hidden, view.hidden);
        try @import("variant_checks.zig").equalBits(&scope, single.logits, view.logits);
    }
    try pass.commit(paths[0..states.len]);
    try std.testing.expectError(error.InvalidRoundStage, pass.commit(paths[0..states.len]));
    pass.deinit();
    pass.deinit();
    try std.testing.expectError(error.StaleRound, pass.view(0));
    for (references, states) |reference, state| try equalCache(reference, state);
    try checkProgress("settled", counts);
}

fn checkArrayForward(m: *model.Model, seeds: []const State) !void {
    const neural = @import("neural_draft.zig");
    const Proposal = @import("drafter.zig").Proposal;
    const sampling = @import("sampling.zig");
    const parents = [_]i32{ -1, 0, 1, 2 };
    const rows = [_]i32{ 0, 1, 2, 3 };
    for ([_][2]usize{ .{ 3, 2 }, .{ 1, 0 }, .{ 0, 0 } }) |keeps| {
        var scope = mx.Scope{};
        defer scope.deinit();
        var states: [2]State = undefined;
        var references: [2]State = undefined;
        var made: usize = 0;
        var copied: usize = 0;
        defer for (states[0..made]) |*state| state.deinit();
        defer for (references[0..copied]) |*state| state.deinit();
        var drafts: [2]neural.Stream(model.Model) = undefined;
        for (&states, &references, &drafts, 0..) |*state, *reference, *draft, i| {
            state.* = try seeds[i].clone();
            made += 1;
            if (state.draft_hidden.ctx == null) state.draft_hidden = try mx.retain(try scope.zeros(&.{ 1, 2688 }, mx.bf16));
            reference.* = try state.clone();
            copied += 1;
            draft.* = .{ .state = state, .first = @intCast(8000 + 19 * i), .budget = 3 - i, .settings = .{ .metal = true, .temperature = if (i == 0) 0 else 0.7, .seed = 0x9876543200000000 + i, .top_k = 12, .top_p = 0.8, .min_p = 0.02 } };
        }
        var expected_proposals: [2]Proposal = undefined;
        try neural.proposeStreams(m, &drafts, &expected_proposals);
        var pending = try neural.proposeStreamsLazy(m, &drafts);
        defer pending.deinit();
        var parts: [2]A = undefined;
        var streams: [2]ArrayStream = undefined;
        var host_streams: [2]Stream = undefined;
        var tokens: [2][4]i32 = undefined;
        var positions: [5]i32 = undefined;
        var settings: [5]sampling.Sampling = undefined;
        var paths: [2][]const i32 = undefined;
        var at: usize = 0;
        for (&streams, &host_streams, &parts, &tokens, &paths, keeps, 0..) |*stream, *host, *part, *ids, *path, keep, i| {
            const count = 3 - i; // Verify a granted prefix shorter than the proposed chain.
            ids[0] = drafts[i].first;
            @memcpy(ids[1..count], expected_proposals[i].tokens[0 .. count - 1]);
            stream.* = .{ .state = &states[i], .count = count, .parents = parents[0..count] };
            host.* = .{ .state = &references[i], .tokens = ids[0..count], .parents = parents[0..count] };
            part.* = try scope.cat(&.{ try scope.cast(try scope.ints(ids[0..1]), mx.c.MLX_UINT32), try scope.slice(pending.tokens[i], 0, 0, @intCast(count - 1)) }, 0);
            path.* = rows[0..keep];
            for (0..count) |j| positions[at + j] = states[i].position + @as(i32, @intCast(j)) + 1;
            @memset(settings[at..][0..count], drafts[i].settings);
            at += count;
        }
        const input = try scope.cat(&parts, 0);
        try std.testing.expectError(error.InvalidStreams, forwardArray(m, &streams, try scope.reshape(input, &.{ 1, 5 })));
        try std.testing.expectError(error.InvalidToken, forwardArray(m, &streams, try scope.cast(input, mx.f32t)));
        {
            const allocator = mx.allocator;
            var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
            mx.allocator = failing.allocator();
            defer mx.allocator = allocator;
            try std.testing.expectError(error.OutOfMemory, forwardArray(m, &streams, input));
        }
        for (states) |state| try std.testing.expect(!state.borrowed);
        try std.testing.expectEqual(round.Stage.idle, m.round_owner.stage);
        var actual = try forwardArray(m, &streams, input);
        defer actual.deinit();
        const picked = try @import("gpu_sampling.zig").sampleRows(&m.kernels, &scope, actual.logits, &positions, &settings, null);
        var proposals: [8]A = undefined;
        const ready = pending.arrays(&proposals);
        var evaluate: [9]A = undefined;
        evaluate[0] = picked;
        @memcpy(evaluate[1..][0..ready.len], ready);
        try mx.evalMany(evaluate[0 .. ready.len + 1], false);
        var actual_proposals: [2]Proposal = undefined;
        try pending.read(&actual_proposals);
        for (actual_proposals, expected_proposals) |got, want| try std.testing.expectEqualSlices(i32, want.tokens[0..want.len], got.tokens[0..got.len]);
        const actual_logits = try scope.own(try mx.retain(actual.logits));
        const actual_hidden = try scope.own(try mx.retain(actual.hidden));
        try actual.commit(&paths);
        actual.deinit();
        pending.deinit();
        var expected = try forward(m, &host_streams);
        defer expected.deinit();
        try @import("variant_checks.zig").equalBits(&scope, expected.logits, actual_logits);
        try @import("variant_checks.zig").equalBits(&scope, expected.hidden, actual_hidden);
        const expected_ids = try sampling.streamRowsMapped(&m.kernels, &scope, expected.logits, &positions, &settings, null);
        defer mx.allocator.free(expected_ids);
        for (expected_ids, mx.c.mlx_array_data_uint32(picked)[0..expected_ids.len]) |want, got| try std.testing.expectEqual(want, @as(i32, @intCast(got)));
        try expected.commit(&paths);
        expected.deinit();
        for (states, references) |state, reference| {
            try equalCache(reference, state);
            try std.testing.expectEqual(reference.head_cache.a.ctx == null, state.head_cache.a.ctx == null);
            try std.testing.expectEqual(reference.draft_hidden.ctx == null, state.draft_hidden.ctx == null);
        }
        var cancelled = try neural.proposeStreamsLazy(m, &drafts);
        defer cancelled.deinit();
        var cancel_scope = mx.Scope{};
        defer cancel_scope.deinit();
        for (&parts, drafts, 0..) |*part, draft, i| part.* = try cancel_scope.cat(&.{ try cancel_scope.cast(try cancel_scope.ints(&.{draft.first}), mx.c.MLX_UINT32), try cancel_scope.slice(cancelled.tokens[i], 0, 0, @intCast(streams[i].count - 1)) }, 0);
        var abandoned = try forwardArray(m, &streams, try cancel_scope.cat(&parts, 0));
        abandoned.deinit();
        cancelled.deinit();
        for (states, references) |state, reference| try equalCache(reference, state);
    }
    std.debug.print("Nemotron lazy target: exact host logits/draws/caches after grant trimming, mixed sampling, full/partial/empty commit, cancellation and failed allocation\n", .{});
}

fn checkLookahead(m: *model.Model, seed_state: *const State) !void {
    var actual = try seed_state.clone();
    defer actual.deinit();
    var expected = try seed_state.clone();
    defer expected.deinit();
    var scope = mx.Scope{};
    defer scope.deinit();
    var first_logits: A = undefined;
    var next_logits: A = undefined;
    var token: i32 = undefined;
    {
        actual.swap(m);
        defer actual.swap(m);
        var first = try m.forwardQueued(&.{9001});
        defer first.deinit();
        const sampled = try scope.argmax(first.logits);
        var next = try m.forwardAfter(&first, sampled);
        defer next.deinit();
        try std.testing.expectEqual(first.start + 1, next.start);
        try std.testing.expectEqual(first.start, m.position);
        try mx.evalMany(&.{ sampled, next.logits }, false);
        token = @intCast(mx.c.mlx_array_data_uint32(sampled)[0]);
        first_logits = try scope.own(try mx.retain(first.logits));
        next_logits = try scope.own(try mx.retain(next.logits));
        try m.commit(&first, 1);
        try m.commit(&next, 1);
        try std.testing.expectError(error.InvalidPreview, m.forwardAfter(&first, sampled));
    }
    {
        expected.swap(m);
        defer expected.swap(m);
        var first = try m.forward(&.{9001});
        defer first.deinit();
        try @import("variant_checks.zig").equalBits(&scope, first.logits, first_logits);
        try m.commit(&first, 1);
        var next = try m.forward(&.{token});
        defer next.deinit();
        try @import("variant_checks.zig").equalBits(&scope, next.logits, next_logits);
        try m.commit(&next, 1);
    }
    try equalCache(expected, actual);
    {
        actual.swap(m);
        defer actual.swap(m);
        var first = try m.forwardQueued(&.{9003});
        defer first.deinit();
        const sampled = try scope.argmax(first.logits);
        try std.testing.expectError(error.InvalidSamplingShape, m.forwardAfter(&first, try scope.cast(sampled, mx.i32t)));
        {
            const allocator = mx.allocator;
            var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
            mx.allocator = failing.allocator();
            defer mx.allocator = allocator;
            try std.testing.expectError(error.OutOfMemory, m.forwardAfter(&first, sampled));
        }
        var abandoned = try m.forwardAfter(&first, sampled);
        abandoned.deinit();
    }
    try equalCache(expected, actual);
    {
        actual.swap(m);
        defer actual.swap(m);
        var stale = try m.forwardQueued(&.{9005});
        defer stale.deinit();
        var committed = try m.forward(&.{9007});
        defer committed.deinit();
        try m.commit(&committed, 1);
        try std.testing.expect(stale.staged_ready);
        try std.testing.expectError(error.InvalidPreview, m.forwardAfter(&stale, try scope.argmax(stale.logits)));
    }
    std.debug.print("Nemotron singleton preview: exact sequential logits/cache, stale/shape/failure rejection and unevaluated cancellation\n", .{});
}

pub fn check(m: *model.Model) !void {
    var memory = try @import("memory_runtime.zig").Runtime.init(@import("bonsai.zig").memory_limit);
    defer memory.deinit();
    var previous: usize = 0;
    try mx.check(mx.c.mlx_set_memory_limit(&previous, @intCast(try memory.admissionBudget(std.Options.debug_io, .nemotron))));
    m.reset();
    defer m.reset();
    try m.checkQkv();
    try m.checkBlocks();
    var states: [2]State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized]) |*state| state.deinit();
    for (&states, 0..) |*state, index| {
        state.* = try State.init(m);
        initialized += 1;
        state.swap(m);
        defer state.swap(m);
        const prefix = [_]i32{ 31, 47, 61, 79, 97 };
        var pass = try m.forward(prefix[0 .. 1 + 4 * index]);
        defer pass.deinit();
        try m.commit(&pass, pass.count);
    }
    try checkLookahead(m, &states[0]);
    if (m.mtp) try checkArrayForward(m, &states);
    try expectPool(m, &states, null);
    try exercise(m, &states, &.{ 3, 2 }, &.{ 2, 0 });
    try expectPool(m, &states, null);
    try exercise(m, &states, &.{ 8, 8 }, &.{ 8, 3 });
    try expectPool(m, &states, &.{ 7, 10 });
    try exercise(m, &states, &.{ 1, 3 }, &.{ 1, 3 });
    try expectPool(m, &states, &.{ 0, 3 });
    try exercise(m, &states, &.{ 1, 1 }, &.{ 0, 1 });
    try expectPool(m, &states, null);
    try exercise(m, &states, &.{ 2, 1 }, &.{ 2, 1 });
    try expectPool(m, &states, &.{ 1, 2 });
    {
        var wide: [max_streams]State = undefined;
        var cloned: usize = 0;
        defer for (wide[0..cloned]) |*state| state.deinit();
        for (&wide) |*state| {
            state.* = try states[0].clone();
            cloned += 1;
        }
        const repeated: [max_streams]i32 = @splat(1);
        try expectPool(m, &wide, &repeated);
        const counts: [max_streams]usize = @splat(1);
        try exercise(m, &wide, &counts, &counts);
        var kept: [max_streams]i32 = undefined;
        for (&kept, 0..) |*row, index| row.* = @intCast(index);
        try expectPool(m, &wide, &kept);
        var partial: [max_streams]usize = undefined;
        for (&partial, 0..) |*count, index| count.* = index % 3;
        try exercise(m, &wide, &@as([max_streams]usize, @splat(2)), &partial);
        try exercise(m, &wide, &counts, &counts);
        try expectPool(m, &wide, &kept);
    }
    {
        var wide: [8]State = undefined;
        var cloned: usize = 0;
        defer for (wide[0..cloned]) |*state| state.deinit();
        for (&wide) |*state| {
            state.* = try states[0].clone();
            cloned += 1;
        }
        try exercise(m, &wide, &@as([8]usize, @splat(16)), &.{ 16, 0, 1, 7, 15, 3, 8, 16 });
        try expectPool(m, &wide, null);
        try exercise(m, &wide, &.{ 1, 3, 2, 16, 5, 7, 11, 1 }, &.{ 1, 3, 2, 16, 5, 7, 11, 1 });
        try expectPool(m, &wide, &.{ 0, 3, 5, 21, 26, 33, 44, 45 });
    }
    {
        const kv = @import("kv_buffer.zig");
        const previous_buffers = kv.enabled;
        defer kv.enabled = previous_buffers;
        kv.enabled = false;
        var uncached: [2]State = undefined;
        var made: usize = 0;
        defer for (uncached[0..made]) |*state| state.deinit();
        for (&uncached) |*state| {
            state.* = try State.init(m);
            made += 1;
        }
        try exercise(m, &uncached, &.{ 3, 1 }, &.{ 2, 1 });
        try exercise(m, &uncached, &.{ 1, 2 }, &.{ 1, 2 });
    }
    var before = try states[0].clone();
    defer before.deinit();
    const stream = Stream{ .state = &states[0], .tokens = &.{ 113, 127 }, .parents = &.{ -1, 0 } };
    var abandoned = try forward(m, &.{stream});
    abandoned.deinit();
    try equalCache(before, states[0]);
    var invalid = try forward(m, &.{stream});
    defer invalid.deinit();
    try std.testing.expectError(error.InvalidCommit, invalid.commit(&.{&.{1}}));
    try std.testing.expectError(error.InvalidRoundStage, invalid.commit(&.{&.{0}}));
    invalid.deinit();
    try equalCache(before, states[0]);
    if (m.mtp) try @import("nemotron_head_checks.zig").check(m);
    std.debug.print("Shared Nemotron rounds: exact hidden/logits and caches, 128 shared rows, ragged commits, continuation, both cache modes, cancellation and failed settlement\n", .{});
}

fn checkProgress(phase: []const u8, counts: []const usize) !void {
    var total: usize = 0;
    for (counts) |count| total += count;
    const active = try @import("memory_runtime.zig").activeBytes();
    var cached: usize = 0;
    var peak: usize = 0;
    try mx.check(mx.c.mlx_get_cache_memory(&cached));
    try mx.check(mx.c.mlx_get_peak_memory(&peak));
    std.debug.print("Nemotron shared check {s}, streams={d}, rows={d}: active={d}, cache={d}, peak={d} bytes\n", .{ phase, counts.len, total, active, cached, peak });
}

test "Nemotron shared rounds validate independent chain streams before execution" {
    var m: model.Model = undefined;
    m.round_owner = .{};
    m.kinds = @splat('M');
    var state = try State.init(&m);
    defer state.deinit();
    var other = try State.init(&m);
    defer other.deinit();
    const stream = Stream{ .state = &state, .tokens = &.{123}, .parents = &.{-1} };
    try std.testing.expectEqual(@as(usize, 3), try validate(&m, &.{ stream, .{ .state = &other, .tokens = &.{ 124, 125 }, .parents = &.{ -1, 0 } } }));
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
    state.position = 1;
    try std.testing.expectError(error.InvalidCacheState, forward(&m, &.{stream}));
    state.position = 0;
    try std.testing.expectError(error.InvalidToken, forward(&m, &.{.{ .state = &state, .tokens = &.{model.Model.vocab}, .parents = &.{-1} }}));
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{.{ .state = &state, .tokens = &.{123}, .parents = &.{} }}));
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{.{ .state = &state, .tokens = &.{}, .parents = &.{} }}));
    try std.testing.expectError(error.InvalidTree, forward(&m, &.{.{ .state = &state, .tokens = &.{ 123, 124, 125 }, .parents = &.{ -1, 0, 0 } }}));
    const full_tokens: [max_shared_rows]i32 = @splat(123);
    var full_parents: [max_shared_rows]i32 = undefined;
    for (&full_parents, 0..) |*parent, index| parent.* = @as(i32, @intCast(index)) - 1;
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{ stream, .{ .state = &other, .tokens = &full_tokens, .parents = &full_parents } }));
    const ticket = try m.round_owner.begin();
    defer ticket.release();
    try std.testing.expectError(error.ModelRoundActive, forward(&m, &.{stream}));
}

test "Nemotron shared commits accept cancellation and consecutive prefixes only" {
    try validatePath(&.{}, 3);
    try validatePath(&.{0}, 3);
    try validatePath(&.{ 0, 1, 2 }, 3);
    try std.testing.expectError(error.InvalidCommit, validatePath(&.{1}, 3));
    try std.testing.expectError(error.InvalidCommit, validatePath(&.{ 0, 2 }, 3));
    try std.testing.expectError(error.InvalidCommit, validatePath(&.{ 0, 1, 2, 3 }, 3));
}

test "Nemotron shared rows admit eight full windows and reject excess rows" {
    var m: model.Model = undefined;
    m.round_owner = .{};
    m.kinds = @splat('M');
    var states: [9]State = undefined;
    var made: usize = 0;
    defer for (states[0..made]) |*state| state.deinit();
    const ids: [16]i32 = @splat(123);
    const parents = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14 };
    var streams: [9]Stream = undefined;
    for (&states, &streams) |*state, *stream| {
        state.* = try State.init(&m);
        made += 1;
        stream.* = .{ .state = state, .tokens = &ids, .parents = &parents };
    }
    try std.testing.expectEqual(@as(usize, 128), try validate(&m, streams[0..8]));
    streams[8].tokens = ids[0..1];
    streams[8].parents = parents[0..1];
    try std.testing.expectError(error.InvalidStreams, validate(&m, &streams));
}

test "Nemotron recurrent pools select retained rows and fall back for mixed rounds" {
    const first = model.RecurrentRows{ .conv = .{ .ctx = @ptrFromInt(1) }, .ssm = .{ .ctx = @ptrFromInt(2) }, .owner = 11, .epoch = 7, .row = 2 };
    var second = first;
    second.conv.ctx = @ptrFromInt(3);
    second.ssm.ctx = @ptrFromInt(4);
    second.row = 6;
    const pool = commonPool(&.{ first, second }).?;
    try std.testing.expectEqual(first.conv.ctx, pool.conv.ctx);
    try std.testing.expectEqual(first.ssm.ctx, pool.ssm.ctx);
    try std.testing.expectEqualSlices(i32, &.{ 2, 6 }, pool.slots[0..2]);
    try std.testing.expect(commonPool(&.{}) == null);
    try std.testing.expect(commonPool(&.{ first, .{} }) == null);
    second.epoch += 1;
    try std.testing.expect(commonPool(&.{ first, second }) == null);
    var third = second;
    third.row = 0;
    try std.testing.expectEqualSlices(i32, &.{ 6, 0 }, commonPool(&.{ second, third }).?.slots[0..2]);
    third.owner += 1;
    try std.testing.expect(commonPool(&.{ second, third }) == null);
}
