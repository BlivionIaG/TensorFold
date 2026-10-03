const std = @import("std");
const mx = @import("mlx.zig");
const sampling = @import("sampling.zig");
const qwen = @import("model.zig");
pub const Drafter = @import("drafter.zig").Drafter;
const Proposal = @import("drafter.zig").Proposal;
const max_streams = nemotron.Model.max_shared_streams;

pub const Depth = struct {
    rates: [15]f64 = @import("draft_depth.zig").flash_prior,
    count: usize = 15,
    choices: usize = 0,
    rounds: usize = 0,
    next_depth: ?usize = null,

    pub fn init(comptime M: type) Depth {
        var d = Depth{};
        if (@hasDecl(M, "draft_prior")) {
            d.count = M.draft_prior.len;
            @memcpy(d.rates[0..d.count], M.draft_prior);
        }
        return d;
    }

    pub fn chances(d: Depth, output: []f64) !void {
        try @import("draft_allocation.zig").chainProbabilities(d.rates[0..d.count], output);
    }

    pub fn choose(d: *Depth, costs: *const @import("draft_depth.zig").Adaptive, budget: usize, room: usize) usize {
        var policy = costs.*;
        policy.budget = @min(policy.budget, budget);
        policy.rates = d.rates;
        policy.rate_count = @min(d.count, @max(1, policy.budget));
        policy.choices = d.choices;
        const depth = policy.choose(room);
        d.choices = policy.choices;
        return depth;
    }

    pub fn observe(d: *Depth, proposed: usize, accepted: usize) void {
        for (0..@min(proposed, d.count)) |j| {
            if (accepted < j) break;
            d.rates[j] += 0.15 * ((if (accepted > j) @as(f64, 1) else 0) - d.rates[j]);
        }
    }
};

test "draft acceptance updates only verified depths and compounds reach probability" {
    var d = Depth{};
    const before = d.rates;
    d.observe(4, 1);
    try std.testing.expectEqual(before[0] + 0.15 * (1 - before[0]), d.rates[0]);
    try std.testing.expectEqual(before[1] * 0.85, d.rates[1]);
    try std.testing.expectEqualSlices(f64, before[2..], d.rates[2..]);
    var chances: [3]f64 = undefined;
    try d.chances(&chances);
    try std.testing.expectEqual(d.rates[0] * d.rates[1] * d.rates[2], chances[2]);
}

pub const Options = struct {
    enabled: bool = false,
    directory: ?[]const u8 = null,
    bits: i32 = 8,
    max_draft: usize = 3,
    calibration: ?[]const u8 = null,

    pub fn validate(o: Options) !void {
        if (o.max_draft > 15) return error.InvalidDraftBudget;
        if (o.bits != 0 and o.bits != 4 and o.bits != 8) return error.InvalidDraftBits;
    }
};

pub fn enabled(m: anytype, d: ?*Drafter) bool {
    const M = @TypeOf(m.*);
    return if (M == qwen.Model) d != null else if (@hasField(M, "mtp")) m.mtp else m.has_mtp;
}

fn hiddenRows(scope: *mx.Scope, hidden: mx.Array, rows: []const i32) !mx.Array {
    if (rows.len > 0 and rows[0] >= 0 and rows[rows.len - 1] < mx.dim(hidden, 0)) {
        for (rows, 0..) |row, i| {
            if (@as(i64, row) != @as(i64, rows[0]) + @as(i64, @intCast(i))) break;
        } else return scope.slice(hidden, 0, rows[0], rows[rows.len - 1] + 1);
    }
    return scope.take(hidden, try scope.ints(rows), 0);
}

pub fn absorb(m: anytype, state: anytype, d: ?*Drafter, pass: anytype, tokens: []const i32, rows: []const i32) !void {
    const M = @TypeOf(m.*);
    if (!enabled(m, d)) return;
    if (M == qwen.Model) {
        var offset: usize = 0;
        while (offset < rows.len) {
            const end = @min(offset + 128, rows.len);
            try d.?.absorb(m, pass, rows[offset..end], tokens);
            offset = end;
        }
    } else {
        const on_commit = if (@hasDecl(M, "draftAbsorbsOnCommit")) m.draftAbsorbsOnCommit() else false;
        const hidden = if (@hasDecl(M, "draftHidden")) M.draftHidden(pass) else pass.hidden;
        if (M == @import("nemotron.zig").Model or M == @import("flash.zig").Model) {
            if (rows.len == 0) return;
            if (state.borrowed) return error.RequestRoundActive;
            var pending = try @import("request_state.zig").State(M).init(m);
            defer pending.deinit();
            pending.head_cache = try state.head_cache.clone();
            if (state.draft_hidden.ctx != null) pending.draft_hidden = try mx.retain(state.draft_hidden);
            var offset: usize = 0;
            const chunk_rows = if (M == nemotron.Model) M.max_shared_rows else 16;
            while (offset < rows.len) {
                const end = @min(offset + chunk_rows, rows.len);
                try absorbStreams(m, &.{.{ .state = &pending, .hidden = hidden, .tokens = tokens, .rows = rows[offset..end] }});
                offset = end;
            }
            std.mem.swap(M.DraftCache, &state.head_cache, &pending.head_cache);
            std.mem.swap(mx.Array, &state.draft_hidden, &pending.draft_hidden);
            if (@hasDecl(M, "HeadPrediction")) state.head_prediction.deinit();
            return;
        }
        if (comptime @hasDecl(M, "absorbDraftContext")) {
            if (rows.len == 0) return;
            const selected = try hiddenRows(&pass.scope, hidden, rows);
            const n: i32 = @intCast(rows.len);
            const prefix = try pass.scope.slice(selected, 0, 0, n - 1);
            const previous = state.draft_hidden.ctx != null;
            const context = if (previous) try pass.scope.cat(&.{ state.draft_hidden, prefix }, 0) else prefix;
            const next = try mx.allocator.alloc(i32, rows.len);
            defer mx.allocator.free(next);
            for (rows, next) |row, *token| token.* = tokens[@intCast(row)];
            try m.absorbDraftContext(context, next[if (previous) 0 else 1..]);
            try mx.replace(&state.draft_hidden, try pass.scope.slice(selected, 0, n - 1, n));
            return;
        }
        for (rows) |row| {
            if (state.draft_hidden.ctx != null and !on_commit) {
                if (@hasDecl(M, "DraftCache")) {
                    _ = try m.draftStep(&pass.scope, state.draft_hidden, tokens[@intCast(row)], &state.head_cache);
                } else if (@hasDecl(M, "forwardMtp")) {
                    var head = try m.forwardMtp(state.draft_hidden, tokens[@intCast(row)..][0..1]);
                    defer head.deinit();
                    try m.commitMtp(&head, 1);
                }
            }
            try mx.replace(&state.draft_hidden, try pass.scope.slice(hidden, 0, row, row + 1));
        }
    }
}

pub fn propose(m: anytype, state: anytype, d: ?*Drafter, first: i32, budget: usize, settings: sampling.Sampling) !Proposal {
    const M = @TypeOf(m.*);
    if (budget == 0 or !enabled(m, d)) return .{};
    if (M == qwen.Model) return d.?.propose(m, first, budget, settings);
    var result = Proposal{};
    result.len = @min(15, if (@hasDecl(M, "maxDrafts")) @min(budget, m.maxDrafts()) else budget);
    if (result.len == 0) return result;
    var tokens: [16]i32 = undefined;
    tokens[0] = first;
    if (@hasDecl(M, "DraftCache")) {
        var scope = mx.Scope{};
        defer scope.deinit();
        var cache = try state.head_cache.clone();
        defer cache.deinit();
        var hidden = state.draft_hidden;
        for (0..result.len) |j| {
            hidden = try m.draftStep(&scope, hidden, tokens[j], &cache);
            const ids = try sampling.rowsMapped(&m.kernels, &scope, try m.draftHead(&scope, hidden), &.{m.position + @as(i32, @intCast(j)) + 1}, settings, m.weights.arrays.get("draft_ids"));
            defer mx.allocator.free(ids);
            tokens[j + 1] = ids[0];
        }
    } else try m.propose(state.draft_hidden, first, tokens[0 .. result.len + 1], settings);
    for (0..result.len) |j| {
        result.tokens[j] = tokens[j + 1];
        result.parents[j] = @as(i32, @intCast(j)) - 1;
        result.scores[j] = 0;
        result.probabilities[j] = 1;
    }
    return result;
}

pub fn Stream(comptime M: type) type {
    return struct {
        state: *@import("request_state.zig").State(M),
        first: i32,
        budget: usize,
        settings: sampling.Sampling,
    };
}

pub fn AbsorbStream(comptime M: type) type {
    return struct {
        state: *@import("request_state.zig").State(M),
        hidden: mx.Array,
        tokens: []const i32,
        rows: []const i32,
    };
}

fn cacheArrays(value: anytype, arrays: *std.ArrayList(mx.Array)) !void {
    const T = @TypeOf(value);
    if (T == mx.Array) {
        if (value.ctx != null) try arrays.append(mx.allocator, value);
        return;
    }
    if (T == @import("kv_buffer.zig").Write) return;
    switch (@typeInfo(T)) {
        .@"struct" => inline for (comptime std.meta.fieldNames(T)) |field| try cacheArrays(@field(value, field), arrays),
        .array => for (value) |item| try cacheArrays(item, arrays),
        else => {},
    }
}

pub fn absorbStreams(m: anytype, streams: []const AbsorbStream(@TypeOf(m.*))) !void {
    const M = @TypeOf(m.*);
    const capacity = M.max_shared_streams;
    if (streams.len > capacity) return error.InvalidDraftRows;
    if (!enabled(m, null) or streams.len == 0) return;
    var total: usize = 0;
    for (streams, 0..) |stream, i| {
        if (stream.state.borrowed) return error.RequestRoundActive;
        for (streams[0..i]) |other| if (stream.state == other.state) return error.DuplicateStream;
        if (stream.rows.len == 0) continue;
        if (stream.hidden.ctx == null or mx.shape(stream.hidden).len != 2) return error.InvalidDraftRows;
        for (stream.rows, 0..) |row, j| {
            if (row < 0 or row >= stream.tokens.len or row >= mx.dim(stream.hidden, 0)) return error.InvalidDraftRows;
            if (j > 0 and row <= stream.rows[j - 1]) return error.InvalidDraftRows;
        }
        total = try std.math.add(usize, total, stream.rows.len);
        if (total > M.max_shared_rows) return error.InvalidDraftRows;
    }
    var scope = mx.Scope{};
    defer scope.deinit();
    var caches: [capacity]M.DraftCache = @splat(.{});
    defer for (&caches) |*cache| cache.deinit();
    var last: [capacity]mx.Array = @splat(mx.empty);
    defer for (last) |hidden| mx.free(hidden);
    var pointers: [capacity]*M.DraftCache = undefined;
    var lengths: [capacity]usize = @splat(0);
    var contexts: [capacity]mx.Array = undefined;
    var parts: usize = 0;
    var tokens: [128]i32 = undefined;
    var token_count: usize = 0;
    for (streams, 0..) |stream, i| {
        pointers[i] = &caches[i];
        if (stream.rows.len == 0) continue;
        caches[i] = try stream.state.head_cache.clone();
        const selected = try hiddenRows(&scope, stream.hidden, stream.rows);
        const n: i32 = @intCast(stream.rows.len);
        last[i] = try mx.retain(try scope.slice(selected, 0, n - 1, n));
        const previous = stream.state.draft_hidden.ctx != null;
        const skip: usize = if (previous) 0 else 1;
        lengths[i] = stream.rows.len - skip;
        if (lengths[i] == 0) continue;
        const prefix = try scope.slice(selected, 0, 0, n - 1);
        contexts[parts] = if (previous) try scope.cat(&.{ stream.state.draft_hidden, prefix }, 0) else prefix;
        parts += 1;
        for (stream.rows[skip..]) |row| {
            tokens[token_count] = stream.tokens[@intCast(row)];
            token_count += 1;
        }
    }
    if (token_count > 0) try m.absorbDraftStreams(&scope, try scope.cat(contexts[0..parts], 0), try scope.ints(tokens[0..token_count]), lengths[0..streams.len], pointers[0..streams.len]);
    var arrays: std.ArrayList(mx.Array) = .empty;
    defer arrays.deinit(mx.allocator);
    for (streams, 0..) |stream, i| if (stream.rows.len > 0) {
        try cacheArrays(caches[i], &arrays);
        try arrays.append(mx.allocator, last[i]);
    };
    if (arrays.items.len > 0) try mx.evalMany(arrays.items, true);
    for (streams, 0..) |stream, i| if (stream.rows.len > 0) {
        std.mem.swap(M.DraftCache, &stream.state.head_cache, &caches[i]);
        std.mem.swap(mx.Array, &stream.state.draft_hidden, &last[i]);
        if (@hasDecl(M, "HeadPrediction")) stream.state.head_prediction.deinit();
    };
}

fn equalAbsorbedCache(scope: *mx.Scope, expected: anytype, actual: @TypeOf(expected)) !void {
    const T = @TypeOf(expected);
    if (T == mx.Array) {
        try std.testing.expectEqual(expected.ctx == null, actual.ctx == null);
        if (expected.ctx != null) try @import("variant_checks.zig").equalBits(scope, expected, actual);
        return;
    }
    if (T == @import("kv_buffer.zig").Buffer or T == @import("kv_buffer.zig").Write or T == @import("nemotron.zig").RecurrentRows) return;
    switch (@typeInfo(T)) {
        .@"struct" => inline for (comptime std.meta.fieldNames(T)) |field| try equalAbsorbedCache(scope, @field(expected, field), @field(actual, field)),
        .array => for (expected, actual) |before, after| try equalAbsorbedCache(scope, before, after),
        .int, .float, .bool, .@"enum" => try std.testing.expectEqual(expected, actual),
        else => {},
    }
}

pub fn checkAbsorbStreams(m: anytype, hidden: mx.Array, tokens: []const i32, seeds: anytype) !void {
    const M = @TypeOf(m.*);
    const State = @import("request_state.zig").State(M);
    const lengths = [_]usize{ 3, 0, 1, 2, 4, 1, 3, 2 };
    const rows = [_]i32{ 0, 1, 2, 3 };
    var scope = mx.Scope{};
    defer scope.deinit();
    for ([_][]const i32{ &.{0}, &.{ 0, 1, 2, 3 }, &.{ 1, 2, 3 }, &.{ 0, 2, 3 } }) |selected| {
        try @import("variant_checks.zig").equalBits(&scope, try scope.take(hidden, try scope.ints(selected), 0), try hiddenRows(&scope, hidden, selected));
    }
    var states: [8]State = undefined;
    var references: [8]State = undefined;
    var initialized: usize = 0;
    var cloned: usize = 0;
    defer for (states[0..initialized]) |*state| state.deinit();
    defer for (references[0..cloned]) |*state| state.deinit();
    var streams: [8]AbsorbStream(M) = undefined;
    var offset: usize = 0;
    for (&states, &references, &streams, lengths, 0..) |*state, *reference, *stream, n, i| {
        state.* = try State.init(m);
        initialized += 1;
        if (i != 0) {
            state.head_cache = try seeds[i].clone();
            state.draft_hidden = try mx.retain(try scope.slice(hidden, 0, @intCast(i), @intCast(i + 1)));
        }
        reference.* = try state.clone();
        cloned += 1;
        stream.* = .{ .state = state, .hidden = try scope.slice(hidden, 0, @intCast(offset), @intCast(offset + n)), .tokens = tokens[offset..][0..n], .rows = rows[0..n] };
        offset += n;
        var pass = struct { scope: mx.Scope = .{}, hidden: mx.Array }{ .hidden = stream.hidden };
        defer pass.scope.deinit();
        try absorb(m, reference, null, &pass, stream.tokens, stream.rows);
    }
    const cancelled_hidden = states[1].draft_hidden.ctx;
    const cancelled_cache = states[1].head_cache.a.ctx;
    try std.testing.expectError(error.DuplicateStream, absorbStreams(m, &.{ streams[0], streams[0] }));
    states[2].borrowed = true;
    const rejected = absorbStreams(m, &streams);
    states[2].borrowed = false;
    try std.testing.expectError(error.RequestRoundActive, rejected);
    if (@import("kv_buffer.zig").enabled) {
        var saved = try states[2].head_cache.keys.clone();
        defer saved.deinit();
        states[2].head_cache.keys.deinit();
        states[2].head_cache.keys.current = try mx.retain(states[2].head_cache.a);
        states[2].head_cache.keys.offset = -1;
        defer std.mem.swap(@import("kv_buffer.zig").Buffer, &states[2].head_cache.keys, &saved);
        var cache_handles: [8]?*anyopaque = undefined;
        var hidden_handles: [8]?*anyopaque = undefined;
        for (states, &cache_handles, &hidden_handles) |state, *cache, *last| {
            cache.* = state.head_cache.a.ctx;
            last.* = state.draft_hidden.ctx;
        }
        try std.testing.expectError(error.InvalidCacheOffset, absorbStreams(m, &streams));
        for (states, cache_handles, hidden_handles) |state, cache, last| {
            try std.testing.expectEqual(cache, state.head_cache.a.ctx);
            try std.testing.expectEqual(last, state.draft_hidden.ctx);
        }
    }
    try absorbStreams(m, &streams);
    for (states, references, 0..) |state, reference, i| {
        errdefer std.debug.print("Shared MTP absorption mismatch at stream {d}\n", .{i});
        try equalAbsorbedCache(&scope, reference.head_cache, state.head_cache);
        try @import("variant_checks.zig").equalBits(&scope, reference.draft_hidden, state.draft_hidden);
    }
    try std.testing.expectEqual(cancelled_hidden, states[1].draft_hidden.ctx);
    try std.testing.expectEqual(cancelled_cache, states[1].head_cache.a.ctx);
    try absorbStreams(m, &.{});
    std.debug.print("Shared MTP absorption: exact serial cache/hidden for ragged kept rows, fresh context and cancellation\n", .{});
}

const nemotron = @import("nemotron.zig");
const NemotronState = @import("request_state.zig").State(nemotron.Model);
pub const HeadStream = struct {
    state: *NemotronState,
    anchor: i32,
    count: usize,
    settings: sampling.Sampling,
};

pub const HeadUpdate = struct {
    cache: nemotron.Cache = .{},
    hidden: mx.Array = mx.empty,
    prediction: nemotron.HeadPrediction = .{},

    pub fn deinit(u: *HeadUpdate) void {
        u.cache.deinit();
        mx.free(u.hidden);
        u.prediction.deinit();
        u.* = .{};
    }

    pub fn publish(u: *HeadUpdate, state: *NemotronState) void {
        std.debug.assert(!state.borrowed);
        std.mem.swap(nemotron.Cache, &state.head_cache, &u.cache);
        std.mem.swap(mx.Array, &state.draft_hidden, &u.hidden);
        std.mem.swap(nemotron.HeadPrediction, &state.head_prediction, &u.prediction);
    }
};

pub const HeadVerification = struct {
    scope: mx.Scope = .{},
    model: ?*nemotron.Model = null,
    bases: [max_streams]nemotron.Cache = @splat(.{}),
    records: [max_streams]nemotron.Cache = @splat(.{}),
    lengths: [max_streams]usize = @splat(0),
    offsets: [max_streams]i32 = @splat(0),
    positions: [max_streams]i32 = @splat(0),
    settings: [max_streams]sampling.Sampling = undefined,
    count: usize = 0,
    target_hidden: mx.Array = mx.empty,
    rows: nemotron.HeadRows = .{},

    pub const Keep = struct { slot: usize, count: usize, token: i32, predict: bool };

    pub fn deinit(v: *HeadVerification) void {
        for (&v.bases) |*base| base.deinit();
        v.scope.deinit();
        v.* = .{};
    }

    pub fn init(m: *nemotron.Model, streams: []const HeadStream, hidden: mx.Array, following: mx.Array) !HeadVerification {
        if (streams.len == 0 or streams.len > max_streams) return error.InvalidDraftRows;
        var total: usize = 0;
        for (streams, 0..) |stream, i| {
            if (stream.count == 0 or stream.count > 16 or stream.state.draft_hidden.ctx == null or stream.anchor < 0 or stream.anchor >= nemotron.Model.vocab) return error.InvalidDraftRows;
            if (!stream.settings.metal and stream.settings.temperature != 0) return error.RequiresGPUSampling;
            try stream.settings.validate();
            if (stream.state.position < 0) return error.InvalidSamplingPosition;
            _ = std.math.add(i32, stream.state.position, @as(i32, @intCast(stream.count)) + 1) catch return error.InvalidSamplingPosition;
            for (streams[0..i]) |other| if (stream.state == other.state) return error.DuplicateStream;
            total += stream.count;
        }
        const rows: i32 = @intCast(total);
        if (hidden.ctx == null or following.ctx == null or mx.dtype(hidden) != mx.bf16 or mx.dtype(following) != mx.c.MLX_UINT32 or !std.mem.eql(i32, mx.shape(hidden), &.{ rows, 2688 }) or !std.mem.eql(i32, mx.shape(following), &.{rows})) return error.InvalidDraftRows;
        var v = HeadVerification{ .model = m, .count = streams.len };
        errdefer v.deinit();
        const s = &v.scope;
        v.target_hidden = try s.own(try mx.retain(hidden));
        var pointers: [max_streams]*nemotron.Cache = undefined;
        var catchup: [max_streams]*nemotron.Cache = undefined;
        var contexts: [max_streams]mx.Array = undefined;
        var anchors: [max_streams]i32 = undefined;
        const ones: [max_streams]usize = @splat(1);
        var catching: usize = 0;
        var first: usize = 0;
        for (streams, 0..) |stream, i| {
            v.lengths[i] = stream.count;
            v.offsets[i] = @intCast(first);
            v.positions[i] = stream.state.position;
            v.settings[i] = stream.settings;
            const predicted = stream.state.head_prediction.matches(stream.state.position, stream.anchor, stream.settings);
            v.bases[i] = try (if (predicted) stream.state.head_prediction.cache else stream.state.head_cache).clone();
            pointers[i] = &v.bases[i];
            if (!predicted) {
                catchup[catching] = &v.bases[i];
                contexts[catching] = stream.state.draft_hidden;
                anchors[catching] = stream.anchor;
                catching += 1;
            }
            first += stream.count;
        }
        if (catching > 0) try m.absorbDraftStreams(s, try s.cat(contexts[0..catching], 0), try s.ints(anchors[0..catching]), ones[0..catching], catchup[0..catching]);
        v.rows = try m.draftWindowFront(s, hidden, following, v.lengths[0..streams.len], pointers[0..streams.len], v.records[0..streams.len]);
        return v;
    }

    pub fn prepare(v: *HeadVerification, slot: usize, keep: usize, token: i32, predict: bool) !HeadUpdate {
        var updates = try v.prepareBatch(&.{.{ .slot = slot, .count = keep, .token = token, .predict = predict }});
        defer for (&updates) |*update| update.deinit();
        const result = updates[slot];
        updates[slot] = .{};
        return result;
    }

    pub fn prepareBatch(v: *HeadVerification, keeps: []const Keep) ![max_streams]HeadUpdate {
        const m = v.model orelse return error.InvalidDraftRows;
        if (keeps.len > v.count) return error.InvalidDraftRows;
        for (keeps, 0..) |keep, i| {
            if (keep.slot >= v.count or keep.count == 0 or keep.count > v.lengths[keep.slot] or keep.token < 0 or keep.token >= nemotron.Model.vocab) return error.InvalidDraftRows;
            for (keeps[0..i]) |other| if (other.slot == keep.slot) return error.DuplicateStream;
        }
        var updates: [max_streams]HeadUpdate = @splat(.{});
        errdefer for (&updates) |*update| update.deinit();
        var slots: [max_streams]usize = undefined;
        var selected: [max_streams]i32 = undefined;
        var positions: [max_streams]i32 = undefined;
        var settings: [max_streams]sampling.Sampling = undefined;
        var predicting: usize = 0;
        for (keeps) |keep| {
            updates[keep.slot] = try v.prepareCache(keep.slot, keep.count, keep.token, keep.predict);
            if (keep.predict) {
                slots[predicting] = keep.slot;
                selected[predicting] = v.offsets[keep.slot] + @as(i32, @intCast(keep.count)) - 1;
                positions[predicting] = v.positions[keep.slot] + @as(i32, @intCast(keep.count)) + 1;
                settings[predicting] = v.settings[keep.slot];
                predicting += 1;
            }
        }
        if (predicting > 0) {
            const hidden = try m.draftWindowTail(&v.scope, v.rows, selected[0..predicting]);
            const firsts = try @import("gpu_sampling.zig").sampleRows(&m.kernels, &v.scope, try m.draftHead(&v.scope, hidden), positions[0..predicting], settings[0..predicting], m.weights.arrays.get("draft_ids"));
            for (slots[0..predicting], 0..) |slot, i| {
                const row: i32 = @intCast(i);
                updates[slot].prediction.hidden = try mx.retain(try v.scope.slice(hidden, 0, row, row + 1));
                updates[slot].prediction.first = try mx.retain(try v.scope.slice(firsts, 0, row, row + 1));
            }
        }
        return updates;
    }

    fn prepareCache(v: *HeadVerification, slot: usize, keep: usize, token: i32, predict: bool) !HeadUpdate {
        if (slot >= v.count or keep == 0 or keep > v.lengths[slot]) return error.InvalidDraftRows;
        const s = &v.scope;
        const base = v.bases[slot];
        const record = v.records[slot];
        const rows: i32 = @intCast(keep);
        // Committed head context trails the target by one row; prediction consumes the pending token too.
        const end = mx.dim(base.a, 2) + rows - 1;
        const at = v.offsets[slot] + rows - 1;
        var u = HeadUpdate{};
        errdefer u.deinit();
        u.cache.a = try mx.retain(try s.slice(record.a, 2, 0, end));
        u.cache.b = try mx.retain(try s.slice(record.b, 2, 0, end));
        u.cache.keys = try base.keys.finish(s, record.key_write, rows - 1);
        u.cache.values = try base.values.finish(s, record.value_write, rows - 1);
        u.hidden = try mx.retain(try s.slice(v.target_hidden, 0, at, at + 1));
        if (predict) {
            u.prediction.cache.a = try mx.retain(try s.slice(record.a, 2, 0, end + 1));
            u.prediction.cache.b = try mx.retain(try s.slice(record.b, 2, 0, end + 1));
            u.prediction.cache.keys = try base.keys.finish(s, record.key_write, rows);
            u.prediction.cache.values = try base.values.finish(s, record.value_write, rows);
            u.prediction.position = v.positions[slot] + rows;
            u.prediction.token = token;
            u.prediction.settings = v.settings[slot];
        }
        return u;
    }
};

pub const PendingProposals = struct {
    tokens: [max_streams]mx.Array = @splat(mx.empty),
    lengths: [max_streams]usize = @splat(0),
    count: usize = 0,

    pub fn deinit(p: *PendingProposals) void {
        for (p.tokens) |token| mx.free(token);
        p.* = .{};
    }

    pub fn arrays(p: *const PendingProposals, out: []mx.Array) []const mx.Array {
        std.debug.assert(out.len >= p.count);
        var count: usize = 0;
        for (p.tokens[0..p.count]) |token| if (token.ctx != null) {
            out[count] = token;
            count += 1;
        };
        return out[0..count];
    }

    pub fn metadata(p: *const PendingProposals, output: []Proposal) !void {
        if (output.len != p.count) return error.InvalidDraftRows;
        for (p.lengths[0..p.count], output) |length, *proposal| {
            proposal.* = .{ .len = length };
            @memset(proposal.tokens[0..length], 0);
            @memset(proposal.scores[0..length], 0);
            @memset(proposal.probabilities[0..length], 1);
            for (proposal.parents[0..length], 0..) |*parent, depth| parent.* = @as(i32, @intCast(depth)) - 1;
        }
    }

    // The caller evaluates these arrays together with its target samples first.
    pub fn read(p: *const PendingProposals, output: []Proposal) !void {
        try p.metadata(output);
        for (p.tokens[0..p.count], output) |token, *proposal| if (proposal.len > 0) {
            for (proposal.tokens[0..proposal.len], mx.c.mlx_array_data_uint32(token)[0..proposal.len]) |*out, id| out.* = @intCast(id);
        };
    }
};

pub fn proposeStreamsLazy(m: anytype, streams: []const Stream(@TypeOf(m.*))) !PendingProposals {
    const M = @TypeOf(m.*);
    const capacity = M.max_shared_streams;
    if (streams.len == 0 or streams.len > capacity) return error.InvalidDraftRows;
    for (streams, 0..) |stream, i| {
        if (stream.budget > 15) return error.InvalidDraftBudget;
        if (!stream.settings.metal and stream.settings.temperature != 0) return error.RequiresGPUSampling;
        try stream.settings.validate();
        if (stream.state.borrowed) return error.RequestRoundActive;
        for (streams[0..i]) |other| if (stream.state == other.state) return error.DuplicateStream;
        if (stream.state.position < 0) return error.InvalidSamplingPosition;
        _ = std.math.add(i32, stream.state.position, @intCast(stream.budget)) catch return error.InvalidSamplingPosition;
        if (stream.first < 0 or stream.first >= M.vocab) return error.InvalidToken;
    }
    var pending = PendingProposals{ .count = streams.len };
    errdefer pending.deinit();
    if (M == @import("gemma.zig").Model) {
        for (streams, 0..) |stream, i| {
            const budget = @min(stream.budget, m.maxDrafts());
            pending.lengths[i] = budget;
            if (budget == 0) continue;
            stream.state.swap(m);
            defer stream.state.swap(m);
            pending.tokens[i] = try m.proposeLazy(stream.first, budget);
        }
        return pending;
    }
    var caches: [capacity]M.DraftCache = undefined;
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*cache| cache.deinit();
    var scope = mx.Scope{};
    defer scope.deinit();
    var hidden: [capacity]mx.Array = undefined;
    var tokens: [capacity]mx.Array = undefined;
    var parts: [capacity][15]mx.Array = undefined;
    var most: usize = 0;
    var predicted: [capacity]bool = @splat(false);
    var cached: [capacity]bool = @splat(false);
    for (streams, 0..) |stream, i| {
        if (@hasDecl(M, "HeadPrediction")) {
            predicted[i] = stream.budget > 0 and stream.state.head_prediction.matches(stream.state.position, stream.first, stream.settings);
        }
        pending.lengths[i] = stream.budget;
        if (M == nemotron.Model and predicted[i] and stream.state.head_prediction.drafts.ctx != null and stream.budget <= mx.c.mlx_array_size(stream.state.head_prediction.drafts)) {
            const drafts = stream.state.head_prediction.drafts;
            pending.tokens[i] = try mx.retain(if (stream.budget == mx.c.mlx_array_size(drafts)) drafts else try scope.slice(drafts, 0, 0, @intCast(stream.budget)));
            caches[i] = .{};
            initialized += 1;
            cached[i] = true;
            continue;
        }
        caches[i] = if (@hasDecl(M, "HeadPrediction") and predicted[i]) try stream.state.head_prediction.cache.clone() else try stream.state.head_cache.clone();
        initialized += 1;
        if (@hasDecl(M, "HeadPrediction") and predicted[i]) {
            hidden[i] = stream.state.head_prediction.hidden;
            tokens[i] = stream.state.head_prediction.first;
            parts[i][0] = tokens[i];
        } else {
            hidden[i] = stream.state.draft_hidden;
            tokens[i] = try scope.ints(&.{stream.first});
        }
        most = @max(most, stream.budget);
    }
    for (0..most) |depth| {
        var active: [capacity]usize = undefined;
        var inputs: [capacity]mx.Array = undefined;
        var token_parts: [capacity]mx.Array = undefined;
        var state: [capacity]*M.DraftCache = undefined;
        var positions: [capacity]i32 = undefined;
        var settings: [capacity]sampling.Sampling = undefined;
        var count: usize = 0;
        for (streams, 0..) |stream, i| if (!cached[i] and stream.budget > depth and !(depth == 0 and predicted[i])) {
            active[count] = i;
            inputs[count] = hidden[i];
            token_parts[count] = tokens[i];
            state[count] = &caches[i];
            positions[count] = stream.state.position + @as(i32, @intCast(depth)) + 1;
            settings[count] = stream.settings;
            count += 1;
        };
        if (count == 0) continue;
        const next = try m.draftStepStreams(&scope, try scope.cat(inputs[0..count], 0), try scope.cat(token_parts[0..count], 0), state[0..count]);
        const selected = try @import("gpu_sampling.zig").sampleRows(&m.kernels, &scope, try m.draftHead(&scope, next), positions[0..count], settings[0..count], m.weights.arrays.get("draft_ids"));
        for (active[0..count], 0..) |i, row| {
            const at: i32 = @intCast(row);
            tokens[i] = try scope.slice(selected, 0, at, at + 1);
            parts[i][depth] = tokens[i];
            hidden[i] = try scope.slice(next, 0, at, at + 1);
        }
    }
    for (streams, 0..) |stream, i| if (!cached[i] and stream.budget > 0) {
        pending.tokens[i] = try mx.retain(try scope.cat(parts[i][0..stream.budget], 0));
    };
    var arrays: [capacity]mx.Array = undefined;
    const ready = pending.arrays(&arrays);
    if (most > 0 and ready.len > 0) try mx.evalMany(ready, true);
    return pending;
}

pub fn proposeStreams(m: anytype, streams: []const Stream(@TypeOf(m.*)), output: []Proposal) !void {
    const M = @TypeOf(m.*);
    const capacity = M.max_shared_streams;
    if (streams.len == 0 or streams.len > capacity or output.len != streams.len) return error.InvalidDraftRows;
    var queued = true;
    for (streams) |stream| queued = queued and (stream.settings.metal or stream.settings.temperature == 0);
    if (queued) {
        var pending = try proposeStreamsLazy(m, streams);
        defer pending.deinit();
        var arrays: [capacity]mx.Array = undefined;
        const ready = pending.arrays(&arrays);
        if (ready.len > 0) try mx.evalMany(ready, false);
        return pending.read(output);
    }
    var caches: [capacity]M.DraftCache = undefined;
    var initialized: usize = 0;
    defer for (caches[0..initialized]) |*cache| cache.deinit();
    var scope = mx.Scope{};
    defer scope.deinit();
    var hidden: [capacity]mx.Array = undefined;
    var tokens: [capacity]i32 = undefined;
    var most: usize = 0;
    for (streams, output, 0..) |stream, *proposal, i| {
        if (stream.budget > 15) return error.InvalidDraftBudget;
        caches[i] = try stream.state.head_cache.clone();
        initialized += 1;
        hidden[i] = stream.state.draft_hidden;
        tokens[i] = stream.first;
        proposal.* = .{ .len = stream.budget };
        most = @max(most, stream.budget);
    }
    for (0..most) |depth| {
        var active: [capacity]usize = undefined;
        var inputs: [capacity]mx.Array = undefined;
        var ids: [capacity]i32 = undefined;
        var state: [capacity]*M.DraftCache = undefined;
        var count: usize = 0;
        for (streams, 0..) |stream, i| if (stream.budget > depth) {
            active[count] = i;
            inputs[count] = hidden[i];
            ids[count] = tokens[i];
            state[count] = &caches[i];
            count += 1;
        };
        const next = try m.draftStepStreams(&scope, try scope.cat(inputs[0..count], 0), try scope.ints(ids[0..count]), state[0..count]);
        const logits = try m.draftHead(&scope, next);
        var positions: [capacity]i32 = undefined;
        var settings: [capacity]sampling.Sampling = undefined;
        for (active[0..count], 0..) |i, row| {
            positions[row] = streams[i].state.position + @as(i32, @intCast(depth)) + 1;
            settings[row] = streams[i].settings;
        }
        const selected = try sampling.streamRowsMapped(&m.kernels, &scope, logits, positions[0..count], settings[0..count], m.weights.arrays.get("draft_ids"));
        defer mx.allocator.free(selected);
        for (active[0..count], 0..) |i, row| {
            tokens[i] = selected[row];
            output[i].tokens[depth] = selected[row];
        }
        for (active[0..count], 0..) |i, row| {
            const at: i32 = @intCast(row);
            hidden[i] = try scope.slice(next, 0, at, at + 1);
            output[i].parents[depth] = @as(i32, @intCast(depth)) - 1;
            output[i].scores[depth] = 0;
        }
    }
}
