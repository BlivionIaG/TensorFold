const std = @import("std");
const mx = @import("mlx.zig");
const gemma = @import("gemma.zig");
const ops = @import("gemma_ops.zig");
const round = @import("decode_round.zig");
const A = mx.Array;
const limit = gemma.Model.max_shared_rows;
const stream_limit = gemma.Model.max_shared_streams;
const window_limit = gemma.Model.max_decode_rows;
pub const State = @import("request_state.zig").State(gemma.Model);
pub const Stream = struct { state: *State, tokens: []const i32, parents: []const i32 };
const Entry = struct { state: *State, first: i32, pass: gemma.Pass };

// The model and request states must remain at stable addresses until deinit.
pub const Pass = struct {
    scope: mx.Scope = .{},
    model: *gemma.Model,
    ticket: round.Ticket,
    entries: []Entry,
    logits: A = mx.empty,
    hidden: A = mx.empty,
    count: usize,

    pub fn view(p: *Pass, index: usize) !*const gemma.Pass {
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
        p.ticket.release();
    }

    pub fn commit(p: *Pass, paths: []const []const i32) !void {
        try p.ticket.expect(.forwarded);
        errdefer p.ticket.owner.stage = .failed;
        if (paths.len != p.entries.len) return error.InvalidCommit;
        for (p.entries, paths) |entry, path| {
            if (!entry.state.borrowed or entry.state.position != entry.pass.position or entry.state.generation != entry.pass.generation or path.len > entry.pass.rows) return error.InvalidCommit;
            for (path, 0..) |row, i| if (row != i) return error.InvalidCommit;
        }
        const next = try mx.allocator.alloc([30]gemma.Cache, p.entries.len);
        defer mx.allocator.free(next);
        @memset(next, @splat(.{}));
        defer for (next) |*cache| for (cache) |*layer| layer.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        var arrays: [stream_limit * 60]A = undefined;
        var count: usize = 0;
        var needs_eval = false;
        for (paths) |path| if (path.len > 0) {
            try mx.eval(p.logits);
            break;
        };
        for (p.entries, paths, next) |*entry, path, *cache| {
            if (path.len == 0) continue;
            needs_eval = needs_eval or !entry.pass.staged_ready or path.len < entry.pass.rows;
            cache.* = try gemma.Model.acceptCache(&scope, entry.state.cache, &entry.pass, path.len);
            for (cache) |layer| {
                arrays[count] = layer.keys;
                arrays[count + 1] = layer.values;
                count += 2;
            }
        }
        if (needs_eval and count > 0) try mx.evalMany(arrays[0..count], false);
        for (next, paths) |cache, path| if (path.len > 0) for (cache) |layer| {
            try layer.observe();
        };
        for (p.entries, paths, next) |entry, path, *cache| {
            if (path.len == 0) continue;
            for (entry.state.cache, cache) |*old, *replacement| {
                old.deinit();
                old.* = replacement.*;
                replacement.* = .{};
            }
            entry.state.position += @intCast(path.len);
            entry.state.generation +%= 1;
        }
        try p.ticket.advance(.forwarded, .settled);
    }
};

pub fn forward(m: *gemma.Model, streams: []const Stream) !Pass {
    if (m.round_owner.stage != .idle) return error.ModelRoundActive;
    if (streams.len == 0 or streams.len > stream_limit) return error.InvalidStreams;
    var rows: usize = 0;
    for (streams, 0..) |stream, index| {
        if (stream.state.borrowed) return error.RequestRoundActive;
        if (stream.state.cache.len != 30 or stream.tokens.len == 0 or stream.tokens.len > window_limit or stream.tokens.len != stream.parents.len or stream.tokens.len > limit - rows) return error.InvalidStreams;
        for (streams[0..index]) |other| if (other.state == stream.state) return error.DuplicateStream;
        for (stream.tokens) |token| if (token < 0 or token >= gemma.Model.vocab) return error.InvalidToken;
        for (stream.parents, 0..) |parent, i| if (parent != @as(i32, @intCast(i)) - 1) return error.UnsupportedTree;
        if (stream.state.position < 0 or stream.state.position > 262144 - stream.tokens.len) return error.InvalidStreams;
        for (stream.state.cache, 0..) |cache, layer| {
            if (cache.keys.ctx == null or cache.values.ctx == null) {
                if (cache.keys.ctx != null or cache.values.ctx != null or stream.state.position != 0) return error.InvalidCacheState;
                continue;
            }
            const g = gemma.Model.geometry(layer);
            const shape = [_]i32{ 1, g.kv_heads, if (gemma.Model.sliding(layer)) 1152 else stream.state.position, g.head_dim };
            if (!std.mem.eql(i32, mx.shape(cache.keys), &shape) or !std.mem.eql(i32, mx.shape(cache.values), &shape) or mx.dtype(cache.keys) != mx.bf16 or mx.dtype(cache.values) != mx.bf16) return error.InvalidCacheState;
        }
        rows += stream.tokens.len;
    }
    const entries = try mx.allocator.alloc(Entry, streams.len);
    errdefer mx.allocator.free(entries);
    var tokens: [limit]i32 = undefined;
    var positions: [limit]i32 = undefined;
    var first: usize = 0;
    for (streams, entries) |stream, *entry| {
        entry.* = .{ .state = stream.state, .first = @intCast(first), .pass = .{ .position = stream.state.position, .generation = stream.state.generation, .rows = stream.tokens.len } };
        @memcpy(tokens[first..][0..stream.tokens.len], stream.tokens);
        for (0..stream.tokens.len) |i| positions[first + i] = stream.state.position + @as(i32, @intCast(i));
        first += stream.tokens.len;
    }
    const ticket = try m.round_owner.begin();
    for (streams) |stream| stream.state.borrowed = true;
    var p = Pass{ .model = m, .ticket = ticket, .entries = entries, .count = rows };
    errdefer {
        p.scope.deinit();
        for (entries) |*entry| entry.pass.deinit();
        for (streams) |stream| stream.state.borrowed = false;
        ticket.release();
    }
    const s = &p.scope;
    if (entries.len == 1) {
        const state = entries[0].state;
        entries[0].pass = try m.forwardState(state.cache, state.position, state.generation, streams[0].tokens);
        for (entries[0].pass.records, 0..) |record, layer| entries[0].pass.record_bytes[layer] = .{ mx.c.mlx_array_nbytes(record.keys), mx.c.mlx_array_nbytes(record.values) };
        p.hidden = entries[0].pass.hidden;
        p.logits = entries[0].pass.logits;
        try ticket.advance(.bound, .forwarded);
        return p;
    }
    const at = try ops.paddedInts(s, positions[0..rows]);
    var attention_rows: [stream_limit][2]ops.Rows = undefined;
    for (entries, attention_rows[0..entries.len]) |entry, *prepared| {
        const first_row: usize = @intCast(entry.first);
        const span = positions[first_row..][0..entry.pass.rows];
        prepared.* = .{ try ops.Rows.init(s, span, 0, 0, 512), try ops.Rows.init(s, span, 1024, 1152, 256) };
    }
    var h = try s.binary(mx.c.mlx_multiply, try m.weights.embed(s, "model.embed_tokens", tokens[0..rows]), try s.cast(try s.scalar(@floatCast(@sqrt(@as(f64, 2816)))), mx.bf16));
    var normed = try s.rms(h, try m.weight(0, "input_layernorm.weight"));
    var carried = [_]A{ mx.empty, mx.empty };
    defer for (carried) |array| mx.free(array);
    var taps: [32]A = undefined;
    var tap_count: usize = 0;
    for (0..30) |layer| {
        var layer_scope = mx.Scope{};
        defer layer_scope.deinit();
        const layer_s = &layer_scope;
        const local = gemma.Model.sliding(layer);
        const g = gemma.Model.geometry(layer);
        const qkv = try m.front(layer_s, layer, normed, at);
        var outputs: [stream_limit]A = undefined;
        for (entries, 0..) |*entry, j| {
            const end = entry.first + @as(i32, @intCast(entry.pass.rows));
            const q = try layer_s.slice(qkv[0], 0, entry.first, end);
            const k = if (entry.pass.rows == rows) qkv[1] else try layer_s.slice(qkv[1], 1, entry.first, end);
            const v = if (entry.pass.rows == rows) qkv[2] else try layer_s.slice(qkv[2], 1, entry.first, end);
            const old = entry.state.cache[layer];
            const keys = if (old.keys.ctx != null) old.attention("keys") else try layer_s.zeros(&.{ 1, g.kv_heads, if (local) 1152 else 1, g.head_dim }, mx.bf16);
            const values = if (old.values.ctx != null) old.attention("values") else try layer_s.zeros(mx.shape(keys), mx.bf16);
            entry.pass.records[layer] = .{ .keys = try entry.pass.scope.reshape(k, &.{ 1, g.kv_heads, @as(i32, @intCast(entry.pass.rows)), g.head_dim }), .values = try entry.pass.scope.reshape(v, &.{ 1, g.kv_heads, @as(i32, @intCast(entry.pass.rows)), g.head_dim }) };
            entry.pass.record_bytes[layer] = .{ mx.c.mlx_array_nbytes(qkv[1]), mx.c.mlx_array_nbytes(qkv[2]) };
            try gemma.Model.stageCacheLayer(entry.state.cache, &entry.pass, layer);
            outputs[j] = try m.attentions[j][@intFromBool(local)].apply(&m.kernels, layer_s, q, keys, values, k, v, attention_rows[j][@intFromBool(local)], 1);
        }
        const attended = if (entries.len == 1) outputs[0] else try layer_s.cat(outputs[0..entries.len], 0);
        const end = try m.back(layer_s, layer, attended, h);
        const next_h = try mx.retain(end[0]);
        const next_normed = mx.retain(end[1]) catch |err| {
            mx.free(next_h);
            return err;
        };
        for (carried) |array| mx.free(array);
        carried = .{ next_h, next_normed };
        h = next_h;
        normed = next_normed;
        if (m.draft) |d| for (d.parsed.value.dflash_config.target_layer_ids) |id| if (id == layer) {
            taps[tap_count] = try s.own(try mx.retain(h));
            tap_count += 1;
        };
        if ((layer + 1) % 8 == 0) {
            var pending: [1 + stream_limit * 16]A = undefined;
            pending[0] = normed;
            var count: usize = 1;
            for (entries) |entry| for (entry.pass.staged[layer - 7 .. layer + 1]) |cache| {
                pending[count] = cache.keys;
                pending[count + 1] = cache.values;
                count += 2;
            };
            try mx.evalMany(pending[0..count], true);
        }
    }
    p.hidden = try s.own(try mx.retain(normed));
    p.logits = try m.activations.call(s, .softcap, &.{ try m.project(s, normed, try m.weights.triple("model.embed_tokens")), try s.scalar(30) });
    var writes: [stream_limit * 60]A = undefined;
    for (entries, 0..) |*entry, i| {
        for (entry.pass.staged, 0..) |cache, layer| {
            writes[i * 60 + layer * 2] = cache.keys;
            writes[i * 60 + layer * 2 + 1] = cache.values;
        }
    }
    p.logits = try gemma.Model.cacheDependency(s, p.logits, writes[0 .. entries.len * 60]);
    const captured = if (tap_count > 0) try s.cat(taps[0..tap_count], -1) else mx.empty;
    for (entries) |*entry| {
        const end = entry.first + @as(i32, @intCast(entry.pass.rows));
        entry.pass.logits = try s.slice(p.logits, 0, entry.first, end);
        entry.pass.hidden = try s.slice(p.hidden, 0, entry.first, end);
        if (captured.ctx != null) entry.pass.taps = try s.slice(captured, 0, entry.first, end);
    }
    try ticket.advance(.bound, .forwarded);
    return p;
}

pub fn check(m: *gemma.Model) !void {
    m.reset();
    defer m.reset();
    var memory = try CheckMemory.init();
    defer memory.runtime.deinit();
    try checkCompiledResidency(m, &memory);
    try checkPreviewCancellation(m);
    try checkPreviewRotation(m);
    try checkBasic(m, &memory);
    try checkWide(m, &memory);
}

fn checkPreviewCancellation(m: *gemma.Model) !void {
    var state = try State.init(m);
    defer state.deinit();
    state.swap(m);
    defer state.swap(m);
    for ([_]i32{ 31, 47, 61 }) |token| {
        var warm = try m.forwardQueued(&.{token});
        defer warm.deinit();
        try m.commit(&warm, 1);
    }
    var scope = mx.Scope{};
    defer scope.deinit();
    var current = try m.forwardQueued(&.{79});
    var current_active = true;
    defer if (current_active) current.deinit();
    const expected_logits = try scope.own(try mx.retain(current.logits));
    var expected: [30]gemma.Cache = @splat(.{});
    defer for (&expected) |*cache| cache.deinit();
    for (current.staged, &expected) |cache, *saved| saved.* = try cache.clone();
    try std.testing.expectError(error.InvalidSamplingShape, m.forwardAfter(&current, mx.empty));
    try std.testing.expectError(error.InvalidSamplingShape, m.forwardAfter(&current, try scope.ints(&.{1})));
    const sample = try current.scope.argmax(current.logits);
    const position = m.position;
    const generation = m.generation;
    {
        var preview = try m.forwardAfter(&current, sample);
        defer preview.deinit();
        try mx.eval(preview.logits);
    }
    current.deinit();
    current_active = false;
    try std.testing.expectEqual(position, m.position);
    try std.testing.expectEqual(generation, m.generation);
    var retry = try m.forwardQueued(&.{79});
    defer retry.deinit();
    try m.commit(&retry, 1);
    var comparisons: [61]A = undefined;
    var same = mx.c.mlx_array_new();
    const first_rc = mx.c.mlx_array_equal(&same, expected_logits, retry.logits, false, mx.stream);
    comparisons[0] = try scope.result(first_rc, same);
    for (expected, m.cache, 0..) |before, after, i| inline for (.{ "keys", "values" }, 0..) |field, j| {
        var equal = mx.c.mlx_array_new();
        const rc = mx.c.mlx_array_equal(&equal, @field(before, field), @field(after, field), false, mx.stream);
        comparisons[1 + 2 * i + j] = try scope.result(rc, equal);
    };
    try mx.evalMany(&comparisons, false);
    for (comparisons) |equal| {
        var value: bool = false;
        try mx.check(mx.c.mlx_array_item_bool(&value, equal));
        try std.testing.expect(value);
    }
    std.debug.print("PASS: Gemma evaluated lookahead cancellation preserves the committed prefix, retry logits and all cache arrays.\n", .{});
}

fn checkPreviewRotation(m: *gemma.Model) !void {
    const kv = @import("kv_buffer.zig");
    const tracking = kv.track_reuse;
    kv.track_reuse = true;
    defer kv.track_reuse = tracking;
    var state = try State.init(m);
    defer state.deinit();
    state.swap(m);
    defer state.swap(m);
    const prefix = [_]i32{ 31, 47, 61 };
    const tokens = [_]i32{ 79, 83, 97, 101, 113, 127, 139, 149 };
    var scope = mx.Scope{};
    defer scope.deinit();
    var expected: [tokens.len]A = undefined;
    var caches: [30]gemma.Cache = @splat(.{});
    defer for (&caches) |*cache| cache.deinit();
    {
        var warm = try m.forward(&prefix);
        defer warm.deinit();
        try m.commit(&warm, prefix.len);
    }
    for (tokens, &expected) |token, *logits| {
        var serial = try m.forward(&.{token});
        defer serial.deinit();
        logits.* = try scope.own(try mx.retain(serial.logits));
        try m.commit(&serial, 1);
    }
    for (m.cache, &caches) |cache, *saved| saved.* = try cache.clone();
    m.reset();
    {
        var warm = try m.forward(&prefix);
        defer warm.deinit();
        try m.commit(&warm, prefix.len);
    }
    var pending = try m.forwardQueued(tokens[0..1]);
    defer pending.deinit();
    for (tokens[1..], 1..) |token, i| {
        const input = try scope.cast(try scope.ints(&.{token}), mx.c.MLX_UINT32);
        var next = try m.forwardAfter(&pending, input);
        errdefer next.deinit();
        try equalBits(pending.logits, expected[i - 1]);
        try equalBits(next.logits, expected[i]);
        for (next.staged, 0..) |cache, layer| if (kv.enabled and !gemma.Model.sliding(layer)) {
            try std.testing.expect(cache.storage.pipelined);
            try std.testing.expect(cache.storage.ring_count <= 2 and cache.storage.recent_count <= 2);
            if (i >= 2) inline for (.{ "keys", "values" }, 0..) |field, j| {
                try std.testing.expect(cache.storage.donors[j] != 0);
                try std.testing.expectEqual(cache.storage.donors[j], kv.address(@field(cache.storage, field).current));
            };
        };
        try m.commit(&pending, 1);
        pending.deinit();
        pending = next;
    }
    try m.commit(&pending, 1);
    for (m.cache, caches) |actual, saved| {
        try equalBits(actual.keys, saved.keys);
        try equalBits(actual.values, saved.values);
    }
    std.debug.print("PASS: Gemma queued singleton rotation reuses all global cache donors and preserves eight serial logits and all cache arrays.\n", .{});
}

fn checkCompiledResidency(m: *gemma.Model, memory: *const CheckMemory) !void {
    var retained: u64 = 0;
    var workspace: u64 = 0;
    var positions: [limit]i32 = undefined;
    for (&positions, 0..) |*position, i| position.* = @intCast(i);
    for ([_]i32{ 64, 1, 2, 3, 5, 8, 9, 16, 17, 31, 32, 48 }, 0..) |rows, i| {
        try memory.progress("compiled weight residency", @intCast(rows), i);
        const before = try @import("memory_runtime.zig").activeBytes();
        if (i == 0) try mx.check(mx.c.mlx_reset_peak_memory());
        {
            var scope = mx.Scope{};
            defer scope.deinit();
            const h = try scope.zeros(&.{ rows, 2816 }, mx.bf16);
            const qkv = try m.front(&scope, 0, h, try ops.paddedInts(&scope, positions[0..@intCast(rows)]));
            const out = try m.back(&scope, 0, try scope.zeros(&.{ rows, 16, 256 }, mx.bf16), h);
            try mx.evalMany(&.{ qkv[0], qkv[1], qkv[2], out[0], out[1] }, false);
            if (i == 0) {
                var peak: usize = 0;
                try mx.check(mx.c.mlx_get_peak_memory(&peak));
                workspace = peak -| before;
            }
        }
        const after = try @import("memory_runtime.zig").activeBytes();
        if (i == 0) retained = after else if (after > retained +| workspace) {
            std.debug.print("Compiled Gemma retained {d} extra bytes at {d} rows; one 64-row layer's measured workspace is {d} bytes\n", .{ after -| retained, rows, workspace });
            return error.CompiledWeightsDuplicated;
        }
    }
    std.debug.print("PASS: Gemma compiled front/back keep model weights shared across twelve row shapes.\n", .{});
}

fn checkBasic(m: *gemma.Model, memory: *const CheckMemory) !void {
    var states = [_]State{ try State.init(m), try State.init(m) };
    defer for (&states) |*state| state.deinit();
    for (&states, 0..) |*state, i| {
        state.swap(m);
        defer state.swap(m);
        const prefix = [_]i32{ 31, 47, 61, 79, 97 };
        try memory.progress("prefix", 1 + 4 * i, i);
        var pass = try m.forward(prefix[0 .. 1 + 4 * i]);
        defer pass.deinit();
        try m.commit(&pass, pass.rows);
    }
    for ([_][2]usize{ .{ 3, 2 }, .{ 8, 8 }, .{ 8, 8 }, .{ 8, 8 } }, 0..) |counts, iteration| {
        var reference = [_]State{ try states[0].clone(), try states[1].clone() };
        defer for (&reference) |*state| state.deinit();
        var ids: [2][8]i32 = undefined;
        const parents = [_]i32{ -1, 0, 1, 2, 3, 4, 5, 6 };
        var streams: [2]Stream = undefined;
        for (&streams, 0..) |*stream, j| {
            for (&ids[j], 0..) |*token, row| token.* = @intCast(103 + row + j * 19);
            stream.* = .{ .state = &states[j], .tokens = ids[j][0..counts[j]], .parents = parents[0..counts[j]] };
        }
        try memory.progress("shared forward", counts[0] + counts[1], iteration);
        var pass = try forward(m, &streams);
        defer pass.deinit();
        for (pass.entries) |entry| {
            try std.testing.expect(entry.pass.staged_ready);
            for (entry.pass.staged) |cache| {
                try std.testing.expect(cache.keys.ctx != null and cache.values.ctx != null);
            }
        }
        try std.testing.expectError(error.ModelRoundActive, forward(m, &streams));
        const kept = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7 };
        const keeps = if (iteration == 0) [2]usize{ 2, 0 } else if (iteration == 3) [2]usize{ 0, 3 } else [2]usize{ 8, 3 };
        for (&reference, 0..) |*state, j| {
            state.swap(m);
            defer state.swap(m);
            try memory.progress("isolated forward", counts[j], j);
            var expected = try m.forward(streams[j].tokens);
            defer expected.deinit();
            const view = try pass.view(j);
            try equalBits(view.logits, expected.logits);
            try equalBits(view.hidden, expected.hidden);
            if (view.taps.ctx != null) try equalBits(view.taps, expected.taps);
            for (view.records, expected.records) |actual, single| {
                try equalBits(actual.keys, single.keys);
                try equalBits(actual.values, single.values);
            }
            if (keeps[j] > 0) try m.commit(&expected, keeps[j]);
        }
        const held = [2]gemma.Cache{ states[0].cache[0], states[1].cache[0] };
        try memory.progress("shared commit", counts[0] + counts[1], iteration);
        try pass.commit(&.{ kept[0..keeps[0]], kept[0..keeps[1]] });
        try std.testing.expectError(error.InvalidRoundStage, pass.commit(&.{ &.{}, &.{} }));
        for (&reference, 0..) |*state, j| {
            if (keeps[j] == 0) {
                try std.testing.expectEqual(held[j].keys.ctx, states[j].cache[0].keys.ctx);
                try std.testing.expectEqual(held[j].values.ctx, states[j].cache[0].values.ctx);
            }
            try std.testing.expectEqual(state.position, states[j].position);
            try std.testing.expectEqual(state.generation, states[j].generation);
            for (state.cache, states[j].cache, 0..) |single, shared, layer| {
                try equalBits(single.keys, shared.keys);
                try equalBits(single.values, shared.values);
                if (@import("kv_buffer.zig").enabled and gemma.Model.sliding(layer) and keeps[j] > 0) {
                    try std.testing.expect(shared.storage.recent_count > 0 and shared.storage.recent_count <= 2);
                    const recent = shared.storage.recent[shared.storage.recent_count - 1];
                    try std.testing.expectEqual(states[j].position - @as(i32, @intCast(keeps[j])), recent.position);
                    const backing = mx.c.mlx_array_nbytes(recent.keys) / keeps[j] * (counts[0] + counts[1]);
                    try std.testing.expectEqual(backing, recent.backing_bytes[0]);
                    try std.testing.expect(shared.nbytes() >= recent.backing_bytes[0] + recent.backing_bytes[1]);
                    var cloned = try shared.clone();
                    defer cloned.deinit();
                    try std.testing.expectEqual(@as(usize, 0), cloned.storage.recent_count);
                    try std.testing.expectEqual(@as(usize, 0), cloned.storage.ring_count);
                } else if (@import("kv_buffer.zig").enabled and !gemma.Model.sliding(layer) and keeps[j] > 0) {
                    var cloned = try shared.clone();
                    defer cloned.deinit();
                    const capacity_bytes = mx.c.mlx_array_nbytes(shared.storage.keys.current) + mx.c.mlx_array_nbytes(shared.storage.values.current);
                    try std.testing.expect(capacity_bytes > mx.c.mlx_array_nbytes(shared.keys) + mx.c.mlx_array_nbytes(shared.values));
                    try std.testing.expectEqual(capacity_bytes, cloned.nbytes());
                    try std.testing.expect(cloned.storage.keys.current.ctx == null and cloned.storage.keys.spare.ctx == null);
                    try std.testing.expect(cloned.storage.values.current.ctx == null and cloned.storage.values.spare.ctx == null);
                    var retained = try cloned.clone();
                    defer retained.deinit();
                    try std.testing.expectEqual(capacity_bytes, retained.nbytes());
                }
            }
            std.mem.swap(@TypeOf(state.draft), &state.draft, &states[j].draft);
        }
        try memory.progress("compared", counts[0] + counts[1], iteration);
    }
    const before = states[0].position;
    var invalid = try forward(m, &.{.{ .state = &states[0], .tokens = &.{ 113, 127 }, .parents = &.{ -1, 0 } }});
    defer invalid.deinit();
    try std.testing.expectError(error.InvalidCommit, invalid.commit(&.{&.{1}}));
    try std.testing.expectEqual(before, states[0].position);
    try std.testing.expectError(error.InvalidRoundStage, invalid.commit(&.{&.{0}}));
    invalid.deinit();
    try std.testing.expectError(error.StaleRound, invalid.view(0));
}

fn checkWide(m: *gemma.Model, memory: *const CheckMemory) !void {
    var states: [5]State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized]) |*state| state.deinit();
    for (&states, 0..) |*state, i| {
        state.* = try State.init(m);
        initialized += 1;
        state.swap(m);
        defer state.swap(m);
        const prefix = [_]i32{ 31, 47, 61, 79, 97 };
        try memory.progress("wide prefix", i + 1, i);
        var pass = try m.forward(prefix[0 .. i + 1]);
        defer pass.deinit();
        try m.commit(&pass, pass.rows);
    }
    const cases = [_][]const usize{ &.{ 9, 8 }, &.{ 10, 11, 10 }, &.{ 7, 9, 16 }, &.{ 13, 16, 11, 8 }, &.{ 15, 9, 16, 11, 13 } };
    for (cases, 0..) |lengths, iteration| {
        var total: usize = 0;
        for (lengths) |count| total += count;
        var reference: [5]State = undefined;
        var copied: usize = 0;
        defer for (reference[0..copied]) |*state| state.deinit();
        var tokens: [5][window_limit]i32 = undefined;
        var parents: [window_limit]i32 = undefined;
        var kept: [window_limit]i32 = undefined;
        for (&parents, &kept, 0..) |*parent, *row, i| {
            row.* = @intCast(i);
            parent.* = row.* - 1;
        }
        var streams: [5]Stream = undefined;
        for (lengths, 0..) |count, j| {
            reference[j] = try states[j].clone();
            copied += 1;
            for (tokens[j][0..count], 0..) |*token, i| token.* = @intCast(109 + 41 * j + i + 101 * iteration);
            streams[j] = .{ .state = &states[j], .tokens = tokens[j][0..count], .parents = parents[0..count] };
        }
        try memory.progress("wide shared forward", total, iteration);
        var pass = try forward(m, streams[0..lengths.len]);
        defer pass.deinit();
        var keeps: [5]usize = undefined;
        var paths: [5][]const i32 = undefined;
        for (reference[0..copied], 0..) |*state, j| {
            state.swap(m);
            defer state.swap(m);
            try memory.progress("wide isolated forward", lengths[j], j);
            var expected = try m.forward(streams[j].tokens);
            defer expected.deinit();
            const view = try pass.view(j);
            try equalBits(view.logits, expected.logits);
            try equalBits(view.hidden, expected.hidden);
            if (view.taps.ctx != null) try equalBits(view.taps, expected.taps);
            keeps[j] = if (j == iteration % lengths.len) 0 else if (j % 2 == 0) lengths[j] else @max(1, lengths[j] / 2);
            paths[j] = kept[0..keeps[j]];
            if (keeps[j] > 0) try m.commit(&expected, keeps[j]);
        }
        try memory.progress("wide shared commit", total, iteration);
        try pass.commit(paths[0..lengths.len]);
        for (reference[0..copied], 0..) |*state, j| {
            try std.testing.expectEqual(state.position, states[j].position);
            try std.testing.expectEqual(state.generation, states[j].generation);
            for (state.cache, states[j].cache) |single, shared| {
                try equalBits(single.keys, shared.keys);
                try equalBits(single.values, shared.values);
            }
            std.mem.swap(@TypeOf(state.draft), &state.draft, &states[j].draft);
        }
        pass.deinit();
        try memory.progress("wide continuations", total, iteration);
        for (reference[0..copied], 0..) |*state, j| {
            const follow = [_]i32{@intCast(701 + j)};
            state.swap(m);
            var single = m.forward(&follow) catch |err| {
                state.swap(m);
                return err;
            };
            state.swap(m);
            defer single.deinit();
            var continuation = try forward(m, &.{.{ .state = &states[j], .tokens = &follow, .parents = &.{-1} }});
            defer continuation.deinit();
            try equalBits((try continuation.view(0)).logits, single.logits);
        }
        try memory.progress("wide compared", total, iteration);
    }
    std.debug.print("PASS: Gemma shared 17/31/32/48/64-row ragged forwards, partial/cancelled commits and continuations.\n", .{});
}

fn equalBits(a: A, b: A) !void {
    var scope = mx.Scope{};
    defer scope.deinit();
    try @import("variant_checks.zig").equalBits(&scope, a, b);
}

const CheckMemory = struct {
    runtime: @import("memory_runtime.zig").Runtime,
    limit: u64,

    fn init() !CheckMemory {
        var runtime = try @import("memory_runtime.zig").Runtime.init(@import("bonsai.zig").memory_limit);
        errdefer runtime.deinit();
        const limit_bytes = try runtime.admissionBudget(std.Options.debug_io, .gemma);
        var previous: usize = 0;
        try mx.check(mx.c.mlx_set_memory_limit(&previous, @intCast(limit_bytes)));
        return .{ .runtime = runtime, .limit = limit_bytes };
    }

    fn progress(memory: *const CheckMemory, phase: []const u8, rows: usize, index: usize) !void {
        const active = try @import("memory_runtime.zig").activeBytes();
        var cached: usize = 0;
        var peak: usize = 0;
        try mx.check(mx.c.mlx_get_cache_memory(&cached));
        try mx.check(mx.c.mlx_get_peak_memory(&peak));
        const rss: u64 = @intCast(@max(0, std.posix.getrusage(std.posix.rusage.SELF).maxrss));
        const gib = @as(f64, @floatFromInt(@import("memory_budget.zig").gib));
        std.debug.print("Gemma shared check {s}: rows={d}, index={d}, active={d:.2} GiB, cache={d:.2} GiB, peak={d:.2} GiB, peak_rss={d:.2} GiB, limit={d:.2} GiB\n", .{ phase, rows, index, @as(f64, @floatFromInt(active)) / gib, @as(f64, @floatFromInt(cached)) / gib, @as(f64, @floatFromInt(peak)) / gib, @as(f64, @floatFromInt(rss)) / gib, @as(f64, @floatFromInt(memory.limit)) / gib });
        if (active >= memory.limit or rss >= memory.runtime.budget) return error.SharedCheckMemoryLimit;
    }
};

test "Gemma shared rounds reject invalid ownership, chains and row capacity" {
    var m: gemma.Model = undefined;
    m.round_owner = .{};
    m.draft = null;
    var state = try State.init(&m);
    defer state.deinit();
    const stream = Stream{ .state = &state, .tokens = &.{123}, .parents = &.{-1} };
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{}));
    try std.testing.expectError(error.DuplicateStream, forward(&m, &.{ stream, stream }));
    state.borrowed = true;
    try std.testing.expectError(error.RequestRoundActive, forward(&m, &.{stream}));
    try std.testing.expectError(error.RequestRoundActive, state.clone());
    state.borrowed = false;
    state.position = -1;
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{stream}));
    state.position = 262144;
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{stream}));
    state.position = 1;
    try std.testing.expectError(error.InvalidCacheState, forward(&m, &.{stream}));
    state.position = 0;
    try std.testing.expectError(error.InvalidToken, forward(&m, &.{.{ .state = &state, .tokens = &.{gemma.Model.vocab}, .parents = &.{-1} }}));
    try std.testing.expectError(error.UnsupportedTree, forward(&m, &.{.{ .state = &state, .tokens = &.{ 123, 124, 125 }, .parents = &.{ -1, 0, 0 } }}));
    const many: [window_limit + 1]i32 = @splat(123);
    const parents: [window_limit + 1]i32 = @splat(-1);
    try std.testing.expectError(error.InvalidStreams, forward(&m, &.{.{ .state = &state, .tokens = &many, .parents = &parents }}));
    const streams: [stream_limit + 1]Stream = @splat(stream);
    try std.testing.expectError(error.InvalidStreams, forward(&m, &streams));
    const ticket = try m.round_owner.begin();
    defer ticket.release();
    try std.testing.expectError(error.ModelRoundActive, forward(&m, &.{stream}));
}
