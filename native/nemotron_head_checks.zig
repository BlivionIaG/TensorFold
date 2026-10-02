const std = @import("std");
const mx = @import("mlx.zig");
const nemotron = @import("nemotron.zig");
const A = mx.Array;
const equal = @import("variant_checks.zig").equalBits;
const past = [_]i32{ 0, 1, 3, 7, 31, 127, 9999, 10000 };

fn initialHidden(s: *mx.Scope) !A {
    var values: [8 * 2688]f32 = undefined;
    for (&values, 0..) |*value, index| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast(index % 2688 % 23)) - 11)) / 64 + @as(f32, @floatFromInt(index / 2688)) / 128;
    }
    return s.cast(try s.data(&values, &.{ 8, 2688 }, mx.f32t), mx.bf16);
}

fn seed(s: *mx.Scope, cache: *nemotron.Cache, index: usize) !void {
    if (past[index] == 0) return;
    const shape = [_]i32{ 1, 2, past[index], 128 };
    const keys = try s.binary(mx.c.mlx_add, try s.zeros(&shape, mx.bf16), try s.cast(try s.scalar(@as(f32, @floatFromInt(index + 1)) / 64), mx.bf16));
    const values = try s.binary(mx.c.mlx_add, try s.zeros(&shape, mx.bf16), try s.cast(try s.scalar(-@as(f32, @floatFromInt(index + 1)) / 32), mx.bf16));
    cache.a = try mx.retain(keys);
    cache.b = try mx.retain(values);
}

fn save(s: *mx.Scope, output: []const u8, name: []const u8, value: A) !void {
    const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ output, name }, 0);
    defer mx.allocator.free(path);
    const converted = try s.cast(value, mx.f32t);
    try mx.eval(converted);
    try mx.check(mx.c.mlx_save(path, converted));
}

fn checkWideAbsorption(m: *nemotron.Model, seeds: *const [8]nemotron.Cache) !void {
    const neural = @import("neural_draft.zig");
    const State = @import("request_state.zig").State(nemotron.Model);
    var scope = mx.Scope{};
    defer scope.deinit();
    const initial = try initialHidden(&scope);
    const hidden = try scope.cat(&.{ initial, initial }, 0);
    var states: [8]State = undefined;
    var references: [8]State = undefined;
    var made: usize = 0;
    var copied: usize = 0;
    defer for (states[0..made]) |*state| state.deinit();
    defer for (references[0..copied]) |*state| state.deinit();
    for (&states, &references, 0..) |*state, *reference, i| {
        state.* = try State.init(m);
        made += 1;
        state.head_cache = try seeds[i].clone();
        state.draft_hidden = try mx.retain(try scope.slice(initial, 0, @intCast(i), @intCast(i + 1)));
        reference.* = try state.clone();
        copied += 1;
    }
    const rows = [_]i32{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    for ([_][8]usize{ @splat(16), .{ 16, 0, 2, 7, 16, 4, 3, 1 } }) |lengths| {
        var tokens: [128]i32 = undefined;
        var streams: [8]neural.AbsorbStream(nemotron.Model) = undefined;
        for (&states, &references, &streams, lengths, 0..) |*state, *reference, *stream, count, i| {
            for (tokens[i * 16 ..][0..16], 0..) |*token, row| token.* = @intCast(4000 + i * 37 + row * 13);
            stream.* = .{ .state = state, .hidden = hidden, .tokens = tokens[i * 16 ..][0..count], .rows = rows[0..count] };
            if (count == 0) continue;
            const end: i32 = @intCast(count);
            const context = try scope.cat(&.{ reference.draft_hidden, try scope.slice(hidden, 0, 0, end - 1) }, 0);
            _ = try m.draftStepArray(&scope, context, try scope.ints(stream.tokens), &reference.head_cache, false);
            try mx.replace(&reference.draft_hidden, try scope.slice(hidden, 0, end - 1, end));
        }
        try neural.absorbStreams(m, &streams);
        for (states, references) |actual, expected| {
            try equal(&scope, expected.head_cache.a, actual.head_cache.a);
            try equal(&scope, expected.head_cache.b, actual.head_cache.b);
            try equal(&scope, expected.draft_hidden, actual.draft_hidden);
        }
    }
    std.debug.print("Nemotron MTP absorption: exact eight16-row windows and ragged continuation with cancellation\n", .{});
}

fn checkPrefillAbsorption(m: *nemotron.Model, seed_cache: nemotron.Cache) !void {
    const neural = @import("neural_draft.zig");
    const State = @import("request_state.zig").State(nemotron.Model);
    var scope = mx.Scope{};
    defer scope.deinit();
    const initial = try initialHidden(&scope);
    const hidden = try scope.cat(&.{ initial, initial, initial, initial, initial }, 0);
    var tokens: [33]i32 = undefined;
    var rows: [33]i32 = undefined;
    for (&tokens, &rows, 0..) |*token, *row, i| {
        token.* = @intCast(5000 + i * 19);
        row.* = @intCast(i);
    }
    for ([_]bool{ false, true }) |previous| for ([_]usize{ 1, 16, 17, 33 }) |count| {
        var actual = try State.init(m);
        defer actual.deinit();
        if (previous) {
            actual.head_cache = try seed_cache.clone();
            actual.draft_hidden = try mx.retain(try scope.slice(initial, 0, 0, 1));
        }
        var expected = try actual.clone();
        defer expected.deinit();
        for (rows[0..count]) |row| {
            var step_scope = mx.Scope{};
            defer step_scope.deinit();
            if (expected.draft_hidden.ctx != null) _ = try m.draftStep(&step_scope, expected.draft_hidden, tokens[@intCast(row)], &expected.head_cache);
            try mx.replace(&expected.draft_hidden, try step_scope.slice(hidden, 0, row, row + 1));
        }
        var pass = struct { scope: mx.Scope = .{}, hidden: A }{ .hidden = hidden };
        defer pass.scope.deinit();
        try neural.absorb(m, &actual, null, &pass, tokens[0..count], rows[0..count]);
        try std.testing.expectEqual(expected.head_cache.a.ctx == null, actual.head_cache.a.ctx == null);
        if (expected.head_cache.a.ctx != null) {
            try equal(&scope, expected.head_cache.a, actual.head_cache.a);
            try equal(&scope, expected.head_cache.b, actual.head_cache.b);
        }
        try equal(&scope, expected.draft_hidden, actual.draft_hidden);
        if (count == 33) {
            var invalid: [17]i32 = undefined;
            @memcpy(&invalid, rows[0..17]);
            invalid[16] = 33;
            const keys = actual.head_cache.a.ctx;
            const last = actual.draft_hidden.ctx;
            try std.testing.expectError(error.InvalidDraftRows, neural.absorb(m, &actual, null, &pass, &tokens, &invalid));
            try std.testing.expectEqual(keys, actual.head_cache.a.ctx);
            try std.testing.expectEqual(last, actual.draft_hidden.ctx);
        }
    };
    std.debug.print("Nemotron prefill MTP absorption: exact scalar cache/hidden for fresh/prior context, 1/16/17/33 rows and failed later chunk\n", .{});
}

fn checkProposals(m: *nemotron.Model, seeds: *const [8]nemotron.Cache) !void {
    const neural = @import("neural_draft.zig");
    const State = @import("request_state.zig").State(nemotron.Model);
    const Proposal = @import("drafter.zig").Proposal;
    var scope = mx.Scope{};
    defer scope.deinit();
    const hidden = try initialHidden(&scope);
    var states: [8]State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized]) |*state| state.deinit();
    var streams: [8]neural.Stream(nemotron.Model) = undefined;
    const budgets = [_]usize{ 0, 1, 3, 2, 3, 1, 2, 3 };
    for (&states, &streams, budgets, 0..) |*state, *stream, budget, i| {
        state.* = try State.init(m);
        initialized += 1;
        state.head_cache = try seeds[i].clone();
        state.draft_hidden = try mx.retain(try scope.slice(hidden, 0, @intCast(i), @intCast(i + 1)));
        state.position = past[i] + 19;
        stream.* = .{ .state = state, .first = @intCast(6000 + i * 37), .budget = budget, .settings = .{ .metal = i % 4 != 1, .temperature = if (i % 4 == 1) 0 else 0.7, .seed = 0x1234567800000000 + i * 97, .top_k = if (i == 3) 0 else 12, .top_p = 0.8, .min_p = if (i % 2 == 0) 0.02 else 0 } };
    }
    const position = m.position;
    defer m.position = position;
    for ([_]bool{ false, true }) |cpu_fallback| {
        streams[3].settings.metal = !cpu_fallback;
        var expected: [8]Proposal = undefined;
        var actual: [8]Proposal = undefined;
        var cache_handles: [8]?*anyopaque = undefined;
        var hidden_handles: [8]?*anyopaque = undefined;
        for (streams, &expected, &cache_handles, &hidden_handles) |stream, *proposal, *cache, *last| {
            m.position = stream.state.position;
            proposal.* = try neural.propose(m, stream.state, null, stream.first, stream.budget, stream.settings);
            cache.* = stream.state.head_cache.a.ctx;
            last.* = stream.state.draft_hidden.ctx;
        }
        try neural.proposeStreams(m, &streams, &actual);
        if (!cpu_fallback) {
            var pending = try neural.proposeStreamsLazy(m, &streams);
            defer pending.deinit();
            var arrays: [8]A = undefined;
            try mx.evalMany(pending.arrays(&arrays), false);
            try pending.read(&actual);
            var cancelled = try neural.proposeStreamsLazy(m, &streams);
            cancelled.deinit();
        }
        for (streams, expected, actual, cache_handles, hidden_handles) |stream, before, after, cache, last| {
            try std.testing.expectEqual(before.len, after.len);
            try std.testing.expectEqualSlices(i32, before.tokens[0..before.len], after.tokens[0..after.len]);
            try std.testing.expectEqualSlices(i32, before.parents[0..before.len], after.parents[0..after.len]);
            try std.testing.expectEqual(cache, stream.state.head_cache.a.ctx);
            try std.testing.expectEqual(last, stream.state.draft_hidden.ctx);
        }
    }
    streams[3].settings.metal = true;
    {
        var deep = streams[2];
        deep.budget = 15;
        m.position = deep.state.position;
        const expected = try neural.propose(m, deep.state, null, deep.first, deep.budget, deep.settings);
        var pending = try neural.proposeStreamsLazy(m, &.{deep});
        defer pending.deinit();
        var arrays: [8]A = undefined;
        try mx.evalMany(pending.arrays(&arrays), false);
        var actual: [1]Proposal = undefined;
        try pending.read(&actual);
        try std.testing.expectEqualSlices(i32, expected.tokens[0..15], actual[0].tokens[0..15]);
    }
    try std.testing.expectError(error.DuplicateStream, neural.proposeStreamsLazy(m, &.{ streams[1], streams[1] }));
    states[1].borrowed = true;
    const borrowed = neural.proposeStreamsLazy(m, streams[1..2]);
    states[1].borrowed = false;
    try std.testing.expectError(error.RequestRoundActive, borrowed);
    {
        var invalid = streams[1];
        invalid.budget = 16;
        try std.testing.expectError(error.InvalidDraftBudget, neural.proposeStreamsLazy(m, &.{invalid}));
        invalid = streams[3];
        invalid.settings.metal = false;
        try std.testing.expectError(error.RequiresGPUSampling, neural.proposeStreamsLazy(m, &.{invalid}));
    }
    const previous = mx.allocator;
    for ([_]usize{ 0, 1 }) |failure| {
        var failing = std.testing.FailingAllocator.init(previous, .{ .fail_index = failure, .resize_fail_index = 0 });
        const cache = states[1].head_cache.a.ctx;
        const last = states[1].draft_hidden.ctx;
        {
            mx.allocator = failing.allocator();
            defer mx.allocator = previous;
            try std.testing.expectError(error.OutOfMemory, neural.proposeStreamsLazy(m, streams[1..2]));
        }
        try std.testing.expect(failing.has_induced_failure);
        try std.testing.expectEqual(cache, states[1].head_cache.a.ctx);
        try std.testing.expectEqual(last, states[1].draft_hidden.ctx);
    }
    var resident: ?usize = null;
    var retained: ?i128 = null;
    var tracking = std.testing.FailingAllocator.init(previous, .{});
    {
        mx.allocator = tracking.allocator();
        defer mx.allocator = previous;
        for (0..3) |_| {
            {
                var pending = try neural.proposeStreamsLazy(m, &streams);
                defer pending.deinit();
                var arrays: [8]A = undefined;
                try mx.evalMany(pending.arrays(&arrays), false);
            }
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            const active: usize = @intCast(try @import("memory_runtime.zig").activeBytes());
            const held = @as(i128, tracking.allocated_bytes) - @as(i128, tracking.freed_bytes);
            if (resident) |before| try std.testing.expectEqual(before, active) else resident = active;
            if (retained) |before| try std.testing.expectEqual(before, held) else retained = held;
        }
    }
    std.debug.print("Nemotron shared proposals: exact serial depths0/1/2/3/15, seeds/mapping, CPU fallback, deferred ownership, cancellation, failures and bounded memory\n", .{});
}

fn checkEarlyPredictions(m: *nemotron.Model, seeds: *const [8]nemotron.Cache) !void {
    const neural = @import("neural_draft.zig");
    const State = @import("request_state.zig").State(nemotron.Model);
    const sampling = @import("sampling.zig");
    var scope = mx.Scope{};
    defer scope.deinit();
    const initial = try initialHidden(&scope);
    const hidden = try scope.cat(&.{ initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial, initial }, 0);
    const saved_position = m.position;
    defer m.position = saved_position;
    for ([_]usize{ 1, 16 }, [_]usize{ 0, 7 }) |count, seed_index| {
        var state = try State.init(m);
        defer state.deinit();
        state.head_cache = try seeds[seed_index].clone();
        state.draft_hidden = try mx.retain(try scope.slice(initial, 0, @intCast(seed_index), @intCast(seed_index + 1)));
        state.position = past[seed_index] + 19;
        const settings = sampling.Sampling{ .metal = count > 1, .temperature = if (count > 1) 0.7 else 0, .seed = 0x1234567800000097, .top_k = 12, .top_p = 0.8, .min_p = 0.02 };
        const input = try scope.slice(hidden, 0, 0, @intCast(count));
        var tokens: [16]i32 = undefined;
        for (tokens[0..count], 0..) |*token, row| token.* = @intCast(8000 + row * 13);
        var verification = try neural.HeadVerification.init(m, &.{.{ .state = &state, .anchor = 6000, .count = count, .settings = settings }}, input, try scope.cast(try scope.ints(tokens[0..count]), mx.c.MLX_UINT32));
        defer verification.deinit();
        var update = try verification.prepare(0, count, tokens[count - 1], true);
        defer update.deinit();
        var serial = try state.head_cache.clone();
        defer serial.deinit();
        _ = try m.draftStep(&scope, state.draft_hidden, 6000, &serial);
        for (0..count) |row| {
            if (row + 1 == count) {
                try equal(&scope, serial.a, update.cache.a);
                try equal(&scope, serial.b, update.cache.b);
            }
            const at: i32 = @intCast(row);
            const out = try m.draftStep(&scope, try scope.slice(input, 0, at, at + 1), tokens[row], &serial);
            if (row + 1 == count) {
                try equal(&scope, out, update.prediction.hidden);
                try equal(&scope, serial.a, update.prediction.cache.a);
                try equal(&scope, serial.b, update.prediction.cache.b);
                const ids = try sampling.rowsMapped(&m.kernels, &scope, try m.draftHead(&scope, out), &.{state.position + @as(i32, @intCast(count)) + 1}, settings, m.weights.arrays.get("draft_ids"));
                defer mx.allocator.free(ids);
                try mx.eval(update.prediction.first);
                try std.testing.expectEqual(ids[0], @as(i32, @intCast(mx.c.mlx_array_data_uint32(update.prediction.first)[0])));
            }
        }
        try equal(&scope, try scope.slice(input, 0, @intCast(count - 1), @intCast(count)), update.hidden);
        try std.testing.expect(state.head_prediction.first.ctx == null);
        if (state.head_cache.a.ctx != null) try equal(&scope, seeds[seed_index].a, state.head_cache.a) else try std.testing.expect(seeds[seed_index].a.ctx == null);
    }
    for ([_][8]usize{ .{ 1, 4, 2, 3, 16, 1, 2, 3 }, @splat(16) }) |lengths| {
        var states: [8]State = undefined;
        var references: [8]State = undefined;
        var made: usize = 0;
        var copied: usize = 0;
        defer for (states[0..made]) |*state| state.deinit();
        defer for (references[0..copied]) |*state| state.deinit();
        var heads: [8]neural.HeadStream = undefined;
        var following: [128]i32 = undefined;
        var total: usize = 0;
        for (&states, &references, &heads, lengths, 0..) |*state, *reference, *head, count, i| {
            state.* = try State.init(m);
            made += 1;
            state.head_cache = try seeds[i].clone();
            state.draft_hidden = try mx.retain(try scope.slice(initial, 0, @intCast(i), @intCast(i + 1)));
            state.position = past[i] + 19;
            reference.* = try state.clone();
            copied += 1;
            head.* = .{ .state = state, .anchor = @intCast(6000 + i * 37), .count = count, .settings = .{ .metal = i % 2 == 0, .temperature = if (i % 2 == 0) 0.7 else 0, .seed = 0x1234567800000000 + i * 97, .top_k = 12, .top_p = 0.8, .min_p = 0.02 } };
            for (following[total..][0..count], 0..) |*token, row| token.* = @intCast(8000 + i * 31 + row * 13);
            total += count;
        }
        const input = try scope.slice(hidden, 0, 0, @intCast(total));
        const tokens = try scope.cast(try scope.ints(following[0..total]), mx.c.MLX_UINT32);
        var verification = try neural.HeadVerification.init(m, &heads, input, tokens);
        defer verification.deinit();
        var all_rows: [128]i32 = undefined;
        var all_positions: [128]i32 = undefined;
        var all_settings: [128]sampling.Sampling = undefined;
        var keeps: [8]neural.HeadVerification.Keep = undefined;
        var offset: usize = 0;
        for (heads, lengths, &keeps, 0..) |head, count, *keep, i| {
            keep.* = .{ .slot = i, .count = if (i % 2 == 0) count else 1, .token = following[offset + (if (i % 2 == 0) count else 1) - 1], .predict = true };
            for (0..count) |row| {
                all_rows[offset + row] = @intCast(offset + row);
                all_positions[offset + row] = head.state.position + @as(i32, @intCast(row)) + 2;
                all_settings[offset + row] = head.settings;
            }
            offset += count;
        }
        const verification_hidden = try m.draftWindowTail(&scope, verification.rows, all_rows[0..total]);
        const reordered = [_]i32{ @intCast(total - 1), 0, @intCast(lengths[0]), 0, @intCast(total - 2) };
        const reordered_hidden = try m.draftWindowTail(&scope, verification.rows, &reordered);
        try equal(&scope, try scope.take(verification_hidden, try scope.ints(&reordered), 0), reordered_hidden);
        const verification_firsts = try @import("gpu_sampling.zig").sampleRows(&m.kernels, &scope, try m.draftHead(&scope, verification_hidden), all_positions[0..total], all_settings[0..total], m.weights.arrays.get("draft_ids"));
        try mx.eval(verification_firsts);
        var updates = try verification.prepareBatch(&keeps);
        defer for (&updates) |*update| update.deinit();
        const subset = [_]neural.HeadVerification.Keep{ keeps[5], .{ .slot = 0, .count = keeps[0].count, .token = keeps[0].token, .predict = false }, keeps[2] };
        var sparse_updates = try verification.prepareBatch(&subset);
        defer for (&sparse_updates) |*update| update.deinit();
        for (subset) |keep| {
            const actual = sparse_updates[keep.slot];
            const full = updates[keep.slot];
            try equal(&scope, full.cache.a, actual.cache.a);
            try equal(&scope, full.cache.b, actual.cache.b);
            try equal(&scope, full.hidden, actual.hidden);
            if (keep.predict) {
                try equal(&scope, full.prediction.hidden, actual.prediction.hidden);
                try mx.evalMany(&.{ full.prediction.first, actual.prediction.first }, false);
                try std.testing.expectEqual(mx.c.mlx_array_data_uint32(full.prediction.first)[0], mx.c.mlx_array_data_uint32(actual.prediction.first)[0]);
            } else {
                try std.testing.expect(actual.prediction.first.ctx == null and actual.prediction.hidden.ctx == null and actual.prediction.cache.a.ctx == null);
                try std.testing.expectEqual(@as(i32, -1), actual.prediction.position);
            }
        }
        for ([_]usize{ 1, 3, 4, 6, 7 }) |slot| try std.testing.expect(sparse_updates[slot].hidden.ctx == null and sparse_updates[slot].cache.a.ctx == null);
        var empty_updates = try verification.prepareBatch(&.{});
        defer for (&empty_updates) |*update| update.deinit();
        for (empty_updates) |update| try std.testing.expect(update.hidden.ctx == null and update.prediction.first.ctx == null);
        try std.testing.expectError(error.InvalidDraftRows, verification.prepareBatch(&.{.{ .slot = 0, .count = 0, .token = keeps[0].token, .predict = true }}));
        try std.testing.expectError(error.DuplicateStream, verification.prepareBatch(&.{ keeps[0], keeps[0] }));
        try std.testing.expectError(error.InvalidDraftRows, m.draftWindowTail(&scope, verification.rows, &.{-1}));
        try std.testing.expectError(error.InvalidDraftRows, m.draftWindowTail(&scope, verification.rows, &.{@intCast(total)}));
        const batch_allocator = mx.allocator;
        for ([_]usize{ 0, 1 }) |failure| {
            var failing = std.testing.FailingAllocator.init(batch_allocator, .{ .fail_index = failure, .resize_fail_index = 0 });
            {
                mx.allocator = failing.allocator();
                defer mx.allocator = batch_allocator;
                try std.testing.expectError(error.OutOfMemory, verification.prepareBatch(&keeps));
            }
            try std.testing.expect(failing.has_induced_failure);
            for (states, references) |state, reference| {
                try std.testing.expectEqual(reference.head_cache.a.ctx != null, state.head_cache.a.ctx != null);
                try std.testing.expect(state.head_prediction.first.ctx == null);
                if (reference.head_cache.a.ctx != null) {
                    try equal(&scope, reference.head_cache.a, state.head_cache.a);
                    try equal(&scope, reference.head_cache.b, state.head_cache.b);
                } else try std.testing.expect(state.head_cache.b.ctx == null);
            }
        }
        var first: usize = 0;
        var proposal_streams: [8]neural.Stream(nemotron.Model) = undefined;
        var expected_proposals: [8]@import("drafter.zig").Proposal = undefined;
        for (&states, &references, heads, lengths, &proposal_streams, &expected_proposals, 0..) |*state, *reference, head, count, *proposal_stream, *expected_proposal, i| {
            const keep = if (i % 2 == 0) count else 1;
            var cache = try reference.head_cache.clone();
            defer cache.deinit();
            _ = try m.draftStep(&scope, reference.draft_hidden, head.anchor, &cache);
            var expected_prediction = nemotron.Cache{};
            defer expected_prediction.deinit();
            for (0..count) |row| {
                const at: i32 = @intCast(first + row);
                const out = try m.draftStep(&scope, try scope.slice(input, 0, at, at + 1), following[first + row], &cache);
                try equal(&scope, out, try scope.slice(verification_hidden, 0, at, at + 1));
                const ids = try sampling.rowsMapped(&m.kernels, &scope, try m.draftHead(&scope, out), &.{state.position + @as(i32, @intCast(row)) + 2}, head.settings, m.weights.arrays.get("draft_ids"));
                defer mx.allocator.free(ids);
                try std.testing.expectEqual(ids[0], @as(i32, @intCast(mx.c.mlx_array_data_uint32(verification_firsts)[first + row])));
                if (row + 1 == keep) expected_prediction = try cache.clone();
            }
            var committed_tokens: [16]i32 = undefined;
            var rows: [16]i32 = undefined;
            committed_tokens[0] = head.anchor;
            @memcpy(committed_tokens[1..keep], following[first..][0 .. keep - 1]);
            for (rows[0..keep], 0..) |*row, j| row.* = @intCast(j);
            try neural.absorbStreams(m, &.{.{ .state = reference, .hidden = try scope.slice(input, 0, @intCast(first), @intCast(first + count)), .tokens = committed_tokens[0..keep], .rows = rows[0..keep] }});
            const next = following[first + keep - 1];
            const update = &updates[i];
            const tail: i32 = @intCast(first + keep - 1);
            try equal(&scope, try scope.slice(verification_hidden, 0, tail, tail + 1), update.prediction.hidden);
            try mx.eval(update.prediction.first);
            try std.testing.expectEqual(mx.c.mlx_array_data_uint32(verification_firsts)[@intCast(tail)], mx.c.mlx_array_data_uint32(update.prediction.first)[0]);
            try equal(&scope, reference.head_cache.a, update.cache.a);
            try equal(&scope, reference.head_cache.b, update.cache.b);
            try equal(&scope, reference.draft_hidden, update.hidden);
            try equal(&scope, expected_prediction.a, update.prediction.cache.a);
            try equal(&scope, expected_prediction.b, update.prediction.cache.b);
            update.publish(state);
            state.position += @intCast(keep);
            reference.position = state.position;
            try std.testing.expect(state.head_prediction.matches(state.position, next, head.settings));
            var snapshot = try state.clone();
            defer snapshot.deinit();
            try std.testing.expect(snapshot.head_prediction.first.ctx == null);
            proposal_stream.* = .{ .state = state, .first = next, .budget = if (i == 0) 15 else 3, .settings = head.settings };
            m.position = reference.position;
            expected_proposal.* = try neural.propose(m, reference, null, next, proposal_stream.budget, head.settings);
            first += count;
        }
        {
            var proposed = try neural.proposeStreamsLazy(m, &proposal_streams);
            defer proposed.deinit();
            var arrays: [8]A = undefined;
            try mx.evalMany(proposed.arrays(&arrays), false);
            var actual: [8]@import("drafter.zig").Proposal = undefined;
            try proposed.read(&actual);
            for (expected_proposals, actual) |expected, got| try std.testing.expectEqualSlices(i32, expected.tokens[0..expected.len], got.tokens[0..got.len]);
            for (&states, proposed.tokens[0..8], 0..) |*state, tokens_, i| {
                state.head_prediction.drafts = try mx.retain(try scope.slice(tokens_, 0, 0, @intCast(if (i % 2 == 0) proposal_streams[i].budget else 1)));
            }
        }
        for ([_]bool{ true, false }) |prefix_only| {
            var streams = proposal_streams;
            if (prefix_only) for (&streams) |*stream| {
                stream.budget = 1;
            };
            var proposed = try neural.proposeStreamsLazy(m, &streams);
            defer proposed.deinit();
            var arrays: [8]A = undefined;
            try mx.evalMany(proposed.arrays(&arrays), false);
            var actual: [8]@import("drafter.zig").Proposal = undefined;
            try proposed.read(&actual);
            for (expected_proposals, actual) |expected, got| try std.testing.expectEqualSlices(i32, expected.tokens[0..got.len], got.tokens[0..got.len]);
        }
        proposal_streams[0].first += 1;
        m.position = states[0].position;
        const changed = try neural.propose(m, &references[0], null, proposal_streams[0].first, 15, proposal_streams[0].settings);
        var fallback = try neural.proposeStreamsLazy(m, proposal_streams[0..1]);
        defer fallback.deinit();
        var arrays: [8]A = undefined;
        try mx.evalMany(fallback.arrays(&arrays), false);
        var actual: [1]@import("drafter.zig").Proposal = undefined;
        try fallback.read(&actual);
        try std.testing.expectEqualSlices(i32, changed.tokens[0..15], actual[0].tokens[0..15]);
        var discarded = try verification.prepare(0, 1, 123, false);
        defer discarded.deinit();
        try std.testing.expect(discarded.prediction.first.ctx == null);
        try std.testing.expectError(error.InvalidDraftRows, verification.prepare(0, 0, 123, true));
        try std.testing.expectError(error.DuplicateStream, neural.HeadVerification.init(m, &.{ heads[0], heads[0] }, try scope.slice(input, 0, 0, @intCast(heads[0].count * 2)), try scope.slice(tokens, 0, 0, @intCast(heads[0].count * 2))));
        const keys = states[0].head_cache.a.ctx;
        const prediction = states[0].head_prediction.first.ctx;
        var cancelled = try neural.HeadVerification.init(m, heads[0..1], try scope.slice(input, 0, 0, @intCast(heads[0].count)), try scope.slice(tokens, 0, 0, @intCast(heads[0].count)));
        cancelled.deinit();
        try std.testing.expectEqual(keys, states[0].head_cache.a.ctx);
        try std.testing.expectEqual(prediction, states[0].head_prediction.first.ctx);
        const previous_allocator = mx.allocator;
        const short_hidden = try scope.slice(input, 0, 0, @intCast(heads[0].count));
        const short_tokens = try scope.slice(tokens, 0, 0, @intCast(heads[0].count));
        for ([_]usize{ 0, 1 }) |failure| {
            var failing = std.testing.FailingAllocator.init(previous_allocator, .{ .fail_index = failure, .resize_fail_index = 0 });
            {
                mx.allocator = failing.allocator();
                defer mx.allocator = previous_allocator;
                try std.testing.expectError(error.OutOfMemory, neural.HeadVerification.init(m, heads[0..1], short_hidden, short_tokens));
            }
            try std.testing.expect(failing.has_induced_failure);
            try std.testing.expectEqual(keys, states[0].head_cache.a.ctx);
            try std.testing.expectEqual(prediction, states[0].head_prediction.first.ctx);
        }
    }
    std.debug.print("Nemotron selected MTP tails: exact isolated head rows, kept/cancelled cache prefixes, depth3/15 preview reuse, adjusted-token fallback and snapshot isolation for 1/16/ragged/128-row windows\n", .{});
}

fn run(m: *nemotron.Model, output: ?[]const u8) !void {
    if (!m.mtp) return error.MissingDraftHead;
    var scope = mx.Scope{};
    defer scope.deinit();
    var batched: [8]nemotron.Cache = @splat(.{});
    defer for (&batched) |*cache| cache.deinit();
    for (&batched, 0..) |*cache, index| try seed(&scope, cache, index);
    var isolated: [8]nemotron.Cache = @splat(.{});
    defer for (&isolated) |*cache| cache.deinit();
    for (batched, &isolated) |source, *cache| cache.* = try source.clone();
    var hidden = try initialHidden(&scope);
    var trace = std.StringHashMap(A).init(mx.allocator);
    defer trace.deinit();
    const previous_trace = m.head_trace;
    defer m.head_trace = previous_trace;
    if (output != null) m.head_trace = &trace;
    var buf: [128]u8 = undefined;
    for ([_]usize{ 8, 3, 1 }, 0..) |count, step| {
        const rows: i32 = @intCast(count);
        const input = try scope.slice(hidden, 0, 0, rows);
        var ids: [8]i32 = undefined;
        var caches: [8]*nemotron.Cache = undefined;
        var expected: [8]A = undefined;
        for (0..count) |index| {
            ids[index] = @intCast(1000 + 37 * index + 17 * step);
            caches[index] = &batched[index];
            if (output == null) {
                const row: i32 = @intCast(index);
                expected[index] = try m.draftStep(&scope, try scope.slice(input, 0, row, row + 1), ids[index], &isolated[index]);
            }
        }
        hidden = try m.draftStepStreams(&scope, input, try scope.ints(ids[0..count]), caches[0..count]);
        const logits = try m.draftHead(&scope, hidden);
        try mx.eval(logits);
        if (output) |directory| {
            try save(&scope, directory, try std.fmt.bufPrint(&buf, "hidden-{d}", .{step}), hidden);
            try save(&scope, directory, try std.fmt.bufPrint(&buf, "logits-{d}", .{step}), logits);
            var values = trace.iterator();
            while (values.next()) |entry| try save(&scope, directory, try std.fmt.bufPrint(&buf, "stage-{d}-{s}", .{ step, entry.key_ptr.* }), entry.value_ptr.*);
            trace.clearRetainingCapacity();
        }
        for (0..count) |index| {
            errdefer std.debug.print("Nemotron shared MTP mismatch at step {d}, stream {d}, initial past {d}\n", .{ step, index, past[index] });
            const row: i32 = @intCast(index);
            if (output) |directory| {
                try save(&scope, directory, try std.fmt.bufPrint(&buf, "keys-{d}-{d}", .{ step, index }), batched[index].a);
                try save(&scope, directory, try std.fmt.bufPrint(&buf, "values-{d}-{d}", .{ step, index }), batched[index].b);
            } else {
                try equal(&scope, expected[index], try scope.slice(hidden, 0, row, row + 1));
                try equal(&scope, try m.draftHead(&scope, expected[index]), try scope.slice(logits, 0, row, row + 1));
                try equal(&scope, isolated[index].a, batched[index].a);
                try equal(&scope, isolated[index].b, batched[index].b);
            }
        }
    }
    const absorption_hidden = try scope.cat(&.{ try initialHidden(&scope), try initialHidden(&scope) }, 0);
    var absorption_tokens: [16]i32 = undefined;
    for (&absorption_tokens, 0..) |*token, i| token.* = @intCast(3000 + 13 * i);
    const lengths = [_]usize{ 3, 0, 1, 2, 4, 1, 3, 2 };
    var absorption_caches: [8]*nemotron.Cache = undefined;
    var offset: i32 = 0;
    for (&absorption_caches, lengths, 0..) |*cache, count, i| {
        cache.* = &batched[i];
        const end = offset + @as(i32, @intCast(count));
        if (output == null and count > 0) _ = try m.draftStepArray(&scope, try scope.slice(absorption_hidden, 0, offset, end), try scope.ints(absorption_tokens[@intCast(offset)..@intCast(end)]), &isolated[i], false);
        offset = end;
    }
    try m.absorbDraftStreams(&scope, absorption_hidden, try scope.ints(&absorption_tokens), &lengths, &absorption_caches);
    for (batched, isolated, 0..) |actual, expected, i| {
        if (output) |directory| {
            try save(&scope, directory, try std.fmt.bufPrint(&buf, "absorbed-keys-{d}", .{i}), actual.a);
            try save(&scope, directory, try std.fmt.bufPrint(&buf, "absorbed-values-{d}", .{i}), actual.b);
        } else {
            try equal(&scope, expected.a, actual.a);
            try equal(&scope, expected.b, actual.b);
        }
    }
    if (output == null) {
        const two = try scope.cat(&.{ hidden, hidden }, 0);
        try std.testing.expectError(error.DuplicateStream, m.draftStepStreams(&scope, two, try scope.ints(&.{ 123, 124 }), &.{ &batched[0], &batched[0] }));
        try @import("neural_draft.zig").checkAbsorbStreams(m, absorption_hidden, &absorption_tokens, &batched);
        try checkWideAbsorption(m, &batched);
        try checkPrefillAbsorption(m, batched[1]);
        try checkProposals(m, &batched);
        try checkEarlyPredictions(m, &batched);
        std.debug.print("Shared Nemotron MTP: exact isolated hidden/logit/cache rows for 8, 3, 1 streams, unequal contexts and 10K attention switch\n", .{});
    }
}

pub fn check(m: *nemotron.Model) !void {
    try run(m, null);
}

pub fn oracle(io: std.Io, m: *nemotron.Model, output: []const u8) !void {
    try std.Io.Dir.cwd().createDirPath(io, output);
    try run(m, output);
}

pub fn checkOracle(io: std.Io, model_directory: []const u8, output: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var model = try nemotron.Model.init(io, model_directory, true);
    defer model.deinit();
    try oracle(io, &model, output);
}
