const std = @import("std");
const mx = @import("mlx.zig");
const grouped = @import("qwen_streams.zig");
const shared = @import("qwen_shared.zig");
const model = @import("model.zig");
const equal = @import("sampling_checks.zig").equal;
const Store = @import("checkpoint.zig").Store;
const A = mx.Array;

fn ints(s: *mx.Scope, value: A) ![]const i32 {
    if (mx.c.mlx_array_size(value) == 0) return &.{};
    const v = try s.contiguous(value);
    try mx.eval(v);
    const data = mx.c.mlx_array_data_int32(v);
    return data[0..mx.c.mlx_array_size(v)];
}

fn sameInts(s: *mx.Scope, store: *Store, name: []const u8, values: []const i32) !void {
    const actual = try ints(s, try store.get(name));
    try std.testing.expect(actual.len >= values.len);
    try std.testing.expectEqualSlices(i32, values, actual[0..values.len]);
}

pub fn checkKernels(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var k = mx.Kernels.init();
    defer k.deinit();
    for (0..6) |index| {
        var buf: [4096]u8 = undefined;
        const path = try std.fmt.bufPrint(&buf, "{s}/streams{d}.safetensors", .{ dir, index });
        var store = Store.init(128);
        defer store.deinit();
        try store.loadFile(io, path, "", "");
        var s = mx.Scope{};
        defer s.deinit();
        const starts = try ints(&s, try store.get("starts"));
        var parents: [8][]const i32 = undefined;
        var paths: [8][]const i32 = undefined;
        var convs: [8]A = undefined;
        var states: [8]A = undefined;
        var keys: [8]A = undefined;
        var values: [8]A = undefined;
        for (0..starts.len) |st| {
            var name: [64]u8 = undefined;
            parents[st] = try ints(&s, try store.get(try std.fmt.bufPrint(&name, "parents{d}", .{st})));
            paths[st] = try ints(&s, try store.get(try std.fmt.bufPrint(&name, "kept{d}", .{st})));
            convs[st] = try store.get(try std.fmt.bufPrint(&name, "conv{d}", .{st}));
            states[st] = try store.get(try std.fmt.bufPrint(&name, "state{d}", .{st}));
            keys[st] = try store.get(try std.fmt.bufPrint(&name, "keys{d}", .{st}));
            values[st] = try store.get(try std.fmt.bufPrint(&name, "values{d}", .{st}));
        }
        const p = try grouped.Layout.init(parents[0..starts.len], starts);
        const c = try grouped.Commit.init(&p, paths[0..starts.len]);
        try sameInts(&s, &store, "windows", p.windows[0 .. 4 * p.rows]);
        try sameInts(&s, &store, "row_stream", p.row_stream[0..p.rows]);
        try sameInts(&s, &store, "tree_meta", p.tree_meta[0 .. 2 * p.streams]);
        try sameInts(&s, &store, "commit_rows", c.rows[0..c.count]);
        try sameInts(&s, &store, "commit_meta", c.meta[0 .. 2 * p.streams]);
        try sameInts(&s, &store, "commit_tails", c.tails[0 .. 3 * p.streams]);
        try sameInts(&s, &store, "attention_base", &p.base);
        try sameInts(&s, &store, "tile_stream", p.tile_stream[0..@intCast(p.tiles)]);
        try sameInts(&s, &store, "q_rows", p.q_rows[0..@intCast(16 * p.tiles)]);
        try sameInts(&s, &store, "nodes", p.nodes[0 .. 2 * p.rows]);
        try sameInts(&s, &store, "paths", p.paths[0 .. 128 * p.rows]);
        const qkv = try store.get("qkv");
        const pre = try grouped.pre(&k, &s, &p, qkv, convs[0..p.streams], try store.get("cw"), try store.get("a"), try store.get("b"), try store.get("alog"), try store.get("dt"));
        for (pre, 0..) |v, j| {
            var name: [64]u8 = undefined;
            try equal(&s, v, try store.get(try std.fmt.bufPrint(&name, "pre{d}", .{j})));
        }
        try equal(&s, try grouped.recurrence(&k, &s, &p, pre, states[0..p.streams]), try store.get("y"));
        const replayed = try grouped.replay(&k, &s, &c, pre, states[0..p.streams]);
        const tails = try grouped.tails(&k, &s, &c, qkv, convs[0..p.streams]);
        for (0..p.streams) |st| {
            var name: [64]u8 = undefined;
            try equal(&s, replayed[st], try store.get(try std.fmt.bufPrint(&name, "replay{d}", .{st})));
            try equal(&s, tails[st], try store.get(try std.fmt.bufPrint(&name, "tail{d}", .{st})));
        }
        if (mx.tensor_units) try equal(&s, try grouped.attention(&k, &s, &p, try store.get("query"), keys[0..p.streams], values[0..p.streams]), try store.get("attention"));
        std.debug.print("Shared Qwen kernel case {d}: exact layouts, recurrence, commits{s}\n", .{ index, if (mx.tensor_units) ", tensor attention" else " (tensor attention unavailable)" });
    }
}

fn prefill(m: *model.Model, state: *shared.State, count: usize, token: i32) !void {
    if (count == 0) return;
    const prompt = try mx.allocator.alloc(i32, count);
    defer mx.allocator.free(prompt);
    @memset(prompt, token);
    state.swap(m);
    defer state.swap(m);
    var pass = try m.prefill(prompt);
    defer pass.deinit();
    const path = try mx.allocator.alloc(i32, count);
    defer mx.allocator.free(path);
    for (path, 0..) |*v, i| v.* = @intCast(i);
    try m.commit(&pass, path);
}

fn sameCache(s: *mx.Scope, expected: shared.State, actual: shared.State) !void {
    try std.testing.expectEqual(expected.position, actual.position);
    try std.testing.expectEqual(expected.rope_delta, actual.rope_delta);
    for (expected.cache, actual.cache, 0..) |a, b, i| {
        errdefer std.debug.print("Cache mismatch at Qwen layer {d}\n", .{i});
        if (a.a.ctx == null) {
            try std.testing.expect(b.a.ctx == null and b.b.ctx == null);
        } else {
            try equal(s, a.a, b.a);
            try equal(s, a.b, b.b);
        }
    }
}

fn exercise(m: *model.Model, states: []shared.State, parents: []const []const i32, paths: []const []const i32) !void {
    const a = mx.allocator;
    const refs = try a.alloc(shared.State, states.len);
    defer a.free(refs);
    var initialized: usize = 0;
    defer for (refs[0..initialized]) |*r| r.deinit();
    for (states, refs) |*state, *ref| {
        ref.* = try state.clone();
        initialized += 1;
    }
    const streams = try a.alloc(shared.Stream, states.len);
    defer a.free(streams);
    var tokens: [128]i32 = undefined;
    var first: usize = 0;
    for (states, parents, streams, 0..) |*state, rp, *stream, st| {
        for (0..rp.len) |j| tokens[first + j] = @intCast(100 + st * 17 + j);
        stream.* = .{ .state = state, .tokens = tokens[first..][0..rp.len], .parents = rp };
        first += rp.len;
    }
    // Standalone references must run before the shared pass acquires the model.
    const standalone = try a.alloc(model.Pass, states.len);
    defer a.free(standalone);
    var passed: usize = 0;
    defer for (standalone[0..passed]) |*p| p.deinit();
    for (streams, refs, standalone) |stream, *ref, *pass| {
        ref.swap(m);
        defer ref.swap(m);
        pass.* = try m.forward(stream.tokens, stream.parents);
        passed += 1;
    }
    var shared_pass = try m.forwardStreams(streams);
    defer shared_pass.deinit();
    try std.testing.expectEqual(first, shared_pass.count);
    try std.testing.expectEqual(@as(i32, 0), m.position);
    for (m.cache) |cache| try std.testing.expect(cache.a.ctx == null and cache.b.ctx == null);
    try std.testing.expectError(error.ModelRoundActive, m.forwardStreams(streams));
    try std.testing.expectError(error.RequestRoundActive, states[0].clone());
    var s = mx.Scope{};
    defer s.deinit();
    for (standalone, 0..) |pass, i| {
        errdefer std.debug.print("Shared model mismatch stream {d}, start {d}, width {d}\n", .{ i, pass.start, pass.count });
        const view = try shared_pass.view(i);
        try equal(&s, pass.hidden, view.hidden);
        try equal(&s, pass.logits, view.logits);
        for (pass.taps, view.taps) |expected, actual| try equal(&s, expected, actual);
    }
    try shared_pass.commit(paths);
    try std.testing.expectError(error.InvalidRoundStage, shared_pass.commit(paths));
    shared_pass.deinit();
    shared_pass.deinit();
    try std.testing.expectError(error.StaleRound, shared_pass.commit(paths));
    for (refs, standalone, paths, states) |*ref, *pass, path, state| {
        if (path.len > 0) {
            ref.swap(m);
            defer ref.swap(m);
            try m.commit(pass, path);
        }
        try sameCache(&s, ref.*, state);
    }
}

pub fn checkModel(io: std.Io, dir: []const u8, simd: bool) !void {
    mx.force_simd = simd;
    try mx.init();
    defer mx.shutdown();
    var m = try model.Model.init(io, dir);
    defer m.deinit();
    var states: [9]shared.State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized]) |*s| s.deinit();
    for (&states, 0..) |*state, i| {
        state.* = try shared.State.init(&m);
        initialized += 1;
        if (i < 3) try prefill(&m, state, ([_]usize{ 0, 63, 511 })[i], @intCast(123 + i));
    }
    states[1].rope_delta = 9;
    const parents = [_][]const i32{ &.{ -1, 0, 0, 2 }, &.{ -1, 0 }, &.{ -1, 0, 1, 1, 3 } };
    try exercise(&m, states[0..3], &parents, &.{ &.{ 0, 2, 3 }, &.{}, &.{ 0, 1, 3, 4 } });
    try exercise(&m, states[0..3], &.{ &.{ -1, 0 }, &.{-1}, &.{ -1, 0, 1 } }, &.{ &.{ 0, 1 }, &.{0}, &.{ 0, 1, 2 } });
    var all_parents: [9][]const i32 = @splat(&.{ -1, 0 });
    const all_paths: [9][]const i32 = @splat(&.{ 0, 1 });
    all_parents[8] = &.{ -1, 0, 0 };
    try exercise(&m, &states, &all_parents, &all_paths);
    {
        var wide: [64]shared.State = undefined;
        var made: usize = 0;
        defer for (wide[0..made]) |*state| state.deinit();
        for (&wide) |*state| {
            state.* = try states[0].clone();
            made += 1;
        }
        const wide_parents: [64][]const i32 = @splat(&.{ -1, 0 });
        const wide_paths: [64][]const i32 = @splat(&.{ 0, 1 });
        try exercise(&m, &wide, &wide_parents, &wide_paths);
        // All 128 rows across eight kernel groups, with copy-on-write prefixes.
        try exercise(&m, wide[0..2], &.{ &.{-1}, &.{-1} }, &.{ &.{0}, &.{0} });
        std.debug.print("Shared maximum round: 64 streams / 128 rows and prefix-copy continuation exact\n", .{});
    }
    {
        const cache_enabled = @import("kv_buffer.zig").enabled;
        defer @import("kv_buffer.zig").enabled = cache_enabled;
        @import("kv_buffer.zig").enabled = false;
        var uncached: [2]shared.State = undefined;
        var made: usize = 0;
        defer for (uncached[0..made]) |*state| state.deinit();
        for (&uncached) |*state| {
            state.* = try shared.State.init(&m);
            made += 1;
        }
        try exercise(&m, &uncached, &.{ &.{ -1, 0, 0 }, &.{-1} }, &.{ &.{ 0, 2 }, &.{0} });
        try exercise(&m, &uncached, &.{ &.{-1}, &.{ -1, 0 } }, &.{ &.{0}, &.{ 0, 1 } });
    }
    // Abandoning a forward preserves committed state and releases every lease.
    var before = try states[0].clone();
    defer before.deinit();
    const stream = [_]shared.Stream{.{ .state = &states[0], .tokens = &.{123}, .parents = &.{-1} }};
    var pass = try m.forwardStreams(&stream);
    pass.deinit();
    var scope = mx.Scope{};
    defer scope.deinit();
    try sameCache(&scope, before, states[0]);
    var failed = try m.forwardStreams(&stream);
    defer failed.deinit();
    try std.testing.expectError(error.InvalidCommit, failed.commit(&.{&.{1}}));
    try std.testing.expectError(error.InvalidRoundStage, failed.commit(&.{&.{0}}));
    failed.deinit();
    try sameCache(&scope, before, states[0]);
    try std.testing.expectError(error.DuplicateStream, m.forwardStreams(&.{ stream[0], stream[0] }));
    std.debug.print("Shared Qwen/Bonsai rounds: exact hidden/logits/taps, ragged commits, continuation, cache modes, cancellation and failed settlement\n", .{});
}
