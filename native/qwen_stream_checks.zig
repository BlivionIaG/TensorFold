const std = @import("std");
const mx = @import("mlx.zig");
const grouped = @import("qwen_streams.zig");
const shared = @import("qwen_shared.zig");
const model = @import("model.zig");
const lanes = @import("lanes.zig");
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
    for (0..8) |index| {
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
        var p = try grouped.Layout.init(parents[0..starts.len], starts);
        var c = try grouped.Commit.init(&p, paths[0..starts.len]);
        try p.prepare(&s);
        try c.prepare(&s);
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
        const stacked_pre = try grouped.preStack(&k, &s, &p, qkv, convs[0..p.streams], try store.get("cw"), try store.get("zba"), try store.get("alog"), try store.get("dt"));
        for (pre, 0..) |v, j| {
            var name: [64]u8 = undefined;
            try equal(&s, v, try store.get(try std.fmt.bufPrint(&name, "pre{d}", .{j})));
            try @import("variant_checks.zig").equalBits(&s, stacked_pre[j], v);
        }
        const rows: i32 = @intCast(p.rows);
        const padded = @divTrunc(rows + 15, 16) * 16;
        const post = try k.run(&s, @import("kernel_sources.zig").lane_fuse_gdn_post, &.{ try store.get("y"), try store.get("zba"), try store.get("norm"), try s.scalar(1e-6), try s.ints(&.{ rows, padded }) }, &.{ mx.ti("NV", 48), mx.ti("DV", 128), mx.ti("ZS", 6240) }, .{ 32, 48, padded }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ 1, rows, 6144 } }, .{ .shape = &.{ 96, padded }, .dtype = mx.f32t } });
        try @import("variant_checks.zig").equalBits(&s, post[0], try store.get("post"));
        try @import("variant_checks.zig").equalBits(&s, post[1], try store.get("post_sums"));
        try equal(&s, try grouped.recurrence(&k, &s, &p, pre, states[0..p.streams]), try store.get("y"));
        if (p.rows == p.streams) {
            const result = try grouped.step(&k, &s, &p, pre, states[0..p.streams]);
            const exact = @import("variant_checks.zig").equalBits;
            try exact(&s, result.y, try store.get("y"));
            for (0..p.streams) |st| {
                var name: [64]u8 = undefined;
                try exact(&s, result.states[st], try store.get(try std.fmt.bufPrint(&name, "step_state{d}", .{st})));
            }
        }
        const replayed = try grouped.replay(&k, &s, &c, pre, states[0..p.streams]);
        const tails = try grouped.tails(&k, &s, &c, qkv, convs[0..p.streams]);
        for (0..p.streams) |st| {
            var name: [64]u8 = undefined;
            try equal(&s, replayed[st], try store.get(try std.fmt.bufPrint(&name, "replay{d}", .{st})));
            try equal(&s, tails[st], try store.get(try std.fmt.bufPrint(&name, "tail{d}", .{st})));
        }
        if (mx.tensor_units) {
            const query = try store.get("query");
            const expected = try store.get("attention");
            try equal(&s, try grouped.attention(&k, &s, &p, query, keys[0..p.streams], values[0..p.streams]), expected);
            const meta = p.attention_meta;
            try equal(&s, try grouped.attention(&k, &s, &p, query, keys[0..p.streams], values[0..p.streams]), expected);
            try std.testing.expectEqual(meta.ctx, p.attention_meta.ctx);
            var padded_keys = keys;
            var padded_values = values;
            padded_keys[0] = try s.contiguous(try s.cat(&.{ keys[0], try s.zeros(&.{ 1, 4, 7, 256 }, mx.bf16) }, 2));
            const last = p.streams - 1;
            padded_values[last] = try s.contiguous(try s.cat(&.{ values[last], try s.zeros(&.{ 1, 4, 9, 256 }, mx.bf16) }, 2));
            try equal(&s, try grouped.attention(&k, &s, &p, query, padded_keys[0..p.streams], padded_values[0..p.streams]), expected);
            try std.testing.expect(meta.ctx != p.attention_meta.ctx);
            try equal(&s, try grouped.attention(&k, &s, &p, query, keys[0..p.streams], values[0..p.streams]), expected);
        }
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

fn checkMlp(m: *model.Model) !void {
    for ([_]usize{ 1, 8, 17, 32, 33, 128 }) |rows| {
        var s = mx.Scope{};
        defer s.deinit();
        var cache = lanes.ProjectionCache{};
        const previous = m.projection_cache;
        m.projection_cache = &cache;
        defer m.projection_cache = previous;
        var tokens: [128]i32 = undefined;
        for (tokens[0..rows], 0..) |*token, i| token.* = @intCast(100 + 17 * i);
        const h = try m.weights.embedArray(&s, try s.ints(tokens[0..rows]));
        const normalized = try lanes.norm(&m.kernels, &s, h, null, try m.weight(0, "post_attention_layernorm.weight"));
        const gate = try m.project(&s, 0, "mlp.gate_proj", normalized.x);
        const up = try m.project(&s, 0, "mlp.up_proj", normalized.x);
        const separate = try lanes.mlp(&m.kernels, &s, gate, up);
        const fused = try m.mlpAct(&s, 0, normalized.x);
        try equal(&s, separate.x, fused.x);
        try equal(&s, separate.sums.?, fused.sums.?);
        try equal(&s, separate.dimensions.?, fused.dimensions.?);
        try equal(&s, try m.project(&s, 0, "mlp.down_proj", separate), try m.project(&s, 0, "mlp.down_proj", fused));
        if (mx.tensor_units) {
            var stacked = false;
            for (cache.outputs) |output| stacked = stacked or output != null;
            try std.testing.expectEqual(rows < 17 or rows > 32, stacked);
        }
        if (cache.dimensions.ctx != null) {
            try std.testing.expectEqual(@as(i32, @intCast(rows)), cache.rows);
            try std.testing.expectEqual(normalized.x.dimensions.?.ctx, cache.dimensions.ctx);
        }
    }
    std.debug.print("Qwen/Bonsai MLP consumers: exact activations, group sums and down projections at 1/8/17/32/33/128 rows\n", .{});
}

fn compiledPostAttempt(norm: A, linears: [3]lanes.Linear, stack: ?lanes.Linear, h: A, r: A) !void {
    var post = try model.CompiledPost.init(norm, linears, stack);
    defer post.deinit();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var s = mx.Scope{};
    defer s.deinit();
    const values = try post.call(&kernels, &s, h, r);
    try mx.evalMany(&values, false);
}

fn checkCompiledPost(m: *model.Model) !void {
    if (!mx.tensor_units or m.weights.bonsai_form != null) return;
    const exact = @import("variant_checks.zig").equalBits;
    const names = [_][]const u8{ "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj" };
    var stack = try m.weights.fused(names[0..2]);
    var linears: [3]lanes.Linear = undefined;
    for (&linears, names) |*linear, name| linear.* = try m.weights.linear(name);
    const norm = try m.weight(0, "post_attention_layernorm.weight");
    var post: model.CompiledPost = .{};
    defer post.deinit();
    {
        var lazy = mx.Scope{};
        defer lazy.deinit();
        const zero: u32 = 0;
        const z = try lazy.data(&zero, &.{}, mx.c.MLX_UINT32);
        for (&linears) |*linear| {
            try std.testing.expectEqual(@as(mx.c.mlx_dtype, mx.c.MLX_UINT32), mx.dtype(linear.weight));
            linear.weight = try lazy.binary(mx.c.mlx_add, linear.weight, z);
        }
        if (stack) |*linear| linear.weight = try lazy.binary(mx.c.mlx_add, linear.weight, z);
        post = try model.CompiledPost.init(norm, linears, stack);
    }
    // The capture owns its handles even after the source lazy graphs are released.
    for (&linears, names) |*linear, name| linear.* = try m.weights.linear(name);
    stack = try m.weights.fused(names[0..2]);
    var fixture = mx.Scope{};
    defer fixture.deinit();
    const width = linears[0].k;
    const values = try mx.allocator.alloc(f32, @as(usize, @intCast(width)) * 256);
    defer mx.allocator.free(values);
    for (values, 0..) |*value, j| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(j * 17 % 101)) - 50)) / 64;
    const input = try fixture.cast(try fixture.data(values.ptr, &.{ 1, 256, width }, mx.f32t), mx.bf16);
    try mx.eval(input);
    try mx.check(mx.c.mlx_clear_cache());
    const before = try @import("memory_runtime.zig").activeBytes();
    const cases = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 24, 31, 32, 33, 48, 64, 127, 128 };
    var largest_growth: u64 = 0;
    for (0..2) |repetition| for (cases) |rows| {
        {
            var s = mx.Scope{};
            defer s.deinit();
            var inputs = mx.Scope{};
            const actual = blk: {
                defer inputs.deinit();
                const offset: i32 = @intCast(repetition * 128);
                const h = try inputs.slice(input, 1, offset, offset + rows);
                const r = try inputs.slice(input, 1, 64, 64 + rows);
                break :blk try post.call(&m.kernels, &s, h, r);
            };
            const offset: i32 = @intCast(repetition * 128);
            const h = try s.slice(input, 1, offset, offset + rows);
            const r = try s.slice(input, 1, 64, 64 + rows);
            var cache = lanes.ProjectionCache{};
            const previous = m.projection_cache;
            m.projection_cache = &cache;
            defer m.projection_cache = previous;
            const normalized = try lanes.norm(&m.kernels, &s, h, r, norm);
            const act = try m.mlpAct(&s, 0, normalized.x);
            const expected = [_]A{ normalized.h, try m.project(&s, 0, "mlp.down_proj", act) };
            for (actual, expected) |got, want| {
                try std.testing.expectEqualSlices(i32, mx.shape(want), mx.shape(got));
                try std.testing.expectEqual(mx.dtype(want), mx.dtype(got));
                try exact(&s, got, want);
            }
        }
        try mx.check(mx.c.mlx_synchronize(mx.stream));
        try mx.check(mx.c.mlx_clear_cache());
        const growth = (try @import("memory_runtime.zig").activeBytes()) -| before;
        largest_growth = @max(largest_growth, growth);
        if (growth > 8 * 1024 * 1024) return error.CompiledWeightsDuplicated;
    };
    const h = try fixture.slice(input, 1, 0, 1);
    const r = try fixture.slice(input, 1, 128, 129);
    const previous = mx.allocator;
    var probe = std.testing.FailingAllocator.init(previous, .{ .resize_fail_index = 0 });
    {
        mx.allocator = probe.allocator();
        defer mx.allocator = previous;
        try compiledPostAttempt(norm, linears, stack, h, r);
    }
    try std.testing.expectEqual(probe.allocated_bytes, probe.freed_bytes);
    for (0..probe.alloc_index) |failure_index| {
        var failing = std.testing.FailingAllocator.init(previous, .{ .fail_index = failure_index, .resize_fail_index = 0 });
        const attempted = blk: {
            mx.allocator = failing.allocator();
            defer mx.allocator = previous;
            break :blk compiledPostAttempt(norm, linears, stack, h, r);
        };
        try std.testing.expectError(error.OutOfMemory, attempted);
        try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
    }
    try compiledPostAttempt(norm, linears, stack, h, r);
    var retained = mx.Scope{};
    defer retained.deinit();
    const escaped = try post.call(&m.kernels, &retained, h, r);
    post.deinit();
    try mx.evalMany(&escaped, false);
    std.debug.print("Qwen compiled post: exact arrays at 25 row shapes repeated with new inputs, owned captures, {d} allocation failures; retained growth={d} bytes\n", .{ probe.alloc_index, largest_growth });
}

fn checkGdnProjections(m: *model.Model) !void {
    const exact = @import("variant_checks.zig").equalBits;
    for ([_]usize{ 1, 8, 17, 32, 33, 128 }) |rows| {
        var s = mx.Scope{};
        defer s.deinit();
        var cache = lanes.ProjectionCache{};
        const previous = m.projection_cache;
        m.projection_cache = &cache;
        defer m.projection_cache = previous;
        var tokens: [128]i32 = undefined;
        for (tokens[0..rows], 0..) |*token, i| token.* = @intCast(100 + 17 * i);
        const hidden = try m.weights.embedArray(&s, try s.ints(tokens[0..rows]));
        const norm = try lanes.norm(&m.kernels, &s, hidden, null, try m.weight(0, "input_layernorm.weight"));
        const suffixes = [_][]const u8{ "linear_attn.in_proj_z", "linear_attn.in_proj_b", "linear_attn.in_proj_a" };
        var separate: [3]A = undefined;
        inline for (suffixes, 0..) |suffix, i| separate[i] = try m.project(&s, 0, suffix, norm.x);
        if (try m.projectStack(&s, 0, &suffixes, norm.x)) |stack| {
            try exact(&s, stack, try s.cat(&separate, -1));
            const repeated = (try m.projectStack(&s, 0, &suffixes, norm.x)).?;
            try std.testing.expectEqual(stack.ctx, repeated.ctx);
        } else if (mx.tensor_units and m.weights.bonsai_form == null) return error.MissingProjectionFusion;
    }
    std.debug.print("Qwen/Bonsai GDN projections: bit-exact stacked z/b/a at 1/8/17/32/33/128 rows\n", .{});
}

fn checkAttentionProjections(m: *model.Model) !void {
    for ([_]i32{ 1, 8, 17, 32, 33, 128 }) |rows| {
        var s = mx.Scope{};
        defer s.deinit();
        var cache = lanes.ProjectionCache{};
        const previous = m.projection_cache;
        m.projection_cache = &cache;
        defer m.projection_cache = previous;
        var tokens: [128]i32 = undefined;
        for (tokens[0..@intCast(rows)], 0..) |*token, i| token.* = @intCast(100 + 17 * i);
        const hidden = try m.weights.embedArray(&s, try s.ints(tokens[0..@intCast(rows)]));
        const norm = try lanes.norm(&m.kernels, &s, hidden, null, try m.weight(3, "input_layernorm.weight"));
        const qg = try s.reshape(try m.project(&s, 3, "self_attn.q_proj", norm.x), &.{ 1, rows, 24, 512 });
        const queries = try s.rms(try s.slice(qg, 3, 0, 256), try m.weight(3, "self_attn.q_norm.weight"));
        const gate = try s.reshape(try s.slice(qg, 3, 256, 512), &.{ 1, rows, 6144 });
        const keys = try s.rms(try s.reshape(try m.project(&s, 3, "self_attn.k_proj", norm.x), &.{ 1, rows, 4, 256 }), try m.weight(3, "self_attn.k_norm.weight"));
        const values = try s.reshape(try m.project(&s, 3, "self_attn.v_proj", norm.x), &.{ 1, rows, 4, 256 });
        const actual = try shared.attentionProjections(m, &s, 3, norm.x);
        const exact = @import("variant_checks.zig").equalBits;
        try exact(&s, queries, actual.queries);
        try exact(&s, gate, actual.gate);
        try exact(&s, keys, actual.keys);
        try exact(&s, values, actual.values);
    }
    std.debug.print("Qwen/Bonsai attention projections: bit-exact normalized queries/keys, values and gates at 1/8/17/32/33/128 rows\n", .{});
}

fn exercise(m: *model.Model, states: []shared.State, parents: []const []const i32, paths: []const []const i32) !void {
    for ([_]bool{ false, true }) |queued| try exerciseMode(m, states, parents, paths, true, queued);
}

fn exerciseMode(m: *model.Model, states: []shared.State, parents: []const []const i32, paths: []const []const i32, evaluate_before_commit: bool, queued: bool) !void {
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
    var shared_pass = if (queued) try shared.forwardQueued(m, streams) else try m.forwardStreams(streams);
    defer shared_pass.deinit();
    try std.testing.expectEqual(first, shared_pass.count);
    try std.testing.expectEqual(@as(i32, 0), m.position);
    for (m.cache) |cache| try std.testing.expect(cache.a.ctx == null and cache.b.ctx == null);
    try std.testing.expectError(error.ModelRoundActive, m.forwardStreams(streams));
    try std.testing.expectError(error.RequestRoundActive, states[0].clone());
    var s = mx.Scope{};
    defer s.deinit();
    for (0..streams.len) |i| {
        const context = try shared_pass.contextView(i);
        try std.testing.expectEqual(streams[i].tokens.len, context.count);
        for (context.records, 0..) |record, layer| if (layer % 4 != 3) {
            for (record.values[0..5]) |value| try std.testing.expect(value.ctx == null);
            try std.testing.expect(record.values[6].ctx == null);
        };
    }
    try std.testing.expectError(error.InvalidStreams, shared_pass.contextView(streams.len));
    try std.testing.expectError(error.InvalidStreams, shared_pass.view(streams.len));
    if (evaluate_before_commit) {
        const previous = mx.allocator;
        const owned = (try shared_pass.contextView(0)).scope.arrays.items.len;
        var materialized = false;
        for (0..32) |failure_index| {
            var failing = std.testing.FailingAllocator.init(previous, .{ .fail_index = failure_index, .resize_fail_index = 0 });
            const attempted = blk: {
                mx.allocator = failing.allocator();
                defer mx.allocator = previous;
                break :blk shared_pass.view(0);
            };
            if (attempted) |_| {
                try std.testing.expect(!failing.has_induced_failure);
                materialized = true;
                break;
            } else |err| try std.testing.expectEqual(error.OutOfMemory, err);
            try std.testing.expectEqual(failing.allocated_bytes, failing.freed_bytes);
            const context = try shared_pass.contextView(0);
            try std.testing.expectEqual(owned, context.scope.arrays.items.len);
            for (context.records, 0..) |record, layer| if (layer % 4 != 3) {
                for (record.values[0..5]) |value| try std.testing.expect(value.ctx == null);
                try std.testing.expect(record.values[6].ctx == null);
            };
        }
        try std.testing.expect(materialized);
    }
    if (evaluate_before_commit) for (standalone, 0..) |pass, i| {
        errdefer std.debug.print("Shared model mismatch stream {d}, start {d}, width {d}\n", .{ i, pass.start, pass.count });
        const view = try shared_pass.view(i);
        try equal(&s, pass.hidden, view.hidden);
        try equal(&s, pass.logits, view.logits);
        for (pass.taps, view.taps) |expected, actual| try equal(&s, expected, actual);
        try equal(&s, try s.slice(pass.records[0].values[6], 1, 0, 3), view.records[0].conv_state);
        try equal(&s, try s.slice(pass.records[0].values[6], 1, 3, 3 + @as(i32, @intCast(pass.count))), view.records[0].values[6]);
        for (pass.records[62].values[0..6], view.records[62].values[0..6]) |expected, actual| try equal(&s, expected, actual);
        for (pass.records[63].values[0..2], view.records[63].values[0..2]) |expected, actual| try equal(&s, expected, actual);
        const repeated = try shared_pass.view(i);
        try std.testing.expectEqual(view.records[62].values[0].ctx, repeated.records[62].values[0].ctx);
    };
    var committed: [64][64][2]?*anyopaque = undefined;
    var staged: [64][64][2]?*anyopaque = undefined;
    if (shared_pass.staged) |caches| for (states, caches, 0..) |state, cache, i| {
        for (state.cache, cache, &committed[i], &staged[i]) |old, candidate, *old_handles, *new_handles| {
            old_handles.* = .{ old.a.ctx, old.b.ctx };
            new_handles.* = .{ candidate.a.ctx, candidate.b.ctx };
        }
    };
    try shared_pass.commit(paths);
    if (shared_pass.staged) |caches| for (states, paths, caches, 0..) |state, path, cache, i| {
        for (state.cache, cache, committed[i], staged[i]) |actual, candidate, old_handles, new_handles| {
            const expected = if (path.len == 0) old_handles else new_handles;
            try std.testing.expectEqual(expected[0], actual.a.ctx);
            try std.testing.expectEqual(expected[1], actual.b.ctx);
            if (path.len != 0) {
                try std.testing.expect(candidate.a.ctx == null and candidate.b.ctx == null);
                for ([_]A{ actual.a, actual.b }) |value| {
                    var available = false;
                    try mx.check(mx.c._mlx_array_is_available(&available, value));
                    try std.testing.expect(available);
                }
            }
        }
    };
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
    try checkMlp(&m);
    try checkCompiledPost(&m);
    try checkGdnProjections(&m);
    try checkAttentionProjections(&m);
    var states: [9]shared.State = undefined;
    var initialized: usize = 0;
    defer for (states[0..initialized]) |*s| s.deinit();
    for (&states, 0..) |*state, i| {
        state.* = try shared.State.init(&m);
        initialized += 1;
        if (i < 3) try prefill(&m, state, ([_]usize{ 0, 63, 511 })[i], @intCast(123 + i));
    }
    states[1].rope_delta = 9;
    const singleton_parents: [9][]const i32 = @splat(&.{-1});
    const singleton_paths: [9][]const i32 = @splat(&.{0});
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
        const root_parents: [64][]const i32 = @splat(&.{-1});
        const root_paths: [64][]const i32 = @splat(&.{0});
        try exerciseMode(&m, &wide, &root_parents, &root_paths, true, true);
        // All 128 rows across eight kernel groups, with copy-on-write prefixes.
        try exercise(&m, wide[0..2], &.{ &.{-1}, &.{-1} }, &.{ &.{0}, &.{0} });
        std.debug.print("Shared maximum round: 64 streams / 128 rows and prefix-copy continuation exact\n", .{});
    }
    for ([_]bool{ false, true }) |queued| for ([_]usize{ 1, 4, 8, 9 }) |count| try exerciseMode(&m, states[0..count], singleton_parents[0..count], singleton_paths[0..count], false, queued);
    try exercise(&m, states[0..3], singleton_parents[0..3], &.{ &.{0}, &.{}, &.{0} });
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
        for ([_]bool{ false, true }) |queued| try exerciseMode(&m, &uncached, singleton_parents[0..2], &.{ &.{0}, &.{} }, false, queued);
    }
    // Abandoning a forward preserves committed state and releases every lease.
    for ([_]bool{ false, true }) |queued| {
        var before = try states[0].clone();
        defer before.deinit();
        const stream = [_]shared.Stream{.{ .state = &states[0], .tokens = &.{123}, .parents = &.{-1} }};
        var pass = if (queued) try shared.forwardQueued(&m, &stream) else try m.forwardStreams(&stream);
        pass.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        try sameCache(&scope, before, states[0]);
        var evaluated = if (queued) try shared.forwardQueued(&m, &stream) else try m.forwardStreams(&stream);
        errdefer evaluated.deinit();
        try mx.eval(evaluated.logits);
        evaluated.deinit();
        try sameCache(&scope, before, states[0]);
        var zero = if (queued) try shared.forwardQueued(&m, &stream) else try m.forwardStreams(&stream);
        errdefer zero.deinit();
        var available = false;
        try mx.check(mx.c._mlx_array_is_available(&available, zero.logits));
        try std.testing.expect(!available);
        try zero.commit(&.{&.{}});
        try mx.check(mx.c._mlx_array_is_available(&available, zero.logits));
        try std.testing.expect(!available);
        zero.deinit();
        try sameCache(&scope, before, states[0]);
        for ([_][]const i32{ &.{1}, &.{ 0, 0 } }) |invalid| {
            var failed = if (queued) try shared.forwardQueued(&m, &stream) else try m.forwardStreams(&stream);
            defer failed.deinit();
            try std.testing.expectError(error.InvalidCommit, failed.commit(&.{invalid}));
            try std.testing.expectError(error.InvalidRoundStage, failed.commit(&.{&.{0}}));
            failed.deinit();
            try sameCache(&scope, before, states[0]);
        }
        try exercise(&m, states[0..1], singleton_parents[0..1], singleton_paths[0..1]);
        try std.testing.expectError(error.DuplicateStream, m.forwardStreams(&.{ stream[0], stream[0] }));
    }
    std.debug.print("Shared Qwen/Bonsai rounds: exact hidden/logits/taps, ragged commits, continuation, cache modes, cancellation and failed settlement\n", .{});
}
