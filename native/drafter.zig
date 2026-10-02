//! DFlash2: target taps -> cached context K/V -> parallel block -> best-first tree.
const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
const model = @import("model.zig");
const sampling = @import("sampling.zig");
const weights = @import("weights.zig");
const calibration = @import("draft_calibration.zig");
const A = mx.Array;
const conv_spec = @import("kernel_sources.zig").Spec{ .name = "dflash_dynamic_conv_v1", .inputs = &.{ "H", "DYNAMIC", "BASE", "dims" }, .outputs = &.{"OUT"}, .source = @embedFile("metal/dynamic_conv.metal"), .header = "", .contiguous = true };
const conv_streams_spec = @import("kernel_sources.zig").Spec{ .name = "dflash_dynamic_conv_streams_v1", .inputs = &.{ "H", "DYNAMIC", "BASE", "dims", "START" }, .outputs = &.{"OUT"}, .source = @embedFile("metal/dynamic_conv_streams.metal"), .header = "", .contiguous = true, .bake_templates = true };
pub const Proposal = struct { tokens: [31]i32 = undefined, parents: [31]i32 = undefined, scores: [31]f64 = undefined, probabilities: [31]f64 = undefined, len: usize = 0 };
pub const Stream = struct {
    state: *@import("request_state.zig").State(model.Model),
    anchor: i32,
    budget: usize,
    settings: sampling.Sampling,
};
pub const AbsorbStream = struct {
    state: *@import("request_state.zig").State(model.Model),
    pass: *model.Pass,
    rows: []const i32,
    tokens: []const i32,
};
const BlockPart = enum { pre, post };
const AttentionMasks = struct {
    const Entry = struct { rows: i32, context: i32, value: A };
    entries: [40]Entry = undefined,
    count: usize = 0,

    fn get(m: *AttentionMasks, s: *mx.Scope, rows: i32, context: i32) !A {
        for (m.entries[0..m.count]) |entry| if (entry.rows == rows and entry.context == context) return entry.value;
        const value = try attentionMask(s, rows, context);
        if (m.count < m.entries.len) {
            m.entries[m.count] = .{ .rows = rows, .context = context, .value = value };
            m.count += 1;
        }
        return value;
    }
};

fn attentionMask(s: *mx.Scope, rows: i32, context: i32) !A {
    const total = context + rows;
    const mask = try mx.allocator.alloc(u8, @intCast(rows * total));
    defer mx.allocator.free(mask);
    for (0..@intCast(rows)) |row| for (0..@intCast(total)) |col| {
        mask[row * @as(usize, @intCast(total)) + col] = @intFromBool(col >= context or @as(i32, @intCast(row)) + context - @as(i32, @intCast(col)) < 2048);
    };
    return s.data(mask.ptr, &.{ rows, total }, mx.c.MLX_BOOL);
}

fn projectionInput(k: *mx.Kernels, s: *mx.Scope, x: A) !lanes.Act {
    if (!mx.tensor_units) return .{ .x = x };
    const width = mx.dim(x, -1);
    const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(width)));
    const padded = @divTrunc(rows + 15, 16) * 16;
    const groups = @divExact(width, 64);
    const dims = try s.ints(&.{ rows, padded });
    const sums = (try k.run(s, @import("kernel_sources.zig").lane_qmm_xsum, &.{ try s.reshape(x, &.{ rows, width }), dims }, &.{ mx.ti("K", width), mx.ti("GS", 64) }, .{ groups, padded, 1 }, .{ @min(groups, 256), 1, 1 }, &.{.{ .shape = &.{ groups, padded }, .dtype = mx.f32t }}))[0];
    return .{ .x = x, .sums = sums, .dimensions = dims };
}

const CompiledPart = struct {
    closure: mx.c.mlx_closure = .{ .ctx = null },
    payload: ?*Payload = null,
    const linear_arrays = .{ "weight", "sb", "scales", "biases", "signs" };
    const max_weights = 3 + 4 * linear_arrays.len;

    const Payload = struct {
        kind: BlockPart,
        kernels: mx.Kernels,
        norm: A,
        attention_base: A,
        mlp_base: A,
        dynamic: lanes.Linear,
        gate: lanes.Linear,
        up: lanes.Linear,
        down: lanes.Linear,
        failure: ?anyerror = null,

        fn weights(p: *const Payload, output: []A) usize {
            output[0] = p.norm;
            output[1] = p.attention_base;
            var count: usize = 2;
            if (p.kind == .post) {
                output[count] = p.mlp_base;
                count += 1;
            }
            const linears = [_]lanes.Linear{ p.dynamic, p.gate, p.up, p.down };
            for (linears[0..if (p.kind == .pre) @as(usize, 1) else 4]) |linear| inline for (linear_arrays) |field| {
                const value = @field(linear, field);
                if (value.ctx != null) {
                    output[count] = value;
                    count += 1;
                }
            };
            return count;
        }

        fn destroy(raw: ?*anyopaque) callconv(.c) void {
            const p: *Payload = @ptrCast(@alignCast(raw.?));
            p.kernels.deinit();
            mx.allocator.destroy(p);
        }
        fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
            const p: *Payload = @ptrCast(@alignCast(raw.?));
            return p.graph(out, ins) catch |err| {
                p.failure = err;
                return -1;
            };
        }
        fn graph(p: *Payload, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
            var s = mx.Scope{};
            defer s.deinit();
            const arity: usize = if (p.kind == .pre) 1 else 3;
            const count = mx.c.mlx_vector_array_size(ins);
            if (count < arity or count > arity + 8) return error.InvalidDraftStreams;
            var args: [11]A = undefined;
            for (args[0..count], 0..) |*arg, j| {
                var value = mx.c.mlx_array_new();
                const rc = mx.c.mlx_vector_array_get(&value, ins, j);
                arg.* = try s.result(rc, value);
            }
            var lengths: [8]i32 = undefined;
            const blocks = if (count == arity) @as(usize, 1) else count - arity;
            if (count == arity) {
                lengths[0] = mx.dim(args[0], 1);
            } else {
                // Shape-only views specialize ragged convolutions without capturing request state.
                for (args[arity..count], lengths[0..blocks]) |view, *length| length.* = mx.dim(view, 1);
            }
            var total: i32 = 0;
            for (lengths[0..blocks]) |length| {
                if (length < 1 or length > 16) return error.InvalidDraftBlock;
                total += length;
            }
            if (total != mx.dim(args[0], 1)) return error.InvalidDraftBlock;
            const starts = try blockStarts(&s, lengths[0..blocks]);
            if (p.kind == .pre) {
                const x = try s.rms(args[0], p.norm);
                const dynamic = try p.dynamic.apply(&p.kernels, &s, .{ .x = x });
                const values = [_]A{ try p.convolve(&s, x, dynamic, p.attention_base, 0, total, starts), dynamic };
                return mx.c.mlx_vector_array_set_data(out, &values, values.len);
            }
            const h = try s.binary(mx.c.mlx_add, args[0], try p.convolve(&s, args[1], args[2], p.attention_base, 1, total, starts));
            const normed = try s.rms(h, p.norm);
            const dynamic = try p.dynamic.apply(&p.kernels, &s, .{ .x = normed });
            const x = try p.convolve(&s, normed, dynamic, p.mlp_base, 0, total, starts);
            const gate = try p.gate.apply(&p.kernels, &s, .{ .x = x });
            const up = try p.up.apply(&p.kernels, &s, .{ .x = x });
            const act = try s.binary(mx.c.mlx_multiply, try s.binary(mx.c.mlx_multiply, gate, try s.unary(mx.c.mlx_sigmoid, gate)), up);
            const down = try p.down.apply(&p.kernels, &s, .{ .x = act });
            const value = try s.binary(mx.c.mlx_add, h, try p.convolve(&s, down, dynamic, p.mlp_base, 1, total, starts));
            return mx.c.mlx_vector_array_set_data(out, &value, 1);
        }
        fn blockStarts(s: *mx.Scope, lengths: []const i32) !?A {
            if (lengths.len == 1) return null;
            var starts: [128]i32 = undefined;
            var at: usize = 0;
            for (lengths) |length| {
                const count: usize = @intCast(length);
                @memset(starts[at..][0..count], @intCast(at));
                at += count;
            }
            return try s.ints(starts[0..at]);
        }
        fn convolve(p: *Payload, s: *mx.Scope, h: A, dynamic: A, base: A, part: i32, rows: i32, starts: ?A) !A {
            const inputs = [_]A{ h, dynamic, base, try s.ints(&.{rows}), starts orelse mx.empty };
            return (try p.kernels.run(s, if (starts != null) conv_streams_spec else conv_spec, inputs[0..if (starts != null) @as(usize, 5) else 4], &.{ mx.ti("N", 5120), mx.ti("PART", part) }, .{ rows * 5120, 1, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ 1, rows, 5120 } }}))[0];
        }
    };

    fn init(d: *Drafter, layer: usize, kind: BlockPart) !CompiledPart {
        return create(.{
            .kind = kind,
            .kernels = mx.Kernels.init(),
            .norm = try d.get(layer, if (kind == .pre) "input_layernorm.weight" else "post_attention_layernorm.weight"),
            .attention_base = try d.get(layer, "attention_conv.base_kernel"),
            .mlp_base = try d.get(layer, "mlp_conv.base_kernel"),
            .dynamic = try d.linear(layer, if (kind == .pre) "attention_conv.kernel_projection" else "mlp_conv.kernel_projection"),
            .gate = try d.linear(layer, "mlp.gate_proj"),
            .up = try d.linear(layer, "mlp.up_proj"),
            .down = try d.linear(layer, "mlp.down_proj"),
        });
    }
    fn create(value: Payload) !CompiledPart {
        var arrays: [max_weights]A = undefined;
        const count = value.weights(&arrays);
        // MLX clones unevaluated captured graphs for every compiled shape.
        try mx.evalMany(arrays[0..count], false);
        const payload = try mx.allocator.create(Payload);
        payload.* = value;
        const fun = mx.c.mlx_closure_new_func_payload(Payload.callback, payload, Payload.destroy);
        if (fun.ctx == null) {
            Payload.destroy(payload);
            return error.MlxFailure;
        }
        defer _ = mx.c.mlx_closure_free(fun);
        var closure = mx.c.mlx_closure{ .ctx = null };
        errdefer if (closure.ctx != null) {
            _ = mx.c.mlx_closure_free(closure);
        };
        try mx.check(mx.c.mlx_compile(&closure, fun, false));
        return .{ .closure = closure, .payload = payload };
    }
    fn deinit(p: *CompiledPart) void {
        if (p.closure.ctx != null) _ = mx.c.mlx_closure_free(p.closure);
        p.* = .{};
    }
    fn call(p: *CompiledPart, s: *mx.Scope, args: []const A, layout: []const A, output: []A) !void {
        return p.apply(s, args, layout, output, true);
    }
    fn apply(p: *CompiledPart, s: *mx.Scope, args: []const A, layout: []const A, output: []A, compiled: bool) !void {
        if (args.len != (if (p.payload.?.kind == .pre) @as(usize, 1) else 3) or layout.len > 8) return error.InvalidDraftStreams;
        var inputs: [11]A = undefined;
        @memcpy(inputs[0..args.len], args);
        @memcpy(inputs[args.len..][0..layout.len], layout);
        const ins = mx.c.mlx_vector_array_new_data(&inputs, args.len + layout.len);
        defer _ = mx.c.mlx_vector_array_free(ins);
        var outs = mx.c.mlx_vector_array_new();
        defer _ = mx.c.mlx_vector_array_free(outs);
        p.payload.?.failure = null;
        const rc = if (compiled) mx.c.mlx_closure_apply(&outs, p.closure, ins) else try p.payload.?.graph(&outs, ins);
        if (p.payload.?.failure) |err| return err;
        try mx.check(rc);
        if (mx.c.mlx_vector_array_size(outs) != output.len) return error.InvalidDraftBlock;
        for (output, 0..) |*value, j| {
            var array = mx.c.mlx_array_new();
            const result = mx.c.mlx_vector_array_get(&array, outs, j);
            value.* = try s.result(result, array);
        }
    }
};
pub const Drafter = struct {
    weights: weights.Weights,
    parts: [5][2]CompiledPart = @splat(@splat(.{})),
    cache: [5]model.Cache = @splat(.{}),
    offset: i32 = 0,
    head: lanes.Linear,
    head_checked: bool = false,
    full_head: bool = false,
    vocabulary_tail: i32 = 248032,
    pred: A,
    succ: A,
    calibration_tables: std.json.Parsed(calibration.File),
    capture: ?*@import("draft_capture.zig").Writer = null,
    pub fn init(io: std.Io, dir: []const u8, target: *model.Model) !Drafter {
        var w = weights.Weights.init();
        errdefer w.deinit();
        try w.loadDraft(io, dir);
        var s = mx.Scope{};
        defer s.deinit();
        const head = try target.weights.linear("lm_head");
        const vocabulary_tail = 248032 - @mod(@as(i32, 248032), if (head.tiled) head.tile_width else 1);
        var selected = try head.selectRanges(&s, &.{ .{ 0, 98304 }, .{ vocabulary_tail, 248320 } });
        errdefer selected.deinit();
        const pred = try s.cast(try w.get("candidate_selector.predecessor_codebook"), mx.f32t);
        const succ = try s.cast(try w.get("candidate_selector.successor_codebook"), mx.f32t);
        try mx.evalMany(&.{ pred, succ }, false);
        const pown = try mx.retain(pred);
        errdefer mx.free(pown);
        const sown = try mx.retain(succ);
        errdefer mx.free(sown);
        w.releaseArray("candidate_selector.predecessor_codebook");
        w.releaseArray("candidate_selector.successor_codebook");
        const tables = try calibration.parse(mx.allocator, @import("native_runtime").dflash_calibration);
        errdefer tables.deinit();
        return .{ .weights = w, .head = selected, .vocabulary_tail = vocabulary_tail, .pred = pown, .succ = sown, .calibration_tables = tables };
    }
    pub fn reset(d: *Drafter) void {
        if (d.capture) |writer| writer.deinit();
        d.capture = null;
        for (&d.cache) |*v| v.deinit();
        d.offset = 0;
    }
    pub fn deinit(d: *Drafter) void {
        d.reset();
        for (&d.parts) |*parts| for (parts) |*part| part.deinit();
        d.weights.deinit();
        d.head.deinit();
        mx.free(d.pred);
        mx.free(d.succ);
        d.calibration_tables.deinit();
    }
    pub fn loadCalibration(d: *Drafter, io: std.Io, path: []const u8) !void {
        const tables = try calibration.load(mx.allocator, io, path);
        d.calibration_tables.deinit();
        d.calibration_tables = tables;
    }
    fn get(d: *Drafter, i: usize, suffix: []const u8) !A {
        var buf: [160]u8 = undefined;
        return d.weights.get(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, suffix }));
    }
    fn project(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, suffix: []const u8, x: A) !A {
        return (try d.linear(i, suffix)).apply(k, s, .{ .x = x });
    }
    fn projectKV(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, input: lanes.Act) ![2]A {
        var names_buf: [2][160]u8 = undefined;
        const names = [_][]const u8{
            try std.fmt.bufPrint(&names_buf[0], "layers.{d}.self_attn.k_proj", .{i}),
            try std.fmt.bufPrint(&names_buf[1], "layers.{d}.self_attn.v_proj", .{i}),
        };
        if (try d.weights.fused(&names)) |stack| {
            const value = try stack.apply(k, s, input);
            return .{ try s.slice(value, 2, 0, 1024), try s.slice(value, 2, 1024, 2048) };
        }
        return .{ try (try d.weights.linear(names[0])).apply(k, s, input), try (try d.weights.linear(names[1])).apply(k, s, input) };
    }
    fn linear(d: *Drafter, i: usize, suffix: []const u8) !lanes.Linear {
        var buf: [160]u8 = undefined;
        return d.weights.linear(try std.fmt.bufPrint(&buf, "layers.{d}.{s}", .{ i, suffix }));
    }
    fn blockPart(d: *Drafter, s: *mx.Scope, layer: usize, kind: BlockPart, args: []const A, layout: []const A, output: []A) !void {
        const part = &d.parts[layer][@backingInt(kind)];
        if (part.closure.ctx == null) part.* = try CompiledPart.init(d, layer, kind);
        try part.call(s, args, layout, output);
    }
    /// Only committed target rows enter the drafter cache; rejected siblings never do.
    pub fn absorb(d: *Drafter, target: *model.Model, p: *model.Pass, rows: []const i32, tokens: []const i32) !void {
        const s = &p.scope;
        const k = &target.kernels;
        const ids = try s.ints(rows);
        var taps: [5]A = undefined;
        for (p.taps, 0..) |v, i| taps[i] = try s.take(v, ids, 1);
        const ctx = try s.rms(try (try d.weights.linear("fc")).apply(k, s, .{ .x = try s.cat(&taps, -1) }), try d.weights.get("hidden_norm.weight"));
        const count: i32 = @intCast(rows.len);
        var positions: [128]i32 = undefined;
        for (rows, 0..) |_, j| positions[j] = d.offset + @as(i32, @intCast(j));
        const pos = try s.ints(positions[0..rows.len]);
        var next: [5]model.Cache = @splat(.{});
        errdefer for (&next) |*v| v.deinit();
        for (0..5) |i| {
            var keys = try s.rms(try s.reshape(try d.project(k, s, i, "self_attn.k_proj", ctx), &.{ 1, count, 8, 128 }), try d.get(i, "self_attn.k_norm.weight"));
            keys = try s.transpose(try s.rope(try s.transpose(keys, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
            var values = try s.transpose(try s.reshape(try d.project(k, s, i, "self_attn.v_proj", ctx), &.{ 1, count, 8, 128 }), &.{ 0, 2, 1, 3 });
            if (d.cache[i].a.ctx != null) {
                keys = try s.cat(&.{ d.cache[i].a, keys }, 2);
                values = try s.cat(&.{ d.cache[i].b, values }, 2);
            }
            const n = mx.dim(keys, 2);
            if (n > 2047) {
                keys = try s.slice(keys, 2, n - 2047, n);
                values = try s.slice(values, 2, n - 2047, n);
            }
            next[i].a = try mx.retain(try s.contiguous(keys));
            next[i].b = try mx.retain(try s.contiguous(values));
        }
        var arrays: [10]A = undefined;
        for (next, 0..) |v, i| {
            arrays[i * 2] = v.a;
            arrays[i * 2 + 1] = v.b;
        }
        try mx.evalMany(&arrays, false);
        for (&d.cache) |*v| v.deinit();
        d.cache = next;
        if (d.capture) |writer| if (!writer.failed.load(.acquire)) {
            d.captureContext(s, ctx, rows, tokens, d.offset) catch |err| writer.disable(err);
        };
        d.offset += count;
    }
    pub fn absorbStreams(d: *Drafter, target: *model.Model, streams: []const AbsorbStream) !void {
        if (streams.len > 8) return error.InvalidDraftStreams;
        var total: usize = 0;
        for (streams, 0..) |stream, j| {
            if (stream.state.borrowed) return error.RequestRoundActive;
            for (streams[0..j]) |previous| if (stream.state == previous.state) return error.DuplicateDraftStream;
            if (stream.rows.len > 128 or stream.state.dflash_offset < 0 or stream.state.dflash_offset > std.math.maxInt(i32) - @as(i32, @intCast(stream.rows.len))) return error.InvalidDraftStream;
            if (stream.rows.len == 0) continue;
            for (stream.rows, 0..) |row, r| {
                if (row < 0 or row >= stream.tokens.len or row >= stream.pass.count) return error.InvalidDraftRows;
                for (stream.rows[0..r]) |earlier| if (earlier == row) return error.InvalidDraftRows;
            }
            for (stream.pass.taps) |tap| {
                if (tap.ctx == null or mx.shape(tap).len != 3 or mx.dim(tap, 0) != 1 or mx.dim(tap, 1) != stream.pass.count or mx.dim(tap, 2) != 5120 or mx.dtype(tap) != mx.bf16) return error.InvalidDraftTaps;
            }
            for (stream.state.dflash_cache) |cache| {
                if ((cache.a.ctx == null) != (cache.b.ctx == null)) return error.InvalidDraftCache;
                if (cache.a.ctx != null and (mx.shape(cache.a).len != 4 or !std.mem.eql(i32, mx.shape(cache.a), mx.shape(cache.b)) or mx.dim(cache.a, 0) != 1 or mx.dim(cache.a, 1) != 8 or mx.dim(cache.a, 2) < 1 or mx.dim(cache.a, 2) > 2047 or mx.dim(cache.a, 3) != 128 or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16)) return error.InvalidDraftCache;
            }
            total += stream.rows.len;
            if (total > 128) return error.InvalidDraftStreams;
        }
        if (total == 0) return;
        var scope = mx.Scope{};
        defer scope.deinit();
        var inputs: [8]A = undefined;
        var count: usize = 0;
        for (streams) |stream| if (stream.rows.len > 0) {
            const ids = try scope.ints(stream.rows);
            var taps: [5]A = undefined;
            for (stream.pass.taps, &taps) |tap, *selected| selected.* = try scope.take(tap, ids, 1);
            inputs[count] = try scope.cat(&taps, -1);
            count += 1;
        };
        const k = &target.kernels;
        const ctx = try scope.rms(try (try d.weights.linear("fc")).apply(k, &scope, .{ .x = try scope.cat(inputs[0..count], 1) }), try d.weights.get("hidden_norm.weight"));
        const ctx_input = try projectionInput(k, &scope, ctx);
        var next: [8][5]model.Cache = @splat(@splat(.{}));
        defer for (&next) |*caches| {
            for (caches) |*cache| cache.deinit();
        };
        for (0..5) |layer| {
            const projected = try d.projectKV(k, &scope, layer, ctx_input);
            var at: i32 = 0;
            for (streams, 0..) |stream, j| {
                if (stream.rows.len == 0) continue;
                const n: i32 = @intCast(stream.rows.len);
                var positions: [128]i32 = undefined;
                for (positions[0..stream.rows.len], 0..) |*position, row| position.* = stream.state.dflash_offset + @as(i32, @intCast(row));
                var keys = try scope.rms(try scope.reshape(try scope.slice(projected[0], 1, at, at + n), &.{ 1, n, 8, 128 }), try d.get(layer, "self_attn.k_norm.weight"));
                keys = try scope.transpose(try scope.rope(try scope.transpose(keys, &.{ 1, 2, 0, 3 }), try scope.ints(positions[0..stream.rows.len]), 128), &.{ 2, 1, 0, 3 });
                var values = try scope.transpose(try scope.reshape(try scope.slice(projected[1], 1, at, at + n), &.{ 1, n, 8, 128 }), &.{ 0, 2, 1, 3 });
                const old = stream.state.dflash_cache[layer];
                if (old.a.ctx != null) {
                    keys = try scope.cat(&.{ old.a, keys }, 2);
                    values = try scope.cat(&.{ old.b, values }, 2);
                }
                const length = mx.dim(keys, 2);
                if (length > 2047) {
                    keys = try scope.slice(keys, 2, length - 2047, length);
                    values = try scope.slice(values, 2, length - 2047, length);
                }
                next[j][layer].a = try mx.retain(try scope.contiguous(keys));
                next[j][layer].b = try mx.retain(try scope.contiguous(values));
                at += n;
            }
        }
        var arrays: [80]A = undefined;
        var n_arrays: usize = 0;
        for (streams, next[0..streams.len]) |stream, caches| if (stream.rows.len > 0) {
            for (caches) |cache| {
                arrays[n_arrays] = cache.a;
                arrays[n_arrays + 1] = cache.b;
                n_arrays += 2;
            }
        };
        try mx.evalMany(arrays[0..n_arrays], true);
        var at: i32 = 0;
        for (streams, next[0..streams.len]) |stream, *caches| {
            if (stream.rows.len == 0) continue;
            const n: i32 = @intCast(stream.rows.len);
            if (d.capture) |writer| if (!writer.failed.load(.acquire)) {
                d.captureStream(&scope, ctx, at, at + n, stream) catch |err| writer.disable(err);
            };
            for (&stream.state.dflash_cache) |*cache| cache.deinit();
            stream.state.dflash_cache = caches.*;
            caches.* = @splat(.{});
            stream.state.dflash_offset += n;
            at += n;
        }
    }
    fn captureStream(d: *Drafter, s: *mx.Scope, ctx: A, start: i32, end: i32, stream: AbsorbStream) !void {
        try d.captureContext(s, try s.slice(ctx, 1, start, end), stream.rows, stream.tokens, stream.state.dflash_offset);
    }
    fn captureContext(d: *Drafter, s: *mx.Scope, ctx: A, rows: []const i32, tokens: []const i32, offset: i32) !void {
        var selected: [128]i32 = undefined;
        if (rows.len > selected.len) return error.InvalidCaptureShape;
        for (rows, 0..) |row, i| {
            if (row < 0 or row >= tokens.len) return error.InvalidCaptureShape;
            selected[i] = tokens[@intCast(row)];
        }
        const x = try s.contiguous(try s.cast(ctx, mx.bf16));
        var view = mx.c.mlx_array_new();
        const rc = mx.c.mlx_view(&view, x, mx.c.MLX_UINT16, mx.stream);
        view = try s.result(rc, view);
        try mx.eval(view);
        const width: usize = @intCast(mx.dim(ctx, -1));
        try d.capture.?.recordContext(offset, width, selected[0..rows.len], mx.c.mlx_array_data_uint16(view)[0 .. rows.len * width]);
    }
    pub fn captureTarget(d: *Drafter, target: *model.Model, p: *model.Pass, rows: []const i32, positions: []const i32) void {
        const writer = d.capture orelse return;
        if (writer.failed.load(.acquire)) return;
        d.captureTargetRows(target, p, rows, positions) catch |err| writer.disable(err);
    }
    fn captureTargetRows(d: *Drafter, target: *model.Model, p: *model.Pass, rows: []const i32, positions: []const i32) !void {
        if (rows.len == 0) return;
        var selected: [32]i64 = undefined;
        if (rows.len > selected.len) return error.InvalidCaptureShape;
        for (rows, 0..) |row, i| {
            if (row < 0 or row >= positions.len) return error.InvalidCaptureShape;
            selected[i] = positions[@intCast(row)];
        }
        const s = &p.scope;
        const logits = try s.take(p.logits, try s.ints(rows), 1);
        const ranked = try @import("gpu_sampling.zig").topk(&target.kernels, s, logits, 16);
        try mx.evalMany(&ranked, false);
        try d.capture.?.recordTarget(selected[0..rows.len], 16, mx.c.mlx_array_data_int32(ranked[0])[0 .. rows.len * 16], mx.c.mlx_array_data_float32(ranked[1])[0 .. rows.len * 16]);
    }
    fn attention(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, x: A) !A {
        const q = try d.project(k, s, i, "self_attn.q_proj", x);
        const keys = try d.project(k, s, i, "self_attn.k_proj", x);
        const values = try d.project(k, s, i, "self_attn.v_proj", x);
        return d.project(k, s, i, "self_attn.o_proj", try d.attend(s, i, d.cache[i], d.offset, q, keys, values, null));
    }
    fn attend(d: *Drafter, s: *mx.Scope, i: usize, cache: model.Cache, offset: i32, query: A, key: A, value: A, masks: ?*AttentionMasks) !A {
        const n = mx.dim(query, 1);
        var positions: [16]i32 = undefined;
        for (0..@intCast(n)) |j| positions[j] = offset + @as(i32, @intCast(j));
        const pos = try s.ints(positions[0..@intCast(n)]);
        var q = try s.rms(try s.reshape(query, &.{ 1, n, 32, 128 }), try d.get(i, "self_attn.q_norm.weight"));
        q = try s.transpose(try s.rope(try s.transpose(q, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        var keys = try s.rms(try s.reshape(key, &.{ 1, n, 8, 128 }), try d.get(i, "self_attn.k_norm.weight"));
        keys = try s.transpose(try s.rope(try s.transpose(keys, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const values = try s.transpose(try s.reshape(value, &.{ 1, n, 8, 128 }), &.{ 0, 2, 1, 3 });
        const ctx = mx.dim(cache.a, 2);
        const mask_array = if (masks) |shared| try shared.get(s, n, ctx) else try attentionMask(s, n, ctx);
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, q, try s.cat(&.{ cache.a, keys }, 2), try s.cat(&.{ cache.b, values }, 2), 0.08838834764831845, "", mask_array, mx.empty, false, mx.stream);
        out = try s.result(rc, out);
        return s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, n, 4096 });
    }
    const Lattice = struct { layers: [5]A, candidates: A, scores: A, projection: A };
    fn attendNormalized(s: *mx.Scope, cache: model.Cache, offset: i32, query: A, key: A, value: A, masks: *AttentionMasks) !A {
        const n = mx.dim(query, 1);
        var positions: [16]i32 = undefined;
        for (0..@intCast(n)) |j| positions[j] = offset + @as(i32, @intCast(j));
        const pos = try s.ints(positions[0..@intCast(n)]);
        const q = try s.transpose(try s.rope(try s.transpose(query, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const keys = try s.transpose(try s.rope(try s.transpose(key, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const values = try s.transpose(value, &.{ 0, 2, 1, 3 });
        const mask = try masks.get(s, n, mx.dim(cache.a, 2));
        var out = mx.c.mlx_array_new();
        const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, q, try s.cat(&.{ cache.a, keys }, 2), try s.cat(&.{ cache.b, values }, 2), 0.08838834764831845, "", mask, mx.empty, false, mx.stream);
        out = try s.result(rc, out);
        return s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, n, 4096 });
    }
    fn attentionStreams(d: *Drafter, k: *mx.Kernels, s: *mx.Scope, i: usize, x: A, streams: []const Stream, masks: *AttentionMasks) !A {
        const input = try projectionInput(k, s, x);
        const rows = mx.dim(x, 1);
        const q = try s.rms(try s.reshape(try (try d.linear(i, "self_attn.q_proj")).apply(k, s, input), &.{ 1, rows, 32, 128 }), try d.get(i, "self_attn.q_norm.weight"));
        const projected = try d.projectKV(k, s, i, input);
        const keys = try s.rms(try s.reshape(projected[0], &.{ 1, rows, 8, 128 }), try d.get(i, "self_attn.k_norm.weight"));
        const values = try s.reshape(projected[1], &.{ 1, rows, 8, 128 });
        var positions: [128]i32 = undefined;
        var position_count: usize = 0;
        for (streams) |stream| {
            const n = @as(usize, @min(15, stream.budget)) + 1;
            for (positions[position_count..][0..n], 0..) |*position, j| position.* = stream.state.dflash_offset + @as(i32, @intCast(j));
            position_count += n;
        }
        const pos = try s.ints(positions[0..position_count]);
        const rotated_q = try s.transpose(try s.rope(try s.transpose(q, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const rotated_keys = try s.transpose(try s.rope(try s.transpose(keys, &.{ 1, 2, 0, 3 }), pos, 128), &.{ 2, 1, 0, 3 });
        const all_values = try s.transpose(values, &.{ 0, 2, 1, 3 });
        var outputs: [8]A = undefined;
        var at: i32 = 0;
        for (streams, 0..) |stream, j| {
            const n: i32 = @as(i32, @intCast(@min(15, stream.budget))) + 1;
            const cache = stream.state.dflash_cache[i];
            const query = try s.slice(rotated_q, 2, at, at + n);
            const key = try s.cat(&.{ cache.a, try s.slice(rotated_keys, 2, at, at + n) }, 2);
            const value = try s.cat(&.{ cache.b, try s.slice(all_values, 2, at, at + n) }, 2);
            const mask = try masks.get(s, n, mx.dim(cache.a, 2));
            var out = mx.c.mlx_array_new();
            const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, query, key, value, 0.08838834764831845, "", mask, mx.empty, false, mx.stream);
            out = try s.result(rc, out);
            outputs[j] = try s.reshape(try s.transpose(out, &.{ 0, 2, 1, 3 }), &.{ 1, n, 4096 });
            at += n;
        }
        return d.project(k, s, i, "self_attn.o_proj", try s.cat(outputs[0..streams.len], 1));
    }
    fn latticeStreams(d: *Drafter, target: *model.Model, s: *mx.Scope, streams: []const Stream) !Lattice {
        var ids: [128]i32 = @splat(248070);
        var count: usize = 0;
        for (streams) |stream| {
            ids[count] = stream.anchor;
            count += @as(usize, @min(15, stream.budget)) + 1;
        }
        const k = &target.kernels;
        var h = try target.weights.embed(s, ids[0..count]);
        var layout: [8]A = undefined;
        var start: i32 = 0;
        for (streams, 0..) |stream, j| {
            const n: i32 = @as(i32, @intCast(@min(15, stream.budget))) + 1;
            layout[j] = try s.slice(h, 1, start, start + n);
            start += n;
        }
        var layers: [5]A = undefined;
        var masks = AttentionMasks{};
        for (0..5) |i| {
            var pre: [2]A = undefined;
            try d.blockPart(s, i, .pre, &.{h}, layout[0..streams.len], &pre);
            const attended = try d.attentionStreams(k, s, i, pre[0], streams, &masks);
            var post: [1]A = undefined;
            try d.blockPart(s, i, .post, &.{ h, attended, pre[1] }, layout[0..streams.len], &post);
            h = post[0];
            layers[i] = h;
            if (i == 0 or i == 2) try mx.evalMany(&.{h}, true);
        }
        var hidden_rows: [8]A = undefined;
        var at: i32 = 0;
        for (streams, 0..) |stream, j| {
            const n: i32 = @as(i32, @intCast(@min(15, stream.budget))) + 1;
            hidden_rows[j] = try s.slice(h, 1, at + 1, at + n);
            at += n;
        }
        const hidden = try s.rms(try s.cat(hidden_rows[0..streams.len], 1), try d.weights.get("norm.weight"));
        const input = try projectionInput(k, s, hidden);
        const logits = try d.candidateLogits(target, s, input);
        const ranked = try @import("gpu_sampling.zig").topk(k, s, logits, 16);
        const projection = try s.cast(try (try d.weights.linear("candidate_selector.hidden_projection")).apply(k, s, input), mx.f32t);
        return .{ .layers = layers, .candidates = ranked[0], .scores = ranked[1], .projection = projection };
    }
    pub fn proposeStreams(d: *Drafter, target: *model.Model, streams: []const Stream, output: []Proposal) !void {
        if (streams.len > 8 or streams.len != output.len) return error.InvalidDraftStreams;
        var active: [8]Stream = undefined;
        var indexes: [8]usize = undefined;
        var count: usize = 0;
        var common_depth: usize = 0;
        for (streams, 0..) |stream, j| {
            if (stream.state.borrowed) return error.RequestRoundActive;
            for (streams[0..j]) |previous| if (stream.state == previous.state) return error.DuplicateDraftStream;
            common_depth = @max(common_depth, @min(15, stream.budget));
            if (stream.budget == 0 or stream.state.dflash_cache[0].a.ctx == null) continue;
            if (stream.anchor < 0 or stream.anchor >= 248320 or stream.state.dflash_offset < 0 or stream.state.dflash_offset > std.math.maxInt(i32) - 16) return error.InvalidDraftStream;
            for (stream.state.dflash_cache) |cache| {
                if (cache.a.ctx == null or cache.b.ctx == null or mx.shape(cache.a).len != 4 or !std.mem.eql(i32, mx.shape(cache.a), mx.shape(cache.b)) or mx.dim(cache.a, 0) != 1 or mx.dim(cache.a, 1) != 8 or mx.dim(cache.a, 2) < 1 or mx.dim(cache.a, 2) > 2047 or mx.dim(cache.a, 3) != 128 or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16) return error.InvalidDraftCache;
            }
            active[count] = stream;
            indexes[count] = j;
            count += 1;
        }
        @memset(output, .{});
        if (count == 0) return;
        for (active[0..count]) |*stream| stream.budget = common_depth;
        var s = mx.Scope{};
        defer s.deinit();
        const lattice = try d.latticeStreams(target, &s, active[0..count]);
        try mx.evalMany(&.{ lattice.candidates, lattice.scores, lattice.projection }, false);
        const ids = mx.c.mlx_array_data_int32(lattice.candidates);
        const values = mx.c.mlx_array_data_float32(lattice.scores);
        const projected = mx.c.mlx_array_data_float32(lattice.projection);
        var at: usize = 0;
        for (active[0..count], indexes[0..count]) |stream, j| {
            output[j] = try d.finish(ids[at * 16 ..][0 .. common_depth * 16], values[at * 16 ..][0 .. common_depth * 16], projected[at * 256 ..][0 .. common_depth * 256], stream.state.dflash_offset, stream.anchor, common_depth, stream.settings);
            output[j].len = @min(output[j].len, streams[j].budget);
            at += common_depth;
        }
    }
    pub fn checkStreams(d: *Drafter, target: *model.Model) !void {
        try d.checkCompiledParts();
        try d.checkProjections(target);
        const State = @import("request_state.zig").State(model.Model);
        const equal = @import("variant_checks.zig").equalBits;
        var states: [8]State = undefined;
        var initialized: usize = 0;
        defer for (states[0..initialized]) |*state| state.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        const lengths = [_]i32{ 1, 2, 7, 15, 31, 64, 127, 2047 };
        var handles: [8][5][2]?*anyopaque = undefined;
        for (&states, lengths, 0..) |*state, past, j| {
            state.* = try State.init(target);
            initialized += 1;
            state.dflash_offset = past + @as(i32, @intCast(j)) * 97;
            const values = try mx.allocator.alloc(f32, @as(usize, @intCast(past)) * 8 * 128);
            defer mx.allocator.free(values);
            for (&state.dflash_cache, 0..) |*cache, layer| {
                for (values, 0..) |*value, p| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((p * 7 + j * 11 + layer * 3) % 29)) - 14)) / 64;
                cache.a = try mx.retain(try scope.cast(try scope.data(values.ptr, &.{ 1, 8, past, 128 }, mx.f32t), mx.bf16));
                for (values, 0..) |*value, p| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((p * 13 + j * 5 + layer * 7) % 31)) - 15)) / 32;
                cache.b = try mx.retain(try scope.cast(try scope.data(values.ptr, &.{ 1, 8, past, 128 }, mx.f32t), mx.bf16));
                handles[j][layer] = .{ cache.a.ctx, cache.b.ctx };
            }
        }
        try d.checkAttentionStreams(target, &states);
        var streams: [8]Stream = undefined;
        var actual: [8]Proposal = undefined;
        const budgets = [_][8]usize{ @splat(15), .{ 15, 7, 3, 1, 0, 15, 7, 3 }, .{ 3, 7, 15, 0, 1, 3, 7, 15 }, .{ 7, 3, 1, 0, 7, 3, 1, 0 }, .{ 3, 1, 0, 3, 1, 0, 3, 1 } };
        for (budgets, 0..) |depths, pass| {
            for (&streams, &states, 0..) |*stream, *state, j| stream.* = .{
                .state = state,
                .anchor = @intCast(1000 + 37 * j),
                .budget = depths[j],
                .settings = .{ .temperature = if (pass == 0 or j % 2 == 0) 0 else 0.75, .seed = @intCast(1234 + j * 17) },
            };
            try d.proposeStreams(target, &streams, &actual);
            var active: [8]Stream = undefined;
            var count: usize = 0;
            for (streams) |stream| if (stream.budget > 0) {
                active[count] = stream;
                count += 1;
            };
            var batched_scope = mx.Scope{};
            defer batched_scope.deinit();
            const batched = try d.latticeStreams(target, &batched_scope, active[0..count]);
            var row_at: i32 = 0;
            var depth_at: i32 = 0;
            for (streams, actual, 0..) |stream, proposed, j| {
                errdefer std.debug.print("Shared Qwen DFlash mismatch: pass {d}, stream {d}, depth {d}\n", .{ pass, j, stream.budget });
                stream.state.swapDFlash(d);
                var expected = d.propose(target, stream.anchor, if (stream.budget == 0) 0 else std.mem.max(usize, &depths), stream.settings) catch |err| {
                    stream.state.swapDFlash(d);
                    return err;
                };
                stream.state.swapDFlash(d);
                expected.len = @min(expected.len, stream.budget);
                try std.testing.expectEqual(expected.len, proposed.len);
                try std.testing.expectEqualSlices(i32, expected.tokens[0..expected.len], proposed.tokens[0..proposed.len]);
                try std.testing.expectEqualSlices(i32, expected.parents[0..expected.len], proposed.parents[0..proposed.len]);
                try std.testing.expectEqualSlices(f64, expected.scores[0..expected.len], proposed.scores[0..proposed.len]);
                try std.testing.expectEqualSlices(f64, expected.probabilities[0..expected.len], proposed.probabilities[0..proposed.len]);
                if (stream.budget == 0) continue;
                var isolated_scope = mx.Scope{};
                defer isolated_scope.deinit();
                const isolated = try d.latticeStreams(target, &isolated_scope, &.{stream});
                const depth: i32 = @intCast(@min(15, stream.budget));
                for (isolated.layers, batched.layers) |single, shared| try equal(&isolated_scope, single, try isolated_scope.slice(shared, 1, row_at, row_at + depth + 1));
                try equal(&isolated_scope, isolated.candidates, try isolated_scope.slice(batched.candidates, 0, depth_at, depth_at + depth));
                try equal(&isolated_scope, isolated.scores, try isolated_scope.slice(batched.scores, 0, depth_at, depth_at + depth));
                try equal(&isolated_scope, isolated.projection, try isolated_scope.slice(batched.projection, 1, depth_at, depth_at + depth));
                row_at += depth + 1;
                depth_at += depth;
            }
            for (states, handles, lengths, 0..) |state, saved, past, j| {
                try std.testing.expectEqual(past + @as(i32, @intCast(j)) * 97, state.dflash_offset);
                for (state.dflash_cache, saved) |cache, pair| {
                    try std.testing.expectEqual(pair[0], cache.a.ctx);
                    try std.testing.expectEqual(pair[1], cache.b.ctx);
                }
            }
        }
        try std.testing.expectError(error.InvalidDraftStreams, d.proposeStreams(target, &streams, actual[0..7]));
        try std.testing.expectError(error.DuplicateDraftStream, d.proposeStreams(target, &.{ streams[0], streams[0] }, actual[0..2]));
        states[0].borrowed = true;
        const rejected = d.proposeStreams(target, streams[0..1], actual[0..1]);
        states[0].borrowed = false;
        try std.testing.expectError(error.RequestRoundActive, rejected);
        var invalid_part: [2]A = undefined;
        try std.testing.expectError(error.InvalidDraftBlock, d.blockPart(&scope, 0, .pre, &.{try scope.zeros(&.{ 1, 17, 5120 }, mx.bf16)}, &.{}, &invalid_part));
        try d.proposeStreams(target, streams[0..1], actual[0..1]);
        try d.checkAbsorbStreams(target, &states);
        std.debug.print("Shared Qwen DFlash: exact common-block calibrated proposals, independent ragged lattices, maxima15/7/3, mixed sampling and unequal sliding contexts; caches unchanged\n", .{});
    }
    fn checkAttentionStreams(d: *Drafter, target: *model.Model, states: []@import("request_state.zig").State(model.Model)) !void {
        const equal = @import("variant_checks.zig").equalBits;
        var fixtures = mx.Scope{};
        defer fixtures.deinit();
        const values = try mx.allocator.alloc(f32, 128 * 5120);
        defer mx.allocator.free(values);
        for (values, 0..) |*value, j| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(j * 19 % 59)) - 29)) / 32;
        const all = try fixtures.cast(try fixtures.data(values.ptr, &.{ 1, 128, 5120 }, mx.f32t), mx.bf16);
        const cases = [_][]const usize{ &.{15}, &.{ 15, 15, 15, 15 }, &.{ 15, 7, 3, 1 }, &.{ 15, 15, 15, 15, 15, 15, 15, 15 }, &.{ 15, 7, 3, 1, 1, 3, 7, 15 } };
        for (cases) |budgets| {
            var streams: [8]Stream = undefined;
            var rows: i32 = 0;
            for (streams[0..budgets.len], states[0..budgets.len], budgets) |*stream, *state, budget| {
                stream.* = .{ .state = state, .anchor = 1000, .budget = budget, .settings = .{} };
                rows += @intCast(budget + 1);
            }
            for (0..5) |layer| {
                var scope = mx.Scope{};
                defer scope.deinit();
                var masks = AttentionMasks{};
                const x = try scope.slice(all, 1, 0, rows);
                const actual = try d.attentionStreams(&target.kernels, &scope, layer, x, streams[0..budgets.len], &masks);
                var at: i32 = 0;
                for (streams[0..budgets.len]) |stream| {
                    const end = at + @as(i32, @intCast(stream.budget + 1));
                    stream.state.swapDFlash(d);
                    defer stream.state.swapDFlash(d);
                    const expected = try d.attention(&target.kernels, &scope, layer, try scope.slice(x, 1, at, end));
                    try equal(&scope, expected, try scope.slice(actual, 1, at, end));
                    at = end;
                }
            }
        }
        std.debug.print("PASS: DFlash shared Q/K normalization and RoPE match serial attention for 1/4/8 streams, common/ragged rows, independent offsets and sliding caches\n", .{});
    }
    fn checkProjections(d: *Drafter, target: *model.Model) !void {
        const equal = @import("variant_checks.zig").equalBits;
        const source_head = try target.weights.linear("lm_head");
        const tail: i32 = if (source_head.tile_width == 64) 248000 else 248032;
        try std.testing.expectEqual(tail, d.vocabulary_tail);
        try std.testing.expectEqual(source_head.tile_width, d.head.tile_width);
        try std.testing.expectEqual(@as(i32, 98303), d.candidateID(98303));
        try std.testing.expectEqual(tail, d.candidateID(98304));
        try std.testing.expectEqual(tail + 31, d.candidateID(98335));
        try std.testing.expectEqual(tail + 32, d.candidateID(98336));
        try std.testing.expectEqual(@as(i32, 248319), d.candidateID(d.head.n - 1));
        var fixtures = mx.Scope{};
        defer fixtures.deinit();
        const values = try mx.allocator.alloc(f32, 128 * 5120);
        defer mx.allocator.free(values);
        for (values, 0..) |*value, j| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(j * 13 % 47)) - 23)) / 32;
        const all = try fixtures.cast(try fixtures.data(values.ptr, &.{ 1, 128, 5120 }, mx.f32t), mx.bf16);
        if (d.head.tiled) {
            const saved = d.head;
            const checked = d.head_checked;
            const full = d.full_head;
            d.head = try copyHead(saved);
            defer {
                d.head.deinit();
                d.head = saved;
                d.head_checked = checked;
                d.full_head = full;
            }
            const x = try fixtures.slice(all, 1, 0, 1);
            d.head_checked = false;
            try equal(&fixtures, try d.candidateLogits(target, &fixtures, .{ .x = x }), try saved.apply(&target.kernels, &fixtures, .{ .x = x }));
            try std.testing.expect(d.head_checked and !d.full_head);
            const bad = try mx.retain(try fixtures.zeros(mx.shape(d.head.sb), mx.dtype(d.head.sb)));
            mx.free(d.head.sb);
            d.head.sb = bad;
            d.head_checked = false;
            try equal(&fixtures, try d.candidateLogits(target, &fixtures, .{ .x = x }), try source_head.apply(&target.kernels, &fixtures, .{ .x = x }));
            try std.testing.expect(d.head_checked and d.full_head);
            try std.testing.expectEqual(@as(i32, 98304), d.candidateID(98304));
        }
        for ([_]i32{ 1, 8, 17, 32, 33, 128 }) |rows| {
            var scope = mx.Scope{};
            defer scope.deinit();
            const x = try scope.slice(all, 1, 0, rows);
            const input = try projectionInput(&target.kernels, &scope, x);
            for (0..5) |layer| {
                const expected = [_]A{
                    try d.project(&target.kernels, &scope, layer, "self_attn.k_proj", x),
                    try d.project(&target.kernels, &scope, layer, "self_attn.v_proj", x),
                };
                const actual = try d.projectKV(&target.kernels, &scope, layer, input);
                for (actual, expected) |got, want| try equal(&scope, got, want);
                const query = try d.linear(layer, "self_attn.q_proj");
                try equal(&scope, try query.apply(&target.kernels, &scope, input), try query.apply(&target.kernels, &scope, .{ .x = x }));
            }
        }
        std.debug.print("PASS: DFlash stacked KV and shared activation sums match separate projections across all five layers and lane tile boundaries\n", .{});
    }
    fn checkCompiledParts(d: *Drafter) !void {
        const equal = @import("variant_checks.zig").equalBits;
        var fixtures = mx.Scope{};
        defer fixtures.deinit();
        var parts: [2]CompiledPart = @splat(.{});
        defer for (&parts) |*part| part.deinit();
        for (&parts, [_]BlockPart{ .pre, .post }) |*part, kind| {
            var original = try CompiledPart.init(d, 0, kind);
            defer original.deinit();
            var payload = original.payload.?.*;
            payload.kernels = mx.Kernels.init();
            payload.failure = null;
            // Exact, lazy packed weights expose per-specialization constant copies.
            const zero: u32 = 0;
            const z = try fixtures.data(&zero, &.{}, mx.c.MLX_UINT32);
            inline for (.{ "dynamic", "gate", "up", "down" }) |field| {
                if (kind == .post or comptime std.mem.eql(u8, field, "dynamic")) {
                    const projection = &@field(payload, field);
                    try std.testing.expectEqual(@as(mx.c.mlx_dtype, mx.c.MLX_UINT32), mx.dtype(projection.weight));
                    projection.weight = try fixtures.binary(mx.c.mlx_add, projection.weight, z);
                }
            }
            part.* = try CompiledPart.create(payload);
        }
        const values = try mx.allocator.alloc(f32, 128 * 5120);
        defer mx.allocator.free(values);
        for (values, 0..) |*value, index| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((index * 7) % 61)) - 30)) / 32;
        const input = try fixtures.cast(try fixtures.data(values.ptr, &.{ 1, 128, 5120 }, mx.f32t), mx.bf16);
        try mx.eval(input);
        try mx.check(mx.c.mlx_clear_cache());
        const before = try @import("memory_runtime.zig").activeBytes();
        const cases = [_][]const i32{
            &.{1},      &.{2},      &.{3},      &.{4},      &.{5},        &.{6},                        &.{7},                        &.{8},                                &.{9}, &.{10}, &.{11}, &.{12}, &.{13}, &.{14}, &.{15}, &.{16},
            &.{ 1, 1 }, &.{ 1, 3 }, &.{ 3, 1 }, &.{ 8, 8 }, &.{ 16, 16 }, &.{ 8, 8, 8, 8, 8, 8, 8, 8 }, &.{ 1, 2, 3, 4, 5, 6, 7, 8 }, &.{ 16, 16, 16, 16, 16, 16, 16, 16 },
        };
        var largest_growth: usize = 0;
        for (0..2) |repetition| for (cases) |lengths| {
            {
                var s = mx.Scope{};
                defer s.deinit();
                var total: i32 = 0;
                for (lengths) |length| total += length;
                const x = try s.slice(input, 1, 0, total);
                var layout: [8]A = undefined;
                var at: i32 = 0;
                for (lengths, 0..) |length, j| {
                    layout[j] = try s.slice(x, 1, at, at + length);
                    at += length;
                }
                const views = layout[0..if (lengths.len == 1) @as(usize, 0) else lengths.len];
                var pre: [2]A = undefined;
                var expected_pre: [2]A = undefined;
                try parts[0].call(&s, &.{x}, views, &pre);
                try parts[0].apply(&s, &.{x}, views, &expected_pre, false);
                for (pre, expected_pre) |actual, expected| try equal(&s, actual, expected);
                if (lengths.len > 1) {
                    const payload = parts[0].payload.?;
                    const normalized = try s.rms(x, payload.norm);
                    const starts = try CompiledPart.Payload.blockStarts(&s, lengths);
                    for ([_]A{ payload.attention_base, parts[1].payload.?.mlp_base }) |base| for (0..2) |tap_part| {
                        const actual = try payload.convolve(&s, normalized, pre[1], base, @intCast(tap_part), total, starts);
                        var independent: [8]A = undefined;
                        var first: i32 = 0;
                        for (lengths, 0..) |length, j| {
                            independent[j] = try payload.convolve(&s, try s.slice(normalized, 1, first, first + length), try s.slice(pre[1], 1, first, first + length), base, @intCast(tap_part), length, null);
                            first += length;
                        }
                        try equal(&s, actual, try s.cat(independent[0..lengths.len], 1));
                    };
                }
                var post: [1]A = undefined;
                var expected_post: [1]A = undefined;
                try parts[1].call(&s, &.{ x, pre[0], pre[1] }, views, &post);
                try parts[1].apply(&s, &.{ x, expected_pre[0], expected_pre[1] }, views, &expected_post, false);
                try equal(&s, post[0], expected_post[0]);
            }
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            try mx.check(mx.c.mlx_clear_cache());
            const growth = (try @import("memory_runtime.zig").activeBytes()) -| before;
            largest_growth = @max(largest_growth, growth);
            if (growth > 8 * 1024 * 1024) {
                std.debug.print("DFlash compiled retention: repetition={d}, lengths={any}, growth={d} bytes\n", .{ repetition, lengths, growth });
                return error.CompiledWeightsDuplicated;
            }
        };
        std.debug.print("PASS: DFlash pre/post compiled arrays match uncompiled through24 shapes/partitions repeated twice; retained growth={d} bytes\n", .{largest_growth});
    }
    fn checkAbsorbStreams(d: *Drafter, target: *model.Model, states: []@import("request_state.zig").State(model.Model)) !void {
        const State = @import("request_state.zig").State(model.Model);
        const equal = @import("variant_checks.zig").equalBits;
        const previous_capture = d.capture;
        d.capture = null;
        defer d.capture = previous_capture;
        for (&states[0].dflash_cache) |*cache| cache.deinit();
        states[0].dflash_offset = 0;
        for (states[1..], 1..) |*state, j| if (j % 2 == 0) {
            for (&state.dflash_cache) |*cache| {
                cache.keys.current = try mx.retain(cache.a);
                cache.keys.offset = mx.dim(cache.a, 2);
                cache.values.current = try mx.retain(cache.b);
                cache.values.offset = mx.dim(cache.b, 2);
            }
        };
        var isolated: [8]State = undefined;
        var initialized: usize = 0;
        defer for (isolated[0..initialized]) |*state| state.deinit();
        for (states, &isolated) |*state, *copy| {
            copy.* = try state.clone();
            initialized += 1;
        }
        var passes: [8]model.Pass = @splat(.{});
        defer for (&passes) |*pass| pass.deinit();
        var tokens: [8][16]i32 = undefined;
        var paths: [8][16]i32 = undefined;
        var streams: [8]AbsorbStream = undefined;
        const values = try mx.allocator.alloc(f32, 16 * 5120);
        defer mx.allocator.free(values);
        for (&passes, 0..) |*pass, j| {
            pass.count = 16;
            for (&pass.taps, 0..) |*tap, layer| {
                for (values, 0..) |*value, p| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((p * 7 + j * 11 + layer * 3) % 61)) - 30)) / 32;
                tap.* = try pass.scope.cast(try pass.scope.data(values.ptr, &.{ 1, 16, 5120 }, mx.f32t), mx.bf16);
            }
            for (&tokens[j], 0..) |*token, row| token.* = @intCast(1000 + j * 17 + row);
        }
        const lengths = [_]usize{ 0, 1, 2, 3, 5, 7, 8, 16 };
        for (0..2) |round| {
            for (&streams, states, &passes, lengths, 0..) |*stream, *state, *pass, ragged, j| {
                const count: usize = if (round == 0) 16 else ragged;
                @memset(pass.parents[0..16], 0);
                pass.parents[0] = -1;
                for (paths[j][0..count], 0..) |*row, r| {
                    row.* = @intCast(if (round == 1 and count <= 8) r * 2 else r);
                    pass.parents[@intCast(row.*)] = if (r == 0) -1 else paths[j][r - 1];
                }
                stream.* = .{ .state = state, .pass = pass, .tokens = &tokens[j], .rows = paths[j][0..count] };
            }
            for (streams, &isolated) |stream, *state| if (stream.rows.len > 0) {
                state.swapDFlash(d);
                d.absorb(target, stream.pass, stream.rows, stream.tokens) catch |err| {
                    state.swapDFlash(d);
                    return err;
                };
                state.swapDFlash(d);
            };
            const empty_handle = states[0].dflash_cache[0].a.ctx;
            try d.absorbStreams(target, &streams);
            if (round == 1) try std.testing.expectEqual(empty_handle, states[0].dflash_cache[0].a.ctx);
            var scope = mx.Scope{};
            defer scope.deinit();
            for (states, isolated, 0..) |state, reference, j| {
                errdefer std.debug.print("Shared DFlash context mismatch: round {d}, stream {d}\n", .{ round, j });
                try std.testing.expectEqual(reference.dflash_offset, state.dflash_offset);
                for (state.dflash_cache, reference.dflash_cache) |actual, expected| {
                    try std.testing.expectEqualSlices(i32, mx.shape(expected.a), mx.shape(actual.a));
                    try equal(&scope, expected.a, actual.a);
                    try equal(&scope, expected.b, actual.b);
                    try std.testing.expect(actual.keys.current.ctx == null and actual.values.current.ctx == null);
                }
            }
        }
        const before_offset = states[2].dflash_offset;
        const before_handle = states[2].dflash_cache[0].a.ctx;
        var bad = streams[1];
        bad.rows = &.{16};
        try std.testing.expectError(error.InvalidDraftRows, d.absorbStreams(target, &.{ streams[2], bad }));
        try std.testing.expectError(error.DuplicateDraftStream, d.absorbStreams(target, &.{ streams[0], streams[0] }));
        try std.testing.expectEqual(before_offset, states[2].dflash_offset);
        try std.testing.expectEqual(before_handle, states[2].dflash_cache[0].a.ctx);
        var wide = model.Pass{ .count = 128 };
        defer wide.deinit();
        var wide_tokens: [128]i32 = undefined;
        var wide_rows: [128]i32 = undefined;
        for (&wide.taps, passes[7].taps) |*tap, source| {
            const pieces: [8]A = @splat(source);
            tap.* = try wide.scope.cat(&pieces, 1);
        }
        for (&wide_tokens, &wide_rows, 0..) |*token, *row, j| {
            token.* = @intCast(3000 + j);
            row.* = @intCast(j);
        }
        isolated[7].swapDFlash(d);
        d.absorb(target, &wide, &wide_rows, &wide_tokens) catch |err| {
            isolated[7].swapDFlash(d);
            return err;
        };
        isolated[7].swapDFlash(d);
        const wide_stream = AbsorbStream{ .state = &states[7], .pass = &wide, .rows = &wide_rows, .tokens = &wide_tokens };
        try d.absorbStreams(target, &.{wide_stream});
        try std.testing.expectEqual(isolated[7].dflash_offset, states[7].dflash_offset);
        for (states[7].dflash_cache, isolated[7].dflash_cache) |actual, expected| {
            try equal(&wide.scope, expected.a, actual.a);
            try equal(&wide.scope, expected.b, actual.b);
        }
        const wide_offset = states[7].dflash_offset;
        try std.testing.expectError(error.InvalidDraftStreams, d.absorbStreams(target, &.{ wide_stream, streams[2] }));
        try std.testing.expectEqual(wide_offset, states[7].dflash_offset);
        std.debug.print("Shared Qwen DFlash absorption: exact isolated caches/offsets for 8x16 and 1x128 rows, ragged accepted tree paths, empty commits, buffer-backed inputs and sliding rollover\n", .{});
    }
    fn saveArray(s: *mx.Scope, directory: []const u8, name: []const u8, value: A) !void {
        const path = try std.fmt.allocPrintSentinel(mx.allocator, "{s}/{s}.npy", .{ directory, name }, 0);
        defer mx.allocator.free(path);
        const converted = try s.cast(value, mx.f32t);
        try mx.eval(converted);
        try mx.saveArray(path, converted);
    }
    fn saveLattice(d: *Drafter, s: *mx.Scope, output: []const u8, prefix: []const u8, lattice: Lattice, proposals: []const Proposal) !void {
        var buffer: [160]u8 = undefined;
        for (lattice.layers, 0..) |layer, i| try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}layer-{d}", .{ prefix, i }), layer);
        try mx.eval(lattice.candidates);
        var ids: [8 * 15 * 16]i32 = undefined;
        const count = mx.c.mlx_array_size(lattice.candidates);
        if (count > ids.len) return error.InvalidDraftStreams;
        for (ids[0..count], mx.c.mlx_array_data_int32(lattice.candidates)[0..count]) |*id, index| id.* = d.candidateID(index);
        try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}candidates", .{prefix}), try s.data(&ids, mx.shape(lattice.candidates), mx.c.MLX_INT32));
        try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}scores", .{prefix}), lattice.scores);
        try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}projection", .{prefix}), lattice.projection);
        for (proposals, 0..) |proposal, j| {
            try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}tokens-{d}", .{ prefix, j }), try s.ints(proposal.tokens[0..proposal.len]));
            try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}parents-{d}", .{ prefix, j }), try s.ints(proposal.parents[0..proposal.len]));
            var probabilities: [15]f32 = undefined;
            for (proposal.probabilities[0..proposal.len], probabilities[0..proposal.len]) |p, *value| value.* = @floatCast(p);
            try saveArray(s, output, try std.fmt.bufPrint(&buffer, "{s}probabilities-{d}", .{ prefix, j }), try s.data(&probabilities, &.{@intCast(proposal.len)}, mx.f32t));
        }
    }
    pub fn oracleStreams(d: *Drafter, target: *model.Model, io: std.Io, fixture: []const u8, output: []const u8) !void {
        const State = @import("request_state.zig").State(model.Model);
        const Meta = struct { offsets: [8]i32, anchors: [8]i32, pasts: [8]i32 };
        var buf: [4096]u8 = undefined;
        const bytes = try weights.readFile(io, try std.fmt.bufPrint(&buf, "{s}/streams.json", .{fixture}));
        defer mx.allocator.free(bytes);
        const meta = try std.json.parseFromSlice(Meta, mx.allocator, bytes, .{});
        defer meta.deinit();
        try std.Io.Dir.cwd().createDirPath(io, output);
        var states: [8]State = undefined;
        var initialized: usize = 0;
        defer for (states[0..initialized]) |*state| state.deinit();
        var scope = mx.Scope{};
        defer scope.deinit();
        const boundary = [_]i32{ 0, 98303, 98304, 98335, 98336, d.head.n - 1 };
        var boundary_ids: [boundary.len]i32 = undefined;
        for (boundary, &boundary_ids) |index, *id| id.* = d.candidateID(index);
        try saveArray(&scope, output, "vocabulary-ids", try scope.ints(&boundary_ids));
        var head_input: [5120]f32 = undefined;
        for (&head_input, 0..) |*value, j| value.* = @as(f32, @floatFromInt(@as(i32, @intCast(j * 7 % 29)) - 14)) / 64;
        const head_logits = try d.head.apply(&target.kernels, &scope, .{ .x = try scope.cast(try scope.data(&head_input, &.{ 1, 1, 5120 }, mx.f32t), mx.bf16) });
        try saveArray(&scope, output, "vocabulary-logits", try scope.take(head_logits, try scope.ints(&boundary), 2));
        const cpu = mx.c.mlx_default_cpu_stream_new();
        defer _ = mx.c.mlx_stream_free(cpu);
        var streams: [8]Stream = undefined;
        var passes: [8]model.Pass = @splat(.{});
        defer for (&passes) |*pass| pass.deinit();
        var absorption: [8]AbsorbStream = undefined;
        for (&states, &streams, 0..) |*state, *stream, j| {
            state.* = try State.init(target);
            initialized += 1;
            state.dflash_offset = meta.value.offsets[j] - 1;
            if (meta.value.pasts[j] > 0) for (&state.dflash_cache, 0..) |*cache, layer| {
                inline for (.{ "keys", "values" }, .{ "a", "b" }) |part, field| {
                    var array = mx.c.mlx_array_new();
                    const rc = mx.c.mlx_load(&array, try std.fmt.bufPrintSentinel(&buf, "{s}/input-cache-{d}-{d}-{s}.npy", .{ fixture, j, layer, part }, 0), cpu);
                    @field(cache, field) = try mx.retain(try scope.cast(try scope.result(rc, array), mx.bf16));
                }
            };
            var taps = mx.c.mlx_array_new();
            const rc = mx.c.mlx_load(&taps, try std.fmt.bufPrintSentinel(&buf, "{s}/taps-{d}.npy", .{ fixture, j }, 0), cpu);
            taps = try scope.cast(try scope.result(rc, taps), mx.bf16);
            for (&passes[j].taps, 0..) |*tap, layer| {
                const at: i32 = @intCast(layer * 5120);
                tap.* = try passes[j].scope.own(try mx.retain(try scope.slice(taps, 2, at, at + 5120)));
            }
            passes[j].count = 1;
            absorption[j] = .{ .state = state, .pass = &passes[j], .rows = &.{0}, .tokens = meta.value.anchors[j..][0..1] };
            stream.* = .{ .state = state, .anchor = meta.value.anchors[j], .budget = 15, .settings = .{ .temperature = if (j % 2 == 0) 0 else 0.75, .seed = @intCast(1234 + j * 17) } };
        }
        try d.absorbStreams(target, &absorption);
        for (states, 0..) |state, j| {
            try std.testing.expectEqual(meta.value.offsets[j], state.dflash_offset);
            for (state.dflash_cache, 0..) |cache, layer| {
                try saveArray(&scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-keys", .{ j, layer }), cache.a);
                try saveArray(&scope, output, try std.fmt.bufPrint(&buf, "cache-{d}-{d}-values", .{ j, layer }), cache.b);
            }
        }
        var proposals: [8]Proposal = undefined;
        try d.proposeStreams(target, &streams, &proposals);
        const lattice = try d.latticeStreams(target, &scope, &streams);
        try d.saveLattice(&scope, output, "", lattice, &proposals);
        const cases = [_]struct { prefix: []const u8, budgets: [8]usize }{
            .{ .prefix = "ragged15-", .budgets = .{ 15, 7, 3, 1, 15, 7, 3, 1 } },
            .{ .prefix = "ragged7-", .budgets = .{ 7, 3, 1, 7, 3, 1, 7, 1 } },
            .{ .prefix = "ragged3-", .budgets = .{ 3, 1, 3, 1, 3, 1, 3, 1 } },
        };
        for (cases) |case| {
            var case_scope = mx.Scope{};
            defer case_scope.deinit();
            for (&streams, case.budgets) |*stream, budget| stream.budget = budget;
            try d.proposeStreams(target, &streams, &proposals);
            var common = streams;
            for (&common) |*stream| stream.budget = std.mem.max(usize, &case.budgets);
            try d.saveLattice(&case_scope, output, case.prefix, try d.latticeStreams(target, &case_scope, &common), &proposals);
        }
        std.debug.print("Saved shared Qwen DFlash layers, lattice and proposals for upstream comparison\n", .{});
    }
    pub fn propose(d: *Drafter, target: *model.Model, anchor: i32, budget: usize, settings: sampling.Sampling) !Proposal {
        if (budget == 0 or d.cache[0].a.ctx == null) return .{};
        const n: usize = @min(16, budget + 1);
        var block: [16]i32 = @splat(248070);
        block[0] = anchor;
        var s = mx.Scope{};
        defer s.deinit();
        const k = &target.kernels;
        var h = try target.weights.embed(&s, block[0..n]);
        for (0..5) |i| {
            var pre: [2]A = undefined;
            try d.blockPart(&s, i, .pre, &.{h}, &.{}, &pre);
            const attended = try d.attention(k, &s, i, pre[0]);
            var post: [1]A = undefined;
            try d.blockPart(&s, i, .post, &.{ h, attended, pre[1] }, &.{}, &post);
            h = post[0];
            if (i == 0 or i == 2) try mx.evalMany(&.{h}, true);
        }
        const hidden = try s.rms(try s.slice(h, 1, 1, @intCast(n)), try d.weights.get("norm.weight"));
        const logits = try d.candidateLogits(target, &s, .{ .x = hidden });
        const ranked = try @import("gpu_sampling.zig").topk(k, &s, logits, 16);
        const projection = try s.cast(try (try d.weights.linear("candidate_selector.hidden_projection")).apply(k, &s, .{ .x = hidden }), mx.f32t);
        try mx.evalMany(&.{ ranked[0], ranked[1], projection }, false);
        return d.finish(mx.c.mlx_array_data_int32(ranked[0])[0 .. (n - 1) * 16], mx.c.mlx_array_data_float32(ranked[1])[0 .. (n - 1) * 16], mx.c.mlx_array_data_float32(projection)[0 .. (n - 1) * 256], d.offset, anchor, budget, settings);
    }
    fn candidateID(d: *const Drafter, index: i32) i32 {
        return if (d.full_head or index < 98304) index else index - 98304 + d.vocabulary_tail;
    }

    fn copyHead(source: lanes.Linear) !lanes.Linear {
        var owned = source;
        inline for (.{ "weight", "sb", "scales", "biases", "signs" }) |field| @field(owned, field) = mx.empty;
        errdefer owned.deinit();
        inline for (.{ "weight", "sb", "scales", "biases", "signs" }) |field| {
            const array = @field(source, field);
            if (array.ctx != null) @field(owned, field) = try mx.retain(array);
        }
        return owned;
    }

    fn candidateLogits(d: *Drafter, target: *model.Model, s: *mx.Scope, input: lanes.Act) !A {
        const logits = try d.head.apply(&target.kernels, s, input);
        if (d.head_checked or !d.head.tiled) return logits;
        const source = try target.weights.linear("lm_head");
        const full = try source.apply(&target.kernels, s, input);
        var check = mx.Scope{};
        defer check.deinit();
        const kept = try check.cast(try check.cat(&.{ try check.slice(full, 2, 0, 98304), try check.slice(full, 2, d.vocabulary_tail, 248320) }, -1), mx.f32t);
        const delta = try check.unary(mx.c.mlx_abs, try check.binary(mx.c.mlx_subtract, try check.cast(logits, mx.f32t), kept));
        var maxima: [2]A = undefined;
        for ([_]A{ delta, try check.unary(mx.c.mlx_abs, kept) }, &maxima) |array, *out| {
            out.* = mx.c.mlx_array_new();
            const rc = mx.c.mlx_max(out, array, false, mx.stream);
            out.* = try check.result(rc, out.*);
        }
        try mx.evalMany(&maxima, false);
        const difference = mx.c.mlx_array_data_float32(maxima[0])[0];
        const scale = mx.c.mlx_array_data_float32(maxima[1])[0];
        if (@as(f64, difference) <= 0.02 * @max(@as(f64, scale), 1e-6)) {
            d.head_checked = true;
            return logits;
        }
        const replacement = try copyHead(source);
        d.head.deinit();
        d.head = replacement;
        d.full_head = true;
        d.head_checked = true;
        std.debug.print("DFlash reduced head differs from full head ({d} against {d}); using full head\n", .{ difference, scale });
        return full;
    }
    fn finish(d: *Drafter, indices: []const i32, values: []const f32, projection: []const f32, offset: i32, anchor: i32, budget: usize, settings: sampling.Sampling) !Proposal {
        const depth = indices.len / 16;
        var candidates: [15][16]sampling.Candidate = undefined;
        for (0..depth) |row| {
            for (0..16) |j| {
                const id = indices[row * 16 + j];
                candidates[row][j] = .{ .id = d.candidateID(id), .value = values[row * 16 + j] };
            }
        }
        var result = d.search(candidates[0..depth], projection, offset, anchor, @min(budget, 15), settings);
        var arena = std.heap.ArenaAllocator.init(mx.allocator);
        defer arena.deinit();
        const table = d.calibration_tables.value.tables.map.get(if (settings.temperature > 0) "sampled" else "greedy");
        const calibrated = try calibration.rank(arena.allocator(), result.tokens[0..result.len], result.parents[0..result.len], result.scores[0..result.len], table);
        @memcpy(result.tokens[0..result.len], calibrated.tokens);
        @memcpy(result.parents[0..result.len], calibrated.parents);
        @memcpy(result.scores[0..result.len], calibrated.scores);
        @memcpy(result.probabilities[0..result.len], calibrated.probabilities);
        return result;
    }
    fn search(d: *Drafter, candidates: []const [16]sampling.Candidate, hproj: []const f32, offset: i32, anchor: i32, budget: usize, settings: sampling.Sampling) Proposal {
        const Node = struct { score: f64, depth: usize, token: i32, parent: i32 };
        var queue: [64]Node = undefined;
        var len: usize = 0;
        var result = Proposal{};
        var parent: i32 = -1;
        var predecessor = anchor;
        var depth: usize = 0;
        var path_score: f64 = 0;
        const pred = mx.c.mlx_array_data_float32(d.pred);
        const succ = mx.c.mlx_array_data_float32(d.succ);
        while (true) {
            if (depth < candidates.len) {
                var scores: [16]f64 = undefined;
                var max: f64 = -std.math.inf(f64);
                const temp = if (settings.temperature > 0) @max(settings.temperature, 1e-6) else 1;
                for (candidates[depth], 0..) |candidate, j| {
                    var edge: f64 = 0;
                    for (0..256) |r| edge += @as(f64, pred[@as(usize, @intCast(predecessor)) * 256 + r]) * @as(f64, hproj[depth * 256 + r]) * @as(f64, succ[@as(usize, @intCast(candidate.id)) * 256 + r]);
                    const noise = if (settings.temperature > 0) sampling.noise(settings.seed, @intCast(offset + 1 + @as(i32, @intCast(depth))), @intCast(candidate.id)) else 0;
                    scores[j] = (candidate.value / temp + 0.6 * edge / temp + 0.7 * noise) / 1.5;
                    max = @max(max, scores[j]);
                }
                var sum: f64 = 0;
                for (scores) |score| sum += @exp(score - max);
                const normalizer = max + @log(sum);
                var chosen: [16]bool = @splat(false);
                for (0..4) |_| {
                    var best: usize = 0;
                    var best_score: f64 = -std.math.inf(f64);
                    for (scores, 0..) |score, j| if (!chosen[j] and score > best_score) {
                        best = j;
                        best_score = score;
                    };
                    chosen[best] = true;
                    queue[len] = .{ .score = path_score + scores[best] - normalizer, .depth = depth, .token = candidates[depth][best].id, .parent = parent };
                    len += 1;
                }
            }
            if (len == 0 or result.len >= budget) break;
            var best: usize = 0;
            for (queue[0..len], 0..) |node, j| if (node.score > queue[best].score) {
                best = j;
            };
            const node = queue[best];
            len -= 1;
            queue[best] = queue[len];
            result.tokens[result.len] = node.token;
            result.parents[result.len] = node.parent;
            result.scores[result.len] = node.score;
            parent = @intCast(result.len);
            predecessor = node.token;
            depth = node.depth + 1;
            path_score = node.score;
            result.len += 1;
        }
        return result;
    }
};
