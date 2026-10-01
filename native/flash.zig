//! Qwen3.8 Flash Next: four residual streams, GDN, sparse attention, MoE, PLE, MTP.
const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const FlashWeight = @import("flash_ops.zig").Weight;

pub const MoEInputs = struct {
    router: A,
    gate: FlashWeight,
    up: FlashWeight,
    shared_gate: FlashWeight,
    shared_up: FlashWeight,
    down: FlashWeight,
    shared_down: FlashWeight,
};

pub const CompiledMoE = struct {
    closure: mx.c.mlx_closure = .{ .ctx = null },
    payload: ?*Payload = null,
    generation: u32 = 0,
    const weights = .{ "gate", "up", "shared_gate", "shared_up", "down", "shared_down" };
    const Payload = struct {
        scope: mx.Scope = .{},
        kernels: mx.Kernels,
        inputs: MoEInputs,
        generation: u32,
        failure: ?anyerror = null,

        fn destroy(raw: ?*anyopaque) callconv(.c) void {
            const p: *Payload = @ptrCast(@alignCast(raw.?));
            p.kernels.deinit();
            p.scope.deinit();
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
            if (mx.c.mlx_vector_array_size(ins) != 3) return error.InvalidKernelArity;
            var s = mx.Scope{};
            defer s.deinit();
            var args: [3]A = undefined;
            for (&args, 0..) |*arg, i| {
                var value = mx.c.mlx_array_new();
                arg.* = try s.result(mx.c.mlx_vector_array_get(&value, ins, i), value);
            }
            const result = try moeGraph(&p.kernels, &s, p.inputs, args[0], args[1], args[2], p.generation);
            return mx.c.mlx_vector_array_set_data(out, &result, 2);
        }
    };

    pub fn init(inputs: MoEInputs) !CompiledMoE {
        var value = Payload{ .kernels = mx.Kernels.init(), .inputs = inputs, .generation = mx.gpu_generation };
        var transferred = false;
        errdefer if (!transferred) {
            value.kernels.deinit();
            value.scope.deinit();
        };
        value.inputs.router = try value.scope.own(try mx.retain(value.inputs.router));
        inline for (weights) |field| {
            for (&@field(value.inputs, field).arrays) |*array| {
                array.* = try value.scope.own(try mx.retain(array.*));
            }
        }
        try mx.evalMany(value.scope.arrays.items, false);
        const payload = try mx.allocator.create(Payload);
        payload.* = value;
        transferred = true;
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
        return .{ .closure = closure, .payload = payload, .generation = value.generation };
    }
    pub fn deinit(p: *CompiledMoE) void {
        if (p.closure.ctx != null) _ = mx.c.mlx_closure_free(p.closure);
        p.* = .{};
    }
    pub fn call(p: *CompiledMoE, kernels: *mx.Kernels, s: *mx.Scope, h: A, x: A, inject: A) ![5]A {
        if (p.generation != mx.gpu_generation) return error.InvalidCompiledGeneration;
        p.payload.?.failure = null;
        var result: [5]A = @splat(mx.empty);
        kernels.call(s, p.closure, &.{ h, x, inject }, result[0..2]) catch |err| return p.payload.?.failure orelse err;
        return result;
    }
};

fn moeGraph(kernels: *mx.Kernels, s: *mx.Scope, inputs: MoEInputs, h: A, x: A, inject: A, generation: u32) ![5]A {
    const r = mx.dim(x, 0);
    if (r < 1 or r > Model.max_shared_rows) return error.InvalidLaneWidth;
    var router: [Model.max_shared_rows / 16]A = undefined;
    var router_count: usize = 0;
    var first: i32 = 0;
    while (first < r) : (first += 16) {
        const end = @min(first + 16, r);
        router[router_count] = (try kernels.run(s, src.q4_router_float, &.{ try s.slice(x, 0, first, end), inputs.router, try s.ints(&.{end - first}) }, &.{ ti("D", 2560), ti("NE", 513), ti("T", 256), ti("MAXR", 16) }, .{ @divTrunc(513 + 7, 8) * 256, 1, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ end - first, 513 }, .dtype = mx.f32t }}))[0];
        router_count += 1;
    }
    const logits = if (router_count == 1) router[0] else try s.cat(router[0..router_count], 0);
    const ops = @import("flash_ops.zig");
    const act = try ops.gateUp(kernels, s, x, logits, inputs.gate, inputs.up, .{ inputs.shared_gate, inputs.shared_up }, 10, generation, 4, 2);
    const y = try ops.expertDown(kernels, s, act[0], act[1], inputs.down, inputs.shared_down, generation, 2);
    return kernels.run(s, src.q4_hc_norm_grouped, &.{ h, inject, y, act[2], logits }, &.{ ti("S", 4), ti("D", 2560), ti("TOPK", 10), ti("NL", 513) }, .{ 2560, r, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 10240 } }, .{ .shape = &.{ r, 10, 4 }, .dtype = mx.f32t } });
}
const kv = @import("kv_buffer.zig");
const ProjectionPlan = struct {
    // Packed arrays are owned by weights.flash_dense for the model's lifetime.
    groups: [4]@import("flash_lane.zig").Projection = undefined,
    group_count: usize = 0,
    members: [4]struct { group: usize, first: i32, end: i32 } = undefined,
    member_count: usize = 0,

    fn apply(p: ProjectionPlan, kernels: *mx.Kernels, s: *mx.Scope, x: A) !A {
        if (p.group_count == 1) return p.groups[0].apply(kernels, s, x);
        var projected: [4]A = undefined;
        for (p.groups[0..p.group_count], projected[0..p.group_count]) |group, *output| output.* = try group.apply(kernels, s, x);
        var outputs: [4]A = undefined;
        for (p.members[0..p.member_count], outputs[0..p.member_count]) |member, *output| output.* = try s.slice(projected[member.group], 1, member.first, member.end);
        return s.cat(outputs[0..p.member_count], -1);
    }
};
pub const Cache = struct {
    a: A = mx.empty,
    b: A = mx.empty,
    raw: A = mx.empty,
    pooled: A = mx.empty,
    ple: A = mx.empty,
    token_history: A = mx.empty,
    offset: i32 = 0,
    history: [2]i32 = .{ 248044, 248044 },
    keys: kv.Buffer = .{},
    values: kv.Buffer = .{},
    index_keys: kv.Buffer = .{},
    key_write: kv.Write = .{},
    value_write: kv.Write = .{},
    index_write: kv.Write = .{},
    pub fn deinit(c: *Cache) void {
        inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |f| mx.free(@field(c, f));
        c.keys.deinit();
        c.values.deinit();
        c.index_keys.deinit();
        c.* = .{};
    }
    pub fn clone(c: Cache) !Cache {
        var out = Cache{ .offset = c.offset, .history = c.history };
        errdefer out.deinit();
        inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |f| {
            const v = @field(c, f);
            @field(out, f) = if (v.ctx != null) try mx.retain(v) else mx.empty;
        }
        out.keys = try c.keys.clone();
        out.values = try c.values.clone();
        out.index_keys = try c.index_keys.clone();
        return out;
    }
    // Returned handles belong to the supplied scope, as do forward-pass records.
    pub fn attentionPrefix(c: Cache, s: *mx.Scope, end: i32) !Cache {
        if (end < 0 or end > c.offset) return error.InvalidCommit;
        return .{
            .a = try s.slice(c.a, 2, 0, end),
            .b = try s.slice(c.b, 2, 0, end),
            .raw = try s.slice(c.raw, 0, 0, end),
            // A rejected window may have crossed the sparse threshold. Below it,
            // serial decoding has no pooled cache, including on full rollback.
            .pooled = if (@divTrunc(end, 4) > 512 and c.pooled.ctx != null) try s.slice(c.pooled, 0, 0, @min(@divTrunc(end, 4), mx.dim(c.pooled, 0))) else mx.empty,
            .offset = end,
            .keys = try c.keys.prefix(s, end),
            .values = try c.values.prefix(s, end),
            .index_keys = try c.index_keys.prefix(s, end),
        };
    }
};
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    records: [48]Cache = @splat(.{}),
    tokens: [2048]i32 = undefined,
    prefilled: bool = false,
    count: usize = 0,
    start: i32 = 0,
    pub fn deinit(p: *Pass) void {
        p.scope.deinit();
    }
};
pub const Model = struct {
    round_owner: @import("decode_round.zig").Owner = .{},
    weights: cp.Store,
    kernels: mx.Kernels,
    projection_plans: std.StringHashMapUnmanaged(ProjectionPlan) = .empty,
    moe_plans: std.StringHashMapUnmanaged(CompiledMoE) = .empty,
    prefill_ops: @import("prefill_ops.zig").Ops = .{},
    cache: [48]Cache = @splat(.{}),
    position: i32 = 0,
    mtp: bool = false,
    centered: bool = true,
    trace_dir: ?[]const u8 = null,
    trace_layer: usize = 0,
    trace_gdn: ?usize = null,
    ngram: @import("ngram.zig").NGram = undefined,
    ple_tables: ?@import("ple_tables.zig").Tables = null,
    resident_ple: bool = false,
    wired_before: ?usize = null,
    resident_wired_bytes: usize = 0,
    pub const DraftCache = Cache;
    pub const SerialPass = Pass;
    pub const vocab = 248320;
    pub const max_shared_rows = 64;
    pub const adaptive_mtp_depth = false;
    pub fn eos(id: i32) bool {
        return id == 248044 or id == 248046;
    }
    pub fn gpuTokensEnabled(m: *const Model) bool {
        return m.resident_ple;
    }
    pub fn makeResidentPLE(m: *Model, wire: bool) !void {
        if (m.position != 0) return error.NonemptyCache;
        const device = mx.c.mlx_device_new_type(mx.c.MLX_GPU, 0);
        defer _ = mx.c.mlx_device_free(device);
        var info = mx.c.mlx_device_info_new();
        defer _ = mx.c.mlx_device_info_free(info);
        try mx.check(mx.c.mlx_device_info_get(&info, device));
        var recommended: usize = 0;
        try mx.check(mx.c.mlx_device_info_get_size(&recommended, info, "max_recommended_working_set_size"));
        if (wire and m.wired_before == null) {
            var previous: usize = 0;
            // Safetensors payloads are lazy. Materialize weights before measuring
            // their budget; counting active arrays immediately after init only
            // sees the small normalization/constants validation tensors.
            const arrays = try mx.allocator.alloc(A, m.weights.arrays.count());
            defer mx.allocator.free(arrays);
            var values = m.weights.arrays.iterator();
            var index: usize = 0;
            while (values.next()) |entry| {
                // These lazy shard handles are metadata for schema validation;
                // resident PLE is assembled separately from bounded file reads.
                if (std.mem.startsWith(u8, entry.key_ptr.*, "model.layers.1.ple.ple_embedding.ngram_embedding.")) continue;
                if (!m.mtp and std.mem.startsWith(u8, entry.key_ptr.*, "mtp.")) continue;
                arrays[index] = entry.value_ptr.*;
                index += 1;
            }
            try mx.evalMany(arrays[0..index], false);
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            try mx.check(mx.c.mlx_clear_cache());
            var weights_bytes: usize = 0;
            try mx.check(mx.c.mlx_get_active_memory(&weights_bytes));
            // Pin the existing model weights, before allocating the sparse PLE
            // tables. Leave room for verification states and pageable PLE rows.
            // Wiring the full recommendation can exhaust Metal memory at long
            // contexts even though the same short decode succeeds.
            const budget = @min(weights_bytes, recommended);
            try mx.check(mx.c.mlx_set_wired_limit(&previous, budget));
            m.wired_before = previous;
            m.resident_wired_bytes = budget;
        }
        try m.ple_tables.?.makeResident();
        m.resident_ple = true;
        std.debug.print("Resident Flash wired budget: {d} bytes; Metal recommended working set: {d}\n", .{ m.resident_wired_bytes, recommended });
    }
    pub fn init(io: std.Io, dir: []const u8, drafts: bool) !Model {
        var m = Model{ .weights = cp.Store.init(32), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        try @import("config.zig").flash(cfg.value);
        m.weights.flash_drafts = drafts;
        try m.weights.configure(bytes);
        try m.weights.load(io, dir, "language_model.");
        m.mtp = drafts;
        if (drafts and !m.weights.has("mtp.fc_hidden.weight")) return error.MissingDraftHead;
        try @import("schema.zig").validateConfig(.flash, &m.weights.arrays, drafts, cfg.value);
        try m.prepareConstants();
        // Match the Python loader's storage-convention check on all 48 HC anchors.
        var means: [48]f64 = undefined;
        var above: usize = 0;
        for (0..48) |i| {
            var s = mx.Scope{};
            defer s.deinit();
            const key = try std.fmt.bufPrint(&buf, "model.layers.{d}.attn_hyper_connection.hc_norm.weight", .{i});
            const array = try s.cast(try m.weights.get(key), mx.f32t);
            try mx.eval(array);
            const vals = mx.c.mlx_array_data_float32(array)[0..mx.c.mlx_array_size(array)];
            var sum: f64 = 0;
            for (vals) |v| sum += v;
            means[i] = sum / @as(f64, @floatFromInt(vals.len));
            if (means[i] > 0.5) above += 1;
        }
        std.mem.sort(f64, &means, {}, std.sort.asc(f64));
        const median = (means[23] + means[24]) / 2;
        if (above >= 44 and median >= 0.75 and median <= 1.5) m.centered = false else if (!(above <= 4 and median >= -0.5 and median <= 0.25)) return error.UnknownNormConvention;
        m.ngram = @import("ngram.zig").NGram.init();
        inline for (.{ .{ "layer_multipliers", "multipliers" }, .{ "ngram_heads_vocab_sizes", "sizes" }, .{ "ngram_heads_offsets", "offsets" } }) |entry| {
            const key = try std.fmt.bufPrint(&buf, "model.layers.1.ple.ple_embedding.{s}", .{entry[0]});
            const value = try m.weights.get(key);
            try mx.eval(value);
            const expected = &@field(m.ngram, entry[1]);
            if (mx.dtype(value) != mx.c.MLX_INT64 or mx.c.mlx_array_size(value) != expected.len or !std.mem.eql(i64, mx.c.mlx_array_data_int64(value)[0..expected.len], expected)) return error.NGramConstantsMismatch;
        }
        m.ple_tables = try @import("ple_tables.zig").Tables.init(io, dir);
        return m;
    }
    fn prepareConstants(m: *Model) !void {
        var scope = mx.Scope{};
        defer scope.deinit();
        const values = [_]A{ try scope.scalar(1e-6), try scope.scalar(@log2(@as(f32, 10000000))), try scope.scalar(0.0625) };
        try mx.evalMany(&values, false);
        inline for (.{ "decode.eps", "decode.log_base", "decode.attention_scale" }, 0..) |name, i| try m.weights.put(name, values[i]);
    }
    pub fn reset(m: *Model) void {
        for (&m.cache) |*c| c.deinit();
        m.position = 0;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        if (m.ple_tables) |*tables| tables.deinit();
        var plans = m.projection_plans.keyIterator();
        while (plans.next()) |key| mx.allocator.free(key.*);
        m.projection_plans.deinit(mx.allocator);
        var moes = m.moe_plans.iterator();
        while (moes.next()) |entry| {
            entry.value_ptr.deinit();
            mx.allocator.free(entry.key_ptr.*);
        }
        m.moe_plans.deinit(mx.allocator);
        m.weights.deinit();
        m.kernels.deinit();
        m.prefill_ops.deinit();
        if (m.wired_before) |previous| {
            _ = mx.c.mlx_synchronize(mx.stream);
            var ignored: usize = 0;
            _ = mx.c.mlx_set_wired_limit(&ignored, previous);
        }
    }
    pub fn lin(m: *Model, s: *mx.Scope, base: []const u8, suffix: []const u8, x: A) !A {
        var buf: [256]u8 = undefined;
        return m.projectNamed(s, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, suffix }), x);
    }
    fn projectNamed(m: *Model, s: *mx.Scope, name: []const u8, x: A) !A {
        if (mx.tensor_units and m.weights.flash_drafts != null) if (m.weights.flash_dense.get(name)) |projection| return projection.apply(&m.kernels, s, x);
        return m.weights.linear(&m.kernels, s, name, x, true);
    }
    pub fn embedResidual(m: *Model, s: *mx.Scope, tokens: A) !A {
        if (tokens.ctx == null or mx.shape(tokens).len != 1 or (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32)) return error.InvalidToken;
        const rows = mx.dim(tokens, 0);
        if (rows < 1 or rows > max_shared_rows) return error.InvalidLaneWidth;
        const name = "model.embed_tokens";
        if (try m.weights.format(name)) |format| if (format.bits == 4 and format.group_size == 32) {
            const weight = try m.weights.affine(name);
            const shape = try weight.geometry(2);
            if (shape.k != 2560 or shape.n != vocab) return error.InvalidTensorShape;
            return (try m.kernels.run(s, src.q4_embed_rows, &.{ tokens, weight.arrays[0], weight.arrays[1], weight.arrays[2] }, &.{ ti("DIMS", 2560), ti("TILE", 4) }, .{ 2560, rows, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ rows, 10240 } }}))[0];
        };
        const e = try m.weights.embedArray(s, name, tokens);
        return s.cat(&.{ e, e, e, e }, -1);
    }
    fn trace(m: *Model, s: *mx.Scope, label: []const u8, value: A) !void {
        const dir = m.trace_dir orelse return;
        var buf: [4096]u8 = undefined;
        const path = if (m.trace_gdn) |layer_index| blk: {
            if (m.trace_layer != layer_index or (!std.mem.eql(u8, label, "mixed") and !std.mem.eql(u8, label, "branch"))) return;
            break :blk try std.fmt.bufPrint(&buf, "{s}/{d:0>6}-{d:0>2}-{s}.npy", .{ dir, @as(usize, @intCast(m.position)), m.trace_layer, label });
        } else try std.fmt.bufPrint(&buf, "{s}/{d:0>2}-{s}.npy", .{ dir, m.trace_layer, label });
        const zpath = try mx.allocator.dupeSentinel(u8, path, 0);
        defer mx.allocator.free(zpath);
        const value_f32 = try s.cast(value, mx.f32t);
        try mx.eval(value_f32);
        try mx.check(mx.c.mlx_save(zpath, value_f32));
    }
    pub fn f(m: *Model, base: []const u8, suffix: []const u8) !A {
        return m.weights.field(base, suffix);
    }
    pub fn scale(m: *Model, s: *mx.Scope, base: []const u8, suffix: []const u8) !A {
        var buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "{s}.{s}.native_scale", .{ base, suffix });
        if (m.weights.arrays.get(key)) |v| return v;
        var value = try s.cast(try m.f(base, suffix), mx.f32t);
        if (m.centered) value = try s.binary(mx.c.mlx_add, value, try s.scalar(1));
        try mx.eval(value);
        try m.weights.put(key, value);
        return m.weights.get(key);
    }
    fn centeredNorm(m: *Model, s: *mx.Scope, x: A, name: []const u8, group: i32) !A {
        const width = mx.dim(x, -1);
        const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(width)));
        const sc = try m.scale(s, name, "weight");
        return (try m.kernels.run(s, src.q4_rms_rows, &.{ try s.reshape(x, &.{ rows, width }), sc, try m.weights.get("decode.eps") }, &.{ ti("W", width), ti("G", group), ti("SW", mx.dim(sc, -1)) }, .{ 1024 * @divExact(width, group), rows, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{ rows, width } }}))[0];
    }
    fn pleNorm(m: *Model, s: *mx.Scope, x: A, name: []const u8) !A {
        // PLE uses the original CenteredRMSNorm's separate square/mean operations.
        // The fused MTP RMS kernel changes fp32 reduction rounding before BF16 storage.
        const y = try s.reshape(try s.cast(x, mx.f32t), &.{ mx.dim(x, 0), 4, 2560 });
        var mean = mx.c.mlx_array_new();
        const rc = mx.c.mlx_mean_axis(&mean, try s.unary(mx.c.mlx_square, y), -1, true, mx.stream);
        const rinv = try s.unary(mx.c.mlx_rsqrt, try s.binary(mx.c.mlx_add, try s.result(rc, mean), try m.weights.get("decode.eps")));
        const sc = try s.reshape(try m.scale(s, name, "weight"), &.{ 4, 2560 });
        return s.cast(try s.reshape(try s.binary(mx.c.mlx_multiply, try s.binary(mx.c.mlx_multiply, y, rinv), sc), mx.shape(x)), mx.dtype(x));
    }
    pub fn hcNorm(m: *Model, s: *mx.Scope, h: A, branch: ?A, inject: A) ![2]A {
        const r = mx.dim(h, 0);
        const out = if (branch) |b| try m.kernels.run(s, src.q4_hc_norm_plain, &.{ h, inject, b }, &.{ ti("S", 4), ti("D", 2560) }, .{ 2560, r, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 10240 } }, .{ .shape = &.{ r, 10, 4 }, .dtype = mx.f32t } }) else try m.kernels.run(s, src.q4_hc_norm_none, &.{h}, &.{ ti("S", 4), ti("D", 2560) }, .{ 2560, r, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 10240 } }, .{ .shape = &.{ r, 10, 4 }, .dtype = mx.f32t } });
        return out[0..2].*;
    }
    pub fn hcWeights(m: *Model, s: *mx.Scope, base: []const u8, inject: bool) !struct { down: @import("flash_ops.zig").Weight, up: @import("flash_ops.zig").Weight, scale: A } {
        var buf: [256]u8 = undefined;
        const stacked = try std.fmt.bufPrint(&buf, "{s}.native_down", .{base});
        if (!m.weights.has(stacked)) {
            const down = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_down", .{base}));
            const inj = if (inject) try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.block_inject_weight", .{base})) else down;
            const parts = [_]@import("flash_ops.zig").Weight{ down, inj };
            const combined = try @import("flash_ops.zig").stack(s, parts[0..if (inject) @as(usize, 2) else 1]);
            try mx.evalMany(&combined.arrays, false);
            try m.weights.putAffine(try std.fmt.bufPrint(&buf, "{s}.native_down", .{base}), combined);
            try m.weights.put(try std.fmt.bufPrint(&buf, "{s}.native_down", .{base}), combined.arrays[0]);
        }
        const down = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.native_down", .{base}));
        const up = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.input_mix_weight_up", .{base}));
        const sc = try m.scale(s, base, "hc_norm.weight");
        return .{ .down = down, .up = up, .scale = sc };
    }
    pub fn hcProject(m: *Model, s: *mx.Scope, base: []const u8, h: A, ssp: A, inject: bool) ![5]A {
        const w = try m.hcWeights(s, base, inject);
        const out = if (mx.tensor_units) try @import("flash_lane.zig").hyper(&m.kernels, s, h, ssp, w.down, w.up, w.scale, try m.weights.get("decode.eps"), 4, 320) else try @import("flash_ops.zig").hyper(&m.kernels, s, h, ssp, w.down, w.up, w.scale, try m.weights.get("decode.eps"), 4, 320, mx.gpu_generation);
        return .{ out[0], out[1], mx.empty, mx.empty, mx.empty };
    }
    pub fn projectStack(m: *Model, s: *mx.Scope, base: []const u8, names: []const []const u8, x: A) !A {
        return m.projectStackNamed(s, base, names, x, "native_decode_stack");
    }
    pub fn projectCacheStack(m: *Model, s: *mx.Scope, base: []const u8, x: A) !A {
        var buf: [256]u8 = undefined;
        const key = try std.fmt.bufPrint(&buf, "{s}.native_index_key", .{base});
        if (!m.weights.has(key)) {
            var name: [256]u8 = undefined;
            var selected = try m.weights.affine(try std.fmt.bufPrint(&name, "{s}.indexer.index_qk_proj", .{base}));
            if ((try selected.geometry(2)).n != 640) return error.InvalidProjectionGroup;
            for (&selected.arrays) |*array| array.* = try s.slice(array.*, 0, 512, 640);
            try mx.evalMany(&selected.arrays, false);
            try m.weights.putAffine(key, selected);
            try m.weights.put(key, selected.arrays[0]);
        }
        return m.projectStackNamed(s, base, &.{ "k_proj", "v_proj", "native_index_key" }, x, "native_absorb_stack");
    }
    fn projectStackNamed(m: *Model, s: *mx.Scope, base: []const u8, names: []const []const u8, x: A, label: []const u8) !A {
        if (names.len == 0 or names.len > 4) return error.InvalidProjectionGroup;
        var plan_buffer: [512]u8 = undefined;
        var plan_key: []const u8 = "";
        var cache_plan = false;
        var plan = ProjectionPlan{ .member_count = names.len };
        if (mx.tensor_units and m.weights.flash_drafts != null) {
            plan_key = try std.fmt.bufPrint(&plan_buffer, "{s}.{s}|{s}|{s}|{s}|{s}", .{ base, label, names[0], if (names.len > 1) names[1] else "", if (names.len > 2) names[2] else "", if (names.len > 3) names[3] else "" });
            if (m.projection_plans.get(plan_key)) |cached| return cached.apply(&m.kernels, s, x);
            cache_plan = m.projection_plans.count() < 64;
        }
        var parts: [4]@import("flash_ops.zig").Weight = undefined;
        var outputs: [4]A = undefined;
        var done: [4]bool = @splat(false);
        var buf: [256]u8 = undefined;
        for (names, 0..) |name, i| {
            if (!mx.tensor_units) {
                outputs[i] = try m.lin(s, base, name, x);
            } else parts[i] = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, name }));
        }
        if (mx.tensor_units) for (names, 0..) |_, i| {
            if (done[i]) continue;
            var group: [4]@import("flash_ops.zig").Weight = undefined;
            var count: usize = 0;
            for (parts[0..names.len]) |part| if (part.format.bits == parts[i].format.bits) {
                group[count] = part;
                count += 1;
            };
            const key = try std.fmt.bufPrint(&buf, "{s}.{s}_{d}", .{ base, label, parts[i].format.bits });
            if (!m.weights.has(key)) {
                const combined = try @import("flash_ops.zig").stack(s, group[0..count]);
                try mx.evalMany(&combined.arrays, false);
                try m.weights.putAffine(key, combined);
                try m.weights.put(key, combined.arrays[0]);
            }
            const projected = try m.projectNamed(s, key, x);
            const group_index = plan.group_count;
            if (cache_plan) {
                plan.groups[group_index] = m.weights.flash_dense.get(key).?;
                plan.group_count += 1;
            }
            if (count == names.len) {
                if (cache_plan) try m.cacheProjectionPlan(plan_key, plan);
                return projected;
            }
            var at: i32 = 0;
            for (parts[0..names.len], 0..) |part, j| if (part.format.bits == parts[i].format.bits) {
                const n = (try part.geometry(2)).n;
                outputs[j] = try s.slice(projected, 1, at, at + n);
                if (cache_plan) plan.members[j] = .{ .group = group_index, .first = at, .end = at + n };
                at += n;
                done[j] = true;
            };
        };
        const output = try s.cat(outputs[0..names.len], -1);
        if (cache_plan) try m.cacheProjectionPlan(plan_key, plan);
        return output;
    }
    fn cacheProjectionPlan(m: *Model, key: []const u8, plan: ProjectionPlan) !void {
        const name = try mx.allocator.dupe(u8, key);
        errdefer mx.allocator.free(name);
        try m.projection_plans.put(mx.allocator, name, plan);
    }
    fn gdn(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: Cache, record: *Cache) !A {
        const r = mx.dim(x, 0);
        const p = try m.projectStack(s, base, &.{ "in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a" }, x);
        const cs = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 3, 10240 }, mx.bf16);
        const state = if (cache.b.ctx != null) cache.b else try s.zeros(&.{1}, mx.f32t);
        const out = try m.kernels.run(s, src.q4_gdn_step, &.{ p, cs, state, try s.reshape(try m.f(base, "conv1d.weight"), &.{ 10240, 4 }), try m.f(base, "A_log"), try m.f(base, "dt_bias"), try m.f(base, "norm.weight"), try m.weights.get("decode.eps"), try s.ints(&.{r}) }, &.{ ti("NK", 16), ti("NV", 48), ti("DK", 128), ti("DV", 128), ti("TAPS", 4), ti("HAS_STATE", @intFromBool(cache.b.ctx != null)) }, .{ 48 * 1024, 1, 1 }, .{ 1024, 1, 1 }, &.{ .{ .shape = &.{ r, 6144 } }, .{ .shape = &.{ r, 3, 10240 } }, .{ .shape = &.{ r, 48, 128, 128 }, .dtype = mx.f32t } });
        record.a = out[1];
        record.b = out[2];
        return m.lin(s, base, "out_proj", out[0]);
    }
    pub fn attention(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: *Cache, record: *Cache) !A {
        const r = mx.dim(x, 0);
        const start = cache.offset;
        const end = start + r;
        const p = try m.projectStack(s, base, &.{ "q_proj", "k_proj", "v_proj", "indexer.index_qk_proj" }, x);
        var pos: [16]i32 = undefined;
        for (0..@intCast(r)) |j| pos[j] = start + @as(i32, @intCast(j));
        const prep = try m.kernels.run(s, src.q4_attn_prep, &.{ p, try s.ints(pos[0..@intCast(r)]), try m.scale(s, base, "q_norm.weight"), try m.scale(s, base, "k_norm.weight"), try m.scale(s, base, "indexer.q_layernorm.weight"), try m.weights.get("decode.eps"), try m.weights.get("decode.log_base") }, &.{ ti("NQ", 24), ti("NKV", 2), ti("HD", 256), ti("RD", 64), ti("PW", 13952), ti("NI", 4), ti("IHD", 128) }, .{ 256, 30, r }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 24, 256 } }, .{ .shape = &.{ r, 2, 256 } }, .{ .shape = &.{ r, 4, 128 } } });
        var keys = try s.transpose(try s.reshape(prep[1], &.{ 1, r, 2, 256 }), &.{ 0, 2, 1, 3 });
        var values = try s.transpose(try s.reshape(try s.slice(p, 1, 12800, 13312), &.{ 1, r, 2, 256 }), &.{ 0, 2, 1, 3 });
        var raw = try s.slice(p, 1, 13824, 13952);
        if (kv.enabled) {
            record.key_write = try cache.keys.append(s, cache.a, keys, 2);
            record.value_write = try cache.values.append(s, cache.b, values, 2);
            record.index_write = try cache.index_keys.append(s, cache.raw, raw, 0);
            keys = record.key_write.view;
            values = record.value_write.view;
            raw = record.index_write.view;
        } else if (cache.a.ctx != null) {
            keys = try s.cat(&.{ cache.a, keys }, 2);
            values = try s.cat(&.{ cache.b, values }, 2);
            raw = try s.cat(&.{ cache.raw, raw }, 0);
        }
        record.a = keys;
        record.b = values;
        record.raw = raw;
        record.offset = end;
        record.pooled = if (cache.pooled.ctx != null) try s.own(try mx.retain(cache.pooled)) else mx.empty;
        var ids = try s.zeros(&.{ r, 1 }, mx.i32t);
        var counts: [16]i32 = undefined;
        var sparse: [16]i32 = undefined;
        var complete: [16]i32 = undefined;
        var ends: [16]i32 = undefined;
        for (0..@intCast(r)) |j| {
            ends[j] = start + @as(i32, @intCast(j)) + 1;
            complete[j] = @divTrunc(ends[j], 4);
            sparse[j] = @intFromBool(complete[j] > 512);
            counts[j] = if (sparse[j] != 0) 2048 + @mod(ends[j], 4) else ends[j];
        }
        if (@divTrunc(end, 4) > 512) {
            const done = if (cache.pooled.ctx != null) mx.dim(cache.pooled, 0) else 0;
            const blocks = @divTrunc(end, 4);
            var pooled = cache.pooled;
            if (blocks > done) {
                const fresh = (try m.kernels.run(s, src.q4_idx_pool, &.{ raw, try s.ints(&.{done}), try m.scale(s, base, "indexer.k_layernorm.weight"), try m.weights.get("decode.eps"), try m.weights.get("decode.log_base") }, &.{ ti("DI", 128), ti("RD", 64) }, .{ 128, blocks - done, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ blocks - done, 128 } }}))[0];
                pooled = if (done > 0) try s.cat(&.{ pooled, fresh }, 0) else fresh;
            }
            record.pooled = pooled;
            const ca = try s.ints(complete[0..@intCast(r)]);
            const block_group: i32 = if (r == 1 or blocks < 4096) 1 else if (blocks < 8192) 2 else if (blocks < 16384) 4 else 8;
            const scores = (try m.kernels.run(s, src.q4_idx_scores, &.{ prep[2], pooled, ca }, &.{ ti("HI", 4), ti("DI", 128), ti("TOP", 512), ti("BB", block_group), ti("RB", 8) }, .{ @divTrunc(blocks + 8 * block_group - 1, 8 * block_group) * 256, @divTrunc(r + 7, 8), 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ r, blocks }, .dtype = mx.f32t }}))[0];
            ids = (try m.kernels.run(s, src.q4_idx_select, &.{ scores, ca, try s.ints(ends[0..@intCast(r)]) }, &.{ ti("TOP", 512), ti("KW", 2051) }, .{ 1024 * r, 1, 1 }, .{ 1024, 1, 1 }, &.{.{ .shape = &.{ r, 2051 }, .dtype = mx.i32t }}))[0];
        }
        // The shader has separate physical-capacity and logical-key-count inputs.
        // Passing full storage avoids its contiguous adapter copying a prefix view.
        const kc = if (kv.enabled) record.key_write.capacity else keys;
        const vc = if (kv.enabled) record.value_write.capacity else values;
        const partial = try m.kernels.run(s, src.q4_attn_parts, &.{ prep[0], kc, vc, ids, try s.ints(counts[0..@intCast(r)]), try s.ints(sparse[0..@intCast(r)]), try m.weights.get("decode.attention_scale") }, &.{ ti("H", 24), ti("KVH", 2), ti("D", 256), ti("P", 16) }, .{ 256 * 24, r, 16 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 24, 16, 256 }, .dtype = mx.f32t }, .{ .shape = &.{ r, 24, 16, 2 }, .dtype = mx.f32t } });
        const att = (try m.kernels.run(s, src.q4_attn_merge, &.{ partial[0], partial[1] }, &.{ ti("H", 24), ti("D", 256), ti("P", 16) }, .{ 256, 24, r }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ r, 24, 256 } }}))[0];
        const gated = (try m.kernels.run(s, src.q4_attn_gate, &.{ att, p }, &.{ ti("NQ", 24), ti("HD", 256), ti("PW", 13952) }, .{ r * 6144, 1, 1 }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ r, 6144 } }}))[0];
        return m.lin(s, base, "o_proj", gated);
    }
    pub fn moeInputs(m: *Model, s: *mx.Scope, base: []const u8) !MoEInputs {
        var buf: [256]u8 = undefined;
        const router_key = try std.fmt.bufPrint(&buf, "{s}.native_router", .{base});
        if (!m.weights.has(router_key)) {
            var nb: [256]u8 = undefined;
            const sg = try m.weights.dequant(s, try std.fmt.bufPrint(&nb, "{s}.shared_expert_gate", .{base}));
            const router = try s.cat(&.{ try m.f(base, "gate.weight"), sg }, 0);
            try mx.eval(router);
            try m.weights.put(router_key, router);
        }
        const router = try m.weights.get(router_key);
        return .{
            .router = router,
            .gate = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.gate_proj", .{base})),
            .up = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.up_proj", .{base})),
            .shared_gate = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.shared_expert.gate_proj", .{base})),
            .shared_up = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.shared_expert.up_proj", .{base})),
            .down = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.down_proj", .{base})),
            .shared_down = try m.weights.affine(try std.fmt.bufPrint(&buf, "{s}.shared_expert.down_proj", .{base})),
        };
    }
    pub fn moeDirect(m: *Model, s: *mx.Scope, base: []const u8, h: A, x: A, inject: A) ![5]A {
        return moeGraph(&m.kernels, s, try m.moeInputs(s, base), h, x, inject, mx.gpu_generation);
    }
    pub fn moe(m: *Model, s: *mx.Scope, base: []const u8, h: A, x: A, inject: A) ![5]A {
        if (mx.tensor_units and m.weights.flash_drafts != null and m.trace_dir == null) {
            if (m.moe_plans.getPtr(base)) |plan| {
                if (plan.generation == mx.gpu_generation) return plan.call(&m.kernels, s, h, x, inject);
            } else if (m.moe_plans.count() < 49) {
                var setup = mx.Scope{};
                defer setup.deinit();
                var plan = try CompiledMoE.init(try m.moeInputs(&setup, base));
                var stored = false;
                errdefer if (!stored) plan.deinit();
                const key = try mx.allocator.dupe(u8, base);
                errdefer if (!stored) mx.allocator.free(key);
                try m.moe_plans.put(mx.allocator, key, plan);
                stored = true;
                return m.moe_plans.getPtr(base).?.call(&m.kernels, s, h, x, inject);
            }
        }
        return m.moeDirect(s, base, h, x, inject);
    }
    fn ple(m: *Model, s: *mx.Scope, h: A, tokens: []const i32, cache: Cache, record: *Cache) !A {
        var hist = cache.history;
        var ids: [16 * 16]i64 = undefined;
        for (tokens, 0..) |token, row| {
            @memcpy(ids[row * 16 ..][0..16], &m.ngram.ids(hist, token));
            hist = .{ hist[1], token };
        }
        const r: i32 = @intCast(tokens.len);
        const emb = try s.reshape(try m.ple_tables.?.gather(s, ids[0 .. tokens.len * 16]), &.{ r, 2560 });
        record.history = hist;
        return m.pleEmbedding(s, h, emb, r, cache, record);
    }
    fn pleArray(m: *Model, s: *mx.Scope, h: A, tokens: A, cache: Cache, record: *Cache) !A {
        const previous = if (cache.token_history.ctx != null) cache.token_history else try s.ints(&.{ 248044, 248044 });
        const joined = try s.cat(&.{ previous, try s.cast(tokens, mx.i32t) }, 0);
        record.token_history = joined;
        const ids = try m.ngram.idsArray(s, joined);
        const emb = try m.ple_tables.?.resident.?.gather(&m.kernels, s, ids);
        return m.pleEmbedding(s, h, emb, mx.dim(tokens, 0), cache, record);
    }
    fn pleEmbedding(m: *Model, s: *mx.Scope, h: A, emb: A, r: i32, cache: Cache, record: *Cache) !A {
        if (mx.dim(h, 0) != r) return error.InvalidTensorShape;
        const out = try m.pleGate(s, h, emb);
        return m.pleConv(s, h, out[0], out[1], cache, record);
    }
    pub fn pleGate(m: *Model, s: *mx.Scope, h: A, emb: A) ![2]A {
        const base = "model.layers.1.ple";
        const r = mx.dim(h, 0);
        if (mx.tensor_units) {
            const keys = try s.reshape(try m.pleNorm(s, try m.lin(s, base, "key_proj", emb), base ++ ".norm_key"), &.{ r, 4, 2560 });
            const queries = try s.reshape(try m.pleNorm(s, h, base ++ ".norm_query"), &.{ r, 4, 2560 });
            const values = try m.lin(s, base, "value_proj", emb);
            var sum = mx.c.mlx_array_new();
            const rc = mx.c.mlx_sum_axis(&sum, try s.binary(mx.c.mlx_multiply, keys, queries), -1, true, mx.stream);
            var gate = try s.binary(mx.c.mlx_divide, try s.result(rc, sum), try s.cast(try s.scalar(@sqrt(@as(f32, 2560))), mx.bf16));
            gate = try s.binary(mx.c.mlx_multiply, try s.unary(mx.c.mlx_sign, gate), try s.unary(mx.c.mlx_sqrt, try s.binary(mx.c.mlx_maximum, try s.unary(mx.c.mlx_abs, gate), try s.cast(try m.weights.get("decode.eps"), mx.bf16))));
            const gated = try s.reshape(try s.binary(mx.c.mlx_multiply, try s.unary(mx.c.mlx_sigmoid, gate), try s.reshape(values, &.{ r, 1, 2560 })), &.{ r, 10240 });
            const normed = try m.pleNorm(s, gated, base ++ ".norm_conv");
            return .{ gated, normed };
        }
        const kv_rows = try s.cat(&.{ try m.lin(s, base, "key_proj", emb), try m.lin(s, base, "value_proj", emb) }, -1);
        const ops = @import("flash_ops.zig");
        return ops.pleGate(&m.kernels, s, kv_rows, h, .{ try m.scale(s, base ++ ".norm_key", "weight"), try m.scale(s, base ++ ".norm_query", "weight"), try m.scale(s, base ++ ".norm_conv", "weight") }, try m.weights.get("decode.eps"), 4);
    }
    pub fn pleConv(m: *Model, s: *mx.Scope, h: A, gated: A, normed: A, cache: Cache, record: *Cache) !A {
        const base = "model.layers.1.ple";
        const r = mx.dim(h, 0);
        const tail = if (cache.ple.ctx != null) cache.ple else try s.zeros(&.{ 9, 10240 }, mx.bf16);
        const conv_in = try s.cat(&.{ tail, normed }, 0);
        record.ple = conv_in;
        if (mx.tensor_units) {
            var conv = mx.c.mlx_array_new();
            const cr = mx.c.mlx_conv1d(&conv, try s.reshape(conv_in, &.{ 1, r + 9, 10240 }), try m.f(base, "conv1d.weight"), 1, 0, 3, 10240, mx.stream);
            const branch = try s.binary(mx.c.mlx_add, gated, try s.reshape(try m.prefill_ops.call(s, .silu, &.{try s.result(cr, conv)}), &.{ r, 10240 }));
            return s.binary(mx.c.mlx_add, h, branch);
        }
        const weight = try s.cast(try s.reshape(try m.f(base, "conv1d.weight"), &.{ 10240, 4 }), mx.f32t);
        return @import("flash_ops.zig").pleConv(&m.kernels, s, conv_in, weight, gated, h, 4, 3);
    }
    fn layer(m: *Model, s: *mx.Scope, base: []const u8, hn: [2]A, cache: *Cache, record: *Cache, linear: bool) ![2]A {
        var buf: [256]u8 = undefined;
        const mix = try m.hcProject(s, try std.fmt.bufPrint(&buf, "{s}.attn_hyper_connection", .{base}), hn[0], hn[1], true);
        try m.trace(s, "mixed", mix[0]);
        const branch = if (linear) try m.gdn(s, try std.fmt.bufPrint(&buf, "{s}.linear_attn", .{base}), mix[0], cache.*, record) else try m.attention(s, try std.fmt.bufPrint(&buf, "{s}.self_attn", .{base}), mix[0], cache, record);
        try m.trace(s, "branch", branch);
        const post = try m.hcNorm(s, hn[0], branch, mix[1]);
        const mm = try m.hcProject(s, try std.fmt.bufPrint(&buf, "{s}.mlp_hyper_connection", .{base}), post[0], post[1], true);
        try m.trace(s, "moe-input", mm[0]);
        const out = try m.moe(s, try std.fmt.bufPrint(&buf, "{s}.mlp", .{base}), post[0], mm[0], mm[1]);
        try m.trace(s, "output", out[0]);
        return out[0..2].*;
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        var p = try m.forwardQueued(tokens);
        errdefer p.deinit();
        try mx.eval(p.logits);
        try observeBuffers(&p);
        return p;
    }
    pub fn forwardStreams(m: *Model, streams: []const @import("flash_shared.zig").Stream) !@import("flash_shared.zig").Pass {
        return @import("flash_shared.zig").forward(m, streams);
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len > 16) return @import("flash_prefill.zig").forward(m, tokens);
        if (m.trace_dir != null) return m.forward(tokens);
        if (tokens.len == 0) return error.InvalidLaneWidth;
        for (tokens) |token| if (token < 0 or token >= vocab) return error.InvalidToken;
        var p = Pass{ .count = tokens.len, .start = m.position };
        errdefer p.deinit();
        @memcpy(p.tokens[0..tokens.len], tokens);
        const ids = try p.scope.ints(tokens);
        var hn: [2]A = @splat(mx.empty);
        defer for (hn) |value| mx.free(value);
        {
            var s = mx.Scope{};
            defer s.deinit();
            hn = try retainPair(try m.hcNorm(&s, try m.embedResidual(&s, ids), null, mx.empty));
        }
        var buf: [256]u8 = undefined;
        for (0..48) |i| {
            var s = mx.Scope{};
            defer s.deinit();
            m.trace_layer = i;
            var input = hn;
            if (i == 1) {
                const h = if (m.gpuTokensEnabled()) try m.pleArray(&s, hn[0], ids, m.cache[i], &p.records[i]) else try m.ple(&s, hn[0], tokens, m.cache[i], &p.records[i]);
                input = try m.hcNorm(&s, h, null, mx.empty);
            }
            const next = try retainPair(try m.layer(&s, try std.fmt.bufPrint(&buf, "model.layers.{d}", .{i}), input, &m.cache[i], &p.records[i], i % 4 != 3));
            for (hn) |value| mx.free(value);
            hn = next;
            const record = &p.records[i];
            inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
                const value = @field(record, field);
                if (value.ctx != null) @field(record, field) = try p.scope.own(try mx.retain(value));
            }
            inline for (.{ "key_write", "value_write", "index_write" }) |field| {
                const write = &@field(record, field);
                inline for (.{ "capacity", "added", "view" }) |part| {
                    const value = @field(write, part);
                    if (value.ctx != null) @field(write, part) = try p.scope.own(try mx.retain(value));
                }
            }
            try mx.evalMany(&.{hn[0]}, true);
        }
        const last: i32 = @intCast(tokens.len - 1);
        p.hidden = try p.scope.own(try mx.retain(hn[0]));
        p.logits = try m.headWithNorm(&p.scope, .{ try p.scope.slice(hn[0], 0, last, last + 1), try p.scope.slice(hn[1], 0, last, last + 1) });
        try mx.eval(p.logits);
        try observeBuffers(&p);
        return p;
    }
    fn retainPair(values: [2]A) ![2]A {
        const first = try mx.retain(values[0]);
        errdefer mx.free(first);
        return .{ first, try mx.retain(values[1]) };
    }
    pub fn observeBuffers(p: *Pass) !void {
        if (!kv.track_reuse) return;
        for (p.records) |rec| {
            try kv.observe(rec.key_write);
            try kv.observe(rec.value_write);
            try kv.observe(rec.index_write);
        }
    }
    pub fn forwardQueued(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16) return error.InvalidLaneWidth;
        for (tokens) |token| if (token < 0 or token >= vocab) return error.InvalidToken;
        var s = mx.Scope{};
        defer s.deinit();
        return m.forwardImpl(try s.ints(tokens), tokens);
    }
    pub fn forwardArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null or (mx.dtype(tokens) != mx.i32t and mx.dtype(tokens) != mx.c.MLX_UINT32)) return error.InvalidToken;
        if (mx.shape(tokens).len != 1 or mx.dim(tokens, 0) < 1 or mx.dim(tokens, 0) > 16) return error.InvalidLaneWidth;
        if (!m.gpuTokensEnabled()) return error.RequiresResidentPLE;
        return m.forwardImpl(tokens, null);
    }
    pub fn forwardSerialArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null or mx.c.mlx_array_size(tokens) != 1) return error.InvalidToken;
        return m.forwardArray(tokens);
    }
    fn forwardImpl(m: *Model, tokens: A, host: ?[]const i32) !Pass {
        var p = Pass{ .count = @intCast(mx.dim(tokens, 0)), .start = m.position };
        errdefer p.deinit();
        if (host) |ids| @memcpy(p.tokens[0..ids.len], ids);
        const s = &p.scope;
        var hn = try m.hcNorm(s, try m.embedResidual(s, tokens), null, mx.empty);
        var buf: [256]u8 = undefined;
        for (0..48) |i| {
            m.trace_layer = i;
            if (i == 1) {
                const h = if (m.gpuTokensEnabled()) try m.pleArray(s, hn[0], tokens, m.cache[i], &p.records[i]) else try m.ple(s, hn[0], host.?, m.cache[i], &p.records[i]);
                hn = try m.hcNorm(s, h, null, mx.empty);
            }
            try m.trace(s, "input", hn[0]);
            hn = try m.layer(s, try std.fmt.bufPrint(&buf, "model.layers.{d}", .{i}), hn, &m.cache[i], &p.records[i], i % 4 != 3);
            try mx.evalMany(&.{hn[0]}, true);
        }
        p.hidden = hn[0];
        p.logits = try m.headWithNorm(s, hn);
        return p;
    }
    pub fn head(m: *Model, s: *mx.Scope, h: A) !A {
        return m.headWithNorm(s, try m.hcNorm(s, h, null, mx.empty));
    }
    pub fn headWithNorm(m: *Model, s: *mx.Scope, hn: [2]A) !A {
        const mixed = try m.hcProject(s, "model.hyper_connection_mixer", hn[0], hn[1], false);
        try m.trace(s, "head-mixed", mixed[0]);
        const logits = try m.projectNamed(s, "lm_head", mixed[0]);
        try m.trace(s, "logits", logits);
        return logits;
    }
    pub fn draftHead(m: *Model, s: *mx.Scope, h: A) !A {
        const hn = try m.hcNorm(s, h, null, mx.empty);
        const mixed = try m.hcProject(s, "mtp.hyper_connection_mixer", hn[0], hn[1], false);
        return m.projectNamed(s, if (m.weights.has("draft_ids")) "draft_lm_head" else "lm_head", mixed[0]);
    }
    pub const draft_vocabulary = @import("draft_vocab.zig").data.flash;
    pub const draft_prior = &@import("draft_depth.zig").flash_prior;
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        return m.commitImpl(p, keep, true);
    }
    pub fn commitSerialQueued(m: *Model, p: *Pass) !void {
        if (p.count != 1) return error.InvalidCommit;
        return m.commitImpl(p, 1, false);
    }
    pub fn committedCache(m: *Model, p: *Pass, old: []const Cache, keep: usize) ![48]Cache {
        if (keep == 0 or keep > p.count or old.len != 48) return error.InvalidCommit;
        if (p.prefilled and keep != p.count) return error.InvalidCommit;
        var next: [48]Cache = @splat(.{});
        errdefer for (&next) |*c| c.deinit();
        for (&next, old, 0..) |*target, cache, layer_index| target.* = try m.committedCacheLayer(&p.scope, p, cache, layer_index, keep);
        return next;
    }
    pub fn committedCacheLayer(m: *Model, s: *mx.Scope, p: *const Pass, old: Cache, layer_index: usize, keep: usize) !Cache {
        if (layer_index >= 48 or keep == 0 or keep > p.count or (p.prefilled and keep != p.count)) return error.InvalidCommit;
        const n: i32 = @intCast(keep);
        const end = std.math.add(i32, p.start, n) catch return error.InvalidCommit;
        const rec = p.records[layer_index];
        var next = Cache{ .offset = end };
        errdefer next.deinit();
        if (layer_index % 4 != 3) {
            const conv = if (p.prefilled or (n == 1 and mx.dim(rec.a, 0) == 1)) rec.a else try s.slice(rec.a, 0, n - 1, n);
            const state = if (p.prefilled or (n == 1 and mx.dim(rec.b, 0) == 1)) rec.b else try s.slice(rec.b, 0, n - 1, n);
            next.a = try mx.retain(if (p.prefilled) conv else try s.reshape(conv, &.{ 3, 10240 }));
            next.b = try mx.retain(if (p.prefilled) state else try s.reshape(state, &.{ 48, 128, 128 }));
        } else {
            next = try (try rec.attentionPrefix(s, end)).clone();
            next.keys = try old.keys.finish(s, rec.key_write, n);
            next.values = try old.values.finish(s, rec.value_write, n);
            next.index_keys = try old.index_keys.finish(s, rec.index_write, n);
        }
        if (layer_index == 1) {
            next.ple = try mx.retain(if (p.prefilled) rec.ple else try s.slice(rec.ple, 0, n, n + 9));
            if (m.gpuTokensEnabled()) {
                next.token_history = try mx.retain(try s.slice(rec.token_history, 0, n, n + 2));
            } else {
                next.history = old.history;
                for (p.tokens[0..keep]) |token| next.history = .{ next.history[1], token };
            }
        }
        return next;
    }
    fn commitImpl(m: *Model, p: *Pass, keep: usize, evaluate: bool) !void {
        if (m.position != p.start) return error.InvalidCommit;
        var next = try m.committedCache(p, &m.cache, keep);
        errdefer for (&next) |*c| c.deinit();
        if (evaluate) {
            var arrays: [48 * 6]A = undefined;
            var count: usize = 0;
            for (next) |cache| inline for (.{ "a", "b", "raw", "pooled", "ple", "token_history" }) |field| {
                const value = @field(cache, field);
                if (value.ctx != null) {
                    arrays[count] = value;
                    count += 1;
                }
            };
            try mx.evalMany(arrays[0..count], false);
        }
        for (&m.cache) |*c| c.deinit();
        m.cache = next;
        m.position = p.start + @as(i32, @intCast(keep));
    }
    pub fn draftStep(m: *Model, s: *mx.Scope, hidden: A, token: i32, cache: *Cache) !A {
        return m.draftStepArray(s, hidden, try s.ints(&.{token}), cache, false);
    }
    pub fn draftInput(m: *Model, s: *mx.Scope, hidden: A, token: A) !A {
        const rows = mx.dim(hidden, 0);
        if (rows < 1 or rows > max_shared_rows or mx.c.mlx_array_size(token) != @as(usize, @intCast(rows))) return error.InvalidDraftRows;
        const e = try m.lin(s, "mtp", "fc_embedding", try m.centeredNorm(s, try m.weights.embedArray(s, "model.embed_tokens", token), "mtp.pre_fc_norm_embedding", 2560));
        const hn = try m.centeredNorm(s, hidden, "mtp.pre_fc_norm_hidden", 10240);
        const streams = try s.reshape(hn, &.{ rows * 4, 2560 });
        const step: i32 = if (mx.tensor_units) 128 else 16;
        const projected = if (rows * 4 <= step) try m.lin(s, "mtp", "fc_hidden", streams) else blk: {
            var parts: [max_shared_rows * 4 / 16]A = undefined;
            var count: usize = 0;
            var begin: i32 = 0;
            while (begin < rows * 4) : (begin += step) {
                parts[count] = try m.lin(s, "mtp", "fc_hidden", try s.slice(streams, 0, begin, @min(begin + step, rows * 4)));
                count += 1;
            }
            break :blk try s.cat(parts[0..count], 0);
        };
        const hs = try s.reshape(projected, &.{ rows, 4, 2560 });
        return s.reshape(try s.binary(mx.c.mlx_add, hs, try s.reshape(e, &.{ rows, 1, 2560 })), &.{ rows, 10240 });
    }
    pub fn draftStepStreams(m: *Model, s: *mx.Scope, hidden: A, tokens: A, caches: []const *Cache) !A {
        return @import("flash_shared.zig").draftStep(m, s, hidden, tokens, caches);
    }
    pub fn absorbDraftStreams(m: *Model, s: *mx.Scope, hidden: A, tokens: A, lengths: []const usize, caches: []const *Cache) !void {
        return @import("flash_shared.zig").absorbDraft(m, s, hidden, tokens, lengths, caches);
    }
    pub fn draftStepArray(m: *Model, s: *mx.Scope, hidden: A, token: A, cache: *Cache, queued: bool) !A {
        if (mx.dim(hidden, 0) < 1 or mx.dim(hidden, 0) > 16) return error.InvalidDraftRows;
        const h = try m.draftInput(s, hidden, token);
        const rows = mx.dim(h, 0);
        var rec = Cache{};
        const out = try m.layer(s, "mtp.layers.0", try m.hcNorm(s, h, null, mx.empty), cache, &rec, false);
        if (!queued) try mx.eval(out[0]);
        var next = try rec.clone();
        errdefer next.deinit();
        next.keys = try cache.keys.finish(s, rec.key_write, rows);
        next.values = try cache.values.finish(s, rec.value_write, rows);
        next.index_keys = try cache.index_keys.finish(s, rec.index_write, rows);
        cache.deinit();
        cache.* = next;
        // MTP residual streams are fed to the next chained step; its own final mixer
        // supplies the prediction, while target hidden states use the trunk mixer.
        return out[0];
    }
    pub fn draftPrefix(s: *mx.Scope, cache: Cache, rows: usize, keep: usize) !Cache {
        if (rows == 0 or keep > rows or rows > @as(usize, @intCast(cache.offset))) return error.InvalidCommit;
        const end = cache.offset - @as(i32, @intCast(rows - keep));
        if (end == 0) return .{};
        return (try cache.attentionPrefix(s, end)).clone();
    }
    pub fn checkAttention(io: std.Io, dir: []const u8) !void {
        try mx.init();
        defer mx.shutdown();
        var m = Model{ .weights = cp.Store.init(32), .kernels = mx.Kernels.init(), .centered = false };
        defer m.deinit();
        try m.prepareConstants();
        var path: [4096]u8 = undefined;
        try m.weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
        defer mx.allocator.free(bytes);
        const Case = struct { key: []const u8, past: i32, pooled: i32 };
        const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
        defer cases.deinit();
        const equal = @import("sampling_checks.zig").equal;
        for (cases.value) |case| {
            var s = mx.Scope{};
            defer s.deinit();
            const x = try m.weights.field(case.key, "x");
            var cache = Cache{ .offset = case.past, .a = try m.weights.field(case.key, "a"), .b = try m.weights.field(case.key, "b"), .raw = try m.weights.field(case.key, "raw"), .pooled = if (case.pooled > 0) try m.weights.field(case.key, "pooled") else mx.empty };
            var batch = Cache{};
            const base = "model.layers.3.self_attn";
            const out = try m.attention(&s, base, x, &cache, &batch);
            try equal(&s, out, try m.weights.field(case.key, "expected"));
            try equal(&s, batch.pooled, try m.weights.field(case.key, "pooled_expected"));
            for (0..8) |row| {
                const j: i32 = @intCast(row);
                const input = try s.slice(x, 0, j, j + 1);
                var next = Cache{};
                const serial = try m.attention(&s, base, input, &cache, &next);
                try equal(&s, serial, try s.slice(out, 0, j, j + 1));
                if (row > 0) {
                    var rollback = try batch.attentionPrefix(&s, case.past + j);
                    var continued = Cache{};
                    try equal(&s, serial, try m.attention(&s, base, input, &rollback, &continued));
                    try equal(&s, next.a, continued.a);
                    try equal(&s, next.b, continued.b);
                    try equal(&s, next.raw, continued.raw);
                    if (next.pooled.ctx != null) try equal(&s, next.pooled, continued.pooled);
                }
                cache = next;
            }
        }
        try @import("flash_shared.zig").checkAttention(&m, cases.value);
        std.debug.print("PASS: {d} Flash sparse-boundary fixtures, all 8 rows, pooling and rollback continuations match exactly.\n", .{cases.value.len});
    }
    pub fn checkPleNorm(io: std.Io, dir: []const u8) !void {
        try mx.init();
        defer mx.shutdown();
        var m = Model{ .weights = cp.Store.init(32), .kernels = mx.Kernels.init() };
        defer m.deinit();
        try m.prepareConstants();
        var path: [4096]u8 = undefined;
        try m.weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/arrays.safetensors", .{dir}), "", "");
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cases = try std.json.parseFromSlice([]const []const u8, mx.allocator, bytes, .{});
        defer cases.deinit();
        if (cases.value.len == 0) return error.EmptyFixtures;
        for (cases.value) |key| {
            var s = mx.Scope{};
            defer s.deinit();
            const actual = try m.pleNorm(&s, try m.weights.field(key, "x"), key);
            try @import("sampling_checks.zig").equal(&s, actual, try m.weights.field(key, "expected"));
        }
        std.debug.print("PASS: {d} PLE normalization fixtures match the original square/mean arithmetic exactly\n", .{cases.value.len});
    }
};
