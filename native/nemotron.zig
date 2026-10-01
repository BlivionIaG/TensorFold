//! Nemotron-H: Mamba2, NoPE attention, routed/shared ReLU² experts, and MTP.
const std = @import("std");
const mx = @import("mlx.zig");
const cp = @import("checkpoint.zig");
const src = @import("kernel_sources.zig");
const A = mx.Array;
const ti = mx.ti;
const kv = @import("kv_buffer.zig");
const sampling = @import("sampling.zig");
pub const HeadPrediction = struct {
    cache: Cache = .{},
    hidden: A = mx.empty,
    first: A = mx.empty,
    position: i32 = -1,
    token: i32 = -1,
    settings: sampling.Sampling = .{},

    pub fn deinit(p: *HeadPrediction) void {
        p.cache.deinit();
        mx.free(p.hidden);
        mx.free(p.first);
        p.* = .{};
    }
    pub fn matches(p: *const HeadPrediction, position: i32, token: i32, settings: sampling.Sampling) bool {
        return p.first.ctx != null and p.hidden.ctx != null and p.position == position and p.token == token and std.meta.eql(p.settings, settings);
    }
};
pub const RecurrentRows = struct {
    conv: A = mx.empty,
    ssm: A = mx.empty,
    owner: usize = 0,
    epoch: u64 = 0,
    row: i32 = 0,

    fn clone(rows: RecurrentRows) !RecurrentRows {
        var out = RecurrentRows{ .owner = rows.owner, .epoch = rows.epoch, .row = rows.row };
        errdefer out.deinit();
        if (rows.conv.ctx != null) out.conv = try mx.retain(rows.conv);
        if (rows.ssm.ctx != null) out.ssm = try mx.retain(rows.ssm);
        return out;
    }
    fn deinit(rows: *RecurrentRows) void {
        mx.free(rows.conv);
        mx.free(rows.ssm);
        rows.* = .{};
    }
};
pub const Cache = struct {
    a: A = mx.empty,
    b: A = mx.empty,
    keys: kv.Buffer = .{},
    values: kv.Buffer = .{},
    key_write: kv.Write = .{},
    value_write: kv.Write = .{},
    recurrent: RecurrentRows = .{},

    pub fn clone(cache: Cache) !Cache {
        var out = Cache{};
        errdefer out.deinit();
        if (cache.a.ctx != null) out.a = try mx.retain(cache.a);
        if (cache.b.ctx != null) out.b = try mx.retain(cache.b);
        out.keys = try cache.keys.clone();
        out.values = try cache.values.clone();
        out.recurrent = try cache.recurrent.clone();
        return out;
    }
    pub fn deinit(cache: *Cache) void {
        mx.free(cache.a);
        mx.free(cache.b);
        cache.keys.deinit();
        cache.values.deinit();
        cache.recurrent.deinit();
        cache.* = .{};
    }
    pub fn nbytes(cache: Cache) u64 {
        const conv = if (cache.recurrent.conv.ctx != null) cache.recurrent.conv else if (cache.keys.current.ctx != null) cache.keys.current else cache.a;
        const ssm = if (cache.recurrent.ssm.ctx != null) cache.recurrent.ssm else if (cache.values.current.ctx != null) cache.values.current else cache.b;
        var total = bytes(conv) +| bytes(ssm);
        inline for (.{ cache.keys, cache.values }) |buffer| total +|= bytes(buffer.spare) +| bytes(buffer.recent);
        return total;
    }
    fn bytes(array: A) u64 {
        return if (array.ctx == null) 0 else mx.c.mlx_array_nbytes(array);
    }
};
pub const Pass = struct {
    scope: mx.Scope = .{},
    logits: A = mx.empty,
    hidden: A = mx.empty,
    start: i32 = 0,
    count: usize = 0,
    prefilled: bool = false,
    records: [52]Cache = @splat(.{}),
    staged: [52]Cache = @splat(.{}),
    staged_ready: bool = false,
    pub fn deinit(p: *Pass) void {
        if (p.staged_ready) for (&p.staged) |*cache| cache.deinit();
        p.scope.deinit();
    }
};
pub const HeadRows = struct {
    context: A = mx.empty,
    attended: A = mx.empty,
};
fn capturedLinear(m: *Model, weights: *cp.Store, kernels: *mx.Kernels, scope: *mx.Scope, name: []const u8) !void {
    if (m.weights.dense.get(name)) |original| {
        var copy = original;
        inline for (.{ "weight", "sb", "scales", "biases", "signs" }) |field| @field(copy, field) = mx.empty;
        errdefer copy.deinit();
        inline for (.{ "weight", "sb", "scales", "biases", "signs" }) |field| if (@field(original, field).ctx != null) {
            @field(copy, field) = try mx.retain(@field(original, field));
        };
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try weights.dense.put(key, copy);
    } else {
        const width = mx.dim(try weights.field(name, "weight"), -1) * 8;
        _ = try weights.linear(kernels, scope, name, try scope.zeros(&.{ 1, width }, mx.bf16), true);
    }
}
const Block = struct {
    weights: cp.Store,
    kernels: mx.Kernels,
    base: []const u8,
    next: A,
    kind: u8,
    shared: bool,

    fn init(m: *Model, layer: usize, shared: bool) !*Block {
        const p = try mx.allocator.create(Block);
        errdefer mx.allocator.destroy(p);
        const base = try std.fmt.allocPrint(mx.allocator, "backbone.layers.{d}.mixer", .{layer});
        errdefer mx.allocator.free(base);
        var weights = cp.Store.init(m.weights.group);
        errdefer weights.deinit();
        inline for (.{ "decode.eps", "decode.limits", "decode.scaling" }) |name| try weights.put(name, try m.weights.get(name));
        var buffer: [256]u8 = undefined;
        const next_name = if (layer + 1 == m.kinds.len) "backbone.norm_f.weight" else try std.fmt.bufPrint(&buffer, "backbone.layers.{d}.norm.weight", .{layer + 1});
        try weights.put(next_name, try m.weights.get(next_name));
        const next = try weights.get(next_name);
        var entries = m.weights.arrays.iterator();
        while (entries.next()) |entry| {
            const name = entry.key_ptr.*;
            if (name.len > base.len and name[base.len] == '.' and std.mem.startsWith(u8, name, base)) try weights.put(name, entry.value_ptr.*);
        }
        var scope = mx.Scope{};
        defer scope.deinit();
        var kernels = mx.Kernels.init();
        errdefer kernels.deinit();
        const projections: []const []const u8 = if (m.kinds[layer] == 'M') &.{ "in_proj", "out_proj" } else &.{ "shared_experts.up_proj", "shared_experts.down_proj" };
        for (projections) |suffix| {
            const name = try std.fmt.bufPrint(&buffer, "{s}.{s}", .{ base, suffix });
            try capturedLinear(m, &weights, &kernels, &scope, name);
        }
        p.* = .{ .weights = weights, .kernels = kernels, .base = base, .next = next, .kind = m.kinds[layer], .shared = shared };
        return p;
    }
    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        const p: *Block = @ptrCast(@alignCast(raw.?));
        p.kernels.deinit();
        p.weights.deinit();
        mx.allocator.free(p.base);
        mx.allocator.destroy(p);
    }
    fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *Block = @ptrCast(@alignCast(raw.?));
        return p.graph(out, ins) catch -1;
    }
    fn graph(p: *Block, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
        var scope = mx.Scope{};
        defer scope.deinit();
        var args: [9]A = undefined;
        const count: usize = if (p.shared) 9 else if (p.kind == 'M') 5 else 3;
        for (args[0..count], 0..) |*arg, index| {
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_vector_array_get(&value, ins, index);
            arg.* = try scope.result(rc, value);
        }
        var model = Model{ .weights = p.weights, .kernels = p.kernels };
        defer {
            p.weights = model.weights;
            p.kernels = model.kernels;
        }
        const previous_sums = args[count - 1];
        const sums: ?A = if (mx.shape(previous_sums).len == 2) previous_sums else null;
        if (p.kind == 'M') {
            var record = Cache{};
            const delta = if (p.shared) try model.mambaSharedSums(&scope, p.base, args[0], .{ .conv = args[2], .ssm = args[3], .segments = args[4], .starts = args[5], .slots = args[6], .dimensions = args[7] }, &record, sums) else try model.mambaSums(&scope, p.base, args[0], .{ .a = args[2], .b = args[3] }, &record, sums);
            const normalized = try model.addNormSums(&scope, args[1], delta, p.next);
            const values = [_]A{ normalized[0], normalized[1], record.a, record.b, if (normalized[4].ctx != null) normalized[4] else previous_sums };
            return mx.c.mlx_vector_array_set_data(out, &values, values.len);
        }
        const normalized = try model.moeImpl(&scope, p.base, args[1], args[0], p.next, sums, true);
        const values = [_]A{ normalized[0], normalized[1], if (normalized[4].ctx != null) normalized[4] else previous_sums };
        return mx.c.mlx_vector_array_set_data(out, &values, values.len);
    }
};
const HeadPlan = struct {
    weights: cp.Store,
    kernels: mx.Kernels,
    front: bool,

    fn init(m: *Model, front: bool) !*HeadPlan {
        const p = try mx.allocator.create(HeadPlan);
        errdefer mx.allocator.destroy(p);
        var weights = cp.Store.init(m.weights.group);
        errdefer weights.deinit();
        inline for (.{ "decode.eps", "decode.scaling" }) |name| try weights.put(name, try m.weights.get(name));
        var entries = m.weights.arrays.iterator();
        while (entries.next()) |entry| {
            const name = entry.key_ptr.*;
            if (std.mem.startsWith(u8, name, "mtp.") or (front and std.mem.startsWith(u8, name, "backbone.embeddings."))) try weights.put(name, entry.value_ptr.*);
        }
        var scope = mx.Scope{};
        defer scope.deinit();
        var kernels = mx.Kernels.init();
        errdefer kernels.deinit();
        const projections: []const []const u8 = if (front) &.{ "mtp.layers.0.eh_proj", "mtp.layers.0.mixer.qkv_proj" } else &.{ "mtp.layers.0.mixer.o_proj", "mtp.layers.1.mixer.shared_experts.up_proj", "mtp.layers.1.mixer.shared_experts.down_proj" };
        for (projections) |name| try capturedLinear(m, &weights, &kernels, &scope, name);
        p.* = .{ .weights = weights, .kernels = kernels, .front = front };
        return p;
    }
    fn destroy(raw: ?*anyopaque) callconv(.c) void {
        const p: *HeadPlan = @ptrCast(@alignCast(raw.?));
        p.kernels.deinit();
        p.weights.deinit();
        mx.allocator.destroy(p);
    }
    fn callback(out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array, raw: ?*anyopaque) callconv(.c) c_int {
        const p: *HeadPlan = @ptrCast(@alignCast(raw.?));
        return p.graph(out, ins) catch -1;
    }
    fn graph(p: *HeadPlan, out: [*c]mx.c.mlx_vector_array, ins: mx.c.mlx_vector_array) !c_int {
        var scope = mx.Scope{};
        defer scope.deinit();
        var args: [2]A = undefined;
        for (&args, 0..) |*arg, i| {
            var value = mx.c.mlx_array_new();
            arg.* = try scope.result(mx.c.mlx_vector_array_get(&value, ins, i), value);
        }
        var m = Model{ .weights = p.weights, .kernels = p.kernels };
        defer {
            p.weights = m.weights;
            p.kernels = m.kernels;
        }
        if (p.front) {
            const result = try m.headFrontGraph(&scope, args[0], args[1]);
            return mx.c.mlx_vector_array_set_data(out, &result, result.len);
        }
        const result = [_]A{try m.headBackGraph(&scope, args[0], args[1])};
        return mx.c.mlx_vector_array_set_data(out, &result, result.len);
    }
};
pub const Model = struct {
    pub const SerialPass = Pass;
    pub const DraftCache = Cache;
    pub const HeadPrediction = @import("nemotron.zig").HeadPrediction;
    pub const max_shared_rows = 128;
    pub const adaptive_mtp_depth = true;
    round_owner: @import("decode_round.zig").Owner = .{},
    weights: cp.Store,
    kernels: mx.Kernels,
    blocks: [52]mx.c.mlx_closure = @splat(.{ .ctx = null }),
    shared_blocks: [52]mx.c.mlx_closure = @splat(.{ .ctx = null }),
    head_plans: [2]mx.c.mlx_closure = @splat(.{ .ctx = null }),
    cache: [52]Cache = @splat(.{}),
    kinds: [52]u8 = undefined,
    position: i32 = 0,
    mtp: bool = false,
    head_trace: ?*std.StringHashMap(A) = null,
    prefill_ops: @import("prefill_ops.zig").Ops = .{},
    prefill_route: @import("nemotron_prefill.zig").Route = .{},
    pub const vocab = 131072;
    pub fn eos(id: i32) bool {
        return id == 2 or id == 11;
    }
    pub fn init(io: std.Io, dir: []const u8, drafts: bool) !Model {
        var m = Model{ .weights = cp.Store.init(64), .kernels = mx.Kernels.init() };
        errdefer m.deinit();
        var buf: [4096]u8 = undefined;
        const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&buf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        try @import("config.zig").nemotron(cfg.value);
        for (cfg.value.object.get("layers_block_type").?.array.items, 0..) |v, i| {
            m.kinds[i] = if (std.mem.eql(u8, v.string, "mamba")) 'M' else if (std.mem.eql(u8, v.string, "moe")) 'E' else '*';
        }
        try m.weights.load(io, dir, "");
        if (drafts) {
            m.weights.loadFile(io, try std.fmt.bufPrint(&buf, "{s}/mtp-4bit.safetensors", .{dir}), "mtp.", "") catch |err| return if (err == error.FileNotFound) error.MissingDraftHead else err;
            m.mtp = true;
        }
        try @import("schema.zig").validate(.nemotron, &m.weights.arrays, drafts);
        {
            var scope = mx.Scope{};
            defer scope.deinit();
            try m.weights.put("decode.eps", try scope.scalar(1e-5));
            try m.weights.put("decode.limits", try scope.data(&[_]f32{ 0, std.math.inf(f32) }, &.{2}, mx.f32t));
            try m.weights.put("decode.scaling", try scope.scalar(2.5));
        }
        // Small constants are prepared once; expert tables remain in their packed format.
        for (m.kinds, 0..) |kind, i| if (kind == 'M') {
            var s = mx.Scope{};
            defer s.deinit();
            const key = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.conv1d.weight", .{i});
            const v = try m.weights.get(key);
            const cw = try s.cast(try s.transpose(try s.reshape(v, &.{ 6144, 4 }), &.{ 1, 0 }), mx.f32t);
            try mx.eval(cw);
            try m.weights.put(key, cw);
            try m.prepareDecodeFloats(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer", .{i}), &.{ "conv1d.bias", "A_log", "D", "dt_bias" });
        } else if (kind == '*') {
            var scope = mx.Scope{};
            defer scope.deinit();
            var arrays: [3]A = undefined;
            inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, field| {
                var parts: [3]A = undefined;
                inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |projection, j| {
                    parts[j] = try m.weights.get(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.{s}.{s}", .{ i, projection, suffix }));
                }
                arrays[field] = try scope.cat(&parts, 0);
                try m.weights.put(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer.qkv_proj.{s}", .{ i, suffix }), arrays[field]);
            }
            try mx.evalMany(&arrays, false);
        } else if (kind == 'E') {
            try m.prepareDecodeFloats(try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer", .{i}), &.{"gate.e_score_correction_bias"});
        };
        if (drafts) {
            try m.prepareDecodeFloats("mtp.layers.1.mixer", &.{"gate.e_score_correction_bias"});
            var scope = mx.Scope{};
            defer scope.deinit();
            var arrays: [3]A = undefined;
            inline for (.{ "weight", "scales", "biases" }, 0..) |suffix, field| {
                var parts: [3]A = undefined;
                inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |projection, j| parts[j] = try m.weights.get(try std.fmt.bufPrint(&buf, "mtp.layers.0.mixer.{s}.{s}", .{ projection, suffix }));
                arrays[field] = try scope.cat(&parts, 0);
                try m.weights.put(try std.fmt.bufPrint(&buf, "mtp.layers.0.mixer.qkv_proj.{s}", .{suffix}), arrays[field]);
            }
            try mx.evalMany(&arrays, false);
        }
        // MLX specializes captured lazy loads per shape; compiled blocks share realized weights.
        const arrays = try mx.allocator.alloc(A, m.weights.arrays.count());
        defer mx.allocator.free(arrays);
        var entries = m.weights.arrays.valueIterator();
        for (arrays) |*array| array.* = entries.next().?.*;
        try mx.evalMany(arrays, false);
        return m;
    }
    fn prepareDecodeFloats(m: *Model, base: []const u8, suffixes: []const []const u8) !void {
        var scope = mx.Scope{};
        defer scope.deinit();
        var buffer: [256]u8 = undefined;
        for (suffixes) |suffix| {
            const value = try scope.cast(try m.weights.field(base, suffix), mx.f32t);
            try m.weights.put(try std.fmt.bufPrint(&buffer, "{s}.{s}_f32", .{ base, suffix }), value);
        }
    }
    pub fn reset(m: *Model) void {
        for (&m.cache) |*c| c.deinit();
        m.position = 0;
    }
    pub fn deinit(m: *Model) void {
        m.reset();
        for (m.blocks) |fun| if (fun.ctx != null) {
            _ = mx.c.mlx_closure_free(fun);
        };
        for (m.shared_blocks) |fun| if (fun.ctx != null) {
            _ = mx.c.mlx_closure_free(fun);
        };
        for (m.head_plans) |fun| if (fun.ctx != null) {
            _ = mx.c.mlx_closure_free(fun);
        };
        m.weights.deinit();
        m.kernels.deinit();
        m.prefill_ops.deinit();
        m.prefill_route.deinit();
    }
    pub fn norm(m: *Model, s: *mx.Scope, x: A, name: []const u8) !A {
        return cp.norm(s, x, try m.weights.field(name, "weight"), 1e-5);
    }
    fn traceHead(m: *Model, name: []const u8, value: A) !void {
        if (m.head_trace) |trace| try trace.put(name, value);
    }
    fn lin(m: *Model, s: *mx.Scope, name: []const u8, x: A) !A {
        return m.linSums(s, name, x, null);
    }
    fn linSums(m: *Model, s: *mx.Scope, name: []const u8, x: A, sums: ?A) !A {
        const dims = mx.dim(x, -1);
        const rows: i32 = @intCast(mx.c.mlx_array_size(x) / @as(usize, @intCast(dims)));
        if (mx.tensor_units and sums != null) {
            if (!m.weights.dense.contains(name)) _ = try m.weights.linear(&m.kernels, s, name, x, true);
            if (m.weights.dense.get(name)) |linear| return s.reshape(try linear.apply(&m.kernels, s, .{ .x = x, .sums = sums }), &.{ rows, linear.n });
        }
        if (!mx.tensor_units and rows > 16 and rows <= max_shared_rows) {
            const input = try s.reshape(x, &.{ rows, dims });
            var outputs: [max_shared_rows / 16]A = undefined;
            var start: i32 = 0;
            var count: usize = 0;
            while (start < rows) : (start += 16) {
                outputs[count] = try m.weights.linear(&m.kernels, s, name, try s.slice(input, 0, start, @min(start + 16, rows)), true);
                count += 1;
            }
            return s.cat(outputs[0..count], 0);
        }
        return m.weights.linear(&m.kernels, s, name, x, true);
    }
    fn f(m: *Model, base: []const u8, suffix: []const u8) !A {
        return m.weights.field(base, suffix);
    }
    pub fn project(m: *Model, s: *mx.Scope, base: []const u8, suffix: []const u8, x: A) !A {
        return m.projectSums(s, base, suffix, x, null);
    }
    pub fn projectSums(m: *Model, s: *mx.Scope, base: []const u8, suffix: []const u8, x: A, sums: ?A) !A {
        var buf: [256]u8 = undefined;
        return m.linSums(s, try std.fmt.bufPrint(&buf, "{s}.{s}", .{ base, suffix }), x, sums);
    }
    pub fn qkv(m: *Model, s: *mx.Scope, base: []const u8, x: A, sums: ?A) ![3]A {
        const rows = mx.dim(x, 0);
        const projected = try m.projectSums(s, base, "qkv_proj", x, sums);
        return .{
            try s.transpose(try s.reshape(try s.slice(projected, 1, 0, 4096), &.{ 1, rows, 32, 128 }), &.{ 0, 2, 1, 3 }),
            try s.transpose(try s.reshape(try s.slice(projected, 1, 4096, 4352), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 }),
            try s.transpose(try s.reshape(try s.slice(projected, 1, 4352, 4608), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 }),
        };
    }
    pub fn checkQkv(m: *Model) !void {
        const first = std.mem.indexOfScalar(u8, &m.kinds, '*').?;
        for (m.kinds, 0..) |kind, layer| if (kind == '*') {
            const widths: []const usize = if (layer == first) &.{ 1, 3, 16, 128 } else &.{3};
            for (widths) |count| {
                var scope = mx.Scope{};
                defer scope.deinit();
                var ids: [max_shared_rows]i32 = undefined;
                for (ids[0..count], 0..) |*token, i| token.* = @intCast(1900 + i * 17);
                const hidden = try m.weights.embed(&scope, "backbone.embeddings", ids[0..count]);
                var buffer: [256]u8 = undefined;
                const x = try m.norm(&scope, hidden, try std.fmt.bufPrint(&buffer, "backbone.layers.{d}.norm", .{layer}));
                const base = try std.fmt.bufPrint(&buffer, "backbone.layers.{d}.mixer", .{layer});
                const rows: i32 = @intCast(count);
                var sums: ?A = null;
                if (mx.tensor_units) {
                    const padded = @divTrunc(rows + 15, 16) * 16;
                    sums = (try m.kernels.run(&scope, src.lane_qmm_xsum, &.{ x, try scope.ints(&.{ rows, padded }) }, &.{ ti("K", 2688), ti("GS", 64) }, .{ 42, padded, 1 }, .{ 42, 1, 1 }, &.{.{ .shape = &.{ 42, padded }, .dtype = mx.f32t }}))[0];
                }
                const actual = try m.qkv(&scope, base, x, sums);
                const generated = try m.qkv(&scope, base, x, null);
                inline for (.{ "q_proj", "k_proj", "v_proj" }, 0..) |projection, i| {
                    const expected = try scope.transpose(try scope.reshape(try m.project(&scope, base, projection, x), &.{ 1, rows, if (i == 0) @as(i32, 32) else 2, 128 }), &.{ 0, 2, 1, 3 });
                    try @import("variant_checks.zig").equalBits(&scope, expected, actual[i]);
                    try @import("variant_checks.zig").equalBits(&scope, expected, generated[i]);
                }
            }
        };
        std.debug.print("Nemotron stacked QKV: exact separate real-weight projections in every layer, widths1/3/16/128 with supplied or computed sums\n", .{});
    }
    pub fn forward(m: *Model, tokens: []const i32) !Pass {
        var p = try m.forwardQueued(tokens);
        errdefer p.deinit();
        try mx.eval(p.logits);
        try observeBuffers(&p);
        return p;
    }
    pub fn forwardStreams(m: *Model, streams: []const @import("nemotron_shared.zig").Stream) !@import("nemotron_shared.zig").Pass {
        return @import("nemotron_shared.zig").forward(m, streams);
    }
    pub fn prefill(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len <= 16) return m.forward(tokens);
        return @import("nemotron_prefill.zig").forward(m, tokens);
    }
    pub fn observeBuffers(p: *Pass) !void {
        if (!kv.track_reuse) return;
        for (p.records) |rec| {
            try kv.observe(rec.key_write);
            try kv.observe(rec.value_write);
        }
    }
    pub fn forwardQueued(m: *Model, tokens: []const i32) !Pass {
        if (tokens.len == 0 or tokens.len > 16) return error.InvalidLaneWidth;
        var s = mx.Scope{};
        defer s.deinit();
        return m.forwardArray(try s.ints(tokens));
    }
    /// Internal GPU-token entry point; IDs come from the validated prompt or sampler.
    pub fn forwardArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null) return error.InvalidToken;
        if (mx.c.mlx_array_ndim(tokens) != 1 or mx.dim(tokens, 0) < 1 or mx.dim(tokens, 0) > 16) return error.InvalidLaneWidth;
        const dtype = mx.c.mlx_array_dtype(tokens);
        if (dtype != mx.c.MLX_INT32 and dtype != mx.c.MLX_UINT32) return error.InvalidToken;
        return m.forwardInput(tokens, &m.cache, m.position);
    }
    pub fn forwardAfter(m: *Model, previous: *Pass, sampled: A) !Pass {
        if (previous.count != 1 or !previous.staged_ready or previous.start != m.position) return error.InvalidPreview;
        if (sampled.ctx == null or mx.dtype(sampled) != mx.c.MLX_UINT32 or !std.mem.eql(i32, mx.shape(sampled), &.{1})) return error.InvalidSamplingShape;
        const position = std.math.add(i32, previous.start, 1) catch return error.InvalidPreview;
        if (position < 1 or position > std.math.maxInt(i32) - 2048) return error.InvalidPreview;
        return m.forwardInput(sampled, &previous.staged, position);
    }
    fn forwardInput(m: *Model, tokens: A, cache: []Cache, position: i32) !Pass {
        var p = Pass{ .start = position, .count = @intCast(mx.dim(tokens, 0)) };
        errdefer p.deinit();
        const s = &p.scope;
        var h = try m.weights.embedArray(s, "backbone.embeddings", tokens);
        var x = try m.norm(s, h, "backbone.layers.0.norm");
        var sums: ?A = null;
        var buf: [256]u8 = undefined;
        var nb: [256]u8 = undefined;
        for (m.kinds, 0..) |kind, i| {
            const base = try std.fmt.bufPrint(&buf, "backbone.layers.{d}.mixer", .{i});
            const next = if (i + 1 == 52) "backbone.norm_f" else try std.fmt.bufPrint(&nb, "backbone.layers.{d}.norm", .{i + 1});
            const nw = try m.weights.field(next, "weight");
            if (kind == 'M') {
                const both = try m.blockSums(s, i, x, h, cache[i], sums);
                p.records[i] = .{ .a = both[2], .b = both[3] };
                h = both[0];
                x = both[1];
                sums = if (both[4].ctx != null) both[4] else null;
            } else if (kind == '*') {
                const delta = try m.attentionSums(s, base, x, &cache[i], &p.records[i], sums);
                const both = try m.addNormSums(s, h, delta, nw);
                h = both[0];
                x = both[1];
                sums = if (both[4].ctx != null) both[4] else null;
            } else {
                const both = try m.blockSums(s, i, x, h, .{}, sums);
                h = both[0];
                x = both[1];
                sums = if (both[4].ctx != null) both[4] else null;
            }
            if ((i + 1) % 8 == 0) try mx.evalMany(&.{ h, x }, true);
        }
        p.hidden = x;
        p.logits = try m.linSums(s, "lm_head", x, sums);
        if (p.count == 1) {
            p.staged = try m.prepareCommit(&p, cache, position, 1);
            p.staged_ready = true;
        }
        return p;
    }
    pub fn block(m: *Model, s: *mx.Scope, layer: usize, x: A, h: A, cache: Cache) ![5]A {
        return m.blockSums(s, layer, x, h, cache, null);
    }
    pub fn blockSums(m: *Model, s: *mx.Scope, layer: usize, x: A, h: A, cache: Cache, sums: ?A) ![5]A {
        if (layer >= m.kinds.len or (m.kinds[layer] != 'M' and m.kinds[layer] != 'E')) return error.InvalidLayerKind;
        const recurrent = m.kinds[layer] == 'M';
        if (x.ctx == null or h.ctx == null or mx.shape(x).len != 2 or mx.dim(x, 0) < 1 or mx.dim(x, 0) > (if (recurrent) @as(i32, 16) else max_shared_rows) or mx.dim(x, 1) != 2688 or !std.mem.eql(i32, mx.shape(x), mx.shape(h)) or mx.dtype(x) != mx.bf16 or mx.dtype(h) != mx.bf16) return error.InvalidBlockInputs;
        if (sums) |value| if (value.ctx == null or !std.mem.eql(i32, mx.shape(value), &.{ 42, @divTrunc(mx.dim(x, 0) + 15, 16) * 16 }) or mx.dtype(value) != mx.f32t) return error.InvalidBlockInputs;
        if (recurrent and ((cache.a.ctx == null) != (cache.b.ctx == null))) return error.InvalidCacheState;
        if (recurrent and cache.a.ctx != null and (!std.mem.eql(i32, mx.shape(cache.a), &.{ 1, 3, 6144 }) or !std.mem.eql(i32, mx.shape(cache.b), &.{ 1, 64, 64, 128 }) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.f32t)) return error.InvalidCacheState;
        if (m.blocks[layer].ctx == null) {
            const payload = try Block.init(m, layer, false);
            const fun = mx.c.mlx_closure_new_func_payload(Block.callback, payload, Block.destroy);
            if (fun.ctx == null) {
                Block.destroy(payload);
                return error.MlxFailure;
            }
            defer _ = mx.c.mlx_closure_free(fun);
            try mx.check(mx.c.mlx_compile(&m.blocks[layer], fun, false));
        }
        var arguments = [_]A{ x, h, mx.empty, mx.empty, mx.empty };
        if (recurrent) {
            arguments[2] = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 1, 3, 6144 }, mx.bf16);
            arguments[3] = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 1, 64, 64, 128 }, mx.f32t);
        }
        const count: usize = if (recurrent) 5 else 3;
        arguments[count - 1] = sums orelse try s.zeros(&.{1}, mx.f32t);
        var values: [5]A = undefined;
        try m.kernels.call(s, m.blocks[layer], arguments[0..count], values[0..count]);
        var result: [5]A = @splat(mx.empty);
        for (values[0..count], 0..) |value, index| {
            if (index == count - 1) {
                if (mx.shape(value).len == 2) result[4] = value;
            } else result[index] = value;
        }
        return result;
    }
    pub const SharedMamba = struct { conv: A, ssm: A, segments: A, starts: A, slots: A, dimensions: A };
    pub fn blockSharedSums(m: *Model, s: *mx.Scope, layer: usize, x: A, h: A, inputs: SharedMamba, sums: ?A) ![5]A {
        if (layer >= m.kinds.len or m.kinds[layer] != 'M') return error.InvalidLayerKind;
        if (x.ctx == null or h.ctx == null or mx.shape(x).len != 2 or mx.dim(x, 0) < 1 or mx.dim(x, 0) > max_shared_rows or mx.dim(x, 1) != 2688 or !std.mem.eql(i32, mx.shape(x), mx.shape(h)) or mx.dtype(x) != mx.bf16 or mx.dtype(h) != mx.bf16) return error.InvalidBlockInputs;
        if (sums) |value| if (value.ctx == null or !std.mem.eql(i32, mx.shape(value), &.{ 42, @divTrunc(mx.dim(x, 0) + 15, 16) * 16 }) or mx.dtype(value) != mx.f32t) return error.InvalidBlockInputs;
        if (inputs.conv.ctx == null or inputs.ssm.ctx == null or mx.shape(inputs.conv).len != 3 or mx.dim(inputs.conv, 0) < 1 or mx.dim(inputs.conv, 0) > max_shared_rows or !std.mem.eql(i32, mx.shape(inputs.conv), &.{ mx.dim(inputs.conv, 0), 3, 6144 }) or !std.mem.eql(i32, mx.shape(inputs.ssm), &.{ mx.dim(inputs.conv, 0), 64, 64, 128 }) or mx.dtype(inputs.conv) != mx.bf16 or mx.dtype(inputs.ssm) != mx.f32t) return error.InvalidCacheState;
        inline for (.{ "segments", "starts", "slots", "dimensions" }) |field| {
            const value = @field(inputs, field);
            if (value.ctx == null or mx.shape(value).len != 1 or mx.dim(value, 0) < 8 or mx.dtype(value) != mx.i32t) return error.InvalidBlockInputs;
        }
        if (mx.dim(inputs.segments, 0) < mx.dim(x, 0)) return error.InvalidBlockInputs;
        if (m.shared_blocks[layer].ctx == null) {
            const payload = try Block.init(m, layer, true);
            const fun = mx.c.mlx_closure_new_func_payload(Block.callback, payload, Block.destroy);
            if (fun.ctx == null) {
                Block.destroy(payload);
                return error.MlxFailure;
            }
            defer _ = mx.c.mlx_closure_free(fun);
            try mx.check(mx.c.mlx_compile(&m.shared_blocks[layer], fun, false));
        }
        const arguments = [_]A{ x, h, inputs.conv, inputs.ssm, inputs.segments, inputs.starts, inputs.slots, inputs.dimensions, sums orelse try s.zeros(&.{1}, mx.f32t) };
        var result: [5]A = undefined;
        try m.kernels.call(s, m.shared_blocks[layer], &arguments, &result);
        if (mx.shape(result[4]).len != 2) result[4] = mx.empty;
        return result;
    }
    pub fn checkBlocks(m: *Model) !void {
        var invalid_scope = mx.Scope{};
        defer invalid_scope.deinit();
        try std.testing.expectError(error.InvalidLayerKind, m.block(&invalid_scope, m.kinds.len, mx.empty, mx.empty, .{}));
        for ([_]u8{ 'M', 'E' }) |kind| {
            const layer = std.mem.indexOfScalar(u8, &m.kinds, kind).?;
            var buffer: [256]u8 = undefined;
            const base = try std.fmt.bufPrint(&buffer, "backbone.layers.{d}.mixer", .{layer});
            var next_buffer: [256]u8 = undefined;
            const next = try m.weights.get(if (layer + 1 == m.kinds.len) "backbone.norm_f.weight" else try std.fmt.bufPrint(&next_buffer, "backbone.layers.{d}.norm.weight", .{layer + 1}));
            var closure: ?*anyopaque = null;
            var resident: ?u64 = null;
            const widths: []const usize = if (kind == 'M') &.{ 1, 3, 16, 1 } else &.{ 1, 3, 16, 128, 1 };
            for (widths) |rows| {
                {
                    var scope = mx.Scope{};
                    defer scope.deinit();
                    const conv = try scope.binary(mx.c.mlx_add, try scope.zeros(&.{ 1, 3, 6144 }, mx.bf16), try scope.cast(try scope.scalar(0.03125), mx.bf16));
                    const ssm = try scope.binary(mx.c.mlx_add, try scope.zeros(&.{ 1, 64, 64, 128 }, mx.f32t), try scope.scalar(0.00390625));
                    const cache = Cache{ .a = conv, .b = ssm };
                    var ids: [max_shared_rows]i32 = undefined;
                    for (ids[0..rows], 0..) |*token, i| token.* = @intCast(1200 + 17 * i);
                    const hidden = try m.weights.embed(&scope, "backbone.embeddings", ids[0..rows]);
                    var norm_buffer: [256]u8 = undefined;
                    const x = try m.norm(&scope, hidden, try std.fmt.bufPrint(&norm_buffer, "backbone.layers.{d}.norm", .{layer}));
                    var record = Cache{};
                    const expected = if (kind == 'M') try m.addNorm(&scope, hidden, try m.mamba(&scope, base, x, cache, &record), next) else try m.moe(&scope, base, hidden, x, next);
                    const actual = try m.block(&scope, layer, x, hidden, if (kind == 'M') cache else .{});
                    try @import("variant_checks.zig").equalBits(&scope, expected[0], actual[0]);
                    try @import("variant_checks.zig").equalBits(&scope, expected[1], actual[1]);
                    if (kind == 'M') {
                        try @import("variant_checks.zig").equalBits(&scope, record.a, actual[2]);
                        try @import("variant_checks.zig").equalBits(&scope, record.b, actual[3]);
                    }
                    if (mx.tensor_units) {
                        const width: i32 = @intCast(rows);
                        const padded = @divTrunc(width + 15, 16) * 16;
                        const dimensions = try scope.ints(&.{ width, padded });
                        const input_sums = (try m.kernels.run(&scope, src.lane_qmm_xsum, &.{ x, dimensions }, &.{ ti("K", 2688), ti("GS", 64) }, .{ 42, padded, 1 }, .{ 42, 1, 1 }, &.{.{ .shape = &.{ 42, padded }, .dtype = mx.f32t }}))[0];
                        const expected_sums = (try m.kernels.run(&scope, src.lane_qmm_xsum, &.{ expected[1], dimensions }, &.{ ti("K", 2688), ti("GS", 64) }, .{ 42, padded, 1 }, .{ 42, 1, 1 }, &.{.{ .shape = &.{ 42, padded }, .dtype = mx.f32t }}))[0];
                        try @import("variant_checks.zig").equalBits(&scope, expected_sums, actual[4]);
                        const supplied = try m.blockSums(&scope, layer, x, hidden, if (kind == 'M') cache else .{}, input_sums);
                        try @import("variant_checks.zig").equalBits(&scope, expected[0], supplied[0]);
                        try @import("variant_checks.zig").equalBits(&scope, expected[1], supplied[1]);
                        try @import("variant_checks.zig").equalBits(&scope, expected_sums, supplied[4]);
                        if (kind == 'M') {
                            try @import("variant_checks.zig").equalBits(&scope, record.a, supplied[2]);
                            try @import("variant_checks.zig").equalBits(&scope, record.b, supplied[3]);
                        }
                        try std.testing.expectError(error.InvalidBlockInputs, m.blockSums(&scope, layer, x, hidden, cache, try scope.zeros(&.{ 42, 1 }, mx.f32t)));
                    }
                    if (closure) |previous| try std.testing.expectEqual(previous, m.blocks[layer].ctx.?) else closure = m.blocks[layer].ctx;
                    try std.testing.expectError(error.InvalidBlockInputs, m.block(&scope, layer, x, try scope.zeros(&.{ 1, 1 }, mx.bf16), cache));
                    if (kind == 'M') try std.testing.expectError(error.InvalidCacheState, m.block(&scope, layer, x, hidden, .{ .a = conv }));
                }
                try mx.check(mx.c.mlx_synchronize(mx.stream));
                const active = try @import("memory_runtime.zig").activeBytes();
                std.debug.print("Nemotron compiled block {c}, rows={d}: active={d} bytes\n", .{ kind, rows, active });
                if (resident) |baseline| {
                    if (active > baseline + 64 * 1024 * 1024) return error.CompiledBlockRetainsWeightsPerShape;
                } else resident = active;
            }
        }
        try m.checkSharedBlocks();
        std.debug.print("Compiled Nemotron Mamba/MoE: exact residual/norm/state and fused sums, widths1/3/16 and MoE128, cached closure reuse and invalid input rejection\n", .{});
    }
    fn checkSharedBlocks(m: *Model) !void {
        const layer = std.mem.indexOfScalar(u8, &m.kinds, 'M').?;
        var buffer: [256]u8 = undefined;
        const base = try std.fmt.bufPrint(&buffer, "backbone.layers.{d}.mixer", .{layer});
        var next_buffer: [256]u8 = undefined;
        const next = try m.weights.get(if (layer + 1 == m.kinds.len) "backbone.norm_f.weight" else try std.fmt.bufPrint(&next_buffer, "backbone.layers.{d}.norm.weight", .{layer + 1}));
        var closure: ?*anyopaque = null;
        var resident: ?u64 = null;
        for ([_]usize{ 1, 3, 16, 128, 1 }, 0..) |rows, step| {
            {
                var scope = mx.Scope{};
                defer scope.deinit();
                const counts: []const usize = switch (rows) {
                    1 => &.{1},
                    3 => &.{ 1, 2 },
                    16 => &.{ 1, 3, 5, 7 },
                    128 => &@as([8]usize, @splat(16)),
                    else => unreachable,
                };
                var conv_states: [8]A = undefined;
                var ssm_states: [8]A = undefined;
                for (counts, 0..) |_, i| {
                    conv_states[i] = try scope.binary(mx.c.mlx_add, try scope.zeros(&.{ 1, 3, 6144 }, mx.bf16), try scope.cast(try scope.scalar(@as(f32, @floatFromInt(i + step + 1)) / 32), mx.bf16));
                    ssm_states[i] = try scope.binary(mx.c.mlx_add, try scope.zeros(&.{ 1, 64, 64, 128 }, mx.f32t), try scope.scalar(@as(f32, @floatFromInt(i + step + 1)) / 256));
                }
                var ids: [max_shared_rows]i32 = undefined;
                for (ids[0..rows], 0..) |*token, i| token.* = @intCast(2100 + step * 71 + i * 17);
                const h = try m.weights.embed(&scope, "backbone.embeddings", ids[0..rows]);
                var norm_buffer: [256]u8 = undefined;
                const x = try m.norm(&scope, h, try std.fmt.bufPrint(&norm_buffer, "backbone.layers.{d}.norm", .{layer}));
                var segments: [max_shared_rows]i32 = @splat(0);
                var starts: [8]i32 = @splat(0);
                var slots: [8]i32 = @splat(0);
                var expected_h: [8]A = undefined;
                var expected_x: [8]A = undefined;
                var expected_conv: [8]A = undefined;
                var expected_ssm: [8]A = undefined;
                var first: usize = 0;
                for (counts, 0..) |count, i| {
                    const slot = counts.len - i - 1;
                    starts[i] = @intCast(first);
                    slots[i] = @intCast(slot);
                    @memset(segments[first..][0..count], @intCast(i));
                    const lo: i32 = @intCast(first);
                    const hi: i32 = @intCast(first + count);
                    var record = Cache{};
                    const expected = try m.addNorm(&scope, try scope.slice(h, 0, lo, hi), try m.mamba(&scope, base, try scope.slice(x, 0, lo, hi), .{ .a = conv_states[slot], .b = ssm_states[slot] }, &record), next);
                    expected_h[i] = expected[0];
                    expected_x[i] = expected[1];
                    expected_conv[i] = record.a;
                    expected_ssm[i] = record.b;
                    first += count;
                }
                const inputs = SharedMamba{ .conv = try scope.cat(conv_states[0..counts.len], 0), .ssm = try scope.cat(ssm_states[0..counts.len], 0), .segments = try scope.ints(segments[0..@max(rows, 8)]), .starts = try scope.ints(&starts), .slots = try scope.ints(&slots), .dimensions = try scope.ints(&.{ @intCast(rows), 0, 0, 0, 0, 0, 0, 0 }) };
                const expected = [_]A{ try scope.cat(expected_h[0..counts.len], 0), try scope.cat(expected_x[0..counts.len], 0), try scope.cat(expected_conv[0..counts.len], 0), try scope.cat(expected_ssm[0..counts.len], 0) };
                const actual = try m.blockSharedSums(&scope, layer, x, h, inputs, null);
                for (expected, actual[0..4]) |reference, got| try @import("variant_checks.zig").equalBits(&scope, reference, got);
                if (mx.tensor_units) {
                    const width: i32 = @intCast(rows);
                    const padded = @divTrunc(width + 15, 16) * 16;
                    const dimensions = try scope.ints(&.{ width, padded });
                    const input_sums = (try m.kernels.run(&scope, src.lane_qmm_xsum, &.{ x, dimensions }, &.{ ti("K", 2688), ti("GS", 64) }, .{ 42, padded, 1 }, .{ 42, 1, 1 }, &.{.{ .shape = &.{ 42, padded }, .dtype = mx.f32t }}))[0];
                    const expected_sums = (try m.kernels.run(&scope, src.lane_qmm_xsum, &.{ expected[1], dimensions }, &.{ ti("K", 2688), ti("GS", 64) }, .{ 42, padded, 1 }, .{ 42, 1, 1 }, &.{.{ .shape = &.{ 42, padded }, .dtype = mx.f32t }}))[0];
                    const supplied = try m.blockSharedSums(&scope, layer, x, h, inputs, input_sums);
                    for (expected, supplied[0..4]) |reference, got| try @import("variant_checks.zig").equalBits(&scope, reference, got);
                    try @import("variant_checks.zig").equalBits(&scope, expected_sums, actual[4]);
                    try @import("variant_checks.zig").equalBits(&scope, expected_sums, supplied[4]);
                }
                if (closure) |previous| try std.testing.expectEqual(previous, m.shared_blocks[layer].ctx.?) else closure = m.shared_blocks[layer].ctx;
                var invalid = inputs;
                invalid.dimensions = try scope.ints(&.{@intCast(rows)});
                try std.testing.expectError(error.InvalidBlockInputs, m.blockSharedSums(&scope, layer, x, h, invalid, null));
                invalid = inputs;
                invalid.conv = mx.empty;
                try std.testing.expectError(error.InvalidCacheState, m.blockSharedSums(&scope, layer, x, h, invalid, null));
            }
            try mx.check(mx.c.mlx_synchronize(mx.stream));
            const active = try @import("memory_runtime.zig").activeBytes();
            if (resident) |baseline| {
                if (active > baseline + 64 * 1024 * 1024) return error.CompiledBlockRetainsWeightsPerShape;
            } else resident = active;
        }
        std.debug.print("Compiled shared Nemotron Mamba: exact serial residual/norm/conv/SSM and fused sums for ragged/permuted1/3/16/128 rows, changed inputs, cached closure and bounded weight residency\n", .{});
    }
    pub fn forwardSerialArray(m: *Model, tokens: A) !Pass {
        if (tokens.ctx == null or mx.c.mlx_array_size(tokens) != 1) return error.InvalidToken;
        return m.forwardArray(tokens);
    }
    pub fn addNorm(m: *Model, s: *mx.Scope, h: A, delta: A, nw: A) ![5]A {
        const r = mx.dim(h, 0);
        return m.kernels.run(s, src.nemotron_add_norm_plain, &.{ h, delta, nw, try m.weights.get("decode.eps") }, &.{ ti("D", 2688), ti("T", 896) }, .{ 896 * r, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } } });
    }
    pub fn addNormSums(m: *Model, s: *mx.Scope, h: A, delta: A, nw: A) ![5]A {
        if (!mx.tensor_units) return m.addNorm(s, h, delta, nw);
        const r = mx.dim(h, 0);
        const padded = @divTrunc(r + 15, 16) * 16;
        var result = try m.kernels.run(s, src.nemotron_add_norm_plain_xs, &.{ h, delta, nw, try m.weights.get("decode.eps"), try s.ints(&.{ r, padded }) }, &.{ ti("D", 2688), ti("T", 896) }, .{ 896 * padded, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ 42, padded }, .dtype = mx.f32t } });
        result[4] = result[2];
        result[2] = mx.empty;
        return result;
    }
    fn mamba(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: Cache, record: *Cache) !A {
        return m.mambaSums(s, base, x, cache, record, null);
    }
    fn mambaSums(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: Cache, record: *Cache, sums: ?A) !A {
        const r = mx.dim(x, 0);
        const p = try m.projectSums(s, base, "in_proj", x, sums);
        const cs = if (cache.a.ctx != null) cache.a else try s.zeros(&.{ 3, 6144 }, mx.bf16);
        const st = if (cache.b.ctx != null) cache.b else try s.zeros(&.{ 64, 64, 128 }, mx.f32t);
        const out = try m.kernels.run(s, src.nemotron_mamba_step, &.{ p, cs, st, try m.f(base, "conv1d.weight"), try m.f(base, "conv1d.bias_f32"), try m.f(base, "A_log_f32"), try m.f(base, "D_f32"), try m.f(base, "dt_bias_f32"), try m.weights.get("decode.limits"), try s.ints(&.{r}) }, &.{ ti("H", 64), ti("DH", 64), ti("NG", 8), ti("DS", 128), ti("XD", 4096), ti("KC", 4), ti("PROJ", 10304), ti("XOFF", 4096), ti("DTOFF", 10240), ti("MAXR", 16), ti("TGY", 8), ti("SSZ", 524288) }, .{ 32, 64, 64 }, .{ 32, 8, 1 }, &.{ .{ .shape = &.{ r, 4096 } }, .{ .shape = &.{ r, 3, 6144 } }, .{ .shape = &.{ r, 64, 64, 128 }, .dtype = mx.f32t } });
        record.* = .{ .a = out[1], .b = out[2] };
        const normed = (try m.kernels.run(s, src.nemotron_group_norm, &.{ out[0], try m.f(base, "norm.weight"), try m.weights.get("decode.eps") }, &.{ ti("XD", 4096), ti("GS", 512) }, .{ 1024, r, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ r, 4096 } }}))[0];
        return m.project(s, base, "out_proj", normed);
    }
    fn mambaSharedSums(m: *Model, s: *mx.Scope, base: []const u8, x: A, inputs: SharedMamba, record: *Cache, sums: ?A) !A {
        const r = mx.dim(x, 0);
        const projected = try m.projectSums(s, base, "in_proj", x, sums);
        const conv = try m.kernels.run(s, src.nemotron_mamba_conv, &.{ projected, inputs.conv, try m.f(base, "conv1d.weight"), try m.f(base, "conv1d.bias_f32"), inputs.segments, inputs.starts, inputs.slots }, &.{ ti("XD", 4096), ti("NG", 8), ti("DS", 128), ti("KC", 4), ti("PROJ", 10304), ti("XOFF", 4096) }, .{ 6144, r, 1 }, .{ 256, 1, 1 }, &.{ .{ .shape = &.{ r, 6144 } }, .{ .shape = &.{ r, 3, 6144 } } });
        const scan = try m.kernels.run(s, src.nemotron_mamba_scan, &.{ projected, conv[0], inputs.ssm, try m.f(base, "A_log_f32"), try m.f(base, "D_f32"), try m.f(base, "dt_bias_f32"), try m.weights.get("decode.limits"), inputs.dimensions, inputs.segments, inputs.slots }, &.{ ti("H", 64), ti("DH", 64), ti("NG", 8), ti("DS", 128), ti("XD", 4096), ti("PROJ", 10304), ti("DTOFF", 10240), ti("SSZ", 524288) }, .{ 32, 64, 64 }, .{ 32, 8, 1 }, &.{ .{ .shape = &.{ r, 4096 } }, .{ .shape = &.{ r, 64, 64, 128 }, .dtype = mx.f32t } });
        record.* = .{ .a = conv[1], .b = scan[1] };
        const normed = (try m.kernels.run(s, src.nemotron_group_norm, &.{ scan[0], try m.f(base, "norm.weight"), try m.weights.get("decode.eps") }, &.{ ti("XD", 4096), ti("GS", 512) }, .{ 1024, r, 1 }, .{ 128, 1, 1 }, &.{.{ .shape = &.{ r, 4096 } }}))[0];
        return m.project(s, base, "out_proj", normed);
    }
    fn attention(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: *Cache, record: *Cache) !A {
        const r = mx.dim(x, 0);
        const q = try s.transpose(try s.reshape(try m.project(s, base, "q_proj", x), &.{ 1, r, 32, 128 }), &.{ 0, 2, 1, 3 });
        const keys = try s.transpose(try s.reshape(try m.project(s, base, "k_proj", x), &.{ 1, r, 2, 128 }), &.{ 0, 2, 1, 3 });
        const values = try s.transpose(try s.reshape(try m.project(s, base, "v_proj", x), &.{ 1, r, 2, 128 }), &.{ 0, 2, 1, 3 });
        return m.project(s, base, "o_proj", try m.attend(s, q, keys, values, cache, record));
    }
    fn attentionSums(m: *Model, s: *mx.Scope, base: []const u8, x: A, cache: *Cache, record: *Cache, sums: ?A) !A {
        const projected = try m.qkv(s, base, x, sums);
        return m.project(s, base, "o_proj", try m.attend(s, projected[0], projected[1], projected[2], cache, record));
    }
    pub fn attend(m: *Model, s: *mx.Scope, q: A, new_keys: A, new_values: A, cache: *Cache, record: *Cache) !A {
        const r = mx.dim(q, 2);
        var keys = new_keys;
        var values = new_values;
        var kw = kv.Write{};
        var vw = kv.Write{};
        if (kv.enabled) {
            kw = try cache.keys.append(s, cache.a, keys, 2);
            vw = try cache.values.append(s, cache.b, values, 2);
            keys = kw.view;
            values = vw.view;
        } else if (cache.a.ctx != null) {
            keys = try s.cat(&.{ cache.a, keys }, 2);
            values = try s.cat(&.{ cache.b, values }, 2);
        }
        record.* = .{ .a = keys, .b = values, .key_write = kw, .value_write = vw };
        // Each query uses the serial kernel and exactly its own visible key length.
        var rows: [16]A = undefined;
        const start = mx.dim(keys, 2) - r;
        for (0..@intCast(r)) |i| {
            const j: i32 = @intCast(i);
            if (mx.tensor_units and start + j + 1 >= 10000) {
                rows[i] = try @import("lanes.zig").sdpa(&m.kernels, s, try s.slice(q, 2, j, j + 1), try s.slice(keys, 2, 0, start + j + 1), try s.slice(values, 2, 0, start + j + 1), 0.08838834764831845);
                continue;
            }
            var out = mx.c.mlx_array_new();
            const rc = mx.c.mlx_fast_scaled_dot_product_attention(&out, try s.slice(q, 2, j, j + 1), try s.slice(keys, 2, 0, start + j + 1), try s.slice(values, 2, 0, start + j + 1), 0.08838834764831845, "", mx.empty, mx.empty, false, mx.stream);
            rows[i] = try s.result(rc, out);
        }
        return s.reshape(try s.transpose(try s.cat(rows[0..@intCast(r)], 2), &.{ 0, 2, 1, 3 }), &.{ r, 4096 });
    }
    pub fn moe(m: *Model, s: *mx.Scope, base: []const u8, h: A, x: A, nw: A) ![5]A {
        return m.moeImpl(s, base, h, x, nw, null, false);
    }
    fn moeImpl(m: *Model, s: *mx.Scope, base: []const u8, h: A, x: A, nw: A, sums: ?A, group_sums: bool) ![5]A {
        const r = mx.dim(x, 0);
        const logits = (try m.kernels.run(s, src.nemotron_router, &.{ x, try m.f(base, "gate.weight"), try s.ints(&.{r}) }, &.{ ti("D", 2688), ti("NE", 128), ti("SG", 8), ti("MAXR", 16) }, .{ 256, 128, @divTrunc(r + 15, 16) }, .{ 256, 1, 1 }, &.{.{ .shape = &.{ r, 128 } }}))[0];
        try m.traceHead("router", logits);
        const route = try m.kernels.run(s, src.nemotron_route, &.{ logits, try m.f(base, "gate.e_score_correction_bias_f32"), try m.weights.get("decode.scaling") }, &.{ ti("NE", 128), ti("K", 6), ti("OK", 6) }, .{ 32 * r, 1, 1 }, .{ 32, 1, 1 }, &.{ .{ .shape = &.{ r, 6 }, .dtype = mx.c.MLX_UINT32 }, .{ .shape = &.{ r, 6 }, .dtype = mx.f32t } });
        try m.traceHead("expert-ids", route[0]);
        try m.traceHead("expert-weights", route[1]);
        var buf: [256]u8 = undefined;
        const up = try m.weights.triple(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.fc1", .{base}));
        const down = try m.weights.triple(try std.fmt.bufPrint(&buf, "{s}.switch_mlp.fc2", .{base}));
        const ids = try s.reshape(route[0], &.{r * 6});
        const act = (try m.kernels.run(s, src.nemotron_rows_expert_up, &.{ x, ids, up[0], up[1], up[2] }, &.{ ti("K", 2688), ti("N", 1856), ti("GS", 64), ti("RPS", 4), ti("SG", 2), ti("TOPK", 6) }, .{ 64, 232, r * 6 }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ r * 6, 1856 } }}))[0];
        const routed = (try m.kernels.run(s, src.nemotron_rows_expert_down, &.{ act, ids, down[0], down[1], down[2] }, &.{ ti("K", 1856), ti("N", 2688), ti("GS", 64), ti("RPS", 4), ti("SG", 2) }, .{ 64, 336, r * 6 }, .{ 64, 1, 1 }, &.{.{ .shape = &.{ r, 6, 2688 } }}))[0];
        try m.traceHead("routed", routed);
        const shared_up = try s.binary(mx.c.mlx_maximum, try m.projectSums(s, base, "shared_experts.up_proj", x, sums), try s.cast(try s.scalar(0), mx.bf16));
        const shared = try m.project(s, base, "shared_experts.down_proj", try s.binary(mx.c.mlx_multiply, shared_up, shared_up));
        try m.traceHead("shared", shared);
        if (group_sums and mx.tensor_units) {
            const padded = @divTrunc(r + 15, 16) * 16;
            var result = try m.kernels.run(s, src.nemotron_add_norm_moe_xs, &.{ h, routed, route[1], shared, nw, try m.weights.get("decode.eps"), try s.ints(&.{ r, padded }) }, &.{ ti("D", 2688), ti("T", 896), ti("E", 6) }, .{ 896 * padded, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ 42, padded }, .dtype = mx.f32t } });
            result[4] = result[2];
            result[2] = mx.empty;
            return result;
        }
        return m.kernels.run(s, src.nemotron_add_norm_moe, &.{ h, routed, route[1], shared, nw, try m.weights.get("decode.eps") }, &.{ ti("D", 2688), ti("T", 896), ti("E", 6) }, .{ 896 * r, 1, 1 }, .{ 896, 1, 1 }, &.{ .{ .shape = &.{ r, 2688 } }, .{ .shape = &.{ r, 2688 } } });
    }
    pub fn commit(m: *Model, p: *Pass, keep: usize) !void {
        return m.commitImpl(p, keep, true);
    }
    pub fn commitSerialQueued(m: *Model, p: *Pass) !void {
        if (mx.dim(p.hidden, 0) != 1) return error.InvalidCommit;
        return m.commitImpl(p, 1, false);
    }
    fn commitImpl(m: *Model, p: *Pass, keep: usize, evaluate: bool) !void {
        if (m.position != p.start) return error.InvalidCommit;
        var next = if (p.staged_ready and keep == 1) p.staged else try m.prepareCommit(p, &m.cache, m.position, keep);
        if (p.staged_ready and keep == 1) {
            p.staged = @splat(.{});
            p.staged_ready = false;
        }
        errdefer for (&next) |*c| c.deinit();
        if (evaluate) {
            var arrays: [104]A = undefined;
            var count: usize = 0;
            for (m.kinds, next) |kind, cache| if (kind != 'E') {
                arrays[count] = cache.a;
                arrays[count + 1] = cache.b;
                count += 2;
            };
            try mx.evalMany(arrays[0..count], false);
        }
        for (&m.cache) |*c| c.deinit();
        m.cache = next;
        m.position += @intCast(keep);
    }
    pub fn prepareCommit(m: *const Model, p: *Pass, cache: []const Cache, position: i32, keep: usize) ![52]Cache {
        if (keep == 0 or keep > @as(usize, @intCast(mx.dim(p.hidden, 0)))) return error.InvalidCommit;
        if (p.prefilled and keep != mx.dim(p.hidden, 0)) return error.InvalidCommit;
        if (cache.len != m.kinds.len or position < 0) return error.InvalidCommit;
        var next: [52]Cache = @splat(.{});
        errdefer for (&next) |*c| c.deinit();
        const n: i32 = @intCast(keep);
        const end = std.math.add(i32, position, n) catch return error.InvalidCommit;
        for (m.kinds, 0..) |kind, i| {
            const rec = p.records[i];
            if (kind == 'M') {
                next[i].a = try mx.retain(if (p.prefilled) rec.a else try p.scope.slice(rec.a, 0, n - 1, n));
                next[i].b = try mx.retain(if (p.prefilled) rec.b else try p.scope.slice(rec.b, 0, n - 1, n));
                if (rec.recurrent.ssm.ctx != null) {
                    next[i].recurrent = try rec.recurrent.clone();
                    next[i].recurrent.row += n - 1;
                }
            }
            if (kind == '*') {
                next[i].a = try mx.retain(try p.scope.slice(rec.a, 2, 0, end));
                next[i].b = try mx.retain(try p.scope.slice(rec.b, 2, 0, end));
                next[i].keys = try cache[i].keys.finish(&p.scope, rec.key_write, n);
                next[i].values = try cache[i].values.finish(&p.scope, rec.value_write, n);
            }
        }
        return next;
    }
    pub fn draftStep(m: *Model, s: *mx.Scope, hidden: A, token: i32, cache: *Cache) !A {
        return m.draftStepArray(s, hidden, try s.ints(&.{token}), cache, false);
    }
    pub fn draftStepArray(m: *Model, s: *mx.Scope, hidden: A, token: A, cache: *Cache, queued: bool) !A {
        const rows = mx.dim(hidden, 0);
        if (rows < 1 or rows > 16 or mx.c.mlx_array_size(token) != @as(usize, @intCast(rows))) return error.InvalidDraftRows;
        const e = try m.norm(s, try m.weights.embedArray(s, "backbone.embeddings", token), "mtp.layers.0.enorm");
        const h = try m.norm(s, hidden, "mtp.layers.0.hnorm");
        const x = try m.lin(s, "mtp.layers.0.eh_proj", try s.cat(&.{ e, h }, -1));
        var record = Cache{};
        const delta = try m.attention(s, "mtp.layers.0.mixer", try m.norm(s, x, "mtp.layers.0.norm"), cache, &record);
        const normalized = try m.addNorm(s, x, delta, try m.weights.get("mtp.layers.1.norm.weight"));
        const out = try m.moe(s, "mtp.layers.1.mixer", normalized[0], normalized[1], try m.weights.get("mtp.layers.1.final_layernorm.weight"));
        if (!queued) try mx.evalMany(&.{ out[1], record.a, record.b }, false);
        var next = try record.clone();
        errdefer next.deinit();
        next.keys = try cache.keys.finish(s, record.key_write, rows);
        next.values = try cache.values.finish(s, record.value_write, rows);
        cache.deinit();
        cache.* = next;
        return out[1];
    }
    pub fn draftStepStreams(m: *Model, s: *mx.Scope, hidden: A, tokens: A, caches: []const *Cache) !A {
        if (caches.len == 0 or caches.len > 8 or hidden.ctx == null or tokens.ctx == null) return error.InvalidDraftRows;
        const rows: i32 = @intCast(caches.len);
        if (!std.mem.eql(i32, mx.shape(hidden), &.{ rows, 2688 }) or !std.mem.eql(i32, mx.shape(tokens), &.{rows}) or mx.dtype(hidden) != mx.bf16) return error.InvalidDraftRows;
        if (mx.dtype(tokens) != mx.c.MLX_INT32 and mx.dtype(tokens) != mx.c.MLX_UINT32) return error.InvalidToken;
        for (caches, 0..) |cache, index| {
            for (caches[0..index]) |other| if (cache == other) return error.DuplicateStream;
            if ((cache.a.ctx == null) != (cache.b.ctx == null)) return error.InvalidCacheState;
            if (cache.a.ctx != null) {
                const shape = mx.shape(cache.a);
                if (shape.len != 4 or shape[0] != 1 or shape[1] != 2 or shape[2] < 1 or shape[3] != 128 or !std.mem.eql(i32, shape, mx.shape(cache.b)) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16) return error.InvalidCacheState;
            }
        }
        const front = try m.headFront(s, hidden, tokens);
        var outputs: [8]A = undefined;
        var records: [8]Cache = @splat(.{});
        for (caches, 0..) |cache, index| {
            const row: i32 = @intCast(index);
            outputs[index] = try m.attend(s, try s.slice(front[1], 2, row, row + 1), try s.slice(front[2], 2, row, row + 1), try s.slice(front[3], 2, row, row + 1), cache, &records[index]);
        }
        const out = try m.headBack(s, front[0], try s.cat(outputs[0..caches.len], 0));
        var next: [8]Cache = @splat(.{});
        defer for (&next) |*cache| cache.deinit();
        for (caches, 0..) |cache, index| {
            next[index] = try records[index].clone();
            next[index].keys = try cache.keys.finish(s, records[index].key_write, 1);
            next[index].values = try cache.values.finish(s, records[index].value_write, 1);
        }
        for (caches, next[0..caches.len]) |cache, *replacement| {
            cache.deinit();
            cache.* = replacement.*;
            replacement.* = .{};
        }
        return out;
    }
    pub fn draftStepWindows(m: *Model, s: *mx.Scope, hidden: A, tokens: A, lengths: []const usize, caches: []const *Cache, records: []Cache) !A {
        const rows = try m.draftWindowFront(s, hidden, tokens, lengths, caches, records);
        return m.headBack(s, rows.context, rows.attended);
    }
    pub fn draftWindowFront(m: *Model, s: *mx.Scope, hidden: A, tokens: A, lengths: []const usize, caches: []const *Cache, records: []Cache) !HeadRows {
        if (caches.len == 0 or caches.len > 8 or lengths.len != caches.len or records.len != caches.len) return error.InvalidDraftRows;
        var total: usize = 0;
        for (caches, lengths, 0..) |cache, count, i| {
            if (count == 0 or count > 16 or total > max_shared_rows - count) return error.InvalidDraftRows;
            total += count;
            for (caches[0..i]) |other| if (cache == other) return error.DuplicateStream;
            if ((cache.a.ctx == null) != (cache.b.ctx == null)) return error.InvalidCacheState;
            if (cache.a.ctx != null) {
                const shape = mx.shape(cache.a);
                if (shape.len != 4 or shape[0] != 1 or shape[1] != 2 or shape[2] < 1 or shape[3] != 128 or !std.mem.eql(i32, shape, mx.shape(cache.b)) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16) return error.InvalidCacheState;
            }
        }
        const rows: i32 = @intCast(total);
        if (hidden.ctx == null or tokens.ctx == null or !std.mem.eql(i32, mx.shape(hidden), &.{ rows, 2688 }) or !std.mem.eql(i32, mx.shape(tokens), &.{rows}) or mx.dtype(hidden) != mx.bf16) return error.InvalidDraftRows;
        if (mx.dtype(tokens) != mx.c.MLX_INT32 and mx.dtype(tokens) != mx.c.MLX_UINT32) return error.InvalidToken;
        const front = try m.headFront(s, hidden, tokens);
        var outputs: [8]A = undefined;
        var first: i32 = 0;
        for (caches, lengths, records, 0..) |cache, count, *record, i| {
            const end = first + @as(i32, @intCast(count));
            outputs[i] = try m.attend(s, try s.slice(front[1], 2, first, end), try s.slice(front[2], 2, first, end), try s.slice(front[3], 2, first, end), cache, record);
            first = end;
        }
        return .{ .context = front[0], .attended = try s.cat(outputs[0..caches.len], 0) };
    }
    pub fn draftWindowTail(m: *Model, s: *mx.Scope, rows: HeadRows, selected: []const i32) !A {
        if (rows.context.ctx == null or rows.attended.ctx == null or selected.len == 0 or selected.len > max_shared_rows) return error.InvalidDraftRows;
        const shape = mx.shape(rows.context);
        if (shape.len != 2 or shape[0] < 1 or shape[0] > max_shared_rows or shape[1] != 2688 or !std.mem.eql(i32, mx.shape(rows.attended), &.{ shape[0], 4096 }) or mx.dtype(rows.context) != mx.bf16 or mx.dtype(rows.attended) != mx.bf16) return error.InvalidDraftRows;
        for (selected) |row| if (row < 0 or row >= shape[0]) return error.InvalidDraftRows;
        const indices = try s.ints(selected);
        return m.headBack(s, try s.take(rows.context, indices, 0), try s.take(rows.attended, indices, 0));
    }

    fn headFrontGraph(m: *Model, s: *mx.Scope, hidden: A, tokens: A) ![4]A {
        const e = try m.norm(s, try m.weights.embedArray(s, "backbone.embeddings", tokens), "mtp.layers.0.enorm");
        const h = try m.norm(s, hidden, "mtp.layers.0.hnorm");
        const x = try m.lin(s, "mtp.layers.0.eh_proj", try s.cat(&.{ e, h }, -1));
        try m.traceHead("x", x);
        const normalized = try m.norm(s, x, "mtp.layers.0.norm");
        try m.traceHead("attention-input", normalized);
        const projected = try m.qkv(s, "mtp.layers.0.mixer", normalized, null);
        return .{ x, projected[0], projected[1], projected[2] };
    }
    fn headBackGraph(m: *Model, s: *mx.Scope, x: A, attended: A) !A {
        const delta = try m.project(s, "mtp.layers.0.mixer", "o_proj", attended);
        try m.traceHead("attention-output", delta);
        const normalized = try m.addNorm(s, x, delta, try m.weights.get("mtp.layers.1.norm.weight"));
        try m.traceHead("residual", normalized[0]);
        try m.traceHead("moe-input", normalized[1]);
        const out = try m.moe(s, "mtp.layers.1.mixer", normalized[0], normalized[1], try m.weights.get("mtp.layers.1.final_layernorm.weight"));
        return out[1];
    }
    fn headPlan(m: *Model, front: bool) !mx.c.mlx_closure {
        const index: usize = if (front) 0 else 1;
        if (m.head_plans[index].ctx == null) {
            const payload = try HeadPlan.init(m, front);
            const fun = mx.c.mlx_closure_new_func_payload(HeadPlan.callback, payload, HeadPlan.destroy);
            if (fun.ctx == null) {
                HeadPlan.destroy(payload);
                return error.MlxFailure;
            }
            defer _ = mx.c.mlx_closure_free(fun);
            try mx.check(mx.c.mlx_compile(&m.head_plans[index], fun, false));
        }
        return m.head_plans[index];
    }
    fn headFront(m: *Model, s: *mx.Scope, hidden: A, tokens: A) ![4]A {
        if (m.head_trace != null) return m.headFrontGraph(s, hidden, tokens);
        var result: [4]A = undefined;
        try m.kernels.call(s, try m.headPlan(true), &.{ hidden, tokens }, &result);
        return result;
    }
    fn headBack(m: *Model, s: *mx.Scope, x: A, attended: A) !A {
        if (m.head_trace != null) return m.headBackGraph(s, x, attended);
        var result: [1]A = undefined;
        try m.kernels.call(s, try m.headPlan(false), &.{ x, attended }, &result);
        return result[0];
    }

    pub fn absorbDraftStreams(m: *Model, s: *mx.Scope, hidden: A, tokens: A, lengths: []const usize, caches: []const *Cache) !void {
        if (caches.len > 8 or lengths.len != caches.len) return error.InvalidDraftRows;
        var total: usize = 0;
        for (caches, lengths, 0..) |cache, count, i| {
            for (caches[0..i]) |other| if (cache == other) return error.DuplicateStream;
            if (count > 16 or total > max_shared_rows - count) return error.InvalidDraftRows;
            total += count;
            if ((cache.a.ctx == null) != (cache.b.ctx == null)) return error.InvalidCacheState;
            if (cache.a.ctx != null) {
                const shape = mx.shape(cache.a);
                if (shape.len != 4 or shape[0] != 1 or shape[1] != 2 or shape[2] < 1 or shape[3] != 128 or !std.mem.eql(i32, shape, mx.shape(cache.b)) or mx.dtype(cache.a) != mx.bf16 or mx.dtype(cache.b) != mx.bf16) return error.InvalidCacheState;
            }
        }
        if (total == 0) return;
        const rows: i32 = @intCast(total);
        if (hidden.ctx == null or tokens.ctx == null or !std.mem.eql(i32, mx.shape(hidden), &.{ rows, 2688 }) or !std.mem.eql(i32, mx.shape(tokens), &.{rows}) or mx.dtype(hidden) != mx.bf16) return error.InvalidDraftRows;
        if (mx.dtype(tokens) != mx.c.MLX_INT32 and mx.dtype(tokens) != mx.c.MLX_UINT32) return error.InvalidToken;
        const embedded = try m.norm(s, try m.weights.embedArray(s, "backbone.embeddings", tokens), "mtp.layers.0.enorm");
        const previous = try m.norm(s, hidden, "mtp.layers.0.hnorm");
        const projected = try m.lin(s, "mtp.layers.0.eh_proj", try s.cat(&.{ embedded, previous }, -1));
        const normalized = try m.norm(s, projected, "mtp.layers.0.norm");
        const keys = try s.transpose(try s.reshape(try m.project(s, "mtp.layers.0.mixer", "k_proj", normalized), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 });
        const values = try s.transpose(try s.reshape(try m.project(s, "mtp.layers.0.mixer", "v_proj", normalized), &.{ 1, rows, 2, 128 }), &.{ 0, 2, 1, 3 });
        var replacements: [8]Cache = @splat(.{});
        defer for (&replacements) |*cache| cache.deinit();
        var offset: i32 = 0;
        for (caches, lengths, 0..) |cache, count, i| {
            if (count == 0) continue;
            const n: i32 = @intCast(count);
            var added_keys = try s.slice(keys, 2, offset, offset + n);
            var added_values = try s.slice(values, 2, offset, offset + n);
            if (kv.enabled) {
                const kw = try cache.keys.append(s, cache.a, added_keys, 2);
                const vw = try cache.values.append(s, cache.b, added_values, 2);
                added_keys = kw.view;
                added_values = vw.view;
                replacements[i].keys = try cache.keys.finish(s, kw, n);
                replacements[i].values = try cache.values.finish(s, vw, n);
            } else if (cache.a.ctx != null) {
                added_keys = try s.cat(&.{ cache.a, added_keys }, 2);
                added_values = try s.cat(&.{ cache.b, added_values }, 2);
            }
            replacements[i].a = try mx.retain(added_keys);
            replacements[i].b = try mx.retain(added_values);
            offset += n;
        }
        for (caches, lengths, replacements[0..caches.len]) |cache, count, *replacement| if (count > 0) {
            std.mem.swap(Cache, cache, replacement);
        };
    }
    pub fn draftPrefix(s: *mx.Scope, cache: Cache, rows: usize, keep: usize) !Cache {
        if (rows == 0 or keep > rows or rows > @as(usize, @intCast(mx.dim(cache.a, 2)))) return error.InvalidCommit;
        const end = mx.dim(cache.a, 2) - @as(i32, @intCast(rows - keep));
        if (end == 0) return .{};
        return (Cache{ .a = try s.slice(cache.a, 2, 0, end), .b = try s.slice(cache.b, 2, 0, end), .keys = try cache.keys.prefix(s, end), .values = try cache.values.prefix(s, end) }).clone();
    }
    pub fn head(m: *Model, s: *mx.Scope, h: A) !A {
        return m.lin(s, "lm_head", h);
    }
    pub fn headSums(m: *Model, s: *mx.Scope, h: A, sums: ?A) !A {
        return m.linSums(s, "lm_head", h, sums);
    }
    pub fn draftHead(m: *Model, s: *mx.Scope, h: A) !A {
        return m.lin(s, if (m.weights.has("draft_ids")) "draft_lm_head" else "lm_head", h);
    }
    pub const draft_vocabulary = @import("draft_vocab.zig").data.nemotron;
    pub const draft_prior = &@import("draft_depth.zig").nemotron_prior;
};
