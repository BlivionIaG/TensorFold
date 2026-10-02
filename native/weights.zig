const std = @import("std");
const mx = @import("mlx.zig");
const lanes = @import("lanes.zig");
pub fn readFile(io: std.Io, path: []const u8) ![]u8 {
    const f = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer f.close(io);
    var buffer: [8192]u8 = undefined;
    var reader = f.reader(io, &buffer);
    return reader.interface.allocRemaining(mx.allocator, .limited(128 * 1024 * 1024));
}
pub const Weights = struct {
    arrays: std.StringHashMap(mx.Array),
    linears: std.StringHashMap(lanes.Linear),
    stacks: std.StringHashMap(lanes.Linear),
    rotations: std.ArrayList(mx.Array) = .empty,
    embedding_format: ?@import("quantization.zig").Spec = .{},
    embedding_signs: mx.Array = mx.empty,
    bonsai_form: ?@import("bonsai.zig").Form = null,
    embedding_kernels: mx.Kernels,
    pub fn init() Weights {
        return .{ .arrays = std.StringHashMap(mx.Array).init(mx.allocator), .linears = std.StringHashMap(lanes.Linear).init(mx.allocator), .stacks = std.StringHashMap(lanes.Linear).init(mx.allocator), .embedding_kernels = mx.Kernels.init() };
    }
    pub fn deinit(w: *Weights) void {
        mx.free(w.embedding_signs);
        w.embedding_kernels.deinit();
        var ls = w.linears.iterator();
        while (ls.next()) |e| {
            e.value_ptr.deinit();
            mx.allocator.free(e.key_ptr.*);
        }
        w.linears.deinit();
        var stacks = w.stacks.iterator();
        while (stacks.next()) |entry| {
            entry.value_ptr.deinit();
            mx.allocator.free(entry.key_ptr.*);
        }
        w.stacks.deinit();
        for (w.rotations.items) |signs| mx.free(signs);
        w.rotations.deinit(mx.allocator);
        var it = w.arrays.iterator();
        while (it.next()) |e| {
            mx.free(e.value_ptr.*);
            mx.allocator.free(e.key_ptr.*);
        }
        w.arrays.deinit();
    }
    pub fn get(w: *const Weights, name: []const u8) !mx.Array {
        return w.arrays.get(name) orelse {
            @import("server_live.zig").print("Missing weight: {s}\n", .{name});
            return error.MissingWeight;
        };
    }
    pub fn linear(w: *const Weights, name: []const u8) !lanes.Linear {
        return w.linears.get(name) orelse error.MissingLinear;
    }
    // Both helpers take ownership, including when map insertion fails.
    pub fn putArray(w: *Weights, name: []const u8, value: mx.Array) !void {
        errdefer mx.free(value);
        if (w.arrays.contains(name)) return error.DuplicateWeight;
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try w.arrays.put(key, value);
    }
    pub fn putLinear(w: *Weights, name: []const u8, value: lanes.Linear) !void {
        var owned = value;
        errdefer owned.deinit();
        if (w.linears.contains(name)) return error.DuplicateWeight;
        owned.rotation_id = 0;
        if (owned.signs.ctx != null and mx.dtype(owned.signs) == mx.f32t) {
            try mx.eval(owned.signs);
            const count = mx.c.mlx_array_size(owned.signs);
            const signs = mx.c.mlx_array_data_float32(owned.signs)[0..count];
            for (w.rotations.items, 0..) |existing, i| {
                if (mx.c.mlx_array_size(existing) == count and std.mem.eql(f32, mx.c.mlx_array_data_float32(existing)[0..count], signs)) {
                    owned.rotation_id = i + 1;
                    break;
                }
            }
            if (owned.rotation_id == 0) {
                const held = try mx.retain(owned.signs);
                errdefer mx.free(held);
                try w.rotations.append(mx.allocator, held);
                owned.rotation_id = w.rotations.items.len;
            }
        }
        const key = try mx.allocator.dupe(u8, name);
        errdefer mx.allocator.free(key);
        try w.linears.put(key, owned);
    }
    pub fn fused(w: *Weights, names: []const []const u8) !?lanes.Linear {
        if (names.len < 2 or names.len > 4) return error.InvalidProjectionGroup;
        var key_buffer: [1024]u8 = undefined;
        var key_length: usize = 0;
        for (names) |name| {
            if (name.len + 1 > key_buffer.len - key_length) return error.InvalidProjectionGroup;
            @memcpy(key_buffer[key_length..][0..name.len], name);
            key_buffer[key_length + name.len] = 0;
            key_length += name.len + 1;
        }
        const stack_name = key_buffer[0..key_length];
        if (w.stacks.get(stack_name)) |stack| return stack;
        var members: [4]lanes.Linear = undefined;
        var weight_parts: [4]mx.Array = undefined;
        var pairs: [4]mx.Array = undefined;
        var width: i32 = 0;
        var tiled_prefix = names.len;
        for (names, 0..) |name, i| {
            members[i] = try w.linear(name);
            if (!lanes.Linear.stackCompatible(members[0], members[i])) return null;
            if (!members[i].tiled and tiled_prefix == names.len) tiled_prefix = i;
            if (members[i].tiled and tiled_prefix < i) return null;
            weight_parts[i] = members[i].weight;
            pairs[i] = members[i].sb;
            width = try std.math.add(i32, width, members[i].n);
        }
        var s = mx.Scope{};
        defer s.deinit();
        const mixed = tiled_prefix > 0 and tiled_prefix < names.len;
        if (mixed) {
            if (members[0].tile_width != 32) return null;
            var tail_width: i32 = 0;
            var tail_bytes: usize = 0;
            for (members[tiled_prefix..names.len]) |member| {
                tail_width += member.n;
                tail_bytes += mx.c.mlx_array_nbytes(member.weight);
            }
            if (@mod(tail_width, 32) != 0 or tail_bytes > 8 * 1024 * 1024) return null;
            const tail = try s.cat(weight_parts[tiled_prefix..names.len], 0);
            const group = members[0].format.?.group_size;
            const words = @divExact(group * members[0].format.?.bits, 32);
            weight_parts[tiled_prefix] = try s.contiguous(try s.reshape(try s.transpose(try s.reshape(tail, &.{ @divExact(tail_width, 32), 32, @divExact(members[0].k, group), words }), &.{ 0, 2, 1, 3 }), mx.shape(tail)));
        }
        const weight = try s.cat(weight_parts[0..if (mixed) tiled_prefix + 1 else names.len], 0);
        const sb = try s.cat(pairs[0..names.len], 1);
        try mx.evalMany(&.{ weight, sb }, false);
        var stack = lanes.Linear{
            .weight = try mx.retain(weight),
            .sb = mx.empty,
            .n = width,
            .k = members[0].k,
            .tiled = tiled_prefix > 0,
            .tile_width = if (tiled_prefix > 0) members[0].tile_width else 32,
            .format = members[0].format,
            .generic = true,
            .split_k = members[0].splitK(),
        };
        errdefer stack.deinit();
        stack.sb = try mx.retain(sb);
        var views: [4]mx.Array = @splat(mx.empty);
        errdefer for (views) |value| mx.free(value);
        var scale_views: [4]mx.Array = @splat(mx.empty);
        errdefer for (scale_views) |value| mx.free(value);
        var offset: i32 = 0;
        const shared_members = if (mixed) tiled_prefix else names.len;
        for (members[0..names.len], 0..) |member, i| {
            if (i < shared_members) views[i] = try mx.retain(try s.slice(weight, 0, offset, offset + member.n));
            scale_views[i] = try mx.retain(try s.slice(sb, 1, offset, offset + member.n));
            offset += member.n;
        }
        const key = try mx.allocator.dupe(u8, stack_name);
        errdefer mx.allocator.free(key);
        try w.stacks.put(key, stack);
        for (names, 0..) |name, i| {
            const member = w.linears.getPtr(name).?;
            if (i < shared_members) {
                mx.free(member.weight);
                member.weight = views[i];
            }
            mx.free(member.sb);
            member.sb = scale_views[i];
        }
        return stack;
    }
    pub fn releaseLinearSources(w: *Weights) !void {
        var it = w.linears.keyIterator();
        while (it.next()) |name| try w.releaseLinearSource(name.*);
    }
    fn releaseLinearSource(w: *Weights, name: []const u8) !void {
        var buffer: [256]u8 = undefined;
        for ([_][]const u8{ ".weight", ".scales", ".biases" }) |suffix| {
            const key = try std.fmt.bufPrint(&buffer, "{s}{s}", .{ name, suffix });
            w.releaseArray(key);
        }
    }
    pub fn releaseArray(w: *Weights, name: []const u8) void {
        if (w.arrays.fetchRemove(name)) |entry| {
            mx.free(entry.value);
            mx.allocator.free(entry.key);
        }
    }
    pub fn loadDraft(w: *Weights, io: std.Io, dir: []const u8) !void {
        var pathbuf: [4096]u8 = undefined;
        const bytes = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(bytes);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, bytes, .{});
        defer cfg.deinit();
        try @import("config.zig").draft(cfg.value);
        const path = try std.fmt.bufPrintSentinel(&pathbuf, "{s}/model.safetensors", .{dir}, 0);
        try @import("safetensors.zig").validateFile(io, path);
        var map = mx.c.mlx_map_string_to_array_new();
        defer _ = mx.c.mlx_map_string_to_array_free(map);
        var meta = mx.c.mlx_map_string_to_string_new();
        defer _ = mx.c.mlx_map_string_to_string_free(meta);
        const cpu = mx.c.mlx_default_cpu_stream_new();
        defer _ = mx.c.mlx_stream_free(cpu);
        try mx.check(mx.c.mlx_load_safetensors(&map, &meta, path, cpu));
        const iter = mx.c.mlx_map_string_to_array_iterator_new(map);
        defer _ = mx.c.mlx_map_string_to_array_iterator_free(iter);
        while (true) {
            var key: [*c]const u8 = null;
            var value = mx.c.mlx_array_new();
            const rc = mx.c.mlx_map_string_to_array_iterator_next(&key, &value, iter);
            if (rc != 0 or key == null) {
                mx.free(value);
                break;
            }
            try w.putArray(std.mem.span(key), value);
        }
        try @import("schema.zig").validate(.dflash, &w.arrays, false);
        var it = w.arrays.iterator();
        while (it.next()) |e| {
            const name = e.key_ptr.*;
            const value = e.value_ptr.*;
            if (!std.mem.endsWith(u8, name, ".weight") or mx.shape(value).len != 2 or std.mem.indexOf(u8, name, "codebook") != null) continue;
            var s = mx.Scope{};
            defer s.deinit();
            var quant = mx.c.mlx_vector_array_new();
            defer _ = mx.c.mlx_vector_array_free(quant);
            try mx.check(mx.c.mlx_quantize(&quant, value, mx.opt(64), mx.opt(4), "affine", mx.empty, mx.stream));
            var arrays: [3]mx.Array = undefined;
            for (0..3) |j| {
                var a = mx.c.mlx_array_new();
                const rc = mx.c.mlx_vector_array_get(&a, quant, j);
                arrays[j] = try s.result(rc, a);
            }
            const l = try lanes.Linear.initWide(&s, arrays[0], arrays[1], arrays[2], true);
            try w.putLinear(name[0 .. name.len - 7], l);
        }
        try w.releaseLinearSources();
    }
    pub fn load(w: *Weights, io: std.Io, dir: []const u8) !void {
        var pathbuf: [4096]u8 = undefined;
        const config = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/config.json", .{dir}));
        defer mx.allocator.free(config);
        const cfg = try std.json.parseFromSlice(std.json.Value, mx.allocator, config, .{});
        defer cfg.deinit();
        try @import("config.zig").target(cfg.value);
        if (std.mem.eql(u8, cfg.value.object.get("model_type").?.string, "prism_hadamard_qwen35")) return @import("bonsai.zig").load(w, io, dir, cfg.value);
        if (mx.tensor_units and !try @import("quantization.zig").nativeLanes(cfg.value)) {
            mx.tensor_units = false;
            std.debug.print("Checkpoint quantization selects the upstream SIMD backend.\n", .{});
        }
        const index = try readFile(io, try std.fmt.bufPrint(&pathbuf, "{s}/model.safetensors.index.json", .{dir}));
        defer mx.allocator.free(index);
        const parsed = try std.json.parseFromSlice(std.json.Value, mx.allocator, index, .{});
        defer parsed.deinit();
        var shards = std.StringHashMap(void).init(mx.allocator);
        defer shards.deinit();
        if (parsed.value != .object) return error.InvalidWeightIndex;
        const weight_map = parsed.value.object.get("weight_map") orelse return error.InvalidWeightIndex;
        if (weight_map != .object) return error.InvalidWeightIndex;
        var it = weight_map.object.iterator();
        while (it.next()) |e| if (std.mem.startsWith(u8, e.key_ptr.*, "language_model.")) {
            if (e.value_ptr.* != .string) return error.InvalidWeightIndex;
            try @import("safetensors.zig").shardName(e.value_ptr.string);
            try shards.put(e.value_ptr.string, {});
        };
        var files = shards.keyIterator();
        if (shards.count() == 0) return error.MissingWeights;
        while (files.next()) |name| {
            const path = try std.fmt.bufPrintSentinel(&pathbuf, "{s}/{s}", .{ dir, name.* }, 0);
            try @import("safetensors.zig").validateFile(io, path);
            std.debug.print("Loading {s}\n", .{name.*});
            var map = mx.c.mlx_map_string_to_array_new();
            defer _ = mx.c.mlx_map_string_to_array_free(map);
            var meta = mx.c.mlx_map_string_to_string_new();
            defer _ = mx.c.mlx_map_string_to_string_free(meta);
            const cpu = mx.c.mlx_default_cpu_stream_new();
            defer _ = mx.c.mlx_stream_free(cpu);
            try mx.check(mx.c.mlx_load_safetensors(&map, &meta, path, cpu));
            const iter = mx.c.mlx_map_string_to_array_iterator_new(map);
            defer _ = mx.c.mlx_map_string_to_array_iterator_free(iter);
            while (true) {
                var key: [*c]const u8 = null;
                var value = mx.c.mlx_array_new();
                const rc = mx.c.mlx_map_string_to_array_iterator_next(&key, &value, iter);
                if (rc != 0 or key == null) {
                    mx.free(value);
                    break;
                }
                const n = std.mem.span(key);
                if (!std.mem.startsWith(u8, n, "language_model.") or std.mem.indexOf(u8, n, ".mtp.") != null) {
                    mx.free(value);
                    continue;
                }
                try w.putArray(n[15..], value);
            }
        }
        try @import("schema.zig").validateConfig(.qwen, &w.arrays, false, cfg.value);
        w.embedding_format = try @import("quantization.zig").resolve(cfg.value, "model.embed_tokens");
        if (w.embedding_format != null and mx.dtype(try w.get("model.embed_tokens.scales")) != mx.dtype(try w.get("model.embed_tokens.biases"))) return error.InvalidTensorDType;
        // MLX-format checkpoints already carry shifted RMS weights and [C,4,1] convs.
        // Refuse raw HF tensors rather than silently applying the wrong normalization.
        if (mx.dim(try w.get("model.layers.0.linear_attn.conv1d.weight"), -1) != 1) return error.UnsanitizedCheckpoint;
        var projections: std.ArrayList([]const u8) = .empty;
        defer projections.deinit(mx.allocator);
        var entries = w.arrays.iterator();
        while (entries.next()) |e| {
            if (!std.mem.endsWith(u8, e.key_ptr.*, ".weight") or mx.shape(e.value_ptr.*).len != 2 or std.mem.indexOf(u8, e.key_ptr.*, "embed_tokens") != null) continue;
            try projections.append(mx.allocator, e.key_ptr.*[0 .. e.key_ptr.len - 7]);
        }
        for (projections.items) |name| {
            var s = mx.Scope{};
            defer s.deinit();
            const weight = try w.get(try std.fmt.bufPrint(&pathbuf, "{s}.weight", .{name}));
            const format = try @import("quantization.zig").resolve(cfg.value, name);
            const scales = if (format != null) try w.get(try std.fmt.bufPrint(&pathbuf, "{s}.scales", .{name})) else mx.empty;
            const biases = if (format != null) try w.get(try std.fmt.bufPrint(&pathbuf, "{s}.biases", .{name})) else mx.empty;
            const linear_ = try lanes.Linear.initFormatWide(&s, weight, scales, biases, format, !std.mem.endsWith(u8, name, "in_proj_z"));
            try w.putLinear(name, linear_);
            // Use the retained key: releasing the source also frees its name.
            try w.releaseLinearSource(w.linears.getKey(name).?);
        }
    }
    pub fn embed(w: *Weights, s: *mx.Scope, tokens: []const i32) !mx.Array {
        return w.embedArray(s, try s.ints(tokens));
    }
    pub fn embedArray(w: *Weights, s: *mx.Scope, ids: mx.Array) !mx.Array {
        if (w.embedding_signs.ctx != null) {
            const count: i32 = @intCast(mx.c.mlx_array_size(ids));
            return (try w.embedding_kernels.run(s, @import("kernel_sources.zig").prism_embed, &.{ try s.cast(try s.reshape(ids, &.{count}), mx.c.MLX_UINT32), try w.get("model.embed_tokens.weight"), try w.get("model.embed_tokens.scales"), try w.get("model.embed_tokens.biases"), w.embedding_signs }, &.{ mx.ti("K", 5120), mx.ti("G", 128) }, .{ 512 * 5, count, 1 }, .{ 512, 1, 1 }, &.{.{ .shape = &.{ 1, count, 5120 } }}))[0];
        }
        const weight = try s.take(try w.get("model.embed_tokens.weight"), ids, 0);
        const out = if (w.embedding_format) |format| blk: {
            var result = mx.c.mlx_array_new();
            const rc = mx.c.mlx_dequantize(&result, weight, try s.take(try w.get("model.embed_tokens.scales"), ids, 0), try s.take(try w.get("model.embed_tokens.biases"), ids, 0), mx.opt(format.group_size), mx.opt(format.bits), "affine", mx.empty, .{ .value = mx.bf16, .has_value = true }, mx.stream);
            break :blk try s.result(rc, result);
        } else weight;
        return s.reshape(out, &.{ 1, @intCast(mx.c.mlx_array_size(ids)), 5120 });
    }
    /// Exercise the owning insertion helpers under the allocation diagnostic.
    pub fn checkOwnedInsertions() !void {
        var w = Weights.init();
        defer w.deinit();
        var s = mx.Scope{};
        defer s.deinit();
        const value = try s.zeros(&.{ 32, 8 }, mx.c.MLX_UINT32);
        const scales = try s.zeros(&.{ 32, 1 }, mx.bf16);
        try w.putArray("projection.weight", try mx.retain(value));
        try w.putArray("projection.scales", try mx.retain(scales));
        try w.putArray("projection.biases", try mx.retain(scales));
        try w.putLinear("projection", try lanes.Linear.init(&s, value, scales, scales));
        try std.testing.expectError(error.DuplicateWeight, w.putArray("projection.weight", try mx.retain(value)));
        try std.testing.expectError(error.DuplicateWeight, w.putLinear("projection", try lanes.Linear.init(&s, value, scales, scales)));
        try w.releaseLinearSources();
        try std.testing.expectEqual(@as(usize, 0), w.arrays.count());
        if (mx.tensor_units) {
            const tail = try s.zeros(&.{ 48, 8 }, mx.c.MLX_UINT32);
            const tail_scales = try s.zeros(&.{ 48, 1 }, mx.bf16);
            try w.putLinear("tail0", try lanes.Linear.init(&s, tail, tail_scales, tail_scales));
            try w.putLinear("tail1", try lanes.Linear.init(&s, tail, tail_scales, tail_scales));
            const tail_handle = (try w.linear("tail0")).weight.ctx;
            const names = [_][]const u8{ "projection", "tail0", "tail1" };
            const stack = (try w.fused(&names)).?;
            try std.testing.expect(stack.tiled and stack.n == 128);
            try std.testing.expectEqual(stack.weight.ctx, (try w.fused(&names)).?.weight.ctx);
            try std.testing.expectEqual(tail_handle, (try w.linear("tail0")).weight.ctx);
            try std.testing.expect(try w.fused(&.{ "projection", "tail0" }) == null);
            try std.testing.expect(try w.fused(&.{ "tail0", "projection" }) == null);
            const wide_weight = try s.zeros(&.{ 64, 8 }, mx.c.MLX_UINT32);
            const wide_scales = try s.zeros(&.{ 64, 1 }, mx.bf16);
            for ([_][]const u8{ "wide0", "wide1" }) |name| try w.putLinear(name, try lanes.Linear.initWide(&s, wide_weight, wide_scales, wide_scales, true));
            const wide_stack = (try w.fused(&.{ "wide0", "wide1" })).?;
            try std.testing.expectEqual(@as(i32, 64), wide_stack.tile_width);
            try std.testing.expectEqual(@as(i32, 128), wide_stack.n);
            try std.testing.expectEqual(wide_stack.weight.ctx, (try w.fused(&.{ "wide0", "wide1" })).?.weight.ctx);
            try std.testing.expect(try w.fused(&.{ "wide0", "projection" }) == null);
            try std.testing.expect(try w.fused(&.{ "wide0", "tail0", "tail1" }) == null);
        }
    }
};
