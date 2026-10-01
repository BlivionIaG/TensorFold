//! Replay original kernel launches using embedded sources and compare raw output bits.
const std = @import("std");
const mx = @import("mlx.zig");
const src = @import("kernel_sources.zig");
const Parameter = struct {
    name: [:0]const u8,
    kind: enum { integer, boolean, dtype },
    integer: i32 = 0,
    boolean: bool = false,
    dtype: []const u8 = "",
};
const Case = struct {
    name: []const u8,
    kernel: []const u8,
    source_sha256: []const u8,
    @"test": []const u8,
    templates: []const Parameter,
    grid: [3]i32,
    group: [3]i32,
    input_count: usize,
    output_count: usize,
    mutated_inputs: []const usize = &.{},
};
fn find(name: []const u8) !src.Spec {
    inline for (comptime std.meta.declarations(src)) |decl| {
        const value = @field(src, decl);
        if (@TypeOf(value) == src.Spec) {
            if (std.mem.eql(u8, value.name, name)) return value;
        }
    }
    return error.UnknownKernelFixture;
}
fn dtype(name: []const u8) !mx.c.mlx_dtype {
    const names = .{ "bfloat16", "float16", "float32", "int32", "uint32", "uint16", "uint8", "int64", "uint64", "bool_" };
    const types = .{ mx.bf16, mx.c.MLX_FLOAT16, mx.f32t, mx.i32t, mx.c.MLX_UINT32, mx.c.MLX_UINT16, mx.c.MLX_UINT8, mx.c.MLX_INT64, mx.c.MLX_UINT64, mx.c.MLX_BOOL };
    inline for (names, types) |label, value| if (std.mem.eql(u8, name, label)) return value;
    return error.UnsupportedFixtureDType;
}
fn raw(s: *mx.Scope, value: mx.Array) !mx.Array {
    var out = mx.c.mlx_array_new();
    const rc = mx.c.mlx_view(&out, try s.contiguous(try s.reshape(value, &.{-1})), mx.c.MLX_UINT8, mx.stream);
    return s.result(rc, out);
}
pub fn check(io: std.Io, dir: []const u8) !void {
    try mx.init();
    defer mx.shutdown();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    var gemma_attention = @import("gemma_ops.zig").Attention{};
    defer gemma_attention.deinit();
    var gemma_cache_checked = false;
    var path: [4096]u8 = undefined;
    const bytes = try @import("weights.zig").readFile(io, try std.fmt.bufPrint(&path, "{s}/cases.json", .{dir}));
    defer mx.allocator.free(bytes);
    const cases = try std.json.parseFromSlice([]const Case, mx.allocator, bytes, .{});
    defer cases.deinit();
    if (cases.value.len == 0) return error.EmptyFixtures;
    var covered = std.StringHashMap(usize).init(mx.allocator);
    defer covered.deinit();
    for (cases.value) |case| {
        errdefer std.debug.print("Variant failure: {s}: {s} ({s})\n", .{ case.name, case.kernel, case.@"test" });
        const spec = try find(case.kernel);
        if (case.input_count != spec.inputs.len or case.output_count != spec.outputs.len) return error.InvalidKernelArity;
        var hash = std.crypto.hash.sha2.Sha256.init(.{});
        hash.update(spec.header);
        hash.update(&.{0});
        hash.update(spec.source);
        var digest: [32]u8 = undefined;
        hash.final(&digest);
        if (!std.mem.eql(u8, &std.fmt.bytesToHex(digest, .lower), case.source_sha256)) return error.KernelSourceMismatch;
        var weights = @import("checkpoint.zig").Store.init(64);
        defer weights.deinit();
        try weights.loadFile(io, try std.fmt.bufPrint(&path, "{s}/{s}.safetensors", .{ dir, case.name }), "", "");
        var scope = mx.Scope{};
        defer scope.deinit();
        const inputs = try mx.allocator.alloc(mx.Array, case.input_count);
        defer mx.allocator.free(inputs);
        const outputs = try mx.allocator.alloc(mx.Output, case.output_count);
        defer mx.allocator.free(outputs);
        const expected = try mx.allocator.alloc(mx.Array, case.output_count);
        defer mx.allocator.free(expected);
        const results = try mx.allocator.alloc(mx.Array, case.output_count);
        defer mx.allocator.free(results);
        const params = try mx.allocator.alloc(mx.Template, case.templates.len);
        defer mx.allocator.free(params);
        for (inputs, 0..) |*input, i| input.* = try weights.get(try std.fmt.bufPrint(&path, "input{d}", .{i}));
        for (expected, outputs, 0..) |*value, *output, i| {
            value.* = try weights.get(try std.fmt.bufPrint(&path, "output{d}", .{i}));
            output.* = .{ .shape = mx.shape(value.*), .dtype = mx.dtype(value.*) };
        }
        for (params, case.templates) |*param, value| param.* = switch (value.kind) {
            .integer => mx.ti(value.name, value.integer),
            .boolean => mx.tb(value.name, value.boolean),
            .dtype => mx.td(value.name, try dtype(value.dtype)),
        };
        try kernels.runInto(&scope, spec, inputs, params, case.grid, case.group, outputs, results, 0);
        if (case.mutated_inputs.len > 0) {
            try mx.evalMany(results, false);
            for (case.mutated_inputs) |i| {
                if (i >= inputs.len) return error.InvalidFixture;
                try equalBits(&scope, inputs[i], try weights.get(try std.fmt.bufPrint(&path, "mutation{d}", .{i})));
            }
        }
        const gemma = @import("gemma_ops.zig");
        const large = @import("large_family_ops.zig");
        if (std.mem.startsWith(u8, case.kernel, "flash_")) {
            const flash = @import("flash_ops.zig");
            const generation: u32 = if (std.mem.endsWith(u8, case.kernel, "_h")) 15 else if (std.mem.endsWith(u8, case.kernel, "_x")) 13 else 17;
            const generic = std.mem.startsWith(u8, case.kernel, "flash_qa_");
            if (std.mem.eql(u8, case.kernel, "flash_q4_ple_gate")) {
                const actual = try flash.pleGate(&kernels, &scope, inputs[0], inputs[1], inputs[2..5].*, inputs[5], parameter(case, "S"));
                for (actual, expected) |got, want| try equalBits(&scope, got, want);
            } else if (std.mem.eql(u8, case.kernel, "flash_q4_ple_conv")) {
                try equalBits(&scope, try flash.pleConv(&kernels, &scope, inputs[0], inputs[1], inputs[2], inputs[3], parameter(case, "S"), parameter(case, "DIL")), expected[0]);
            } else if (std.mem.eql(u8, case.kernel, "flash_qa_qmv_rows_mma")) {
                const w = flash.Weight{ .arrays = inputs[2..5].*, .format = .{ .bits = parameter(case, "BITS"), .group_size = parameter(case, "GS") } };
                try equalBits(&scope, try flash.projectTiles(&kernels, &scope, inputs[0], w), expected[0]);
                if (weights.has("dispatch_output")) {
                    const want = try weights.get("dispatch_output");
                    try equalBits(&scope, try flash.project(&kernels, &scope, inputs[0], w, generation, 4), want);
                    const batch = try scope.stack(&.{ inputs[0], inputs[0] }, 0);
                    try equalBits(&scope, try flash.project(&kernels, &scope, batch, w, generation, 4), try scope.stack(&.{ want, want }, 0));
                }
            } else if (std.mem.indexOf(u8, case.kernel, "qmv_rows") != null) {
                const format = @import("quantization.zig").Spec{ .bits = if (generic) parameter(case, "BITS") else 4, .group_size = if (generic) parameter(case, "GS") else 32 };
                const w = flash.Weight{ .arrays = .{ inputs[1], inputs[2], inputs[3] }, .format = format };
                try equalBits(&scope, try flash.project(&kernels, &scope, inputs[0], w, generation, parameter(case, "RPS")), expected[0]);
                const batch = try scope.stack(&.{ inputs[0], inputs[0] }, 0);
                try equalBits(&scope, try flash.project(&kernels, &scope, batch, w, generation, parameter(case, "RPS")), try scope.stack(&.{ expected[0], expected[0] }, 0));
            } else if (std.mem.eql(u8, case.kernel, "flash_q4_hc_up_tiles")) {
                const down = flash.Weight{ .arrays = .{ try weights.get("down_weight"), try weights.get("down_scales"), try weights.get("down_biases") } };
                const up = flash.Weight{ .arrays = inputs[4..7].* };
                const actual = try flash.hyperTiles(&kernels, &scope, inputs[0], inputs[1], down, up, inputs[2], inputs[7], parameter(case, "S"), parameter(case, "LOW"));
                try equalBits(&scope, actual[0], expected[0]);
                if (parameter(case, "ND") > parameter(case, "LOW")) try equalBits(&scope, actual[1], try scope.slice(expected[1], 0, 0, mx.dim(inputs[0], 0)));
                if (weights.has("dispatch_output")) {
                    const gen = try weights.get("generation");
                    try mx.eval(gen);
                    const dispatched = try flash.hyper(&kernels, &scope, inputs[0], inputs[1], down, up, inputs[2], inputs[7], parameter(case, "S"), parameter(case, "LOW"), @intCast(mx.c.mlx_array_data_int32(gen)[0]));
                    try equalBits(&scope, dispatched[0], try weights.get("dispatch_output"));
                    if (weights.has("dispatch_inject")) try equalBits(&scope, dispatched[1], try weights.get("dispatch_inject"));
                }
            } else if (std.mem.indexOf(u8, case.kernel, "hc_up_row") != null or std.mem.indexOf(u8, case.kernel, "hc_up_mma") != null) {
                const df = try weights.get("down_format");
                try mx.eval(df);
                const values = mx.c.mlx_array_data_int32(df)[0..2];
                const down = flash.Weight{ .arrays = .{ try weights.get("down_weight"), try weights.get("down_scales"), try weights.get("down_biases") }, .format = .{ .bits = values[0], .group_size = values[1] } };
                const up = flash.Weight{ .arrays = inputs[4..7].*, .format = .{ .bits = if (generic) parameter(case, "BITS") else 4, .group_size = if (generic) parameter(case, "GS") else 32 } };
                const actual = try @import("flash_lane.zig").hyper(&kernels, &scope, inputs[0], inputs[1], down, up, inputs[2], inputs[7], parameter(case, "S"), parameter(case, "LOW"));
                try equalBits(&scope, actual[0], expected[0]);
                if (parameter(case, "ND") > parameter(case, "LOW")) try equalBits(&scope, actual[1], try scope.slice(expected[1], 0, 0, mx.dim(inputs[0], 0)));
            } else if (std.mem.indexOf(u8, case.kernel, "hc_up2") != null) {
                const df = try weights.get("down_format");
                try mx.eval(df);
                const values = mx.c.mlx_array_data_int32(df)[0..2];
                const down = flash.Weight{ .arrays = .{ try weights.get("down_weight"), try weights.get("down_scales"), try weights.get("down_biases") }, .format = .{ .bits = values[0], .group_size = values[1] } };
                const up = flash.Weight{ .arrays = .{ inputs[3], inputs[4], inputs[5] }, .format = .{ .bits = if (generic) parameter(case, "BITS") else 4, .group_size = if (generic) parameter(case, "GS") else 32 } };
                const actual = try flash.hyper(&kernels, &scope, inputs[0], inputs[1], down, up, inputs[6], inputs[7], parameter(case, "S"), parameter(case, "LOW"), generation);
                try equalBits(&scope, actual[0], expected[0]);
                if (parameter(case, "ND") > parameter(case, "LOW")) try equalBits(&scope, actual[1], try scope.slice(expected[1], 0, 0, mx.dim(inputs[0], 0)));
            } else if (std.mem.indexOf(u8, case.kernel, "expert_") != null) {
                const format = @import("quantization.zig").Spec{ .bits = if (generic) parameter(case, "WB") else 4, .group_size = if (generic) parameter(case, "WG") else 32 };
                const shared_format = @import("quantization.zig").Spec{ .bits = if (generic) parameter(case, "SWB") else 4, .group_size = if (generic) parameter(case, "SWG") else 32 };
                if (std.mem.indexOf(u8, case.kernel, "gateup") != null) {
                    const gate = flash.Weight{ .arrays = inputs[2..5].*, .format = format };
                    const up = flash.Weight{ .arrays = inputs[5..8].*, .format = format };
                    const shared: ?[2]flash.Weight = if (parameter(case, "SHARED") == 0) null else .{ .{ .arrays = inputs[8..11].*, .format = shared_format }, .{ .arrays = inputs[11..14].*, .format = shared_format } };
                    const actual = try flash.gateUp(&kernels, &scope, inputs[0], inputs[1], gate, up, shared, parameter(case, "TOPK"), generation, parameter(case, "RPS"), parameter(case, "SG"));
                    for (actual, expected) |got, want| try equalBits(&scope, got, want);
                } else try equalBits(&scope, try flash.expertDown(&kernels, &scope, inputs[0], inputs[1], .{ .arrays = inputs[2..5].*, .format = format }, .{ .arrays = inputs[5..8].*, .format = shared_format }, generation, parameter(case, "SG")), expected[0]);
            }
        }
        if (std.mem.eql(u8, case.kernel, "glm_qmv_rows64") or std.mem.eql(u8, case.kernel, "glm_qmv_rows_b") or std.mem.eql(u8, case.kernel, "ds4_qmv_rows_f32")) {
            const bits = if (std.mem.eql(u8, case.kernel, "glm_qmv_rows_b")) parameter(case, "BITS") else 4;
            try equalBits(&scope, try large.project(&kernels, &scope, inputs[0], .{ inputs[1], inputs[2], inputs[3] }, bits, std.mem.eql(u8, case.kernel, "ds4_qmv_rows_f32"), parameter(case, "RPS")), expected[0]);
        }
        if (std.mem.eql(u8, case.kernel, "ds4_norm_rope")) try equalBits(&scope, try large.normRope(&kernels, &scope, inputs[0], if (parameter(case, "WEIGHTED") == 1) inputs[1] else null, inputs[2], inputs[3], inputs[4], parameter(case, "NORM") == 1, parameter(case, "INVERSE") == 1), expected[0]);
        if (std.mem.eql(u8, case.kernel, "glm_indexed_attention")) {
            try mx.evalMany(inputs[3..5], false);
            try equalBits(&scope, try large.indexedAttention(&kernels, &scope, inputs[0], inputs[1], inputs[2], mx.c.mlx_array_data_int32(inputs[4])[0], mx.c.mlx_array_data_float32(inputs[3])[0]), expected[0]);
        }
        if (std.mem.eql(u8, case.kernel, "glm_kda_rows")) {
            const actual = try large.kda(&kernels, &scope, .{ .heads = parameter(case, "H"), .dims = parameter(case, "D"), .taps = parameter(case, "TAPS"), .f_bits = parameter(case, "FB"), .g_bits = parameter(case, "GB") }, inputs[0..15].*);
            for (actual, expected) |got, want| try equalBits(&scope, got, want);
        }
        if (std.mem.eql(u8, case.kernel, "gemma_qkv_rows") and parameter(case, "SG") == 32) {
            const geometry = gemma.Geometry{ .heads = parameter(case, "NQ"), .kv_heads = parameter(case, "NK"), .head_dim = parameter(case, "DH"), .values_are_keys = parameter(case, "VK") == 1 };
            const actual = try gemma.qkv(&kernels, &scope, geometry, inputs[0], .{ inputs[1], inputs[2], inputs[3] }, inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], parameter(case, "GS"));
            for (actual, expected) |got, want| try equalBits(&scope, got, want);
        }
        if (std.mem.eql(u8, case.kernel, "gemma_route")) {
            const actual = try gemma.route(&kernels, &scope, inputs[0], inputs[1], parameter(case, "K"));
            const logical = mx.dim(inputs[0], 0) * parameter(case, "K");
            for (actual, expected) |got, want| try equalBits(&scope, try scope.slice(got, 0, 0, logical), try scope.slice(want, 0, 0, logical));
        }
        if (std.mem.eql(u8, case.kernel, "gemma_router")) try equalBits(&scope, try gemma.router(&kernels, &scope, inputs[0], .{ inputs[1], inputs[2], inputs[3] }, parameter(case, "GS")), expected[0]);
        if (std.mem.eql(u8, case.kernel, "gemma_expert_gateup")) try equalBits(&scope, try gemma.gateUp(&kernels, &scope, inputs[0], inputs[1], parameter(case, "TOPK"), .{ inputs[2], inputs[3], inputs[4] }, .{ inputs[5], inputs[6], inputs[7] }, parameter(case, "GS")), expected[0]);
        if (std.mem.eql(u8, case.kernel, "gemma_expert_down")) try equalBits(&scope, try gemma.down(&kernels, &scope, inputs[0], inputs[1], inputs[2], parameter(case, "TOPK"), .{ inputs[3], inputs[4], inputs[5] }, parameter(case, "GS")), expected[0]);
        if (std.mem.eql(u8, case.kernel, "gemma_attention_partial")) {
            try mx.evalMany(inputs[5..8], false);
            const rows: usize = @intCast(mx.dim(inputs[0], 0));
            const positions = mx.c.mlx_array_data_int32(inputs[5])[0..rows];
            const lows = mx.c.mlx_array_data_int32(inputs[6])[0..rows];
            const ring = mx.c.mlx_array_data_int32(inputs[7])[2];
            const window = if (lows[rows - 1] > 0) positions[rows - 1] - lows[rows - 1] + 1 else 0;
            const actual = try gemma.attention(&kernels, &scope, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], positions, window, ring, @bitCast(parameter(case, "SCALE_BITS")));
            const heads = mx.dim(inputs[0], 1);
            const dims = mx.dim(inputs[0], 2);
            const want = (try kernels.run(&scope, src.gemma_attention_merge, &.{ expected[0], expected[1], expected[2], inputs[7] }, &.{ mx.ti("D", dims), mx.ti("H", heads) }, .{ 32, heads, @intCast(rows) }, .{ 32, 1, 1 }, &.{.{ .shape = mx.shape(inputs[0]) }}))[0];
            try equalBits(&scope, actual, want);
            const prepared = try gemma.Rows.init(&scope, positions, window, ring, dims);
            try equalBits(&scope, try gemma_attention.apply(&kernels, &scope, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], prepared, @bitCast(parameter(case, "SCALE_BITS"))), want);
            if (!gemma_cache_checked) {
                try checkGemmaAttentionCache();
                gemma_cache_checked = true;
            }
        }
        if (std.mem.startsWith(u8, case.@"test", "flash-lane-") and std.mem.startsWith(u8, case.kernel, "lane_qmm_") and !std.mem.eql(u8, case.kernel, "lane_qmm_xsum")) {
            const fmt = try weights.get("original_format");
            try mx.eval(fmt);
            const values = mx.c.mlx_array_data_int32(fmt)[0..2];
            var projection = try @import("flash_lane.zig").Projection.init(&scope, .{ .arrays = .{ try weights.get("original_weight"), try weights.get("original_scales"), try weights.get("original_biases") }, .format = .{ .bits = values[0], .group_size = values[1] } });
            defer projection.deinit();
            try equalBits(&scope, try projection.apply(&kernels, &scope, inputs[0]), expected[0]);
        }
        if (std.mem.startsWith(u8, case.kernel, "lane_qmm_lowbit") or std.mem.startsWith(u8, case.kernel, "lane_qmm_bytes") or std.mem.startsWith(u8, case.kernel, "lane_qmm_main") or std.mem.eql(u8, case.kernel, "lane_qmm_coop")) {
            const cooperative = std.mem.eql(u8, case.kernel, "lane_qmm_coop");
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const bits = if (cooperative or std.mem.startsWith(u8, case.kernel, "lane_qmm_main")) 4 else parameter(case, "BITS");
            const group = if (bits == 4 or std.mem.endsWith(u8, case.kernel, "_grouped")) parameter(case, "GS") else 64;
            const groups = @divExact(width, group);
            const words = @divExact(group * bits, 32);
            const tiled = if (bits == 4) cooperative or std.mem.endsWith(u8, case.kernel, "_tiled") else parameter(case, "TILED") == 1;
            const nt: i32 = if (cooperative) 64 else 32;
            const w = if (tiled) try scope.contiguous(try scope.reshape(try scope.transpose(try scope.reshape(inputs[2], &.{ @divExact(n, nt), groups, nt, words }), &.{ 0, 2, 1, 3 }), &.{ n, groups * words })) else inputs[2];
            const sb = try scope.transpose(inputs[3], &.{ 1, 0, 2 });
            const sc = try scope.reshape(try scope.slice(sb, 2, 0, 1), &.{ n, groups });
            const bs = try scope.reshape(try scope.slice(sb, 2, 1, 2), &.{ n, groups });
            var linear = try @import("lanes.zig").Linear.initFormatWide(&scope, w, sc, bs, .{ .bits = bits, .group_size = group }, cooperative);
            defer linear.deinit();
            try std.testing.expectEqual(mx.tensor_units and @mod(n, 32) == 0 and (group == 64 or bits == 4), linear.tiled);
            try std.testing.expectEqual(nt, linear.tile_width);
            if (linear.tiled) {
                var selected = try linear.selectRanges(&scope, &.{.{ 0, @min(n, 3) }});
                defer selected.deinit();
                try equalBits(&scope, selected.weight, try scope.slice(w, 0, 0, @min(n, 3)));
                try equalBits(&scope, selected.scales, try scope.slice(sc, 0, 0, @min(n, 3)));
                try equalBits(&scope, selected.biases, try scope.slice(bs, 0, 0, @min(n, 3)));
            }
            // Only default launch reductions are the production dispatch contract.
            var split: i32 = 1;
            while (split < 8 and @divTrunc(n + 31, 32) * split < 1024 and @divTrunc(@divExact(width, 64), split * 2) >= 8) split *= 2;
            if (split == parameter(case, "SK")) {
                try equalBits(&scope, try linear.tensorRows(&kernels, &scope, .{ .x = inputs[0], .sums = inputs[1] }), expected[0]);
                if (cooperative) {
                    try equalBits(&scope, try linear.apply(&kernels, &scope, .{ .x = inputs[0], .sums = inputs[1] }), expected[0]);
                    var narrow = try @import("lanes.zig").Linear.initFormat(&scope, w, sc, bs, .{ .bits = bits, .group_size = group });
                    defer narrow.deinit();
                    try equalBits(&scope, try narrow.apply(&kernels, &scope, .{ .x = inputs[0], .sums = inputs[1] }), expected[0]);
                    if (mx.dim(inputs[0], 0) == 1) {
                        const ranges = [_][2]i32{ .{ 0, 32 }, .{ n - 32, n } };
                        var selected = try linear.selectRanges(&scope, &ranges);
                        defer selected.deinit();
                        var selected_narrow = try narrow.selectRanges(&scope, &ranges);
                        defer selected_narrow.deinit();
                        try std.testing.expectEqual(@as(i32, 64), selected.tile_width);
                        try equalBits(&scope, try selected.apply(&kernels, &scope, .{ .x = inputs[0] }), try selected_narrow.apply(&kernels, &scope, .{ .x = inputs[0] }));
                        const repeated: [129]mx.Array = @splat(inputs[0]);
                        const prompt = try scope.reshape(try scope.cat(&repeated, 0), &.{ 1, 129, width });
                        try equalBits(&scope, try linear.prefill(&kernels, &scope, prompt), try narrow.prefill(&kernels, &scope, prompt));
                    }
                }
            }
        }
        if (std.mem.startsWith(u8, case.@"test", "bonsai-reuse-") and std.mem.eql(u8, case.kernel, "lane_qmm_lowbit")) {
            try checkBonsaiReuse(&scope, &weights, expected[0]);
        }
        if (std.mem.startsWith(u8, case.@"test", "plain-stack-") and (std.mem.startsWith(u8, case.kernel, "lane_qmm_main") or std.mem.eql(u8, case.kernel, "lane_qmm_coop"))) {
            try checkPlainStack(&scope, &weights, expected[0], std.mem.indexOf(u8, case.@"test", "-wide-") != null);
        }
        if (std.mem.eql(u8, case.kernel, "simd_qmm_mma")) {
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const group = parameter(case, "GS");
            const split: i32 = if (n <= 64) 32 else if (n <= 6144) 16 else 8;
            if (parameter(case, "S") == split and mx.dim(inputs[0], 0) <= 128) {
                const was_tensor = mx.tensor_units;
                mx.tensor_units = false;
                defer mx.tensor_units = was_tensor;
                var linear = try @import("lanes.zig").Linear.initFormat(&scope, try matrix(&scope, inputs[1], n, @divExact(width, 8)), try matrix(&scope, inputs[2], n, @divExact(width, group)), try matrix(&scope, inputs[3], n, @divExact(width, group)), .{ .bits = 4, .group_size = group });
                defer linear.deinit();
                try equalBits(&scope, try linear.apply(&kernels, &scope, .{ .x = inputs[0] }), expected[0]);
            }
        }
        if (std.mem.eql(u8, case.kernel, "affine_rows")) {
            const n = parameter(case, "N");
            const width = parameter(case, "K");
            const format = @import("quantization.zig").Spec{ .bits = parameter(case, "BITS"), .group_size = parameter(case, "GS") };
            const w = try matrix(&scope, inputs[1], n, @divExact(width * format.bits, 32));
            const sc = try matrix(&scope, inputs[2], n, @divExact(width, format.group_size));
            const bs = try matrix(&scope, inputs[3], n, @divExact(width, format.group_size));
            var linear = try @import("lanes.zig").Linear.initFormat(&scope, w, sc, bs, format);
            defer linear.deinit();
            // The pre-existing 4/64 path has its own lane/SIMD arithmetic.
            if (linear.generic) {
                const projected = try linear.apply(&kernels, &scope, .{ .x = inputs[0] });
                try equalBits(&scope, projected, expected[0]);
                var selected = try linear.selectRanges(&scope, &.{.{ 0, @min(n, 3) }});
                defer selected.deinit();
                try equalBits(&scope, try selected.apply(&kernels, &scope, .{ .x = inputs[0] }), try scope.slice(expected[0], 1, 0, @min(n, 3)));
            }
        }
        for (results, expected, 0..) |result, want, i| {
            const a = try raw(&scope, result);
            const b = try raw(&scope, want);
            try mx.evalMany(&.{ a, b }, false);
            const n = mx.c.mlx_array_size(a);
            if (n != mx.c.mlx_array_size(b) or !std.mem.eql(u8, mx.c.mlx_array_data_uint8(a)[0..n], mx.c.mlx_array_data_uint8(b)[0..n])) {
                std.debug.print("Output {d} differs, shape {any}\n", .{ i, mx.shape(result) });
                return error.KernelVariantMismatch;
            }
        }
        const entry = try covered.getOrPut(case.kernel);
        if (!entry.found_existing) entry.value_ptr.* = 0;
        entry.value_ptr.* += 1;
    }
    var it = covered.iterator();
    while (it.next()) |entry| std.debug.print("PASS: {s}: {d} launches, every output bit exact\n", .{ entry.key_ptr.*, entry.value_ptr.* });
    std.debug.print("PASS: {d} native launches across {d} embedded Metal variants\n", .{ cases.value.len, covered.count() });
}

fn parameter(case: Case, name: []const u8) i32 {
    for (case.templates) |p| if (std.mem.eql(u8, p.name, name)) return p.integer;
    unreachable;
}
fn checkPlainStack(s: *mx.Scope, fixture: *@import("checkpoint.zig").Store, expected: mx.Array, wide: bool) !void {
    const lanes = @import("lanes.zig");
    var weights = @import("weights.zig").Weights.init();
    defer weights.deinit();
    var kernels = mx.Kernels.init();
    defer kernels.deinit();
    const x = try fixture.get("source_input");
    const count_array = try fixture.get("member_count");
    try mx.eval(count_array);
    const count: usize = @intCast(mx.c.mlx_array_data_int32(count_array)[0]);
    const names = [_][]const u8{ "left", "right", "tail" };
    var projections: [3]mx.Array = undefined;
    var original: [3]lanes.Linear = undefined;
    for (names[0..count], 0..) |name, i| {
        var buffer: [64]u8 = undefined;
        const w = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_weight", .{i}));
        const scales = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_scales", .{i}));
        const biases = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_biases", .{i}));
        const group = @divExact(mx.dim(x, -1), mx.dim(scales, -1));
        try weights.putLinear(name, try lanes.Linear.initFormatWide(s, w, scales, biases, .{ .bits = 4, .group_size = group }, wide and count == 2));
        original[i] = try weights.linear(name);
        projections[i] = try original[i].apply(&kernels, s, .{ .x = x });
    }
    const stack = (try weights.fused(names[0..count])).?;
    try std.testing.expectEqual(original[0].tiled, stack.tiled);
    try std.testing.expectEqual(original[0].tile_width, stack.tile_width);
    try std.testing.expectEqual(original[0].splitK(), stack.splitK());
    try std.testing.expectEqual(stack.weight.ctx, (try weights.fused(names[0..count])).?.weight.ctx);
    try equalBits(s, try stack.apply(&kernels, s, .{ .x = x }), expected);
    try equalBits(s, try s.cat(projections[0..count], -1), expected);
    for (names[0..count], 0..) |name, i| {
        const member = try weights.linear(name);
        try equalBits(s, try member.apply(&kernels, s, .{ .x = x }), projections[i]);
        if (original[0].tiled and !original[i].tiled) try std.testing.expectEqual(original[i].weight.ctx, member.weight.ctx);
    }
    var different = original[0];
    if (original[0].tiled) {
        different.tile_width = if (original[0].tile_width == 32) 64 else 32;
        try std.testing.expect(!lanes.Linear.stackCompatible(original[0], different));
    }
    different = original[0];
    different.format.?.group_size = if (original[0].format.?.group_size == 32) 64 else 32;
    try std.testing.expect(!lanes.Linear.stackCompatible(original[0], different));
    different = original[0];
    different.split_k = original[0].splitK() + 1;
    try std.testing.expect(!lanes.Linear.stackCompatible(original[0], different));
    different = original[0];
    different.signs = x;
    different.rotation_id = 1;
    try std.testing.expect(!lanes.Linear.stackCompatible(original[0], different));
    if (original[0].n == 16384) {
        var unforced = stack;
        unforced.split_k = null;
        try std.testing.expect(stack.splitK() != unforced.splitK());
    }
}
fn checkBonsaiReuse(s: *mx.Scope, fixture: *@import("checkpoint.zig").Store, expected: mx.Array) !void {
    const lanes = @import("lanes.zig");
    var model = @import("model.zig").Model{ .weights = @import("weights.zig").Weights.init(), .kernels = mx.Kernels.init() };
    defer model.deinit();
    const x = try fixture.get("source_input");
    const signs = try fixture.get("source_signs");
    const names = [_][]const u8{ "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "opposite" };
    for (names, 0..) |name, i| {
        var buffer: [64]u8 = undefined;
        const part = @min(i, 1);
        const w = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_weight", .{part}));
        const scales = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_scales", .{part}));
        const biases = try fixture.get(try std.fmt.bufPrint(&buffer, "member{d}_biases", .{part}));
        const transform = if (i == 2) try s.unary(mx.c.mlx_negative, signs) else signs;
        var linear = try lanes.Linear.initFormat(s, w, scales, biases, .{ .bits = 2, .group_size = 64 });
        linear.signs = mx.retain(transform) catch |err| {
            linear.deinit();
            return err;
        };
        try model.weights.putLinear(name, linear);
    }
    const left = try model.weights.linear(names[0]);
    const right = try model.weights.linear(names[1]);
    const opposite = try model.weights.linear(names[2]);
    try std.testing.expect(left.tiled and right.tiled and opposite.tiled);
    try std.testing.expectEqual(left.rotation_id, right.rotation_id);
    try std.testing.expect(left.rotation_id != opposite.rotation_id);
    if (left.n == 64 and mx.dim(x, 1) == 1) {
        var original = left;
        original.weight = try fixture.get("member0_weight");
        original.tiled = false;
        const repeated: [129]mx.Array = @splat(x);
        const prompt = try s.cat(&repeated, 1);
        try equalBits(s, try left.prefill(&model.kernels, s, prompt), try original.prefill(&model.kernels, s, prompt));
        var selected = try left.selectRanges(s, &.{ .{ 1, 4 }, .{ 11, 16 } });
        defer selected.deinit();
        var reference = try original.selectRanges(s, &.{ .{ 1, 4 }, .{ 11, 16 } });
        defer reference.deinit();
        try std.testing.expect(!selected.tiled);
        try equalBits(s, selected.weight, reference.weight);
        try equalBits(s, try selected.apply(&model.kernels, s, .{ .x = x }), try reference.apply(&model.kernels, s, .{ .x = x }));
        try equalBits(s, try selected.prefill(&model.kernels, s, prompt), try reference.prefill(&model.kernels, s, prompt));
    }
    var cache = lanes.ProjectionCache{};
    model.projection_cache = &cache;
    const input = lanes.Act{ .x = x };
    const rotated = try cache.prepare(left, &model.kernels, s, input);
    const repeated = try cache.prepare(right, &model.kernels, s, input);
    try std.testing.expect(rotated.x.ctx == repeated.x.ctx and rotated.sums.?.ctx == repeated.sums.?.ctx);
    const changed = try cache.prepare(opposite, &model.kernels, s, input);
    try std.testing.expect(rotated.x.ctx != changed.x.ctx);
    var plain = opposite;
    plain.signs = mx.empty;
    try equalBits(s, try plain.apply(&model.kernels, s, changed), try opposite.apply(&model.kernels, s, input));
    const other_input = lanes.Act{ .x = try s.unary(mx.c.mlx_negative, x) };
    const other = try cache.prepare(left, &model.kernels, s, other_input);
    try std.testing.expect(other.x.ctx != rotated.x.ctx);
    plain = left;
    plain.signs = mx.empty;
    try equalBits(s, try plain.apply(&model.kernels, s, other), try left.apply(&model.kernels, s, other_input));
    const a = try model.project(s, 0, "mlp.gate_proj", input);
    const b = try model.project(s, 0, "mlp.up_proj", input);
    try equalBits(s, try s.cat(&.{ a, b }, -1), expected);
    const stack = (try model.weights.fused(names[0..2])).?;
    try std.testing.expect(stack.tiled);
    try std.testing.expectEqual(left.splitK(), stack.splitK());
    if (left.n == 16384) {
        var unforced = stack;
        unforced.split_k = null;
        try std.testing.expect(stack.splitK() != unforced.splitK());
    }
    try std.testing.expect(try model.weights.fused(&.{ names[0], names[2] }) == null);
    var different_format = right;
    different_format.format.?.bits = 4;
    try std.testing.expect(!lanes.Linear.stackCompatible(left, different_format));
    model.projection_cache = null;
    try std.testing.expectError(error.MissingWeight, model.forward(&.{1}, &.{-1}));
    try std.testing.expect(model.projection_cache == null);
}
fn matrix(s: *mx.Scope, a: mx.Array, n: i32, width: i32) !mx.Array {
    return s.reshape(try s.slice(try s.reshape(a, &.{-1}), 0, 0, n * width), &.{ n, width });
}
pub fn equalBits(s: *mx.Scope, a: mx.Array, b: mx.Array) !void {
    const x = try raw(s, a);
    const y = try raw(s, b);
    try mx.evalMany(&.{ x, y }, false);
    const count = mx.c.mlx_array_size(x);
    if (count != mx.c.mlx_array_size(y) or !std.mem.eql(u8, mx.c.mlx_array_data_uint8(x)[0..count], mx.c.mlx_array_data_uint8(y)[0..count])) return error.NativeAffineMismatch;
}

fn gemmaInput(s: *mx.Scope, shape: []const i32, seed: usize) !mx.Array {
    var values: [2 * 160 * 64]f32 = undefined;
    var count: usize = 1;
    for (shape) |dim| count *= @intCast(dim);
    for (values[0..count], 0..) |*value, i| value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + seed * 7) % 61)) - 30)) / 128;
    return s.cast(try s.data(values[0..count].ptr, shape, mx.f32t), mx.bf16);
}

fn gemmaActiveBytes() !usize {
    try mx.check(mx.c.mlx_synchronize(mx.stream));
    var bytes: usize = 0;
    try mx.check(mx.c.mlx_get_active_memory(&bytes));
    return bytes;
}

fn checkGemmaAttentionCache() !void {
    const gemma = @import("gemma_ops.zig");
    {
        var kernels = mx.Kernels.init();
        defer kernels.deinit();
        var cache = gemma.Attention{};
        defer cache.deinit();
        var results = mx.Scope{};
        defer results.deinit();
        var actual: [8]mx.Array = undefined;
        var expected: [8]mx.Array = undefined;
        for (0..actual.len) |step| {
            var scope = mx.Scope{};
            defer scope.deinit();
            var inputs = [_]mx.Array{
                try gemmaInput(&scope, &.{ 2, 4, 64 }, step),
                try gemmaInput(&scope, &.{ 1, 2, if (step == 7) 160 else 128, 64 }, step + 1),
                try gemmaInput(&scope, &.{ 1, 2, if (step == 7) 160 else 128, 64 }, step + 2),
                try gemmaInput(&scope, &.{ 2, 2, 64 }, step + 3),
                try gemmaInput(&scope, &.{ 2, 2, 64 }, step + 4),
            };
            if (step == 2) for (&inputs) |*input| {
                const axis: usize = mx.shape(input.*).len - 1;
                const width = mx.dim(input.*, @intCast(axis));
                const padded = try scope.contiguous(try scope.cat(&.{ input.*, input.* }, @intCast(axis)));
                input.* = try scope.slice(padded, axis, 0, width);
                try mx.eval(input.*);
                try std.testing.expect(mx.c.mlx_array_strides(input.*)[axis - 1] > @as(usize, @intCast(width)));
            };
            var rows = try gemma.Rows.init(&scope, if (step >= 6) &.{ 127, 128 } else if (step == 1) &.{ 126, 127 } else &.{ 62, 63 }, 32, 128, 64);
            if (step >= 3) {
                rows.positions = try scope.reshape(rows.positions, &.{ 1, 8 });
                rows.lows = try scope.reshape(rows.lows, &.{ 1, 8 });
                rows.meta = try scope.reshape(rows.meta, &.{ 1, 8 });
            }
            if (step >= 4) {
                const padding = try scope.zeros(&.{ 1, 8 }, mx.c.MLX_INT32);
                rows.positions = try scope.cat(&.{ rows.positions, padding }, 1);
                rows.lows = try scope.cat(&.{ rows.lows, padding }, 1);
                rows.meta = try scope.cat(&.{ rows.meta, padding }, 1);
            }
            const scale: f32 = if (step >= 5) 0.5 else 1;
            const previous = cache.closure.ctx;
            actual[step] = try cache.apply(&kernels, &results, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], rows, scale);
            expected[step] = try gemma.attentionRows(&kernels, &results, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], rows, scale);
            if (step == 1 or step == 2) try std.testing.expectEqual(previous, cache.closure.ctx) else if (step > 0) try std.testing.expect(previous != cache.closure.ctx);
            const current = cache.closure.ctx;
            try std.testing.expectError(error.InvalidGemmaAttention, cache.apply(&kernels, &results, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], rows, std.math.nan(f32)));
            var invalid = rows;
            invalid.ring = 0;
            invalid.first_position = mx.dim(inputs[1], 2) + 1;
            try std.testing.expectError(error.InvalidGemmaAttention, cache.apply(&kernels, &results, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], invalid, scale));
            if (step == 0) {
                const allocator = mx.allocator;
                var failing = std.testing.FailingAllocator.init(allocator, .{ .fail_index = 0 });
                mx.allocator = failing.allocator();
                defer mx.allocator = allocator;
                try std.testing.expectError(error.OutOfMemory, cache.apply(&kernels, &results, inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], rows, 2));
            }
            try std.testing.expectEqual(current, cache.closure.ctx);
        }
        cache.deinit();
        for (actual, expected) |got, want| try equalBits(&results, got, want);
    }
    const previous = mx.allocator;
    var tracking = std.testing.FailingAllocator.init(previous, .{});
    mx.allocator = tracking.allocator();
    defer mx.allocator = previous;
    const baseline = try gemmaActiveBytes();
    {
        var kernels = mx.Kernels.init();
        defer kernels.deinit();
        var caches: [2]gemma.Attention = @splat(.{});
        defer for (&caches) |*cache| cache.deinit();
        var retained: usize = 0;
        for (0..3) |cycle| {
            for ([_]i32{ 1, 2, 3, 5, 8, 9, 16 }, 0..) |count, index| {
                {
                    var scope = mx.Scope{};
                    defer scope.deinit();
                    var positions: [16]i32 = undefined;
                    for (positions[0..@intCast(count)], 0..) |*position, i| position.* = @intCast(i + cycle * 64);
                    for (&caches, 0..) |*cache, kind| {
                        const dims: i32 = if (kind == 0) 256 else 512;
                        const q = try scope.zeros(&.{ count, 4, dims }, mx.bf16);
                        const old = try scope.zeros(&.{ 1, 2, @as(i32, @intCast(256 + index * 64)), dims }, mx.bf16);
                        const fresh = try scope.zeros(&.{ 2, count, dims }, mx.bf16);
                        const rows = try gemma.Rows.init(&scope, positions[0..@intCast(count)], 0, 0, dims);
                        try mx.eval(try cache.apply(&kernels, &scope, q, old, old, fresh, fresh, rows, 1));
                    }
                }
                try std.testing.expectEqual(baseline, try gemmaActiveBytes());
            }
            const held = tracking.allocated_bytes - tracking.freed_bytes;
            if (cycle == 0) retained = held else try std.testing.expectEqual(retained, held);
        }
    }
    try std.testing.expectEqual(tracking.allocated_bytes, tracking.freed_bytes);
    try std.testing.expectEqual(baseline, try gemmaActiveBytes());
    std.debug.print("PASS: compiled Gemma attention preserves runtime metadata, strided inputs and pending graphs without retaining tensors or shape history.\n", .{});
}
