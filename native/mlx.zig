//! MLX-C ownership boundary. A Scope owns temporary graph handles; persistent
//! weights/cache entries explicitly retain their own handles with `retain`.
const std = @import("std");
pub const c = @import("mlx_c");
pub const Array = c.mlx_array;
pub const empty: Array = .{ .ctx = null };
pub const bf16 = c.MLX_BFLOAT16;
pub const f32t = c.MLX_FLOAT32;
pub const i32t = c.MLX_INT32;
pub var stream: c.mlx_stream = .{ .ctx = null };
// Replaced only by the single-threaded allocation diagnostic; production uses libc.
pub var allocator: std.mem.Allocator = std.heap.c_allocator;
pub var tensor_units = false;
pub var force_simd = false;
pub var gpu_generation: u32 = 0;
// M1/M2 pipelines can have register-dependent limits. Eight physical groups
// stay within their guaranteed limit without changing arithmetic chunk counts.
pub var simd_groups: i32 = 8;

fn onError(msg: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    @import("server_live.zig").print("MLX: {s}\n", .{msg});
}
pub fn check(rc: c_int) !void {
    if (rc != 0) return error.MlxFailure;
}
pub fn saveArray(path: [:0]const u8, value: Array) !void {
    try @import("storage.zig").check(path, c.mlx_array_nbytes(value) +| 65536);
    try check(c.mlx_save(path, value));
}
pub fn checkVersion() !void {
    var version_string = c.mlx_string_new();
    defer _ = c.mlx_string_free(version_string);
    try check(c.mlx_version(&version_string));
    const version = std.mem.span(c.mlx_string_data(version_string));
    const expected = @import("native_runtime").mlx_version;
    if (!std.mem.eql(u8, version, expected)) {
        std.debug.print("Native MLX version {s} differs from pin {s}; rebuild the MLX prefix from native/dependencies.json.\n", .{ version, expected });
        return error.MlxVersionMismatch;
    }
}
pub fn init() !void {
    try checkVersion();
    c.mlx_set_error_handler(onError, null, null);
    const dev = c.mlx_device_new_type(c.MLX_GPU, 0);
    defer _ = c.mlx_device_free(dev);
    var info = c.mlx_device_info_new();
    defer _ = c.mlx_device_info_free(info);
    try check(c.mlx_device_info_get(&info, dev));
    var arch: [*c]const u8 = null;
    try check(c.mlx_device_info_get_string(&arch, info, "architecture"));
    const name = std.mem.span(arch);
    const prefix = "applegpu_g";
    if (std.mem.startsWith(u8, name, prefix)) {
        var end: usize = prefix.len;
        while (end < name.len and std.ascii.isDigit(name[end])) : (end += 1) {}
        gpu_generation = std.fmt.parseInt(u32, name[prefix.len..end], 10) catch 0;
        simd_groups = if (gpu_generation >= 15) 16 else 8;
        tensor_units = gpu_generation >= 17 and !force_simd;
    }
    std.debug.print("Metal: {s}, {s} backend\n", .{ name, if (tensor_units) "tensor" else "SIMD" });
    try check(c.mlx_get_default_stream(&stream, dev));
    var previous: usize = 0;
    try check(c.mlx_set_cache_limit(&previous, 2 * 1024 * 1024 * 1024));
}
pub fn shutdown() void {
    _ = c.mlx_stream_free(stream);
    stream = .{ .ctx = null };
}
pub fn free(a: Array) void {
    if (a.ctx != null) _ = c.mlx_array_free(a);
}
pub fn retain(a: Array) !Array {
    var out = c.mlx_array_new();
    errdefer free(out);
    try check(c.mlx_array_set(&out, a));
    return out;
}
pub fn replace(dst: *Array, src: Array) !void {
    const a = try retain(src);
    free(dst.*);
    dst.* = a;
}
pub fn dim(a: Array, axis: c_int) c_int {
    return c.mlx_array_dim(a, axis);
}
pub fn dtype(a: Array) c.mlx_dtype {
    return c.mlx_array_dtype(a);
}
pub fn shape(a: Array) []const c_int {
    return c.mlx_array_shape(a)[0..c.mlx_array_ndim(a)];
}
pub fn eval(a: Array) !void {
    try check(c.mlx_array_eval(a));
}
pub fn evalMany(arrays: []const Array, async_: bool) !void {
    const v = c.mlx_vector_array_new_data(arrays.ptr, arrays.len);
    defer _ = c.mlx_vector_array_free(v);
    try check(if (async_) c.mlx_async_eval(v) else c.mlx_eval(v));
}
pub fn opt(n: c_int) c.mlx_optional_int {
    return .{ .value = n, .has_value = true };
}

pub const Scope = struct {
    arrays: std.ArrayList(Array) = .empty,
    pub fn deinit(s: *Scope) void {
        s.clear();
        s.arrays.deinit(allocator);
    }
    pub fn clear(s: *Scope) void {
        for (s.arrays.items) |a| free(a);
        s.arrays.clearRetainingCapacity();
    }
    pub fn own(s: *Scope, a: Array) !Array {
        errdefer free(a);
        if (a.ctx == null) return error.MlxFailure;
        try s.arrays.append(allocator, a);
        return a;
    }
    pub fn result(s: *Scope, rc: c_int, a: Array) !Array {
        check(rc) catch |err| {
            free(a);
            return err;
        };
        return s.own(a);
    }
    pub fn data(s: *Scope, ptr: anytype, dims: []const c_int, dt: c.mlx_dtype) !Array {
        return s.own(c.mlx_array_new_data(@ptrCast(ptr), dims.ptr, @intCast(dims.len), dt));
    }
    pub fn ints(s: *Scope, values: []const i32) !Array {
        return s.data(values.ptr, &.{@intCast(values.len)}, i32t);
    }
    pub fn scalar(s: *Scope, value: f32) !Array {
        return s.data(&value, &.{1}, f32t);
    }
    pub fn zeros(s: *Scope, dims: []const c_int, dt: c.mlx_dtype) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_zeros(&a, dims.ptr, dims.len, dt, stream);
        return s.result(rc, a);
    }
    pub fn reshape(s: *Scope, x: Array, dims: []const c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_reshape(&a, x, dims.ptr, dims.len, stream);
        return s.result(rc, a);
    }
    pub fn transpose(s: *Scope, x: Array, axes: []const c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_transpose_axes(&a, x, axes.ptr, axes.len, stream);
        return s.result(rc, a);
    }
    pub fn cast(s: *Scope, x: Array, dt: c.mlx_dtype) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_astype(&a, x, dt, stream);
        return s.result(rc, a);
    }
    pub fn contiguous(s: *Scope, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_contiguous(&a, x, false, stream);
        return s.result(rc, a);
    }
    pub fn slice(s: *Scope, x: Array, axis: usize, start: c_int, end: c_int) !Array {
        var starts: [8]c_int = @splat(0);
        var stops: [8]c_int = undefined;
        const steps: [8]c_int = @splat(1);
        const dims = shape(x);
        @memcpy(stops[0..dims.len], dims);
        starts[axis] = start;
        stops[axis] = end;
        var a = c.mlx_array_new();
        const rc = c.mlx_slice(&a, x, &starts, dims.len, &stops, dims.len, &steps, dims.len, stream);
        return s.result(rc, a);
    }
    pub fn take(s: *Scope, x: Array, ids: Array, axis: c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_take_axis(&a, x, ids, axis, stream);
        return s.result(rc, a);
    }
    pub fn cat(s: *Scope, xs: []const Array, axis: c_int) !Array {
        const v = c.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = c.mlx_vector_array_free(v);
        var a = c.mlx_array_new();
        const rc = c.mlx_concatenate_axis(&a, v, axis, stream);
        return s.result(rc, a);
    }
    pub fn stack(s: *Scope, xs: []const Array, axis: c_int) !Array {
        const v = c.mlx_vector_array_new_data(xs.ptr, xs.len);
        defer _ = c.mlx_vector_array_free(v);
        var a = c.mlx_array_new();
        const rc = c.mlx_stack_axis(&a, v, axis, stream);
        return s.result(rc, a);
    }
    pub fn unary(s: *Scope, comptime op: anytype, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = op(&a, x, stream);
        return s.result(rc, a);
    }
    pub fn binary(s: *Scope, comptime op: anytype, x: Array, y: Array) !Array {
        var a = c.mlx_array_new();
        const rc = op(&a, x, y, stream);
        return s.result(rc, a);
    }
    pub fn rms(s: *Scope, x: Array, w: Array) !Array {
        return s.rmsEpsilon(x, w, 1e-6);
    }
    pub fn rmsEpsilon(s: *Scope, x: Array, w: Array, epsilon: f32) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_fast_rms_norm(&a, x, w, epsilon, stream);
        return s.result(rc, a);
    }
    pub fn rope(s: *Scope, x: Array, positions: Array, dims: c_int) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_fast_rope_dynamic(&a, x, dims, false, .{ .value = 10000000, .has_value = true }, 1, positions, empty, stream);
        return s.result(rc, a);
    }
    pub fn dequant(s: *Scope, w: Array, scales: Array, biases: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_dequantize(&a, w, scales, biases, opt(64), opt(4), "affine", empty, .{ .value = bf16, .has_value = true }, stream);
        return s.result(rc, a);
    }
    pub fn argmax(s: *Scope, x: Array) !Array {
        var a = c.mlx_array_new();
        const rc = c.mlx_argmax_axis(&a, x, -1, false, stream);
        return s.result(rc, a);
    }
};

pub const Template = struct { name: [:0]const u8, value: union(enum) { int: c_int, dtype: c.mlx_dtype, boolean: bool } };
pub fn ti(name: [:0]const u8, value: c_int) Template {
    return .{ .name = name, .value = .{ .int = value } };
}
pub fn td(name: [:0]const u8, value: c.mlx_dtype) Template {
    return .{ .name = name, .value = .{ .dtype = value } };
}
pub fn tb(name: [:0]const u8, value: bool) Template {
    return .{ .name = name, .value = .{ .boolean = value } };
}
pub const Output = struct { shape: []const c_int, dtype: c.mlx_dtype = bf16 };
const KernelKey = struct {
    name: []const u8,
    reserve: i64,
    specialized: bool,
    templates: []const Template,
    names: []u8 = &.{},

    fn clone(key: KernelKey) !KernelKey {
        var bytes = key.name.len;
        for (key.templates) |t| bytes = try std.math.add(usize, bytes, try std.math.add(usize, t.name.len, 1));
        const names = try allocator.alloc(u8, bytes);
        errdefer allocator.free(names);
        const templates = try allocator.dupe(Template, key.templates);
        errdefer allocator.free(templates);
        @memcpy(names[0..key.name.len], key.name);
        var at = key.name.len;
        for (templates) |*t| {
            @memcpy(names[at..][0..t.name.len], t.name);
            names[at + t.name.len] = 0;
            t.name = names[at..][0..t.name.len :0];
            at += t.name.len + 1;
        }
        return .{ .name = names[0..key.name.len], .reserve = key.reserve, .specialized = key.specialized, .templates = templates, .names = names };
    }

    fn deinit(key: KernelKey) void {
        allocator.free(key.templates);
        allocator.free(key.names);
    }

    const Context = struct {
        pub fn hash(_: Context, key: KernelKey) u64 {
            var h = std.hash.Wyhash.init(0);
            std.hash.autoHash(&h, key.name.len);
            h.update(key.name);
            std.hash.autoHash(&h, key.reserve);
            std.hash.autoHash(&h, key.specialized);
            std.hash.autoHash(&h, key.templates.len);
            for (key.templates) |t| {
                std.hash.autoHash(&h, t.name.len);
                h.update(t.name);
                std.hash.autoHash(&h, std.meta.activeTag(t.value));
                switch (t.value) {
                    inline else => |value| std.hash.autoHash(&h, value),
                }
            }
            return h.final();
        }

        pub fn eql(_: Context, a: KernelKey, b: KernelKey) bool {
            if (a.reserve != b.reserve or a.specialized != b.specialized or a.templates.len != b.templates.len or !std.mem.eql(u8, a.name, b.name)) return false;
            for (a.templates, b.templates) |left, right| {
                if (!std.mem.eql(u8, left.name, right.name) or std.meta.activeTag(left.value) != std.meta.activeTag(right.value)) return false;
                switch (left.value) {
                    inline else => |value, tag| if (value != @field(right.value, @tagName(tag))) return false,
                }
            }
            return true;
        }
    };
};

const LaunchConfig = struct {
    handle: c.mlx_fast_metal_kernel_config,
    grid: [3]c_int,
    group: [3]c_int,
    init_bits: ?u32,
    outputs: []Output,
    shapes: []c_int,
    templates: []Template,
    names: []u8,

    fn init(grid: [3]c_int, group: [3]c_int, outputs: []const Output, templates: []const Template, init_value: ?f32) !LaunchConfig {
        var shape_count: usize = 0;
        for (outputs) |output| shape_count = try std.math.add(usize, shape_count, output.shape.len);
        const owned_outputs = try allocator.dupe(Output, outputs);
        errdefer allocator.free(owned_outputs);
        const shapes = try allocator.alloc(c_int, shape_count);
        errdefer allocator.free(shapes);
        var at: usize = 0;
        for (owned_outputs, outputs) |*owned, output| {
            const dims = shapes[at..][0..output.shape.len];
            @memcpy(dims, output.shape);
            owned.shape = dims;
            at += dims.len;
        }
        var name_bytes: usize = 0;
        for (templates) |t| name_bytes = try std.math.add(usize, name_bytes, try std.math.add(usize, t.name.len, 1));
        const names = try allocator.alloc(u8, name_bytes);
        errdefer allocator.free(names);
        const owned_templates = try allocator.dupe(Template, templates);
        errdefer allocator.free(owned_templates);
        at = 0;
        for (owned_templates) |*t| {
            @memcpy(names[at..][0..t.name.len], t.name);
            names[at + t.name.len] = 0;
            t.name = names[at..][0..t.name.len :0];
            at += t.name.len + 1;
        }
        return .{
            .handle = try newLaunchConfig(grid, group, outputs, templates, init_value),
            .grid = grid,
            .group = group,
            .init_bits = if (init_value) |value| @bitCast(value) else null,
            .outputs = owned_outputs,
            .shapes = shapes,
            .templates = owned_templates,
            .names = names,
        };
    }

    fn deinit(config: *LaunchConfig) void {
        c.mlx_fast_metal_kernel_config_free(config.handle);
        allocator.free(config.outputs);
        allocator.free(config.shapes);
        allocator.free(config.templates);
        allocator.free(config.names);
    }

    fn matches(config: *const LaunchConfig, grid: [3]c_int, group: [3]c_int, outputs: []const Output, templates: []const Template, init_value: ?f32) bool {
        const init_bits: ?u32 = if (init_value) |value| @bitCast(value) else null;
        if (!std.mem.eql(c_int, &config.grid, &grid) or !std.mem.eql(c_int, &config.group, &group) or config.init_bits != init_bits or config.outputs.len != outputs.len or config.templates.len != templates.len) return false;
        for (config.templates, templates) |cached, t| {
            if (!std.mem.eql(u8, cached.name, t.name) or std.meta.activeTag(cached.value) != std.meta.activeTag(t.value)) return false;
            switch (cached.value) {
                inline else => |value, tag| if (value != @field(t.value, @tagName(tag))) return false,
            }
        }
        for (config.outputs, outputs) |cached, output| {
            if (cached.dtype != output.dtype or !std.mem.eql(c_int, cached.shape, output.shape)) return false;
        }
        return true;
    }
};

fn newLaunchConfig(grid: [3]c_int, group: [3]c_int, outputs: []const Output, templates: []const Template, init_value: ?f32) !c.mlx_fast_metal_kernel_config {
    const cfg = c.mlx_fast_metal_kernel_config_new();
    if (cfg.ctx == null) return error.MlxFailure;
    errdefer c.mlx_fast_metal_kernel_config_free(cfg);
    if (init_value) |value| try check(c.mlx_fast_metal_kernel_config_set_init_value(cfg, value));
    try check(c.mlx_fast_metal_kernel_config_set_grid(cfg, grid[0], grid[1], grid[2]));
    try check(c.mlx_fast_metal_kernel_config_set_thread_group(cfg, group[0], group[1], group[2]));
    for (templates) |t| try check(switch (t.value) {
        .int => |v| c.mlx_fast_metal_kernel_config_add_template_arg_int(cfg, t.name, v),
        .dtype => |v| c.mlx_fast_metal_kernel_config_add_template_arg_dtype(cfg, t.name, v),
        .boolean => |v| c.mlx_fast_metal_kernel_config_add_template_arg_bool(cfg, t.name, v),
    });
    for (outputs) |o| try check(c.mlx_fast_metal_kernel_config_add_output_arg(cfg, o.shape.ptr, o.shape.len, o.dtype));
    return cfg;
}

const Kernel = struct {
    handle: c.mlx_fast_metal_kernel,
    launch: ?LaunchConfig = null,

    fn config(kernel: *Kernel, grid: [3]c_int, group: [3]c_int, outputs: []const Output, templates: []const Template, init_value: ?f32) !c.mlx_fast_metal_kernel_config {
        if (kernel.launch) |*cached| if (cached.matches(grid, group, outputs, templates, init_value)) return cached.handle;
        const replacement = try LaunchConfig.init(grid, group, outputs, templates, init_value);
        if (kernel.launch) |*previous| previous.deinit();
        kernel.launch = replacement;
        return replacement.handle;
    }
};

const LaunchVectors = struct {
    blank: c.mlx_vector_array = .{ .ctx = null },
    inputs: c.mlx_vector_array = .{ .ctx = null },
    outputs: c.mlx_vector_array = .{ .ctx = null },
    active: bool = false,

    fn prepare(vectors: *LaunchVectors) !void {
        if (vectors.active) return error.ReentrantKernelLaunch;
        errdefer vectors.deinit();
        inline for (.{ "blank", "inputs", "outputs" }) |field| {
            const value = &@field(vectors, field);
            if (value.ctx == null) value.* = c.mlx_vector_array_new();
            if (value.ctx == null) return error.MlxFailure;
        }
        vectors.active = true;
    }

    fn clear(vectors: *LaunchVectors) void {
        inline for (.{ "inputs", "outputs" }) |field| {
            const value = &@field(vectors, field);
            if (value.ctx != null and c.mlx_vector_array_set(value, vectors.blank) != 0) {
                _ = c.mlx_vector_array_free(value.*);
                value.* = .{ .ctx = null };
            }
        }
        vectors.active = false;
    }

    fn read(vectors: *LaunchVectors, s: *Scope, results: []Array) !void {
        for (results, 0..) |*result, i| {
            var value = c.mlx_array_new();
            const rc = c.mlx_vector_array_get(&value, vectors.outputs, i);
            result.* = try s.result(rc, value);
        }
    }

    fn deinit(vectors: *LaunchVectors) void {
        inline for (.{ "blank", "inputs", "outputs" }) |field| {
            const value = @field(vectors, field);
            if (value.ctx != null) _ = c.mlx_vector_array_free(value);
        }
        vectors.* = .{};
    }
};

pub const Kernels = struct {
    const Map = std.HashMap(KernelKey, Kernel, KernelKey.Context, std.hash_map.default_max_load_percentage);
    const Constant = struct {
        bytes: [32]u8 = @splat(0),
        len: usize = 0,
        dtype: c.mlx_dtype = i32t,
        value: Array = empty,
    };
    items: Map,
    constants: [64]Constant = @splat(.{}),
    next_constant: usize = 0,
    vectors: LaunchVectors = .{},
    affine: @import("deepseek_dense.zig").Dense = .{},
    flash_prefill: @import("flash_prefill_mm.zig").State = .{},
    flash_rows: @import("flash_ops.zig").State = .{},
    pub fn init() Kernels {
        return .{ .items = Map.init(allocator) };
    }
    pub fn deinit(k: *Kernels) void {
        for (k.constants) |entry| free(entry.value);
        k.vectors.deinit();
        k.affine.deinit();
        k.flash_rows.deinit();
        var it = k.items.iterator();
        while (it.next()) |entry| {
            if (entry.value_ptr.launch) |*launch| launch.deinit();
            c.mlx_fast_metal_kernel_free(entry.value_ptr.handle);
            entry.key_ptr.deinit();
        }
        k.items.deinit();
    }
    fn constant(k: *Kernels, s: *Scope, comptime T: type, values: []const T, element_type: c.mlx_dtype) !Array {
        const bytes = std.mem.sliceAsBytes(values);
        if (bytes.len > 32) return s.data(values.ptr, &.{@intCast(values.len)}, element_type);
        for (&k.constants) |*entry| {
            if (entry.value.ctx != null and entry.dtype == element_type and entry.len == bytes.len and std.mem.eql(u8, entry.bytes[0..entry.len], bytes))
                return s.own(try retain(entry.value));
        }
        const value = try s.data(values.ptr, &.{@intCast(values.len)}, element_type);
        const held = try retain(value);
        const entry = &k.constants[k.next_constant];
        free(entry.value);
        entry.* = .{ .len = bytes.len, .dtype = element_type, .value = held };
        @memcpy(entry.bytes[0..bytes.len], bytes);
        k.next_constant = (k.next_constant + 1) % k.constants.len;
        return value;
    }
    pub fn constantInts(k: *Kernels, s: *Scope, values: []const i32) !Array {
        return k.constant(s, i32, values, i32t);
    }
    pub fn constantScalar(k: *Kernels, s: *Scope, value: f32) !Array {
        return k.constant(s, f32, &.{value}, f32t);
    }
    pub fn run(k: *Kernels, s: *Scope, spec: @import("kernel_sources.zig").Spec, inputs: []const Array, templates: []const Template, grid: [3]c_int, group: [3]c_int, outputs: []const Output) ![5]Array {
        if (outputs.len > 5) return error.TooManyKernelOutputs;
        var result_: [5]Array = @splat(empty);
        try k.runInto(s, spec, inputs, templates, grid, group, outputs, result_[0..outputs.len], null);
        return result_;
    }

    // A closure's host callback must use its own Kernels instance.
    pub fn call(k: *Kernels, s: *Scope, closure: c.mlx_closure, inputs: []const Array, results: []Array) !void {
        try k.vectors.prepare();
        defer k.vectors.clear();
        try check(c.mlx_vector_array_append_data(k.vectors.inputs, inputs.ptr, inputs.len));
        try check(c.mlx_closure_apply(&k.vectors.outputs, closure, k.vectors.inputs));
        if (c.mlx_vector_array_size(k.vectors.outputs) != results.len) return error.InvalidKernelArity;
        try k.vectors.read(s, results);
    }

    pub fn runInto(k: *Kernels, s: *Scope, spec: @import("kernel_sources.zig").Spec, inputs: []const Array, templates: []const Template, grid: [3]c_int, group: [3]c_int, outputs: []const Output, result_: []Array, init_value: ?f32) !void {
        if (result_.len != outputs.len or outputs.len != spec.outputs.len or inputs.len != spec.inputs.len) return error.InvalidKernelArity;
        if (group[0] < 1 or group[1] < 1 or group[2] < 1) return error.InvalidThreadgroup;
        const threads = @as(i64, group[0]) * group[1] * group[2];
        if (threads > 1024) return error.InvalidThreadgroup;
        const reserve: i64 = if (spec.reserve > 0) spec.reserve else if (spec.reserve_launch and threads > 256) threads else 0;
        const specialize = spec.bake_templates or spec.reserve_launch or spec.reserve > 0;
        if (specialize) for (templates) |t| if (t.value == .dtype) return error.UnsupportedReservedTemplate;
        const key = KernelKey{ .name = spec.name, .reserve = reserve, .specialized = specialize, .templates = if (specialize) templates else &.{} };
        const kernel = k.items.getPtr(key) orelse blk: {
            var name: [2048]u8 = undefined;
            const cache_name = if (specialize) named: {
                var used = (try std.fmt.bufPrint(&name, "{s}_r{d}", .{ spec.name, reserve })).len;
                for (templates) |t| used += (try switch (t.value) {
                    .int => |v| std.fmt.bufPrint(name[used..], "_{s}_i{x}", .{ t.name, @as(u32, @bitCast(v)) }),
                    .boolean => |v| std.fmt.bufPrint(name[used..], "_{s}_b{d}", .{ t.name, @intFromBool(v) }),
                    .dtype => unreachable,
                }).len;
                if (used == name.len) return error.NoSpaceLeft;
                name[used] = 0;
                break :named name[0..used :0];
            } else spec.name;
            const owned_key = try key.clone();
            errdefer owned_key.deinit();
            const ins = c.mlx_vector_string_new();
            defer _ = c.mlx_vector_string_free(ins);
            const outs = c.mlx_vector_string_new();
            defer _ = c.mlx_vector_string_free(outs);
            for (spec.inputs) |n| try check(c.mlx_vector_string_append_value(ins, n));
            for (spec.outputs) |n| try check(c.mlx_vector_string_append_value(outs, n));
            const reserved = if (reserve > 0) try std.fmt.allocPrintSentinel(allocator, "{s}\n[[max_total_threads_per_threadgroup({d})]]\n", .{ spec.header, reserve }, 0) else null;
            defer if (reserved) |text| allocator.free(text);
            var body: std.ArrayList(u8) = .empty;
            defer body.deinit(allocator);
            if (specialize) {
                // MLX-C inserts template declarations after the header, separating
                // its threadgroup attribute from the function. Upstream bakes constants.
                for (templates) |t| switch (t.value) {
                    .int => |v| try body.print(allocator, "  constexpr int {s} = {d};\n", .{ t.name, v }),
                    .boolean => |v| try body.print(allocator, "  constexpr bool {s} = {s};\n", .{ t.name, if (v) "true" else "false" }),
                    .dtype => return error.UnsupportedReservedTemplate,
                };
                try body.appendSlice(allocator, spec.source);
                try body.append(allocator, 0);
            }
            const source = if (specialize) body.items[0 .. body.items.len - 1 :0] else spec.source;
            const handle = c.mlx_fast_metal_kernel_new(cache_name, ins, outs, source, reserved orelse spec.header, spec.contiguous, false);
            if (handle.ctx == null) return error.MlxFailure;
            errdefer c.mlx_fast_metal_kernel_free(handle);
            try k.items.putNoClobber(owned_key, .{ .handle = handle });
            break :blk k.items.getPtr(key).?;
        };
        const cfg = try kernel.config(grid, group, outputs, if (specialize) &.{} else templates, init_value);
        try k.vectors.prepare();
        defer k.vectors.clear();
        // Direct Metal apply constructs graph nodes without invoking host closures.
        // Empty assignment keeps vector capacity while releasing all tensor handles.
        try check(c.mlx_vector_array_append_data(k.vectors.inputs, inputs.ptr, inputs.len));
        try check(c.mlx_fast_metal_kernel_apply(&k.vectors.outputs, kernel.handle, k.vectors.inputs, cfg, stream));
        try k.vectors.read(s, result_);
    }
};
