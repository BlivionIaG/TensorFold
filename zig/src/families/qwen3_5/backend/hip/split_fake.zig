//! Test support: a recording stand-in for the HIP runtime (launches logged by symbol, nothing computed) and a 4B view.

const std = @import("std");
const hip = @import("hip");
const view = @import("model/view.zig");

const abi = hip.abi;

/// The kernels module names a GPU, so a launcher loads on the stand-in runtime.
pub fn available() bool {
    return hip.kernels.arch.len != 0;
}

pub const max_log = 1 << 15;

var names: [max_log][*:0]const u8 = undefined;
var logged: usize = 0;
var next_device: u64 = 1 << 36;
pub const fake_stream: abi.Stream = @ptrFromInt(0x2);

/// The kernels launched since `clear`, by symbol name.
pub fn launches() []const [*:0]const u8 {
    return names[0..logged];
}

pub fn clear() void {
    logged = 0;
}

pub fn nameAt(i: usize) []const u8 {
    return std.mem.span(names[i]);
}

fn ok() callconv(.c) abi.Result {
    return 0;
}

fn loadData(out: *abi.Module, _: [*]const u8) callconv(.c) abi.Result {
    out.* = @ptrFromInt(0x1000);
    return 0;
}

fn getFunction(out: *abi.Function, _: abi.Module, name: [*:0]const u8) callconv(.c) abi.Result {
    out.* = @ptrCast(@constCast(name));
    return 0;
}

fn getGlobal(ptr: *abi.DevicePtr, len: *usize, _: abi.Module, _: [*:0]const u8) callconv(.c) abi.Result {
    ptr.* = @ptrFromInt(1 << 35);
    len.* = 1 << 16;
    return 0;
}

fn launch(f: abi.Function, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: c_uint, _: abi.Stream, _: ?[*]?*anyopaque, _: ?[*]?*anyopaque) callconv(.c) abi.Result {
    if (logged == max_log) return 1;
    names[logged] = @ptrCast(f.?);
    logged += 1;
    return 0;
}

fn malloc(out: *abi.DevicePtr, n: usize) callconv(.c) abi.Result {
    out.* = @ptrFromInt(next_device);
    next_device += std.mem.alignForward(u64, n, 1 << 16) + (1 << 16);
    return 0;
}

fn hostMalloc(out: *abi.DevicePtr, n: usize, _: c_uint) callconv(.c) abi.Result {
    out.* = std.c.malloc(n);
    return 0;
}

fn hostFree(p: abi.DevicePtr) callconv(.c) abi.Result {
    std.c.free(p);
    return 0;
}

fn createStream(out: *abi.Stream, _: c_uint) callconv(.c) abi.Result {
    out.* = fake_stream;
    return 0;
}

fn memInfo(free: *usize, total: *usize) callconv(.c) abi.Result {
    free.* = 1 << 34;
    total.* = 1 << 34;
    return 0;
}

fn attribute(out: *c_int, _: abi.FunctionAttribute, _: abi.Function) callconv(.c) abi.Result {
    out.* = 64;
    return 0;
}

/// The stand-in runtime: every call succeeds, launches are logged, the rest of the API does nothing.
pub fn runtime() hip.Runtime {
    var r: hip.Runtime = .{ .lib = undefined, .api = undefined };
    inline for (@typeInfo(abi.Api).@"struct".field_names) |name| @field(r.api, name) = @ptrCast(&ok);
    r.api.hipModuleLoadData = loadData;
    r.api.hipModuleGetFunction = getFunction;
    r.api.hipModuleGetGlobal = getGlobal;
    r.api.hipModuleLaunchKernel = launch;
    r.api.hipMalloc = malloc;
    r.api.hipHostMalloc = hostMalloc;
    r.api.hipHostFree = hostFree;
    r.api.hipStreamCreateWithFlags = createStream;
    r.api.hipMemGetInfo = memInfo;
    r.api.hipFuncGetAttribute = attribute;
    return r;
}

/// The Qwen3.5-4B text tower's shape: hidden 2560, 32 layers, full attention every fourth, MLX 4-bit groups of 64.
pub const spec_4b: view.Spec = blk: {
    var s: view.Spec = .{
        .hidden = 2560,
        .intermediate = 9216,
        .n_layers = 32,
        .heads = 16,
        .kv_heads = 4,
        .head_dim = 256,
        .key_heads = 16,
        .value_heads = 32,
        .key_dim = 128,
        .value_dim = 128,
        .conv = 4,
        .vocab = 248320,
        .eps = 1e-6,
        .rope_theta = 10_000_000,
        .rotary_dim = 64,
        .bits = 4,
        .group = 64,
        .experts = 0,
        .top_k = 0,
        .moe_width = 0,
        .shared_width = 0,
        .mtp_layers = 1,
    };
    for (0..32) |i| s.full_layers[i] = i % 4 == 3;
    break :blk s;
};

var next_weight: u64 = 1 << 38;

fn address(bytes: usize) u64 {
    const at = next_weight;
    next_weight += std.mem.alignForward(u64, bytes, 1 << 12) + (1 << 12);
    return at;
}

fn projection(n: u32, k: u32) view.Projection {
    return .{ .n = n, .k = k, .handle = .{ .mlx = .{
        .words = address(@as(usize, n) * k / 2),
        .scale = address(@as(usize, n) * k / 32),
        .bias = address(@as(usize, n) * k / 32),
        .tables = .bf16,
        .bits = 4,
        .group = 64,
    } } };
}

fn mlp(s: view.Spec) view.Mlp {
    const h: u32 = @intCast(s.hidden);
    const i: u32 = @intCast(s.intermediate);
    return .{ .dense = .{ .gate = projection(i, h), .up = projection(i, h), .down = projection(h, i) } };
}

fn layerOf(s: view.Spec, index: usize) view.Layer {
    const h: u32 = @intCast(s.hidden);
    const norm = address(s.hidden * 4);
    if (s.full(index)) {
        const hd = s.head_dim;
        return .{ .full = .{
            .input_norm = norm,
            .post_norm = address(s.hidden * 4),
            .q = projection(@intCast(s.heads * 2 * hd), h),
            .k = projection(@intCast(s.kv_heads * hd), h),
            .v = projection(@intCast(s.kv_heads * hd), h),
            .o = projection(h, @intCast(s.heads * hd)),
            .q_norm = address(hd * 4),
            .k_norm = address(hd * 4),
            .mlp = mlp(s),
        } };
    }
    return .{ .linear = .{
        .input_norm = norm,
        .post_norm = address(s.hidden * 4),
        .qkv = projection(@intCast(view.convChannels(s)), h),
        .z = projection(@intCast(s.valueWidth()), h),
        .a = projection(@intCast(s.value_heads), h),
        .b = projection(@intCast(s.value_heads), h),
        .out = projection(h, @intCast(s.valueWidth())),
        .conv = address(view.convChannels(s) * s.conv * 4),
        .a_log = address(s.value_heads * 4),
        .dt_bias = address(s.value_heads * 4),
        .gnorm = address(s.value_dim * 4),
        .mlp = mlp(s),
    } };
}

/// A view of the 4B over the layers of `range`, its tensors fake addresses; free with `deinit`.
pub const Fixture = struct {
    gpa: std.mem.Allocator,
    layers: []view.Layer,
    model: view.Model,

    pub fn init(gpa: std.mem.Allocator, range: view.Range, act: view.Kind) !Fixture {
        const s = spec_4b;
        const r = range.within(s.n_layers);
        try r.check(s.n_layers);
        const layers = try gpa.alloc(view.Layer, r.count(s.n_layers));
        for (layers, r.first..) |*l, i| l.* = layerOf(s, i);
        const h: u32 = @intCast(s.hidden);
        return .{ .gpa = gpa, .layers = layers, .model = .{
            .spec = s,
            .act = act,
            .range = r,
            .embed = if (r.embeds()) projection(@intCast(s.vocab), h) else null,
            .layers = layers,
            .final_norm = if (r.heads(s.n_layers)) address(s.hidden * 4) else 0,
            .head = if (r.heads(s.n_layers)) projection(@intCast(s.vocab), h) else null,
            .qk = .{ .q_weight = address(s.key_dim * 4), .k_weight = address(s.key_dim * 4), .eps = 1e-6 },
        } };
    }

    pub fn deinit(f: *Fixture) void {
        f.gpa.free(f.layers);
    }
};
