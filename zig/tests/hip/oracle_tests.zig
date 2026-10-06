//! The kernel library against the Python ROCm engine's own outputs (tools/zig/qwen_rocm_dump.py), bit for bit.

const std = @import("std");
const hip = @import("hip");
const npy = @import("npy");
const check = @import("check.zig");
const Gpu = check.Gpu;

/// A fixture directory: manifest.json and its .npy files.
const Dir = struct {
    gpu: Gpu,
    dir: []const u8,
    parsed: std.json.Parsed(std.json.Value),

    fn open(gpu: Gpu, dir: []const u8) !Dir {
        const path = try std.fs.path.join(gpu.gpa, &.{ dir, "manifest.json" });
        defer gpu.gpa.free(path);
        const text = try std.Io.Dir.cwd().readFileAlloc(gpu.io, path, gpu.gpa, .limited(1 << 24));
        defer gpu.gpa.free(text);
        return .{ .gpu = gpu, .dir = dir, .parsed = try std.json.parseFromSlice(std.json.Value, gpu.gpa, text, .{}) };
    }

    fn deinit(self: *Dir) void {
        self.parsed.deinit();
    }

    /// The whole .npy file (the caller frees it) and its parsed header.
    fn read(self: Dir, name: []const u8) !struct { bytes: []u8, array: npy.Array } {
        const path = try std.fs.path.join(self.gpu.gpa, &.{ self.dir, name });
        defer self.gpu.gpa.free(path);
        const bytes = try std.Io.Dir.cwd().readFileAlloc(self.gpu.io, path, self.gpu.gpa, .limited(1 << 31));
        return .{ .bytes = bytes, .array = try npy.parse(bytes) };
    }

    /// The array's data on the device.
    fn upload(self: Dir, name: []const u8) !hip.DeviceBuffer {
        const f = try self.read(name);
        defer self.gpu.gpa.free(f.bytes);
        return hip.DeviceBuffer.fromHost(self.gpu.d, f.array.data);
    }
};

fn kindOf(act: []const u8) !hip.ops.Kind {
    if (std.mem.eql(u8, act, "float16")) return .f16;
    if (std.mem.eql(u8, act, "bfloat16")) return .bf16;
    return error.UnknownDtype;
}

/// Every affine case: the activation-dtype product and the fp32 product equal the engine's bytes.
pub fn affine(gpu: Gpu, dir_path: []const u8) !void {
    var dir = try Dir.open(gpu, dir_path);
    defer dir.deinit();
    const root = dir.parsed.value.object;
    const act = try kindOf(root.get("act_dtype").?.string);
    const cap = try gpu.ctx.capability();
    var lib = try hip.rocm.Library.open(gpu.d, hip.rocm.familyOf(cap) orelse return error.UnsupportedGpu);
    defer lib.close();
    var stream = try hip.Stream.init(gpu.d, true);
    defer stream.deinit();
    var arena = try hip.Arena.init(gpu.d, 64 << 20);
    defer arena.deinit();
    const o: hip.ops.Ops = .{ .lib = &lib, .stream = stream.handle, .arena = &arena };
    var cases: usize = 0;
    for (root.get("cases").?.array.items) |case| {
        const c = case.object;
        const files = c.get("files").?.object;
        const m: usize = @intCast(c.get("m").?.integer);
        var x = try dir.upload(files.get("x").?.string);
        defer x.free();
        var words = try dir.upload(files.get("words").?.string);
        defer words.free();
        var scale = try dir.upload(files.get("scale").?.string);
        defer scale.free();
        var bias = try dir.upload(files.get("bias").?.string);
        defer bias.free();
        const w: hip.ops.Affine = .{
            .words = words.ptr,
            .scale = scale.ptr,
            .bias = bias.ptr,
            .tables = .bf16,
            .n = @intCast(c.get("n").?.integer),
            .k = @intCast(c.get("k").?.integer),
            .bits = @intCast(c.get("bits").?.integer),
            .group = @intCast(c.get("group").?.integer),
        };
        inline for (.{ "out", "out_f32" }, .{ false, true }) |key, wide| {
            arena.reset();
            const y = try o.affine(.{ .ptr = x.ptr, .kind = act }, w, m, wide);
            try stream.synchronize();
            const want = try dir.read(files.get(key).?.string);
            defer gpu.gpa.free(want.bytes);
            const got = try gpu.gpa.alloc(u8, want.array.data.len);
            defer gpu.gpa.free(got);
            try gpu.d.check(gpu.d.api.hipMemcpyDtoH(got.ptr, y.ptr, got.len), "download");
            var name_buf: [96]u8 = undefined;
            const what = try std.fmt.bufPrint(&name_buf, "affine b{d} g{d} m{d} {s}", .{ w.bits, w.group, m, key });
            try check.sameBytes(what, got, want.array.data);
        }
        cases += 1;
    }
    check.pass("affine: {d} cases (bits 2-8, groups 32-128, 1-300 rows), activation and fp32 products bit-exact against the Python engine", .{cases});
}
