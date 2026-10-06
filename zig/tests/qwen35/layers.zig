//! `layers <model dir> <fixture dir>`: the prompt and greedy decode steps of tools/zig/qwen_rocm_dump.py's `layers`
//! run on the HIP forward, every layer's residual, the final rows, the logits and the tokens compared byte for byte.

const std = @import("std");
const hip = @import("hip");
const npy = @import("npy");
const qwen35 = @import("qwen35");

const view = qwen35.view;
const Tensor = hip.ops.Tensor;

const Gpu = struct { d: *const hip.Driver, gpa: std.mem.Allocator, io: std.Io };

/// One fixture file's data (the caller frees `bytes`).
fn read(g: Gpu, dir: []const u8, stem: []const u8, act: view.Kind) !struct { bytes: []u8, data: []const u8 } {
    const name = try std.fmt.allocPrint(g.gpa, "{s}{s}", .{ stem, if (act == .bf16 and !std.mem.eql(u8, stem, "tokens") and !std.mem.eql(u8, stem, "sampled")) ".bf16.npy" else ".npy" });
    defer g.gpa.free(name);
    const path = try std.fs.path.join(g.gpa, &.{ dir, name });
    defer g.gpa.free(path);
    const bytes = try std.Io.Dir.cwd().readFileAlloc(g.io, path, g.gpa, .limited(1 << 31));
    return .{ .bytes = bytes, .data = (try npy.parse(bytes)).data };
}

fn download(g: Gpu, ptr: u64, n: usize) ![]u8 {
    const out = try g.gpa.alloc(u8, n);
    try g.d.check(g.d.api.hipMemcpyDtoH(out.ptr, ptr, n), "download");
    return out;
}

/// The first differing value of two activation rows' bytes, or null.
fn firstDiff(got: []const u8, want: []const u8) ?usize {
    const i = std.mem.indexOfDiff(u8, got, want) orelse return null;
    return i / 2;
}

const Check = struct {
    g: Gpu,
    want: []const u8, // (layers, rows, hidden) of the step
    hidden: usize,
    stream: hip.Stream,
    failed: ?usize = null,

    fn layer(ctx: *anyopaque, index: usize, x: Tensor, rows: usize) anyerror!void {
        const c: *Check = @ptrCast(@alignCast(ctx));
        try c.stream.synchronize();
        const n = rows * c.hidden * 2;
        const got = try download(c.g, x.ptr, n);
        defer c.g.gpa.free(got);
        if (c.failed == null) if (firstDiff(got, c.want[index * n ..][0..n])) |at| {
            std.debug.print("FAIL layer {d}: residual differs first at row {d} column {d}\n", .{ index, at / c.hidden, at % c.hidden });
            c.failed = index;
        };
    }
};

pub fn run(g: Gpu, model_dir: []const u8, dir: []const u8) !void {
    var ctx = try hip.Context.init(g.d, 0);
    defer ctx.deinit();
    const family = hip.rocm.familyOf(try ctx.capability()) orelse return error.UnsupportedGpu;
    var lib = try hip.rocm.Library.open(family);
    defer lib.close();
    const act: view.Kind = if (family == .rdna2) .f16 else .bf16;
    const dtype: qwen35.sample.Dtype = if (act == .f16) .f16 else .bf16;
    var model = try qwen35.Model.load(g.gpa, g.io, g.d, model_dir);
    defer model.deinit();
    const br = try qwen35.bridge.Bridge.init(g.gpa, g.d, &model, act);
    defer br.deinit();
    const m = &br.model;
    const s = m.spec;
    var stream = try hip.Stream.init(g.d, true);
    defer stream.deinit();
    var arena = try hip.Arena.init(g.d, 2 << 30);
    defer arena.deinit();
    const o: hip.ops.Ops = .{ .lib = &lib, .stream = stream.handle, .arena = &arena };

    const tokens_f = try read(g, dir, "tokens", act);
    defer g.gpa.free(tokens_f.bytes);
    const sampled_f = try read(g, dir, "sampled", act);
    defer g.gpa.free(sampled_f.bytes);
    const tokens = std.mem.bytesAsSlice(i64, tokens_f.data);
    const sampled = std.mem.bytesAsSlice(i64, sampled_f.data);
    const len = tokens.len;
    var caches = try qwen35.state.Caches.init(g.gpa, g.d, m, len + sampled.len + 1);
    defer caches.deinit(g.gpa);

    const ids = try g.gpa.alloc(i32, len);
    defer g.gpa.free(ids);
    for (ids, tokens) |*i, t| i.* = @intCast(t);
    var ids_dev = try hip.DeviceBuffer.fromHost(g.d, std.mem.sliceAsBytes(ids));
    defer ids_dev.free();

    const want_layers = try read(g, dir, "prefill.layers", act);
    defer g.gpa.free(want_layers.bytes);
    var check: Check = .{ .g = g, .want = want_layers.data, .hidden = s.hidden, .stream = stream };
    const trace: qwen35.forward.Trace = .{ .ctx = &check, .layer = Check.layer };
    const hidden = try qwen35.forward.span(o, m, &caches, ids_dev.ptr, len, 0, trace);
    try stream.synchronize();
    if (check.failed) |l| return fail("prefill", l);
    std.debug.print("PASS prefill: {d} rows, {d} layers' residuals equal\n", .{ len, s.n_layers });
    try compare(g, dir, "prefill.hidden", act, hidden.ptr, len * s.hidden * 2);
    var token = try logitsToken(g, o, m, dir, "prefill.logits", act, dtype, hidden, len - 1);
    try expectToken(token, sampled[0], "prefill");

    var snaps: [512]qwen35.window.Snapshot = undefined;
    for (0..sampled.len - 1) |step| {
        arena.reset();
        const pos = len + step;
        var one = [1]i32{@intCast(token)};
        var tok_dev = try hip.DeviceBuffer.fromHost(g.d, std.mem.asBytes(&one));
        defer tok_dev.free();
        var at = [1]i32{@intCast(pos)};
        var at_dev = try hip.DeviceBuffer.fromHost(g.d, std.mem.asBytes(&at));
        defer at_dev.free();
        var windows = [1]qwen35.window.Window{.{ .caches = &caches, .pos = pos, .rows = 1, .at32 = at_dev.ptr, .snaps = snaps[0..s.n_layers] }};
        var name_buf: [32]u8 = undefined;
        const stem = try std.fmt.bufPrint(&name_buf, "decode{d}", .{step});
        const want = try read(g, dir, try std.fmt.allocPrint(g.gpa, "{s}.layers", .{stem}), act);
        defer g.gpa.free(want.bytes);
        check = .{ .g = g, .want = want.data, .hidden = s.hidden, .stream = stream };
        const h = try qwen35.window.forward(o, m, &windows, tok_dev.ptr, trace);
        try qwen35.window.commit(o, m, windows[0], 1);
        try stream.synchronize();
        if (check.failed) |l| return fail(stem, l);
        try compare(g, dir, try std.fmt.allocPrint(g.gpa, "{s}.hidden", .{stem}), act, h.ptr, s.hidden * 2);
        token = try logitsToken(g, o, m, dir, try std.fmt.allocPrint(g.gpa, "{s}.logits", .{stem}), act, dtype, h, 0);
        try expectToken(token, sampled[step + 1], stem);
        std.debug.print("PASS {s}: residuals, final row, logits and token {d} equal\n", .{ stem, token });
    }
}

fn fail(what: []const u8, layer: usize) error{Mismatch} {
    std.debug.print("FAIL {s}: first differing layer {d}\n", .{ what, layer });
    return error.Mismatch;
}

fn compare(g: Gpu, dir: []const u8, stem: []const u8, act: view.Kind, ptr: u64, n: usize) !void {
    const want = try read(g, dir, stem, act);
    defer g.gpa.free(want.bytes);
    const got = try download(g, ptr, n);
    defer g.gpa.free(got);
    if (firstDiff(got, want.data)) |at| {
        std.debug.print("FAIL {s}: differs first at value {d}\n", .{ stem, at });
        return error.Mismatch;
    }
}

/// The row's logits in the activation dtype (one-row projection, as the engine) compared, then its greedy token.
fn logitsToken(g: Gpu, o: hip.ops.Ops, m: *const view.Model, dir: []const u8, stem: []const u8, act: view.Kind, dtype: qwen35.sample.Dtype, hidden: Tensor, row: usize) !u32 {
    const s = m.spec;
    const x: Tensor = .{ .ptr = hidden.ptr + row * s.hidden * 2, .kind = act };
    const logits = try o.affine(x, m.head, 1, false);
    try g.d.check(g.d.api.hipStreamSynchronize(o.stream), "sync");
    try compare(g, dir, stem, act, logits.ptr, m.head.n * 2);
    const bytes = try download(g, logits.ptr, m.head.n * 2);
    defer g.gpa.free(bytes);
    return qwen35.sample.argmax(std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(bytes))), dtype);
}

fn expectToken(got: u32, want: i64, what: []const u8) !void {
    if (got == want) return;
    std.debug.print("FAIL {s}: token {d}, the engine drew {d}\n", .{ what, got, want });
    return error.Mismatch;
}
