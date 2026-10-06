//! Qwen3.5 / 3.6 loader checks: `check <model dir> [first layers]` uploads the model and reads every buffer back (GPU);
//! `digest <model dir>` prints a SHA-256 per host tensor, to compare with the Python loader's.

const std = @import("std");
const hip = @import("hip");
const qwen35 = @import("qwen35");

const Buf = qwen35.weights.Buf;
const Tensor = qwen35.table.Tensor;

const usage = "usage: tf-qwen35-test check <model dir> [first layers] | digest <model dir> | layers <model dir> <fixture dir>\n";

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) {
        std.debug.print(usage, .{});
        return 2;
    }
    if (std.mem.eql(u8, args[1], "digest")) {
        digest(init.gpa, init.io, args[2]) catch |e| {
            std.debug.print("FAIL {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (std.mem.eql(u8, args[1], "layers") and args.len > 3) {
        var d = try hip.Driver.open();
        defer d.close();
        @import("layers.zig").run(.{ .d = &d, .gpa = init.gpa, .io = init.io }, args[2], args[3]) catch |e| {
            std.debug.print("FAIL {t}\n", .{e});
            return 1;
        };
        return 0;
    }
    if (!std.mem.eql(u8, args[1], "check")) {
        std.debug.print(usage, .{});
        return 2;
    }
    var driver = try hip.Driver.open();
    defer driver.close();
    var ctx = try hip.Context.init(&driver, 0);
    defer ctx.deinit();
    const limit = if (args.len > 3) try std.fmt.parseInt(usize, args[3], 10) else std.math.maxInt(usize);
    run(init.gpa, init.io, &driver, args[2], limit) catch |e| {
        std.debug.print("FAIL {t}\n", .{e});
        return 1;
    };
    return 0;
}

/// One line per host tensor: its path, dtype, shape and the SHA-256 of its bytes.
fn line(path: []const u8, t: Tensor) void {
    var sum: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(t.bytes, &sum, .{});
    std.debug.print("{s} {t} {any} {s}\n", .{ path, t.dtype, t.shape[0..t.rank], std.fmt.bytesToHex(sum, .lower) });
}

fn emit(gpa: std.mem.Allocator, path: []const u8, v: anytype) !void {
    const T = @TypeOf(v);
    switch (@typeInfo(T)) {
        .@"struct" => if (T == Tensor) line(path, v) else inline for (@typeInfo(T).@"struct".field_names) |name| {
            if (comptime std.mem.eql(u8, name, "arena") or std.mem.eql(u8, name, "owned") or std.mem.eql(u8, name, "io")) continue;
            const sub = try std.fmt.allocPrint(gpa, "{s}.{s}", .{ path, name });
            defer gpa.free(sub);
            try emit(gpa, sub, @field(v, name));
        },
        .@"union" => switch (v) {
            inline else => |payload, tag| {
                const sub = try std.fmt.allocPrint(gpa, "{s}.{s}", .{ path, @tagName(tag) });
                defer gpa.free(sub);
                try emit(gpa, sub, payload);
            },
        },
        .optional => if (v) |x| try emit(gpa, path, x),
        .int, .bool => std.debug.print("{s} {any}\n", .{ path, v }),
        else => {},
    }
}

fn digest(gpa: std.mem.Allocator, io: std.Io, dir: []const u8) !void {
    var ck = try qwen35.Checkpoint.open(gpa, io, dir);
    defer ck.close();
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    try emit(gpa, "embed", qwen35.host.Projection{ .affine = try ck.embed(scratch.allocator()) });
    try emit(gpa, "final_norm", try ck.finalNorm(scratch.allocator()));
    try emit(gpa, "head", try ck.head(scratch.allocator()));
    for (0..ck.spec().n_layers) |i| {
        var layer = try ck.layer(i);
        defer layer.deinit();
        const path = try std.fmt.allocPrint(gpa, "L{d}", .{i});
        defer gpa.free(path);
        try emit(gpa, path, layer.body);
    }
    var mtp = try ck.mtp();
    defer if (mtp) |*m| m.deinit();
    try emit(gpa, "mtp", mtp);
}

const Check = struct {
    gpa: std.mem.Allocator,
    driver: *const hip.Driver,
    buffers: usize = 0,
    bytes: usize = 0,

    /// Reads `b` back and compares dtype, shape and bytes with the host tensor.
    fn buf(c: *Check, b: Buf, t: Tensor) !void {
        if (b.dtype != t.dtype or b.rank != t.rank or !std.mem.eql(u8, std.mem.sliceAsBytes(&b.shape), std.mem.sliceAsBytes(&t.shape)) or b.len != t.bytes.len) return error.ShapeMismatch;
        const got = try c.gpa.alloc(u8, b.len);
        defer c.gpa.free(got);
        const device: hip.DeviceBuffer = .{ .d = c.driver, .ptr = b.ptr, .len = b.len };
        try device.download(0, got);
        if (!std.mem.eql(u8, got, t.bytes)) return error.BytesDiffer;
        c.buffers += 1;
        c.bytes += b.len;
    }

    /// Walks a device struct and its host twin (same field and tag names) and checks every buffer.
    fn walk(c: *Check, dev: anytype, h: anytype) !void {
        const T = @TypeOf(dev);
        switch (@typeInfo(T)) {
            .@"struct" => if (T == Buf) try c.buf(dev, h) else inline for (@typeInfo(T).@"struct".field_names) |name| try c.walk(@field(dev, name), if (@hasField(@TypeOf(h), name)) @field(h, name) else {}),
            .@"union" => switch (dev) {
                inline else => |payload, tag| try c.walk(payload, @field(h, @tagName(tag))),
            },
            .optional => if (dev) |d| try c.walk(d, h.?) else if (h != null) return error.ShapeMismatch,
            else => {},
        }
    }
};

fn run(gpa: std.mem.Allocator, io: std.Io, driver: *const hip.Driver, dir: []const u8, limit: usize) !void {
    var ck = try qwen35.Checkpoint.open(gpa, io, dir);
    defer ck.close();
    var model = try qwen35.Model.fromCheckpoint(gpa, driver, &ck, limit);
    defer model.deinit();
    var c: Check = .{ .gpa = gpa, .driver = driver };
    var scratch: std.heap.ArenaAllocator = .init(gpa);
    defer scratch.deinit();
    try c.walk(model.embed, (try ck.embed(scratch.allocator())));
    try c.walk(model.final_norm, try ck.finalNorm(scratch.allocator()));
    try c.walk(model.head, try ck.head(scratch.allocator()));
    for (model.layers, 0..) |layer, i| {
        var host_layer = try ck.layer(i);
        defer host_layer.deinit();
        try c.walk(layer, host_layer.body);
    }
    if (model.layers.len == ck.spec().n_layers) {
        var host_mtp = try ck.mtp();
        defer if (host_mtp) |*m| m.deinit();
        try c.walk(model.mtp, host_mtp);
    }
    std.debug.print("PASS {d} layers, {d} buffers, {d} MiB read back equal to the host index\n", .{ model.layers.len, c.buffers, c.bytes >> 20 });
}
