//! Real LFM3-VL tower rows from an independent patch fixture, without Python in the native process.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const q = tf.lfm3;
const Dump = struct {
    io: std.Io,
    a: std.mem.Allocator,
    dir: []const u8,

    fn take(ptr: *anyopaque, name: []const u8, value: q.vision.Ref, rows: usize, width: usize) !void {
        const self: *Dump = @ptrCast(@alignCast(ptr));
        const path = try std.fmt.allocPrint(self.a, "{s}/{s}.bf16", .{ self.dir, name });
        defer self.a.free(path);
        try std.Io.Dir.cwd().writeFile(self.io, .{
            .sub_path = path,
            .data = (value.buf.contents() + value.off)[0 .. rows * width * 2],
        });
    }
};

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len != 6 and args.len != 7) return error.ExpectedModelPatchesHeightWidthOutput;
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const gh = try std.fmt.parseInt(u32, args[3], 10);
    const gw = try std.fmt.parseInt(u32, args[4], 10);
    if (gh == 0 or gw == 0 or gh % 2 != 0 or gw % 2 != 0 or @as(u64, gh) * gw > 16384)
        return error.BadPatchGrid;
    const tokens = gh * gw / 4;
    const bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, args[2], init.gpa, .limited(128 << 20));
    defer init.gpa.free(bytes);
    const input = try tf.npy.parse(bytes);
    if (!std.mem.eql(u8, input.descr, "<f4") or input.rank != 2 or input.shape[0] != gh * gw or
        input.shape[1] != 1536) return error.ExpectedPatchMatrix;
    const pixels = try init.gpa.alloc(f32, input.count());
    defer init.gpa.free(pixels);
    @memcpy(std.mem.sliceAsBytes(pixels), input.data);
    const model = try q.Model.load(init.gpa, init.io, args[1]);
    defer model.deinit();
    const vision = try q.vision.Vision.load(init.gpa, model.device, &model.checkpoint, tokens);
    defer vision.deinit();
    const output = try model.device.buffer(@as(usize, tokens) * q.config.hidden * 2, mtl.ResourceOptions.shared);
    defer output.deinit();
    if (args.len == 7) {
        try std.Io.Dir.cwd().createDirPath(init.io, args[6]);
        var dump: Dump = .{ .io = init.io, .a = init.gpa, .dir = args[6] };
        try vision.imageTrace(model.queue, pixels, gh, gw, .{ .buf = output }, .{ .ctx = &dump, .take = Dump.take });
    } else try vision.image(model.queue, pixels, gh, gw, .{ .buf = output });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = output.contents()[0..output.length()] });
    std.debug.print("native LFM3-VL tower: {d} image rows, bf16 outputs\n", .{tokens});
}
