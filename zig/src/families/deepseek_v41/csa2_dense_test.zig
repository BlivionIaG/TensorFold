const std = @import("std");
const mtl = @import("metal");
const csa2 = @import("csa2.zig");

// The one-dispatch dense table must be bit-identical to the three per-section dispatches it replaces.
test "the one-dispatch dense table matches the per-section decode" {
    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const device = try mtl.Device.init();
    defer device.deinit();
    const queue = try device.queue();
    defer queue.deinit();
    var p = try csa2.Pipelines.load(device);
    defer p.deinit();
    if (!p.tensor_units) return error.SkipZigTest;

    const width: u32 = 512;
    const ring_rows: u32 = 128;
    const chunk_rows: u32 = 8;
    const comp_rows: u32 = 164;
    const ring_stride: u32 = 528;
    const chunk_stride: u32 = 528;
    const main_stride: u32 = 288;
    const window: u32 = 128;
    const total = ring_rows + chunk_rows + comp_rows;

    const ring = try device.buffer(ring_rows * ring_stride, mtl.ResourceOptions.shared);
    defer ring.deinit();
    const chunk = try device.buffer(chunk_rows * chunk_stride, mtl.ResourceOptions.shared);
    defer chunk.deinit();
    const main = try device.buffer(comp_rows * main_stride, mtl.ResourceOptions.shared);
    defer main.deinit();
    const a = try device.buffer(total * width * 2, mtl.ResourceOptions.shared);
    defer a.deinit();
    const b = try device.buffer(total * width * 2, mtl.ResourceOptions.shared);
    defer b.deinit();
    const picks = try device.buffer(comp_rows * 4, mtl.ResourceOptions.shared);
    defer picks.deinit();

    var state: u64 = 0x243f6a8885a308d3;
    const next = struct {
        fn draw(s: *u64) u32 {
            s.* = s.* *% 6364136223846793005 +% 1442695040888963407;
            return @intCast(s.* >> 32);
        }
    }.draw;
    for (ring.contents()[0 .. ring_rows * ring_stride]) |*x| x.* = @intCast(next(&state) & 0xff);
    for (chunk.contents()[0 .. chunk_rows * chunk_stride]) |*x| x.* = @intCast(next(&state) & 0xff);
    for (main.contents()[0 .. comp_rows * main_stride]) |*x| x.* = @intCast(next(&state) & 0xff);

    // The reference: one dispatch a section, exactly as the decode issued them before.
    const cb = queue.commandBuffer();
    const enc = cb.compute(.serial);
    enc.setPipeline(p.dense_kv);
    enc.setBuffer(ring, 0, 0);
    enc.setBuffer(a, 0, 1);
    enc.setValue(csa2.DenseKv{ .rows = ring_rows, .width = width, .stride = ring_stride, .first_row = 0, .source = 0 }, 2);
    enc.dispatchThreads(mtl.Size.of(ring_rows, width / 4, 1), mtl.Size.of(32, 4, 1));
    enc.setBuffer(chunk, 0, 0);
    enc.setValue(csa2.DenseKv{ .rows = chunk_rows, .width = width, .stride = chunk_stride, .first_row = window, .source = 0 }, 2);
    enc.dispatchThreads(mtl.Size.of(chunk_rows, width / 4, 1), mtl.Size.of(32, 4, 1));
    enc.setBuffer(main, 0, 0);
    enc.setValue(csa2.DenseKv{ .rows = comp_rows, .width = width, .stride = main_stride, .first_row = window + chunk_rows, .source = 1 }, 2);
    enc.dispatchThreads(mtl.Size.of(comp_rows, width / 4, 1), mtl.Size.of(32, 4, 1));
    try submit(cb, enc);

    // The landed form: all three sections in one dispatch.
    const cb2 = queue.commandBuffer();
    const enc2 = cb2.compute(.serial);
    enc2.setPipeline(p.dense_all);
    enc2.setBuffer(ring, 0, 0);
    enc2.setBuffer(chunk, 0, 1);
    enc2.setBuffer(main, 0, 2);
    enc2.setBuffer(b, 0, 3);
    enc2.setValue(csa2.DenseKvAll{ .ring_rows = ring_rows, .chunk_rows = chunk_rows, .comp_rows = comp_rows, .width = width, .ring_stride = ring_stride, .chunk_stride = chunk_stride, .main_stride = main_stride, .window = window, .comp_first = window + chunk_rows, .picked = 0 }, 4);
    enc2.setBuffer(picks, 0, 5);
    enc2.dispatchThreads(mtl.Size.of(total, width / 4, 1), mtl.Size.of(32, 4, 1));
    try submit(cb2, enc2);

    try std.testing.expectEqualSlices(u16, a.slice(u16, total * width), b.slice(u16, total * width));
}
