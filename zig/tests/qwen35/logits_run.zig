//! `logits <model dir> <ids.npy> <out.npy> [--f32] [--decode]`: every row's logits for the ids (int32 or int64), from
//! one cold prefill or (--decode) a token at a time through the decode rounds, in the activation dtype the engine
//! draws from (bf16 as `<u2`) or in fp32, for tools/truth/score.py to measure against the high-precision reference.

const std = @import("std");
const hip = @import("hip");
const npy = @import("npy");
const qwen35 = @import("qwen35");

/// Rows a head projection covers at once.
const chunk = 32;

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 3) return error.MissingArgument;
    var wide = false;
    var decode = false;
    for (args[3..]) |a| {
        if (std.mem.eql(u8, a, "--f32")) wide = true else if (std.mem.eql(u8, a, "--decode")) decode = true else return error.UnknownOption;
    }
    const file = try std.Io.Dir.cwd().readFileAlloc(io, args[1], gpa, .limited(1 << 28));
    defer gpa.free(file);
    const a = try npy.parse(file);
    const ids = try gpa.alloc(u32, a.count());
    defer gpa.free(ids);
    for (ids, 0..) |*t, i| t.* = if (std.mem.eql(u8, a.descr, "<i8"))
        @intCast(std.mem.readInt(i64, a.data[i * 8 ..][0..8], .little))
    else
        @intCast(std.mem.readInt(i32, a.data[i * 4 ..][0..4], .little));

    const e = try qwen35.engine.Engine.open(gpa, io, args[0], .{ .capacity = ids.len + 64, .batch_rows = 32 });
    defer e.deinit();
    const m = e.model();
    var caches = try e.newCaches(ids.len + 8);
    defer caches.deinit(gpa);
    e.prompts.reset();
    const o: hip.ops.Ops = .{ .lib = &e.lib, .stream = e.stream.handle, .arena = &e.prompts };
    var ids_dev = try hip.DeviceBuffer.fromHost(&e.driver, std.mem.sliceAsBytes(ids));
    defer ids_dev.free();
    const hidden = try qwen35.forward.span(o, m, &caches, ids_dev.ptr, if (decode) 1 else ids.len, 0, null);

    const vocab = m.spec.vocab;
    const size: usize = if (wide) 4 else 2;
    var header_buf: [128]u8 = undefined;
    const descr = if (wide) "<f4" else if (m.act == .f16) "<f2" else "<u2";
    const dict = try std.fmt.bufPrint(&header_buf, "{{'descr': '{s}', 'fortran_order': False, 'shape': ({d}, {d}), }}", .{ descr, ids.len, vocab });
    const pad = 64 - (10 + dict.len + 1) % 64;
    const out = try gpa.alloc(u8, 10 + dict.len + pad + 1 + ids.len * vocab * size);
    defer gpa.free(out);
    @memcpy(out[0..8], "\x93NUMPY\x01\x00");
    std.mem.writeInt(u16, out[8..10], @intCast(dict.len + pad + 1), .little);
    @memcpy(out[10..][0..dict.len], dict);
    @memset(out[10 + dict.len ..][0..pad], ' ');
    out[10 + dict.len + pad] = '\n';
    const body = out[10 + dict.len + pad + 1 ..];

    if (decode) {
        // row 0 from its one-row prefill, then each token a one-row round at its slot, kept before the next
        const first = try o.affine(hidden, m.head, 1, wide);
        try e.stream.synchronize();
        try e.driver.check(e.driver.api.hipMemcpyDtoH(body.ptr, first.ptr, vocab * size), "download");
        var wins: [1]qwen35.window.Window = undefined;
        const snaps = try gpa.alloc(qwen35.window.Snapshot, m.spec.n_layers);
        defer gpa.free(snaps);
        for (1..ids.len) |i| {
            const rows = [1]qwen35.engine.Engine.Rows{.{ .caches = &caches, .pos = i, .tokens = ids[i..][0..1] }};
            const reqs = [1]qwen35.draw.Request{.{ .sampling = null, .position = i + 1 }};
            var drawn: [1]u32 = undefined;
            const r = try e.verify(&rows, &wins, snaps, &reqs, &drawn);
            const round: hip.ops.Ops = .{ .lib = &e.lib, .stream = e.stream.handle, .arena = &e.rounds };
            const logits = try round.affine(r.hidden, m.head, 1, wide);
            try e.stream.synchronize();
            try e.driver.check(e.driver.api.hipMemcpyDtoH(body[i * vocab * size ..].ptr, logits.ptr, vocab * size), "download");
            try e.keep(wins[0], 1);
        }
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out });
        std.debug.print("wrote {d} decoded rows x {d} logits to {s}\n", .{ ids.len, vocab, args[2] });
        return;
    }
    var row: usize = 0;
    while (row < ids.len) : (row += chunk) {
        const rows = @min(chunk, ids.len - row);
        const at = e.prompts.mark();
        const x = qwen35.forward.at(hidden, row * m.spec.hidden);
        const logits = try o.affine(x, m.head, rows, wide);
        try e.stream.synchronize();
        const n = rows * vocab * size;
        try e.driver.check(e.driver.api.hipMemcpyDtoH(body[row * vocab * size ..].ptr, logits.ptr, n), "download");
        e.prompts.release(at);
    }
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = args[2], .data = out });
    std.debug.print("wrote {d} rows x {d} logits to {s}\n", .{ ids.len, vocab, args[2] });
}
