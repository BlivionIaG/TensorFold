//! One EXL3 window on a real device: a captured graph, a prefix KV cache, and an MTP accept.
//! Replay writes `expert_ids` and `kv_index` in place. It does not recapture.
//! Rank 1, when a second device exists, gets its own graph and does not share rank 0's key.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const window = @import("exl3_window.zig");

const Gpu = check.Gpu;

fn fp16(gpa: std.mem.Allocator, n: usize, bits: u16) ![]u8 {
    const bytes = try gpa.alloc(u8, n * 2);
    var i: usize = 0;
    while (i < n) : (i += 1) std.mem.writeInt(u16, bytes[i * 2 ..][0..2], bits, .little);
    return bytes;
}

const Body = struct {
    ex: *const cuda.exl3.Exl3,
    stream: cuda.Stream,
    x: u64,
    ids: u64,
    trellis: u64,
    suh: u64,
    svh: u64,
    y_expert: cuda.DeviceBuffer,
    gate: u64,
    up: u64,
    y_swiglu: cuda.DeviceBuffer,
    kv_src: u64,
    kv_out: cuda.DeviceBuffer,
};

fn play(b: Body) !void {
    try b.ex.experts(b.stream, .gate_up, b.x, b.ids, b.trellis, b.suh, b.svh, b.y_expert.ptr, 1, 128, 128, 2, 1, 2, 0);
    try b.ex.swiglu(b.stream, b.gate, b.up, b.y_swiglu.ptr, 1, 128);
    try b.kv_out.copyFrom(0, b.kv_src, b.kv_out.len, b.stream.handle);
}

fn zeros(gpu: Gpu, n: usize) !cuda.DeviceBuffer {
    var b = try cuda.DeviceBuffer.alloc(gpu.d, n);
    errdefer b.free();
    try b.fill8(0, null);
    return b;
}

fn captureRank(gpu: Gpu, ex: *const cuda.exl3.Exl3, rank: u8) !void {
    var stream = try cuda.Stream.init(gpu.d, true);
    defer stream.deinit();
    const span = try cuda.exl3.tileSpan(128, 128, 2);
    var trellis = try zeros(gpu, span * 2 * 2);
    defer trellis.free();
    var suh_b = try zeros(gpu, 128 * 2 * 2);
    defer suh_b.free();
    var svh_b = try zeros(gpu, 128 * 2 * 2);
    defer svh_b.free();
    const ones = try fp16(gpu.gpa, 128, 0x3c00);
    defer gpu.gpa.free(ones);
    var x = try cuda.DeviceBuffer.fromHost(gpu.d, ones);
    defer x.free();
    var ids = try cuda.DeviceBuffer.fromHost(gpu.d, &[_]u8{ 0, 0, 0, 0 });
    defer ids.free();
    var y_expert = try zeros(gpu, 256);
    defer y_expert.free();
    var gate = try cuda.DeviceBuffer.fromHost(gpu.d, ones);
    defer gate.free();
    var up = try cuda.DeviceBuffer.fromHost(gpu.d, ones);
    defer up.free();
    var y_swiglu = try zeros(gpu, 256);
    defer y_swiglu.free();
    var y_swiglu_eager = try zeros(gpu, 256);
    defer y_swiglu_eager.free();
    var kv_src = try cuda.DeviceBuffer.fromHost(gpu.d, &[_]u8{ 1, 2, 3, 4, 5, 6, 7, 8 });
    defer kv_src.free();
    var kv_out = try zeros(gpu, 8);
    defer kv_out.free();
    var kv_eager = try zeros(gpu, 8);
    defer kv_eager.free();

    const body = Body{
        .ex = ex,
        .stream = stream,
        .x = x.ptr,
        .ids = ids.ptr,
        .trellis = trellis.ptr,
        .suh = suh_b.ptr,
        .svh = svh_b.ptr,
        .y_expert = y_expert,
        .gate = gate.ptr,
        .up = up.ptr,
        .y_swiglu = y_swiglu,
        .kv_src = kv_src.ptr,
        .kv_out = kv_out,
    };

    try cuda.graph.beginCapture(stream, .thread_local);
    try play(body);
    var captured = try cuda.graph.endCapture(stream);
    defer captured.deinit();
    const nodes = try captured.nodeCount();
    try check.expect(nodes >= 3, "rank {d} graph has {d} nodes", .{ rank, nodes });
    const still = try check.download(gpu, y_swiglu);
    defer gpu.gpa.free(still);
    try check.expect(std.mem.allEqual(u8, still, 0), "rank {d} capture ran the window", .{rank});

    var exec = try captured.instantiate();
    defer exec.deinit();
    try exec.upload(stream);
    try exec.launchOn(stream);
    try stream.synchronize();

    var eager = body;
    eager.y_swiglu = y_swiglu_eager;
    eager.kv_out = kv_eager;
    try play(eager);
    try stream.synchronize();
    const got_y = try check.download(gpu, y_swiglu);
    defer gpu.gpa.free(got_y);
    const want_y = try check.download(gpu, y_swiglu_eager);
    defer gpu.gpa.free(want_y);
    try check.sameBytes("swiglu graph", got_y, want_y);
    const got_kv = try check.download(gpu, kv_out);
    defer gpu.gpa.free(got_kv);
    const want_kv = try check.download(gpu, kv_eager);
    defer gpu.gpa.free(want_kv);
    try check.sameBytes("kv graph", got_kv, want_kv);

    const zero_gate = try fp16(gpu.gpa, 128, 0);
    defer gpu.gpa.free(zero_gate);
    try gate.upload(0, zero_gate);
    try ids.upload(0, &[_]u8{ 1, 0, 0, 0 });
    try kv_src.upload(0, &[_]u8{ 9, 9, 9, 9, 9, 9, 9, 9 });
    try exec.launchOn(stream);
    try stream.synchronize();
    try y_swiglu_eager.fill8(0, null);
    try kv_eager.fill8(0, null);
    try play(eager);
    try stream.synchronize();
    const got_y2 = try check.download(gpu, y_swiglu);
    defer gpu.gpa.free(got_y2);
    const want_y2 = try check.download(gpu, y_swiglu_eager);
    defer gpu.gpa.free(want_y2);
    try check.sameBytes("swiglu replay", got_y2, want_y2);
    try check.expect(!std.mem.eql(u8, got_y, got_y2), "replay kept the first SwiGLU row", .{});
    const got_kv2 = try check.download(gpu, kv_out);
    defer gpu.gpa.free(got_kv2);
    try check.expect(std.mem.eql(u8, got_kv2, &[_]u8{ 9, 9, 9, 9, 9, 9, 9, 9 }), "kv_index replay did not publish the new slot", .{});
    check.pass("rank {d}: one graph, replayed ids and kv", .{rank});
}

pub fn run(gpu: Gpu) !void {
    var ex = try cuda.exl3.Exl3.load(gpu.d);
    defer ex.unload();
    try captureRank(gpu, &ex, 0);

    var cache = window.PrefixCache.init(gpu.gpa);
    defer cache.deinit();
    const prompt = [_]u32{ 10, 20, 30, 40 };
    try check.expect(cache.lookup(&prompt) == null, "empty prefix cache hit", .{});
    try cache.store(prompt[0..3], 3);
    const resumed = window.startAt(cache.lookup(&prompt));
    try check.expect(resumed == 2, "prefix resume {d}", .{resumed});
    const again = window.action(false, &.{.hit}, &.{.resident}, &.{.{ .rank = 0, .layer = 0, .window = 1 }}, .{ .rank = 0, .layer = 0, .window = 1 });
    try check.expect(again == .rewrite_ids, "a stored window recaptured", .{});
    check.pass("prefix of 3 tokens resumes at index 2 and the stored key rewrites", .{});

    const main_logits = try gpu.gpa.alloc(f32, window.logit_len);
    defer gpu.gpa.free(main_logits);
    const mtp_logits = try gpu.gpa.alloc(f32, window.logit_len);
    defer gpu.gpa.free(mtp_logits);
    @memset(main_logits, 0);
    @memset(mtp_logits, 0);
    main_logits[40] = 1;
    mtp_logits[40] = 1;
    const kept = try window.draftAndVerify(main_logits, mtp_logits);
    try check.expect(kept.accepted and kept.advance == 2 and kept.token == 40, "MTP dropped a matching draft", .{});
    mtp_logits[40] = 0;
    mtp_logits[9] = 1;
    const dropped = try window.draftAndVerify(main_logits, mtp_logits);
    try check.expect(!dropped.accepted and dropped.advance == 1 and dropped.token == 40, "MTP kept a mismatched draft", .{});
    check.pass("MTP accepted token 40 and rejected token 9", .{});

    const n = try gpu.d.deviceCount();
    if (n < 2) {
        check.pass("one device, rank 1 skipped", .{});
        return;
    }
    {
        // Rank 1's module and graph live in device 1's primary context. Unload
        // them before switching back: cuModuleUnload follows the current context.
        var ctx1 = try cuda.Context.init(gpu.d, 1);
        defer ctx1.deinit();
        var ex1 = try cuda.exl3.Exl3.load(gpu.d);
        defer ex1.unload();
        const other = window.action(false, &.{.hit}, &.{.resident}, &.{.{ .rank = 0, .layer = 0, .window = 1 }}, .{ .rank = 1, .layer = 0, .window = 1 });
        try check.expect(other == .capture, "rank 1 reused rank 0's graph key", .{});
        try captureRank(gpu, &ex1, 1);
    }
    try gpu.ctx.makeCurrent();
}
