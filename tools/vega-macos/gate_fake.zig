//! gate_fake: a stand-in for `tensorfold-native serve` and `tf-qwen35-test logits`, so tf-vega-gate and tf-moe-gate run offline (zig build test-vega-tools).
//!
//!     gate_fake serve MODEL [--port P] ...        listens on 127.0.0.1:P, answers /v1/chat/completions, logs `done <id> ... sha=` like the real server
//!     gate_fake logits MODEL IDS OUT [--decode]   writes the logits matrix of `logits()` below to OUT
//!
//! The token ids of a reply are `tokens()` (a function of thinking, length and FAKE_VARIANT), the sha in the log is the sha256 of their comma-joined text,
//! exactly what the gates compute from a reference, so a gate passes unless FAKE_VARIANT (or FAKE_FLIP for the logits) differs from the reference's.
//! FAKE_EXIT=1: print a line and exit at once (a server that dies on start).
const std = @import("std");
const npy = @import("npy.zig");
const Sha256 = std.crypto.hash.sha2.Sha256;

pub const logit_rows = 4;
pub const logit_cols = 16;

/// The token ids the fake server "generates".
pub fn tokens(a: std.mem.Allocator, thinking: bool, n: usize, variant: u32) ![]u32 {
    const out = try a.alloc(u32, n);
    for (out, 0..) |*t, i| t.* = @intCast((i * 31 + (if (thinking) @as(usize, 5) else 3) + variant) % 1000);
    return out;
}

pub fn shaOf(a: std.mem.Allocator, ids: []const u32) ![]u8 {
    var joined: std.Io.Writer.Allocating = .init(a);
    defer joined.deinit();
    for (ids, 0..) |t, i| {
        if (i > 0) try joined.writer.writeByte(',');
        try joined.writer.print("{d}", .{t});
    }
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(joined.written(), &digest, .{});
    const hex = std.fmt.bytesToHex(digest, .lower);
    return a.dupe(u8, hex[0..12]);
}

/// The text of a reply: a function of the prompt and the variant; it differs between variants only after the prompt, at code point 10 + the prompt's + 2.
pub fn textOf(a: std.mem.Allocator, prompt: []const u8, variant: u32) ![]u8 {
    return std.fmt.allocPrint(a, "answer to {s} v{d} \u{e9}\u{20ac}", .{ prompt, variant });
}

/// The logits `tf-qwen35-test logits` "computes": one clear maximum a row; `flip` moves it.
pub fn logit(row: usize, col: usize, flip: bool) f32 {
    const peak = (row * 3 + 1 + @intFromBool(flip)) % logit_cols;
    if (col == peak) return 100;
    return @floatFromInt((col * 7 + row) % 10);
}

pub fn logitsFile(a: std.mem.Allocator, flip: bool) ![]u8 {
    var vals: [logit_rows * logit_cols]f32 = undefined;
    for (0..logit_rows) |r| for (0..logit_cols) |c| {
        vals[r * logit_cols + c] = logit(r, c, flip);
    };
    return npy.encode(a, "<f4", &.{ logit_rows, logit_cols }, std.mem.sliceAsBytes(&vals));
}

fn envInt(env: *const std.process.Environ.Map, key: []const u8) u32 {
    return std.fmt.parseInt(u32, env.get(key) orelse return 0, 10) catch 0;
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    const a = init.gpa;
    const argv = try init.minimal.args.toSlice(init.arena.allocator());
    var out_buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    const out = &stdout.interface;
    if (argv.len < 2) return 2;

    if (std.mem.eql(u8, argv[1], "logits")) {
        if (argv.len < 5) return 2;
        const bytes = try logitsFile(a, envInt(init.environ_map, "FAKE_FLIP") != 0);
        defer a.free(bytes);
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = argv[4], .data = bytes });
        try out.writeAll("fake logits written\n");
        try out.flush();
        return if (envInt(init.environ_map, "FAKE_LOGITS_FAIL") != 0) 3 else 0;
    }
    if (!std.mem.eql(u8, argv[1], "serve")) return 2;

    if (envInt(init.environ_map, "FAKE_EXIT") != 0) {
        try out.writeAll("boom: fake server dies on start\n");
        try out.flush();
        return 1;
    }
    var port: u16 = 18099;
    var i: usize = 2;
    while (i + 1 < argv.len) : (i += 1) if (std.mem.eql(u8, argv[i], "--port")) {
        port = try std.fmt.parseInt(u16, argv[i + 1], 10);
    };
    const variant = envInt(init.environ_map, "FAKE_VARIANT");

    const address = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    var server = try address.listen(io, .{ .reuse_address = true });
    defer server.deinit(io);
    try out.print("[tensorfold] serving fake at http://127.0.0.1:{d}/v1 (sampling: greedy; drafts: off; loaded in 0.1s)\n", .{port});
    try out.flush();

    var counter: u32 = 0;
    while (true) {
        const stream = try server.accept(io);
        defer stream.close(io);
        var rbuf: [8192]u8 = undefined;
        var wbuf: [8192]u8 = undefined;
        var reader = stream.reader(io, &rbuf);
        var writer = stream.writer(io, &wbuf);
        var http = std.http.Server.init(&reader.interface, &writer.interface);
        var request = http.receiveHead() catch continue;
        var arena_state = std.heap.ArenaAllocator.init(a);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        const len: usize = @intCast(request.head.content_length orelse 0);
        var body_buf: [4096]u8 = undefined;
        const body = try request.readerExpectNone(&body_buf).readAlloc(arena, len);
        const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{});
        const n: usize = @intCast(doc.object.get("max_tokens").?.integer);
        const prompt = doc.object.get("messages").?.array.items[0].object.get("content").?.string;
        const thinking = if (doc.object.get("chat_template_kwargs")) |kw| (if (kw.object.get("enable_thinking")) |t| t.bool else false) else false;

        counter += 1;
        const ids = try tokens(arena, thinking, n, variant);
        const sha = try shaOf(arena, ids);
        const id = try std.fmt.allocPrint(arena, "chatcmpl-{x:0>8}", .{counter});
        try out.print("[tensorfold] done {s} prompt={d} cached=0 thinking={s} effort=none tokens={d} sha={s} finish=stop rounds={d} accepted=0/0\n", .{ id, prompt.len, if (thinking) "True" else "False", n, sha, n });
        try out.flush();
        var reply: std.Io.Writer.Allocating = .init(arena);
        try reply.writer.print("{{\"id\": \"{s}\", \"choices\": [{{\"message\": {{\"content\": ", .{id});
        try std.json.Stringify.value(try textOf(arena, prompt, variant), .{}, &reply.writer);
        try reply.writer.print("}}}}], \"usage\": {{\"completion_tokens\": {d}}}, \"tensorfold\": {{\"engine\": \"fake\", \"tokens_per_second\": {d}.5, \"seconds\": 1.25, \"drafts\": false, \"widths\": []}}}}", .{ n, 80 + counter });
        try request.respond(reply.written(), .{ .extra_headers = &.{.{ .name = "content-type", .value = "application/json" }}, .keep_alive = false });
    }
}
