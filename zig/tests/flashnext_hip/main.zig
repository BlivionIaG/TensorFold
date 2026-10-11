//! Flash Next model checks on HIP: `tf-flashnext-hip-test generate <dir> [group flags] [--text T] [--chat] [--max N]`.
const std = @import("std");
const hip = @import("hip");
const fx = @import("flashnext");
const tokenizer = @import("tokenizer");

/// No per-thread alternate signal stack: its thread-local leaves RCCL's and HIP's threads too little stack to start.
pub const std_options: std.Options = .{ .signal_stack_size = null };

const usage = "usage: tf-flashnext-hip-test generate <model dir> [--tp N --rank R --master H --port P --policy P] [--text T] [--chat] [--max N] [--capacity N] [--repeat N]\n";

const Args = struct {
    dir: []const u8 = "",
    world: usize = 1,
    rank: usize = 0,
    master: []const u8 = "127.0.0.1",
    port: u16 = 29561,
    policy: []const u8 = "",
    text: []const u8 = "The capital of France is",
    chat: bool = false,
    max: usize = 32,
    capacity: usize = 8192,
    repeat: usize = 1,
};

fn parse(args: []const [:0]const u8) !Args {
    var a: Args = .{ .dir = args[0] };
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const name = args[i];
        if (std.mem.eql(u8, name, "--chat")) {
            a.chat = true;
            continue;
        }
        i += 1;
        if (i >= args.len) return error.MissingArgument;
        const v = args[i];
        if (std.mem.eql(u8, name, "--tp")) a.world = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, name, "--rank")) a.rank = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, name, "--master")) a.master = v else if (std.mem.eql(u8, name, "--port")) a.port = try std.fmt.parseInt(u16, v, 10) else if (std.mem.eql(u8, name, "--policy")) a.policy = v else if (std.mem.eql(u8, name, "--text")) a.text = v else if (std.mem.eql(u8, name, "--max")) a.max = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, name, "--capacity")) a.capacity = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, name, "--repeat")) a.repeat = try std.fmt.parseInt(usize, v, 10) else return error.UnknownOption;
    }
    return a;
}

fn nowNs(io: std.Io) i96 {
    return std.Io.Timestamp.now(io, .awake).nanoseconds;
}

/// Every rank runs the same prompt and greedy steps (the collectives pair up); rank 0 prints the text and the timings.
fn generate(gpa: std.mem.Allocator, io: std.Io, a: Args) !void {
    var driver = try hip.Runtime.open();
    const device: c_int = @intCast(a.rank);
    const caps = try hip.Device.kernelCaps(&driver, device);
    driver.close();
    var notes: hip.Policy.Notes = .{};
    var policy = try hip.Policy.resolve(a.policy, .current, &notes);
    const group: ?hip.collective.Group = if (a.world > 1) try hip.joinGroup(gpa, io, @intCast(a.rank), @intCast(a.world), a.master, a.port, caps, &policy) else null;
    defer if (group) |g| g.close(gpa);
    const t0 = nowNs(io);
    const e = try fx.engine.Engine.open(gpa, io, a.dir, .{ .device = device, .rank = a.rank, .world = a.world, .group = group, .policy = policy, .capacity = a.capacity });
    defer e.deinit();
    const loud = a.rank == 0;
    if (loud) std.debug.print("loaded in {d:.1} s, {d:.2} GiB on the device\n", .{ @as(f64, @floatFromInt(nowNs(io) - t0)) / 1e9, @as(f64, @floatFromInt(hip.usage(false).device)) / (1 << 30) });
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const tk_path = try std.fs.path.join(arena.allocator(), &.{ a.dir, "tokenizer.json" });
    var tok = try tokenizer.loadTokenizer(io, arena.allocator(), tk_path);
    const text = if (a.chat) try std.fmt.allocPrint(arena.allocator(), "<|im_start|>user\n{s}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", .{a.text}) else a.text;
    const prompt = try tok.encode(arena.allocator(), text);
    for (0..a.repeat) |_| {
        var c = try e.caches(a.capacity);
        defer c.deinit(gpa);
        const m = e.model_();
        var at: usize = 0;
        var start: usize = 0;
        const p0 = nowNs(io);
        var out: fx.forward.Out = undefined;
        while (at < prompt.len) {
            const n = @min(e.o.chunk, prompt.len - at);
            out = try e.prompt(&c, prompt[at..][0..n]);
            start = at;
            at += n;
        }
        const last = fx.forward.at(out.mixed, (prompt.len - 1 - start) * m.dims.hidden);
        var o = e.ops(&e.arena);
        var next: [1]u32 = undefined;
        try e.drawer.draw(o, e.stream, try e.logits(o, last, 1), &.{.{ .sampling = null, .position = prompt.len }}, &next, false);
        const p1 = nowNs(io);
        var reply: std.ArrayList(u32) = .empty;
        try reply.append(arena.allocator(), next[0]);
        for (1..a.max) |_| {
            const step = [1]u32{next[0]};
            const w = try e.run(&c, &step, c.len, true);
            o = e.ops(&e.arena);
            try e.drawer.draw(o, e.stream, try e.logits(o, w.mixed, 1), &.{.{ .sampling = null, .position = c.len + 1 }}, &next, false);
            try e.keepWindow(&c, &step, 1);
            try reply.append(arena.allocator(), next[0]);
        }
        try e.stream.synchronize();
        const p2 = nowNs(io);
        if (!loud) continue;
        const words = try tok.decode(arena.allocator(), reply.items, false);
        const pre = @as(f64, @floatFromInt(p1 - p0)) / 1e9;
        const dec = @as(f64, @floatFromInt(p2 - p1)) / 1e9;
        std.debug.print("prompt {d} tokens in {d:.3} s ({d:.1} tok/s); {d} tokens in {d:.3} s ({d:.2} tok/s)\n", .{ prompt.len, pre, @as(f64, @floatFromInt(prompt.len)) / pre, a.max - 1, dec, @as(f64, @floatFromInt(a.max - 1)) / dec });
        std.debug.print("ids {any}\n---\n{s}{s}\n---\n", .{ reply.items, text, words });
    }
}

pub fn main(init: std.process.Init) !u8 {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3 or !std.mem.eql(u8, args[1], "generate")) {
        std.debug.print(usage, .{});
        return 2;
    }
    const a = parse(args[2..]) catch {
        std.debug.print(usage, .{});
        return 2;
    };
    generate(init.gpa, init.io, a) catch |err| {
        std.debug.print("FAIL {t}\n", .{err});
        if (@errorReturnTrace()) |t| std.debug.dumpStackTrace(t);
        return 1;
    };
    return 0;
}
