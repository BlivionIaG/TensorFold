//! The host stage of a split pipeline in another process. The runtime that owns an external card can be a program that
//! cannot also hold the host's GPU engine (one build links a source file into one module), and the card allows one
//! client: so the pipeline runs where the card is, and its host stage is served over a unix socket by the process that
//! owns the Mac's GPU. Both map the window (the card's pinned host memory, exported as a shared-memory descriptor and
//! sent with `SCM_RIGHTS`): rows and tokens never cross the socket, only the calls that start a step and wait for it.
const std = @import("std");
const pipeline = @import("pipeline.zig");

const c = std.c;

/// The most sequences one server hosts.
pub const max_streams = 8;

pub const Error = error{ Socket, PeerClosed, BadMessage, PeerFailed, Timeout, MapFailed };

const magic: u32 = 0x5350_4c54;

const Op = enum(u8) { hello, step, finish, reset, bye };

/// One request: `a`..`d` carry the operation's words (hello: window offset in the object, window length, object size;
/// step: input offset and length, output offset; the output length, the rows and the sequence ride in `len`, `rows` and
/// `seq`). A step's input and output are in the window: the pipeline stages the tokens it hands the stage there.
/// `stream` names the sequence's stage when the server hosts several.
const Request = extern struct {
    magic: u32 = magic,
    op: u8,
    level: u8 = 0,
    stream: u16 = 0,
    rows: u32 = 1,
    seq: u32 = 0,
    len: u32 = 0,
    a: u64 = 0,
    b: u64 = 0,
    d: u64 = 0,
};

const Reply = extern struct { status: i32, n: u32 };

fn readAll(fd: c_int, buf: []u8) Error!void {
    var at: usize = 0;
    while (at < buf.len) {
        const n = c.read(fd, buf[at..].ptr, buf.len - at);
        if (n == 0) return error.PeerClosed;
        if (n < 0) {
            if (std.c._errno().* == @backingInt(c.E.INTR)) continue;
            return error.Socket;
        }
        at += @intCast(n);
    }
}

fn writeAll(fd: c_int, buf: []const u8) Error!void {
    var at: usize = 0;
    while (at < buf.len) {
        const n = c.write(fd, buf[at..].ptr, buf.len - at);
        if (n < 0) {
            if (std.c._errno().* == @backingInt(c.E.INTR)) continue;
            return error.Socket;
        }
        at += @intCast(n);
    }
}

/// A request with one descriptor attached.
fn sendWithFd(sock: c_int, req: *const Request, fd: c_int) Error!void {
    const Control = extern struct { hdr: c.cmsghdr, fd: c_int };
    var ctl: Control = .{ .hdr = .{ .len = @sizeOf(Control), .level = c.SOL.SOCKET, .type = c.SCM.RIGHTS }, .fd = fd };
    var iov = [1]c.iovec_const{.{ .base = std.mem.asBytes(req).ptr, .len = @sizeOf(Request) }};
    const msg: c.msghdr_const = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &ctl, .controllen = @sizeOf(Control), .flags = 0 };
    if (c.sendmsg(sock, &msg, 0) != @sizeOf(Request)) return error.Socket;
}

/// A request and the descriptor sent with it, if any (-1).
fn recvWithFd(sock: c_int, req: *Request) Error!c_int {
    const Control = extern struct { hdr: c.cmsghdr, fd: c_int };
    var ctl: Control = undefined;
    var iov = [1]c.iovec{.{ .base = std.mem.asBytes(req).ptr, .len = @sizeOf(Request) }};
    var msg: c.msghdr = .{ .name = null, .namelen = 0, .iov = &iov, .iovlen = 1, .control = &ctl, .controllen = @sizeOf(Control), .flags = 0 };
    const n = c.recvmsg(sock, &msg, 0);
    if (n == 0) return error.PeerClosed;
    if (n != @sizeOf(Request)) return error.Socket;
    if (msg.controllen >= @sizeOf(Control) and ctl.hdr.level == c.SOL.SOCKET and ctl.hdr.type == c.SCM.RIGHTS) return ctl.fd;
    return -1;
}

fn unixAddr(path: []const u8) Error!c.sockaddr.un {
    var addr: c.sockaddr.un = .{ .path = @splat(0) };
    if (path.len >= addr.path.len) return error.Socket;
    @memcpy(addr.path[0..path.len], path);
    return addr;
}

/// The pipeline's side: it owns the window (`bytes` is the card's pinned memory, mapped here), and its host stage lives
/// in the process that `serve`s.
pub const Client = struct {
    fd: c_int,
    window: []u8,
    lanes: [max_streams]Lane = undefined,

    const Lane = struct { client: *Client, stream: u16 };

    /// Connects to the server at `path` (it may still be starting: retried until `timeout_ns`) and hands it the window
    /// as `shm_fd`: the object's descriptor, where the window starts in it, and the object's size.
    pub fn connect(path: []const u8, window: []u8, shm_fd: c_int, offset: u64, object_size: u64, timeout_ns: u64) Error!Client {
        const addr = try unixAddr(path);
        const t0 = nowNs();
        const fd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
        if (fd < 0) return error.Socket;
        errdefer _ = c.close(fd);
        while (c.connect(fd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0) {
            if (nowNs() - t0 > timeout_ns) return error.Timeout;
            sleepNs(20_000_000);
        }
        const hello: Request = .{ .op = @backingInt(Op.hello), .a = offset, .b = window.len, .d = object_size };
        try sendWithFd(fd, &hello, shm_fd);
        var cl: Client = .{ .fd = fd, .window = window };
        _ = try cl.reply();
        return cl;
    }

    /// A socket pair's end as the client (tests, and a server forked in-process).
    pub fn over(fd: c_int, window: []u8) Client {
        return .{ .fd = fd, .window = window };
    }

    fn reply(cl: *Client) Error!Reply {
        var r: Reply = undefined;
        try readAll(cl.fd, std.mem.asBytes(&r));
        if (r.status != 0) return error.PeerFailed;
        return r;
    }

    fn call(cl: *Client, req: Request) Error!Reply {
        try writeAll(cl.fd, std.mem.asBytes(&req));
        return cl.reply();
    }

    fn offsetOf(cl: *const Client, bytes: []const u8) !u64 {
        const base = @intFromPtr(cl.window.ptr);
        const at = @intFromPtr(bytes.ptr);
        if (at < base or at + bytes.len > base + cl.window.len) return error.BadMessage;
        return at - base;
    }

    /// The host stage of sequence 0.
    pub fn peer(cl: *Client) pipeline.Peer {
        return cl.peerOf(0);
    }

    /// A `Peer` for the server's stage `stream`; the stream's number rides in a context the client keeps.
    pub fn peerOf(cl: *Client, stream: u16) pipeline.Peer {
        std.debug.assert(stream < max_streams);
        cl.lanes[stream] = .{ .client = cl, .stream = stream };
        return .{ .ptr = &cl.lanes[stream], .step = stepFn, .finish = finishFn };
    }

    fn stepFn(ptr: *anyopaque, seq: u32, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const lane: *Lane = @ptrCast(@alignCast(ptr));
        const cl = lane.client;
        const req: Request = .{ .op = @backingInt(Op.step), .level = @backingInt(level), .stream = lane.stream, .rows = rows, .seq = seq, .len = @intCast(out.len), .a = try cl.offsetOf(input), .b = input.len, .d = try cl.offsetOf(out) };
        return (try cl.call(req)).n;
    }

    fn finishFn(ptr: *anyopaque, seq: u32) anyerror!void {
        const lane: *Lane = @ptrCast(@alignCast(ptr));
        _ = try lane.client.call(.{ .op = @backingInt(Op.finish), .stream = lane.stream, .seq = seq });
    }

    /// The server's stage back to position 0 (before a new pipeline runs over the window).
    pub fn reset(cl: *Client) Error!void {
        _ = try cl.call(.{ .op = @backingInt(Op.reset) });
    }

    pub fn close(cl: *Client) void {
        _ = cl.call(.{ .op = @backingInt(Op.bye) }) catch {};
        _ = c.close(cl.fd);
    }
};

/// The server's stages: `bind` is told the window as this process maps it (it may keep it until `unbind`), `reset`
/// returns every stage to position 0, `peers` are the stages themselves, one a sequence.
pub const Host = struct {
    ptr: *anyopaque,
    bind: *const fn (ptr: *anyopaque, window: []u8) anyerror!void,
    unbind: *const fn (ptr: *anyopaque) void,
    reset: *const fn (ptr: *anyopaque) anyerror!void,
    peers: []const pipeline.Peer,
};

/// Listens at `path` for one client and serves it until it says bye or hangs up. `ready` is called once the socket
/// listens, so the caller can print it. The error of a failed call goes to stderr (the client sees only that it failed).
pub fn serve(path: []const u8, host: Host, ready: ?*const fn () void) Error!void {
    const addr = try unixAddr(path);
    _ = c.unlink(@ptrCast(&addr.path));
    const lfd = c.socket(c.AF.UNIX, c.SOCK.STREAM, 0);
    if (lfd < 0) return error.Socket;
    defer {
        _ = c.close(lfd);
        _ = c.unlink(@ptrCast(&addr.path));
    }
    if (c.bind(lfd, @ptrCast(&addr), @sizeOf(c.sockaddr.un)) != 0) return error.Socket;
    if (c.listen(lfd, 1) != 0) return error.Socket;
    if (ready) |f| f();
    const fd = c.accept(lfd, null, null);
    if (fd < 0) return error.Socket;
    defer _ = c.close(fd);
    return serveOver(fd, host);
}

/// The serving loop over a connected socket.
pub fn serveOver(fd: c_int, host: Host) Error!void {
    var mapped: ?[]align(16 * 1024) u8 = null;
    var window: ?[]u8 = null;
    defer {
        if (window != null) host.unbind(host.ptr);
        if (mapped) |m| _ = c.munmap(m.ptr, m.len);
    }
    while (true) {
        var req: Request = undefined;
        const shm = recvWithFd(fd, &req) catch |err| return if (err == error.PeerClosed) {} else err;
        if (req.magic != magic) return error.BadMessage;
        var reply: Reply = .{ .status = 0, .n = 0 };
        const op = std.enums.fromInt(Op, req.op) orelse return error.BadMessage;
        switch (op) {
            .hello => {
                if (shm < 0 or window != null) return error.BadMessage;
                defer _ = c.close(shm);
                const m = c.mmap(null, req.d, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, shm, 0);
                if (m == c.MAP_FAILED) return error.MapFailed;
                mapped = @as([*]align(16 * 1024) u8, @ptrCast(@alignCast(m)))[0..req.d];
                if (req.a + req.b > req.d) return error.BadMessage;
                const w = mapped.?[req.a..][0..req.b];
                if (host.bind(host.ptr, w)) |_| window = w else |err| {
                    std.debug.print("split remote: bind: {s}\n", .{@errorName(err)});
                    reply.status = 1;
                }
            },
            .step => {
                const w = window orelse return error.BadMessage;
                if (req.d + req.len > w.len or req.a + req.b > w.len) return error.BadMessage;
                const level: pipeline.Level = if (req.level == 0) .a else .b;
                if (req.stream >= host.peers.len) return error.BadMessage;
                const pe = host.peers[req.stream];
                if (pe.step(pe.ptr, req.seq, req.rows, w[req.a..][0..req.b], w[req.d..][0..req.len], level)) |n| reply.n = @intCast(n) else |err| {
                    std.debug.print("split remote: step {d}: {s}\n", .{ req.seq, @errorName(err) });
                    reply.status = 1;
                }
            },
            .finish => if (req.stream >= host.peers.len) return error.BadMessage else if (host.peers[req.stream].finish(host.peers[req.stream].ptr, req.seq)) |_| {} else |err| {
                std.debug.print("split remote: finish {d}: {s}\n", .{ req.seq, @errorName(err) });
                reply.status = 1;
            },
            .reset => {
                const w = window orelse return error.BadMessage;
                host.unbind(host.ptr);
                window = null;
                if (host.reset(host.ptr)) |_| {
                    if (host.bind(host.ptr, w)) |_| window = w else |err| {
                        std.debug.print("split remote: rebind: {s}\n", .{@errorName(err)});
                        reply.status = 1;
                    }
                } else |err| {
                    std.debug.print("split remote: reset: {s}\n", .{@errorName(err)});
                    reply.status = 1;
                }
            },
            .bye => {},
        }
        try writeAll(fd, std.mem.asBytes(&reply));
        if (op == .bye) return;
    }
}

fn nowNs() u64 {
    var ts: c.timespec = undefined;
    _ = c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn sleepNs(ns: u64) void {
    const ts: c.timespec = .{ .sec = @intCast(ns / std.time.ns_per_s), .nsec = @intCast(ns % std.time.ns_per_s) };
    _ = c.nanosleep(&ts, null);
}

/// A stage that adds one to every input byte and writes it to the output; hands back what `step` was called with.
const Adder = struct {
    window: ?[]u8 = null,
    resets: u32 = 0,
    last_level: pipeline.Level = .a,
    last_rows: u32 = 0,
    finished: u32 = 0,

    peers: [2]pipeline.Peer = undefined,

    fn host(a: *Adder) Host {
        a.peers = .{ .{ .ptr = a, .step = step, .finish = finish }, .{ .ptr = a, .step = step, .finish = finish } };
        return .{ .ptr = a, .bind = bind, .unbind = unbind, .reset = reset, .peers = &a.peers };
    }
    fn bind(ptr: *anyopaque, w: []u8) anyerror!void {
        @as(*Adder, @ptrCast(@alignCast(ptr))).window = w;
    }
    fn unbind(ptr: *anyopaque) void {
        @as(*Adder, @ptrCast(@alignCast(ptr))).window = null;
    }
    fn reset(ptr: *anyopaque) anyerror!void {
        @as(*Adder, @ptrCast(@alignCast(ptr))).resets += 1;
    }
    fn step(ptr: *anyopaque, seq: u32, rows: u32, input: []const u8, out: []u8, level: pipeline.Level) anyerror!usize {
        const a: *Adder = @ptrCast(@alignCast(ptr));
        a.last_rows = rows;
        if (seq == 99) return error.Boom;
        a.last_level = level;
        for (input, 0..) |b, i| out[i] = b +% 1;
        return input.len;
    }
    fn finish(ptr: *anyopaque, _: u32) anyerror!void {
        @as(*Adder, @ptrCast(@alignCast(ptr))).finished += 1;
    }
};

fn serveThread(fd: c_int, adder: *Adder) void {
    serveOver(fd, adder.host()) catch |err| std.debug.print("serve: {s}\n", .{@errorName(err)});
}

extern "c" fn mkstemp(template: [*:0]u8) c_int;

test "a step, a finish and a reset cross a socket pair; the window is the same memory on both sides" {
    // a file stands in for the card's shared-memory object
    var name = "/tmp/tf-split-remote-XXXXXX".*;
    const file = mkstemp(&name);
    try std.testing.expect(file >= 0);
    defer _ = c.unlink(&name);
    defer _ = c.close(file);
    const size: usize = 64 * 1024;
    try std.testing.expectEqual(@as(c_int, 0), c.ftruncate(file, @intCast(size)));
    const m = c.mmap(null, size, .{ .READ = true, .WRITE = true }, .{ .TYPE = .SHARED }, file, 0);
    try std.testing.expect(m != c.MAP_FAILED);
    defer _ = c.munmap(@ptrCast(@alignCast(m)), size);
    const mine = @as([*]u8, @ptrCast(m))[0..size];

    var pair: [2]c_int = undefined;
    try std.testing.expectEqual(@as(c_int, 0), c.socketpair(c.AF.UNIX, c.SOCK.STREAM, 0, &pair));
    defer _ = c.close(pair[0]);
    var adder: Adder = .{};
    const thread = try std.Thread.spawn(.{}, serveThread, .{ pair[1], &adder });
    var cl = Client.over(pair[0], mine[16384..][0..16384]);
    const dup = c.dup(file);
    try sendWithFd(pair[0], &.{ .op = @backingInt(Op.hello), .a = 16384, .b = 16384, .d = size }, dup);
    _ = c.close(dup);
    _ = try cl.reply();

    const p = cl.peer();
    @memcpy(cl.window[0..4], &[4]u8{ 1, 2, 3, 4 });
    const n = try p.step(p.ptr, 7, 1, cl.window[0..4], cl.window[64..68], .b);
    try std.testing.expectEqual(@as(usize, 4), n);
    try std.testing.expectEqualSlices(u8, &.{ 2, 3, 4, 5 }, cl.window[64..68]);
    try p.finish(p.ptr, 7);
    try cl.reset();
    try std.testing.expectError(error.PeerFailed, p.step(p.ptr, 99, 1, cl.window[0..4], cl.window[64..68], .a));
    // the rows of a chunk travel as a count
    try std.testing.expectEqual(@as(usize, 4), try p.step(p.ptr, 8, 3, cl.window[0..4], cl.window[64..68], .a));
    try std.testing.expectEqual(@as(u32, 3), adder.last_rows);
    // a second stage answers to its own number
    const p1 = cl.peerOf(1);
    try std.testing.expectEqual(@as(usize, 4), try p1.step(p1.ptr, 1, 1, cl.window[0..4], cl.window[64..68], .a));
    try p1.finish(p1.ptr, 1);
    try std.testing.expectEqual(@as(u32, 2), adder.finished);
    // bytes outside the window are refused before they are sent
    var stray: [4]u8 = undefined;
    try std.testing.expectError(error.BadMessage, p.step(p.ptr, 9, 1, &stray, cl.window[64..68], .a));
    _ = c.shutdown(pair[0], c.SHUT.WR);
    thread.join();
    try std.testing.expectEqual(@as(u32, 1), adder.resets);
    try std.testing.expectEqual(@as(u32, 2), adder.finished);
    try std.testing.expectEqual(pipeline.Level.a, adder.last_level);
    try std.testing.expect(adder.window == null);
}
