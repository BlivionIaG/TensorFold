const std = @import("std");
// Darwin sys/ttycom.h; Zig exposes only TIOCGWINSZ.
const tiocswinsz: c_int = @bitCast(@as(u32, 0x80087467));
extern "c" fn openpty(master: *c_int, slave: *c_int, name: ?[*]u8, term: ?*const std.c.termios, size: ?*const std.c.winsize) c_int;

const Terminal = struct {
    master: std.Io.File,
    slave: std.Io.File,

    fn init() !Terminal {
        var master: c_int = undefined;
        var slave: c_int = undefined;
        const size = std.c.winsize{ .row = 24, .col = 120, .xpixel = 0, .ypixel = 0 };
        if (openpty(&master, &slave, null, null, &size) != 0) return error.OpenTerminalFailed;
        return .{ .master = .{ .handle = master, .flags = .{ .nonblocking = false } }, .slave = .{ .handle = slave, .flags = .{ .nonblocking = false } } };
    }

    fn read(t: Terminal, a: std.mem.Allocator) ![]const u8 {
        var output: std.ArrayList(u8) = .empty;
        var buffer: [8192]u8 = undefined;
        while (true) {
            var ready = [_]std.c.pollfd{.{ .fd = t.master.handle, .events = std.c.POLL.IN, .revents = 0 }};
            const polled = std.c.poll(&ready, 1, 0);
            if (polled < 0) return error.TerminalReadFailed;
            if (polled == 0 or ready[0].revents & std.c.POLL.IN == 0) break;
            const count = std.c.read(t.master.handle, &buffer, buffer.len);
            if (count <= 0) return error.TerminalReadFailed;
            try output.appendSlice(a, buffer[0..@intCast(count)]);
        }
        return output.items;
    }
};
const long_request = "{\"prompt\":\"Count upwards, one number per line.\",\"max_tokens\":200000,\"ignore_eos\":true,\"temperature\":0,\"stream\":true}";

fn connect(io: std.Io, port: u16) !std.Io.net.Stream {
    const address = try std.Io.net.IpAddress.parse("127.0.0.1", port);
    return address.connect(io, .{ .mode = .stream });
}

fn headers(io: std.Io, socket: std.Io.net.Stream, length: usize) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.print("POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{length});
    try writer.interface.flush();
}

fn write(io: std.Io, socket: std.Io.net.Stream, bytes: []const u8) !void {
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.writeAll(bytes);
    try writer.interface.flush();
}

fn post(io: std.Io, port: u16, body: []const u8) !std.Io.net.Stream {
    return postRoute(io, port, "/v1/completions", body);
}

fn postRoute(io: std.Io, port: u16, route: []const u8, body: []const u8) !std.Io.net.Stream {
    const socket = try connect(io, port);
    errdefer socket.close(io);
    var buffer: [2048]u8 = undefined;
    var writer = socket.writer(io, &buffer);
    try writer.interface.print("POST {s} HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n\r\n", .{ route, body.len });
    try writer.interface.flush();
    try write(io, socket, body);
    return socket;
}

fn readAll(a: std.mem.Allocator, io: std.Io, socket: std.Io.net.Stream) ![]u8 {
    var buffer: [8192]u8 = undefined;
    var reader = socket.reader(io, &buffer);
    return reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
}

fn memoryWaiting(a: std.mem.Allocator, io: std.Io, port: u16) !i64 {
    return (try health(a, io, port)).object.get("memory").?.object.get("waiting_requests").?.integer;
}

fn health(a: std.mem.Allocator, io: std.Io, port: u16) !std.json.Value {
    const socket = try connect(io, port);
    defer socket.close(io);
    try write(io, socket, "GET /health HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n");
    const response = try readAll(a, io, socket);
    const start = (std.mem.indexOf(u8, response, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
    const body = try std.json.parseFromSlice(std.json.Value, a, response[start..], .{});
    return body.value;
}

const CacheCounts = struct {
    enabled: bool,
    bytes: u64,
    entries: usize,
    hits: u64,
    misses: u64,
    evictions: u64,

    fn read(a: std.mem.Allocator, io: std.Io, port: u16) !CacheCounts {
        const result = try std.json.parseFromValue(CacheCounts, a, (try health(a, io, port)).object.get("prompt_cache").?, .{});
        return result.value;
    }
};

fn waitForMemory(a: std.mem.Allocator, io: std.Io, port: u16, expected: i64) !void {
    for (0..1000) |_| {
        if (try memoryWaiting(a, io, port) == expected) return;
        try std.Io.sleep(io, .fromMilliseconds(25), .awake);
    }
    return error.MissingMemoryWaitState;
}

fn firstEvent(io: std.Io, socket: std.Io.net.Stream) !void {
    var buffer: [8192]u8 = undefined;
    var reader = socket.reader(io, &buffer);
    if (!std.mem.startsWith(u8, try reader.interface.takeSentinel('\n'), "HTTP/1.1 200")) return error.ExpectedStreamingResponse;
    while (true) {
        const line = try reader.interface.takeSentinel('\n');
        if (std.mem.startsWith(u8, line, "data: ")) {
            if (std.mem.indexOf(u8, line, "\"error\"") != null or std.mem.indexOf(u8, line, "[DONE]") != null) return error.ExpectedGeneratedToken;
            return;
        }
    }
}

fn assertCancelled(bytes: []const u8) !void {
    if (std.mem.indexOf(u8, bytes, "\"finish_reason\":\"length\"") != null or std.mem.indexOf(u8, bytes, "\"finish_reason\":\"stop\"") != null) return error.CancelledRequestCompleted;
    if (bytes.len != 0 and std.mem.indexOf(u8, bytes, "RequestTimedOut") == null and std.mem.indexOf(u8, bytes, "ServerStopping") == null and std.mem.indexOf(u8, bytes, "data: ") == null) {
        std.debug.print("Unexpected cancellation response: {s}\n", .{bytes});
        return error.MissingCancellation;
    }
}

const Output = struct {
    content: []const u8,
    reasoning: []const u8,
    finish: []const u8,
    usage: ?[]const u8 = null,

    fn parse(a: std.mem.Allocator, bytes: []const u8, streaming: bool) !Output {
        if (!std.mem.startsWith(u8, bytes, "HTTP/1.1 200")) {
            std.debug.print("Unexpected HTTP response: {s}\n", .{bytes[0..@min(bytes.len, 4096)]});
            return error.HttpRequestFailed;
        }
        if (!streaming) {
            const start = (std.mem.indexOf(u8, bytes, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
            const body = try std.json.parseFromSlice(std.json.Value, a, bytes[start..], .{});
            const choice = body.value.object.get("choices").?.array.items[0];
            const message = choice.object.get("message");
            return .{ .content = if (message) |m| m.object.get("content").?.string else choice.object.get("text").?.string, .reasoning = if (message) |m| m.object.get("reasoning_content").?.string else "", .finish = choice.object.get("finish_reason").?.string, .usage = try std.json.Stringify.valueAlloc(a, body.value.object.get("usage").?, .{}) };
        }
        var content: std.ArrayList(u8) = .empty;
        var reasoning: std.ArrayList(u8) = .empty;
        var finish: ?[]const u8 = null;
        var usage: ?[]const u8 = null;
        var done = false;
        var lines = std.mem.splitScalar(u8, bytes, '\n');
        while (lines.next()) |line| {
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const data = std.mem.trimEnd(u8, line[6..], "\r");
            if (std.mem.eql(u8, data, "[DONE]")) {
                done = true;
                continue;
            }
            const body = try std.json.parseFromSlice(std.json.Value, a, data, .{});
            if (body.value.object.contains("error")) {
                std.debug.print("Stream error: {s}\n", .{data});
                return error.StreamFailed;
            }
            const choice = body.value.object.get("choices").?.array.items[0];
            if (choice.object.get("text")) |text| try content.appendSlice(a, text.string);
            if (choice.object.get("delta")) |delta| {
                if (delta.object.get("content")) |text| try content.appendSlice(a, text.string);
                if (delta.object.get("reasoning_content")) |text| try reasoning.appendSlice(a, text.string);
            }
            if (choice.object.get("finish_reason")) |reason| if (reason == .string) {
                finish = reason.string;
                usage = try std.json.Stringify.valueAlloc(a, body.value.object.get("usage") orelse return error.MissingStreamUsage, .{});
            };
        }
        if (!done or finish == null) {
            std.debug.print("Incomplete HTTP stream: {s}\n", .{bytes[bytes.len -| 4096..]});
            return error.IncompleteStream;
        }
        return .{ .content = content.items, .reasoning = reasoning.items, .finish = finish.?, .usage = usage orelse return error.MissingStreamUsage };
    }

    fn compare(expected: Output, actual: Output) !void {
        return expected.compareWithCache(actual, false);
    }

    fn compareWithCache(expected: Output, actual: Output, comptime strict_cache: bool) !void {
        if (!std.mem.eql(u8, expected.content, actual.content) or !std.mem.eql(u8, expected.reasoning, actual.reasoning) or !std.mem.eql(u8, expected.finish, actual.finish)) return error.ConcurrentOutputMismatch;
        const lhs = try std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, expected.usage orelse return error.MissingUsage, .{});
        defer lhs.deinit();
        const rhs = try std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, actual.usage orelse return error.MissingUsage, .{});
        defer rhs.deinit();
        try @import("native_http_checks.zig").compareUsage(lhs.value, rhs.value);
        if (strict_cache and lhs.value.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer != rhs.value.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer) return error.CachedUsageMismatch;
    }
};

const Scenario = struct {
    init: std.process.Init,
    child: std.process.Child,
    idle: bool,
    rounds: bool = false,
    memory: bool = false,
    prefixes: bool = false,
    live: bool = false,
    drafts: bool = false,
    responses: bool = false,
    background: bool = false,
    background_lanes: usize = 1,
    neural: bool = false,
    neural_enabled: bool = true,
    synthetic: bool = false,
    require_acceptance: bool = true,
    terminal: ?Terminal = null,
    live_enabled: bool = false,
    cache_enabled: bool = true,
    cache_oversize: bool = false,
    image: []const u8 = "",
    http_checks: []const u8 = "",
    disk_phase: ?usize = null,
    disk_expected: ?*[2]?Output = null,
    warming_phase: ?usize = null,
    warming_expected: ?*?Output = null,
    benchmark: ?*Benchmark = null,
    python_port: ?u16 = null,

    fn checkSharing(s: *Scenario, port: u16) !void {
        const status = try health(s.init.arena.allocator(), s.init.io, port);
        if (!status.object.get("shared_decode").?.bool) return;
        const stats = try s.liveSnapshot(port);
        if (stats.object.get("max_shared_streams")) |value| try std.testing.expect(value.integer >= 2);
    }

    fn checkWarming(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const phase = s.warming_phase.?;
        const initial = try health(a, io, port);
        try std.testing.expectEqual(phase == 1 or phase == 4, initial.object.get("warming").?.bool);
        if (phase == 4) {
            try std.posix.kill(s.child.id.?, .INT);
            if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
            std.debug.print("PASS: SIGINT cancels active prefix warming and releases its queued work\n", .{});
            return;
        }
        if (phase == 1) {
            const foreground = try post(io, port, long_request);
            var opened = true;
            defer if (opened) foreground.close(io);
            try firstEvent(io, foreground);
            const during = try health(a, io, port);
            try std.testing.expect(during.object.get("warming").?.bool);
            try std.testing.expect(during.object.get("background_preemptions").?.integer > 0);
            foreground.close(io);
            opened = false;
            var finished = false;
            for (0..6000) |_| {
                if (!(try health(a, io, port)).object.get("warming").?.bool) {
                    finished = true;
                    break;
                }
                try std.Io.sleep(io, .fromMilliseconds(10), .awake);
            }
            if (!finished) return error.WarmingDidNotFinish;
            const cache = try CacheCounts.read(a, io, port);
            try std.testing.expect(cache.entries > 0);
        }
        const system = try a.alloc(u8, 3000 * 5);
        for (0..3000) |i| @memcpy(system[i * 5 ..][0..5], "word ");
        const body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = "Name three colors." } }, .reasoning_effort = "none", .max_tokens = @as(usize, 12), .ignore_eos = true, .temperature = @as(f64, 0.7), .seed = @as(usize, 123) }, .{});
        const socket = try postRoute(io, port, "/v1/chat/completions", body);
        defer socket.close(io);
        const actual = try Output.parse(a, try readAll(a, io, socket), false);
        if (s.warming_expected.?.*) |expected| try expected.compare(actual) else s.warming_expected.?.* = actual;
        const usage = (try std.json.parseFromSlice(std.json.Value, a, actual.usage.?, .{})).value;
        const cached = usage.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer;
        if (phase == 1 or phase == 2) try std.testing.expect(cached >= 3000) else try std.testing.expectEqual(@as(i64, 0), cached);
        try std.posix.kill(s.child.id.?, .INT);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: snapshot warming phase {d}: exact sampled output, rebuilt prefix reuse, foreground preemption and startup policy\n", .{phase});
    }

    fn interruptBackground(s: *Scenario, port: u16, initial_decoded: i64, count: i64) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        var decoded = initial_decoded;
        for (0..2) |round| {
            var progressed = false;
            for (0..1000) |_| {
                const current = (try s.liveSnapshot(port)).object.get("decoded_tokens").?.integer;
                if (current >= decoded + 8) {
                    progressed = true;
                    break;
                }
                try std.Io.sleep(io, .fromMilliseconds(5), .awake);
            }
            if (!progressed) return error.BackgroundDidNotProgress;
            const foreground = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":1,\"temperature\":0}");
            defer foreground.close(io);
            _ = try Output.parse(a, try readAll(a, io, foreground), false);
            const interruptions: i64 = if (s.background_lanes == 1) @as(i64, @intCast(round)) + 1 else 0;
            try std.testing.expectEqual(count + interruptions, (try health(a, io, port)).object.get("background_preemptions").?.integer);
            decoded = (try s.liveSnapshot(port)).object.get("decoded_tokens").?.integer;
        }
    }

    fn checkBackground(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const raw = "{\"prompt\":\"Count upwards, one number per line:\",\"max_tokens\":96,\"ignore_eos\":true,\"temperature\":0.7,\"seed\":123}";
        const chat = "{\"messages\":[{\"role\":\"user\",\"content\":\"Explain why the sky is blue.\"}],\"max_tokens\":96,\"ignore_eos\":true,\"temperature\":0.7,\"seed\":123,\"thinking_budget\":24}";
        const pixels = try std.Io.Dir.cwd().readFileAlloc(io, "build/native-checks/session-image/image.png", a, .limited(4 * 1024 * 1024));
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(pixels.len));
        const url = try std.fmt.allocPrint(a, "data:image/png;base64,{s}", .{std.base64.standard.Encoder.encode(encoded, pixels)});
        const image = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image." }, .{ .type = "image_url", .image_url = .{ .url = url, .detail = "low" } } } }}, .reasoning_effort = "none", .max_tokens = @as(usize, 96), .ignore_eos = true, .temperature = @as(f64, 0.7), .seed = @as(usize, 123) }, .{});
        if (s.background_lanes > 1) {
            const system = try a.alloc(u8, 16384 * 5);
            for (0..16384) |i| @memcpy(system[i * 5 ..][0..5], "word ");
            const body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = "Name three colors." } }, .priority = "background", .reasoning_effort = "none", .max_tokens = @as(usize, 4096), .ignore_eos = true, .stream = true }, .{});
            const count = (try health(a, io, port)).object.get("background_preemptions").?.integer;
            const filling = try postRoute(io, port, "/v1/chat/completions", body);
            var filling_open = true;
            defer if (filling_open) filling.close(io);
            try firstEvent(io, filling);
            const foreground = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":1,\"temperature\":0}");
            defer foreground.close(io);
            _ = try Output.parse(a, try readAll(a, io, foreground), false);
            try std.testing.expectEqual(count, (try health(a, io, port)).object.get("background_preemptions").?.integer);
            filling.close(io);
            filling_open = false;
            _ = try s.waitForCounts(port, 0, 0);
        }
        for ([_][]const u8{ raw, chat, image }, 0..) |source, kind| {
            const route = if (kind == 0) "/v1/completions" else "/v1/chat/completions";
            var request = try std.json.parseFromSlice(std.json.Value, a, source, .{});
            const isolated = try postRoute(io, port, route, source);
            defer isolated.close(io);
            const expected = try Output.parse(a, try readAll(a, io, isolated), false);
            for ([_]bool{ false, true }) |streaming| {
                _ = try s.waitForCounts(port, 0, 0);
                try request.value.object.put(a, "priority", .{ .string = "background" });
                try request.value.object.put(a, "stream", .{ .bool = streaming });
                const decoded = (try s.liveSnapshot(port)).object.get("decoded_tokens").?.integer;
                const count = (try health(a, io, port)).object.get("background_preemptions").?.integer;
                const ongoing = try postRoute(io, port, route, try std.json.Stringify.valueAlloc(a, request.value, .{}));
                defer ongoing.close(io);
                try s.interruptBackground(port, decoded, count);
                try expected.compare(try Output.parse(a, try readAll(a, io, ongoing), streaming));
                _ = try s.waitForCounts(port, 0, 0);
            }
        }
        if (s.background_lanes == 1) {
            try @import("native_responses_checks.zig").checkBackground(s.init, port, s, Scenario.interruptBackground);
            const count = (try health(a, io, port)).object.get("background_preemptions").?.integer;
            const low = "{\"prompt\":\"Count upwards.\",\"priority\":\"background\",\"max_tokens\":4096,\"ignore_eos\":true,\"stream\":true}";
            const active = try post(io, port, low);
            var active_open = true;
            defer if (active_open) active.close(io);
            try firstEvent(io, active);
            const queued = try post(io, port, low);
            var queued_open = true;
            defer if (queued_open) queued.close(io);
            _ = try s.waitForCounts(port, 2, 1);
            try std.testing.expectEqual(count, (try health(a, io, port)).object.get("background_preemptions").?.integer);
            queued.close(io);
            queued_open = false;
            _ = try s.waitForCounts(port, 1, 0);
            const foreground = try post(io, port, long_request);
            var foreground_open = true;
            defer if (foreground_open) foreground.close(io);
            try firstEvent(io, foreground);
            _ = try s.waitForCounts(port, 2, 1);
            try std.testing.expectEqual(count + 1, (try health(a, io, port)).object.get("background_preemptions").?.integer);
            active.close(io);
            active_open = false;
            _ = try s.waitForCounts(port, 1, 0);
            foreground.close(io);
            foreground_open = false;
            _ = try s.waitForCounts(port, 0, 0);
            const recovery = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":1}");
            defer recovery.close(io);
            _ = try Output.parse(a, try readAll(a, io, recovery), false);
        }
        try std.posix.kill(s.child.id.?, .INT);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: background lanes={d}: seeded JSON/SSE text, reasoning, images and usage; preemption only when needed, cancellation and recovery\n", .{s.background_lanes});
    }

    fn checkDisk(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const phase = s.disk_phase.?;
        const initial = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(phase == 1, initial.entries > 0);
        var tokens: [2051]i32 = undefined;
        for (&tokens, 0..) |*id, i| id.* = @intCast(10 + i % 93);
        const body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .seed = @as(usize, 123) }, .{});
        const before = try s.liveSnapshot(port);
        const response = try post(io, port, body);
        defer response.close(io);
        const actual = try Output.parse(a, try readAll(a, io, response), false);
        if (s.disk_expected.?[0]) |expected| try expected.compare(actual) else s.disk_expected.?[0] = actual;
        const after = try s.liveSnapshot(port);
        const fed = after.object.get("prefilled_tokens").?.integer - before.object.get("prefilled_tokens").?.integer;
        try std.testing.expectEqual(@as(i64, if (phase == 1 or phase == 2) 3 else 2051), fed);
        const raw_usage = try std.json.parseFromSlice(std.json.Value, a, actual.usage.?, .{});
        try std.testing.expectEqual(@as(i64, 2051) - fed, raw_usage.value.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer);
        if (phase == 0) {
            tokens[0] = 101;
            const changed = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 1), .ignore_eos = true }, .{});
            const different = try post(io, port, changed);
            defer different.close(io);
            _ = try Output.parse(a, try readAll(a, io, different), false);
            const start = try s.liveSnapshot(port);
            const repeated = try post(io, port, body);
            defer repeated.close(io);
            try actual.compare(try Output.parse(a, try readAll(a, io, repeated), false));
            const finish = try s.liveSnapshot(port);
            try std.testing.expectEqual(@as(i64, 3), finish.object.get("prefilled_tokens").?.integer - start.object.get("prefilled_tokens").?.integer);
        }
        const system = try a.alloc(u8, 5 * 600);
        for (0..600) |i| @memcpy(system[i * 5 ..][0..5], "word ");
        const chat = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = system } }, .reasoning_effort = "none", .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .seed = @as(usize, 21) }, .{});
        const chat_before = try s.liveSnapshot(port);
        const chat_response = try postRoute(io, port, "/v1/chat/completions", chat);
        defer chat_response.close(io);
        const chat_actual = try Output.parse(a, try readAll(a, io, chat_response), false);
        if (s.disk_expected.?[1]) |expected| try expected.compare(chat_actual) else s.disk_expected.?[1] = chat_actual;
        const chat_after = try s.liveSnapshot(port);
        const usage = try std.json.parseFromSlice(std.json.Value, a, chat_actual.usage.?, .{});
        const prompt = usage.value.object.get("prompt_tokens").?.integer;
        const chat_fed = chat_after.object.get("prefilled_tokens").?.integer - chat_before.object.get("prefilled_tokens").?.integer;
        try std.testing.expectEqual(prompt - chat_fed, usage.value.object.get("prompt_tokens_details").?.object.get("cached_tokens").?.integer);
        if (phase == 1 or phase == 2) try std.testing.expect(chat_fed <= prompt - 512) else try std.testing.expectEqual(prompt, chat_fed);
        try std.posix.kill(s.child.id.?, .INT);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: HTTP snapshot phase {d}: spill, restart/on-demand reuse and corrupt-file fallback preserve seeded output\n", .{phase});
    }

    fn liveSnapshot(s: *Scenario, port: u16) !std.json.Value {
        return (try health(s.init.arena.allocator(), s.init.io, port)).object.get("inference").?;
    }

    fn callFunctions(a: std.mem.Allocator, response: []const u8) ![]const u8 {
        const start = (std.mem.indexOf(u8, response, "\r\n\r\n") orelse return error.MissingHttpBody) + 4;
        const body = try std.json.parseFromSlice(std.json.Value, a, response[start..], .{});
        const calls = body.value.object.get("choices").?.array.items[0].object.get("message").?.object.get("tool_calls").?;
        var functions: std.ArrayList(std.json.Value) = .empty;
        for (calls.array.items) |call| try functions.append(a, call.object.get("function").?);
        if (functions.items.len == 0) return error.MissingToolCall;
        return std.json.Stringify.valueAlloc(a, functions.items, .{});
    }

    fn checkNeural(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        var stage: []const u8 = "isolated completions";
        errdefer std.debug.print("Neural HTTP failure during {s}\n", .{stage});
        var expected: [2]Output = undefined;
        const fixture_prompts = try std.json.parseFromSlice(std.json.Value, a, "[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17],[21,22,23,24,25,26,27,28]]", .{});
        const prompts = [_]std.json.Value{
            if (s.synthetic) fixture_prompts.value.array.items[0] else .{ .string = "Explain why the sky is blue:" },
            if (s.synthetic) fixture_prompts.value.array.items[1] else .{ .string = "A short story about a fox:" },
        };
        for (&expected, prompts, 0..) |*output, prompt, i| {
            const socket = try post(io, port, try std.json.Stringify.valueAlloc(a, .{ .prompt = prompt, .max_tokens = 32, .ignore_eos = true, .temperature = @as(f64, if (i == 0) 0 else 0.7), .seed = 819, .draft = false }, .{}));
            defer socket.close(io);
            output.* = try Output.parse(a, try readAll(a, io, socket), false);
        }
        _ = try s.waitForCounts(port, 0, 0);
        if ((try s.liveSnapshot(port)).object.get("neural_proposed").?.integer != 0) return error.SerialRequestUsedNeuralDrafts;
        for ([_]bool{ false, true }) |streaming| {
            stage = if (streaming) "concurrent SSE" else "concurrent JSON";
            var sockets: [2]std.Io.net.Stream = undefined;
            var count: usize = 0;
            defer for (sockets[0..count]) |socket| socket.close(io);
            for (&sockets, prompts, 0..) |*socket, prompt, i| {
                socket.* = try post(io, port, try std.json.Stringify.valueAlloc(a, .{ .prompt = prompt, .max_tokens = 32, .ignore_eos = true, .temperature = @as(f64, if (i == 0) 0 else 0.7), .seed = 819, .draft = true, .stream = streaming }, .{}));
                count += 1;
            }
            for (sockets, expected) |socket, reference| try reference.compare(try Output.parse(a, try readAll(a, io, socket), streaming));
        }
        _ = try s.waitForCounts(port, 0, 0);
        const stats = try s.liveSnapshot(port);
        const proposed = stats.object.get("neural_proposed").?.integer;
        const accepted = stats.object.get("neural_accepted").?.integer;
        if (!s.synthetic and stats.object.contains("shared_rounds")) try s.checkSharing(port);
        if ((proposed > 0) != s.neural_enabled) return error.NeuralDraftActivationMismatch;
        if (s.neural_enabled and !s.synthetic and s.require_acceptance and accepted == 0) return error.NoNeuralDraftsAccepted;
        stage = "cancellation and recovery";
        const abandoned = try post(io, port, try std.json.Stringify.valueAlloc(a, .{ .prompt = prompts[0], .max_tokens = 4096, .ignore_eos = true, .stream = true }, .{}));
        var bytes: [128]u8 = undefined;
        var reader = abandoned.reader(io, &bytes);
        _ = try reader.interface.takeByte();
        abandoned.close(io);
        _ = try s.waitForCounts(port, 0, 0);
        const recovery = try post(io, port, try std.json.Stringify.valueAlloc(a, .{ .prompt = prompts[0], .max_tokens = 32, .ignore_eos = true, .temperature = @as(f64, 0), .seed = 819 }, .{}));
        defer recovery.close(io);
        try expected[0].compare(try Output.parse(a, try readAll(a, io, recovery), false));
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: neural HTTP enabled={any}, {d}/{d} accepted; serial/concurrent JSON/SSE parity, seeded sampling, request opt-out and cancellation recovery\n", .{ s.neural_enabled, accepted, proposed });
    }

    fn checkDrafts(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const request =
            \\{"messages":[{"role":"user","content":"Call read_file with path /tmp/report.txt and offset 3."}],"tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"},"offset":{"type":"integer"}},"required":["path","offset"]}}}],"tool_choice":{"type":"function","function":{"name":"read_file"}},"reasoning_effort":"none","max_tokens":160,"temperature":0,"seed":913}
        ;
        var body = try std.json.parseFromSlice(std.json.Value, a, request, .{});
        for ([_]f64{ 0, 0.7 }) |temperature| {
            try body.value.object.put(a, "temperature", .{ .float = temperature });
            var expected: ?Output = null;
            var functions: ?[]const u8 = null;
            for ([_]bool{ false, true }) |drafting| {
                try body.value.object.put(a, "draft", .{ .bool = drafting });
                const socket = try postRoute(io, port, "/v1/chat/completions", try std.json.Stringify.valueAlloc(a, body.value, .{}));
                defer socket.close(io);
                const response = try readAll(a, io, socket);
                const actual = try Output.parse(a, response, false);
                const calls = try callFunctions(a, response);
                if (expected) |value| {
                    try value.compare(actual);
                    try std.testing.expectEqualStrings(functions.?, calls);
                } else {
                    expected = actual;
                    functions = calls;
                }
                _ = try s.waitForCounts(port, 0, 0);
            }
        }
        const stats = try s.liveSnapshot(port);
        const proposed = stats.object.get("structural_proposed").?.integer;
        const accepted = stats.object.get("structural_accepted").?.integer;
        if (accepted <= 0 or proposed <= accepted) return error.MissingStructuralAcceptanceAndRejection;
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
        for ([_][]const u8{ "--tools-only", "--tool-stream-only", "--controls-only" }) |mode| {
            const result = try std.process.run(a, io, .{ .argv = &.{ s.http_checks, url, mode }, .stderr_limit = .limited(4 * 1024 * 1024) });
            std.debug.print("{s}", .{result.stderr});
            if (!result.term.success()) return error.HttpDraftCompatibilityFailed;
        }
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: greedy/sampled tool drafts match serial calls and usage; {d}/{d} structural tokens accepted, including rejections; SSE and forced controls verified\n", .{ accepted, proposed });
    }

    fn waitForCounts(s: *Scenario, port: u16, connections: i64, waiting: i64) !std.json.Value {
        for (0..1000) |_| {
            const value = try s.liveSnapshot(port);
            if (value.object.get("connections").?.integer == connections and value.object.get("waiting_requests").?.integer == waiting) return value;
            try std.Io.sleep(s.init.io, .fromMilliseconds(25), .awake);
        }
        return error.IncorrectLiveRequestCounts;
    }

    fn waitTerminal(s: *Scenario, terminal: Terminal, suffix: []const u8) !void {
        const a = s.init.arena.allocator();
        var output: std.ArrayList(u8) = .empty;
        for (0..30) |_| {
            try output.appendSlice(a, try terminal.read(a));
            if (std.mem.endsWith(u8, output.items, suffix)) return;
            try std.Io.sleep(s.init.io, .fromMilliseconds(100), .awake);
        }
        std.debug.print("Terminal output: {f}\nExpected suffix: {f}\n", .{ std.json.fmt(output.items, .{}), std.json.fmt(suffix, .{}) });
        return error.MissingTerminalStatus;
    }

    fn checkLive(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        _ = try s.waitForCounts(port, 0, 0);
        const bad = try post(io, port, "{\"prompt\":[],\"max_tokens\":1}");
        defer bad.close(io);
        if (!std.mem.startsWith(u8, try readAll(a, io, bad), "HTTP/1.1 400")) return error.ExpectedInvalidPrompt;
        _ = try s.waitForCounts(port, 0, 0);
        const empty = try post(io, port, "{\"prompt\":[10],\"max_tokens\":0}");
        defer empty.close(io);
        _ = try Output.parse(a, try readAll(a, io, empty), false);
        const untouched = try s.waitForCounts(port, 0, 0);
        try std.testing.expectEqual(@as(i64, 0), untouched.object.get("prefilled_tokens").?.integer);
        try std.testing.expectEqual(@as(i64, 0), untouched.object.get("decoded_tokens").?.integer);
        var tokens: [2051]i32 = undefined;
        for (&tokens, 0..) |*token, i| token.* = @intCast(10 + i % 93);
        const body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0) }, .{});
        var expected: ?Output = null;
        for (0..2) |iteration| {
            const socket = try post(io, port, body);
            defer socket.close(io);
            const result = try Output.parse(a, try readAll(a, io, socket), false);
            if (expected) |value| try value.compare(result) else expected = result;
            const metrics = try s.waitForCounts(port, 0, 0);
            try std.testing.expectEqual(@as(i64, @intCast(2051 + 3 * iteration)), metrics.object.get("prefilled_tokens").?.integer);
            try std.testing.expectEqual(@as(i64, @intCast(8 * (iteration + 1))), metrics.object.get("decoded_tokens").?.integer);
            if (try rate(metrics, "decode_tokens_per_second") <= 0) return error.MissingDecodeRate;
        }
        const long = "{\"prompt\":[10,11],\"max_tokens\":4096,\"ignore_eos\":true,\"stream\":true}";
        var sockets: [9]std.Io.net.Stream = undefined;
        var opened: usize = 0;
        defer for (sockets[0..opened]) |socket| socket.close(io);
        sockets[0] = try post(io, port, long);
        opened = 1;
        try firstEvent(io, sockets[0]);
        for (sockets[1..]) |*socket| {
            socket.* = try post(io, port, long);
            opened += 1;
        }
        _ = try s.waitForCounts(port, 9, 8);
        const rejected = try post(io, port, long);
        defer rejected.close(io);
        if (!std.mem.startsWith(u8, try readAll(a, io, rejected), "HTTP/1.1 503")) return error.ExpectedFullQueue;
        _ = try s.waitForCounts(port, 9, 8);
        for (sockets) |socket| socket.close(io);
        opened = 0;
        _ = try s.waitForCounts(port, 0, 0);
        try std.Io.sleep(io, .fromMilliseconds(2100), .awake);
        const idle = try s.liveSnapshot(port);
        try std.testing.expectEqual(@as(f64, 0), try rate(idle, "decode_tokens_per_second"));
        try std.testing.expectEqual(@as(f64, 0), try rate(idle, "prefill_tokens_per_second"));
        if (s.terminal) |terminal| {
            if (s.live_enabled) {
                try s.waitTerminal(terminal, "[tensorfold] 0 connections · decode 0 tok/s · prefill 0 tok/s");
                const size = std.c.winsize{ .row = 24, .col = 24, .xpixel = 0, .ypixel = 0 };
                if (std.c.ioctl(terminal.slave.handle, tiocswinsz, &size) != 0) return error.ResizeTerminalFailed;
                try s.waitTerminal(terminal, "[tensorfold] 0 connections"[0..23]);
            } else try std.testing.expectEqual(@as(usize, 0), (try terminal.read(a)).len);
        }
        try std.posix.kill(s.child.id.?, .TERM);
        if (s.terminal == null) {
            var buffer: [1024]u8 = undefined;
            var reader = s.child.stdout.?.reader(io, &buffer);
            try std.testing.expectEqual(@as(usize, 0), (try reader.interface.allocRemaining(a, .limited(65536))).len);
        }
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        if (s.terminal) |terminal| {
            const final = try terminal.read(a);
            if (s.live_enabled) {
                if (!std.mem.endsWith(u8, final, "\r\x1b[2K")) return error.MissingTerminalCleanup;
            } else try std.testing.expectEqual(@as(usize, 0), final.len);
        }
        std.debug.print("PASS: live counts, queue overflow, cancellation, cached prefill, token totals, idle expiry and terminal mode={s}\n", .{if (s.live_enabled) "enabled" else if (s.terminal != null) "disabled" else "redirected"});
    }

    fn rate(value: std.json.Value, key: []const u8) !f64 {
        return switch (value.object.get(key).?) {
            .float => |v| v,
            .integer => |v| @floatFromInt(v),
            else => error.ExpectedRate,
        };
    }

    fn checkPrefixes(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        var tokens: [2051]i32 = undefined;
        for (&tokens, 0..) |*id, i| id.* = @intCast(10 + i % 93);
        const body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 16), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 123) }, .{});
        const cold = try post(io, port, body);
        defer cold.close(io);
        const expected = try Output.parse(a, try readAll(a, io, cold), false);
        var counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(s.cache_enabled, counts.enabled);
        try std.testing.expectEqual(@as(u64, 0), counts.hits);
        try std.testing.expectEqual(@as(usize, @intFromBool(s.cache_enabled)), counts.entries);
        if (s.cache_enabled) try std.testing.expect(counts.bytes > 0);
        if (s.cache_oversize) try std.testing.expect(counts.bytes > 1074);
        for ([_]bool{ false, true }) |stream| {
            var request = try std.json.parseFromSlice(std.json.Value, a, body, .{});
            try request.value.object.put(a, "stream", .{ .bool = stream });
            const repeated = try post(io, port, try std.json.Stringify.valueAlloc(a, request.value, .{}));
            defer repeated.close(io);
            try expected.compare(try Output.parse(a, try readAll(a, io, repeated), stream));
        }
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 2 else 0), counts.hits);
        {
            var sockets: [3]std.Io.net.Stream = undefined;
            var opened: usize = 0;
            defer for (sockets[0..opened]) |socket| socket.close(io);
            for (&sockets) |*socket| {
                socket.* = try post(io, port, body);
                opened += 1;
            }
            for (sockets) |socket| try expected.compare(try Output.parse(a, try readAll(a, io, socket), false));
        }
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 5 else 0), counts.hits);
        tokens[0] = 101;
        const changed = try std.json.Stringify.valueAlloc(a, .{ .prompt = &tokens, .max_tokens = @as(usize, 1), .ignore_eos = true }, .{});
        const different = try post(io, port, changed);
        defer different.close(io);
        _ = try Output.parse(a, try readAll(a, io, different), false);
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 1 else 0), counts.evictions);
        const restored = try post(io, port, body);
        defer restored.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, restored), false));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 3 else 0), counts.misses);
        var ongoing = try std.json.parseFromSlice(std.json.Value, a, body, .{});
        try ongoing.value.object.put(a, "stream", .{ .bool = true });
        try ongoing.value.object.put(a, "max_tokens", .{ .integer = 10000 });
        const cancelled = try post(io, port, try std.json.Stringify.valueAlloc(a, ongoing.value, .{}));
        var open = true;
        defer if (open) cancelled.close(io);
        try firstEvent(io, cancelled);
        cancelled.close(io);
        open = false;
        const recovery = try post(io, port, body);
        defer recovery.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, recovery), false));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 7 else 0), counts.hits);
        const system = try a.alloc(u8, 5 * 320);
        for (0..320) |i| @memcpy(system[i * 5 ..][0..5], "word ");
        const conversation = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = "Reply briefly." } }, .reasoning_effort = "none", .max_tokens = @as(usize, 16), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 21) }, .{});
        const cold_chat = try postRoute(io, port, "/v1/chat/completions", conversation);
        defer cold_chat.close(io);
        const expected_chat = try Output.parse(a, try readAll(a, io, cold_chat), false);
        const usage = try std.json.parseFromSlice(std.json.Value, a, expected_chat.usage.?, .{});
        const prompt_tokens = usage.value.object.get("prompt_tokens").?.integer;
        try std.testing.expect(prompt_tokens > 256 and prompt_tokens < 2048);
        var chat = try std.json.parseFromSlice(std.json.Value, a, conversation, .{});
        try chat.value.object.put(a, "stream", .{ .bool = true });
        const cached_chat = try postRoute(io, port, "/v1/chat/completions", try std.json.Stringify.valueAlloc(a, chat.value, .{}));
        defer cached_chat.close(io);
        try expected_chat.compare(try Output.parse(a, try readAll(a, io, cached_chat), true));
        counts = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(u64, if (s.cache_enabled) 8 else 0), counts.hits);
        for ([_][]const u8{ "Continue briefly.", "Revise the previous answer briefly." }) |followup| {
            const turn = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = system }, .{ .role = "user", .content = "Reply briefly." }, .{ .role = "assistant", .content = system }, .{ .role = "user", .content = followup } }, .reasoning_effort = "none", .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 21) }, .{});
            const first = try postRoute(io, port, "/v1/chat/completions", turn);
            defer first.close(io);
            const wanted = try Output.parse(a, try readAll(a, io, first), false);
            var repeated_turn = try std.json.parseFromSlice(std.json.Value, a, turn, .{});
            try repeated_turn.value.object.put(a, "stream", .{ .bool = true });
            const repeated = try postRoute(io, port, "/v1/chat/completions", try std.json.Stringify.valueAlloc(a, repeated_turn.value, .{}));
            defer repeated.close(io);
            try wanted.compare(try Output.parse(a, try readAll(a, io, repeated), true));
        }
        const turns = try CacheCounts.read(a, io, port);
        // With one slot, the revised last user message replaces the previous history checkpoint.
        try std.testing.expectEqual(counts.hits + @as(u64, if (s.cache_enabled) 3 else 0), turns.hits);
        std.debug.print("PASS: HTTP prefix cache enabled={any}: cold/reused/concurrent JSON/SSE agree; eviction, cache counters and cancellation match policy\n", .{s.cache_enabled});
        std.debug.print("PASS: {d}-token adaptive chat JSON/SSE agree, cache enabled={any}\n", .{ prompt_tokens, s.cache_enabled });
        std.debug.print("PASS: multi-turn history and revised prompts reuse checkpoints with seeded JSON/SSE parity\n", .{});
        const shared_system = try a.alloc(u8, 5 * 600);
        const shared_user = try a.alloc(u8, 5 * 600);
        for (0..600) |i| {
            @memcpy(shared_system[i * 5 ..][0..5], "word ");
            @memcpy(shared_user[i * 5 ..][0..5], "item ");
        }
        const shared_body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{ .{ .role = "system", .content = shared_system }, .{ .role = "user", .content = shared_user } }, .reasoning_effort = "none", .max_tokens = @as(usize, 8), .ignore_eos = true, .temperature = @as(f64, 0.7), .top_k = @as(usize, 12), .top_p = @as(f64, 0.8), .seed = @as(usize, 21) }, .{});
        const warm_shared = try postRoute(io, port, "/v1/chat/completions", shared_body);
        defer warm_shared.close(io);
        const shared_expected = try Output.parse(a, try readAll(a, io, warm_shared), false);
        const pinned = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(@as(usize, if (s.cache_enabled) 2 else 0), pinned.entries);
        const churn = try post(io, port, changed);
        defer churn.close(io);
        _ = try Output.parse(a, try readAll(a, io, churn), false);
        const churned = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(pinned.entries, churned.entries);
        const before_shared = try s.liveSnapshot(port);
        const resumed_shared = try postRoute(io, port, "/v1/chat/completions", shared_body);
        defer resumed_shared.close(io);
        const shared_actual = try Output.parse(a, try readAll(a, io, resumed_shared), false);
        try shared_expected.compare(shared_actual);
        const after_shared = try s.waitForCounts(port, 0, 0);
        const restored_shared = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(churned.hits + @as(u64, @intFromBool(s.cache_enabled)), restored_shared.hits);
        const shared_usage = try std.json.parseFromSlice(std.json.Value, a, shared_actual.usage.?, .{});
        const full_prompt = shared_usage.value.object.get("prompt_tokens").?.integer;
        const fed = after_shared.object.get("prefilled_tokens").?.integer - before_shared.object.get("prefilled_tokens").?.integer;
        if (s.cache_enabled) {
            try std.testing.expect(fed > 0 and fed < full_prompt - 512);
        } else try std.testing.expectEqual(full_prompt, fed);
        std.debug.print("PASS: shared system checkpoint survives ordinary LRU eviction and resumes exact seeded output; cache enabled={any}\n", .{s.cache_enabled});
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn checkMemory(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const memory = (try health(a, io, port)).object.get("memory").?.object;
        try std.testing.expectEqual(@as(i64, 3), memory.get("probe_repeats").?.integer);
        try std.testing.expectEqual(@as(i64, 0), memory.get("growth_waits").?.integer);
        try std.testing.expectEqual(@as(i64, 0), memory.get("growth_ends").?.integer);
        const short = "{\"prompt\":\"Hello\",\"max_tokens\":12,\"temperature\":0}";
        const baseline = try post(io, port, short);
        defer baseline.close(io);
        const expected = try Output.parse(a, try readAll(a, io, baseline), false);
        const reserved = "{\"prompt\":\"Count upwards.\",\"max_tokens\":260000,\"ignore_eos\":true,\"temperature\":0,\"stream\":true}";
        var active = try post(io, port, reserved);
        var active_open = true;
        defer if (active_open) active.close(io);
        try firstEvent(io, active);
        const cancelled = try post(io, port, reserved);
        var cancelled_open = true;
        defer if (cancelled_open) cancelled.close(io);
        try firstEvent(io, cancelled);
        try waitForMemory(a, io, port, 0);
        _ = try s.waitForCounts(port, 2, 0);
        const concurrent = try post(io, port, short);
        defer concurrent.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, concurrent), false));
        cancelled.close(io);
        cancelled_open = false;
        try waitForMemory(a, io, port, 0);
        _ = try s.waitForCounts(port, 1, 0);
        const tokens = try a.alloc(i32, 262000);
        @memset(tokens, 1001);
        const oversized = try std.json.Stringify.valueAlloc(a, .{ .prompt = tokens, .max_tokens = @as(usize, 1) }, .{});
        const waiting = try post(io, port, oversized);
        var waiting_open = true;
        defer if (waiting_open) waiting.close(io);
        try waitForMemory(a, io, port, 1);
        _ = try s.waitForCounts(port, 2, 1);
        waiting.close(io);
        waiting_open = false;
        try waitForMemory(a, io, port, 0);
        _ = try s.waitForCounts(port, 1, 0);
        var next = try post(io, port, oversized);
        var next_open = true;
        defer if (next_open) next.close(io);
        try waitForMemory(a, io, port, 1);
        _ = try s.waitForCounts(port, 2, 1);
        active.close(io);
        active_open = false;
        if (std.mem.indexOf(u8, try readAll(a, io, next), "RequestExceedsMemoryBudget") == null) return error.MissingPromptMemoryRefusal;
        try waitForMemory(a, io, port, 0);
        next.close(io);
        next_open = false;
        const recovery = try post(io, port, short);
        defer recovery.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, recovery), false));
        _ = try s.waitForCounts(port, 0, 0);
        std.debug.print("PASS: long replies share rolling reservations; concurrent output matches isolation; oversized prompts wait, cancel and refuse after release\n", .{});

        var prefix: [2051]i32 = undefined;
        for (&prefix, 0..) |*id, i| id.* = @intCast(10 + i % 93);
        const warm_body = try std.json.Stringify.valueAlloc(a, .{ .prompt = &prefix, .max_tokens = @as(usize, 1), .ignore_eos = true }, .{});
        const warm = try post(io, port, warm_body);
        defer warm.close(io);
        _ = try Output.parse(a, try readAll(a, io, warm), false);
        const retained = try CacheCounts.read(a, io, port);
        try std.testing.expect(retained.entries > 0 and retained.bytes > 0);
        const too_long = try post(io, port, oversized);
        defer too_long.close(io);
        const refusal = try readAll(a, io, too_long);
        if (std.mem.indexOf(u8, refusal, "RequestExceedsMemoryBudget") == null) return error.MissingPromptMemoryRefusal;
        const after_refusal = try CacheCounts.read(a, io, port);
        try std.testing.expectEqual(retained.evictions, after_refusal.evictions);
        try std.testing.expectEqual(retained.bytes, after_refusal.bytes);
        const image = try std.Io.Dir.cwd().readFileAlloc(io, s.image, a, .limited(10 * 1024 * 1024));
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
        _ = std.base64.standard.Encoder.encode(encoded, image);
        const url = try std.mem.concat(a, u8, &.{ "data:image/jpeg;base64,", encoded });
        const image_body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image." }, .{ .type = "image_url", .image_url = .{ .url = url, .detail = "high" } } } }}, .max_tokens = @as(usize, 1) }, .{});
        const too_large = try postRoute(io, port, "/v1/chat/completions", image_body);
        defer too_large.close(io);
        const image_refusal = try readAll(a, io, too_large);
        if (std.mem.indexOf(u8, image_refusal, "RequestExceedsMemoryBudget") == null) return error.MissingImageMemoryRefusal;
        const final = try post(io, port, short);
        defer final.close(io);
        try expected.compare(try Output.parse(a, try readAll(a, io, final), false));
        std.debug.print("PASS: oversized prompt and image workspace refused before inference; server recovers unchanged\n", .{});
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn checkRounds(s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        const image = try std.Io.Dir.cwd().readFileAlloc(io, s.image, a, .limited(10 * 1024 * 1024));
        const encoded = try a.alloc(u8, std.base64.standard.Encoder.calcSize(image.len));
        _ = std.base64.standard.Encoder.encode(encoded, image);
        const image_url = try std.mem.concat(a, u8, &.{ "data:image/png;base64,", encoded });
        const image_body = try std.json.Stringify.valueAlloc(a, .{ .messages = &.{.{ .role = "user", .content = .{ .{ .type = "text", .text = "Describe this image briefly." }, .{ .type = "image_url", .image_url = .{ .url = image_url, .detail = "low" } } } }}, .reasoning_effort = "none", .max_tokens = @as(usize, 12), .temperature = @as(f64, 0.8), .seed = @as(usize, 9182) }, .{});
        const cases = [_]struct { route: []const u8, body: []const u8 }{
            .{ .route = "/v1/completions", .body = "{\"prompt\":\"Name three colors:\",\"max_tokens\":24,\"temperature\":0,\"seed\":12}" },
            .{ .route = "/v1/completions", .body = "{\"prompt\":\"A short story about a fox:\",\"max_tokens\":31,\"temperature\":0.9,\"seed\":919}" },
            .{ .route = "/v1/chat/completions", .body = image_body },
        };
        var expected: [cases.len]Output = undefined;
        for (cases, &expected) |case, *value| {
            const before = try CacheCounts.read(a, io, port);
            const socket = try postRoute(io, port, case.route, case.body);
            defer socket.close(io);
            value.* = try Output.parse(a, try readAll(a, io, socket), false);
            if (std.mem.eql(u8, case.route, "/v1/chat/completions")) {
                const after = try CacheCounts.read(a, io, port);
                try std.testing.expectEqual(before.hits, after.hits);
                try std.testing.expectEqual(before.misses, after.misses);
                try std.testing.expectEqual(before.entries, after.entries);
            }
        }
        for ([_]bool{ false, true }) |stream| {
            const background = try post(io, port, long_request);
            var background_open = true;
            defer if (background_open) background.close(io);
            var buffer: [8192]u8 = undefined;
            var reader = background.reader(io, &buffer);
            while (true) {
                const line = try reader.interface.takeSentinel('\n');
                if (std.mem.startsWith(u8, line, "data: ")) break;
            }
            var sockets: [cases.len]std.Io.net.Stream = undefined;
            var opened: usize = 0;
            defer for (sockets[0..opened]) |socket| socket.close(io);
            for (cases, &sockets) |case, *socket| {
                var body = try std.json.parseFromSlice(std.json.Value, a, case.body, .{});
                try body.value.object.put(a, "stream", .{ .bool = stream });
                socket.* = try postRoute(io, port, case.route, try std.json.Stringify.valueAlloc(a, body.value, .{}));
                opened += 1;
            }
            if (stream) {
                background.close(io);
                background_open = false;
            }
            for (sockets, expected) |socket, prior| {
                const actual = try Output.parse(a, try readAll(a, io, socket), stream);
                try prior.compare(actual);
            }
            if (!stream) {
                // The short requests finished while this unbounded request was still active.
                const remainder = reader.interface.buffered();
                if (std.mem.indexOf(u8, remainder, "[DONE]") != null) return error.BackgroundFinishedBeforeShortRequests;
            }
        }
        std.debug.print("PASS: concurrent greedy/sampled/image requests match isolated JSON/SSE; short requests progress during long inference; disconnect preserves other requests\n", .{});
        try s.checkSharing(port);
        const url = try std.fmt.allocPrint(a, "http://127.0.0.1:{d}/v1/chat/completions", .{port});
        for ([_][]const u8{ "--controls-only", "--tool-stream-only", s.image }) |mode| {
            const result = try std.process.run(a, io, .{ .argv = &.{ s.http_checks, url, mode }, .stderr_limit = .limited(4 * 1024 * 1024) });
            std.debug.print("{s}", .{result.stderr});
            if (!result.term.success()) return error.HttpCompatibilityFailed;
        }
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
    }

    fn run(s: *Scenario) anyerror!void {
        const io = s.init.io;
        const a = s.init.arena.allocator();
        var stderr_buffer: [8192]u8 = undefined;
        var stderr = (if (s.python_port != null) s.child.stdout.? else s.child.stderr.?).reader(io, &stderr_buffer);
        const prefix = "Native inference listening at http://127.0.0.1:";
        const port = while (true) {
            const line = try stderr.interface.takeSentinel('\n');
            if (s.python_port) |python_port| {
                std.debug.print("{s}\n", .{line});
                if (std.mem.startsWith(u8, line, "[tensorfold] serving ")) break python_port;
                continue;
            }
            if (std.mem.startsWith(u8, line, "Native memory admission:") or std.mem.startsWith(u8, line, "Native memory ceiling:") or std.mem.startsWith(u8, line, "Qwen calibrated target forward costs:")) std.debug.print("{s}\n", .{line});
            if (std.mem.indexOf(u8, line, prefix)) |start| {
                const value = line[start + prefix.len ..];
                const end = std.mem.indexOfScalar(u8, value, ' ') orelse return error.InvalidListenAddress;
                break try std.fmt.parseInt(u16, value[0..end], 10);
            }
        };
        if (s.benchmark) |bench| {
            var logs: std.Io.Group = .init;
            defer logs.cancel(io);
            try logs.concurrent(io, drainLogs, .{&stderr.interface});
            return bench.run(s, port);
        }
        if (s.disk_phase != null) return s.checkDisk(port);
        if (s.warming_phase != null) return s.checkWarming(port);
        if (s.background) return s.checkBackground(port);
        if (s.responses) {
            try @import("native_responses_checks.zig").check(s.init, port, s.http_checks);
            try std.posix.kill(s.child.id.?, .INT);
            if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
            return;
        }
        if (s.memory) return s.checkMemory(port);
        if (s.neural) return s.checkNeural(port);
        if (s.drafts) return s.checkDrafts(port);
        if (s.live) return s.checkLive(port);
        if (s.prefixes) return s.checkPrefixes(port);
        if (s.rounds) return s.checkRounds(port);
        if (s.idle) {
            try std.posix.kill(s.child.id.?, .INT);
            if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
            std.debug.print("PASS: SIGINT exits an idle server cleanly\n", .{});
            return;
        }

        const slow_head = try connect(io, port);
        defer slow_head.close(io);
        try write(io, slow_head, "POST /v1/completions HTTP/1.1\r\n");
        const slow_body = try connect(io, port);
        defer slow_body.close(io);
        try headers(io, slow_body, 9999);
        try write(io, slow_body, "{");
        const queued = try connect(io, port);
        defer queued.close(io);
        try headers(io, queued, long_request.len);
        try std.Io.sleep(io, .fromMilliseconds(250), .awake);
        const active = try post(io, port, long_request);
        defer active.close(io);
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try write(io, queued, long_request);
        try assertCancelled(try readAll(a, io, queued));
        try assertCancelled(try readAll(a, io, active));
        _ = try readAll(a, io, slow_head);
        _ = try readAll(a, io, slow_body);
        // A GPU operation already in flight may finish after the socket deadline.
        var recovered = false;
        for (0..16) |_| {
            const recovery = try post(io, port, "{\"prompt\":\"Hello\",\"max_tokens\":0}");
            defer recovery.close(io);
            const response = try readAll(a, io, recovery);
            if (std.mem.indexOf(u8, response, "200 OK") != null and std.mem.indexOf(u8, response, "\"completion_tokens\":0") != null) {
                recovered = true;
                break;
            }
            try assertCancelled(response);
        }
        if (!recovered) return error.ServerDidNotRecover;
        std.debug.print("PASS: deadlines stop active/queued inference and partial requests; next request succeeds\n", .{});

        const generating = try post(io, port, long_request);
        defer generating.close(io);
        var buffer: [8192]u8 = undefined;
        var reader = generating.reader(io, &buffer);
        while (true) {
            const line = try reader.interface.takeSentinel('\n');
            if (std.mem.startsWith(u8, line, "data: ")) break;
        }
        const waiting = try post(io, port, long_request);
        defer waiting.close(io);
        const stalled = try connect(io, port);
        defer stalled.close(io);
        try write(io, stalled, "POST /v1/completions HTTP/1.1\r\n");
        try std.Io.sleep(io, .fromMilliseconds(100), .awake);
        try std.posix.kill(s.child.id.?, .TERM);
        try assertCancelled(try reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024)));
        try assertCancelled(try readAll(a, io, waiting));
        _ = try readAll(a, io, stalled);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        std.debug.print("PASS: SIGTERM cancels active/queued inference, releases stalled clients and exits cleanly\n", .{});
    }
};

fn invalidateWarmSnapshots(init: std.process.Init, directory: []const u8) !void {
    const a = init.arena.allocator();
    const io = init.io;
    var dir = try std.Io.Dir.cwd().openDir(io, directory, .{ .iterate = true });
    defer dir.close(io);
    var iterator = dir.iterate();
    var changed: usize = 0;
    while (try iterator.next(io)) |entry| {
        if (!std.mem.endsWith(u8, entry.name, ".safetensors")) continue;
        const file = try dir.openFile(io, entry.name, .{ .mode = .read_write });
        defer file.close(io);
        var size: [8]u8 = undefined;
        if (try file.readPositionalAll(io, &size, 0) != size.len) return error.InvalidSnapshotHeader;
        const length = std.mem.readInt(u64, &size, .little);
        if (length > 64 * 1024 * 1024) return error.InvalidSnapshotHeader;
        const buffer = try a.alloc(u8, @intCast(length));
        if (try file.readPositionalAll(io, buffer, 8) != buffer.len) return error.InvalidSnapshotHeader;
        var header = try std.json.parseFromSlice(std.json.Value, a, buffer, .{ .allocate = .alloc_always });
        const encoded = header.value.object.getPtr("__metadata__").?.object.getPtr("tensorfold_native").?;
        var metadata = try std.json.parseFromSlice(std.json.Value, a, encoded.string, .{});
        const identity = try a.dupe(u8, metadata.value.object.get("identity").?.string);
        const at = (std.mem.indexOfScalar(u8, identity, '|') orelse return error.MissingSnapshotRevision) + 1;
        @memset(identity[at..], '0');
        try metadata.value.object.put(a, "identity", .{ .string = identity });
        try metadata.value.object.put(a, "dependencies", .{ .string = "obsolete" });
        try metadata.value.object.put(a, "state", .null);
        encoded.* = .{ .string = try std.json.Stringify.valueAlloc(a, metadata.value, .{}) };
        const updated = try std.json.Stringify.valueAlloc(a, header.value, .{});
        if (updated.len > buffer.len) return error.SnapshotHeaderGrew;
        @memset(buffer, ' ');
        @memcpy(buffer[0..updated.len], updated);
        try file.writePositionalAll(io, buffer, 8);
        changed += 1;
    }
    try std.testing.expect(changed > 0);
}

const TimedResponse = struct {
    socket: std.Io.net.Stream,
    started: f64,
    first_token_ms: f64 = 0,
    latency_ms: f64 = 0,
    bytes: []const u8 = &.{},
    failure: ?anyerror = null,

    fn read(r: *TimedResponse, io: std.Io) void {
        r.readInner(io) catch |err| {
            r.failure = err;
        };
    }

    fn readInner(r: *TimedResponse, io: std.Io) !void {
        const a = std.heap.page_allocator;
        var buffer: [8192]u8 = undefined;
        var reader = r.socket.reader(io, &buffer);
        var bytes: std.ArrayList(u8) = .empty;
        errdefer bytes.deinit(a);
        while (true) {
            const line = try reader.interface.takeSentinel('\n');
            try bytes.appendSlice(a, line);
            try bytes.append(a, '\n');
            if (!std.mem.startsWith(u8, line, "data: ")) continue;
            const data = std.mem.trim(u8, line[6..], "\r\n ");
            if (std.mem.eql(u8, data, "[DONE]")) return error.MissingFirstToken;
            const parsed = try std.json.parseFromSlice(std.json.Value, a, data, .{});
            defer parsed.deinit();
            if (parsed.value.object.contains("error")) return error.StreamFailed;
            const choices = parsed.value.object.get("choices") orelse continue;
            if (choices.array.items.len == 0) continue;
            const text = choices.array.items[0].object.get("text") orelse continue;
            if (text == .string and text.string.len > 0) {
                r.first_token_ms = (instant(io) - r.started) * 1000;
                break;
            }
        }
        const rest = try reader.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
        defer a.free(rest);
        try bytes.appendSlice(a, rest);
        r.latency_ms = (instant(io) - r.started) * 1000;
        r.bytes = try bytes.toOwnedSlice(a);
    }
};

fn instant(io: std.Io) f64 {
    return @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds())) / std.time.ns_per_s;
}

fn drainLogs(reader: *std.Io.Reader) void {
    while (true) {
        const line = reader.takeSentinel('\n') catch return;
        std.debug.print("{s}\n", .{line});
    }
}

const Benchmark = struct {
    const ProcessMemory = @import("process_memory").Snapshot;
    const Process = struct {
        variant: []const u8,
        phase: usize,
        startup: ProcessMemory,
        before_shutdown: ProcessMemory,
        peak_rss_bytes: usize,
    };
    const Record = struct {
        variant: []const u8,
        phase: usize,
        repetition: usize,
        streams: usize,
        draft: bool,
        tokens: usize,
        seconds: f64,
        tokens_per_second: f64,
        first_token_ms: []const f64,
        latency_ms: []const f64,
        shared_rounds: i64,
        max_shared_streams: i64,
        inference_before: std.json.Value,
        inference_after: std.json.Value,
        memory: std.json.Value,
        process_memory: ?ProcessMemory = null,
    };
    expected: [64]?Output = @splat(null),
    records: std.ArrayList(Record) = .empty,
    processes: std.ArrayList(Process) = .empty,
    variant: []const u8 = "baseline",
    phase: usize = 0,
    drafter: bool = false,
    expect_sharing: bool = false,
    concurrency: [3]usize = .{ 1, 4, 8 },
    capacity: ?usize = null,
    mismatch_path: []const u8 = "",
    verify_only: bool = false,
    memory_map_prefix: ?[]const u8 = null,

    fn captureMemoryMap(b: *const Benchmark, s: *Scenario, stage: []const u8) !void {
        const prefix = b.memory_map_prefix orelse return;
        const a = s.init.arena.allocator();
        const pid = try std.fmt.allocPrint(a, "{d}", .{s.child.id.?});
        const result = try std.process.run(a, s.init.io, .{ .argv = &.{ "/usr/bin/vmmap", "-summary", pid }, .stdout_limit = .limited(1024 * 1024) });
        if (!result.term.success()) {
            std.debug.print("{s}", .{result.stderr});
            return error.ProcessMemoryMapFailed;
        }
        const path = try std.fmt.allocPrint(a, "{s}.{s}-{d}-{s}.vmmap.txt", .{ prefix, b.variant, b.phase, stage });
        try std.Io.Dir.cwd().writeFile(s.init.io, .{ .sub_path = path, .data = result.stdout });
    }

    fn maxStreams(b: *const Benchmark) usize {
        return std.mem.max(usize, &b.concurrency);
    }

    fn streamCounts(b: *const Benchmark) []const usize {
        return b.concurrency[0..@min(3, b.maxStreams())];
    }

    fn configureCapacity(b: *Benchmark, status: std.json.Value) !void {
        const value = if (status == .object) status.object.get("max_batch_size") orelse return error.MissingBenchmarkCapacity else return error.MissingBenchmarkCapacity;
        if (value != .integer or value.integer < 1 or value.integer > b.expected.len) return error.InvalidBenchmarkCapacity;
        const streams: usize = @intCast(value.integer);
        if (b.capacity) |reference| {
            if (streams != reference) {
                std.debug.print("Benchmark stream capacity mismatch: {s} reports {d}, expected {d}\n", .{ b.variant, streams, reference });
                return error.BenchmarkCapacityMismatch;
            }
        } else {
            b.capacity = streams;
            b.concurrency = .{ 1, @min(streams, @min(8, @max(2, streams / 2))), streams };
        }
    }

    const MemoryComparison = struct {
        python_peak_rss_bytes: usize = 0,
        native_peak_rss_bytes: usize = 0,
        python_peak_footprint_bytes: u64 = 0,
        native_peak_footprint_bytes: u64 = 0,
        passed: bool = false,
    };

    fn memoryComparison(b: *const Benchmark) MemoryComparison {
        var result = MemoryComparison{};
        for (b.processes.items) |process| {
            if (std.mem.eql(u8, process.variant, "python")) {
                result.python_peak_rss_bytes = @max(result.python_peak_rss_bytes, process.peak_rss_bytes);
                result.python_peak_footprint_bytes = @max(result.python_peak_footprint_bytes, process.before_shutdown.peak_footprint_bytes);
            } else if (std.mem.eql(u8, process.variant, "native")) {
                result.native_peak_rss_bytes = @max(result.native_peak_rss_bytes, process.peak_rss_bytes);
                result.native_peak_footprint_bytes = @max(result.native_peak_footprint_bytes, process.before_shutdown.peak_footprint_bytes);
            }
        }
        result.passed = result.native_peak_rss_bytes > 0 and result.native_peak_footprint_bytes > 0 and
            result.native_peak_rss_bytes <= result.python_peak_rss_bytes and result.native_peak_footprint_bytes <= result.python_peak_footprint_bytes;
        return result;
    }

    const Performance = struct {
        const Cell = struct {
            streams: usize,
            draft: bool,
            python_samples: usize,
            native_samples: usize,
            python_median_tokens_per_second: ?f64,
            native_median_tokens_per_second: ?f64,
            passed: bool,
        };
        required_samples: usize = 6,
        criterion: []const u8 = "Native median completion tokens/s >= Python in every concurrency/draft cell; six samples per implementation",
        passed: bool,
        cells: []const Cell,
    };

    const PythonReference = struct {
        const Cell = struct {
            streams: usize,
            draft: bool,
            python_samples: usize,
            python_median_tokens_per_second: ?f64,
            valid: bool,
        };
        required_samples: usize = 6,
        valid: bool,
        cells: []const Cell,
    };

    fn samples(b: *const Benchmark, variant: []const u8, streams: usize, draft: bool) struct { count: usize, median: ?f64 } {
        var values: [6]f64 = undefined;
        var count: usize = 0;
        var valid = true;
        for (b.records.items) |record| {
            if (!std.mem.eql(u8, record.variant, variant) or record.streams != streams or record.draft != draft) continue;
            valid = valid and std.math.isFinite(record.tokens_per_second) and record.tokens_per_second > 0;
            if (count < values.len) values[count] = record.tokens_per_second;
            count += 1;
        }
        if (count != values.len or !valid) return .{ .count = count, .median = null };
        std.mem.sort(f64, &values, {}, std.sort.asc(f64));
        return .{ .count = count, .median = (values[2] + values[3]) / 2 };
    }

    fn performance(b: *const Benchmark, a: std.mem.Allocator) !Performance {
        const counts = b.streamCounts();
        const cells = try a.alloc(Performance.Cell, counts.len * @as(usize, if (b.drafter) 2 else 1));
        var passed = true;
        for (cells, 0..) |*cell, i| {
            const streams = counts[i % counts.len];
            const draft = i >= counts.len;
            const python = b.samples("python", streams, draft);
            const native = b.samples("native", streams, draft);
            const at_least_python = python.median != null and native.median != null and native.median.? >= python.median.?;
            cell.* = .{ .streams = streams, .draft = draft, .python_samples = python.count, .native_samples = native.count, .python_median_tokens_per_second = python.median, .native_median_tokens_per_second = native.median, .passed = at_least_python };
            passed = passed and at_least_python;
        }
        return .{ .passed = passed, .cells = cells };
    }

    fn pythonReference(b: *const Benchmark, a: std.mem.Allocator) !PythonReference {
        const counts = b.streamCounts();
        const cells = try a.alloc(PythonReference.Cell, counts.len * @as(usize, if (b.drafter) 2 else 1));
        var valid = b.records.items.len == cells.len * 6;
        for (cells, 0..) |*cell, i| {
            const streams = counts[i % counts.len];
            const draft = i >= counts.len;
            const python = b.samples("python", streams, draft);
            var cell_valid = python.median != null and std.math.isFinite(python.median.?);
            var seen: u6 = 0;
            for (b.records.items) |record| {
                if (!std.mem.eql(u8, record.variant, "python") or record.streams != streams or record.draft != draft) continue;
                if ((record.phase != 0 and record.phase != 3) or record.repetition >= 3) {
                    cell_valid = false;
                    continue;
                }
                const slot: u3 = @intCast((if (record.phase == 0) @as(usize, 0) else 3) + record.repetition);
                const mask = @as(u6, 1) << slot;
                cell_valid = cell_valid and seen & mask == 0;
                seen |= mask;
                cell_valid = cell_valid and record.tokens == 32 * streams and std.math.isFinite(record.seconds) and record.seconds > 0;
                cell_valid = cell_valid and record.tokens_per_second == @as(f64, @floatFromInt(record.tokens)) / record.seconds;
            }
            cell_valid = cell_valid and seen == 0b111111;
            cell.* = .{ .streams = streams, .draft = draft, .python_samples = python.count, .python_median_tokens_per_second = if (cell_valid) python.median else null, .valid = cell_valid };
            valid = valid and cell_valid;
        }
        return .{ .valid = valid, .cells = cells };
    }

    fn validateOutput(actual: Output) !void {
        const usage = try std.json.parseFromSlice(std.json.Value, std.heap.page_allocator, actual.usage orelse return error.MissingUsage, .{});
        defer usage.deinit();
        try @import("native_http_checks.zig").compareUsage(usage.value, usage.value);
        if (usage.value.object.get("completion_tokens").?.integer != 32) return error.InvalidBenchmarkCompletionCount;
    }

    fn compare(b: *Benchmark, s: *Scenario, i: usize, actual: Output) !void {
        try validateOutput(actual);
        const reference = b.expected[i].?;
        reference.compareWithCache(actual, true) catch |err| {
            const a = s.init.arena.allocator();
            const bytes = try std.json.Stringify.valueAlloc(a, .{ .variant = b.variant, .phase = b.phase, .request = try body(a, i, false), .expected = reference, .actual = actual, .failure = @errorName(err) }, .{ .whitespace = .indent_2 });
            try std.Io.Dir.cwd().writeFile(s.init.io, .{ .sub_path = b.mismatch_path, .data = bytes });
            std.debug.print("Output mismatch request {d}, {s}; saved {s}\n", .{ i, b.variant, b.mismatch_path });
            return err;
        };
    }

    fn body(a: std.mem.Allocator, i: usize, drafting: bool) ![]const u8 {
        const prompts = [_][]const u8{ "Explain why the sky is blue:", "A short story about a fox:", "Write a Fibonacci function in Zig:", "Describe the seasons in Copenhagen:", "Count upwards, one number per line:", "What makes a good unit test?", "Explain how binary search works:", "Describe a walk beside the sea:" };
        return std.json.Stringify.valueAlloc(a, .{ .prompt = prompts[i % prompts.len], .max_tokens = 32, .ignore_eos = true, .temperature = @as(f64, if (i % 2 == 0) 0 else 0.7), .top_k = 20, .top_p = 0.95, .min_p = 0.0, .seed = 819 + i, .draft = drafting, .stream = true }, .{});
    }

    fn run(b: *Benchmark, s: *Scenario, port: u16) !void {
        const a = s.init.arena.allocator();
        const io = s.init.io;
        try b.configureCapacity(try health(a, io, port));
        const startup = try ProcessMemory.read(s.child.id.?);
        try b.captureMemoryMap(s, "startup");
        // Establish each request's serial reference before sharing caches or kernels.
        for (b.expected[0..b.maxStreams()], 0..) |*expected, i| {
            const socket = try post(io, port, try body(a, i, false));
            defer socket.close(io);
            const actual = try Output.parse(a, try readAll(a, io, socket), true);
            if (expected.* != null) try b.compare(s, i, actual) else {
                try validateOutput(actual);
                expected.* = actual;
            }
        }
        for (0..if (b.drafter) @as(usize, 2) else 1) |mode| {
            if (mode == 1) for (0..b.maxStreams()) |i| {
                const socket = try post(io, port, try body(a, i, true));
                defer socket.close(io);
                try b.compare(s, i, try Output.parse(a, try readAll(a, io, socket), true));
            };
            for (b.streamCounts()) |count| {
                for (0..if (b.verify_only) @as(usize, 1) else 3) |repetition| {
                    const before = if (b.verify_only) std.json.Value.null else (try health(a, io, port)).object.get("inference") orelse std.json.Value.null;
                    var replies: [64]TimedResponse = undefined;
                    var opened: usize = 0;
                    defer for (replies[0..opened]) |r| {
                        r.socket.close(io);
                        std.heap.page_allocator.free(r.bytes);
                    };
                    var group: std.Io.Group = .init;
                    defer group.cancel(io);
                    const started = instant(io);
                    for (replies[0..count], 0..) |*reply, i| {
                        const request = try body(a, i, mode == 1);
                        const sent = instant(io);
                        reply.* = .{ .socket = try post(io, port, request), .started = sent };
                        opened += 1;
                        try group.concurrent(io, TimedResponse.read, .{ reply, io });
                    }
                    try group.await(io);
                    const seconds = instant(io) - started;
                    const first = try a.alloc(f64, count);
                    const latency = try a.alloc(f64, count);
                    var total: usize = 0;
                    for (replies[0..count], 0..) |reply, i| {
                        if (reply.failure) |err| return err;
                        const actual = try Output.parse(a, reply.bytes, true);
                        try b.compare(s, i, actual);
                        const usage = (try std.json.parseFromSlice(std.json.Value, a, actual.usage.?, .{})).value;
                        total += @intCast(usage.object.get("completion_tokens").?.integer);
                        first[i] = reply.first_token_ms;
                        latency[i] = reply.latency_ms;
                    }
                    if (s.python_port == null) _ = try s.waitForCounts(port, 0, 0);
                    const status = try health(a, io, port);
                    const stats = status.object.get("inference") orelse std.json.Value{ .object = .empty };
                    try b.records.append(a, .{ .variant = b.variant, .phase = b.phase, .repetition = repetition, .streams = count, .draft = mode == 1, .tokens = total, .seconds = seconds, .tokens_per_second = @as(f64, @floatFromInt(total)) / seconds, .first_token_ms = first, .latency_ms = latency, .shared_rounds = if (stats.object.get("shared_rounds")) |v| v.integer else 0, .max_shared_streams = if (stats.object.get("max_shared_streams")) |v| v.integer else 0, .inference_before = before, .inference_after = stats, .memory = status.object.get("memory").?, .process_memory = try ProcessMemory.read(s.child.id.?) });
                    if (b.verify_only) {
                        std.debug.print("PASS: {s}, {d} requests, draft={any}: exact seeded greedy/sampled SSE output and usage; peak footprint {d}\n", .{ b.variant, count, mode == 1, b.records.items[b.records.items.len - 1].process_memory.?.peak_footprint_bytes });
                        continue;
                    }
                    std.debug.print("BENCH {s} phase={d} streams={d} draft={any} rep={d}: {d:.2} completion tok/s, {d:.3}s, exact output\n", .{ b.variant, b.phase, count, mode == 1, repetition, @as(f64, @floatFromInt(total)) / seconds, seconds });
                }
            }
        }
        if (std.mem.eql(u8, b.variant, "native") and b.expect_sharing) {
            const stats = try s.liveSnapshot(port);
            try std.testing.expectEqual(@as(i64, @intCast(b.maxStreams())), stats.object.get("max_shared_streams").?.integer);
        }
        const before_shutdown = try ProcessMemory.read(s.child.id.?);
        try b.captureMemoryMap(s, "finished");
        try std.posix.kill(s.child.id.?, .TERM);
        if (!(try s.child.wait(io)).success()) return error.UncleanShutdown;
        const peak_rss = s.child.resource_usage_statistics.getMaxRss() orelse return error.ProcessMemoryUnavailable;
        try b.processes.append(a, .{ .variant = b.variant, .phase = b.phase, .startup = startup, .before_shutdown = before_shutdown, .peak_rss_bytes = peak_rss });
    }
};

fn benchmark(init: std.process.Init, all_args: []const []const u8) !void {
    var memory_maps = false;
    var stream_limit: ?usize = null;
    var end = all_args.len;
    while (end > 6) {
        if (std.mem.eql(u8, all_args[end - 1], "--memory-maps") and !memory_maps) {
            memory_maps = true;
            end -= 1;
        } else if (end >= 8 and std.mem.eql(u8, all_args[end - 2], "--streams") and stream_limit == null) {
            stream_limit = try std.fmt.parseInt(usize, all_args[end - 1], 10);
            if (stream_limit.? < 1 or stream_limit.? > 64) return error.InvalidBenchmarkCapacity;
            end -= 2;
        } else break;
    }
    const args = all_args[0..end];
    if (args.len != 6 and args.len != 8) return error.InvalidBenchmarkArguments;
    if (args.len == 8 and !std.mem.eql(u8, args[6], "--drafter")) return error.InvalidBenchmarkArguments;
    const a = init.arena.allocator();
    const verify_only = std.mem.eql(u8, args[1], "--verify-python");
    const python_only = std.mem.eql(u8, args[1], "--benchmark-python");
    if (python_only and !std.mem.eql(u8, args[2], args[3])) return error.InvalidBenchmarkArguments;
    const python = python_only or verify_only or std.mem.eql(u8, args[1], "--compare-python");
    const repeat_native = !python and std.mem.eql(u8, args[2], args[3]);
    const config_path = try std.fs.path.join(a, &.{ args[4], "config.json" });
    const config_bytes = try std.Io.Dir.cwd().readFileAlloc(init.io, config_path, a, .limited(1024 * 1024));
    const config = (try std.json.parseFromSlice(std.json.Value, a, config_bytes, .{})).value;
    const kind = config.object.get("model_type").?.string;
    const external_drafter = args.len == 8 and !std.mem.eql(u8, args[7], "-");
    const gemma = std.mem.startsWith(u8, kind, "gemma4");
    const draft_budget: usize = if (args.len != 8) 0 else if (external_drafter) 15 else 3;
    const draft_bits: usize = if (external_drafter) (if (gemma) 8 else 4) else 0;
    var bench = Benchmark{ .drafter = args.len == 8, .expect_sharing = std.mem.startsWith(u8, kind, "qwen3") or std.mem.startsWith(u8, kind, "qwen4") or std.mem.startsWith(u8, kind, "gemma4") or std.mem.eql(u8, kind, "prism_hadamard_qwen35") or std.mem.eql(u8, kind, "nemotron_h") };
    const streams: usize = stream_limit orelse if (std.mem.startsWith(u8, kind, "qwen3") or std.mem.eql(u8, kind, "prism_hadamard_qwen35") or std.mem.eql(u8, kind, "nemotron_h")) 64 else if (std.mem.startsWith(u8, kind, "qwen4")) 32 else if (gemma) 16 else 1;
    bench.capacity = streams;
    bench.concurrency = .{ 1, @min(streams, @min(8, @max(2, streams / 2))), streams };
    bench.verify_only = verify_only;
    if (memory_maps) bench.memory_map_prefix = args[5];
    bench.mismatch_path = try std.fmt.allocPrint(a, "{s}.mismatch.json", .{args[5]});
    var environment = try init.environ_map.clone(a);
    defer environment.deinit();
    var ram: u64 = 0;
    var ram_size: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &ram, &ram_size, null, 0) != 0 or ram == 0) return error.PhysicalMemoryUnavailable;
    // Both implementations cap this explicit allowance at Metal's recommended working set.
    try environment.put("TENSORFOLD_MEMORY_LIMIT_GB", try std.fmt.allocPrint(a, "{d}", .{ram / (1024 * 1024 * 1024)}));
    const order = [_]usize{ 2, 3, 3, 2 };
    for (order[0..if (verify_only) @as(usize, 2) else 4], 0..) |binary, phase| {
        if (python_only and binary != 2) continue;
        bench.phase = phase;
        bench.variant = if (repeat_native) "native" else if (binary == 2) (if (python) "python" else "baseline") else "native";
        const is_python = python and binary == 2;
        var python_port: ?u16 = null;
        if (is_python) {
            var listener = try (try std.Io.net.IpAddress.parse("127.0.0.1", 0)).listen(init.io, .{});
            python_port = listener.socket.address.getPort();
            listener.deinit(init.io);
        }
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(a, &.{ args[binary], "serve", args[4], "--snapshot-dir", "none", "--port", if (python_port) |p| try std.fmt.allocPrint(a, "{d}", .{p}) else "0", if (is_python) "--parallel" else "--batch-streams", try std.fmt.allocPrint(a, "{d}", .{streams}), "--prompt-cache-gib", "0" });
        if (is_python) {
            try argv.append(a, "--no-update-check");
            if (std.mem.indexOf(u8, args[4], "Flash-Next") != null) try argv.append(a, "--ple-on-ssd");
        }
        if (bench.drafter) {
            if (external_drafter) {
                try argv.appendSlice(a, &.{ "--drafter", args[7], "--drafter-bits", try std.fmt.allocPrint(a, "{d}", .{draft_bits}) });
            } else if (is_python) {
                try argv.appendSlice(a, &.{ "--drafter", "none" });
            }
            if (!is_python or !external_drafter) try argv.appendSlice(a, &.{ if (is_python) "--mtp-drafts" else "--max-draft", try std.fmt.allocPrint(a, "{d}", .{draft_budget}) });
        } else {
            try argv.append(a, "--no-drafts");
            if (is_python) try argv.appendSlice(a, &.{ "--mtp-drafts", "0" });
        }
        var scenario = Scenario{ .init = init, .idle = false, .benchmark = &bench, .python_port = python_port, .child = try std.process.spawn(init.io, .{ .argv = argv.items, .environ_map = &environment, .stdout = if (is_python) .pipe else .inherit, .stderr = if (is_python) .inherit else .pipe, .request_resource_usage_statistics = true }) };
        defer if (scenario.child.id) |id| {
            std.posix.kill(id, .KILL) catch {};
            scenario.child.kill(init.io);
        };
        try scenario.run();
    }
    const performance = if (python and !verify_only and !python_only) try bench.performance(a) else null;
    const python_reference = if (python_only) try bench.pythonReference(a) else null;
    const memory = if (python and !python_only) bench.memoryComparison() else null;
    const bytes = try std.json.Stringify.valueAlloc(a, .{
        .model = args[4],
        .baseline = args[2],
        .candidate = if (python_only) @as(?[]const u8, null) else args[3],
        .comparison = if (python_only) "python_reference" else if (python) "python_vs_native" else if (repeat_native) "native_repeatability" else "before_after",
        .correctness_only = verify_only,
        .outputs = bench.expected[0..bench.maxStreams()],
        .concurrency = bench.streamCounts(),
        .max_batch_size = bench.maxStreams(),
        .memory_limit_gib = ram / (1024 * 1024 * 1024),
        .draft_budget = draft_budget,
        .drafter = if (bench.drafter) args[7] else "none",
        .drafter_bits = draft_bits,
        .draft_policy = "Matched checkpoint, quantization and maximum draft budget; implementations retain their adaptive allocation and proposal policies",
        .method = if (verify_only) "Isolated reference per request, followed by exact concurrent output/usage checks" else "ABBA (Python-only: phases 0 and 3); isolated warmup per request in each process; 3 repetitions per concurrency/draft cell; seeded mixed greedy/sampled SSE; 32 completion tokens per request; prompt caching disabled; throughput includes prefill and HTTP, excludes startup",
        .memory_scope = "Whole child-process lifetime peak RSS including shutdown; physical footprint sampled through the end of inference before shutdown; bytes",
        .memory = memory,
        .processes = bench.processes.items,
        .performance = performance,
        .python_reference = python_reference,
        .records = bench.records.items,
    }, .{ .whitespace = .indent_2 });
    try std.Io.Dir.cwd().writeFile(init.io, .{ .sub_path = args[5], .data = bytes });
    if (memory) |result| if (!result.passed) return error.NativeMemoryExceedsPython;
    if (python_reference) |result| {
        for (result.cells) |cell| std.debug.print("{s}: streams={d} draft={any}: Python {any} median completion tok/s ({d} samples)\n", .{ if (cell.valid) "VALID" else "INVALID", cell.streams, cell.draft, cell.python_median_tokens_per_second, cell.python_samples });
        if (!result.valid) return error.InvalidPythonReference;
    }
    if (performance) |result| {
        for (result.cells) |cell| std.debug.print("{s}: streams={d} draft={any}: native {any}, Python {any} median completion tok/s ({d}/{d} samples)\n", .{ if (cell.passed) "PASS" else "FAIL", cell.streams, cell.draft, cell.native_median_tokens_per_second, cell.python_median_tokens_per_second, cell.native_samples, cell.python_samples });
        if (!result.passed) return error.NativeSlowerThanPython;
    }
}

test "HTTP comparisons require matching stream capacities" {
    const a = std.testing.allocator;
    for ([_]usize{ 1, 2, 16, 32, 64 }) |streams| {
        var selected = Benchmark{};
        const json = try std.json.Stringify.valueAlloc(a, .{ .max_batch_size = streams }, .{});
        defer a.free(json);
        const status = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer status.deinit();
        try selected.configureCapacity(status.value);
        const counts = selected.streamCounts();
        try std.testing.expectEqual(@min(3, streams), counts.len);
        try std.testing.expectEqual(streams, counts[counts.len - 1]);
        for (counts[1..], counts[0 .. counts.len - 1]) |next, previous| try std.testing.expect(next > previous);
    }
    var benchmark_state = Benchmark{};
    const reference = try std.json.parseFromSlice(std.json.Value, a, "{\"max_batch_size\":64}", .{});
    defer reference.deinit();
    try benchmark_state.configureCapacity(reference.value);
    try std.testing.expectEqualSlices(usize, &.{ 1, 8, 64 }, &benchmark_state.concurrency);
    try benchmark_state.configureCapacity(reference.value);
    for ([_][]const u8{ "{\"max_batch_size\":8}", "{\"max_batch_size\":32}" }) |json| {
        const candidate = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer candidate.deinit();
        try std.testing.expectError(error.BenchmarkCapacityMismatch, benchmark_state.configureCapacity(candidate.value));
    }
    for ([_][]const u8{ "{\"max_batch_size\":0}", "{\"max_batch_size\":65}", "{\"max_batch_size\":64.0}", "{\"max_batch_size\":true}" }) |json| {
        const invalid = try std.json.parseFromSlice(std.json.Value, a, json, .{});
        defer invalid.deinit();
        try std.testing.expectError(error.InvalidBenchmarkCapacity, benchmark_state.configureCapacity(invalid.value));
    }
    try std.testing.expectError(error.MissingBenchmarkCapacity, benchmark_state.configureCapacity(.null));
    try std.testing.expectEqual(@as(usize, 64), benchmark_state.maxStreams());
}

test "benchmark responses require exact cached usage and 32 completion tokens" {
    const expected = Output{ .content = "response", .reasoning = "", .finish = "length", .usage = "{\"prompt_tokens\":4,\"completion_tokens\":32,\"total_tokens\":36,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}" };
    try Benchmark.validateOutput(expected);
    try expected.compareWithCache(expected, true);
    var actual = expected;
    actual.usage = "{\"prompt_tokens\":4,\"completion_tokens\":32,\"total_tokens\":36,\"prompt_tokens_details\":{\"cached_tokens\":1},\"completion_tokens_details\":{\"reasoning_tokens\":0}}";
    try std.testing.expectError(error.CachedUsageMismatch, expected.compareWithCache(actual, true));
    try expected.compare(actual);
    actual.usage = "{\"prompt_tokens\":4,\"completion_tokens\":31,\"total_tokens\":35,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}";
    try std.testing.expectError(error.InvalidBenchmarkCompletionCount, Benchmark.validateOutput(actual));
    actual.usage = "{\"prompt_tokens\":4,\"completion_tokens\":33,\"total_tokens\":37,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}";
    try std.testing.expectError(error.InvalidBenchmarkCompletionCount, Benchmark.validateOutput(actual));
    actual.usage = "{\"prompt_tokens\":4,\"completion_tokens\":32,\"total_tokens\":35,\"prompt_tokens_details\":{\"cached_tokens\":0},\"completion_tokens_details\":{\"reasoning_tokens\":0}}";
    try std.testing.expectError(error.InvalidUsage, Benchmark.validateOutput(actual));
    actual.usage = null;
    try std.testing.expectError(error.MissingUsage, Benchmark.validateOutput(actual));
}

test "whole-process memory gate rejects either peak exceeding Python" {
    var b = Benchmark{};
    defer b.processes.deinit(std.testing.allocator);
    const memory = Benchmark.ProcessMemory{ .rss_bytes = 80, .footprint_bytes = 90, .peak_footprint_bytes = 100 };
    try std.testing.expect(!b.memoryComparison().passed);
    for ([_][]const u8{ "python", "native" }) |variant| try b.processes.append(std.testing.allocator, .{ .variant = variant, .phase = 0, .startup = memory, .before_shutdown = memory, .peak_rss_bytes = 100 });
    try std.testing.expect(b.memoryComparison().passed);
    b.processes.items[1].peak_rss_bytes += 1;
    try std.testing.expect(!b.memoryComparison().passed);
    b.processes.items[1].peak_rss_bytes -= 1;
    b.processes.items[1].before_shutdown.peak_footprint_bytes += 1;
    try std.testing.expect(!b.memoryComparison().passed);
}

test "Python performance gate uses every concurrency and draft median" {
    const a = std.testing.allocator;
    var b = Benchmark{ .drafter = true };
    defer b.records.deinit(a);
    const rates = [_]f64{ 1000, 101, 1, 103, 100, 102 };
    for ([_][]const u8{ "python", "native" }) |variant| {
        for ([_]bool{ false, true }) |draft| {
            for (b.concurrency) |streams| {
                for (rates, 0..) |rate_value, i| try b.records.append(a, .{
                    .variant = variant,
                    .phase = i / 3,
                    .repetition = i % 3,
                    .streams = streams,
                    .draft = draft,
                    .tokens = 32 * streams,
                    .seconds = @as(f64, @floatFromInt(32 * streams)) / rate_value,
                    .tokens_per_second = rate_value,
                    .first_token_ms = &.{},
                    .latency_ms = &.{},
                    .shared_rounds = 0,
                    .max_shared_streams = 0,
                    .inference_before = .null,
                    .inference_after = .null,
                    .memory = .null,
                });
            }
        }
    }
    {
        const result = try b.performance(a);
        defer a.free(result.cells);
        try std.testing.expect(result.passed);
        try std.testing.expectEqual(6, result.cells.len);
        for (result.cells) |cell| {
            try std.testing.expectEqual(@as(?f64, 101.5), cell.native_median_tokens_per_second);
            try std.testing.expectEqual(@as(?f64, 101.5), cell.python_median_tokens_per_second);
        }
    }
    b.records.items[b.records.items.len - 1].tokens_per_second = 99;
    {
        const result = try b.performance(a);
        defer a.free(result.cells);
        try std.testing.expect(!result.passed);
        for (result.cells[0..5]) |cell| try std.testing.expect(cell.passed);
        try std.testing.expect(!result.cells[5].passed);
        try std.testing.expectEqual(@as(?f64, 100.5), result.cells[5].native_median_tokens_per_second);
    }
    b.records.items.len -= 1;
    const incomplete = try b.performance(a);
    defer a.free(incomplete.cells);
    try std.testing.expect(!incomplete.passed);
    try std.testing.expectEqual(5, incomplete.cells[5].native_samples);
    try std.testing.expectEqual(null, incomplete.cells[5].native_median_tokens_per_second);
}

test "Python reference requires six exact finite phase samples in every cell" {
    const a = std.testing.allocator;
    var b = Benchmark{ .drafter = true };
    defer b.records.deinit(a);
    for ([_]bool{ false, true }) |draft| {
        for (b.concurrency) |streams| {
            for ([_]f64{ 64, 32, 128, 8, 256, 16 }, 0..) |rate, i| try b.records.append(a, .{
                .variant = "python",
                .phase = if (i < 3) 0 else 3,
                .repetition = i % 3,
                .streams = streams,
                .draft = draft,
                .tokens = 32 * streams,
                .seconds = @as(f64, @floatFromInt(32 * streams)) / rate,
                .tokens_per_second = rate,
                .first_token_ms = &.{},
                .latency_ms = &.{},
                .shared_rounds = 0,
                .max_shared_streams = 0,
                .inference_before = .null,
                .inference_after = .null,
                .memory = .null,
            });
        }
    }
    for ([_]bool{ false, true }) |draft| {
        b.drafter = draft;
        b.records.items.len = if (draft) 36 else 18;
        const result = try b.pythonReference(a);
        defer a.free(result.cells);
        try std.testing.expect(result.valid);
        try std.testing.expectEqual(@as(usize, if (draft) 6 else 3), result.cells.len);
        for (result.cells) |cell| {
            try std.testing.expect(cell.valid);
            try std.testing.expectEqual(6, cell.python_samples);
            try std.testing.expectEqual(@as(?f64, 48), cell.python_median_tokens_per_second);
        }
    }
    const last = &b.records.items[b.records.items.len - 1];
    const saved = last.*;
    for ([_]f64{ 0, -1, std.math.nan(f64), std.math.inf(f64) }) |seconds| {
        last.* = saved;
        last.seconds = seconds;
        const invalid = try b.pythonReference(a);
        defer a.free(invalid.cells);
        try std.testing.expect(!invalid.valid);
        try std.testing.expectEqual(null, invalid.cells[5].python_median_tokens_per_second);
    }
    for ([_]usize{ 0, 1 }) |phase| {
        last.* = saved;
        last.phase = phase;
        const invalid = try b.pythonReference(a);
        defer a.free(invalid.cells);
        try std.testing.expect(!invalid.valid);
    }
    last.* = saved;
    last.tokens_per_second = 17;
    {
        const invalid = try b.pythonReference(a);
        defer a.free(invalid.cells);
        try std.testing.expect(!invalid.valid);
    }
    last.* = saved;
    b.records.items.len -= 1;
    const incomplete = try b.pythonReference(a);
    defer a.free(incomplete.cells);
    try std.testing.expect(!incomplete.valid);
    for (incomplete.cells[0..5]) |cell| try std.testing.expect(cell.valid);
    try std.testing.expectEqual(5, incomplete.cells[5].python_samples);
    try std.testing.expectEqual(null, incomplete.cells[5].python_median_tokens_per_second);
}

pub fn main(init: std.process.Init) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len > 1 and (std.mem.eql(u8, args[1], "--benchmark") or std.mem.eql(u8, args[1], "--compare-python") or std.mem.eql(u8, args[1], "--verify-python") or std.mem.eql(u8, args[1], "--benchmark-python"))) return benchmark(init, args);
    if (args.len != 3 and args.len != 5 and args.len != 6) return error.ExpectedExecutableAndModel;
    if (args.len == 5 and std.mem.eql(u8, args[3], "--warming-only")) {
        const a = init.arena.allocator();
        const root = args[4];
        std.Io.Dir.cwd().deleteTree(init.io, root) catch |err| if (err != error.FileNotFound) return err;
        const directory = try std.fs.path.join(a, &.{ root, "system" });
        var expected: ?Output = null;
        for (0..5) |phase| {
            if (phase == 1 or phase == 4) {
                try invalidateWarmSnapshots(init, directory);
                std.Io.Dir.cwd().deleteTree(init.io, try std.fs.path.join(a, &.{ root, "native-session-snapshots" })) catch |err| if (err != error.FileNotFound) return err;
            }
            var scenario = Scenario{ .init = init, .idle = false, .warming_phase = phase, .warming_expected = &expected, .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0", "--batch-streams", "1", "--snapshot-dir", directory, "--spill-gib", "1", "--checkpoint-slots", "1", "--prompt-cache-gib", if (phase == 3) "0" else "1" }, .stderr = .pipe }) };
            defer if (scenario.child.id) |id| {
                std.posix.kill(id, .KILL) catch {};
                scenario.child.kill(init.io);
            };
            const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
            var events: [2]Event = undefined;
            var select = std.Io.Select(Event).init(init.io, &events);
            defer select.cancelDiscard();
            try select.concurrent(.done, Scenario.run, .{&scenario});
            try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(300), .awake });
            switch (try select.await()) {
                .done => |result| try result,
                .timeout => return error.ServerWarmingCheckTimedOut,
            }
        }
        return;
    }
    if (args.len == 5 and std.mem.eql(u8, args[3], "--disk-only")) {
        const root = args[4];
        std.Io.Dir.cwd().deleteTree(init.io, root) catch |err| if (err != error.FileNotFound) return err;
        const directory = try std.fs.path.join(init.arena.allocator(), &.{ root, "system" });
        var expected: [2]?Output = @splat(null);
        for (0..4) |phase| {
            if (phase == 3) {
                var dir = try std.Io.Dir.cwd().openDir(init.io, root, .{ .iterate = true });
                defer dir.close(init.io);
                var walk = try dir.walk(init.arena.allocator());
                defer walk.deinit();
                while (try walk.next(init.io)) |entry| {
                    if (!std.mem.endsWith(u8, entry.path, ".safetensors")) continue;
                    const file = try dir.createFile(init.io, entry.path, .{});
                    file.close(init.io);
                }
            }
            var scenario = Scenario{ .init = init, .idle = false, .disk_phase = phase, .disk_expected = &expected, .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0", "--snapshot-dir", directory, "--spill-gib", "1", "--max-snapshots", if (phase == 2) "0" else "3", "--checkpoint-slots", "1", "--prompt-cache-gib", "1" }, .stderr = .pipe }) };
            defer if (scenario.child.id) |id| {
                std.posix.kill(id, .KILL) catch {};
                scenario.child.kill(init.io);
            };
            const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
            var events: [2]Event = undefined;
            var select = std.Io.Select(Event).init(init.io, &events);
            defer select.cancelDiscard();
            try select.concurrent(.done, Scenario.run, .{&scenario});
            try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(300), .awake });
            switch (try select.await()) {
                .done => |result| try result,
                .timeout => return error.ServerSnapshotCheckTimedOut,
            }
        }
        return;
    }
    if (args.len >= 5) {
        const memory = std.mem.eql(u8, args[3], "--memory-only");
        const prefixes = std.mem.eql(u8, args[3], "--cache-only");
        const live = std.mem.eql(u8, args[3], "--live-only");
        const drafts = std.mem.eql(u8, args[3], "--drafts-only");
        const responses = std.mem.eql(u8, args[3], "--responses-only");
        const background = std.mem.eql(u8, args[3], "--background-only");
        const neural = std.mem.eql(u8, args[3], "--neural-only") or std.mem.eql(u8, args[3], "--neural-disabled") or std.mem.eql(u8, args[3], "--neural-synthetic") or std.mem.eql(u8, args[3], "--neural-untrained");
        const flash = std.mem.eql(u8, std.fs.path.basename(args[2]), "Qwen3.8-Flash-Next-MLX-4bit-MTP");
        const terminal = if (live and !std.mem.eql(u8, args[4], "redirected")) try Terminal.init() else null;
        defer if (terminal) |t| {
            t.master.close(init.io);
            t.slave.close(init.io);
        };
        var environment = try init.environ_map.clone(init.arena.allocator());
        defer environment.deinit();
        if (memory) try environment.put("TENSORFOLD_MEMORY_LIMIT_GB", "70");
        if (live) {
            try environment.put("TENSORFOLD_NO_LIVE", if (std.mem.eql(u8, args[4], "disabled")) "1" else "0");
            try environment.put("COLUMNS", "0");
        }
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(init.arena.allocator(), &.{ args[1], "serve", args[2], "--snapshot-dir", "none" });
        try argv.appendSlice(init.arena.allocator(), &.{ "--port", "0", "--batch-streams", if (live) "1" else if (background) args[4] else "4", "--shutdown-grace-seconds", "1", "--checkpoint-slots", if (prefixes) "1" else "12", "--prompt-cache-gib", if (prefixes) args[4] else if (flash) "0" else "16" });
        if (neural) {
            try argv.appendSlice(init.arena.allocator(), &.{ "--max-draft", if (flash) "3" else "15" });
            if (!std.mem.eql(u8, args[4], "-")) try argv.appendSlice(init.arena.allocator(), &.{ "--drafter", args[4] });
            if (std.mem.eql(u8, args[3], "--neural-disabled")) try argv.append(init.arena.allocator(), "--no-drafts");
        }
        if (args.len == 6) try argv.appendSlice(init.arena.allocator(), &.{ "--drafter", args[5], "--max-draft", "15" });
        var scenario = Scenario{ .init = init, .idle = false, .rounds = !memory and !prefixes and !live, .memory = memory, .prefixes = prefixes, .live = live, .neural = neural, .neural_enabled = !std.mem.eql(u8, args[3], "--neural-disabled"), .terminal = terminal, .live_enabled = live and std.mem.eql(u8, args[4], "enabled"), .cache_enabled = !std.mem.eql(u8, args[4], "0"), .cache_oversize = std.mem.eql(u8, args[4], "0.000001"), .image = if (memory) args[4] else args[3], .http_checks = args[4], .child = try std.process.spawn(init.io, .{ .argv = argv.items, .environ_map = &environment, .stdout = if (terminal) |t| .{ .file = t.slave } else if (live) .pipe else .inherit, .stderr = .pipe }) };
        scenario.drafts = drafts;
        scenario.responses = responses;
        scenario.background = background;
        if (background) scenario.background_lanes = try std.fmt.parseInt(usize, args[4], 10);
        scenario.synthetic = std.mem.eql(u8, args[3], "--neural-synthetic");
        scenario.require_acceptance = !std.mem.eql(u8, args[3], "--neural-untrained");
        defer if (scenario.child.id) |id| {
            std.posix.kill(id, .KILL) catch {};
            scenario.child.kill(init.io);
        };
        const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
        var events: [2]Event = undefined;
        var select = std.Io.Select(Event).init(init.io, &events);
        defer select.cancelDiscard();
        try select.concurrent(.done, Scenario.run, .{&scenario});
        try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(300), .awake });
        switch (try select.await()) {
            .done => |result| try result,
            .timeout => return error.ServerRoundsCheckTimedOut,
        }
        return;
    }
    var environment = try init.environ_map.clone(init.arena.allocator());
    defer environment.deinit();
    for ([_][]const u8{ "nan", "0", "1" }) |budget| {
        try environment.put("TENSORFOLD_MEMORY_LIMIT_GB", budget);
        const result = try std.process.run(init.arena.allocator(), init.io, .{ .argv = &.{ args[1], "serve", args[2], "--port", "0" }, .environ_map = &environment, .stderr_limit = .limited(64 * 1024) });
        const expected = if (std.mem.eql(u8, budget, "1")) "InsufficientMemoryBudget" else "InvalidMemoryBudget";
        if (result.term.success() or std.mem.indexOf(u8, result.stderr, expected) == null or std.mem.indexOf(u8, result.stderr, "Loading ") != null or std.mem.indexOf(u8, result.stderr, "Native inference listening") != null) return error.InvalidMemoryBudgetLoadedModel;
    }
    std.debug.print("PASS: invalid/insufficient process budgets fail before model weights load\n", .{});
    for ([_]bool{ false, true }) |idle| {
        var scenario = Scenario{ .init = init, .idle = idle, .child = try std.process.spawn(init.io, .{ .argv = &.{ args[1], "serve", args[2], "--snapshot-dir", "none", "--port", "0", "--request-timeout-seconds", if (idle) "0" else "2", "--shutdown-grace-seconds", "1", "--no-thinking" }, .stderr = .pipe }) };
        defer if (scenario.child.id) |id| {
            std.posix.kill(id, .KILL) catch {};
            scenario.child.kill(init.io);
        };
        const Event = union(enum) { done: anyerror!void, timeout: std.Io.Cancelable!void };
        var events: [2]Event = undefined;
        var select = std.Io.Select(Event).init(init.io, &events);
        defer select.cancelDiscard();
        try select.concurrent(.done, Scenario.run, .{&scenario});
        try select.concurrent(.timeout, std.Io.sleep, .{ init.io, std.Io.Duration.fromSeconds(90), .awake });
        switch (try select.await()) {
            .done => |result| try result,
            .timeout => return error.ServerLifecycleCheckTimedOut,
        }
    }
}
