//! The control vector's serve checks: the file, the reader, the family, the checkpoint's own numbers and the line
//! that reports what is in force. Ported from the #1835 startup tests (head c5947a8d2e, BlivionIaG) onto DeepSeek-V4.1.

const std = @import("std");
const builtin = @import("builtin");
const glp = @import("engine_api").glp;
const serve = @import("glp_startup.zig");
const log = @import("log.zig");

const io = std.testing.io;

/// The text model's hidden size and backbone depth DeepSeek-V4.1 has; its last block is layer 39.
const hidden = 5120;
const depth = 40;

/// Whether this platform has an engine that steers: Metal on a Mac, CUDA on Linux.
const steers = builtin.os.tag == .macos or builtin.os.tag == .linux;

/// The lines the server printed, for the report the operator reads.
const Sink = struct {
    list: std.ArrayList(u8) = .empty,
    a: std.mem.Allocator,

    fn lines(self: Sink) usize {
        return std.mem.count(u8, self.list.items, "\n");
    }

    fn has(self: Sink, needle: []const u8) bool {
        return std.mem.indexOf(u8, self.list.items, needle) != null;
    }
};

fn capture(ctx: *anyopaque, text: []const u8) void {
    const sink: *Sink = @ptrCast(@alignCast(ctx));
    sink.list.appendSlice(sink.a, text) catch {};
}

/// What the checkpoint's config.json holds, so a test can bend one number at a time.
const Config = struct {
    hidden: usize = hidden,
    layers: usize = depth,
    hc: ?usize = 4,
    /// A vision tower's numbers beside the text model's, which a reader must never take for the model's.
    decoy: bool = true,
};

/// What a vector file holds.
const Opts = struct {
    width: usize = hidden,
    ids: []const u32 = &.{ 1, 2 },
    mode: glp.Mode = .project,
    /// The strength the file carries as its own.
    alpha: f32 = 1.0,
    /// The first component of every direction: 1 is a unit vector; an add vector may be longer.
    peak: f32 = 1.0,
    base_model: []const u8 = "",
    base_revision: []const u8 = "",
    derived_at: ?[]const u8 = null,
    config: Config = .{},
    /// The strength the operator gave.
    alpha_arg: ?f32 = null,
};

/// A checkpoint folder with a config.json, a vector file beside it and the vector read.
const Fixture = struct {
    tmp: std.testing.TmpDir,
    /// The model's folder, as the serve path names it.
    model: []const u8,
    /// The vector the file holds, read once.
    g: serve.Vec,

    fn make(a: std.mem.Allocator, o: Opts) !Fixture {
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        const model = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..] });
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = try configJson(a, o.config) });
        try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = try vectorBytes(a, o) });
        const path = try std.fs.path.join(a, &.{ model, "vector.gguf" });
        var g: ?serve.Vec = serve.withPath(null, path);
        if (o.alpha_arg) |x| g = serve.withAlpha(g, x);
        if (try serve.read(a, &g)) |why| {
            std.debug.print("refused: {s}\n", .{why});
            return error.UnexpectedRefusal;
        }
        return .{ .tmp = tmp, .model = model, .g = g.? };
    }

    fn setup(self: Fixture, model_type: []const u8, backend: []const u8) serve.Setup {
        return .{ .dir = self.model, .model_type = model_type, .backend = backend };
    }
};

/// The DeepSeek-V4.1 config shape: the text model's numbers in text_config, a vision tower's beside it.
fn configJson(a: std.mem.Allocator, c: Config) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.writeAll("{\"model_type\":\"deepseek_v41\"");
    if (c.decoy) try w.writer.writeAll(",\"vision_config\":{\"hidden_size\":1024,\"num_hidden_layers\":32}");
    try w.writer.print(",\"text_config\":{{\"model_type\":\"deepseek_v41_text\",\"hidden_size\":{d},\"num_hidden_layers\":{d}", .{ c.hidden, c.layers });
    if (c.hc) |hc| try w.writer.print(",\"hc_mult\":{d}", .{hc});
    try w.writer.writeAll("}}");
    return w.toOwnedSlice();
}

/// A vector file with one direction per layer, each along the first axis.
fn vectorBytes(a: std.mem.Allocator, o: Opts) ![]u8 {
    const layers = try a.alloc(glp.Layer, o.ids.len);
    for (o.ids, 0..) |id, i| {
        const dir = try a.alloc(f32, o.width);
        @memset(dir, 0);
        dir[0] = o.peak;
        layers[i] = .{ .id = id, .dir = dir };
    }
    return glp.write(a, .{ .mode = o.mode, .alpha = o.alpha, .base_model = o.base_model, .base_revision = o.base_revision, .derived_at = o.derived_at, .layers = layers });
}

/// The reason the file was refused, as the operator reads it.
fn refused(a: std.mem.Allocator, g: *?serve.Vec) ![]const u8 {
    return (try serve.read(a, g)) orelse return error.UnexpectedlyAccepted;
}

fn contains(text: []const u8, needle: []const u8) bool {
    return std.mem.indexOf(u8, text, needle) != null;
}

/// Why the vector in `f` is refused on this platform's own engine; null when it is taken.
fn refusalOf(a: std.mem.Allocator, f: Fixture) !?[]const u8 {
    return serve.refusal(a, f.setup("deepseek_v41", "auto"), serve.factsOf(a, f.model), f.g);
}

test "a missing file is refused with the path and the reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g: ?serve.Vec = serve.withPath(null, ".zig-cache/tmp/no-such-vector.gguf");
    const why = try refused(a, &g);
    try std.testing.expect(contains(why, "argument --glp:"));
    try std.testing.expect(contains(why, "cannot read"));
    try std.testing.expect(contains(why, "no such file"));
    try std.testing.expect(contains(why, "no-such-vector.gguf"));
}

test "a file the reader refuses is refused with the reader's reason" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "vector.gguf", .data = "not a control vector" });
    const path = try std.fs.path.join(a, &.{ ".zig-cache", "tmp", tmp.sub_path[0..], "vector.gguf" });
    var g: ?serve.Vec = serve.withPath(null, path);
    const why = try refused(a, &g);
    try std.testing.expect(contains(why, "argument --glp:"));
    try std.testing.expect(contains(why, "not a GGUF container"));
}

test "--glp-alpha without --glp is refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var g: ?serve.Vec = serve.withAlpha(null, 0.5);
    try std.testing.expect(contains(try refused(a, &g), "needs --glp"));
}

test "the strength is a finite number from 0, and 0 is asked for as 0" {
    try std.testing.expectEqual(@as(?f32, 0.5), serve.strength("0.5"));
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(serve.strength("0").?)));
    try std.testing.expectEqual(@as(u32, 0), @as(u32, @bitCast(serve.strength("-0").?)));
    for ([_][]const u8{ "-1", "inf", "-inf", "nan", "1e39", "1e100", "1e-100", "1e-40", "abc", "" }) |text| {
        try std.testing.expect(serve.strength(text) == null);
    }
}

test "a strength the operator gave that the engine could not hold is refused at the file" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    var g: ?serve.Vec = serve.withAlpha(serve.withPath(null, f.g.path.?), -1.0);
    try std.testing.expect(contains(try refused(a, &g), "strength"));
}

test "the strength in force is the operator's, else the file's, and 0 is a strength" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var own = try Fixture.make(a, .{ .alpha = 0.5 });
    defer own.tmp.cleanup();
    try std.testing.expectEqual(@as(f32, 0.5), serve.engine(own.g).?.alpha);
    var given = try Fixture.make(a, .{ .alpha = 0.5, .alpha_arg = 0.25 });
    defer given.tmp.cleanup();
    const g = serve.engine(given.g).?;
    try std.testing.expectEqual(@as(f32, 0.25), g.alpha);
    try std.testing.expectEqual(@as(usize, 2), g.vector.layers.len);
    try std.testing.expect(!g.off());
    var zero = try Fixture.make(a, .{ .alpha = 0.5, .alpha_arg = 0 });
    defer zero.tmp.cleanup();
    // zero is loaded, not dropped: the engine is given the vector and bypasses every edit
    try std.testing.expect(serve.engine(zero.g).?.off());
    try std.testing.expect(serve.engine(null) == null);
    try std.testing.expect(serve.engine(serve.withPath(null, "x")) == null);
}

test "a good vector is accepted where an engine steers, with or without the vision decoy" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    for ([_]bool{ true, false }) |decoy| {
        var f = try Fixture.make(a, .{ .config = .{ .decoy = decoy }, .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017" });
        defer f.tmp.cleanup();
        try std.testing.expect((try refusalOf(a, f)) == null);
    }
}

test "the checkpoint's numbers are the text model's, never the vision tower's" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    const facts = serve.factsOf(a, f.model);
    try std.testing.expectEqual(@as(?usize, hidden), facts.model.hidden);
    try std.testing.expectEqual(@as(?usize, depth), facts.model.layers);
    try std.testing.expectEqual(@as(?usize, 4), facts.copies);
    // a layer the vision tower's 32 blocks would refuse is a backbone layer here
    if (!steers) return;
    var deep = try Fixture.make(a, .{ .ids = &.{ 1, 39 } });
    defer deep.tmp.cleanup();
    try std.testing.expect((try refusalOf(a, deep)) == null);
}

test "a checkpoint whose config gives no text size or depth cannot be checked" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    try f.tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"deepseek_v41\"}" });
    const why = (try refusalOf(a, f)).?;
    try std.testing.expect(contains(why, "no text hidden size and depth"));
}

test "a family that is not DeepSeek-V4.1 is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    for ([_][]const u8{ "glm5_next", "gemma4", "qwen3_5", "deepseek_v32" }) |family| {
        const why = (try serve.refusal(a, f.setup(family, "auto"), serve.factsOf(a, f.model), f.g)).?;
        try std.testing.expect(contains(why, family));
        try std.testing.expect(contains(why, "DeepSeek-V4.1"));
    }
}

test "an engine that does not steer is refused by name" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    const facts = serve.factsOf(a, f.model);
    const rocm = (try serve.refusal(a, f.setup("deepseek_v41", "rocm"), facts, f.g)).?;
    try std.testing.expect(contains(rocm, "ROCm"));
    // the engine of the other platform is not this one's
    const other = if (builtin.os.tag == .macos) "cuda" else "mlx";
    const why = (try serve.refusal(a, f.setup("deepseek_v41", other), facts, f.g)).?;
    try std.testing.expect(contains(why, if (builtin.os.tag == .macos) "CUDA" else "Metal"));
}

test "prompt states kept on disk are refused, for they are not keyed by the vector" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    var s = f.setup("deepseek_v41", "auto");
    s.persists = true;
    const why = (try serve.refusal(a, s, serve.factsOf(a, f.model), f.g)).?;
    try std.testing.expect(contains(why, "keyed by the vector"));
    try std.testing.expect(contains(why, "--learn"));
    try std.testing.expect(contains(why, "--spill-gib"));
}

test "a vector the model's width does not match is refused" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .width = 4096 });
    defer f.tmp.cleanup();
    const why = (try refusalOf(a, f)).?;
    try std.testing.expect(contains(why, "4096"));
    try std.testing.expect(contains(why, "5120"));
}

test "the backbone's last layer is steered and the first DSpark block is not" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var last = try Fixture.make(a, .{ .ids = &.{ 1, 39 } });
    defer last.tmp.cleanup();
    try std.testing.expect((try refusalOf(a, last)) == null);
    var mtp = try Fixture.make(a, .{ .ids = &.{ 1, 40 } });
    defer mtp.tmp.cleanup();
    const why = (try refusalOf(a, mtp)).?;
    try std.testing.expect(contains(why, "layer 40"));
    try std.testing.expect(contains(why, "40 backbone layers"));
    try std.testing.expect(contains(why, "deepest steerable layer is 39"));
}

test "a checkpoint that does not carry four stream copies is refused" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var two = try Fixture.make(a, .{ .config = .{ .hc = 2 } });
    defer two.tmp.cleanup();
    const why = (try refusalOf(a, two)).?;
    try std.testing.expect(contains(why, "4 stream copies"));
    try std.testing.expect(contains(why, "carries 2"));
    // a config that names no count is taken at the family's four
    var none = try Fixture.make(a, .{ .config = .{ .hc = null } });
    defer none.tmp.cleanup();
    try std.testing.expect((try refusalOf(a, none)) == null);
}

test "an additive strength BF16 cannot hold is refused at the file and again at admission, a projection is not" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    // the file's own strength keeps its largest component inside BF16; the operator's twice that does not
    var add = try Fixture.make(a, .{ .mode = .add, .peak = 3.0e38 });
    defer add.tmp.cleanup();
    var g: ?serve.Vec = serve.withAlpha(serve.withPath(null, add.g.path.?), 2.0);
    try std.testing.expect(contains(try refused(a, &g), "BF16"));
    // admission checks the strength in force again, whatever read the vector
    var forced = add.g;
    forced.alpha = 2.0;
    try std.testing.expect(contains((try serve.refusal(a, add.setup("deepseek_v41", "auto"), serve.factsOf(a, add.model), forced)).?, "BF16"));
    var project = try Fixture.make(a, .{ .alpha_arg = 1000.0 });
    defer project.tmp.cleanup();
    try std.testing.expect((try refusalOf(a, project)) == null);
}

test "a hub snapshot's repo and commit bind the vector's checkpoint" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const hub = "/cache/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots/abc123def456";
    var facts = serve.factsOf(a, hub);
    try std.testing.expectEqualStrings("deepseek-ai/DeepSeek-V4.1-Flash-1017", facts.model.repo);
    try std.testing.expectEqualStrings("abc123def456", facts.model.revision);
    // the folder holds no config here: the numbers come from the checkpoint's own, so they are given
    facts.model.hidden = hidden;
    facts.model.layers = depth;
    facts.copies = 4;
    const setup: serve.Setup = .{ .dir = hub, .model_type = "deepseek_v41", .backend = "auto" };
    var same = try Fixture.make(a, .{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017", .base_revision = "abc123def456" });
    defer same.tmp.cleanup();
    try std.testing.expect((try serve.refusal(a, setup, facts, same.g)) == null);
    var other = try Fixture.make(a, .{ .base_model = "Qwen/Qwen3.8-27B" });
    defer other.tmp.cleanup();
    try std.testing.expect(contains((try serve.refusal(a, setup, facts, other.g)).?, "the vector is for Qwen/Qwen3.8-27B"));
    var moved = try Fixture.make(a, .{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017", .base_revision = "0123456789ab" });
    defer moved.tmp.cleanup();
    try std.testing.expect(contains((try serve.refusal(a, setup, facts, moved.g)).?, "revision"));
    // a similarly named repo is not the same checkpoint, whichever name is the longer
    var shorter = try Fixture.make(a, .{ .base_model = "deepseek-ai/DeepSeek-V4.1-Flash" });
    defer shorter.tmp.cleanup();
    try std.testing.expect(contains((try serve.refusal(a, setup, facts, shorter.g)).?, "the vector is for deepseek-ai/DeepSeek-V4.1-Flash;"));
    var fork = facts;
    fork.model.repo = "someone/DeepSeek-V4.1-Flash-1017-abliterated";
    try std.testing.expect(contains((try serve.refusal(a, setup, fork, same.g)).?, "the model is someone/DeepSeek-V4.1-Flash-1017-abliterated"));
    var converted = facts;
    converted.model.repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017-EXL3";
    try std.testing.expect((try serve.refusal(a, setup, converted, same.g)) != null);
}

test "the report says steered for a nonzero strength and inactive for 0" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var on = try Fixture.make(a, .{ .alpha = 2.5 });
    defer on.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, on.g, serve.factsOf(a, on.model));
    try std.testing.expect(sink.has("glp on: replies are steered"));
    try std.testing.expect(!sink.has("inactive"));
    try std.testing.expect(sink.has("project vector in"));
    try std.testing.expect(sink.has("hook: hc_fold_pre_engram"));
    try std.testing.expect(sink.has("alpha: 2.5"));
    try std.testing.expect(sink.has("source repo and revision unknown (not a hub snapshot)"));
    try std.testing.expect(sink.has("not the vector's base"));
    const sha = std.fmt.bytesToHex(on.g.vector.?.file_sha256, .lower);
    try std.testing.expect(sink.has(sha[0..]));
    try std.testing.expectEqual(@as(usize, 1), sink.lines());

    var off = try Fixture.make(a, .{ .alpha = 2.5, .alpha_arg = 0 });
    defer off.tmp.cleanup();
    sink.list.clearRetainingCapacity();
    serve.report(a, off.g, serve.factsOf(a, off.model));
    try std.testing.expect(sink.has("glp loaded but inactive"));
    try std.testing.expect(sink.has("replies are unsteered"));
    try std.testing.expect(!sink.has("glp on:"));
    try std.testing.expect(!sink.has("replies are steered"));
    try std.testing.expect(sink.has("alpha: 0"));
    try std.testing.expectEqual(@as(usize, 1), sink.lines());
}

test "the report names an add vector's mode and a hub snapshot's commit" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .mode = .add, .peak = 0.25, .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017", .base_revision = "abc123" });
    defer f.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    var facts = serve.factsOf(a, f.model);
    facts.model.repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
    facts.model.revision = "abc123def456";
    serve.report(a, f.g, facts);
    try std.testing.expect(sink.has("add vector in"));
    try std.testing.expect(sink.has("base: deepseek-ai/DeepSeek-V4.1-Flash-1017 @ abc123"));
    try std.testing.expect(sink.has("checkpoint: deepseek-ai/DeepSeek-V4.1-Flash-1017 @ abc123def456"));
    try std.testing.expect(sink.has("layers: 2 (1-2)"));
}

test "the line shows the file name and the file's own words escaped, so none can add a line" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .base_model = "x/y\nglp off: \"z\"", .derived_at = "site\nwarning: forged" });
    defer f.tmp.cleanup();
    var named = f.g;
    named.path = "dir/a\nglp off: \"x\".gguf";
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, named, serve.factsOf(a, f.model));
    try std.testing.expect(sink.has("dir/a\\nglp off: \\\"x\\\".gguf"));
    try std.testing.expect(sink.has("x/y\\nglp off: \\\"z\\\""));
    try std.testing.expect(sink.has("site\\nwarning: forged"));
    // the report line and the transfer warning, and nothing a file name or a field made up
    try std.testing.expectEqual(@as(usize, 2), sink.lines());
}

test "a vector derived at another site is reported as a warning, not refused" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{ .derived_at = "ffn_out_pre_residual" });
    defer f.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    serve.report(a, f.g, serve.factsOf(a, f.model));
    try std.testing.expect(sink.has("warning:"));
    try std.testing.expect(sink.has("ffn_out_pre_residual"));
    if (steers) try std.testing.expect((try refusalOf(a, f)) == null);
}

test "admission reports once on the serving rank and stays quiet on a follower" {
    if (!steers) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var f = try Fixture.make(a, .{});
    defer f.tmp.cleanup();
    var sink: Sink = .{ .a = a };
    log.Test.sink = .{ .ctx = @ptrCast(&sink), .call = capture };
    defer log.Test.sink = null;
    try std.testing.expect((try serve.admit(a, f.setup("deepseek_v41", "auto"), f.g)) == null);
    try std.testing.expectEqual(@as(usize, 1), sink.lines());
    sink.list.clearRetainingCapacity();
    var follower = f.setup("deepseek_v41", "auto");
    follower.follower = true;
    try std.testing.expect((try serve.admit(a, follower, f.g)) == null);
    try std.testing.expectEqual(@as(usize, 0), sink.lines());
    // a refusal prints nothing: the caller prints the reason
    try std.testing.expect((try serve.admit(a, f.setup("gemma4", "auto"), f.g)) != null);
    try std.testing.expectEqual(@as(usize, 0), sink.lines());
}
