//! Which engine a qwen3_5 checkpoint opens on Macs: the 27B's, or the pinned 2B's.
const std = @import("std");
const tf = @import("tensorfold");
const Allocator = std.mem.Allocator;

/// The 27B engine, unless the checkpoint has the pinned 2B's hidden size (the 2B's engine then), or ties its head and
/// does not say its hidden size (the 2B's engine, as before). A tied head of another size is this engine's.
pub fn owns(a: Allocator, io: std.Io, dir: []const u8) bool {
    const path = std.fs.path.join(a, &.{ dir, "config.json" }) catch return true;
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(16 << 20)) catch return true;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return true;
    if (doc != .object) return true;
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t else doc) else doc;
    var tied = false;
    var sized = false;
    for ([_]std.json.Value{ doc, text }) |o| {
        if (o.object.get("tie_word_embeddings")) |v| tied = tied or (v == .bool and v.bool);
        if (o.object.get("hidden_size")) |v| if (v == .integer) {
            if (v.integer == tf.qwen35.config.hidden) return false;
            sized = true;
        };
    }
    return sized or !tied;
}

test "a qwen3_5 checkpoint opens the 27B engine unless it has the 2B's hidden size or a tied head of no stated size" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try std.fmt.allocPrint(a, ".zig-cache/tmp/{s}", .{tmp.sub_path});
    try std.testing.expect(owns(a, io, dir)); // no config.json: the 27B's loader explains the refusal
    const cases = [_]struct { config: []const u8, big: bool }{
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":false,\"text_config\":{\"tie_word_embeddings\":false}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"hidden_size\":5120}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":4096}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":false,\"hidden_size\":2048}}", .big = false },
        // a tied head is the 2B's only at the 2B's width: the 4B (2,560 wide, tied) is this engine's, at either nesting
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"text_config\":{\"hidden_size\":2560}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true,\"hidden_size\":2560}}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"hidden_size\":4096}", .big = true },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"text_config\":{\"tie_word_embeddings\":true,\"hidden_size\":2048}}", .big = false },
        .{ .config = "{\"model_type\":\"qwen3_5\",\"tie_word_embeddings\":true,\"hidden_size\":2048,\"text_config\":{\"hidden_size\":2560}}", .big = false },
    };
    for (cases) |c| {
        try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = c.config });
        try std.testing.expectEqual(c.big, owns(a, io, dir));
    }
}
