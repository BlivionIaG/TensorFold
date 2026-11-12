const std = @import("std");
const cli = @import("src/server/cli.zig");
const api = @import("engine_api");

test "FSR CLI forwards explicit budgets and refuses unsupported backends and families before loading" {
    if (@import("builtin").os.tag != .macos) return error.SkipZigTest;
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const plain = try cli.request(a, "unused", "qwen4_exp", try cli.parse(a, &.{"model"}, &usage));
    try std.testing.expect(plain.ssd_experts_gib == null and !plain.ple_on_ssd and plain.fsr_trace == null);
    const args = try cli.parse(a, &.{ "model", "--ssd-experts=16.5", "--ple-on-ssd", "--fsr-route-trace", "route.jsonl" }, &usage);
    const on = try cli.request(a, "unused", "qwen4_exp", args);
    try std.testing.expectEqual(@as(?f64, 16.5), on.ssd_experts_gib);
    try std.testing.expect(on.ple_on_ssd);
    try std.testing.expectEqualStrings("route.jsonl", on.fsr_trace.?);
    try std.testing.expect(on.fsrRefusal("metal") == null);
    for ([_][]const u8{ "cuda", "hip", "xpu" }) |backend| try std.testing.expect(on.fsrRefusal(backend) != null);
    var other = on;
    other.model_type = "glm5_next";
    try std.testing.expect(other.fsrRefusal("metal") != null);
    other = on;
    other.speed_up = "pair";
    try std.testing.expect(other.fsrRefusal("metal") != null);
    for ([_][]const u8{ "0", "-1", "nan", "inf", "1e20", "bad" }) |bad| {
        try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", "--ssd-experts", bad }, &usage));
        other = on;
        other.ssd_experts_gib = std.fmt.parseFloat(f64, bad) catch std.math.nan(f64);
        try std.testing.expect(other.fsrRefusal("metal") != null);
    }
    try std.testing.expectError(error.Usage, cli.parse(a, &.{ "model", "--fsr-route-trace=" }, &usage));
    _ = api;
}

test "a HIP Qwen3.5 MoE takes --ssd-experts alone with a finite budget, and no other flag, family or backend does" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var usage: cli.Usage = .{};
    const moe = try cli.request(a, "unused", "qwen3_5_moe", try cli.parse(a, &.{ "model", "--ssd-experts=8" }, &usage));
    try std.testing.expect(moe.fsrRefusal("hip") == null);
    try std.testing.expect(moe.fsrRefusal("cuda") != null);
    var other = moe;
    other.ssd_experts_gib = 0;
    try std.testing.expect(other.fsrRefusal("hip") != null);
    other = moe;
    other.ple_on_ssd = true;
    try std.testing.expect(other.fsrRefusal("hip") != null);
    other = moe;
    other.fsr_trace = "route.jsonl";
    try std.testing.expect(other.fsrRefusal("hip") != null);
    other = moe;
    other.model_type = "qwen3_5";
    try std.testing.expect(other.fsrRefusal("hip") != null);
}
