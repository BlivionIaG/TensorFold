//! GLPv3 publication metadata checks, ported from #1835 c5947a8d2e.

const std = @import("std");
const glp = @import("glp.zig");
const t = @import("glp_fixture.zig");

test "a file that names itself carries what it said, and the model it came from" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{
        .mode = .project,
        .base_model = "TensorFold/DeepSeek-V4.1-Flash-sample",
        .base_revision = "22d8d538154e0e5b6f8dcb1fcc74a60a0798104c",
        .publication = .{
            .name = "DeepSeek-V4.1-Flash-sample-glp-L10-a1",
            .description = "A sample control vector.",
            .author = "TensorFold",
            .license = "Apache-2.0",
            .model_hint = "deepseek_v41",
            .measured_effect = "a sample note on the strength",
            .created = "2026-10-27",
        },
        .layers = &layers,
    });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    const p = read.vector.publication;
    try std.testing.expectEqualStrings("DeepSeek-V4.1-Flash-sample-glp-L10-a1", p.name.?);
    try std.testing.expectEqualStrings("A sample control vector.", p.description.?);
    try std.testing.expectEqualStrings("TensorFold", p.author.?);
    try std.testing.expectEqualStrings("Apache-2.0", p.license.?);
    try std.testing.expectEqualStrings("deepseek_v41", p.model_hint.?);
    try std.testing.expectEqualStrings("a sample note on the strength", p.measured_effect.?);
    try std.testing.expectEqualStrings("2026-10-27", p.created.?);

    // The container itself carries the keys a consumer reads, not only the reader.
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expectEqualStrings("DeepSeek-V4.1-Flash-sample-glp-L10-a1", parsed.file.str("general.name").?);
    try std.testing.expectEqualStrings("TensorFold", parsed.file.str("general.author").?);
    try std.testing.expectEqualStrings("deepseek_v41", parsed.file.str("controlvector.model_hint").?);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("controlvector.layer_count").?);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("general.base_model.count").?);
    try std.testing.expectEqualStrings("DeepSeek-V4.1-Flash-sample", parsed.file.str("general.base_model.0.name").?);
    try std.testing.expectEqualStrings("TensorFold", parsed.file.str("general.base_model.0.organization").?);
    try std.testing.expectEqualStrings("22d8d538154e0e5b6f8dcb1fcc74a60a0798104c", parsed.file.str("general.base_model.0.version").?);
    try std.testing.expectEqualStrings("https://huggingface.co/TensorFold/DeepSeek-V4.1-Flash-sample", parsed.file.str("general.base_model.0.repo_url").?);
}

test "a vector written again keeps what the file said about itself" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{
        .mode = .add,
        .base_model = "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        .publication = .{ .name = "a name", .author = "an author" },
        .layers = &layers,
    });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    const again = try glp.writeVector(a, read.vector);
    defer a.free(again);
    var reread = try t.readOk(again);
    defer reread.deinit();
    try std.testing.expectEqualStrings("a name", reread.vector.publication.name.?);
    try std.testing.expectEqualStrings("an author", reread.vector.publication.author.?);
    try std.testing.expectEqual(glp.Mode.add, reread.vector.mode);
}

test "a file that names no model of its own keeps the base model out of the header" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .mode = .project, .layers = &layers });
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    try std.testing.expect(parsed.file.str("general.base_model.0.repo_url") == null);
    try std.testing.expectEqual(@as(u64, 1), parsed.file.num("controlvector.layer_count").?);
}

test "a file that names no other site names the site it is applied at" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .mode = .project, .layers = &layers });
    defer a.free(bytes);
    var parsed = try t.parseFile(a, bytes);
    defer parsed.deinit();
    // glp.derived_at is the site the direction was captured at, not a time.
    try std.testing.expectEqualStrings("hc_fold_pre_engram", parsed.file.str("glp.derived_at").?);
    var read = try t.readOk(bytes);
    defer read.deinit();
    try std.testing.expect(!read.vector.transferred);
}

test "a vector derived at another site is a warning, not a refusal" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    const bytes = try glp.write(a, .{ .mode = .project, .derived_at = "ffn_out_pre_residual", .layers = &layers });
    defer a.free(bytes);
    var read = try t.readOk(bytes);
    defer read.deinit();
    try std.testing.expect(read.vector.transferred);
    try std.testing.expectEqualStrings("ffn_out_pre_residual", read.vector.derived_at);
    try std.testing.expectEqualStrings("hc_fold_pre_engram", read.vector.hook_point);
}

test "a metadata string of another type is refused, not skipped" {
    try t.expectRefused(.{ .extra = &.{.{ .key = "general.name", .value = .{ .u32v = 7 } }} }, "general.name is not text");
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.created", .value = .{ .f32v = 1.5 } }} }, "glp.created is not text");
    try t.expectRefused(.{ .extra = &.{.{ .key = "glp.derived_at", .value = .{ .i32v = -1 } }} }, "glp.derived_at is not text");
}

const repo = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
const commit = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0";

/// The spec's provenance keys one at a time, as a producer following docs/spec/GLP.md would write them.
const count_one: t.Extra = .{ .key = "general.base_model.count", .value = .{ .u32v = 1 } };
const repo_url: t.Extra = .{ .key = "general.base_model.0.repo_url", .value = .{ .text = "https://huggingface.co/" ++ repo } };
const org: t.Extra = .{ .key = "general.base_model.0.organization", .value = .{ .text = "deepseek-ai" } };
const base_name: t.Extra = .{ .key = "general.base_model.0.name", .value = .{ .text = "DeepSeek-V4.1-Flash-1017" } };
const pin: t.Extra = .{ .key = "general.base_model.0.version", .value = .{ .text = commit } };

fn model(revision: []const u8, name: []const u8) glp.Model {
    return .{ .hidden = 8, .layers = 40, .repo = name, .revision = revision };
}

test "the spec's commit pin binds a vector to one revision of its repo" {
    // a file with the spec's repo_url and .0.version and nothing of ours: the pin must still be read and checked
    var got = try t.accept(.{ .base_model = null, .extra = &.{ count_one, repo_url, pin } });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const v = got.read.vector;
    try std.testing.expectEqualStrings(repo, v.base_model);
    try std.testing.expectEqualStrings(commit, v.base_revision);
    var why: glp.Why = .{};
    try glp.checkModel(v, model(commit, repo), &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, model("ffffffffffffffffffffffffffffffffffffffff", repo), &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "revision") != null);
    // a checkpoint that names no commit is not compared by commit, and one that is not this repo is refused by name
    try glp.checkModel(v, model("", repo), &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, model(commit, "deepseek-ai/DeepSeek-V4.1-Flash-0731"), &why));
}

test "organization and name name the repo when the file gives no URL" {
    var got = try t.accept(.{ .base_model = null, .extra = &.{ count_one, org, base_name, pin } });
    defer std.testing.allocator.free(got.bytes);
    defer got.read.deinit();
    const v = got.read.vector;
    try std.testing.expectEqualStrings(repo, v.base_model);
    try std.testing.expectEqualStrings(commit, v.base_revision);
    var why: glp.Why = .{};
    try glp.checkModel(v, model(commit, repo), &why);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(v, model("", "deepseek-ai/DeepSeek-V4.1-Flash-0731"), &why));
    try std.testing.expect(std.mem.indexOf(u8, why.text(), "the vector is for deepseek-ai/DeepSeek-V4.1-Flash-1017") != null);
}

test "a name with no owner and a URL that is not the hub's cannot be the same repo as an owner's" {
    var bare = try t.accept(.{ .base_model = null, .extra = &.{ count_one, base_name } });
    defer std.testing.allocator.free(bare.bytes);
    defer bare.read.deinit();
    try std.testing.expectEqualStrings("DeepSeek-V4.1-Flash-1017", bare.read.vector.base_model);
    var why: glp.Why = .{};
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(bare.read.vector, model("", repo), &why));
    // nothing is known of the served checkpoint's repo, so nothing is compared
    try glp.checkModel(bare.read.vector, model("", ""), &why);

    const elsewhere: t.Extra = .{ .key = "general.base_model.0.repo_url", .value = .{ .text = "https://example.com/x/DeepSeek-V4.1-Flash-1017" } };
    var other = try t.accept(.{ .base_model = null, .extra = &.{ count_one, elsewhere } });
    defer std.testing.allocator.free(other.bytes);
    defer other.read.deinit();
    try std.testing.expectEqualStrings("https://example.com/x/DeepSeek-V4.1-Flash-1017", other.read.vector.base_model);
    try std.testing.expectError(error.BadGlpVector, glp.checkModel(other.read.vector, model("", repo), &why));
}

test "base fields that disagree or name more than one base are refused" {
    const other_org: t.Extra = .{ .key = "general.base_model.0.organization", .value = .{ .text = "someone" } };
    try t.expectRefused(.{ .base_model = null, .extra = &.{ count_one, repo_url, other_org, base_name } }, "organization and name");
    const two: t.Extra = .{ .key = "general.base_model.count", .value = .{ .u32v = 2 } };
    try t.expectRefused(.{ .base_model = null, .extra = &.{ two, repo_url } }, "2 base models");
    const none: t.Extra = .{ .key = "general.base_model.count", .value = .{ .u32v = 0 } };
    try t.expectRefused(.{ .base_model = null, .extra = &.{ none, base_name } }, "count is 0");
    const number_pin: t.Extra = .{ .key = "general.base_model.0.version", .value = .{ .u32v = 3 } };
    try t.expectRefused(.{ .base_model = null, .extra = &.{ count_one, repo_url, number_pin } }, "general.base_model.0.version is not text");
    const number_url: t.Extra = .{ .key = "general.base_model.0.repo_url", .value = .{ .u32v = 3 } };
    try t.expectRefused(.{ .base_model = null, .extra = &.{ count_one, number_url } }, "general.base_model.0.repo_url is not text");
}

test "a producer's own glp base keys never name the base or override the spec's pin" {
    const legacy = [_]t.Extra{
        .{ .key = "glp.base_model", .value = .{ .text = "someone/Other" } },
        .{ .key = "glp.base_revision", .value = .{ .text = "0000000000000000" } },
    };
    var both = try t.accept(.{ .base_model = null, .extra = &.{ count_one, repo_url, pin, legacy[0], legacy[1] } });
    defer std.testing.allocator.free(both.bytes);
    defer both.read.deinit();
    try std.testing.expectEqualStrings(repo, both.read.vector.base_model);
    try std.testing.expectEqualStrings(commit, both.read.vector.base_revision);
    var alone = try t.accept(.{ .base_model = null, .extra = &legacy });
    defer std.testing.allocator.free(alone.bytes);
    defer alone.read.deinit();
    try std.testing.expectEqual(@as(usize, 0), alone.read.vector.base_model.len);
    try std.testing.expectEqual(@as(usize, 0), alone.read.vector.base_revision.len);
}

test "a written vector names its base only through the spec's fields, and reads back the same" {
    const a = std.testing.allocator;
    const dir = try t.unitDir(a, 8, 0);
    defer a.free(dir);
    const layers = [_]glp.Layer{.{ .id = 10, .dir = dir }};
    // the repo as an id, as the hub's URL, as a name with no owner and as a URL that is not the hub's
    const cases = [_]struct { given: []const u8, kept: []const u8 }{
        .{ .given = repo, .kept = repo },
        .{ .given = "https://huggingface.co/" ++ repo, .kept = repo },
        .{ .given = "DeepSeek-V4.1-Flash-1017", .kept = "DeepSeek-V4.1-Flash-1017" },
        .{ .given = "https://example.com/x/y", .kept = "https://example.com/x/y" },
    };
    for (cases) |c| {
        const bytes = try glp.write(a, .{ .mode = .project, .base_model = c.given, .base_revision = commit, .layers = &layers });
        defer a.free(bytes);
        var parsed = try t.parseFile(a, bytes);
        defer parsed.deinit();
        try std.testing.expect(parsed.file.find("glp.base_model") == null);
        try std.testing.expect(parsed.file.find("glp.base_revision") == null);
        try std.testing.expectEqualStrings(commit, parsed.file.str("general.base_model.0.version").?);
        var read = try t.readOk(bytes);
        defer read.deinit();
        try std.testing.expectEqualStrings(c.kept, read.vector.base_model);
        try std.testing.expectEqualStrings(commit, read.vector.base_revision);
        const again = try glp.writeVector(a, read.vector);
        defer a.free(again);
        try std.testing.expectEqualSlices(u8, bytes, again);
    }
}

test "glp.validation and the informational dose keys are written, and never touch the tensor bytes" {
    const a = std.testing.allocator;
    const dir = try a.alloc(f32, 4);
    defer a.free(dir);
    for (dir, 0..) |*x, i| x.* = if (i == 0) 1 else 0;
    const layers = [_]glp.Layer{.{ .id = 3, .dir = dir }};
    const plain = try glp.write(a, .{ .mode = .project, .layers = &layers });
    defer a.free(plain);
    const with = try glp.write(a, .{
        .mode = .project,
        .layers = &layers,
        .validation = "tool-calling 12/12 at alpha 1; finish-reason 0/24 vanish",
        .dose_cos = "3:0.041",
        .resid_norm_median = "3:12.5",
    });
    defer a.free(with);
    try std.testing.expect(std.mem.indexOf(u8, with, "glp.validation") != null);
    try std.testing.expect(std.mem.indexOf(u8, with, "tool-calling 12/12 at alpha 1; finish-reason 0/24 vanish") != null);
    try std.testing.expect(std.mem.indexOf(u8, with, "glp.dose_cos") != null);
    try std.testing.expect(std.mem.indexOf(u8, with, "glp.resid_norm_median") != null);
    try std.testing.expect(std.mem.indexOf(u8, plain, "glp.validation") == null);
    // The digest the spec makes the contract covers tensor bytes only: metadata may grow around it.
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const pv = try glp.read(arena.allocator(), plain, &why);
    const wv = try glp.read(arena.allocator(), with, &why);
    try std.testing.expectEqualSlices(u8, &pv.sha256, &wv.sha256);
    try std.testing.expectEqualSlices(f32, pv.layers[0].dir, wv.layers[0].dir);
}
