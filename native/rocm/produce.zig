//! gfx1030 produce line. Device code is ROCm 7.14.0 hipcc, one fatbin for that arch.
const std = @import("std");

pub const rocm_pin = "7.14.0";
pub const arch = "gfx1030";
pub const codebook = "3inst";
pub const produce_line = "hipcc --offload-arch=gfx1030 -O3 -cb 3inst";

pub const refused_archs = [_][]const u8{ "gfx900", "gfx906", "gfx1013" };

pub const Error = error{
    CompilerRefused,
    ArchRefused,
    ForeignArch,
    MultipleArch,
    ContractOff,
    CodebookRefused,
    Wave32Required,
    FatbinRefused,
    RocmPinMismatch,
    NameRefused,
};

const prefix = [_][]const u8{
    "hipcc",
    "--offload-arch=gfx1030",
    "-O3",
    "-cb",
    "3inst",
};

pub fn rocmVersionOk(text: []const u8) bool {
    return std.mem.indexOf(u8, text, rocm_pin) != null;
}

pub fn acceptArch(name: []const u8) Error!void {
    if (std.mem.eql(u8, name, arch)) return;
    return error.ArchRefused;
}

pub fn fatbinPath(stem: []const u8, buf: []u8) Error![]const u8 {
    if (stem.len == 0 or std.mem.indexOfScalar(u8, stem, '/') != null or std.mem.indexOfScalar(u8, stem, '.') != null)
        return error.NameRefused;
    for (refused_archs) |foreign| {
        if (std.mem.indexOf(u8, stem, foreign) != null) return error.ForeignArch;
    }
    return std.fmt.bufPrint(buf, "build/rocm/fatbin/gfx1030/{s}.hipfb", .{stem}) catch error.NameRefused;
}

pub const Argv = struct {
    tokens: [10][]const u8,

    pub fn line(self: Argv, buf: []u8) []const u8 {
        var used: usize = 0;
        for (self.tokens, 0..) |token, i| {
            if (i != 0 and used < buf.len) {
                buf[used] = ' ';
                used += 1;
            }
            const n = @min(token.len, buf.len - used);
            @memcpy(buf[used..][0..n], token[0..n]);
            used += n;
        }
        return buf[0..used];
    }
};

/// Locked prefix, then wave32 and a single gfx1030 device object. No other arch is added.
pub fn deviceArgv(stem: []const u8, source: []const u8, path_buf: []u8) Error!Argv {
    const out = try fatbinPath(stem, path_buf);
    var argv = Argv{ .tokens = undefined };
    @memcpy(argv.tokens[0..prefix.len], &prefix);
    argv.tokens[prefix.len] = "-mno-wavefrontsize64";
    argv.tokens[prefix.len + 1] = "--cuda-device-only";
    argv.tokens[prefix.len + 2] = "-o";
    argv.tokens[prefix.len + 3] = out;
    argv.tokens[prefix.len + 4] = source;
    try validate(&argv.tokens);
    return argv;
}

pub fn validate(tokens: []const []const u8) Error!void {
    if (tokens.len < prefix.len) return error.CompilerRefused;
    for (prefix, 0..) |token, i| {
        if (!std.mem.eql(u8, tokens[i], token)) return error.CompilerRefused;
    }
    const base = std.fs.path.basename(tokens[0]);
    if (!std.mem.eql(u8, base, "hipcc")) return error.CompilerRefused;
    var archs: usize = 0;
    var wave32 = false;
    var outputs: usize = 0;
    var saw_cb = false;
    for (tokens, 0..) |token, i| {
        if (std.mem.indexOf(u8, token, "-ffp-contract=off") != null) return error.ContractOff;
        if (std.mem.eql(u8, token, "-ffp-contract") and i + 1 < tokens.len and std.mem.eql(u8, tokens[i + 1], "off"))
            return error.ContractOff;
        for (refused_archs) |foreign| {
            if (std.mem.indexOf(u8, token, foreign) != null) return error.ForeignArch;
        }
        if (std.mem.startsWith(u8, token, "--offload-arch=")) {
            archs += 1;
            if (!std.mem.eql(u8, token, "--offload-arch=gfx1030")) return error.ArchRefused;
        }
        if (std.mem.eql(u8, token, "-mno-wavefrontsize64")) wave32 = true;
        if (std.mem.eql(u8, token, "-o")) outputs += 1;
        if (std.mem.eql(u8, token, "-cb")) {
            saw_cb = true;
            if (i + 1 >= tokens.len or !std.mem.eql(u8, tokens[i + 1], codebook)) return error.CodebookRefused;
        }
        if (std.mem.eql(u8, token, "zig") or std.mem.endsWith(u8, token, "/zig")) return error.CompilerRefused;
    }
    if (archs != 1) return error.MultipleArch;
    if (!wave32) return error.Wave32Required;
    if (!saw_cb) return error.CodebookRefused;
    if (outputs != 1) return error.FatbinRefused;
    const out = tokens[tokens.len - 2];
    if (!std.mem.startsWith(u8, out, "build/rocm/fatbin/gfx1030/") or !std.mem.endsWith(u8, out, ".hipfb"))
        return error.FatbinRefused;
}

/// Integer stage of the 3inst codebook (EXL3 pair decode, before the half2 add).
pub fn mix3inst(s: u32) u32 {
    const x = s *% 89226354 +% 64248484;
    return (x & 0x8FFF8FFF) ^ 0x3B603B60;
}

test "produce line is hipcc gfx1030 -O3 -cb 3inst, one fatbin, wave32" {
    var path: [96]u8 = undefined;
    const argv = try deviceArgv("w4a16_fdot2", "native/rocm/hip/w4a16_fdot2.hip", &path);
    var text: [256]u8 = undefined;
    const rendered = argv.line(&text);
    try std.testing.expect(std.mem.startsWith(u8, rendered, produce_line));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "-ffp-contract=off") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "-mno-wavefrontsize64") != null);
    try std.testing.expect(std.mem.endsWith(u8, rendered, "build/rocm/fatbin/gfx1030/w4a16_fdot2.hipfb native/rocm/hip/w4a16_fdot2.hip"));
    try validate(&argv.tokens);
}

test "refused archs and contraction-off never join the produce line" {
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx900"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx906"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx1013"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx1100"));
    try acceptArch("gfx1030");
    const bad = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-cb", "3inst", "-ffp-contract=off", "-mno-wavefrontsize64", "-o", "build/rocm/fatbin/gfx1030/a.hipfb", "a.hip" };
    try std.testing.expectError(error.ContractOff, validate(&bad));
    const foreign = [_][]const u8{ "hipcc", "--offload-arch=gfx900", "-O3", "-cb", "3inst", "-mno-wavefrontsize64", "-o", "build/rocm/fatbin/gfx900/a.hipfb", "a.hip" };
    try std.testing.expectError(error.CompilerRefused, validate(&foreign));
    const smuggled = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-cb", "3inst", "-mno-wavefrontsize64", "-o", "build/rocm/fatbin/gfx1030/gfx906.hipfb", "a.hip" };
    try std.testing.expectError(error.ForeignArch, validate(&smuggled));
    const bundled = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-cb", "3inst", "--offload-arch=gfx1013", "-mno-wavefrontsize64", "-o", "build/rocm/fatbin/gfx1030/a.hipfb", "a.hip" };
    try std.testing.expectError(error.ForeignArch, validate(&bundled));
    const zig_cc = [_][]const u8{ "zig", "--offload-arch=gfx1030", "-O3", "-cb", "3inst", "-mno-wavefrontsize64", "-o", "build/rocm/fatbin/gfx1030/a.hipfb", "a.hip" };
    try std.testing.expectError(error.CompilerRefused, validate(&zig_cc));
}

test "ROCm pin is 7.14.0 and the 3inst mix is stable" {
    try std.testing.expect(rocmVersionOk("HIP version: 7.14.0\n"));
    try std.testing.expect(!rocmVersionOk("HIP version: 7.2.0\n"));
    try std.testing.expect(!rocmVersionOk("clang 18.0.0"));
    try std.testing.expectEqual(@as(u32, 0x38b431c4), mix3inst(0));
    try std.testing.expectEqual(@as(u32, 0x3245bc76), mix3inst(1));
    try std.testing.expectEqual(@as(u32, 0x3194b552), mix3inst(0xffff));
    try std.testing.expectEqual(@as(u32, 0xb841beac), mix3inst(0x1234));
}
