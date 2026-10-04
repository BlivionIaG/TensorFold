//! gfx1030 produce line. Device code is ROCm 7.14.0 hipcc, one fatbin for that arch.
const std = @import("std");

pub const rocm_pin = "7.14.0";
/// Banner from hipcc in the 7.14.0-3 package. `.info/version` is 7.14.0.
pub const hip_banner = "HIP version: 7.14.60850";
pub const arch = "gfx1030";
pub const codebook = "3inst";
pub const produce_line = "hipcc --offload-arch=gfx1030 -O3 -mno-wavefrontsize64";
pub const fatbin_rel = "native/rocm/fatbin/gfx1030/tensorfold.hipfb";
pub const unity_source = "native/rocm/hip/gfx1030.hip";

pub const refused_archs = [_][]const u8{ "gfx900", "gfx906", "gfx1013" };

pub const Error = error{
    CompilerRefused,
    ArchRefused,
    ForeignArch,
    MultipleArch,
    ContractOff,
    QuantizerSwitchRefused,
    Wave32Required,
    FatbinRefused,
    CudaDeviceOnly,
    BundleRefused,
    RocmPinMismatch,
    NameRefused,
};

const prefix = [_][]const u8{
    "hipcc",
    "--offload-arch=gfx1030",
    "-O3",
    "-mno-wavefrontsize64",
};

const bundle_magic = "__CLANG_OFFLOAD_BUNDLE__";

pub fn rocmVersionOk(text: []const u8) bool {
    if (std.mem.indexOf(u8, text, "7.14.1") != null) return false;
    if (std.mem.indexOf(u8, text, rocm_pin) != null) return true;
    return std.mem.indexOf(u8, text, hip_banner) != null;
}

pub fn acceptArch(name: []const u8) Error!void {
    if (std.mem.eql(u8, name, arch)) return;
    return error.ArchRefused;
}

pub const Argv = struct {
    tokens: [8][]const u8,

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

/// Compile line, then --genco into the one gfx1030 fatbin. `-cb` is not a hipcc flag.
pub fn deviceArgv() Error!Argv {
    var argv = Argv{ .tokens = undefined };
    @memcpy(argv.tokens[0..prefix.len], &prefix);
    argv.tokens[prefix.len] = "--genco";
    argv.tokens[prefix.len + 1] = "-o";
    argv.tokens[prefix.len + 2] = fatbin_rel;
    argv.tokens[prefix.len + 3] = unity_source;
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
    var genco = false;
    var outputs: usize = 0;
    var out: ?[]const u8 = null;
    for (tokens, 0..) |token, i| {
        if (std.mem.eql(u8, token, "-cb") or std.mem.eql(u8, token, codebook)) return error.QuantizerSwitchRefused;
        if (std.mem.eql(u8, token, "--cuda-device-only")) return error.CudaDeviceOnly;
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
        if (std.mem.eql(u8, token, "--genco")) genco = true;
        if (std.mem.eql(u8, token, "-o")) {
            outputs += 1;
            if (i + 1 >= tokens.len) return error.FatbinRefused;
            out = tokens[i + 1];
        }
        if (std.mem.eql(u8, token, "zig") or std.mem.endsWith(u8, token, "/zig")) return error.CompilerRefused;
    }
    if (archs != 1) return error.MultipleArch;
    if (!wave32) return error.Wave32Required;
    if (!genco or outputs != 1) return error.FatbinRefused;
    const produced = out orelse return error.FatbinRefused;
    if (!std.mem.eql(u8, produced, fatbin_rel)) return error.FatbinRefused;
    if (!std.mem.eql(u8, tokens[tokens.len - 1], unity_source)) return error.FatbinRefused;
}

pub const BundlePart = struct {
    triple: []const u8,
    code_object: bool,
};

fn readU64(bytes: []const u8, at: *usize) Error!u64 {
    if (at.* + 8 > bytes.len) return error.BundleRefused;
    const value = std.mem.readInt(u64, bytes[at.*..][0..8], .little);
    at.* += 8;
    return value;
}

fn writeU64(buf: []u8, at: *usize, value: u64) Error!void {
    if (at.* + 8 > buf.len) return error.BundleRefused;
    std.mem.writeInt(u64, buf[at.*..][0..8], value, .little);
    at.* += 8;
}

/// A loadable fatbin is a clang offload bundle with one gfx1030 code object.
pub fn acceptBundle(bytes: []const u8) Error!void {
    if (bytes.len < bundle_magic.len + 8 or !std.mem.startsWith(u8, bytes, bundle_magic)) return error.BundleRefused;
    var at: usize = bundle_magic.len;
    const n = try readU64(bytes, &at);
    if (n == 0 or n > 8) return error.BundleRefused;
    var devices: usize = 0;
    var i: u64 = 0;
    while (i < n) : (i += 1) {
        const off = try readU64(bytes, &at);
        const size = try readU64(bytes, &at);
        const triple_len = try readU64(bytes, &at);
        if (triple_len > 256 or at + triple_len > bytes.len) return error.BundleRefused;
        const triple = bytes[at..][0..@intCast(triple_len)];
        at += @intCast(triple_len);
        for (refused_archs) |foreign| {
            if (std.mem.indexOf(u8, triple, foreign) != null) return error.ForeignArch;
        }
        if (std.mem.indexOf(u8, triple, "amdgcn") == null) continue;
        if (std.mem.indexOf(u8, triple, arch) == null) return error.ArchRefused;
        if (size < 4 or off + size > bytes.len) return error.BundleRefused;
        if (!std.mem.eql(u8, bytes[@intCast(off)..][0..4], &[_]u8{ 0x7f, 'E', 'L', 'F' })) return error.BundleRefused;
        devices += 1;
    }
    if (devices != 1) return error.BundleRefused;
}

pub fn encodeBundle(buf: []u8, entries: []const BundlePart) Error![]const u8 {
    if (entries.len == 0 or entries.len > 8 or buf.len < bundle_magic.len + 8) return error.BundleRefused;
    @memcpy(buf[0..bundle_magic.len], bundle_magic);
    var header: usize = bundle_magic.len + 8;
    for (entries) |entry| header += 24 + entry.triple.len;
    var cursor = header;
    var blobs: [8]struct { at: usize, len: usize } = undefined;
    for (entries, 0..) |entry, i| {
        const blob_len: usize = if (entry.code_object) 4 else 0;
        if (cursor + blob_len > buf.len) return error.BundleRefused;
        if (entry.code_object) buf[cursor..][0..4].* = .{ 0x7f, 'E', 'L', 'F' };
        blobs[i] = .{ .at = cursor, .len = blob_len };
        cursor += blob_len;
    }
    var w: usize = bundle_magic.len;
    try writeU64(buf, &w, entries.len);
    for (entries, 0..) |entry, i| {
        try writeU64(buf, &w, blobs[i].at);
        try writeU64(buf, &w, blobs[i].len);
        try writeU64(buf, &w, entry.triple.len);
        @memcpy(buf[w..][0..entry.triple.len], entry.triple);
        w += entry.triple.len;
    }
    return buf[0..cursor];
}

const built_fatbin = @embedFile("fatbin/gfx1030/tensorfold.hipfb");

pub fn loadFatbin() Error![]const u8 {
    try acceptBundle(built_fatbin);
    return built_fatbin;
}

pub fn gfx1030Bundle(buf: []u8) Error![]const u8 {
    const entries = [_]BundlePart{
        .{ .triple = "host-x86_64-unknown-linux-gnu-", .code_object = false },
        .{ .triple = "hipv4-amdgcn-amd-amdhsa--gfx1030", .code_object = true },
    };
    return encodeBundle(buf, &entries);
}

/// Integer stage of the 3inst codebook (EXL3 pair decode, before the half2 add).
pub fn mix3inst(s: u32) u32 {
    const x = s *% 89226354 +% 64248484;
    return (x & 0x8FFF8FFF) ^ 0x3B603B60;
}

test "produce line is hipcc gfx1030 -O3 wave32, one fatbin" {
    const argv = try deviceArgv();
    var text: [256]u8 = undefined;
    const rendered = argv.line(&text);
    try std.testing.expect(std.mem.startsWith(u8, rendered, produce_line));
    try std.testing.expect(std.mem.indexOf(u8, rendered, "-cb") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, codebook) == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "-ffp-contract=off") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "--cuda-device-only") == null);
    try std.testing.expect(std.mem.indexOf(u8, rendered, "--genco") != null);
    try std.testing.expect(std.mem.endsWith(u8, rendered, fatbin_rel ++ " " ++ unity_source));
    try validate(&argv.tokens);
    const loaded = try loadFatbin();
    try std.testing.expect(std.mem.startsWith(u8, loaded, bundle_magic));
}

test "refused archs, contraction-off, and cuda-device-only never join the produce line" {
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx900"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx906"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx1013"));
    try std.testing.expectError(error.ArchRefused, acceptArch("gfx1100"));
    try acceptArch("gfx1030");
    const bad = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "-ffp-contract=off", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.ContractOff, validate(&bad));
    const quantizer = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "-cb", "3inst", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.QuantizerSwitchRefused, validate(&quantizer));
    const device_only = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "--cuda-device-only", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.CudaDeviceOnly, validate(&device_only));
    const foreign = [_][]const u8{ "hipcc", "--offload-arch=gfx900", "-O3", "-mno-wavefrontsize64", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.CompilerRefused, validate(&foreign));
    const smuggled = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "--genco", "-o", "native/rocm/fatbin/gfx1030/gfx906.hipfb", unity_source };
    try std.testing.expectError(error.ForeignArch, validate(&smuggled));
    const bundled = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "--offload-arch=gfx1013", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.ForeignArch, validate(&bundled));
    const zig_cc = [_][]const u8{ "zig", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "--genco", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.CompilerRefused, validate(&zig_cc));
    const no_genco = [_][]const u8{ "hipcc", "--offload-arch=gfx1030", "-O3", "-mno-wavefrontsize64", "-o", fatbin_rel, unity_source };
    try std.testing.expectError(error.FatbinRefused, validate(&no_genco));
}

test "ROCm pin is 7.14.0 and the 3inst mix is stable" {
    try std.testing.expect(rocmVersionOk("HIP version: 7.14.0\n"));
    try std.testing.expect(rocmVersionOk("7.14.0\n"));
    try std.testing.expect(rocmVersionOk(hip_banner ++ "-0000000\n"));
    try std.testing.expect(!rocmVersionOk("HIP version: 7.2.0\n"));
    try std.testing.expect(!rocmVersionOk("HIP version: 7.14.1\n"));
    try std.testing.expect(!rocmVersionOk("clang 18.0.0"));
    try std.testing.expectEqual(@as(u32, 0x38b431c4), mix3inst(0));
    try std.testing.expectEqual(@as(u32, 0x3245bc76), mix3inst(1));
    try std.testing.expectEqual(@as(u32, 0x3194b552), mix3inst(0xffff));
    try std.testing.expectEqual(@as(u32, 0xb841beac), mix3inst(0x1234));
}

test "a loadable fatbin is one gfx1030 code object" {
    var buf: [256]u8 = undefined;
    const bundle = try gfx1030Bundle(&buf);
    try acceptBundle(bundle);
    try std.testing.expect(std.mem.startsWith(u8, bundle, bundle_magic));
    try std.testing.expectError(error.BundleRefused, acceptBundle(""));
    try std.testing.expectError(error.BundleRefused, acceptBundle(&[_]u8{ 0x7f, 'E', 'L', 'F' }));
    const foreign = [_]BundlePart{
        .{ .triple = "hipv4-amdgcn-amd-amdhsa--gfx900", .code_object = true },
    };
    const foreign_bytes = try encodeBundle(&buf, &foreign);
    try std.testing.expectError(error.ForeignArch, acceptBundle(foreign_bytes));
    const two = [_]BundlePart{
        .{ .triple = "hipv4-amdgcn-amd-amdhsa--gfx1030", .code_object = true },
        .{ .triple = "hipv4-amdgcn-amd-amdhsa--gfx1030", .code_object = true },
    };
    var wide: [384]u8 = undefined;
    const wide_bytes = try encodeBundle(&wide, &two);
    try std.testing.expectError(error.BundleRefused, acceptBundle(wide_bytes));
}
