//! tf-hipcc: a `hipcc --genco`-compatible wrapper over Homebrew LLVM for macOS: HIP device code to an AMDGPU code object, no ROCm install.
//! Only what zig/build/hip.zig (and `hipcc --genco` users) needs: compile one .hip file for --offload-arch=gfxNNN to a linked, unbundled .hsaco.
//! The Zig port of the former tools/vega-macos/hipcc shell script: same arguments, environment, messages, exit statuses and clang++ command line.
//!
//! It is installed as `tf-hipcc` and meant to be symlinked as `<root>/bin/hipcc` (tf-fetch-rocm makes the link):
//! ROCm root = $TF_VEGA_ROOT, else the directory above the directory this program was invoked from (argv[0] as given, made absolute, symlinks NOT resolved);
//! it provides include/hip/** and lib/llvm/amdgcn/bitcode/*.bc (an explicit --rocm-path / --rocm-device-lib-path argument wins).
//! Invoked through PATH (argv[0] without a slash) the root cannot be known: TF_VEGA_ROOT is required then.
//! Mirrors hipcc (rocm-systems hipBin_amd.h): -x hip for .hip, -O3 unless an -O level is given, --genco = --cuda-device-only,
//! -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false. Added for macOS: the SDK libc++ needs, lld path, unbundled output.
//! Env: TF_VEGA_ROOT, TF_LLVM_PREFIX (default /opt/homebrew/opt/llvm), TF_LLD_PREFIX (/opt/homebrew/opt/lld), TF_SDKROOT (else `xcrun --show-sdk-path`),
//!      TF_HIP_COV (code object version, default 5: what the runtime's loader implements; 6 is clang 23's own default).
//! Exit: clang++'s own status for a compile (the process is replaced by clang++), 1 for a missing tool / SDK / root, 2 for a call without --genco.
const std = @import("std");

const default_llvm = "/opt/homebrew/opt/llvm";
const default_lld = "/opt/homebrew/opt/lld";
const default_cov = "5";

/// What the wrapper resolved before it looks at the arguments.
pub const Toolchain = struct {
    root: []const u8,
    llvm: []const u8,
    lld: []const u8,
    sdk: []const u8,
    cov: []const u8,
};

/// What to do with the user's arguments.
pub const Plan = union(enum) {
    /// replace this process by clang++ with this argument vector
    compile: []const []const u8,
    /// `--version` was given: replace this process by `clang --version` (its output and status are the call's)
    version,
    /// print `hipcc: ...` and exit 2
    no_genco,
};

/// The clang++ argument vector (argv[0] is the clang++ path): the wrapper's own arguments, then the user's in order without `--genco`.
pub fn plan(a: std.mem.Allocator, tc: Toolchain, args: []const []const u8) !Plan {
    var rest: std.ArrayList([]const u8) = .empty;
    var have_o = false;
    var have_rocm = false;
    var have_lib = false;
    var genco = false;
    for (args) |arg| {
        if (std.mem.eql(u8, arg, "--genco")) {
            genco = true;
            continue;
        } else if (std.mem.startsWith(u8, arg, "-O")) {
            have_o = true;
        } else if (std.mem.startsWith(u8, arg, "--rocm-path=")) {
            have_rocm = true;
        } else if (std.mem.startsWith(u8, arg, "--rocm-device-lib-path=")) {
            have_lib = true;
        } else if (std.mem.eql(u8, arg, "--version")) {
            return .version;
        }
        try rest.append(a, arg);
    }
    if (!genco) return .no_genco;

    var argv: std.ArrayList([]const u8) = .empty;
    try argv.append(a, try std.fmt.allocPrint(a, "{s}/bin/clang++", .{tc.llvm}));
    try argv.appendSlice(a, &.{ "--no-default-config", "-isysroot", tc.sdk, "-x", "hip", "--cuda-device-only", "--offload-device-only", "--no-gpu-bundle-output" });
    try argv.append(a, try std.fmt.allocPrint(a, "--ld-path={s}/bin/ld.lld", .{tc.lld}));
    try argv.append(a, try std.fmt.allocPrint(a, "-mcode-object-version={s}", .{tc.cov}));
    try argv.appendSlice(a, &.{ "-mllvm", "-amdgpu-early-inline-all=true", "-mllvm", "-amdgpu-function-calls=false" });
    if (!have_rocm) try argv.append(a, try std.fmt.allocPrint(a, "--rocm-path={s}", .{tc.root}));
    if (!have_lib) try argv.append(a, try std.fmt.allocPrint(a, "--rocm-device-lib-path={s}/lib/llvm/amdgcn/bitcode", .{tc.root}));
    if (!have_o) try argv.append(a, "-O3");
    try argv.appendSlice(a, rest.items);
    return .{ .compile = argv.items };
}

/// The ROCm root implied by how the program was invoked: the directory above the directory of `argv0`, lexically (`..` and `.` folded, symlinks
/// not resolved, like `cd "$(dirname "$0")/.." && pwd`); null when `argv0` has no slash. `cwd` joins a relative `argv0`.
pub fn rootOf(a: std.mem.Allocator, argv0: []const u8, cwd: []const u8) !?[]u8 {
    if (std.mem.indexOfScalar(u8, argv0, '/') == null) return null;
    return try std.fs.path.resolveAlloc(a, &.{ cwd, argv0, "..", ".." });
}

fn isExecutable(io: std.Io, path: []const u8) bool {
    std.Io.Dir.cwd().access(io, path, .{ .execute = true }) catch return false;
    return true;
}

fn isDir(io: std.Io, path: []const u8) bool {
    const st = std.Io.Dir.cwd().statFile(io, path, .{}) catch return false;
    return st.kind == .directory;
}

/// A non-empty environment value (the shell's `${VAR:-default}`).
fn envOr(env: *const std.process.Environ.Map, key: []const u8, default: []const u8) []const u8 {
    const v = env.get(key) orelse return default;
    return if (v.len == 0) default else v;
}

/// The working directory as a shell sees it: $PWD when it names the current directory (it may go through symlinks), else the real one.
fn logicalCwd(a: std.mem.Allocator, io: std.Io, env: *const std.process.Environ.Map) ![]const u8 {
    const real = try std.process.currentPathAlloc(io, a);
    const pwd = env.get("PWD") orelse return real;
    if (!std.fs.path.isAbsolute(pwd)) return real;
    const resolved = std.Io.Dir.cwd().realPathFileAlloc(io, pwd, a) catch return real;
    return if (std.mem.eql(u8, resolved, real)) pwd else real;
}

fn xcrunSdk(a: std.mem.Allocator, io: std.Io) []const u8 {
    const r = std.process.run(a, io, .{ .argv = &.{ "xcrun", "--show-sdk-path" } }) catch return "";
    switch (r.term) {
        .exited => |code| if (code != 0) return "",
        else => return "",
    }
    return std.mem.trimEnd(u8, r.stdout, "\n");
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(a);
    const env = init.environ_map;

    var err_buf: [1024]u8 = undefined;
    var stderr = std.Io.File.stderr().writerStreaming(io, &err_buf);
    const err = &stderr.interface;

    const root = blk: {
        if (env.get("TF_VEGA_ROOT")) |r| if (r.len > 0) break :blk r;
        const own = try rootOf(a, argv[0], try logicalCwd(a, io, env));
        break :blk own orelse {
            try err.print("hipcc: cannot tell the ROCm root from argv[0] '{s}' (it has no directory; set TF_VEGA_ROOT or run it by path)\n", .{argv[0]});
            try err.flush();
            return 1;
        };
    };
    const llvm = envOr(env, "TF_LLVM_PREFIX", default_llvm);
    const lld = envOr(env, "TF_LLD_PREFIX", default_lld);
    const clang = try std.fmt.allocPrint(a, "{s}/bin/clang", .{llvm});
    if (!isExecutable(io, clang)) {
        try err.print("hipcc: {s} missing (brew install llvm)\n", .{clang});
        try err.flush();
        return 1;
    }
    if (!isExecutable(io, try std.fmt.allocPrint(a, "{s}/bin/ld.lld", .{lld}))) {
        try err.print("hipcc: {s}/bin/ld.lld missing (brew install lld)\n", .{lld});
        try err.flush();
        return 1;
    }
    const sdk = if (env.get("TF_SDKROOT")) |s| (if (s.len > 0) s else xcrunSdk(a, io)) else xcrunSdk(a, io);
    if (sdk.len == 0 or !isDir(io, try std.fmt.allocPrint(a, "{s}/usr/include", .{sdk}))) {
        try err.writeAll("hipcc: no macOS SDK (set TF_SDKROOT)\n");
        try err.flush();
        return 1;
    }

    const tc: Toolchain = .{ .root = root, .llvm = llvm, .lld = lld, .sdk = sdk, .cov = envOr(env, "TF_HIP_COV", default_cov) };
    const p = try plan(a, tc, argv[1..]);
    switch (p) {
        .no_genco => {
            try err.writeAll("hipcc: only --genco compiles are supported by this wrapper\n");
            try err.flush();
            return 2;
        },
        .version, .compile => {
            const cmd: []const []const u8 = switch (p) {
                .compile => |c| c,
                else => &.{ clang, "--version" },
            };
            const e = std.process.replace(io, .{ .argv = cmd });
            try err.print("hipcc: cannot run {s}: {s}\n", .{ cmd[0], @errorName(e) });
            try err.flush();
            return 1;
        },
    }
}

fn testTc() Toolchain {
    return .{ .root = "/r", .llvm = "/l", .lld = "/d", .sdk = "/s", .cov = "5" };
}

test "plan: plain --genco compile" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const p = try plan(arena.allocator(), testTc(), &.{ "--genco", "--offload-arch=gfx900", "-o", "k.hsaco", "k.hip" });
    try std.testing.expectEqualDeep(@as([]const []const u8, &.{
        "/l/bin/clang++",                                    "--no-default-config",     "-isysroot",                    "/s",
        "-x",                                                "hip",                     "--cuda-device-only",           "--offload-device-only",
        "--no-gpu-bundle-output",                            "--ld-path=/d/bin/ld.lld", "-mcode-object-version=5",      "-mllvm",
        "-amdgpu-early-inline-all=true",                     "-mllvm",                  "-amdgpu-function-calls=false", "--rocm-path=/r",
        "--rocm-device-lib-path=/r/lib/llvm/amdgcn/bitcode", "-O3",                     "--offload-arch=gfx900",        "-o",
        "k.hsaco",                                           "k.hip",
    }), p.compile);
}

test "plan: overrides, --version, no --genco" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const p = try plan(a, testTc(), &.{ "k.hip", "-O2", "--rocm-path=/x", "--rocm-device-lib-path=/y", "--genco" });
    const cmd = p.compile;
    try std.testing.expectEqual(@as(usize, 15 + 4), cmd.len);
    try std.testing.expectEqualStrings("k.hip", cmd[15]);
    for (cmd) |arg| {
        try std.testing.expect(!std.mem.eql(u8, arg, "-O3"));
        try std.testing.expect(!std.mem.eql(u8, arg, "--rocm-path=/r"));
    }
    try std.testing.expectEqual(Plan.version, try plan(a, testTc(), &.{ "-c", "--version" }));
    try std.testing.expectEqual(Plan.no_genco, try plan(a, testTc(), &.{"k.hip"}));
    try std.testing.expectEqual(Plan.version, try plan(a, testTc(), &.{ "--version", "--genco" }));
}

test "rootOf: the directory above argv[0]'s, symlinks untouched" {
    const a = std.testing.allocator;
    const r1 = (try rootOf(a, "/x/rocm/bin/hipcc", "/cwd")).?;
    defer a.free(r1);
    try std.testing.expectEqualStrings("/x/rocm", r1);
    const r2 = (try rootOf(a, "bin/hipcc", "/x/rocm")).?;
    defer a.free(r2);
    try std.testing.expectEqualStrings("/x/rocm", r2);
    const r3 = (try rootOf(a, "./hipcc", "/x/rocm/bin")).?;
    defer a.free(r3);
    try std.testing.expectEqualStrings("/x/rocm", r3);
    try std.testing.expectEqual(@as(?[]u8, null), try rootOf(a, "hipcc", "/x"));
}
