//! hipcc_check HIPCC FAKE DIR [--ab SCRIPT]: runs the tf-hipcc wrapper as a process against hipcc_fake_clang (a stand-in for Homebrew's clang / clang++ that
//! prints its argv), through a `<root>/bin/hipcc` symlink like tf-fetch-rocm makes, and checks the clang++ command line, the exit statuses and the messages.
//! With `--ab SCRIPT` every case also runs through `bash SCRIPT` (the former shell wrapper, linked into the same root) and must give the same status,
//! stdout and stderr. Files land in DIR; exit status 0 when every check holds. No Homebrew, no SDK, no card.
const std = @import("std");

var failures: usize = 0;

fn check(ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (ok) return;
    failures += 1;
    std.debug.print("hipcc_check: FAIL " ++ fmt ++ "\n", args);
}

const Ran = struct { code: u8, stdout: []const u8, stderr: []const u8 };

const Ctx = struct {
    a: std.mem.Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    /// absolute: the ROCm root with bin/hipcc (the Zig wrapper) and bin/hipcc-sh (the shell wrapper)
    root: []const u8,
    llvm: []const u8,
    lld: []const u8,
    sdk: []const u8,
    ab_script: ?[]const u8,

    fn run(c: *Ctx, via: []const []const u8, args: []const []const u8, env: []const [2][]const u8, cwd: ?[]const u8) !Ran {
        var e = try c.base_env.clone(c.a);
        for ([_][]const u8{ "TF_VEGA_ROOT", "TF_LLVM_PREFIX", "TF_LLD_PREFIX", "TF_SDKROOT", "TF_HIP_COV", "FAKE_CLANG_EXIT", "PWD" }) |k| _ = e.swapRemove(k);
        try e.put("TF_LLVM_PREFIX", c.llvm);
        try e.put("TF_LLD_PREFIX", c.lld);
        try e.put("TF_SDKROOT", c.sdk);
        for (env) |kv| try e.put(kv[0], kv[1]);
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.appendSlice(c.a, via);
        try argv.appendSlice(c.a, args);
        const r = try std.process.run(c.a, c.io, .{ .argv = argv.items, .environ_map = &e, .cwd = if (cwd) |d| .{ .path = d } else .inherit });
        return .{ .code = switch (r.term) {
            .exited => |code| code,
            else => 255,
        }, .stdout = r.stdout, .stderr = r.stderr };
    }

    fn path(c: *Ctx, comptime fmt: []const u8, args: anytype) ![]u8 {
        return std.fmt.allocPrint(c.a, fmt, args);
    }

    /// Run the case through the Zig wrapper (and the shell one with --ab) and return the Zig result.
    fn case(c: *Ctx, name: []const u8, args: []const []const u8, env: []const [2][]const u8) !Ran {
        const zig_hipcc = try c.path("{s}/bin/hipcc", .{c.root});
        const r = try c.run(&.{zig_hipcc}, args, env, null);
        if (c.ab_script != null) {
            const sh = try c.run(&.{ "bash", try c.path("{s}/bin/hipcc-sh", .{c.root}) }, args, env, null);
            check(r.code == sh.code, "{s}: exit {d} != shell {d}", .{ name, r.code, sh.code });
            check(std.mem.eql(u8, r.stdout, sh.stdout), "{s}: stdout differs from the shell wrapper's:\n--- zig\n{s}--- shell\n{s}", .{ name, r.stdout, sh.stdout });
            check(std.mem.eql(u8, r.stderr, sh.stderr), "{s}: stderr differs from the shell wrapper's:\n--- zig\n{s}--- shell\n{s}", .{ name, r.stderr, sh.stderr });
        }
        return r;
    }

    /// The fake clang++'s output for a compile with the wrapper's fixed prefix and `rest` after it.
    fn compileOut(c: *Ctx, pre_tail: []const []const u8, rest: []const []const u8, cov: []const u8) ![]u8 {
        var w: std.Io.Writer.Allocating = .init(c.a);
        try w.writer.print("{s}/bin/clang++\n--no-default-config\n-isysroot\n{s}\n-x\nhip\n--cuda-device-only\n--offload-device-only\n--no-gpu-bundle-output\n", .{ c.llvm, c.sdk });
        try w.writer.print("--ld-path={s}/bin/ld.lld\n-mcode-object-version={s}\n-mllvm\n-amdgpu-early-inline-all=true\n-mllvm\n-amdgpu-function-calls=false\n", .{ c.lld, cov });
        for (pre_tail) |l| try w.writer.print("{s}\n", .{l});
        for (rest) |l| try w.writer.print("{s}\n", .{l});
        return w.written();
    }
};

fn expectRan(name: []const u8, r: Ran, code: u8, stdout: ?[]const u8, stderr: ?[]const u8) void {
    check(r.code == code, "{s}: exit {d}, want {d}\n{s}{s}", .{ name, r.code, code, r.stdout, r.stderr });
    if (stdout) |want| check(std.mem.eql(u8, r.stdout, want), "{s}: stdout\n--- got\n{s}--- want\n{s}", .{ name, r.stdout, want });
    if (stderr) |want| check(std.mem.eql(u8, r.stderr, want), "{s}: stderr\n--- got\n{s}--- want\n{s}", .{ name, r.stderr, want });
}

fn mkfile(io: std.Io, p: []const u8, data: []const u8, exec: bool) !void {
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = p, .data = data, .flags = .{ .permissions = if (exec) .executable_file else .default_file } });
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len != 4 and !(argv.len == 6 and std.mem.eql(u8, argv[4], "--ab"))) {
        std.debug.print("usage: hipcc_check HIPCC FAKE DIR [--ab SCRIPT]\n", .{});
        return 2;
    }
    const cwd = std.Io.Dir.cwd();
    const hipcc_exe = try cwd.realPathFileAlloc(io, argv[1], a);
    const fake_exe = try cwd.realPathFileAlloc(io, argv[2], a);
    try cwd.createDirPath(io, argv[3]);
    const dir = try cwd.realPathFileAlloc(io, argv[3], a);

    var c: Ctx = .{
        .a = a,
        .io = io,
        .base_env = init.environ_map,
        .root = try std.fmt.allocPrint(a, "{s}/rocm", .{dir}),
        .llvm = try std.fmt.allocPrint(a, "{s}/llvm", .{dir}),
        .lld = try std.fmt.allocPrint(a, "{s}/lld", .{dir}),
        .sdk = try std.fmt.allocPrint(a, "{s}/sdk", .{dir}),
        .ab_script = if (argv.len == 6) try cwd.realPathFileAlloc(io, argv[5], a) else null,
    };
    // the layout: <root>/bin/hipcc -> tf-hipcc, <llvm>/bin/{clang,clang++} = the fake, <lld>/bin/ld.lld an empty executable, <sdk>/usr/include
    for ([_][]const u8{ "rocm/bin", "llvm/bin", "lld/bin", "sdk/usr/include", "sdk-bare" }) |sub| try cwd.createDirPath(io, try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, sub }));
    try cwd.symLink(io, hipcc_exe, try c.path("{s}/bin/hipcc", .{c.root}), .{});
    if (c.ab_script) |s| try cwd.symLink(io, s, try c.path("{s}/bin/hipcc-sh", .{c.root}), .{});
    for ([_][]const u8{ "clang", "clang++" }) |n| try cwd.copyFile(fake_exe, cwd, try c.path("{s}/bin/{s}", .{ c.llvm, n }), io, .{});
    try mkfile(io, try c.path("{s}/bin/ld.lld", .{c.lld}), "", true);

    const rocm_path = try c.path("--rocm-path={s}", .{c.root});
    const lib_path = try c.path("--rocm-device-lib-path={s}/lib/llvm/amdgcn/bitcode", .{c.root});

    // 1. a plain compile: the wrapper's arguments, -O3 and the root's paths, then the user's arguments in order without --genco
    {
        const r = try c.case("plain", &.{ "--genco", "--offload-arch=gfx900", "-o", "k.hsaco", "k.hip" }, &.{});
        expectRan("plain", r, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{ "--offload-arch=gfx900", "-o", "k.hsaco", "k.hip" }, "5"), "");
    }
    // 1b. --genco anywhere, other flags passed through in order
    {
        const r = try c.case("order", &.{ "-DX=1", "k.hip", "--genco", "-I", "inc", "--offload-arch=gfx906" }, &.{});
        expectRan("order", r, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{ "-DX=1", "k.hip", "-I", "inc", "--offload-arch=gfx906" }, "5"), "");
    }
    // 2. -O level, --rocm-path= and --rocm-device-lib-path= given: theirs win, and stay where the user put them
    {
        const r = try c.case("overrides", &.{ "--genco", "-O2", "--rocm-path=/opt/rocm", "--rocm-device-lib-path=/opt/bc", "k.hip" }, &.{});
        expectRan("overrides", r, 0, try c.compileOut(&.{}, &.{ "-O2", "--rocm-path=/opt/rocm", "--rocm-device-lib-path=/opt/bc", "k.hip" }, "5"), "");
        const r2 = try c.case("override rocm only", &.{ "--genco", "--rocm-path=/opt/rocm", "k.hip" }, &.{});
        expectRan("override rocm only", r2, 0, try c.compileOut(&.{ lib_path, "-O3" }, &.{ "--rocm-path=/opt/rocm", "k.hip" }, "5"), "");
        const r3 = try c.case("override -O0", &.{ "--genco", "-O0", "k.hip" }, &.{});
        expectRan("override -O0", r3, 0, try c.compileOut(&.{ rocm_path, lib_path }, &.{ "-O0", "k.hip" }, "5"), "");
    }
    // 3. TF_HIP_COV, TF_VEGA_ROOT (wins over the directory the wrapper was invoked from)
    {
        const r = try c.case("cov", &.{ "--genco", "k.hip" }, &.{.{ "TF_HIP_COV", "6" }});
        expectRan("cov", r, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "6"), "");
        const r2 = try c.case("vega root", &.{ "--genco", "k.hip" }, &.{.{ "TF_VEGA_ROOT", "/elsewhere/rocm" }});
        expectRan("vega root", r2, 0, try c.compileOut(&.{ "--rocm-path=/elsewhere/rocm", "--rocm-device-lib-path=/elsewhere/rocm/lib/llvm/amdgcn/bitcode", "-O3" }, &.{"k.hip"}, "5"), "");
        // an empty variable is as good as unset (the shell's ${VAR:-default})
        const r3 = try c.case("empty vars", &.{ "--genco", "k.hip" }, &.{ .{ "TF_VEGA_ROOT", "" }, .{ "TF_HIP_COV", "" } });
        expectRan("empty vars", r3, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "5"), "");
    }
    // 4. no --genco: exit 2 and the message, clang is not run
    {
        const r = try c.case("no genco", &.{ "-c", "k.hip" }, &.{});
        expectRan("no genco", r, 2, "", "hipcc: only --genco compiles are supported by this wrapper\n");
        const r2 = try c.case("no args", &.{}, &.{});
        expectRan("no args", r2, 2, "", "hipcc: only --genco compiles are supported by this wrapper\n");
    }
    // 5. missing tools and SDK: exit 1 and the message
    {
        const r = try c.case("no clang", &.{ "--genco", "k.hip" }, &.{.{ "TF_LLVM_PREFIX", "/nonexistent/llvm" }});
        expectRan("no clang", r, 1, "", "hipcc: /nonexistent/llvm/bin/clang missing (brew install llvm)\n");
        const r2 = try c.case("no lld", &.{ "--genco", "k.hip" }, &.{.{ "TF_LLD_PREFIX", "/nonexistent/lld" }});
        expectRan("no lld", r2, 1, "", "hipcc: /nonexistent/lld/bin/ld.lld missing (brew install lld)\n");
        const r3 = try c.case("no sdk", &.{ "--genco", "k.hip" }, &.{.{ "TF_SDKROOT", try c.path("{s}/sdk-bare", .{dir}) }});
        expectRan("no sdk", r3, 1, "", "hipcc: no macOS SDK (set TF_SDKROOT)\n");
    }
    // 6. --version runs `clang --version`, whose output and status are the call's; it wins over a missing --genco and over later arguments
    {
        const want = try c.path("{s}/bin/clang\n--version\n", .{c.llvm});
        const r = try c.case("version", &.{"--version"}, &.{});
        expectRan("version", r, 0, want, "");
        const r2 = try c.case("version + args", &.{ "k.hip", "--version", "--genco" }, &.{.{ "FAKE_CLANG_EXIT", "3" }});
        expectRan("version + args", r2, 3, want, "");
    }
    // 7. the compile's exit status and output are clang++'s (the wrapper execs it)
    {
        const r = try c.case("clang++ status", &.{ "--genco", "k.hip" }, &.{.{ "FAKE_CLANG_EXIT", "7" }});
        expectRan("clang++ status", r, 7, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "5"), "");
    }
    // 8. the root comes from argv[0] as invoked: relative paths are joined with the working directory, symlinks are not resolved, `..` is folded
    {
        const rel = try c.run(&.{"rocm/bin/hipcc"}, &.{ "--genco", "k.hip" }, &.{}, dir);
        expectRan("relative argv0", rel, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "5"), "");
        const dots = try c.run(&.{"./hipcc"}, &.{ "--genco", "k.hip" }, &.{}, try c.path("{s}/bin", .{c.root}));
        expectRan("dot argv0", dots, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "5"), "");
        // a symlink to the root: the root named is the one the wrapper was called through, not its target
        const via = try c.path("{s}/rocm-link", .{dir});
        try cwd.symLink(io, "rocm", via, .{ .is_directory = true });
        const lr = try c.run(&.{try c.path("{s}/bin/hipcc", .{via})}, &.{ "--genco", "k.hip" }, &.{}, null);
        expectRan("symlinked root", lr, 0, try c.compileOut(&.{ try c.path("--rocm-path={s}", .{via}), try c.path("--rocm-device-lib-path={s}/lib/llvm/amdgcn/bitcode", .{via}), "-O3" }, &.{"k.hip"}, "5"), "");
    }
    // 9. argv[0] without a slash: the root is unknown, TF_VEGA_ROOT is required (exec'd with the name as given)
    {
        var bin = try cwd.openDir(io, try c.path("{s}/bin", .{c.root}), .{});
        defer bin.close(io);
        var e = try init.environ_map.clone(a);
        for ([_][]const u8{ "TF_VEGA_ROOT", "PWD" }) |k| _ = e.swapRemove(k);
        try e.put("TF_LLVM_PREFIX", c.llvm);
        try e.put("TF_LLD_PREFIX", c.lld);
        try e.put("TF_SDKROOT", c.sdk);
        const bare = try std.process.run(a, io, .{ .argv = &.{"hipcc"}, .exe = .{ .explicit = .{ .dir = bin, .path = "hipcc" } }, .environ_map = &e });
        expectRan("bare argv0", .{ .code = bare.term.exited, .stdout = bare.stdout, .stderr = bare.stderr }, 1, "", "hipcc: cannot tell the ROCm root from argv[0] 'hipcc' (it has no directory; set TF_VEGA_ROOT or run it by path)\n");
        try e.put("TF_VEGA_ROOT", c.root);
        const ok = try std.process.run(a, io, .{ .argv = &.{ "hipcc", "--genco", "k.hip" }, .exe = .{ .explicit = .{ .dir = bin, .path = "hipcc" } }, .environ_map = &e });
        expectRan("bare argv0 + TF_VEGA_ROOT", .{ .code = ok.term.exited, .stdout = ok.stdout, .stderr = ok.stderr }, 0, try c.compileOut(&.{ rocm_path, lib_path, "-O3" }, &.{"k.hip"}, "5"), "");
    }

    if (failures > 0) {
        std.debug.print("hipcc_check: {d} failure(s)\n", .{failures});
        return 1;
    }
    try std.Io.File.stdout().writeStreamingAll(io, try std.fmt.allocPrint(a, "hipcc_check: ok{s}\n", .{if (c.ab_script != null) " (also identical to the shell wrapper)" else ""}));
    return 0;
}
