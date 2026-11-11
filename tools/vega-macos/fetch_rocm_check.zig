//! fetch_rocm_check FETCH DIR: runs tf-fetch-rocm as a process on a synthetic wheel (a zip with stored and deflated members, the headers and bitcode it installs and
//! members it must skip) and a pins file naming that wheel's sha256, offline: the installed tree, bin/.hipVersion, .pinned, the bin/hipcc link, the early exit of a
//! second run, --keep, the default root, the error exits (sha256 mismatch, missing wheel, not a zip, bad CRC, missing wrapper, bad arguments).
//! The wheel and the pins stay in DIR/fixture (wheel.whl is the wheel's file name, pins.env the pins) for a run of the former fetch-rocm.sh on the same input.
//! Exit status 0 when every check holds. No network.
const std = @import("std");
const flate = std.compress.flate;
const Sha256 = std.crypto.hash.sha2.Sha256;
const Crc32 = std.hash.Crc32;

var failures: usize = 0;

fn check(ok: bool, comptime fmt: []const u8, args: anytype) void {
    if (ok) return;
    failures += 1;
    std.debug.print("fetch_rocm_check: FAIL " ++ fmt ++ "\n", args);
}

const Member = struct {
    name: []const u8,
    data: []const u8,
    deflate: bool = false,
    /// write a wrong CRC-32 into the headers
    bad_crc: bool = false,
};

fn put16(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u16) !void {
    try list.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
}

fn put32(list: *std.ArrayList(u8), a: std.mem.Allocator, v: u32) !void {
    try list.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
}

fn deflated(a: std.mem.Allocator, data: []const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = try .initCapacity(a, 4096);
    var window: [flate.max_window_len]u8 = undefined;
    var c = try flate.Compress.init(&out.writer, &window, .raw, .level_6);
    try c.writer.writeAll(data);
    try c.finish();
    return out.written();
}

/// A zip file the way Python's zipfile writes a wheel: local headers, data, central directory, end record.
fn buildZip(a: std.mem.Allocator, members: []const Member) ![]u8 {
    var zip: std.ArrayList(u8) = .empty;
    var central: std.ArrayList(u8) = .empty;
    const date: u16 = (0 << 9) | (1 << 5) | 1; // 1980-01-01
    for (members) |m| {
        const body = if (m.deflate) try deflated(a, m.data) else m.data;
        const crc = Crc32.hash(m.data) ^ @as(u32, if (m.bad_crc) 0xdeadbeef else 0);
        const method: u16 = if (m.deflate) 8 else 0;
        const offset: u32 = @intCast(zip.items.len);
        try zip.appendSlice(a, "PK\x03\x04");
        try put16(&zip, a, 20); // version needed
        try put16(&zip, a, 0); // flags
        try put16(&zip, a, method);
        try put16(&zip, a, 0); // time
        try put16(&zip, a, date);
        try put32(&zip, a, crc);
        try put32(&zip, a, @intCast(body.len));
        try put32(&zip, a, @intCast(m.data.len));
        try put16(&zip, a, @intCast(m.name.len));
        try put16(&zip, a, 0); // extra
        try zip.appendSlice(a, m.name);
        try zip.appendSlice(a, body);

        try central.appendSlice(a, "PK\x01\x02");
        try put16(&central, a, 20); // made by
        try put16(&central, a, 20); // version needed
        try put16(&central, a, 0);
        try put16(&central, a, method);
        try put16(&central, a, 0);
        try put16(&central, a, date);
        try put32(&central, a, crc);
        try put32(&central, a, @intCast(body.len));
        try put32(&central, a, @intCast(m.data.len));
        try put16(&central, a, @intCast(m.name.len));
        try put16(&central, a, 0); // extra
        try put16(&central, a, 0); // comment
        try put16(&central, a, 0); // disk
        try put16(&central, a, 0); // internal attributes
        try put32(&central, a, 0); // external attributes
        try put32(&central, a, offset);
        try central.appendSlice(a, m.name);
    }
    const cd_offset: u32 = @intCast(zip.items.len);
    try zip.appendSlice(a, central.items);
    try zip.appendSlice(a, "PK\x05\x06");
    try put16(&zip, a, 0);
    try put16(&zip, a, 0);
    try put16(&zip, a, @intCast(members.len));
    try put16(&zip, a, @intCast(members.len));
    try put32(&zip, a, @intCast(central.items.len));
    try put32(&zip, a, cd_offset);
    try put16(&zip, a, 0);
    return zip.items;
}

const version_header =
    \\#ifndef HIP_VERSION_H
    \\#define HIP_VERSION_MAJOR 7
    \\#define HIP_VERSION_MINOR 16
    \\#define HIP_VERSION_PATCH 26332
    \\#endif
    \\
;
const hip_version_text = "HIP_VERSION_MAJOR=7\nHIP_VERSION_MINOR=16\nHIP_VERSION_PATCH=26332\nHIP_VERSION_GITHASH=0\n";

/// Many compressible lines: several deflate blocks, matches across them.
fn bigHeader(a: std.mem.Allocator) ![]u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    for (0..6000) |i| try w.writer.print("#define HIP_AMD_DETAIL_{d} (({d} * 7) + HIP_VERSION_MAJOR)\n", .{ i, i % 97 });
    return w.written();
}

/// Bytes that hardly compress.
fn noise(a: std.mem.Allocator, n: usize, seed: u64) ![]u8 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try a.alloc(u8, n);
    prng.random().bytes(out);
    return out;
}

const Ran = struct { code: u8, stdout: []const u8, stderr: []const u8 };

const Ctx = struct {
    a: std.mem.Allocator,
    io: std.Io,
    base_env: *const std.process.Environ.Map,
    fetch: []const u8,

    fn run(c: *Ctx, args: []const []const u8, env: []const [2][]const u8) !Ran {
        return c.runExe(c.fetch, args, env);
    }

    fn runExe(c: *Ctx, exe: []const u8, args: []const []const u8, env: []const [2][]const u8) !Ran {
        var e = try c.base_env.clone(c.a);
        for ([_][]const u8{ "TF_VEGA_ROOT", "HOME", "TMPDIR" }) |k| _ = e.swapRemove(k);
        for (env) |kv| try e.put(kv[0], kv[1]);
        var argv: std.ArrayList([]const u8) = .empty;
        try argv.append(c.a, exe);
        try argv.appendSlice(c.a, args);
        const r = try std.process.run(c.a, c.io, .{ .argv = argv.items, .environ_map = &e });
        return .{ .code = switch (r.term) {
            .exited => |code| code,
            else => 255,
        }, .stdout = r.stdout, .stderr = r.stderr };
    }

    fn readAll(c: *Ctx, p: []const u8) ![]u8 {
        return std.Io.Dir.cwd().readFileAlloc(c.io, p, c.a, .limited(1 << 26));
    }

    fn exists(c: *Ctx, p: []const u8) bool {
        _ = std.Io.Dir.cwd().statFile(c.io, p, .{ .follow_symlinks = false }) catch return false;
        return true;
    }

    fn write(c: *Ctx, p: []const u8, data: []const u8) !void {
        try std.Io.Dir.cwd().writeFile(c.io, .{ .sub_path = p, .data = data });
    }

    /// Every file and link below `root`, relative, sorted; a link shows as `path -> target`.
    fn tree(c: *Ctx, root: []const u8) ![]const []const u8 {
        var dir = try std.Io.Dir.cwd().openDir(c.io, root, .{ .iterate = true });
        defer dir.close(c.io);
        var walker = try dir.walk(c.a);
        defer walker.deinit();
        var list: std.ArrayList([]const u8) = .empty;
        while (try walker.next(c.io)) |entry| switch (entry.kind) {
            .file => try list.append(c.a, try c.a.dupe(u8, entry.path)),
            .sym_link => {
                var buf: [std.fs.max_path_bytes]u8 = undefined;
                const n = try dir.readLink(c.io, entry.path, &buf);
                try list.append(c.a, try std.fmt.allocPrint(c.a, "{s} -> {s}", .{ entry.path, buf[0..n] }));
            },
            else => {},
        };
        std.mem.sort([]const u8, list.items, {}, struct {
            fn lt(_: void, x: []const u8, y: []const u8) bool {
                return std.mem.lessThan(u8, x, y);
            }
        }.lt);
        return list.items;
    }
};

fn expectRan(name: []const u8, r: Ran, code: u8, stdout: ?[]const u8) void {
    check(r.code == code, "{s}: exit {d}, want {d}\n{s}{s}", .{ name, r.code, code, r.stdout, r.stderr });
    if (stdout) |want| check(std.mem.eql(u8, r.stdout, want), "{s}: stdout '{s}', want '{s}'", .{ name, r.stdout, want });
}

fn contains(name: []const u8, hay: []const u8, needle: []const u8) void {
    check(std.mem.indexOf(u8, hay, needle) != null, "{s}: expected '{s}' in '{s}'", .{ name, needle, hay });
}

fn pinsFile(a: std.mem.Allocator, wheel_name: []const u8, sha: []const u8) ![]u8 {
    return std.fmt.allocPrint(a,
        \\# synthetic pins
        \\ROCM_INDEX=http://127.0.0.1:9/whl/
        \\ROCM_VERSION=0.0.test
        \\ROCM_WHEEL={s}
        \\ROCM_WHEEL_SHA256={s}
        \\
    , .{ wheel_name, sha });
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len != 3) {
        std.debug.print("usage: fetch_rocm_check FETCH DIR\n", .{});
        return 2;
    }
    const cwd = std.Io.Dir.cwd();
    const fetch = try cwd.realPathFileAlloc(io, argv[1], a);
    try cwd.createDirPath(io, argv[2]);
    const dir = try cwd.realPathFileAlloc(io, argv[2], a);
    var c: Ctx = .{ .a = a, .io = io, .base_env = init.environ_map, .fetch = fetch };

    // the fixture: the wheel, its pins, a stand-in for the wrapper
    const fixture = try std.fmt.allocPrint(a, "{s}/fixture", .{dir});
    try cwd.createDirPath(io, fixture);
    const wheel_name = "rocm_sdk_core-0.0.test-py3-none-linux_x86_64.whl";
    const big = try bigHeader(a);
    const ocml = try noise(a, 70_000, 1);
    const ockl = try noise(a, 3_000, 2);
    const nested = try deflated(a, "BC nested");
    const members = [_]Member{
        .{ .name = "_rocm_sdk_core/include/hip/hip_version.h", .data = version_header, .deflate = true },
        .{ .name = "_rocm_sdk_core/include/hip/hip_runtime.h", .data = "#include <hip/hip_version.h>\n" },
        .{ .name = "_rocm_sdk_core/include/hip/", .data = "" },
        .{ .name = "_rocm_sdk_core/include/hip/amd_detail/amd_hip_runtime.h", .data = big, .deflate = true },
        .{ .name = "_rocm_sdk_core/include/rocblas/rocblas.h", .data = "skipped\n", .deflate = true },
        .{ .name = "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ocml.bc", .data = ocml, .deflate = true },
        .{ .name = "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ockl.bc", .data = ockl },
        .{ .name = "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/sub/nested.bc", .data = nested },
        .{ .name = "_rocm_sdk_core/lib/libamdhip64.so.7", .data = ockl },
        .{ .name = "_rocm_sdk_core/bin/clang", .data = big, .deflate = true },
        .{ .name = "rocm_sdk_core-0.0.test.dist-info/METADATA", .data = "Name: rocm-sdk-core\n", .deflate = true },
    };
    const wheel = try buildZip(a, &members);
    const wheel_path = try std.fmt.allocPrint(a, "{s}/{s}", .{ fixture, wheel_name });
    try c.write(wheel_path, wheel);
    var digest: [Sha256.digest_length]u8 = undefined;
    Sha256.hash(wheel, &digest, .{});
    const sha = std.fmt.bytesToHex(digest, .lower);
    const pins_path = try std.fmt.allocPrint(a, "{s}/pins.env", .{fixture});
    try c.write(pins_path, try pinsFile(a, wheel_name, &sha));
    const stub = try std.fmt.allocPrint(a, "{s}/tf-hipcc", .{fixture});
    try cwd.writeFile(io, .{ .sub_path = stub, .data = "#!/bin/sh\n", .flags = .{ .permissions = .executable_file } });

    const root = try std.fmt.allocPrint(a, "{s}/root", .{dir});
    const want_tree = [_][]const u8{
        ".pinned",
        try std.fmt.allocPrint(a, "bin/.hipVersion", .{}),
        try std.fmt.allocPrint(a, "bin/hipcc -> {s}", .{stub}),
        "include/hip/amd_detail/amd_hip_runtime.h",
        "include/hip/hip_runtime.h",
        "include/hip/hip_version.h",
        "lib/llvm/amdgcn/bitcode/nested.bc",
        "lib/llvm/amdgcn/bitcode/ockl.bc",
        "lib/llvm/amdgcn/bitcode/ocml.bc",
    };
    const root_env = [_][2][]const u8{.{ "TF_VEGA_ROOT", root }};
    const common = [_][]const u8{ "--wheel", wheel_path, "--pins", pins_path, "--hipcc", stub };

    // 1. a fresh install: stdout is the root, the tree is exactly the headers and the bitcode (flattened) plus the version file, .pinned and the link
    {
        const r = try c.run(&common, &root_env);
        expectRan("install", r, 0, try std.fmt.allocPrint(a, "{s}\n", .{root}));
        contains("install", r.stderr, "installed 3 header files and 3 bitcode files");
        const got = try c.tree(root);
        var ok = got.len == want_tree.len;
        if (ok) for (got, want_tree) |g, w| {
            if (!std.mem.eql(u8, g, w)) ok = false;
        };
        if (!ok) {
            std.debug.print("fetch_rocm_check: FAIL install: tree\n", .{});
            for (got) |g| std.debug.print("  got  {s}\n", .{g});
            for (want_tree) |w| std.debug.print("  want {s}\n", .{w});
            failures += 1;
        }
        const contents = [_]struct { []const u8, []const u8 }{
            .{ "include/hip/hip_version.h", version_header },
            .{ "include/hip/hip_runtime.h", "#include <hip/hip_version.h>\n" },
            .{ "include/hip/amd_detail/amd_hip_runtime.h", big },
            .{ "lib/llvm/amdgcn/bitcode/ocml.bc", ocml },
            .{ "lib/llvm/amdgcn/bitcode/ockl.bc", ockl },
            .{ "lib/llvm/amdgcn/bitcode/nested.bc", nested },
            .{ "bin/.hipVersion", hip_version_text },
            .{ ".pinned", &sha },
        };
        for (contents) |kv| {
            const data = c.readAll(try std.fmt.allocPrint(a, "{s}/{s}", .{ root, kv[0] })) catch |e| {
                check(false, "install: cannot read {s}: {s}", .{ kv[0], @errorName(e) });
                continue;
            };
            check(std.mem.eql(u8, data, kv[1]), "install: {s} differs ({d} bytes, want {d})", .{ kv[0], data.len, kv[1].len });
        }
        check(!c.exists(try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, wheel_name })), "install: the wheel was kept without --keep", .{});
    }
    // 2. a second run finds the root installed for this pin: prints it, quietly, and touches nothing
    {
        try c.write(try std.fmt.allocPrint(a, "{s}/marker", .{root}), "x");
        const r = try c.run(&common, &root_env);
        expectRan("second run", r, 0, try std.fmt.allocPrint(a, "{s}\n", .{root}));
        check(r.stderr.len == 0, "second run: stderr '{s}'", .{r.stderr});
        check(c.exists(try std.fmt.allocPrint(a, "{s}/marker", .{root})), "second run: reinstalled", .{});
        // without a wheel to read: the early exit needs no download and no wheel
        const r2 = try c.run(&.{ "--pins", pins_path }, &root_env);
        expectRan("early exit without wheel", r2, 0, try std.fmt.allocPrint(a, "{s}\n", .{root}));
    }
    // 3. another pinned sha256 (or a missing .pinned): installs again, over the old tree
    {
        try c.write(try std.fmt.allocPrint(a, "{s}/.pinned", .{root}), "stale");
        const r = try c.run(&common, &root_env);
        expectRan("reinstall", r, 0, try std.fmt.allocPrint(a, "{s}\n", .{root}));
        check(!c.exists(try std.fmt.allocPrint(a, "{s}/marker", .{root})), "reinstall: the old tree stayed", .{});
        check(std.mem.eql(u8, try c.readAll(try std.fmt.allocPrint(a, "{s}/.pinned", .{root})), &sha), "reinstall: .pinned", .{});
    }
    // 4. sha256 mismatch: exit 1, the message, the root untouched
    {
        const other = try std.fmt.allocPrint(a, "{s}/other-pins.env", .{fixture});
        const bogus: [64]u8 = @splat('0');
        try c.write(other, try pinsFile(a, wheel_name, &bogus));
        try c.write(try std.fmt.allocPrint(a, "{s}/marker", .{root}), "x");
        const r = try c.run(&.{ "--wheel", wheel_path, "--pins", other, "--hipcc", stub }, &root_env);
        expectRan("sha mismatch", r, 1, "");
        check(std.mem.eql(u8, r.stderr, try std.fmt.allocPrint(a, "fetch-rocm: {s} sha256 {s} != pinned {s}\n", .{ wheel_path, &sha, bogus })), "sha mismatch: stderr '{s}'", .{r.stderr});
        check(c.exists(try std.fmt.allocPrint(a, "{s}/marker", .{root})), "sha mismatch: the root was touched", .{});
    }
    // 5. a wheel that is not there
    {
        const r = try c.run(&.{ "--wheel", try std.fmt.allocPrint(a, "{s}/absent.whl", .{fixture}), "--pins", pins_path, "--hipcc", stub }, &.{.{ "TF_VEGA_ROOT", try std.fmt.allocPrint(a, "{s}/root-missing", .{dir}) }});
        expectRan("missing wheel", r, 1, "");
        contains("missing wheel", r.stderr, "fetch-rocm: cannot read ");
        contains("missing wheel", r.stderr, "absent.whl");
    }
    // 6. --keep keeps the wheel next to the root (and not otherwise); the root's parent is DIR/keep here
    {
        const keep_root = try std.fmt.allocPrint(a, "{s}/keep/root", .{dir});
        try cwd.createDirPath(io, try std.fmt.allocPrint(a, "{s}/keep", .{dir}));
        const r = try c.run(&(common ++ [_][]const u8{"--keep"}), &.{.{ "TF_VEGA_ROOT", keep_root }});
        expectRan("keep", r, 0, try std.fmt.allocPrint(a, "{s}\n", .{keep_root}));
        const kept = c.readAll(try std.fmt.allocPrint(a, "{s}/keep/{s}", .{ dir, wheel_name })) catch "";
        check(std.mem.eql(u8, kept, wheel), "keep: the kept wheel differs ({d} bytes)", .{kept.len});
        check(std.mem.eql(u8, try c.readAll(wheel_path), wheel), "keep: the source wheel changed", .{});
    }
    // 7. the default root is ~/.cache/tensorfold-vega/rocm-<ROCM_VERSION>; the wrapper link defaults to tf-hipcc beside the running executable
    {
        const home = try std.fmt.allocPrint(a, "{s}/home", .{dir});
        const bin = try std.fmt.allocPrint(a, "{s}/bin", .{dir});
        try cwd.createDirPath(io, bin);
        const exe = try std.fmt.allocPrint(a, "{s}/tf-fetch-rocm", .{bin});
        try cwd.copyFile(fetch, cwd, exe, io, .{});
        const hipcc_beside = try std.fmt.allocPrint(a, "{s}/tf-hipcc", .{bin});
        const r0 = try c.runExe(exe, &.{ "--wheel", wheel_path, "--pins", pins_path }, &.{.{ "HOME", home }});
        expectRan("no wrapper beside", r0, 1, "");
        contains("no wrapper beside", r0.stderr, try std.fmt.allocPrint(a, "fetch-rocm: the hipcc wrapper {s} does not exist", .{hipcc_beside}));
        try cwd.copyFile(stub, cwd, hipcc_beside, io, .{});
        const r = try c.runExe(exe, &.{ "--wheel", wheel_path, "--pins", pins_path }, &.{.{ "HOME", home }});
        const def_root = try std.fmt.allocPrint(a, "{s}/.cache/tensorfold-vega/rocm-0.0.test", .{home});
        expectRan("default root", r, 0, try std.fmt.allocPrint(a, "{s}\n", .{def_root}));
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const n = cwd.readLink(io, try std.fmt.allocPrint(a, "{s}/bin/hipcc", .{def_root}), &buf) catch 0;
        check(std.mem.eql(u8, buf[0..n], hipcc_beside), "default wrapper link '{s}', want {s}", .{ buf[0..n], hipcc_beside });
    }
    // 8. a file that is no zip, with the right sha256; a member with a wrong CRC-32
    {
        const junk = "this is not a wheel";
        var d: [Sha256.digest_length]u8 = undefined;
        Sha256.hash(junk, &d, .{});
        const junk_sha = std.fmt.bytesToHex(d, .lower);
        const junk_path = try std.fmt.allocPrint(a, "{s}/junk.whl", .{fixture});
        const junk_pins = try std.fmt.allocPrint(a, "{s}/junk-pins.env", .{fixture});
        try c.write(junk_path, junk);
        try c.write(junk_pins, try pinsFile(a, "junk.whl", &junk_sha));
        const r = try c.run(&.{ "--wheel", junk_path, "--pins", junk_pins, "--hipcc", stub }, &.{.{ "TF_VEGA_ROOT", try std.fmt.allocPrint(a, "{s}/root-junk", .{dir}) }});
        expectRan("not a zip", r, 1, "");
        contains("not a zip", r.stderr, "is not a zip file");

        const bad = try buildZip(a, &.{.{ .name = "_rocm_sdk_core/include/hip/hip_version.h", .data = version_header, .deflate = true, .bad_crc = true }});
        Sha256.hash(bad, &d, .{});
        const bad_sha = std.fmt.bytesToHex(d, .lower);
        const bad_path = try std.fmt.allocPrint(a, "{s}/bad.whl", .{fixture});
        const bad_pins = try std.fmt.allocPrint(a, "{s}/bad-pins.env", .{fixture});
        try c.write(bad_path, bad);
        try c.write(bad_pins, try pinsFile(a, "bad.whl", &bad_sha));
        const r2 = try c.run(&.{ "--wheel", bad_path, "--pins", bad_pins, "--hipcc", stub }, &.{.{ "TF_VEGA_ROOT", try std.fmt.allocPrint(a, "{s}/root-bad", .{dir}) }});
        expectRan("bad crc", r2, 1, "");
        contains("bad crc", r2.stderr, "bad CRC-32");
    }
    // 9. arguments: unknown, missing value, an incomplete pins file, a missing pins file
    {
        const r = try c.run(&.{"--bogus"}, &root_env);
        expectRan("unknown argument", r, 2, "");
        check(std.mem.eql(u8, r.stderr, "fetch-rocm: unknown argument --bogus\n"), "unknown argument: stderr '{s}'", .{r.stderr});
        const r2 = try c.run(&.{"--wheel"}, &root_env);
        expectRan("missing value", r2, 2, "");
        contains("missing value", r2.stderr, "--wheel needs a value");
        const part = try std.fmt.allocPrint(a, "{s}/partial-pins.env", .{fixture});
        try c.write(part, "ROCM_INDEX=x\nROCM_VERSION=y\n");
        const r3 = try c.run(&.{ "--pins", part }, &root_env);
        expectRan("incomplete pins", r3, 1, "");
        contains("incomplete pins", r3.stderr, "ROCM_WHEEL is not set");
        const r4 = try c.run(&.{ "--pins", try std.fmt.allocPrint(a, "{s}/absent.env", .{fixture}) }, &root_env);
        expectRan("missing pins file", r4, 1, "");
        contains("missing pins file", r4.stderr, "cannot read ");
    }

    if (failures > 0) {
        std.debug.print("fetch_rocm_check: {d} failure(s)\n", .{failures});
        return 1;
    }
    try std.Io.File.stdout().writeStreamingAll(io, "fetch_rocm_check: ok\n");
    return 0;
}
