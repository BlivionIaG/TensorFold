//! tf-fetch-rocm: install the HIP headers and the ROCm device libraries a macOS host needs to build TensorFold's gfx900 (Vega) kernels with Homebrew LLVM.
//!   tf-fetch-rocm [--wheel PATH] [--keep] [--pins FILE] [--hipcc PATH]      -> $TF_VEGA_ROOT (default ~/.cache/tensorfold-vega/rocm-<version>)
//! The Zig port of the former tools/vega-macos/fetch-rocm.sh. Layout (what zig/build/hip.zig expects of a ROCm root, so -Dhipcc=<root>/bin/hipcc is all the build needs):
//!   <root>/bin/hipcc                      symlink to the tf-hipcc executable (the wrapper, built next to this tool; --hipcc PATH names another);
//!                                         <root>/bin/.hipVersion lets clang find the HIP wrapper header
//!   <root>/include/hip/**                 HIP 7.x headers (hip_runtime.h, amd_detail/...)
//!   <root>/lib/llvm/amdgcn/bitcode/*.bc   ocml, ockl, oclc_*, hip, opencl bitcode
//! Needs `brew install llvm lld` for the compiles, nothing else: no python, no curl, no sudo. The pinned wheel (416 MB, pins.env, embedded here; --pins FILE
//! reads another) is downloaded over https, hashed while it streams to a temporary file, verified, and only the members above are inflated from it.
//! It is kept (next to the root) only when --keep is given; --wheel PATH installs from a local copy instead of downloading.
//! stdout is the root and nothing else; progress and diagnostics go to stderr. Exit: 0 done (or already installed for this pin), 1 failure, 2 bad arguments.
const std = @import("std");
const Sha256 = std.crypto.hash.sha2.Sha256;
const Crc32 = std.hash.Crc32;

const embedded_pins = @embedFile("pins.env");

const usage =
    \\usage: tf-fetch-rocm [--wheel PATH] [--keep] [--pins FILE] [--hipcc PATH]
    \\  installs the HIP headers and ROCm device bitcode into $TF_VEGA_ROOT (default ~/.cache/tensorfold-vega/rocm-<version>) and prints that root
    \\  --wheel PATH   install from this wheel instead of downloading the pinned one (it is still verified against the pin)
    \\  --keep         keep the wheel in the directory above the root
    \\  --pins FILE    read the pins (KEY=VALUE lines: ROCM_INDEX, ROCM_VERSION, ROCM_WHEEL, ROCM_WHEEL_SHA256) from FILE instead of the built-in pins.env
    \\  --hipcc PATH   make <root>/bin/hipcc a link to PATH instead of the tf-hipcc executable beside this one
    \\
;

pub const Pins = struct {
    index: []const u8,
    version: []const u8,
    wheel: []const u8,
    sha256: []const u8,
};

/// The value of `key` in KEY=VALUE `text` (the last assignment wins like in a sourced shell file; one pair of quotes around the value is dropped).
fn pinValue(text: []const u8, key: []const u8) ?[]const u8 {
    var found: ?[]const u8 = null;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        const line = std.mem.trim(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        const eq = std.mem.indexOfScalar(u8, line, '=') orelse continue;
        if (!std.mem.eql(u8, std.mem.trim(u8, line[0..eq], " \t"), key)) continue;
        var v = std.mem.trim(u8, line[eq + 1 ..], " \t");
        if (v.len >= 2 and (v[0] == '"' or v[0] == '\'') and v[v.len - 1] == v[0]) v = v[1 .. v.len - 1];
        found = v;
    }
    return found;
}

/// The pins of a pins.env text; `missing` is the first key it lacks.
pub fn parsePins(text: []const u8, missing: *[]const u8) !Pins {
    inline for (.{ "ROCM_INDEX", "ROCM_VERSION", "ROCM_WHEEL", "ROCM_WHEEL_SHA256" }) |key| {
        if (pinValue(text, key) == null) {
            missing.* = key;
            return error.MissingPin;
        }
    }
    return .{
        .index = pinValue(text, "ROCM_INDEX").?,
        .version = pinValue(text, "ROCM_VERSION").?,
        .wheel = pinValue(text, "ROCM_WHEEL").?,
        .sha256 = pinValue(text, "ROCM_WHEEL_SHA256").?,
    };
}

pub const include_prefix = "_rocm_sdk_core/include/hip/";
pub const bitcode_prefix = "_rocm_sdk_core/lib/llvm/amdgcn/bitcode/";

/// Where a wheel member goes below the root, or null when it is not installed: the headers keep their layout below include/hip,
/// the bitcode files are flattened into lib/llvm/amdgcn/bitcode, directories and everything else are skipped.
pub fn destination(name: []const u8, buf: []u8) !?[]const u8 {
    if (name.len == 0 or name[name.len - 1] == '/') return null;
    if (std.mem.startsWith(u8, name, include_prefix)) {
        return try std.fmt.bufPrint(buf, "include/hip/{s}", .{name[include_prefix.len..]});
    }
    if (std.mem.startsWith(u8, name, bitcode_prefix)) {
        return try std.fmt.bufPrint(buf, "lib/llvm/amdgcn/bitcode/{s}", .{name[std.mem.lastIndexOfScalar(u8, name, '/').? + 1 ..]});
    }
    return null;
}

fn isSpace(c: u8) bool {
    return c == ' ' or (c >= '\t' and c <= '\r');
}

/// The digits after the first `#define HIP_VERSION_<which>` followed by white space and a number (what `re.search(r'#define HIP_VERSION_X\s+(\d+)')` finds).
fn defineNumber(header: []const u8, which: []const u8) ?[]const u8 {
    var buf: [64]u8 = undefined;
    const needle = std.fmt.bufPrint(&buf, "#define HIP_VERSION_{s}", .{which}) catch return null;
    var from: usize = 0;
    while (std.mem.indexOfPos(u8, header, from, needle)) |at| {
        from = at + 1;
        var i = at + needle.len;
        const ws = i;
        while (i < header.len and isSpace(header[i])) i += 1;
        if (i == ws) continue;
        const start = i;
        while (i < header.len and header[i] >= '0' and header[i] <= '9') i += 1;
        if (i > start) return header[start..i];
    }
    return null;
}

/// The text of bin/.hipVersion for the include/hip/hip_version.h text `header`: clang only adds its __clang_hip_runtime_wrapper.h (the device math: fmaf,
/// expf, ... next to the host libm) once it reads a HIP version there.
pub fn hipVersionText(a: std.mem.Allocator, header: []const u8, missing: *[]const u8) ![]u8 {
    var out: std.Io.Writer.Allocating = .init(a);
    errdefer out.deinit();
    inline for (.{ "MAJOR", "MINOR", "PATCH" }) |which| {
        const v = defineNumber(header, which) orelse {
            missing.* = "HIP_VERSION_" ++ which;
            return error.NoHipVersion;
        };
        try out.writer.print("HIP_VERSION_{s}={s}\n", .{ which, v });
    }
    try out.writer.writeAll("HIP_VERSION_GITHASH=0\n");
    return out.toOwnedSlice();
}

const Fail = error{Failed};

const Ctx = struct {
    a: std.mem.Allocator,
    io: std.Io,
    err: *std.Io.Writer,

    fn say(c: Ctx, comptime fmt: []const u8, args: anytype) void {
        c.err.print("fetch-rocm: " ++ fmt ++ "\n", args) catch {};
        c.err.flush() catch {};
    }

    fn fail(c: Ctx, comptime fmt: []const u8, args: anytype) Fail {
        c.say(fmt, args);
        return error.Failed;
    }
};

fn cwd() std.Io.Dir {
    return std.Io.Dir.cwd();
}

fn exists(io: std.Io, path: []const u8, kind: std.Io.File.Kind) bool {
    const st = cwd().statFile(io, path, .{}) catch return false;
    return st.kind == kind;
}

/// Already installed for this pin: the wrapper link works, the headers and the bitcode are there and `.pinned` is the pinned sha256.
fn installed(c: Ctx, root: []const u8, pins: Pins) !bool {
    const p = struct {
        fn join(ctx: Ctx, r: []const u8, rel: []const u8) ![]u8 {
            return std.fmt.allocPrint(ctx.a, "{s}/{s}", .{ r, rel });
        }
    };
    cwd().access(c.io, try p.join(c, root, "bin/hipcc"), .{ .execute = true }) catch return false;
    if (!exists(c.io, try p.join(c, root, "include/hip/hip_runtime.h"), .file)) return false;
    if (!exists(c.io, try p.join(c, root, "lib/llvm/amdgcn/bitcode/ocml.bc"), .file)) return false;
    const pinned = cwd().readFileAlloc(c.io, try p.join(c, root, ".pinned"), c.a, .limited(4096)) catch return false;
    return std.mem.eql(u8, std.mem.trimEnd(u8, pinned, "\n"), pins.sha256);
}

fn hex(digest: [Sha256.digest_length]u8) [Sha256.digest_length * 2]u8 {
    return std.fmt.bytesToHex(digest, .lower);
}

fn sha256OfFile(c: Ctx, path: []const u8) ![Sha256.digest_length * 2]u8 {
    var file = cwd().openFile(c.io, path, .{}) catch |e| return c.fail("cannot read {s}: {s}", .{ path, @errorName(e) });
    defer file.close(c.io);
    var rbuf: [1 << 16]u8 = undefined;
    var reader = file.reader(c.io, &rbuf);
    var hasher: Sha256 = .init(.{});
    var chunk: [1 << 16]u8 = undefined;
    while (true) {
        const n = reader.interface.readSliceShort(&chunk) catch |e| return c.fail("cannot read {s}: {s}", .{ path, @errorName(e) });
        hasher.update(chunk[0..n]);
        if (n < chunk.len) break;
    }
    return hex(hasher.finalResult());
}

const Attempt = union(enum) {
    done: [Sha256.digest_length * 2]u8,
    /// a failure worth another try (the network, a 5xx)
    retry: []const u8,
    /// a failure that another try will not change
    fatal: []const u8,
};

const mib = 1 << 20;

fn downloadOnce(c: Ctx, client: *std.http.Client, uri: std.Uri, dest: []const u8) !Attempt {
    var req = client.request(.GET, uri, .{ .headers = .{ .accept_encoding = .{ .override = "identity" } } }) catch |e| switch (e) {
        error.UnsupportedUriScheme, error.UriMissingHost => return .{ .fatal = @errorName(e) },
        else => return .{ .retry = @errorName(e) },
    };
    defer req.deinit();
    req.sendBodiless() catch |e| return .{ .retry = @errorName(e) };
    var redirect_buf: [8 * 1024]u8 = undefined;
    var response = req.receiveHead(&redirect_buf) catch |e| return .{ .retry = @errorName(e) };
    const status = @backingInt(response.head.status);
    if (status < 200 or status >= 300) {
        const msg = try std.fmt.allocPrint(c.a, "the server answered HTTP {d}", .{status});
        return if (status == 408 or status == 429 or status >= 500) .{ .retry = msg } else .{ .fatal = msg };
    }
    const want = response.head.content_length;

    var file = try cwd().createFile(c.io, dest, .{});
    defer file.close(c.io);
    var wbuf: [1 << 16]u8 = undefined;
    var fw = file.writer(c.io, &wbuf);
    var hbuf: [1 << 16]u8 = undefined;
    var hashed = std.Io.Writer.Hashed(Sha256).initHasher(&fw.interface, Sha256.init(.{}), &hbuf);
    var transfer: [4096]u8 = undefined;
    const body = response.reader(&transfer);
    var total: u64 = 0;
    var next_report: u64 = 32 * mib;
    while (true) {
        const n = body.stream(&hashed.writer, .limited(mib)) catch |e| switch (e) {
            error.EndOfStream => break,
            error.ReadFailed => return .{ .retry = if (response.bodyErr()) |be| @errorName(be) else "read failed" },
            error.WriteFailed => return c.fail("cannot write {s}: {s}", .{ dest, if (fw.err) |we| @errorName(we) else "write failed" }),
        };
        total += n;
        if (total >= next_report) {
            if (want) |w| c.say("{d} of {d} MiB", .{ total / mib, w / mib }) else c.say("{d} MiB", .{total / mib});
            next_report += 32 * mib;
        }
    }
    hashed.writer.flush() catch return c.fail("cannot write {s}: {s}", .{ dest, if (fw.err) |we| @errorName(we) else "write failed" });
    fw.interface.flush() catch return c.fail("cannot write {s}: {s}", .{ dest, if (fw.err) |we| @errorName(we) else "write failed" });
    if (want) |w| if (total != w) return .{ .retry = try std.fmt.allocPrint(c.a, "the body ended after {d} of {d} bytes", .{ total, w }) };
    return .{ .done = hex(hashed.hasher.finalResult()) };
}

/// `curl -fL --retry 3`: GET `url` into `dest` (redirects followed), hashing while it streams; a transient failure is tried again up to 3 times after 1, 2, 4 s.
fn download(c: Ctx, environ: *const std.process.Environ.Map, url: []const u8, dest: []const u8) ![Sha256.digest_length * 2]u8 {
    const uri = std.Uri.parse(url) catch |e| return c.fail("bad URL {s}: {s}", .{ url, @errorName(e) });
    var client: std.http.Client = .{ .allocator = c.a, .io = c.io };
    defer client.deinit();
    client.initDefaultProxies(c.a, environ) catch {};
    var retries: u32 = 0;
    while (true) {
        c.say("downloading {s}", .{url});
        const why = switch (try downloadOnce(c, &client, uri, dest)) {
            .done => |sha| return sha,
            .fatal => |msg| return c.fail("download of {s} failed: {s}", .{ url, msg }),
            .retry => |msg| msg,
        };
        if (retries == 3) return c.fail("download of {s} failed: {s}", .{ url, why });
        const wait: i64 = @as(i64, 1) << @intCast(retries);
        retries += 1;
        c.say("download failed ({s}); retry {d} of 3 in {d}s", .{ why, retries, wait });
        c.io.sleep(.fromSeconds(wait), .awake) catch {};
    }
}

fn writeFile(c: Ctx, root: []const u8, rel: []const u8, data: []const u8) !void {
    const path = try std.fmt.allocPrint(c.a, "{s}/{s}", .{ root, rel });
    cwd().writeFile(c.io, .{ .sub_path = path, .data = data }) catch |e| return c.fail("cannot write {s}: {s}", .{ path, @errorName(e) });
}

const Counts = struct { headers: usize = 0, bitcode: usize = 0 };

/// Inflate the headers and the bitcode out of the wheel straight into the root (no full extraction).
fn extractWheel(c: Ctx, wheel: []const u8, root: []const u8) !Counts {
    var file = cwd().openFile(c.io, wheel, .{}) catch |e| return c.fail("cannot read {s}: {s}", .{ wheel, @errorName(e) });
    defer file.close(c.io);
    var rbuf: [1 << 16]u8 = undefined;
    var fr = file.reader(c.io, &rbuf);
    var it = std.zip.Iterator.init(&fr) catch |e| return c.fail("{s} is not a zip file: {s}", .{ wheel, @errorName(e) });
    var counts: Counts = .{};
    var name_buf: [std.math.maxInt(u16)]u8 = undefined;
    var dest_buf: [std.math.maxInt(u16) + 64]u8 = undefined;
    while (it.next() catch |e| return c.fail("{s}: bad zip directory: {s}", .{ wheel, @errorName(e) })) |entry| {
        const name = name_buf[0..entry.filename_len];
        fr.seekTo(entry.header_zip_offset + @sizeOf(std.zip.CentralDirectoryFileHeader)) catch |e| return c.fail("{s}: {s}", .{ wheel, @errorName(e) });
        fr.interface.readSliceAll(name) catch return c.fail("{s}: truncated zip directory", .{wheel});
        const rel = (destination(name, &dest_buf) catch return c.fail("{s}: member name too long", .{wheel})) orelse continue;
        var parts = std.mem.splitScalar(u8, name, '/');
        while (parts.next()) |part| if (std.mem.eql(u8, part, "..")) return c.fail("{s}: unsafe member name {s}", .{ wheel, name });

        const out = try std.fmt.allocPrint(c.a, "{s}/{s}", .{ root, rel });
        defer c.a.free(out);
        if (std.fs.path.dirname(out)) |dir| cwd().createDirPath(c.io, dir) catch |e| return c.fail("cannot create {s}: {s}", .{ dir, @errorName(e) });
        var dest = cwd().createFile(c.io, out, .{}) catch |e| return c.fail("cannot create {s}: {s}", .{ out, @errorName(e) });
        defer dest.close(c.io);
        var wbuf: [1 << 16]u8 = undefined;
        var fw = dest.writer(c.io, &wbuf);
        var crc_buf: [4096]u8 = undefined;
        var crc = std.Io.Writer.Hashed(Crc32).initHasher(&fw.interface, Crc32.init(), &crc_buf);
        entry.extractTo(&fr, &crc.writer) catch |e| return c.fail("{s}: member {s}: {s}", .{ wheel, name, @errorName(e) });
        crc.writer.flush() catch |e| return c.fail("cannot write {s}: {s}", .{ out, @errorName(e) });
        fw.interface.flush() catch |e| return c.fail("cannot write {s}: {s}", .{ out, @errorName(e) });
        if (crc.hasher.final() != entry.crc32) return c.fail("{s}: member {s}: bad CRC-32", .{ wheel, name });
        if (std.mem.startsWith(u8, name, include_prefix)) counts.headers += 1 else counts.bitcode += 1;
    }
    return counts;
}

const Options = struct {
    wheel: ?[]const u8 = null,
    keep: bool = false,
    pins_file: ?[]const u8 = null,
    hipcc: ?[]const u8 = null,
};

/// The wrapper `<root>/bin/hipcc` links to, as an absolute path: --hipcc, else tf-hipcc beside this executable.
fn hipccTarget(c: Ctx, opt: Options) ![]const u8 {
    const given = if (opt.hipcc) |h| h else blk: {
        const dir = std.process.executableDirPathAlloc(c.io, c.a) catch |e| return c.fail("cannot find the directory of this executable ({s}); pass --hipcc PATH", .{@errorName(e)});
        break :blk try std.fmt.allocPrint(c.a, "{s}/tf-hipcc", .{dir});
    };
    if (!exists(c.io, given, .file)) return c.fail("the hipcc wrapper {s} does not exist (zig build tf-hipcc, or pass --hipcc PATH)", .{given});
    if (std.fs.path.isAbsolute(given)) return given;
    const here = std.process.currentPathAlloc(c.io, c.a) catch |e| return c.fail("cannot get the working directory: {s}", .{@errorName(e)});
    return std.fs.path.resolveAlloc(c.a, &.{ here, given });
}

fn run(c: Ctx, environ: *const std.process.Environ.Map, opt: Options, out: *std.Io.Writer) !u8 {
    const pins_text = if (opt.pins_file) |p| cwd().readFileAlloc(c.io, p, c.a, .limited(1 << 20)) catch |e| return c.fail("cannot read {s}: {s}", .{ p, @errorName(e) }) else embedded_pins;
    var missing: []const u8 = "";
    const pins = parsePins(pins_text, &missing) catch return c.fail("{s}: {s} is not set", .{ opt.pins_file orelse "pins.env", missing });

    const root = blk: {
        if (environ.get("TF_VEGA_ROOT")) |r| if (r.len > 0) break :blk r;
        const home = environ.get("HOME") orelse return c.fail("HOME is not set (set TF_VEGA_ROOT)", .{});
        break :blk try std.fmt.allocPrint(c.a, "{s}/.cache/tensorfold-vega/rocm-{s}", .{ home, pins.version });
    };

    if (try installed(c, root, pins)) {
        try out.print("{s}\n", .{root});
        return 0;
    }
    const hipcc = try hipccTarget(c, opt);

    // a temporary directory for the download, removed on the way out
    var tmp_dir: ?[]const u8 = null;
    defer if (tmp_dir) |d| cwd().deleteTree(c.io, d) catch {};
    var wheel: []const u8 = undefined;
    var got: [Sha256.digest_length * 2]u8 = undefined;
    if (opt.wheel) |w| {
        wheel = w;
        got = try sha256OfFile(c, wheel);
    } else {
        var rnd: [6]u8 = undefined;
        c.io.random(&rnd);
        const base = if (environ.get("TMPDIR")) |t| (if (t.len > 0) std.mem.trimEnd(u8, t, "/") else "/tmp") else "/tmp";
        const dir = try std.fmt.allocPrint(c.a, "{s}/tf-fetch-rocm.{x}", .{ base, rnd });
        cwd().createDirPath(c.io, dir) catch |e| return c.fail("cannot create {s}: {s}", .{ dir, @errorName(e) });
        tmp_dir = dir;
        wheel = try std.fmt.allocPrint(c.a, "{s}/{s}", .{ dir, pins.wheel });
        got = try download(c, environ, try std.fmt.allocPrint(c.a, "{s}{s}", .{ pins.index, pins.wheel }), wheel);
    }
    if (!std.mem.eql(u8, &got, pins.sha256)) return c.fail("{s} sha256 {s} != pinned {s}", .{ wheel, got, pins.sha256 });

    cwd().deleteTree(c.io, root) catch |e| return c.fail("cannot remove {s}: {s}", .{ root, @errorName(e) });
    const bin = try std.fmt.allocPrint(c.a, "{s}/bin", .{root});
    cwd().createDirPath(c.io, bin) catch |e| return c.fail("cannot create {s}: {s}", .{ bin, @errorName(e) });

    const counts = try extractWheel(c, wheel, root);
    c.say("installed {d} header files and {d} bitcode files", .{ counts.headers, counts.bitcode });

    const header_path = try std.fmt.allocPrint(c.a, "{s}/include/hip/hip_version.h", .{root});
    const header = cwd().readFileAlloc(c.io, header_path, c.a, .limited(1 << 20)) catch |e| return c.fail("cannot read {s}: {s}", .{ header_path, @errorName(e) });
    const version = hipVersionText(c.a, header, &missing) catch |e| switch (e) {
        error.NoHipVersion => return c.fail("no #define {s} <number> in {s}", .{ missing, header_path }),
        else => return e,
    };
    try writeFile(c, root, "bin/.hipVersion", version);
    const link = try std.fmt.allocPrint(c.a, "{s}/hipcc", .{bin});
    cwd().symLink(c.io, hipcc, link, .{}) catch |e| return c.fail("cannot link {s} to {s}: {s}", .{ link, hipcc, @errorName(e) });
    try writeFile(c, root, ".pinned", pins.sha256);

    if (opt.keep) keepWheel(c, wheel, root, tmp_dir != null);
    try out.print("{s}\n", .{root});
    return 0;
}

/// `cp WHEEL $(dirname ROOT)/`: a failure is reported but not fatal. A wheel downloaded here is moved when the directories are on one file system.
fn keepWheel(c: Ctx, wheel: []const u8, root: []const u8, downloaded: bool) void {
    const parent = std.fs.path.dirname(std.mem.trimEnd(u8, root, "/")) orelse ".";
    const dest = std.fmt.allocPrint(c.a, "{s}/{s}", .{ parent, std.fs.path.basename(wheel) }) catch return;
    const dst_stat = cwd().statFile(c.io, dest, .{}) catch null;
    const src_stat = cwd().statFile(c.io, wheel, .{}) catch null;
    if (dst_stat != null and src_stat != null and dst_stat.?.inode == src_stat.?.inode) return; // already there
    if (downloaded) {
        if (std.Io.Dir.rename(cwd(), wheel, cwd(), dest, c.io)) return else |_| {}
    }
    cwd().copyFile(wheel, cwd(), dest, c.io, .{}) catch |e| c.say("cannot keep {s} as {s}: {s}", .{ wheel, dest, @errorName(e) });
}

pub fn main(init: std.process.Init) !u8 {
    const io = init.io;
    var arena_state = std.heap.ArenaAllocator.init(init.gpa);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const argv = try init.minimal.args.toSlice(a);

    var out_buf: [1024]u8 = undefined;
    var err_buf: [1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writerStreaming(io, &out_buf);
    var stderr = std.Io.File.stderr().writerStreaming(io, &err_buf);
    const out = &stdout.interface;
    const err = &stderr.interface;
    const c: Ctx = .{ .a = a, .io = io, .err = err };

    var opt: Options = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (std.mem.eql(u8, arg, "-h") or std.mem.eql(u8, arg, "--help")) {
            try out.writeAll(usage);
            try out.flush();
            return 0;
        } else if (std.mem.eql(u8, arg, "--keep")) {
            opt.keep = true;
        } else for ([_]struct { []const u8, *?[]const u8 }{ .{ "--wheel", &opt.wheel }, .{ "--pins", &opt.pins_file }, .{ "--hipcc", &opt.hipcc } }) |o| {
            if (!std.mem.eql(u8, arg, o[0])) continue;
            i += 1;
            if (i == argv.len) {
                c.say("{s} needs a value", .{arg});
                return 2;
            }
            o[1].* = argv[i];
            break;
        } else {
            c.say("unknown argument {s}", .{arg});
            return 2;
        }
    }

    const code = run(c, init.environ_map, opt, out) catch |e| switch (e) {
        error.Failed => 1,
        else => blk: {
            c.say("{s}", .{@errorName(e)});
            break :blk 1;
        },
    };
    try out.flush();
    return code;
}

test "pins: parsed from pins.env text, last assignment wins, quotes dropped" {
    var missing: []const u8 = "";
    const pins = try parsePins(
        \\# comment
        \\ROCM_INDEX=https://example.test/whl/
        \\ROCM_VERSION="1.2.3"
        \\ROCM_WHEEL=w-1.2.3.whl
        \\ROCM_WHEEL=w-1.2.4.whl
        \\ROCM_WHEEL_SHA256=abc
        \\LLVM_VERSION=23
    , &missing);
    try std.testing.expectEqualStrings("https://example.test/whl/", pins.index);
    try std.testing.expectEqualStrings("1.2.3", pins.version);
    try std.testing.expectEqualStrings("w-1.2.4.whl", pins.wheel);
    try std.testing.expectEqualStrings("abc", pins.sha256);
    try std.testing.expectError(error.MissingPin, parsePins("ROCM_INDEX=x\nROCM_VERSION=y\n", &missing));
    try std.testing.expectEqualStrings("ROCM_WHEEL", missing);
}

test "the built-in pins.env has every pin and a sha256" {
    var missing: []const u8 = "";
    const pins = try parsePins(embedded_pins, &missing);
    try std.testing.expectEqual(@as(usize, 64), pins.sha256.len);
    try std.testing.expect(std.mem.endsWith(u8, pins.wheel, ".whl"));
    try std.testing.expect(std.mem.startsWith(u8, pins.index, "https://"));
}

test "destination: headers keep their layout, bitcode is flattened, the rest is skipped" {
    var buf: [256]u8 = undefined;
    try std.testing.expectEqualStrings("include/hip/hip_runtime.h", (try destination("_rocm_sdk_core/include/hip/hip_runtime.h", &buf)).?);
    try std.testing.expectEqualStrings("include/hip/amd_detail/a.h", (try destination("_rocm_sdk_core/include/hip/amd_detail/a.h", &buf)).?);
    try std.testing.expectEqualStrings("lib/llvm/amdgcn/bitcode/ocml.bc", (try destination("_rocm_sdk_core/lib/llvm/amdgcn/bitcode/ocml.bc", &buf)).?);
    try std.testing.expectEqualStrings("lib/llvm/amdgcn/bitcode/x.bc", (try destination("_rocm_sdk_core/lib/llvm/amdgcn/bitcode/sub/x.bc", &buf)).?);
    try std.testing.expectEqual(@as(?[]const u8, null), try destination("_rocm_sdk_core/include/hip/", &buf));
    try std.testing.expectEqual(@as(?[]const u8, null), try destination("_rocm_sdk_core/include/other.h", &buf));
    try std.testing.expectEqual(@as(?[]const u8, null), try destination("_rocm_sdk_core/lib/libamdhip64.so", &buf));
    try std.testing.expectEqual(@as(?[]const u8, null), try destination("rocm_sdk_core-1.dist-info/METADATA", &buf));
}

test "hipVersionText: the three defines and a zero git hash" {
    const a = std.testing.allocator;
    var missing: []const u8 = "";
    const text = try hipVersionText(a, "#ifndef X\n#define HIP_VERSION_MAJOR 7\n#define HIP_VERSION_MINOR\t16\n// #define HIP_VERSION_PATCH_X 1\n#define HIP_VERSION_PATCH 26332\n#define HIP_VERSION_PATCH 5\n", &missing);
    defer a.free(text);
    try std.testing.expectEqualStrings("HIP_VERSION_MAJOR=7\nHIP_VERSION_MINOR=16\nHIP_VERSION_PATCH=26332\nHIP_VERSION_GITHASH=0\n", text);
    try std.testing.expectError(error.NoHipVersion, hipVersionText(a, "#define HIP_VERSION_MAJOR 7\n", &missing));
    try std.testing.expectEqualStrings("HIP_VERSION_MINOR", missing);
}
