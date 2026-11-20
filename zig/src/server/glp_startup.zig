//! GLPv3 startup admission/reporting from #1835 c5947a8d2e; format: Matt Suiche/weightless (see GLP recipe).

const std = @import("std");
const builtin = @import("builtin");
const api = @import("engine_api");
const glp = api.glp;
const log = @import("log.zig");
const cfg = @import("checkpoint_config");
const cli_flag = @import("cli_flag.zig");

/// The flags the vector adds to the server's own.
pub const flags = [_]cli_flag.Flag{
    .{ .name = "--glp", .path = true, .native = true },
    .{ .name = "--glp-alpha", .native = true },
};

/// The model type of the family that takes a vector: DeepSeek-V4.1, whose streams carry four copies.
pub const steer_type = "deepseek_v41";

/// The copies of the stream DeepSeek-V4.1 carries: the fold site's mean is over this many.
const hc_copies: usize = 4;

/// What the --glp flags hold: the file the operator named, the strength and the vector the file carries.
pub const Vec = struct {
    /// The file's path, as the operator wrote it (null: no file named).
    path: ?[]const u8 = null,
    /// The strength the operator gave (null: the file's own alpha_default is in force).
    alpha: ?f32 = null,
    /// The vector the file carries, read once at startup (null: the file was not read).
    vector: ?glp.Vector = null,
};

/// The --glp file with the path the operator gave; null when the operator named none.
pub fn withPath(g: ?Vec, path: []const u8) Vec {
    return .{ .path = path, .alpha = if (g) |v| v.alpha else null };
}

/// The --glp file with the strength the operator gave; null when the operator named none.
pub fn withAlpha(g: ?Vec, alpha: f32) Vec {
    return .{ .path = if (g) |v| v.path else null, .alpha = alpha };
}

/// A finite strength from 0 as --glp-alpha gives it, 0 meaning every edit is bypassed; null for anything else.
pub fn strength(v: []const u8) ?f32 {
    return glp.strength(@import("fields.zig").pyFloat(v) orelse return null);
}

/// The strength in force: the operator's, or the file's own.
pub fn alphaOf(v: Vec) f32 {
    return v.alpha orelse v.vector.?.alpha;
}

/// The control vector the engine applies; null when no file was named.
pub fn engine(g: ?Vec) ?glp.Glp {
    const v = g orelse return null;
    const vector = v.vector orelse return null;
    return .{ .vector = vector, .alpha = alphaOf(v) };
}

/// Read the file the operator named and accept it; the message says why not, or null when it is read.
pub fn read(a: std.mem.Allocator, g: *?Vec) error{OutOfMemory}!?[]const u8 {
    const v = g.* orelse return null;
    const path = v.path orelse return "argument --glp-alpha: needs --glp";
    var why: glp.Why = .{};
    // The strings the reader hands back borrow the file's bytes, so those bytes stay as long as the vector.
    const loaded = glp.load(a, startupIo(), path, v.alpha, &why) catch |e| {
        if (e == error.OutOfMemory) return error.OutOfMemory;
        return try std.fmt.allocPrint(a, "argument --glp: {s}", .{why.text()});
    };
    g.* = .{ .path = path, .alpha = v.alpha, .vector = loaded.vector };
    return null;
}

/// The io a startup file read uses: the process's own, since the flags are read before the server takes one.
fn startupIo() std.Io {
    return std.Io.Threaded.global_single_threaded.io();
}

/// What the serve around the vector is: where the checkpoint is, what engine opens it and what it keeps.
pub const Setup = struct {
    dir: []const u8,
    model_type: []const u8,
    backend: []const u8,
    /// Whether prompt states are kept on disk for later servers (--learn and its relatives, --spill-gib).
    persists: bool = false,
    /// A tensor-parallel rank above 0 follows rank 0 and serves no HTTP: it checks the vector but does not report.
    follower: bool = false,
};

/// What the checkpoint says about itself, as far as it says.
pub const Facts = struct {
    model: glp.Model = .{},
    /// The text config's copies of the stream (hc_mult), when it names them.
    copies: ?usize = null,
};

/// The checkpoint's text hidden size, backbone depth and copies from its config, and its hub repo and commit.
pub fn factsOf(a: std.mem.Allocator, dir: []const u8) Facts {
    var f: Facts = .{};
    hubSnapshot(a, dir, &f.model);
    var problem: []const u8 = "";
    const bytes = (cfg.read(a, startupIo(), dir, &problem) catch null) orelse return f;
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return f;
    if (doc != .object) return f;
    // The text model's numbers live in text_config; a vision tower's sit beside it and are never read.
    const text = if (doc.object.get("text_config")) |t| (if (t == .object) t.object else return f) else doc.object;
    f.model.hidden = positive(text, "hidden_size");
    f.model.layers = positive(text, "num_hidden_layers");
    f.copies = positive(text, "hc_mult");
    return f;
}

fn positive(obj: std.json.ObjectMap, key: []const u8) ?usize {
    const v = obj.get(key) orelse return null;
    return if (v == .integer and v.integer > 0) @intCast(v.integer) else null;
}

/// The repo id and commit a hub-cache snapshot folder names ("models--Org--Name/snapshots/COMMIT"); unset otherwise.
fn hubSnapshot(a: std.mem.Allocator, dir: []const u8, model: *glp.Model) void {
    const at = std.mem.lastIndexOf(u8, dir, "models--") orelse return;
    const rest = dir[at + "models--".len ..];
    const snapshots = "/snapshots/";
    const sep = std.mem.indexOf(u8, rest, snapshots) orelse return;
    const owner = rest[0..sep];
    const dash = std.mem.indexOf(u8, owner, "--") orelse return;
    const tail = rest[sep + snapshots.len ..];
    model.revision = tail[0 .. std.mem.indexOfScalar(u8, tail, '/') orelse tail.len];
    model.repo = std.fmt.allocPrint(a, "{s}/{s}", .{ owner[0..dash], owner[dash + 2 ..] }) catch "";
}

/// Check the vector against the serve and the checkpoint, and report what is in force; the message is the refusal.
pub fn admit(a: std.mem.Allocator, setup: Setup, v: Vec) error{OutOfMemory}!?[]const u8 {
    const facts = factsOf(a, setup.dir);
    if (try refusal(a, setup, facts, v)) |why| return why;
    if (!setup.follower) report(a, v, facts);
    return null;
}

/// Why this serve cannot take the vector, as the operator reads it; null: it can.
pub fn refusal(a: std.mem.Allocator, setup: Setup, facts: Facts, v: Vec) error{OutOfMemory}!?[]const u8 {
    const vector = v.vector orelse return null;
    if (!std.mem.eql(u8, setup.model_type, steer_type)) {
        return try std.fmt.allocPrint(a, "--glp: {s} checkpoints are not steered: only DeepSeek-V4.1 takes a GLP vector", .{setup.model_type});
    }
    if (!steers(setup.backend)) {
        const here = switch (builtin.os.tag) {
            .macos => "on a Mac only the Metal engine does",
            .linux => "on Linux only the CUDA engine does",
            else => "no engine on this platform does",
        };
        return try std.fmt.allocPrint(a, "--glp: the {s} engine does not steer DeepSeek-V4.1: {s}", .{ backendName(setup.backend), here });
    }
    if (setup.persists) {
        return "--glp: prompt states kept on disk are not keyed by the vector and would bring back rows made without it: " ++
            "serve without --learn, --learn-dir, --learn-conversations, --prompt-cache-disk and --spill-gib";
    }
    if (facts.model.hidden == null or facts.model.layers == null) {
        return "--glp: the checkpoint's config.json gives no text hidden size and depth, so the vector cannot be checked against it";
    }
    if (facts.copies) |copies| if (copies != hc_copies) {
        return try std.fmt.allocPrint(a, "--glp: the vector is applied to {d} stream copies; this checkpoint carries {d}", .{ hc_copies, copies });
    };
    var why: glp.Why = .{};
    glp.checkModel(vector, facts.model, &why) catch return try std.fmt.allocPrint(a, "--glp: {s}", .{why.text()});
    glp.checkStrength(vector, alphaOf(v), &why) catch return try std.fmt.allocPrint(a, "--glp: {s}", .{why.text()});
    return null;
}

/// The engines that steer DeepSeek-V4.1: Metal on a Mac, CUDA elsewhere; `auto` is whichever this build has.
fn steers(backend: []const u8) bool {
    const auto = std.mem.eql(u8, backend, "auto");
    return switch (builtin.os.tag) {
        .macos => auto or std.mem.eql(u8, backend, "mlx"),
        .linux => auto or std.mem.eql(u8, backend, "cuda"),
        else => false,
    };
}

/// The engine a backend names, as the operator reads it.
fn backendName(backend: []const u8) []const u8 {
    if (std.mem.eql(u8, backend, "auto")) return switch (builtin.os.tag) {
        .macos => "Metal",
        .linux => "CUDA",
        else => "default",
    };
    if (std.mem.eql(u8, backend, "mlx")) return "Metal";
    if (std.mem.eql(u8, backend, "cuda")) return "CUDA";
    if (std.mem.eql(u8, backend, "rocm")) return "ROCm";
    return backend;
}

/// Report escaped file/digests and effective policy, explicitly distinguishing alpha-zero loads from active edits.
pub fn report(a: std.mem.Allocator, v: Vec, facts: Facts) void {
    const vector = v.vector orelse return;
    const alpha = alphaOf(v);
    const sha = std.fmt.bytesToHex(vector.sha256, .lower);
    const file_sha = std.fmt.bytesToHex(vector.file_sha256, .lower);
    const verdict = if (alpha == 0) "glp loaded but inactive: alpha 0 bypasses every edit and replies are unsteered" else "glp on: replies are steered";
    log.line("{s} ({s} vector in \"{f}\"; hook: {f}; file sha256: {s}; alpha: {d}; spec v{d}; layers: {d} ({s}); base: {s}; tensor digest: {s}; checkpoint: {s})", .{
        verdict,
        @tagName(vector.mode),
        std.zig.fmtString(v.path orelse ""),
        std.zig.fmtString(vector.hook_point),
        file_sha[0..],
        alpha,
        vector.spec_version,
        vector.layers.len,
        layerIds(a, vector.layers),
        base(a, vector),
        fmt(a, "{s} ({s})", .{ sha[0..12], if (vector.digest_claimed) "verified" else "the file claims none" }),
        provenance(a, facts.model),
    });
    if (vector.transferred) log.line("warning: this vector was derived at {f}, not at {f}: it is applied at {s}", .{ std.zig.fmtString(vector.derived_at), std.zig.fmtString(vector.hook_point), vector.hook_point });
}

/// Where the served checkpoint came from, when its folder says: a hub snapshot's repo and commit, else unavailable.
fn provenance(a: std.mem.Allocator, m: glp.Model) []const u8 {
    if (m.repo.len == 0) return "source repo and revision unknown (not a hub snapshot): only hidden size, depth and stream copies were checked, not the vector's base";
    return fmt(a, "{f} @ {f}", .{ std.zig.fmtString(m.repo), std.zig.fmtString(m.revision) });
}

/// The checkpoint the vector was derived from, with its revision when it names one; the file's own words, escaped.
fn base(a: std.mem.Allocator, v: glp.Vector) []const u8 {
    if (v.base_model.len == 0) return "an unnamed checkpoint";
    if (v.base_revision.len == 0) return fmt(a, "{f}", .{std.zig.fmtString(v.base_model)});
    return fmt(a, "{f} @ {f}", .{ std.zig.fmtString(v.base_model), std.zig.fmtString(v.base_revision) });
}

/// The layer ids a vector covers: "1-39" when they run and "1, 3, 5" when they do not.
fn layerIds(a: std.mem.Allocator, layers: []const glp.Layer) []const u8 {
    if (layers.len == 0) return "none";
    var run = true;
    for (layers, 0..) |l, i| if (i > 0 and l.id != layers[i - 1].id + 1) {
        run = false;
        break;
    };
    if (run) return fmt(a, "{d}-{d}", .{ layers[0].id, layers[layers.len - 1].id });
    var w: std.Io.Writer.Allocating = .init(a);
    for (layers, 0..) |l, i| {
        if (i > 0) w.writer.print(", ", .{}) catch {};
        w.writer.print("{d}", .{l.id}) catch {};
    }
    return w.toOwnedSlice() catch "?";
}

/// A formatted string, or a mark when the allocator says no.
fn fmt(a: std.mem.Allocator, comptime f: []const u8, args: anytype) []const u8 {
    return std.fmt.allocPrint(a, f, args) catch "?";
}
