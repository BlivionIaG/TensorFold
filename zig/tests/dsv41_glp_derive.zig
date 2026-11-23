//! Derives a DeepSeek-V4.1 GLP vector from the checkpoint's own fold-site activations, then writes the file.
//! The site is hc_fold_pre_engram, spec v3: every block's post-fold stream, the mean over its four copies.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const glp = @import("glp");
const derive = @import("glp_derive");
const prompts = @import("glp_prompts");
const tokenizer = @import("tokenizer");

const ds = tf.deepseek_v41;
const Capture = ds.glp_capture.Capture;

/// libc's own setenv: the family reads its knobs from the environment at open, before there is a model.
extern "c" fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;

const help =
    \\tf-dsv41-glp-derive --pack DIR --out FILE [--alpha 1.0] [--mask 0.005] [--rows last|mean]
    \\                    [--prompts N] [--layers A-B] [--repo ID] [--revision REV] [--stream-dir DIR]
    \\                    [--token-map FILE] [--tokenizer FILE] [--receipt FILE] [--name S] [--description S]
    \\  Derives one direction per backbone block from the checkpoint's own post-fold stream, at
    \\  glp.hook_point = hc_fold_pre_engram, then writes the GLP file.
    \\  --pack DIR is the EXL3 checkpoint; --out FILE is the GLP file to write.
    \\  --rows last takes each prompt's last row, where the model decides what to say next; mean takes every row.
    \\  --mask F leaves the top F of each block's dimensions out of its direction (0: none).
    \\  --layers A-B writes only blocks A to B; block 0 cannot be written, so the default is 1-(layers-1).
    \\  --repo ID and --revision REV are the pack's identity in the file's metadata.
    \\  The harmful set and the form-matched harmless set beside it each run through the model.
;

const Args = struct {
    pack: ?[]const u8 = null,
    out: ?[]const u8 = null,
    stream_dir: ?[]const u8 = null,
    token_map: ?[]const u8 = null,
    tokenizer_path: ?[]const u8 = null,
    receipt: ?[]const u8 = null,
    repo: []const u8 = "",
    revision: []const u8 = "",
    name: []const u8 = "",
    description: []const u8 = "",
    alpha: f32 = 1.0,
    mask: f32 = derive.mask_recommended,
    which: derive.rows = .last,
    limit: usize = std.math.maxInt(usize),
    lo: u32 = 1,
    hi: u32 = std.math.maxInt(u32),
};

fn parse(argv: []const [:0]const u8) !Args {
    var o: Args = .{};
    var i: usize = 1;
    while (i < argv.len) : (i += 1) {
        const arg = argv[i];
        if (i + 1 >= argv.len) return error.MissingValue;
        i += 1;
        const v = argv[i];
        if (std.mem.eql(u8, arg, "--pack")) o.pack = v else if (std.mem.eql(u8, arg, "--out")) o.out = v else if (std.mem.eql(u8, arg, "--stream-dir")) o.stream_dir = v else if (std.mem.eql(u8, arg, "--token-map")) o.token_map = v else if (std.mem.eql(u8, arg, "--tokenizer")) o.tokenizer_path = v else if (std.mem.eql(u8, arg, "--receipt")) o.receipt = v else if (std.mem.eql(u8, arg, "--repo")) o.repo = v else if (std.mem.eql(u8, arg, "--revision")) o.revision = v else if (std.mem.eql(u8, arg, "--name")) o.name = v else if (std.mem.eql(u8, arg, "--description")) o.description = v else if (std.mem.eql(u8, arg, "--alpha")) o.alpha = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, arg, "--mask")) o.mask = try std.fmt.parseFloat(f32, v) else if (std.mem.eql(u8, arg, "--rows")) o.which = std.meta.stringToEnum(derive.rows, v) orelse return error.BadRows else if (std.mem.eql(u8, arg, "--prompts")) o.limit = try std.fmt.parseInt(usize, v, 10) else if (std.mem.eql(u8, arg, "--layers")) parseRange(&o, v) catch return error.BadLayers else return error.UnknownOption;
    }
    if (o.pack == null or o.out == null) return error.BadOptions;
    if (!std.math.isFinite(o.alpha) or o.alpha < 0) return error.BadOptions;
    if (!std.math.isFinite(o.mask) or o.mask < 0 or o.mask >= 1) return error.BadOptions;
    return o;
}

fn parseRange(o: *Args, text: []const u8) !void {
    const dash = std.mem.indexOfScalar(u8, text, '-') orelse return error.BadLayers;
    o.lo = try std.fmt.parseInt(u32, text[0..dash], 10);
    o.hi = try std.fmt.parseInt(u32, text[dash + 1 ..], 10);
    if (o.lo == 0 or o.lo > o.hi) return error.BadLayers;
}

/// The digest of a file the pack is identified by, or empty when it is not there.
fn digestFile(a: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8) ![]const u8 {
    const path = try std.fs.path.join(a, &.{ dir, name });
    defer a.free(path);
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(64 << 20)) catch return "";
    defer a.free(bytes);
    var out: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &out, .{});
    return std.fmt.allocPrint(a, "{x}", .{out});
}

/// The tokenizer and the Engram token map beside the pack, unless the operator named them.
fn sidecars(a: std.mem.Allocator, pack: []const u8, o: Args) !struct { tokenizer: []const u8, map: []const u8 } {
    const tok = if (o.tokenizer_path) |p| p else try std.fs.path.join(a, &.{ pack, "tokenizer.json" });
    const map = if (o.token_map) |p| p else try std.fs.path.join(a, &.{ pack, "engram-token-map.bin" });
    return .{ .tokenizer = tok, .map = map };
}

pub fn main(init: std.process.Init) !void {
    const a = init.arena.allocator();
    const io = init.io;
    const argv = try init.minimal.args.toSlice(a);
    if (argv.len == 1 or (argv.len == 2 and std.mem.eql(u8, argv[1], "--help"))) {
        try std.Io.File.stdout().writeStreamingAll(io, help ++ "\n");
        return;
    }
    var o = parse(argv) catch |e| {
        try std.Io.File.stderr().writeStreamingAll(io, help ++ "\n");
        return e;
    };
    // The capture reads every block's fold; the cone's replay enters blocks past it.
    if (setenv("TF_DSV41_CONE", "0", 1) != 0) return error.BadEnvironment;
    const pack = o.pack.?;
    const files = try sidecars(a, pack, o);

    const pool = mtl.objc.Pool.push();
    defer pool.pop();
    const started = ds.nowNs();
    var m = try ds.Model.open(init.gpa, io, .{
        .dir = pack,
        .stream_dir = o.stream_dir,
        .token_map = files.map,
        .context = 4096,
        .chunk_rows = 4096,
        .lanes = 1,
        .glp = null,
    });
    defer m.deinit();
    const layers: u32 = @intCast(m.cfg.layers);
    const hidden: u32 = @intCast(m.cfg.hidden);
    if (o.hi == std.math.maxInt(u32)) o.hi = layers - 1;
    if (o.hi >= layers) return error.LayerPastBackbone;

    var tok = try tokenizer.loadTokenizer(io, init.gpa, files.tokenizer);
    defer tok.deinit();
    const cap = try Capture.init(init.gpa, m.device, layers, hidden, o.which == .last);
    defer cap.deinit();
    m.glp_capture = cap;
    defer m.glp_capture = null;

    var harmful = try derive.Mean.init(init.gpa, layers, hidden);
    defer harmful.deinit(init.gpa);
    var harmless = try derive.Mean.init(init.gpa, layers, hidden);
    defer harmless.deinit(init.gpa);
    const requests = try prompts.requests(a);
    const twins = try prompts.harmless(a);
    try captureSet(&m, cap, &tok, requests, &harmful, o);
    try captureSet(&m, cap, &tok, twins, &harmless, o);
    if (harmful.prompts == 0 or harmless.prompts == 0) return error.NoPrompts;

    var derived = try derive.derive(init.gpa, harmful, harmless, layers, hidden, o.mask);
    defer derived.deinit(init.gpa);
    var file_layers: std.ArrayList(glp.Layer) = .empty;
    defer file_layers.deinit(init.gpa);
    for (derived.layers) |l| {
        if (l.id < o.lo or l.id > o.hi) continue;
        try file_layers.append(init.gpa, .{ .id = l.id, .dir = l.dir });
    }
    if (file_layers.items.len == 0) return error.NoDirections;

    var stamp_buf: [32]u8 = undefined;
    const created = stamp(io, &stamp_buf);
    const bytes = try glp.write(init.gpa, .{
        .mode = .project,
        .alpha = o.alpha,
        .base_model = o.repo,
        .base_revision = o.revision,
        .method = derive.method,
        .contrast = derive.contrast,
        .publication = .{
            .name = if (o.name.len > 0) o.name else null,
            .description = if (o.description.len > 0) o.description else null,
            .author = "BlivionIaG",
            .license = "Apache-2.0",
            .model_hint = "deepseek_v41",
            .measured_effect = "alpha 1 removes the direction's component at the fold and the attention writer",
            .created = created,
        },
        .layers = file_layers.items,
    });
    defer init.gpa.free(bytes);

    // Fail closed: the file is what the engine reads back, with the geometry this checkpoint has.
    try verify(init.gpa, bytes, .{ .hidden = hidden, .layers = layers, .repo = o.repo, .revision = o.revision });

    const out = o.out.?;
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = out, .data = bytes });
    var file_sha: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &file_sha, .{});

    var arena = std.heap.ArenaAllocator.init(init.gpa);
    defer arena.deinit();
    var why: glp.Why = .{};
    const vector = try glp.read(arena.allocator(), bytes, &why);
    const file_hex = std.fmt.bytesToHex(file_sha, .lower);
    const dir_hex = std.fmt.bytesToHex(vector.sha256, .lower);
    const scratch = try a.alloc(f32, @as(usize, hidden));
    const ratios = try a.alloc(f64, @as(usize, layers));
    for (0..layers) |id| ratios[id] = derive.massiveRatio(harmful, harmless, id, hidden, scratch);

    const receipt = try std.json.Stringify.valueAlloc(a, .{
        .kind = "dsv41_glp_derive",
        .pack = pack,
        .out = out,
        .repo = o.repo,
        .revision = o.revision,
        .hook_point = vector.hook_point,
        .spec_version = vector.spec_version,
        .mode = @tagName(vector.mode),
        .alpha = o.alpha,
        .mask = o.mask,
        .rows = @tagName(o.which),
        .prompts = harmful.prompts,
        .harmless_prompts = harmless.prompts,
        .width = vector.width,
        .backbone_layers = layers,
        .directions = vector.layers.len,
        .first_layer = vector.layers[0].id,
        .last_layer = vector.layers[vector.layers.len - 1].id,
        .dropped = derived.dropped,
        .bytes = bytes.len,
        .file_sha256 = file_hex[0..],
        .directions_sha256 = dir_hex[0..],
        .massive_ratio_max = blk: {
            var top: f64 = 0;
            for (ratios) |r| top = @max(top, r);
            break :blk top;
        },
        .pack_config_sha256 = try digestFile(a, io, pack, "config.json"),
        .pack_index_sha256 = try digestFile(a, io, pack, "model.safetensors.index.json"),
        .pack_tokenizer_sha256 = try digestFile(a, io, pack, "tokenizer.json"),
        .pack_quantization_sha256 = try digestFile(a, io, pack, "quantization_config.json"),
        .seconds = @as(f64, @floatFromInt(ds.nowNs() - started)) / 1e9,
    }, .{});
    try std.Io.File.stdout().writeStreamingAll(io, receipt);
    try std.Io.File.stdout().writeStreamingAll(io, "\n");
    if (o.receipt) |path| try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = receipt });
}

/// Reads the bytes back and holds them to the checkpoint they name; the producer never ships what it cannot read.
fn verify(a: std.mem.Allocator, bytes: []const u8, model: glp.Model) !void {
    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();
    var why: glp.Why = .{};
    const v = try glp.read(arena.allocator(), bytes, &why);
    glp.checkModel(v, model, &why) catch |e| {
        std.log.err("the written vector does not fit the checkpoint: {s}", .{why.text()});
        return e;
    };
    if (v.mode != .project or !std.mem.eql(u8, v.hook_point, glp.site_name)) return error.WrongSite;
}

/// One prompt set's running mean: every prompt's own mean, counted once.
fn captureSet(m: *ds.Model, cap: *Capture, tok: *const tokenizer.Tokenizer, texts: []const []const u8, mean: *derive.Mean, o: Args) !void {
    for (texts, 0..) |text, i| {
        if (i >= o.limit) break;
        const ids = tok.encode(m.gpa, text) catch |e| {
            std.log.err("prompt {d} cannot be encoded ({s})", .{ i, @errorName(e) });
            return e;
        };
        defer m.gpa.free(ids);
        // One chunk a prompt: the last row is the prompt's last row, not a chunk boundary's.
        if (ids.len == 0 or ids.len > m.workspace.plan.rows) return error.BadPrompt;
        cap.reset();
        try m.resetSlot(0);
        _ = try m.forwardOn(0, ids, .last);
        const divisor: f32 = if (o.which == .last) 1 else @floatFromInt(ids.len);
        try mean.addPrompt(cap.sums(), divisor);
    }
}

/// The UTC time of the derivation, as the file's glp.created.
fn stamp(io: std.Io, buf: *[32]u8) []const u8 {
    const ns = std.Io.Clock.real.now(io).nanoseconds;
    const secs: u64 = @intCast(@max(0, @divFloor(ns, std.time.ns_per_s)));
    const es: std.time.epoch.EpochSeconds = .{ .secs = secs };
    const yd = es.getEpochDay().calculateYearDay();
    const md = yd.calculateMonthDay();
    const dsx = es.getDaySeconds();
    return std.fmt.bufPrint(buf, "{d:0>4}-{d:0>2}-{d:0>2}T{d:0>2}:{d:0>2}:{d:0>2}Z", .{
        yd.year, md.month.numeric(), md.day_index + 1, dsx.getHoursIntoDay(), dsx.getMinutesIntoHour(), dsx.getSecondsIntoMinute(),
    }) catch "unknown";
}
