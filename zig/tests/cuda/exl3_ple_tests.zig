//! The PLE n-gram path on the real pack (host test; no GPU is touched).
//! 1. n-gram ids for a fixed real token sequence; the 16 table rows per token gathered through the file mapping and through
//!    explicit-offset preads (the --ple-on-ssd path: no mmap, no populate, no mlock) and through a third, per-row reader:
//!    byte-identical.
//! 2. The rows dequantised by two independent decoders, bit-equal, and both equal to a numpy port of the reference
//!    (`tools` crc32 printed by scratchpad/ple_ref.py on the same ids).
//! 3. The layer's injection vector (see exl3_ple_cpu.zig) from real key/value/norm/conv weights against the f64 oracle.
//! 4. Hot-row cache hit ratios over real tokenised text, and RSS to show the 30 GiB table is not resident.
//! Usage: tf-cuda-test exl3-ple <pack dir> [qbench row (3)] [tokens (24)]

const std = @import("std");
const st = @import("safetensors.zig");
const shape = @import("exl3_shape.zig");
const ngram = @import("exl3_ngram.zig");
const ple = @import("exl3_ple.zig");
const ids_mod = @import("exl3_ple_ids.zig");
const cpu = @import("exl3_ple_cpu.zig");

const prefix = "model.language_model.layers.1.ple."; // ple_layer_ids [2] is one-indexed: decoder layer index 1
const table_prefix = prefix ++ "ple_embedding.ngram_embedding.";
const heads = ids_mod.heads;
const dh = shape.ngram_dh;

// numpy port (scratchpad/ple_ref.py, tf-final ngram.NGram + exl3_mm._ple_rows) on qbench row 3, first 24 tokens.
const py_ids_crc: u32 = 0xd04d5584;
const py_raw_crc: u32 = 0x2d57eafe;
const py_decoded_crc: u32 = 0xdcf520dc;

fn fail(comptime fmt: []const u8, args: anytype) error{TestFailed} {
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return error.TestFailed;
}

fn pass(comptime fmt: []const u8, args: anytype) void {
    std.debug.print("PASS " ++ fmt ++ "\n", args);
}

const Rss = struct { total: u64 = 0, file: u64 = 0, anon: u64 = 0 }; // kB

fn rss(gpa: std.mem.Allocator, io: std.Io) Rss {
    _ = io;
    _ = gpa;
    var buf: [1 << 14]u8 = undefined;
    const fd = c.open("/proc/self/status", 0);
    if (fd < 0) return .{};
    defer _ = c.close(fd);
    var len: usize = 0;
    while (len < buf.len) {
        const n = c.read(fd, buf[len..].ptr, buf.len - len);
        if (n <= 0) break;
        len += @intCast(n);
    }
    const text = buf[0..len];
    var out: Rss = .{};
    var it = std.mem.splitScalar(u8, text, '\n');
    while (it.next()) |line| {
        const fields = [_]struct { key: []const u8, dst: *u64 }{
            .{ .key = "VmRSS:", .dst = &out.total },
            .{ .key = "RssFile:", .dst = &out.file },
            .{ .key = "RssAnon:", .dst = &out.anon },
        };
        for (fields) |f| {
            if (!std.mem.startsWith(u8, line, f.key)) continue;
            var parts = std.mem.tokenizeAny(u8, line[f.key.len..], " \tkB");
            if (parts.next()) |n| f.dst.* = std.fmt.parseInt(u64, n, 10) catch 0;
        }
    }
    return out;
}

fn showRss(gpa: std.mem.Allocator, io: std.Io, what: []const u8) Rss {
    const r = rss(gpa, io);
    std.debug.print("RSS {s}: total {d} MiB, file-backed {d} MiB, anon {d} MiB\n", .{ what, r.total >> 10, r.file >> 10, r.anon >> 10 });
    return r;
}

/// Page-cache mode: rows copied out of the file mapping.
const MapReader = struct {
    base: []const u8,
    reads: u64 = 0,
    pub fn read(self: *MapReader, file: u32, offset: u64, dest: []u8) !void {
        _ = file;
        self.reads += 1;
        @memcpy(dest, self.base[offset..][0..dest.len]);
    }
};

/// The hot-row cache plus a count of reads that reach it.
const Counting = struct {
    hot: *ple.HotRows,
    reads: u64 = 0,
    pub fn read(self: *Counting, file: u32, offset: u64, dest: []u8) !void {
        self.reads += 1;
        return self.hot.read(file, offset, dest);
    }
};

const c = struct {
    extern "c" fn open(path: [*:0]const u8, flags: c_int, ...) c_int;
    extern "c" fn read(fd: c_int, buf: [*]u8, n: usize) isize;
    extern "c" fn close(fd: c_int) c_int;
    extern "c" fn pread(fd: c_int, buf: [*]u8, n: usize, off: i64) isize;
};

fn crc(bytes: []const u8) u32 {
    return std.hash.Crc32.hash(bytes);
}


fn widen(gpa: std.mem.Allocator, t: st.Tensor, offset: f32) ![]f32 {
    const out = try gpa.alloc(f32, t.numel());
    const w: [*]align(1) const u16 = @ptrCast(t.bytes.ptr);
    for (out, 0..) |*o, i| {
        o.* = switch (t.dtype) {
            .f16 => @floatCast(@as(f16, @bitCast(w[i]))),
            .bf16 => @bitCast(@as(u32, w[i]) << 16),
            else => return error.BadDType,
        };
        o.* += offset;
    }
    return out;
}

fn need(f: *const st.File, name: []const u8) !st.Tensor {
    return f.get(name) orelse fail("missing tensor {s}", .{name});
}

/// Direct-mapped hit ratio of `slots` rows over the id stream (the id is hashed to its slot).
fn directMapped(gpa: std.mem.Allocator, stream: []const u64, slots: usize) !f64 {
    const tags = try gpa.alloc(u64, slots);
    defer gpa.free(tags);
    @memset(tags, std.math.maxInt(u64));
    var hits: u64 = 0;
    for (stream) |id| {
        const slot = @as(usize, @truncate((id *% 0x9E3779B97F4A7C15) >> 20)) % slots;
        if (tags[slot] == id) hits += 1 else tags[slot] = id;
    }
    return @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(stream.len));
}

fn distinct(gpa: std.mem.Allocator, stream: []const u64) !usize {
    var seen: std.AutoHashMapUnmanaged(u64, void) = .empty;
    defer seen.deinit(gpa);
    for (stream) |id| try seen.put(gpa, id, {});
    return seen.count();
}

const Qbench = struct { rows: []const struct { input_ids: []const u32 } };

pub fn run(gpu: anytype, rest: []const [:0]const u8) !void {
    const gpa = gpu.gpa;
    const io = gpu.io;
    if (rest.len < 1) return error.MissingArgument;
    const dir = rest[0];
    const qrow: usize = if (rest.len > 1) try std.fmt.parseInt(usize, rest[1], 10) else 3;
    const count: usize = if (rest.len > 2) try std.fmt.parseInt(usize, rest[2], 10) else 24;

    _ = showRss(gpa, io, "start");

    // ---- tokens: real tokenised chat prompts shipped in the pack (qbench_prompts.json) ------------------------------
    var path_buf: [512]u8 = undefined;
    const qpath = try std.fmt.bufPrint(&path_buf, "{s}/qbench_prompts.json", .{dir});
    const qtext = try std.Io.Dir.cwd().readFileAlloc(io, qpath, gpa, .limited(1 << 26));
    defer gpa.free(qtext);
    const parsed = try std.json.parseFromSlice(Qbench, gpa, qtext, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const qrows = parsed.value.rows;
    if (qrow >= qrows.len or count > qrows[qrow].input_ids.len) return fail("qbench row {d} has {d} tokens, asked {d}", .{ qrow, if (qrow < qrows.len) qrows[qrow].input_ids.len else 0, count });
    for (qrows) |r| for (r.input_ids) |t| if (t >= shape.vocab) return fail("token {d} outside the vocabulary", .{t});
    const tokens = qrows[qrow].input_ids[0..count];

    // ---- the table: header only, mapped lazily ---------------------------------------------------------------------
    const tpath = try std.fmt.bufPrint(&path_buf, "{s}/ngram_embedding.safetensors", .{dir});
    var tf = try st.File.open(gpa, io, tpath);
    defer tf.close(io);
    const file_bytes = tf.map.memory.len;
    std.debug.print("table file {s}: {d} bytes ({d:.2} GiB), {d} header entries\n", .{ tpath, file_bytes, @as(f64, @floatFromInt(file_bytes)) / (1 << 30), tf.names.count() });
    if (@import("builtin").os.tag == .linux) _ = std.os.linux.madvise(tf.map.memory.ptr, tf.map.memory.len, std.os.linux.MADV.RANDOM);
    const rss_open = showRss(gpa, io, "after mapping the 30 GiB table (header parsed)");

    const layout = ids_mod.derive(.{}, 0);
    {
        const mult = try need(&tf, table_prefix ++ "layer_multipliers");
        const sizes = try need(&tf, table_prefix ++ "head_vocab_sizes");
        const offs = try need(&tf, table_prefix ++ "head_offsets");
        const bias = try need(&tf, table_prefix ++ "head_bias");
        if (!mult.is(.i64, &.{3}) or !sizes.is(.i64, &.{heads}) or !offs.is(.i64, &.{heads}) or !bias.is(.f16, &.{ heads, dh })) return fail("n-gram side tensors have unexpected dtype/shape", .{});
        const m: [*]align(1) const i64 = @ptrCast(mult.bytes.ptr);
        const s: [*]align(1) const i64 = @ptrCast(sizes.bytes.ptr);
        const o: [*]align(1) const i64 = @ptrCast(offs.bytes.ptr);
        for (0..3) |i| if (m[i] != layout.multipliers[i]) return fail("multiplier {d}: pack {d}, derived {d}", .{ i, m[i], layout.multipliers[i] });
        for (0..heads) |i| if (s[i] != layout.sizes[i] or o[i] != layout.offsets[i]) return fail("head {d}: pack size/offset {d}/{d}, derived {d}/{d}", .{ i, s[i], o[i], layout.sizes[i], layout.offsets[i] });
        pass("derived hash constants (3 multipliers, 16 prime head sizes + offsets) equal the pack's shipped tensors", .{});
    }
    var shards: [128]ple.Shard = undefined;
    var words_per_row: u32 = 0;
    for (&shards, 0..) |*sh, i| {
        var nb: [160]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, "{s}shard_{d}.trellis", .{ table_prefix, i });
        const e = tf.names.get(name) orelse return fail("missing {s}", .{name});
        if (e.dtype != .i16 or e.rank != 2 or e.shape[0] != layout.shardRows(i)) return fail("{s}: dtype/rows differ from the derived layout ({d} vs {d})", .{ name, e.shape[0], layout.shardRows(i) });
        if (i == 0) words_per_row = @intCast(e.shape[1]) else if (e.shape[1] != words_per_row) return fail("shard widths differ", .{});
        sh.* = .{ .file = 0, .offset = tf.data + e.begin, .rows = @intCast(e.shape[0]) };
    }
    const bits = (@as(usize, words_per_row) - 1) * 16 / dh;
    if (bits != shape.ngram_bits or try ngram.wordsPerRow(bits) != words_per_row) return fail("row of {d} words is not 5-bit", .{words_per_row});
    pass("128 shards x {d} rows x {d} words ({d} bytes/row, {d}-bit), {d} rows in all", .{ shards[0].rows, words_per_row, words_per_row * 2, bits, layout.rows });
    const bias_t = try need(&tf, table_prefix ++ "head_bias");
    var head_bias_store: [heads * dh]f16 = undefined; // the mapping is not 2-byte aligned in general
    @memcpy(std.mem.sliceAsBytes(&head_bias_store), bias_t.bytes);
    const head_bias: []const f16 = &head_bias_store;

    // ---- (1) ids and rows, three readers ---------------------------------------------------------------------------
    const hist: [2]i64 = .{ layout.eos, layout.eos };
    const id_rows = try gpa.alloc([heads]u64, tokens.len);
    defer gpa.free(id_rows);
    ids_mod.ids(layout, hist, tokens, id_rows);
    const spec_rows = try gpa.alloc([heads]u64, tokens.len);
    defer gpa.free(spec_rows);
    try ids_mod.idsSpec(gpa, layout, hist, tokens, spec_rows);
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(id_rows), std.mem.sliceAsBytes(spec_rows))) return fail("incremental ids differ from the whole-sequence spec", .{});
    const flat: []const u64 = @as([*]const u64, @ptrCast(id_rows.ptr))[0 .. tokens.len * heads];
    std.debug.print("tokens {d} (qbench row {d}); ids[0][0..4] = {any}; ids[last][13..16] = {any}\n", .{ tokens.len, qrow, id_rows[0][0..4].*, id_rows[tokens.len - 1][13..16].* });
    {
        const ids_le = try gpa.alloc(i64, flat.len);
        defer gpa.free(ids_le);
        for (ids_le, flat) |*d, s| d.* = @intCast(s);
        const got = crc(std.mem.sliceAsBytes(ids_le));
        std.debug.print("ids crc32 {x:0>8}\n", .{got});
        if (qrow == 3 and count == 24 and got != py_ids_crc) return fail("ids crc32 {x:0>8} != numpy/tf-final NGram {x:0>8}", .{ got, py_ids_crc });
    }

    const rb: usize = words_per_row;
    const via_map = try gpa.alloc(i16, flat.len * rb);
    defer gpa.free(via_map);
    const via_ssd = try gpa.alloc(i16, flat.len * rb);
    defer gpa.free(via_ssd);
    const via_row = try gpa.alloc(i16, flat.len * rb);
    defer gpa.free(via_row);

    var mapper: MapReader = .{ .base = tf.map.memory };
    try ple.gatherFrom(&shards, words_per_row, flat, &mapper, via_map);
    const rss_map = showRss(gpa, io, "after gathering the rows through the mapping");

    const zpath = try gpa.dupeSentinel(u8, tpath, 0);
    defer gpa.free(zpath);
    const fd = try ple.Disk.openPath(zpath);
    const fds = [_]c_int{fd};
    var disk: ple.Disk = .{ .fds = &fds, .whole = @as(u64, shards[0].rows) * words_per_row * 2 };
    defer disk.close();
    if (@import("builtin").os.tag == .linux) _ = std.os.linux.fadvise(fd, 0, 0, std.os.linux.POSIX_FADV.RANDOM);
    {
        var hot: ple.HotRows = .{ .disk = &disk, .words = words_per_row };
        var counting: Counting = .{ .hot = &hot };
        // one gather a token, as a decode step issues them
        for (0..tokens.len) |t| try ple.gatherFrom(&shards, words_per_row, flat[t * heads ..][0..heads], &counting, via_ssd[t * heads * rb ..][0 .. heads * rb]);
        std.debug.print("ssd path: {d} row reads, {d} misses, {d} hits (32-slot cache, this sequence)\n", .{ counting.reads, hot.misses, counting.reads - hot.misses });
    }
    // third reader: one pread per row at an offset computed here, from the shard start table, not from planReads/locate
    for (flat, 0..) |id, i| {
        var shard: usize = 0;
        var start: u64 = 0;
        while (id >= start + layout.shardRows(shard)) : (shard += 1) start += layout.shardRows(shard);
        const off = shards[shard].offset + (id - start) * rb * 2;
        const dst = std.mem.sliceAsBytes(via_row[i * rb ..][0..rb]);
        if (c.pread(fd, dst.ptr, dst.len, @intCast(off)) != @as(isize, @intCast(dst.len))) return fail("pread of row {d} failed", .{id});
    }
    const bytes_map = std.mem.sliceAsBytes(via_map);
    if (!std.mem.eql(u8, bytes_map, std.mem.sliceAsBytes(via_ssd))) return fail("mmap rows != pread rows", .{});
    if (!std.mem.eql(u8, bytes_map, std.mem.sliceAsBytes(via_row))) return fail("mmap rows != per-row pread", .{});
    const raw = crc(bytes_map);
    std.debug.print("{d} rows x {d} bytes: raw crc32 {x:0>8}\n", .{ flat.len, rb * 2, raw });
    if (qrow == 3 and count == 24 and raw != py_raw_crc) return fail("raw rows crc32 {x:0>8} != numpy reference {x:0>8}", .{ raw, py_raw_crc });
    pass("rows for {d} tokens ({d} ids): mmap, planned-pread+hot-cache and per-row pread are byte-identical{s}", .{ tokens.len, flat.len, if (qrow == 3 and count == 24) "; crc32 equals the numpy reference" else "" });

    // ---- (2) dequantised vectors, two decoders + the numpy reference crc ---------------------------------------------
    const emb16 = try gpa.alloc(f16, flat.len * dh);
    defer gpa.free(emb16);
    const ref16 = try gpa.alloc(f16, dh);
    defer gpa.free(ref16);
    for (0..flat.len) |i| {
        const words = via_map[i * rb ..][0..rb];
        const bias = head_bias[(i % heads) * dh ..][0..dh];
        try ngram.decodeRow(words, bits, bias, emb16[i * dh ..][0..dh]);
        cpu.decodeRowRef(words, bits, bias, ref16);
        for (emb16[i * dh ..][0..dh], ref16, 0..) |a, b, j| {
            if (@as(u16, @bitCast(a)) != @as(u16, @bitCast(b))) return fail("row {d} value {d}: decodeRow {x:0>4} != reference formulation {x:0>4}", .{ i, j, @as(u16, @bitCast(a)), @as(u16, @bitCast(b)) });
        }
    }
    var peak_emb: f32 = 0;
    for (emb16) |v| peak_emb = @max(peak_emb, @abs(@as(f32, @floatCast(v))));
    const dec = crc(std.mem.sliceAsBytes(emb16));
    std.debug.print("dequantised {d} rows: max |value| {d:.4}, fp16 crc32 {x:0>8}\n", .{ flat.len, peak_emb, dec });
    if (!(peak_emb > 0 and peak_emb < 65504)) return fail("dequantised rows look degenerate", .{});
    if (qrow == 3 and count == 24 and dec != py_decoded_crc) return fail("decoded crc32 {x:0>8} != numpy reference {x:0>8}", .{ dec, py_decoded_crc });
    pass("dequantised rows: two decoders bit-equal on all {d} values{s}", .{ emb16.len, if (qrow == 3 and count == 24) "; crc32 equals the numpy port of the reference" else "" });

    // ---- (3) the layer's injection vector on real weights ----------------------------------------------------------
    const m1 = try std.fmt.bufPrint(&path_buf, "{s}/model-00001-of-00007.safetensors", .{dir});
    var mf = try st.File.open(gpa, io, m1);
    defer mf.close(io);
    const key_t = try need(&mf, prefix ++ "key_proj.weight");
    const value_t = try need(&mf, prefix ++ "value_proj.weight");
    if (!key_t.is(.f16, &.{ cpu.wide, cpu.embed_dim }) or !value_t.is(.f16, &.{ cpu.hidden, cpu.embed_dim })) return fail("key/value projection dtype or shape", .{});
    const conv_t = try need(&mf, prefix ++ "conv1d.weight");
    if (!conv_t.is(.f16, &.{ cpu.wide, 1, cpu.taps })) return fail("conv1d weight dtype or shape", .{});
    const key = try widen(gpa, key_t, 0);
    defer gpa.free(key);
    const value = try widen(gpa, value_t, 0);
    defer gpa.free(value);
    const conv = try widen(gpa, conv_t, 0);
    defer gpa.free(conv);
    // norms are stored gamma - 1 (every EXL3 pack); the scale is 1 + w. Print the means as the check.
    const nk_raw = try widen(gpa, try need(&mf, prefix ++ "norm_key.weight"), 0);
    defer gpa.free(nk_raw);
    var mean: f64 = 0;
    for (nk_raw) |v| mean += v;
    std.debug.print("norm_key stored mean {d:.4} (gamma - 1 storage if near 0)\n", .{mean / @as(f64, @floatFromInt(nk_raw.len))});
    if (@abs(mean / @as(f64, @floatFromInt(nk_raw.len))) > 0.5) return fail("norm_key does not look gamma - 1", .{});
    const norms = try gpa.alloc(f32, 3 * cpu.wide);
    defer gpa.free(norms);
    for ([_][]const u8{ "norm_key", "norm_query", "norm_conv" }, 0..) |n, i| {
        var nb: [96]u8 = undefined;
        const t = try need(&mf, try std.fmt.bufPrint(&nb, "{s}{s}.weight", .{ prefix, n }));
        if (!t.is(.bf16, &.{cpu.wide})) return fail("{s} dtype or shape", .{n});
        const w = try widen(gpa, t, 1.0);
        defer gpa.free(w);
        @memcpy(norms[i * cpu.wide ..][0..cpu.wide], w);
    }
    const weights: cpu.Weights = .{ .key = key, .value = value, .norm_key = norms[0..cpu.wide], .norm_query = norms[cpu.wide..][0..cpu.wide], .norm_conv = norms[2 * cpu.wide ..], .conv = conv };

    // Stand-in layer input h: the HC expansion of the layer-0 input, i.e. the token embedding tiled over 4 streams (the real
    // h at layer index 1 needs layers 0 and the start of 1, which this host test does not run). Real embed_tokens rows.
    const embed_t = try need(&mf, "model.language_model.embed_tokens.weight");
    if (!embed_t.is(.bf16, &.{ shape.vocab, shape.hidden })) return fail("embed_tokens dtype or shape", .{});
    const eb: [*]align(1) const u16 = @ptrCast(embed_t.bytes.ptr);

    var branch = try cpu.Branch.init(gpa, weights);
    defer branch.deinit();
    var oracle = try cpu.Oracle.init(gpa, weights);
    defer oracle.deinit();
    const emb32 = try gpa.alloc(f32, cpu.embed_dim);
    defer gpa.free(emb32);
    const emb64 = try gpa.alloc(f64, cpu.embed_dim);
    defer gpa.free(emb64);
    const h32 = try gpa.alloc(f32, cpu.wide);
    defer gpa.free(h32);
    const h64 = try gpa.alloc(f64, cpu.wide);
    defer gpa.free(h64);
    const got = try gpa.alloc(f32, cpu.wide);
    defer gpa.free(got);
    const want = try gpa.alloc(f64, cpu.wide);
    defer gpa.free(want);
    const first = try gpa.alloc(f32, cpu.wide);
    defer gpa.free(first);
    var worst: f64 = 0;
    var peak: f64 = 0;
    var worst_rel_rms: f64 = 0;
    var h_rms: f64 = 0;
    var distinct_tokens: usize = 0;
    for (tokens, 0..) |tok, t| {
        for (emb32, emb64, emb16[t * heads * dh ..][0..cpu.embed_dim]) |*a, *b, v| {
            a.* = @floatCast(v);
            b.* = a.*;
        }
        for (0..cpu.streams) |s| for (0..cpu.hidden) |d| {
            const v: f32 = @bitCast(@as(u32, eb[@as(usize, tok) * cpu.hidden + d]) << 16);
            h32[s * cpu.hidden + d] = v;
            h64[s * cpu.hidden + d] = v;
        };
        branch.step(emb32, h32, got);
        try oracle.step(emb64, h64, want);
        var se: f64 = 0;
        var sw: f64 = 0;
        var sh: f64 = 0;
        for (got, want, h32) |g, o, hv| {
            const e = @as(f64, g) - o;
            worst = @max(worst, @abs(e));
            peak = @max(peak, @abs(o));
            se += e * e;
            sw += o * o;
            sh += @as(f64, hv) * hv;
        }
        worst_rel_rms = @max(worst_rel_rms, @sqrt(se / sw));
        h_rms += @sqrt(sh / cpu.wide);
        if (t == 0) @memcpy(first, got) else if (!std.mem.eql(f32, first, got)) distinct_tokens += 1;
    }
    std.debug.print("injection vector over {d} tokens: max |engine - f64 oracle| {e:.3}, peak |oracle| {e:.3}, worst per-token rms rel error {e:.3}, mean rms(h) {d:.3}\n", .{ tokens.len, worst, peak, worst_rel_rms, h_rms / @as(f64, @floatFromInt(tokens.len)) });
    if (!(peak > 0) or distinct_tokens + 1 < tokens.len) return fail("injection vector is degenerate (peak {e}, {d} distinct tokens)", .{ peak, distinct_tokens + 1 });
    // bf16 keeps 8 significant bits (2^-9 relative per rounding); the chain rounds keys/values/norms/gate/gated/conv/silu/sum.
    if (worst > 0.03 * peak or worst_rel_rms > 0.02) return fail("injection vector differs from the f64 oracle beyond bf16 rounding", .{});
    pass("PLE injection (added to the 4-stream residual before layer 1's attn hyper-connection) matches the f64 oracle: max abs {e:.2} <= 3% of peak {e:.2}, rms rel {e:.2} <= 2e-2", .{ worst, peak, worst_rel_rms });

    // ---- (4) hot-row cache over real tokenised text; RSS --------------------------------------------------------------
    const cache_sets = [_]struct { name: []const u8, row: ?usize }{ .{ .name = "qbench row 3 prompt (408 tokens)", .row = 3 }, .{ .name = "all 22 qbench prompts (rows reset at each prompt)", .row = null } };
    for (cache_sets) |set| {
        var stream: std.ArrayList(u64) = .empty;
        defer stream.deinit(gpa);
        var ntok: usize = 0;
        var hot: ple.HotRows = .{ .disk = &disk, .words = words_per_row };
        var counting: Counting = .{ .hot = &hot };
        var buf: [heads * 51]i16 = undefined;
        for (qrows, 0..) |r, ri| {
            if (set.row) |only| if (only != ri) continue;
            var h: [2]i64 = hist; // a fresh conversation
            var off: usize = 0;
            while (off < r.input_ids.len) {
                const n = @min(64, r.input_ids.len - off);
                var rows: [64][heads]u64 = undefined;
                ids_mod.ids(layout, h, r.input_ids[off..][0..n], rows[0..n]);
                h = ids_mod.after(h, r.input_ids[off..][0..n]);
                for (rows[0..n]) |row| {
                    try stream.appendSlice(gpa, &row);
                    try ple.gatherFrom(&shards, words_per_row, &row, &counting, &buf);
                }
                ntok += n;
                off += n;
            }
        }
        const look = counting.reads;
        const hits = look - hot.misses;
        const uniq = try distinct(gpa, stream.items);
        std.debug.print("hot rows, {s}: {d} tokens, {d} lookups, {d} distinct rows ({d:.1}% of lookups are repeats = ceiling for any cache)\n", .{ set.name, ntok, look, uniq, 100.0 * (1.0 - @as(f64, @floatFromInt(uniq)) / @as(f64, @floatFromInt(look))) });
        std.debug.print("  existing HotRows (32 slots, 4 KiB): {d} hits / {d} = {d:.2}%\n", .{ hits, look, 100.0 * @as(f64, @floatFromInt(hits)) / @as(f64, @floatFromInt(look)) });
        for ([_]usize{ 256, 1024, 4096, 16384 }) |slots| {
            const ratio = try directMapped(gpa, stream.items, slots);
            std.debug.print("  direct-mapped {d:>5} rows ({d:>5} KiB at 102 B): {d:.2}%\n", .{ slots, slots * 102 / 1024, 100.0 * ratio });
        }
    }
    const rss_end = showRss(gpa, io, "end (dense weights of this layer are resident as f32)");
    const table_mib = file_bytes >> 20;
    std.debug.print("table {d} MiB; file-backed RSS: {d} MiB after mapping, {d} MiB after the mmap gather, {d} MiB at the end\n", .{ table_mib, rss_open.file >> 10, rss_map.file >> 10, rss_end.file >> 10 });
    // The mapping alone must not make the table resident; touching ~400 rows may pull in some pages + readahead, not the file.
    if (rss_map.file > (rss_open.file + 128 * 1024)) return fail("mmap gather pulled {d} MiB of file pages", .{(rss_map.file - rss_open.file) >> 10});
    if (rss_end.file > 512 * 1024) return fail("{d} MiB of file pages resident at the end", .{rss_end.file >> 10});
    pass("30.4 GiB table stays out of memory: file-backed RSS {d} MiB (mapping {d} MiB, mmap gather {d} MiB), anon {d} MiB", .{ rss_end.file >> 10, rss_open.file >> 10, rss_map.file >> 10, rss_end.anon >> 10 });
}
