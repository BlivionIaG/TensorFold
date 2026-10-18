//! The engine's PLE row path (S6b): n-gram ids with a per-sequence history, the 16 table rows of each token gathered through
//! the file mapping (page cache) or exact preads behind a 4096-row hot cache (SSD), decoded to fp16 rows, then
//! `exl3_ple_cpu.Branch.step` (the engine's bf16 roundings). Per-sequence state is the 2-token id history and the 9-row
//! conv tail; `snapshot` / `restore` copy it so a sequence can be forked or rolled back (speculation, prefix reuse).
//! This is the independent implementation; `exl3_fwd_ple.Ple` (f64, S5) stays as the cross-check reference.

const std = @import("std");
const st = @import("safetensors.zig");
const shape = @import("exl3_shape.zig");
const ngram = @import("exl3_ngram.zig");
const ple = @import("exl3_ple.zig");
const ids_mod = @import("exl3_ple_ids.zig");
const cpu = @import("exl3_ple_cpu.zig");

pub const heads = ids_mod.heads;
pub const dh = shape.ngram_dh;
pub const wide = cpu.wide;
pub const hot_rows: usize = 4096;
pub const Hot = ple.HotCache(hot_rows, 128);

pub const Mode = ple.Mode;

const table_prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.";

/// Reads rows out of the file mapping.
const MapReader = struct {
    base: []const u8,
    pub fn read(self: *MapReader, file: u32, offset: u64, dest: []u8) !void {
        _ = file;
        @memcpy(dest, self.base[offset..][0..dest.len]);
    }
};

/// Per-sequence state: the two tokens before the next one and the conv tail (the last 9 normed rows).
pub const State = struct {
    history: [ids_mod.context]i64,
    tail: []f32,

    pub fn init(gpa: std.mem.Allocator) !State {
        const tail = try gpa.alloc(f32, cpu.tail_rows * wide);
        @memset(tail, 0);
        return .{ .history = .{ 0, 0 }, .tail = tail };
    }

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        gpa.free(s.tail);
        s.* = undefined;
    }
};

pub const Rows = struct {
    gpa: std.mem.Allocator,
    layout: ids_mod.Layout,
    shards: [128]ple.Shard,
    words_per_row: u32,
    bits: usize,
    head_bias: [heads * dh]f16,
    mode: Mode,
    map: []const u8,
    fds: [1]c_int,
    disk: ple.Disk,
    hot: ?*Hot,
    branch: cpu.Branch,
    history: [ids_mod.context]i64,
    /// Rows read through the table, rows that missed the hot cache (ssd), calls to `rows`.
    row_reads: u64 = 0,

    /// `weights` slices are owned by the caller and must outlive this.
    pub fn init(gpa: std.mem.Allocator, ng: *st.File, mode: Mode, weights: cpu.Weights) !*Rows {
        const r = try gpa.create(Rows);
        errdefer gpa.destroy(r);
        r.gpa = gpa;
        r.mode = mode;
        r.map = ng.map.memory;
        r.hot = null;
        r.row_reads = 0;
        r.layout = ids_mod.derive(.{}, 0);
        const layout = r.layout;
        // the pack's own constants win: the four tensors must equal what the config derives
        const mult = ng.get(table_prefix ++ "layer_multipliers") orelse return error.MissingTensor;
        const sizes = ng.get(table_prefix ++ "head_vocab_sizes") orelse return error.MissingTensor;
        const offs = ng.get(table_prefix ++ "head_offsets") orelse return error.MissingTensor;
        const bias = ng.get(table_prefix ++ "head_bias") orelse return error.MissingTensor;
        if (!mult.is(.i64, &.{3}) or !sizes.is(.i64, &.{heads}) or !offs.is(.i64, &.{heads}) or !bias.is(.f16, &.{ heads, dh })) return error.BadShape;
        for (0..3) |i| if (std.mem.readInt(i64, mult.bytes[i * 8 ..][0..8], .little) != layout.multipliers[i]) return error.PleConstantsDiffer;
        for (0..heads) |i| {
            if (std.mem.readInt(i64, sizes.bytes[i * 8 ..][0..8], .little) != layout.sizes[i]) return error.PleConstantsDiffer;
            if (std.mem.readInt(i64, offs.bytes[i * 8 ..][0..8], .little) != layout.offsets[i]) return error.PleConstantsDiffer;
        }
        @memcpy(std.mem.sliceAsBytes(&r.head_bias), bias.bytes);
        var wpr: u32 = 0;
        for (&r.shards, 0..) |*sh, i| {
            var nb: [160]u8 = undefined;
            const name = try std.fmt.bufPrint(&nb, "{s}shard_{d}.trellis", .{ table_prefix, i });
            const e = ng.names.get(name) orelse return error.MissingTensor;
            if (e.dtype != .i16 or e.rank != 2 or e.shape[0] != layout.shardRows(i)) return error.BadShape;
            if (i == 0) wpr = @intCast(e.shape[1]) else if (e.shape[1] != wpr) return error.NonUniformShards;
            sh.* = .{ .file = 0, .offset = ng.data + e.begin, .rows = @intCast(e.shape[0]) };
        }
        r.words_per_row = wpr;
        r.bits = (@as(usize, wpr) - 1) * 16 / dh;
        if (r.bits != shape.ngram_bits or try ngram.wordsPerRow(r.bits) != wpr) return error.BadShape;
        r.fds = .{-1};
        r.disk = .{ .fds = &r.fds, .whole = @as(u64, r.shards[0].rows) * wpr * 2 };
        if (mode == .ssd) {
            r.fds[0] = try ple.Disk.openPath(ng.path);
            r.disk.fds = &r.fds;
            const hot = try gpa.create(Hot);
            hot.disk = &r.disk;
            hot.words = wpr;
            hot.misses = 0;
            hot.lookups = 0;
            @memset(&hot.tags, std.math.maxInt(u64));
            r.hot = hot;
        }
        errdefer if (r.hot) |h| gpa.destroy(h);
        errdefer if (mode == .ssd) r.disk.close();
        r.branch = try cpu.Branch.init(gpa, weights);
        r.history = @splat(layout.eos);
        return r;
    }

    pub fn deinit(r: *Rows) void {
        const gpa = r.gpa;
        r.branch.deinit();
        if (r.hot) |h| gpa.destroy(h);
        if (r.mode == .ssd) r.disk.close();
        gpa.destroy(r);
    }

    /// New sequence: history = [eos, eos], conv tail = zeros.
    pub fn reset(r: *Rows) void {
        r.history = @splat(r.layout.eos);
        r.branch.reset();
    }

    pub fn snapshot(r: *const Rows, dst: *State) void {
        dst.history = r.history;
        @memcpy(dst.tail, r.branch.tail);
    }

    pub fn restore(r: *Rows, src: State) void {
        r.history = src.history;
        @memcpy(r.branch.tail, src.tail);
    }

    /// Hot-cache misses and lookups (ssd mode), else zeros.
    pub fn hotStats(r: *const Rows) struct { lookups: u32, misses: u32 } {
        if (r.hot) |h| return .{ .lookups = h.lookups, .misses = h.misses };
        return .{ .lookups = 0, .misses = 0 };
    }

    /// The 16 decoded rows of one token's ids, as the 2560 fp16 values the projections read (f32).
    pub fn embedding(r: *Rows, row_ids: *const [heads]u64, emb: []f32) !void {
        std.debug.assert(emb.len == cpu.embed_dim);
        var words: [heads * 51]i16 = undefined;
        const wpr: usize = r.words_per_row;
        if (wpr != 51) return error.BadShape;
        switch (r.mode) {
            .page_cache => {
                var mr: MapReader = .{ .base = r.map };
                try ple.gatherFrom(&r.shards, r.words_per_row, row_ids, &mr, &words);
            },
            .ssd => try ple.gatherFrom(&r.shards, r.words_per_row, row_ids, r.hot.?, &words),
        }
        r.row_reads += heads;
        for (0..heads) |hd| {
            var row: [dh]f16 = undefined;
            try ngram.decodeRow(words[hd * wpr ..][0..wpr], r.bits, r.head_bias[hd * dh ..][0..dh], &row);
            for (emb[hd * dh ..][0..dh], row) |*d, s| d.* = @floatCast(s);
        }
    }

    /// The add-back for each of `tokens` given the streams `h` [rows][10240] before it; advances history and conv tail.
    pub fn rows(r: *Rows, tokens: []const u32, h: []const f32, out: []f32) !void {
        if (h.len != tokens.len * wide or out.len != h.len) return error.Length;
        const gpa = r.gpa;
        const row_ids = try gpa.alloc([heads]u64, tokens.len);
        defer gpa.free(row_ids);
        ids_mod.ids(r.layout, r.history, tokens, row_ids);
        var emb: [cpu.embed_dim]f32 = undefined;
        for (0..tokens.len) |t| {
            try r.embedding(&row_ids[t], &emb);
            r.branch.step(&emb, h[t * wide ..][0..wide], out[t * wide ..][0..wide]);
        }
        r.history = ids_mod.after(r.history, tokens);
    }
};
