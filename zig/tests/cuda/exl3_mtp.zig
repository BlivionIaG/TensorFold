//! The MTP head of Qwen3.8-Flash-Next EXL3 on the device, built on the `exl3_fwd.Engine` (see exl3_mtp_spec.zig for the maths).
//!
//! Tensors (verified from model.safetensors.index.json and the patch file header, par1-cs13 pack):
//!   mtp.fc_embedding / mtp.fc_hidden   EXL3 2560 -> 2560      mtp.pre_fc_norm_embedding [2560], mtp.pre_fc_norm_hidden [10240]
//!   mtp.layers.0.{self_attn (+indexer), mlp (512 experts + shared + gates), attn/mlp_hyper_connection}   as any attention layer
//!   mtp_hyper_connection_mixer_patch.safetensors: mtp.hyper_connection_mixer.{hc_norm [10240], input_mix_weight_down [320,10240], _up [10240,320]}
//! The head's attention KV is slot 12 of the engine's KV arrays (`Options.extra_kv = 1`), addressed by the head's own position.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("check.zig");
const st = @import("safetensors.zig");
const real = @import("exl3_real_tests.zig");
const fwd = @import("exl3_fwd.zig");
const route = @import("exl3_route.zig");
const plerows = @import("exl3_fwd_plerows.zig");
const embed = @import("exl3_embed.zig");
const spec = @import("exl3_mtp_spec.zig");

const hidden = fwd.hidden;
const wide = fwd.wide;
const vocab = fwd.vocab;
const eps = fwd.eps;

pub const kv_slot = 12;
pub const layer_stem = "mtp.layers.0";
pub const patch_file = "mtp_hyper_connection_mixer_patch.safetensors";

pub const Mtp = struct {
    fc_e: fwd.Lin,
    fc_h: fwd.Lin,
    norm_e: []f32, // 1 + w (the model's centred convention, engine offset), [2560]
    norm_h: []f32, // [10240]
    layer: fwd.Layer,
    attn_host: [4][]f32,
    mixer: fwd.HcW, // host f32 copy (the oracle's)
    mixer_dev: fwd.HcDev, // F16 on the device (the collapse)
    patch: st.File,
    in_e: cuda.DeviceBuffer, // fp16 [R][2560]
    eo: cuda.DeviceBuffer,
    in_h: cuda.DeviceBuffer, // fp16 [R*4][2560]
    hs: cuda.DeviceBuffer,
    /// Next KV position of the head (row t of the head is position t; committed rows only, so no rollback).
    pos: usize = 0,
    /// The last `step`'s output streams / collapsed rows (host), for `absorb` and the oracle test.
    last_mixed: []f32,
    last_streams: []f32,
    last_rows: usize = 0,

    pub fn load(e: *fwd.Engine, dir: []const u8) !Mtp {
        const gpu = e.gpu;
        const gpa = gpu.gpa;
        const pack = e.pack;
        const R = e.opt.max_rows;
        if (e.opt.extra_kv < 1) return error.NoKvSlot;
        if (!e.opt.device_hc) return error.NeedsDeviceHc;
        var m: Mtp = undefined;
        m.pos = 0;
        m.last_rows = 0;
        m.fc_e = try fwd.Lin.load(gpu, pack, "mtp.fc_embedding");
        errdefer m.fc_e.free();
        m.fc_h = try fwd.Lin.load(gpu, pack, "mtp.fc_hidden");
        errdefer m.fc_h.free();
        if (m.fc_e.k != hidden or m.fc_e.n != hidden or m.fc_h.k != hidden or m.fc_h.n != hidden) return error.BadShape;
        m.norm_e = try fwd.f32Of(gpa, try fwd.tensor(pack, "mtp.pre_fc_norm_embedding.weight", .{}));
        m.norm_h = try fwd.f32Of(gpa, try fwd.tensor(pack, "mtp.pre_fc_norm_hidden.weight", .{}));
        if (m.norm_e.len != hidden or m.norm_h.len != wide) return error.BadShape;
        for (m.norm_e) |*v| v.* += e.offset;
        for (m.norm_h) |*v| v.* += e.offset;

        m.layer.index = fwd.layers_total;
        m.layer.stem = layer_stem;
        m.layer.own_heat = null;
        m.layer.gdn = null;
        m.layer.ep = null;
        m.layer.gate_dev = null;
        m.layer.sgate_dev = null;
        m.layer.attn_hc = fwd.HcW.empty;
        m.layer.mlp_hc = fwd.HcW.empty;
        var hcs: [2]fwd.HcDev = undefined;
        hcs[0] = try fwd.HcDev.load(gpu, pack, layer_stem ++ ".attn_hyper_connection", .{}, true);
        hcs[1] = try fwd.HcDev.load(gpu, pack, layer_stem ++ ".mlp_hyper_connection", .{}, true);
        m.layer.hc_dev = hcs;
        const gt = try fwd.tensor(pack, layer_stem ++ ".mlp.gate.weight", .{});
        const sg = try fwd.tensor(pack, layer_stem ++ ".mlp.shared_expert_gate.weight", .{});
        m.layer.gate16 = try fwd.f16Of(gpa, gt);
        m.layer.sgate16 = try fwd.f16Of(gpa, sg);
        if (m.layer.gate16.len != fwd.experts * hidden or m.layer.sgate16.len != hidden) return error.BadShape;
        m.layer.gate_dev = try fwd.dupload(gpu, gt.bytes);
        m.layer.sgate_dev = try fwd.dupload(gpu, sg.bytes);
        m.layer.shared[0] = try fwd.Lin.load(gpu, pack, layer_stem ++ ".mlp.shared_expert.gate_proj");
        m.layer.shared[1] = try fwd.Lin.load(gpu, pack, layer_stem ++ ".mlp.shared_expert.up_proj");
        m.layer.shared[2] = try fwd.Lin.load(gpu, pack, layer_stem ++ ".mlp.shared_expert.down_proj");
        m.attn_host = .{ &.{}, &.{}, &.{}, &.{} };
        // q_norm / k_norm of this layer average ~2.7 stored (the main layers 0.27-0.59); the engine's single gamma-1 convention
        // (e.offset) applies to them as to every CenteredRMSNorm, and the loader has no per-tensor range test to relax.
        m.layer.attn = try fwd.AttnDev.loadStem(gpu, pack, layer_stem ++ ".self_attn", e.offset, &m.attn_host);
        try e.allocExtraPool(&m.layer);

        var pb: [512]u8 = undefined;
        m.patch = try st.File.open(gpa, gpu.io, try std.fmt.bufPrint(&pb, "{s}/" ++ patch_file, .{dir}));
        const pn = m.patch.get("mtp.hyper_connection_mixer.hc_norm.weight") orelse return error.MissingTensor;
        const pd = m.patch.get("mtp.hyper_connection_mixer.input_mix_weight_down.weight") orelse return error.MissingTensor;
        const pu = m.patch.get("mtp.hyper_connection_mixer.input_mix_weight_up.weight") orelse return error.MissingTensor;
        m.mixer = .{ .norm = try fwd.f32Of(gpa, pn), .down = try fwd.f32Of(gpa, pd), .up = try fwd.f32Of(gpa, pu), .inject = null };
        if (m.mixer.norm.len != wide or m.mixer.down.len != 320 * wide or m.mixer.up.len != wide * 320) return error.BadShape;
        m.mixer_dev = .{ .norm = try f16Dev(gpu, m.mixer.norm), .down = try f16Dev(gpu, m.mixer.down), .up = try f16Dev(gpu, m.mixer.up), .inject = null };

        m.in_e = try fwd.dalloc(gpu, R * hidden * 2);
        m.eo = try fwd.dalloc(gpu, R * hidden * 2);
        m.in_h = try fwd.dalloc(gpu, R * wide * 2);
        m.hs = try fwd.dalloc(gpu, R * wide * 2);
        m.last_mixed = try gpa.alloc(f32, R * hidden);
        m.last_streams = try gpa.alloc(f32, R * wide);
        return m;
    }

    pub fn deinit(m: *Mtp, e: *fwd.Engine) void {
        const gpa = e.gpu.gpa;
        m.fc_e.free();
        m.fc_h.free();
        gpa.free(m.norm_e);
        gpa.free(m.norm_h);
        m.layer.deinit(gpa);
        for (m.attn_host) |s| gpa.free(s);
        gpa.free(m.mixer.norm);
        gpa.free(m.mixer.down);
        gpa.free(m.mixer.up);
        m.mixer_dev.free();
        m.patch.close(e.gpu.io);
        m.in_e.free();
        m.eo.free();
        m.in_h.free();
        m.hs.free();
        gpa.free(m.last_mixed);
        gpa.free(m.last_streams);
    }

    pub fn reset(m: *Mtp) void {
        m.pos = 0;
        m.last_rows = 0;
    }

    /// One head step over `tokens.len` rows: row t pairs the main streams `h[t]` ([rows][10240]) with the committed token `tokens[t]`
    /// (the token after the row's position). Appends the rows to the head's KV at its position. Fills `last_streams` (the
    /// layer output) and `last_mixed` (after the final collapse, the head's input).
    pub fn step(m: *Mtp, e: *fwd.Engine, h: []const f32, tokens: []const u32) !void {
        const gpa = e.gpu.gpa;
        const rows = tokens.len;
        if (rows == 0 or rows > e.opt.max_rows or h.len != rows * wide or m.pos + rows > e.opt.cap) return error.Shape;
        const s = e.stream;

        const en16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(en16);
        const hn16 = try gpa.alloc(f16, rows * wide);
        defer gpa.free(hn16);
        for (tokens, 0..) |tk, t| {
            const row = try embed.gather(e.embed_table, tk);
            var r32: [hidden]f32 = undefined;
            for (&r32, 0..) |*d, i| d.* = fwd.bf16(std.mem.readInt(u16, row[i * 2 ..][0..2], .little));
            var n32: [hidden]f32 = undefined;
            spec.centredNorm(&r32, m.norm_e, &n32);
            for (en16[t * hidden ..][0..hidden], n32) |*d, v| d.* = @floatCast(v);
            const nh = try gpa.alloc(f32, wide);
            defer gpa.free(nh);
            spec.centredNorm(h[t * wide ..][0..wide], m.norm_h, nh);
            for (hn16[t * wide ..][0..wide], nh) |*d, v| d.* = @floatCast(v);
        }
        try m.in_e.upload(0, std.mem.sliceAsBytes(en16));
        try m.in_h.upload(0, std.mem.sliceAsBytes(hn16));
        try m.fc_e.run(&e.ex, s, m.in_e.ptr, m.eo.ptr, rows);
        try m.fc_h.run(&e.ex, s, m.in_h.ptr, m.hs.ptr, rows * 4);
        try s.synchronize();
        const eo16 = try gpa.alloc(f16, rows * hidden);
        defer gpa.free(eo16);
        const hs16 = try gpa.alloc(f16, rows * wide);
        defer gpa.free(hs16);
        try m.eo.download(0, std.mem.sliceAsBytes(eo16));
        try m.hs.download(0, std.mem.sliceAsBytes(hs16));
        const x = m.last_streams;
        for (0..rows) |t| for (0..4) |q| for (0..hidden) |d| {
            x[t * wide + q * hidden + d] = @as(f32, @floatCast(eo16[t * hidden + d])) + @as(f32, @floatCast(hs16[t * wide + q * hidden + d]));
        };

        // the decoder layer, as Engine.forwardDev runs one (no PLE), against the head's KV at the head's position
        try e.b.h.upload(0, std.mem.sliceAsBytes(x[0 .. rows * wide]));
        const p0: i32 = @intCast(m.pos);
        try e.b.pos0.upload(0, std.mem.asBytes(&p0));
        try e.gpu.ctx.synchronize();
        try layerRun(e, &m.layer, rows);
        m.pos += rows;
        try s.synchronize();
        try e.b.h.download(0, std.mem.sliceAsBytes(x[0 .. rows * wide]));

        // collapse (the head's own mixer from the patch file, no inject) on the device
        try e.hcg.pre(s, m.mixer_dev.weights(), e.offset, fwd.eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, null, rows);
        try s.synchronize();
        try e.b.x.download(0, std.mem.sliceAsBytes(en16[0 .. rows * hidden])); // reuse: fp16 [rows][2560]
        for (m.last_mixed[0 .. rows * hidden], en16[0 .. rows * hidden]) |*d, v| d.* = v;
        m.last_rows = rows;
    }

    /// Greedy draft after the last row of the last `step`: the main lm_head over the collapsed row (the plain full-head variant).
    pub fn draftLogits(m: *Mtp, e: *fwd.Engine, out: []f32) !void {
        if (m.last_rows == 0 or out.len != vocab) return error.Shape;
        const head = e.head orelse return error.NoHead;
        const gpa = e.gpu.gpa;
        const x16 = try gpa.alloc(f16, hidden);
        defer gpa.free(x16);
        for (x16, m.last_mixed[(m.last_rows - 1) * hidden ..][0..hidden]) |*d, v| d.* = @floatCast(v);
        try e.b.x.upload(0, std.mem.sliceAsBytes(x16));
        try e.gpu.ctx.synchronize();
        try head.run(&e.ex, e.stream, e.b.x.ptr, e.b.logits.ptr, 1);
        try e.stream.synchronize();
        const l16 = try gpa.alloc(f16, vocab);
        defer gpa.free(l16);
        try e.b.logits.download(0, std.mem.sliceAsBytes(l16));
        for (out, l16) |*d, v| d.* = @floatCast(v);
    }
};

fn f16Dev(gpu: check.Gpu, w: []const f32) !cuda.DeviceBuffer {
    const h = try gpu.gpa.alloc(f16, w.len);
    defer gpu.gpa.free(h);
    for (h, w) |*d, v| d.* = @floatCast(v);
    return fwd.dupload(gpu, std.mem.sliceAsBytes(h));
}

/// The per-layer body of `Engine.forwardDev` for the head's attention layer: streams already in `e.b.h`, updated in place.
/// Same engine hooks (`hcg`, `attnRun`, `moeCompute`), none of the profile / trace branches.
fn layerRun(e: *fwd.Engine, lay: *fwd.Layer, rows: usize) !void {
    const gpa = e.gpu.gpa;
    const s = e.stream;
    const hw = lay.hc_dev.?;
    const glog = try gpa.alloc(f32, rows * cuda.exl3_hc.gate_rows);
    defer gpa.free(glog);
    try e.hcg.pre(s, hw[0].weights(), e.offset, fwd.eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, e.b.inj.ptr, rows);
    try e.attnRun(&lay.attn.?, kv_slot, rows);
    try e.hcg.post(s, e.b.h.ptr, e.b.br.ptr, e.b.inj.ptr, rows);
    try e.hcg.pre(s, hw[1].weights(), e.offset, fwd.eps, e.b.h.ptr, e.hc_scratch, e.b.x.ptr, e.b.inj.ptr, rows);
    try e.hcg.gate(s, e.b.x.ptr, lay.gate_dev.?.ptr, lay.sgate_dev.?.ptr, e.b.gate_logits.ptr, rows);
    try s.synchronize();
    try e.b.gate_logits.download(0, std.mem.sliceAsBytes(glog));
    for (0..rows) |t| {
        var picks: [fwd.slots]u16 = undefined;
        var weights: [fwd.slots]f32 = undefined;
        var rscratch: [fwd.experts]f32 = undefined;
        try route.select(glog[t * cuda.exl3_hc.gate_rows ..][0..cuda.exl3_hc.gate_rows], fwd.topk, &picks, &weights, &rscratch);
        try e.moeCompute(lay, &picks, &weights, t);
    }
    try e.hcg.post(s, e.b.h.ptr, e.b.br.ptr, e.b.inj.ptr, rows);
}

// -- state snapshot for a rejected draft --------------------------------------------------------------------------

/// GDN conv tail + recurrent state of every layer, the engine position and the PLE history + conv tail. Attention KV, indexer keys
/// and pooled blocks are written by position before they are read, so rewinding `pos` is their rollback.
pub const Snapshot = struct {
    conv: cuda.DeviceBuffer,
    rec: cuda.DeviceBuffer,
    pos: usize = 0,
    /// The `rows` PLE implementation's per-sequence state (2-token id history + 9-row conv tail), via Engine.pleSnapshot / pleRestore.
    ple: ?plerows.State = null,
    /// The f64 reference PLE (`ple_impl = .f64_reference`).
    history: [2]i64 = undefined,
    tail: []f64 = &.{},

    pub fn init(e: *fwd.Engine) !Snapshot {
        const gpu = e.gpu;
        var s: Snapshot = undefined;
        s.conv = try cuda.DeviceBuffer.alloc(gpu.d, e.table.conv.len);
        errdefer s.conv.free();
        s.rec = try cuda.DeviceBuffer.alloc(gpu.d, e.table.rec.len);
        errdefer s.rec.free();
        s.pos = 0;
        s.ple = null;
        s.tail = &.{};
        if (e.ple_rows != null) s.ple = try e.pleState();
        errdefer if (s.ple) |*p| p.deinit(gpu.gpa);
        if (e.ple) |p| s.tail = try gpu.gpa.alloc(f64, p.tail.len);
        return s;
    }

    pub fn deinit(s: *Snapshot, e: *fwd.Engine) void {
        s.conv.free();
        s.rec.free();
        if (s.ple) |*p| p.deinit(e.gpu.gpa);
        if (s.tail.len != 0) e.gpu.gpa.free(s.tail);
    }

    pub fn take(s: *Snapshot, e: *fwd.Engine) !void {
        try s.conv.copyFrom(0, e.table.conv.ptr, e.table.conv.len, e.stream.handle);
        try s.rec.copyFrom(0, e.table.rec.ptr, e.table.rec.len, e.stream.handle);
        try e.stream.synchronize();
        s.pos = e.pos;
        if (s.ple) |*p| try e.pleSnapshot(p);
        if (e.ple) |p| {
            s.history = p.history;
            @memcpy(s.tail, p.tail);
        }
    }

    pub fn restore(s: *const Snapshot, e: *fwd.Engine) !void {
        try e.table.conv.copyFrom(0, s.conv.ptr, s.conv.len, e.stream.handle);
        try e.table.rec.copyFrom(0, s.rec.ptr, s.rec.len, e.stream.handle);
        try e.stream.synchronize();
        e.pos = s.pos;
        if (s.ple) |p| try e.pleRestore(p);
        if (e.ple) |*p| {
            p.history = s.history;
            @memcpy(p.tail, s.tail);
        }
    }
};

// -- the model adaptor for exl3_mtp_spec's round driver ---------------------------------------------------------------

pub fn argmax(xs: []const f32) u32 {
    var best: usize = 0;
    for (xs, 0..) |v, i| if (v > xs[best]) {
        best = i;
    };
    return @intCast(best);
}

pub const Model = struct {
    e: *fwd.Engine,
    mtp: *Mtp,
    snap: *Snapshot,
    h: []f32, // streams of the last forward, [rows][10240]
    logits: []f32, // [R][vocab]
    rows: usize = 0,
    ms_forward: f64 = 0,
    ms_head: f64 = 0,
    ms_mtp: f64 = 0,
    io: std.Io,
    /// Test hook: every `corrupt_every`-th draft is replaced by a wrong token, forcing the rollback path (0 = off).
    corrupt_every: usize = 0,
    drafts: usize = 0,

    fn now(m: *const Model) f64 {
        return @as(f64, @floatFromInt(check.now(m.io))) / 1e6;
    }

    pub fn forward(m: *Model, tokens: []const u32, next: []u32) !void {
        var t0 = m.now();
        try m.e.forward(tokens, m.h[0 .. tokens.len * wide]);
        m.ms_forward += m.now() - t0;
        t0 = m.now();
        try m.e.logitsOf(m.h[0 .. tokens.len * wide], tokens.len, m.logits[0 .. tokens.len * vocab]);
        m.ms_head += m.now() - t0;
        m.rows = tokens.len;
        for (0..tokens.len) |i| next[i] = argmax(m.logits[i * vocab ..][0..vocab]);
    }

    pub fn snapshot(m: *Model) !void {
        try m.snap.take(m.e);
    }

    pub fn restore(m: *Model) !void {
        try m.snap.restore(m.e);
    }

    pub fn absorb(m: *Model, next_tokens: []const u32) !u32 {
        const t0 = m.now();
        if (next_tokens.len != m.rows) return error.Shape;
        try m.mtp.step(m.e, m.h[0 .. m.rows * wide], next_tokens);
        const lg = m.logits[0..vocab];
        try m.mtp.draftLogits(m.e, lg);
        m.ms_mtp += m.now() - t0;
        m.drafts += 1;
        const d = argmax(lg);
        if (m.corrupt_every != 0 and m.drafts % m.corrupt_every == 0) return (d + 1) % vocab;
        return d;
    }
};
