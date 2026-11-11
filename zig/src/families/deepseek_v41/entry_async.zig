//! Layer-split passes without a host wait between them: the entry rows staged in one of two device buffers while the
const std = @import("std");
const mtl = @import("metal");
const engram = @import("engram.zig");
const gate = @import("layersplit_gate.zig");
const model = @import("model.zig");
const ops = @import("ops.zig");
const Slots = @import("entry_slots.zig").Slots;
const Model = model.Model;

/// The command buffers of one committed pass (retained: the pass outlives its forward's autorelease pool).
pub const Parked = struct {
    cbs: [6]mtl.CommandBuffer = undefined,
    n: u8 = 0,
    /// the entry buffer the pass reads
    buf: ?u1 = null,
};

pub const State = struct {
    bufs: [2]?mtl.Buffer = .{ null, null },
    slots: Slots = .{},
    /// the buffer the pass being encoded reads
    reading: ?u1 = null,
    /// the Engram staging buffer the running pass does not read (it swaps with `Model.stream_stage` a pass)
    stage_alt: ?mtl.Buffer = null,
    event: ?mtl.Event = null,
    /// the last value a committed pass signals when its last command buffer completes
    signalled: u64 = 0,
    /// the forward commits its command buffers and returns without waiting
    deferring: bool = false,
    parked: Parked = .{},
};

/// TF_DSV41_SPLIT_ASYNC=0 waits for every entered pass as the normal forward does.
pub fn enabled() bool {
    return @import("sys").environment.read(bool, "TF_DSV41_SPLIT_ASYNC", true) catch true;
}

fn add(m: *Model, buf: mtl.Buffer) void {
    m.residency.add(buf);
    m.residency.commit();
    m.residency.requestResidency();
}

fn ensure(m: *Model) !void {
    const st = &m.pipe;
    if (st.bufs[0] != null) return;
    const bytes = @as(usize, m.workspace.plan.rows) * (gate.hidden_row + gate.pre_row);
    const opts = mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked;
    const a = try m.device.buffer(bytes, opts);
    errdefer a.deinit();
    const b = try m.device.buffer(bytes, opts);
    errdefer b.deinit();
    add(m, a);
    add(m, b);
    st.bufs = .{ a, b };
}

/// `n` memory rows into the entry buffer no running pass reads: the host's copy overlaps the pass before.
pub fn stage(m: *Model, rows: []const u8, n: u32) !u1 {
    const st = &m.pipe;
    if (n == 0 or n > m.workspace.plan.rows) return error.InvalidPrefillChunk;
    try ensure(m);
    const t = st.slots.take();
    const b = t.slot;
    if (t.wait) try settle(m);
    const at = st.bufs[b].?.contents();
    const x: usize = @as(usize, n) * gate.hidden_row;
    try gate.Gate.deinterleave(rows, at[0..x], at[x..][0 .. @as(usize, n) * gate.pre_row], n);
    return b;
}

/// The staged rows into the workspace's stream and pre, on the GPU, at the head of the pass.
pub fn encodeEntry(m: *Model, enc: mtl.ComputeEncoder, buf: mtl.Buffer, rows: u32) !void {
    const p = &m.pipelines.ops;
    if (p.dump_f32 == null) return error.GateEntry;
    const ws = &m.workspace;
    const x: usize = @as(usize, rows) * gate.hidden_row;
    ops.encodeDump(enc, p, .{ .buffer = buf, .offset = 0 }, ws.view(.stream), @intCast(x / 4), false);
    ops.encodeDump(enc, p, .{ .buffer = buf, .offset = x }, ws.view(.previous_pre), rows * 4, false);
    enc.barrier();
}

/// Before a layer-split pass: later passes encode behind the running one where their host writes cannot reach what it
pub fn prepare(m: *Model) !void {
    const st = &m.pipe;
    if (!enabled() or m.profile != null or m.observer != null or m.host_prof) return;
    if (m.stream == null or m.stream_stage == null) {
        try settle(m);
    } else {
        if (st.stage_alt == null) {
            const bytes = @as(usize, m.workspace.plan.rows) * 2 * engram.gather_dim * @sizeOf(u16);
            const buf = try m.device.buffer(bytes, mtl.ResourceOptions.shared | mtl.ResourceOptions.untracked);
            add(m, buf);
            st.stage_alt = buf;
        }
        std.mem.swap(?mtl.Buffer, &m.stream_stage, &st.stage_alt);
    }
    st.deferring = true;
}

/// After a layer-split pass, whether it ran or not.
pub fn done(m: *Model) void {
    m.pipe.deferring = false;
    m.pipe.reading = null;
}

/// This pass's command buffer starts after the parked pass completes.
pub fn behind(m: *Model, e: mtl.CommandBuffer) void {
    const st = &m.pipe;
    if (st.parked.n != 0) if (st.event) |ev| e.waitForEvent(ev, st.signalled);
}

/// The pass's last command buffer, before its commit, signals the next pass's wait.
pub fn signal(m: *Model, e: mtl.CommandBuffer) !void {
    const st = &m.pipe;
    if (st.event == null) st.event = try m.device.event();
    st.signalled += 1;
    e.signalEvent(st.event.?, st.signalled);
}

/// The committed pass keeps running; the one before it, queued ahead, is waited for now.
pub fn park(m: *Model, cbs: []const mtl.CommandBuffer) !void {
    const st = &m.pipe;
    var next: Parked = .{ .buf = st.reading };
    for (cbs) |cb| {
        next.cbs[next.n] = cb.retain();
        next.n += 1;
    }
    if (st.reading) |b| st.slots.committed(b);
    const prev = st.parked;
    st.parked = next;
    try finish(m, prev);
}

fn finish(m: *Model, p: Parked) !void {
    if (p.n == 0) return;
    var failed: ?anyerror = null;
    var gpu: f64 = 0;
    for (p.cbs[0..p.n]) |cb| {
        cb.wait();
        model.completed(cb) catch |err| {
            failed = err;
        };
        gpu += cb.gpuSeconds();
        cb.release();
    }
    if (p.buf) |b| m.pipe.slots.completed(b);
    m.last_gpu_seconds = gpu;
    if (failed) |err| {
        m.valid = false;
        return err;
    }
}

/// Every parked pass completed (a full sync): before the host reads or writes what the GPU may be using.
pub fn settle(m: *Model) !void {
    const p = m.pipe.parked;
    m.pipe.parked = .{};
    try finish(m, p);
}

/// `settle` where a failure only means the state is rebuilt anyway.
pub fn settleQuiet(m: *Model) void {
    settle(m) catch |err| std.log.warn("layer split: a pass failed on the device: {s}", .{@errorName(err)});
}

pub fn release(m: *Model) void {
    const st = &m.pipe;
    settleQuiet(m);
    for (&st.bufs) |*b| {
        if (b.*) |buf| buf.deinit();
        b.* = null;
    }
    if (st.stage_alt) |buf| buf.deinit();
    st.stage_alt = null;
    if (st.event) |ev| ev.deinit();
    st.event = null;
}
