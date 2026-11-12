//! The offloaded experts' window: pinned host memory the device reads in place, holding control, table, trace, slots.
const std = @import("std");

/// The six tensors of one expert in blob order: fused words, scales, biases, then down's.
pub const parts = 6;

/// What a model offers the store: its layers, the routed experts of a layer, and the bytes of each part of an expert.
pub const Shape = struct {
    layers: usize,
    /// Routed experts a layer; the shared expert is one more, always resident.
    experts: usize,
    /// Router picks a row (the routed ones plus the shared expert's): the trace holds this many ids a row.
    picks: usize,
    /// The most rows a forward stages (a round's bucket, a prompt span).
    trace_rows: usize,
    part_bytes: [parts]usize,

    pub fn blobBytes(s: Shape) usize {
        var n: usize = 0;
        for (s.part_bytes) |b| n += b;
        return n;
    }

    /// Where the down projection starts in a blob.
    pub fn downOffset(s: Shape) usize {
        return s.part_bytes[0] + s.part_bytes[1] + s.part_bytes[2];
    }

    /// Table entries a layer: every routed expert and the shared one, each a (blob address, line address) pair of u64.
    pub fn entries(s: Shape) usize {
        return s.experts + 1;
    }

    pub fn check(s: Shape) error{BadShape}!void {
        if (s.layers == 0 or s.experts == 0 or s.picks == 0 or s.trace_rows == 0) return error.BadShape;
        for (s.part_bytes) |b| if (b == 0 or b % 16 != 0) return error.BadShape;
        if (s.layers * s.experts >= std.math.maxInt(u32)) return error.BadShape;
    }
};

pub const Window = struct {
    bytes: []u8,
    /// The device's address of `bytes[0]`.
    dev: u64,
    ptab_off: usize,
    trace_off: usize,
    slots_off: usize,
    nslots: usize,

    pub const ctl_bytes: usize = 4096;
    /// The word the device writes to ask for a stage, and the word the host writes to answer it.
    pub const req_at: usize = 0;
    pub const ack_at: usize = 64;

    fn tableBytes(shape: Shape) usize {
        return std.mem.alignForward(usize, shape.layers * shape.entries() * 16, 4096);
    }

    /// Bytes before the first slot: control, table and trace.
    pub fn headBytes(shape: Shape) usize {
        return ctl_bytes + tableBytes(shape) + std.mem.alignForward(usize, shape.trace_rows * shape.picks * 4, 4096);
    }

    /// Slots a window of about `want` bytes holds: an even count, one half for each parity of layer.
    pub fn slotsFor(shape: Shape, want: usize) usize {
        return (want -| headBytes(shape)) / shape.blobBytes() / 2 * 2;
    }

    /// Fewest slots a window may have: each half holds a layer's every expert and a few more.
    pub fn leastSlots(shape: Shape) usize {
        return 2 * (shape.experts + 8);
    }

    pub fn bytesFor(shape: Shape, nslots: usize) usize {
        return headBytes(shape) + nslots * shape.blobBytes();
    }

    /// The window over `bytes` (`bytesFor` an even slot count); its control, table and trace words start at zero.
    pub fn init(shape: Shape, bytes: []u8, dev: u64) Window {
        const head = headBytes(shape);
        std.debug.assert(bytes.len >= head and bytes.len % shape.blobBytes() == head % shape.blobBytes());
        @memset(bytes[0..head], 0);
        return .{
            .bytes = bytes,
            .dev = dev,
            .ptab_off = ctl_bytes,
            .trace_off = ctl_bytes + tableBytes(shape),
            .slots_off = head,
            .nslots = (bytes.len - head) / shape.blobBytes(),
        };
    }

    /// Every routed expert's entry starts at the first slot, so an unstaged read stays in the window; shared: its line.
    pub fn initTable(w: Window, shape: Shape, shared_base: u64) void {
        const words = w.tableWords();
        for (0..shape.layers) |l| {
            const row = words[l * shape.entries() * 2 ..][0 .. shape.entries() * 2];
            for (0..shape.experts) |x| row[2 * x ..][0..2].* = .{ w.dev + w.slots_off, 0 };
            row[2 * shape.experts ..][0..2].* = .{ shared_base + l * shape.blobBytes(), 0 };
        }
    }

    /// The table: per layer `entries` pairs of u64 (blob address, cache line address or 0).
    pub fn tableWords(w: Window) []u64 {
        const bytes = w.bytes[w.ptab_off..w.trace_off];
        return @alignCast(std.mem.bytesAsSlice(u64, bytes[0 .. bytes.len / 8 * 8]));
    }

    /// The picks of the stage in flight, `picks` i32 ids a row.
    pub fn traceWords(w: Window) []i32 {
        const bytes = w.bytes[w.trace_off..w.slots_off];
        return @alignCast(std.mem.bytesAsSlice(i32, bytes[0 .. bytes.len / 4 * 4]));
    }

    pub fn word(w: Window, at: usize) *u32 {
        return @ptrCast(@alignCast(w.bytes[at..].ptr));
    }

    pub fn slotOffset(w: Window, shape: Shape, slot: usize) usize {
        return w.slots_off + slot * shape.blobBytes();
    }

    pub fn slotBytes(w: Window, shape: Shape, slot: usize) []u8 {
        return w.bytes[w.slotOffset(shape, slot)..][0..shape.blobBytes()];
    }

    pub fn slotAddr(w: Window, shape: Shape, slot: usize) u64 {
        return w.dev + w.slotOffset(shape, slot);
    }

    /// True when `addr` lies in this window.
    pub fn contains(w: Window, addr: u64) bool {
        return addr >= w.dev and addr < w.dev + w.bytes.len;
    }

    /// The table's device address for `layer`.
    pub fn layerTable(w: Window, shape: Shape, layer: usize) u64 {
        return w.dev + w.ptab_off + layer * shape.entries() * 16;
    }
};
