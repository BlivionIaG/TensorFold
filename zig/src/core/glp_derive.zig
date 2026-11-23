//! Directions for a GLP file from a model's own activations: per-layer means, the massive-activation mask,
//! and the difference of means that becomes one unit direction per layer. No device and no model here.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// The GLP metadata a producer writes: how the numbers were made and what they contrast.
pub const method = "difference_of_means_per_layer";
pub const contrast = "form_matched_harmful_vs_harmless_unpaired";

/// Which rows of a prompt a capture sums: every row, or the last row alone where the model decides.
pub const rows = enum { mean, last };

/// The fraction a direction leaves out when the screen says the model carries massive activations.
pub const mask_recommended = 0.005;

/// The peak-to-median ratio above which a model carries massive activations of its own.
pub const massive_threshold = 50.0;

/// One prompt set's mean, and the largest a dimension's per-prompt mean ever was: every prompt counts once.
pub const Mean = struct {
    /// The per-prompt means, summed: a long prompt does not outweigh a short one.
    total: []f32,
    /// The largest absolute per-prompt mean, dimension by dimension: what the mask screen reads.
    peak: []f32,
    prompts: usize = 0,
    hidden: usize,

    pub fn init(a: Allocator, layers: usize, hidden: usize) !Mean {
        if (layers == 0 or hidden == 0) return error.BadMeanShape;
        const total = try a.alloc(f32, layers * hidden);
        errdefer a.free(total);
        const peak = try a.alloc(f32, layers * hidden);
        @memset(peak, 0);
        return .{ .total = total, .peak = peak, .hidden = hidden };
    }

    pub fn deinit(m: *Mean, a: Allocator) void {
        a.free(m.total);
        a.free(m.peak);
        m.* = .{ .total = &.{}, .peak = &.{}, .hidden = m.hidden };
    }

    /// Adds one prompt's summed rows: its own mean, weighted like every other prompt's, and its peaks.
    pub fn addPrompt(m: *Mean, sums: []const f32, divisor: f32) !void {
        if (divisor <= 0 or sums.len != m.total.len) return error.BadPromptSums;
        for (m.total, m.peak, sums) |*t, *p, s| {
            const v = s / divisor;
            t.* += v;
            p.* = @max(p.*, @abs(v));
        }
        m.prompts += 1;
    }

    /// The set's mean for one layer, written into `out`.
    pub fn mean(m: Mean, id: usize, out: []f32) void {
        const count: f32 = @floatFromInt(@max(m.prompts, 1));
        const base = id * m.hidden;
        for (out, m.total[base..][0..m.hidden]) |*o, v| o.* = v / count;
    }
};

/// One layer's direction: the layer id the GLP tensor names, and the unit vector.
pub const Layer = struct {
    id: u32,
    dir: []f32,
};

/// The derived directions and the layers whose difference was degenerate (dropped, never emitted as zeros).
pub const Derived = struct {
    layers: []Layer,
    dropped: []u32,

    pub fn deinit(d: *Derived, a: Allocator) void {
        for (d.layers) |l| a.free(l.dir);
        a.free(d.layers);
        a.free(d.dropped);
        d.* = .{ .layers = &.{}, .dropped = &.{} };
    }
};

/// The two sets' peaks in one layer: the largest a dimension's activation ever was, over both.
const Peaks = struct {
    harmful: Mean,
    harmless: Mean,
    base: usize,

    fn at(p: Peaks, i: usize) f32 {
        return @max(p.harmful.peak[p.base + i], p.harmless.peak[p.base + i]);
    }
};

/// The dims a layer's direction leaves out: the top `frac` of them by the activation they carry.
fn maskDims(peaks: Peaks, hidden: usize, frac: f32, order: []usize, masked: []bool) void {
    @memset(masked, false);
    const count = @as(usize, @intFromFloat(@round(frac * @as(f32, @floatFromInt(hidden)))));
    const k = @min(hidden, @max(1, count));
    for (order, 0..) |*o, i| o.* = i;
    std.mem.sort(usize, order, peaks, struct {
        fn lessThan(p: Peaks, x: usize, y: usize) bool {
            const px = p.at(x);
            const py = p.at(y);
            if (px != py) return px > py;
            return x < y;
        }
    }.lessThan);
    for (order[0..k]) |i| masked[i] = true;
}

/// The peak-to-median ratio of a layer's activations: how far the massive activations stand out.
pub fn massiveRatio(harmful: Mean, harmless: Mean, id: usize, hidden: usize, scratch: []f32) f64 {
    if (scratch.len < hidden) return 0;
    const peaks = Peaks{ .harmful = harmful, .harmless = harmless, .base = id * hidden };
    for (0..hidden) |i| scratch[i] = peaks.at(i);
    std.mem.sort(f32, scratch[0..hidden], {}, std.sort.asc(f32));
    const median = scratch[hidden / 2];
    if (median <= 0) return 0;
    return @as(f64, scratch[hidden - 1]) / @as(f64, median);
}

/// The two sets' difference of means per layer, each unit-normalised; a layer with no difference is dropped.
/// `mask` is the fraction of dims excluded per layer (0: none), and the excluded dims are not re-added.
pub fn derive(a: Allocator, harmful: Mean, harmless: Mean, layers: usize, hidden: usize, mask: f32) !Derived {
    if (harmful.prompts == 0 or harmless.prompts == 0) return error.NoPrompts;
    if (harmful.total.len != layers * hidden or harmless.total.len != layers * hidden) return error.BadMeanShape;
    var kept: std.ArrayList(Layer) = .empty;
    errdefer {
        for (kept.items) |l| a.free(l.dir);
        kept.deinit(a);
    }
    var dropped: std.ArrayList(u32) = .empty;
    errdefer dropped.deinit(a);
    const left = try a.alloc(f32, hidden);
    defer a.free(left);
    const right = try a.alloc(f32, hidden);
    defer a.free(right);
    const diff = try a.alloc(f32, hidden);
    defer a.free(diff);
    const order = try a.alloc(usize, hidden);
    defer a.free(order);
    const masked = try a.alloc(bool, hidden);
    defer a.free(masked);
    for (0..layers) |id| {
        harmful.mean(id, left);
        harmless.mean(id, right);
        for (left, right, diff) |l, r, *d| d.* = l - r;
        if (mask > 0) {
            const peaks = Peaks{ .harmful = harmful, .harmless = harmless, .base = id * hidden };
            maskDims(peaks, hidden, mask, order, masked);
            for (diff, masked) |*d, m| {
                if (m) d.* = 0;
            }
        }
        var sum: f64 = 0;
        for (diff) |d| sum += @as(f64, d) * @as(f64, d);
        const norm = @sqrt(sum);
        if (!std.math.isFinite(norm) or norm == 0) {
            try dropped.append(a, @intCast(id));
            continue;
        }
        var finite = true;
        for (diff) |*d| {
            d.* = @floatCast(@as(f64, d.*) / norm);
            if (!std.math.isFinite(d.*)) finite = false;
        }
        if (!finite) {
            try dropped.append(a, @intCast(id));
            continue;
        }
        const dir = try a.alloc(f32, hidden);
        errdefer a.free(dir);
        @memcpy(dir, diff);
        try kept.append(a, .{ .id = @intCast(id), .dir = dir });
    }
    return .{ .layers = try kept.toOwnedSlice(a), .dropped = try dropped.toOwnedSlice(a) };
}
