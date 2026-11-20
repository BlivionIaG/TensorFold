//! In-place Metal GLPv3 fold/writer edits against FP64, with independently observable planted faults.
const std = @import("std");
const mtl = @import("metal");
const tf = @import("tensorfold");
const sources = @import("kernel_sources");
const contract = @import("contract");
const rig_ = @import("rig.zig");
const ref = @import("dsv41_glp_ref.zig");

const c = contract.spec;
const g = contract.gates;
const glp = tf.dsv41_metal.glp;
const base = "dsv41 glp ";
const dim: usize = 5120;
const rows_17: usize = 17;

/// The four-copy stream after a block's FFN fold, or the attention writer's single copy.
pub const Kind = enum { fold, writer };

pub const Variant = struct { kind: Kind, project: bool, ctl: u8 = 0 };

/// Project: ceil(dim / 256) + 20 roundings (see dsv41_glp_ref.zig); an add rounds its product, its sum and its output.
fn projectDepth(s: c.Shape) usize {
    return (s.k + 255) / 256 + 20;
}

fn addDepth(_: c.Shape) usize {
    return 3;
}

/// The kernel takes any width: the served widths prove 5120 (the model's), the odd ones its tails (no engine call).
const served = [_]c.Shape{
    .{ .rows = 16, .n = 1, .k = dim },
    .{ .rows = 1, .n = 1, .k = dim },
    .{ .rows = 5, .n = 1, .k = dim },
    .{ .rows = 64, .n = 1, .k = dim },
    .{ .rows = 3, .n = 1, .k = 5000 },
    .{ .rows = 7, .n = 1, .k = 37 },
};

pub fn spec(v: Variant, name: []const u8) c.Spec {
    return .{
        .name = name,
        .accum = if (v.project) .f32 else .none,
        .depth = if (v.project) projectDepth else addDepth,
        .out = .bf16,
        .tiles = .{ .{ .size = 16 }, .{}, .{ .size = 256, .step = 32, .min = 32 } },
        .served = &served,
        // the stress rows' fp16 and bf16 totals miss by far more than the default 100x or the gate fails
        .sums = .{ .why = "each thread's chain, the butterfly and the eight partials in fp32; the output is one bf16" },
    };
}

fn gateOf(ctl: u8) ?g.Gate {
    return switch (ctl) {
        0 => null,
        1 => .nonfinite,
        3 => .sums,
        6 => .repeat,
        else => .shapes,
    };
}

const vt = g.hooks(Proj);

pub const Proj = struct {
    a: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,
    rig: rig_.Rig,
    v: Variant,
    name: []const u8,
    pipes: glp.Pipelines,
    case: ?ref.Case = null,
    /// The next prepare makes the rank-one probe's inputs (a one-hot direction, strength 1).
    one_hot: bool = false,
    xin: rig_.Ref = undefined,
    din: rig_.Ref = undefined,
    out: rig_.Ref = undefined,

    pub fn init(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, v: Variant, name: []const u8) !Proj {
        var pipes = if (v.ctl == 0) try glp.Pipelines.load(device) else blk: {
            const text = try std.fmt.allocPrint(a, "#define TF_CTL {d}\n{s}", .{ v.ctl, sources.dsv41.glp });
            defer a.free(text);
            break :blk try glp.Pipelines.loadSource(device, text);
        };
        errdefer pipes.deinit();
        return .{ .a = a, .arena = std.heap.ArenaAllocator.init(a), .rig = .{ .a = a, .device = device, .queue = queue }, .v = v, .name = name, .pipes = pipes };
    }

    pub fn deinit(p: *Proj) void {
        p.pipes.deinit();
        p.rig.deinit();
        p.arena.deinit();
    }

    pub fn kernel(p: *Proj) g.Kernel {
        return .{ .spec = spec(p.v, p.name), .ctx = p, .vt = &vt, .control = gateOf(p.v.ctl) };
    }

    fn copies(p: *const Proj) usize {
        return if (p.v.kind == .fold) 4 else 1;
    }

    fn begin(p: *Proj) void {
        p.rig.reset();
        _ = p.arena.reset(.retain_capacity);
    }

    fn bind(p: *Proj, cs: ref.Case) !void {
        p.case = cs;
        p.xin = try p.rig.input(std.mem.sliceAsBytes(cs.x));
        p.din = try p.rig.input(std.mem.sliceAsBytes(cs.d));
        p.out = try p.rig.output(cs.x.len * 2);
    }

    pub fn prepare(p: *Proj, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.k == 0) return false;
        p.begin();
        var cs = try ref.Case.init(p.arena.allocator(), p.copies(), p.v.project, shape.rows, shape.k);
        cs.fill(fill, seed, p.one_hot);
        try p.bind(cs);
        return true;
    }

    /// The other kind's inputs: a fold's four copies each a writer's copy, or a writer's copy a fold's copy 0.
    fn prepareFrom(p: *Proj, src: ref.Case) !void {
        p.begin();
        var cs = try ref.Case.init(p.arena.allocator(), p.copies(), p.v.project, src.rows, src.dim);
        @memcpy(cs.d, src.d);
        cs.alpha = src.alpha;
        for (0..cs.rows) |r| for (0..cs.copies) |cc| {
            const from = if (src.copies == cs.copies) cc else 0;
            @memcpy(cs.x[(r * cs.copies + cc) * cs.dim ..][0..cs.dim], src.x[(r * src.copies + from) * cs.dim ..][0..cs.dim]);
        };
        try p.bind(cs);
    }

    /// The kernel edits in place: every launch starts from the input's bytes.
    fn reload(p: *Proj) void {
        const n = p.case.?.x.len * 2;
        @memcpy(p.rig.bytes(p.out, n), p.rig.bytes(p.xin, n));
    }

    /// Rows r0 .. r0 + n of the prepared call as one launch.
    fn encode(p: *Proj, e: mtl.ComputeEncoder, r0: usize, n: usize) void {
        const cs = p.case.?;
        e.setPipeline(if (p.v.kind == .fold) p.pipes.fold else p.pipes.writer);
        e.setBuffer(p.out.buf, p.out.off + r0 * cs.copies * cs.dim * 2, 0);
        e.setBuffer(p.din.buf, p.din.off, 1);
        e.setValue(glp.Args{ .dim = @intCast(cs.dim), .rows = @intCast(n), .alpha = cs.alpha, .project = @intFromBool(cs.project) }, 2);
        e.dispatchGroups(mtl.Size.of(n, 1, 1), mtl.Size.of(glp.threads, 1, 1));
    }

    pub fn run(p: *Proj) !void {
        p.reload();
        const pass = p.rig.begin();
        p.encode(pass.e, 0, p.case.?.rows);
        try p.rig.finish(pass);
    }

    pub fn count(p: *Proj) usize {
        return p.case.?.count();
    }

    pub fn reference(p: *Proj, want: []f64, mag: []f64) !void {
        try p.case.?.reference(p.arena.allocator(), want, mag, .none, .sequential);
    }

    /// Add uses three FP32 roundings plus BF16 output; project retains exact stress and depth-based allowances.
    pub fn bounds(p: *Proj, want: []const f64, mag: []const f64, allow: []f64) !void {
        if (p.case.?.project) return;
        for (allow, want, mag) |*l, w, m| {
            const work = c.gamma(3, 0x1p-24) * m;
            l.* = work + 0x1p-8 * (@abs(w) + work) + 0x1p-133;
        }
    }

    /// The outputs if the dot's sum ran in `acc` in `order`; everything else as the kernel does it.
    pub fn simulate(p: *Proj, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try p.arena.allocator().alloc(f64, out.len);
        try p.case.?.reference(p.arena.allocator(), out, mag, acc, order);
    }
};

fn fail(comptime fmt: []const u8, args: anytype) usize {
    std.debug.print("FAIL " ++ fmt ++ "\n", args);
    return 1;
}

/// Every registration and its planted controls, plus the probes that go beyond the generic gates.
pub const Suite = struct {
    a: std.mem.Allocator,
    probes: std.ArrayList(*Proj) = .empty,

    const E = struct { v: Variant, name: []const u8 };

    fn ent(comptime kind: Kind, comptime project: bool, comptime ctl: u8, comptime text: []const u8) E {
        const k = @tagName(kind);
        const m = if (project) "project" else "add";
        const tail = if (ctl == 0) "tf_dsv41_glp_" ++ k else "control: " ++ text;
        return .{ .v = .{ .kind = kind, .project = project, .ctl = ctl }, .name = base ++ k ++ " " ++ m ++ " " ++ tail };
    }

    const list = [_]E{
        ent(.fold, false, 0, ""),
        ent(.fold, true, 0, ""),
        ent(.writer, false, 0, ""),
        ent(.writer, true, 0, ""),
        ent(.fold, false, 1, "non-finite inputs read as zero"),
        ent(.fold, true, 1, "non-finite inputs read as zero"),
        ent(.writer, false, 1, "non-finite inputs read as zero"),
        ent(.writer, true, 1, "non-finite inputs read as zero"),
        ent(.fold, true, 2, "the copies' sum for their mean"),
        ent(.fold, true, 3, "the dot's totals in fp16"),
        ent(.writer, true, 3, "the dot's totals in fp16"),
        ent(.fold, true, 4, "project run as add"),
        ent(.writer, true, 4, "project run as add"),
        ent(.fold, false, 5, "only copy 0 edited"),
        ent(.fold, true, 5, "only copy 0 edited"),
        ent(.fold, false, 6, "writes one row past its end"),
        ent(.writer, true, 6, "writes one row past its end"),
    };

    pub fn create(a: std.mem.Allocator, device: mtl.Device, queue: mtl.Queue, kernels: *std.ArrayList(g.Kernel), only: ?[]const u8) !*Suite {
        const s = try a.create(Suite);
        s.* = .{ .a = a };
        errdefer s.destroy();
        for (list) |entry| {
            if (only) |text| if (std.mem.indexOf(u8, entry.name, text) == null) continue;
            const p = try a.create(Proj);
            errdefer a.destroy(p);
            p.* = try Proj.init(a, device, queue, entry.v, entry.name);
            errdefer p.deinit();
            try kernels.append(a, p.kernel());
            try s.probes.append(a, p);
        }
        return s;
    }

    pub fn destroy(s: *Suite) void {
        for (s.probes.items) |p| {
            p.deinit();
            s.a.destroy(p);
        }
        s.probes.deinit(s.a);
        s.a.destroy(s);
    }

    fn find(s: *Suite, kind: Kind, project: bool, ctl: u8) ?*Proj {
        for (s.probes.items) |p| {
            if (p.v.kind == kind and p.v.project == project and p.v.ctl == ctl) return p;
        }
        return null;
    }

    /// Row separation, copy symmetry, the rank-one law, and the kernels against each other and their controls.
    pub fn invariance(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        var launches: usize = 0;
        var kernels: usize = 0;
        for (s.probes.items) |p| {
            if (p.v.ctl != 0) continue;
            kernels += 1;
            bad += try s.split(p, seed, &launches);
            bad += try s.poison(p, seed);
            if (p.v.kind == .fold) bad += try sameCorrection(p, seed);
            bad += try alphaZero(p, seed);
            if (p.v.project) bad += try s.rankOne(p, seed);
        }
        if (kernels == 0) return 0;
        bad += try s.foldIsWriter(seed);
        bad += try s.maxMean(seed);
        bad += try s.controlsDiffer(seed);
        std.debug.print("{s} invariance {s}: {d} kernels (rows apart, split launches, same correction on every copy, rank one, fold = writer on equal copies, planted controls differ), {d} launches, {d} failures\n", .{ if (bad == 0) "PASS" else "FAIL", base, kernels, launches, bad });
        return bad;
    }

    /// Launches of 1, 5 and 7 rows equal one launch of 17: a row alone is a row among others, every byte.
    fn split(s: *Suite, p: *Proj, seed: u64, launches: *usize) !usize {
        var bad: usize = 0;
        if (!try p.prepare(.{ .rows = rows_17, .n = 1, .k = dim }, .random, seed)) return error.Refused;
        try p.run();
        const whole = try s.a.dupe(u8, try p.rig.outputBytes());
        defer s.a.free(whole);
        for ([_]usize{ 1, 5, 7 }) |step| {
            p.reload();
            const pass = p.rig.begin();
            var r0: usize = 0;
            while (r0 < rows_17) : (r0 += step) {
                p.encode(pass.e, r0, @min(step, rows_17 - r0));
                launches.* += 1;
            }
            try p.rig.finish(pass);
            if (!std.mem.eql(u8, whole, try p.rig.outputBytes())) bad += fail("row invariance {s}: {d} rows a launch differ from one launch", .{ p.name, step });
        }
        return bad;
    }

    /// A row of NaN in a 17-row launch leaves every other row's bytes as they were, and is NaN throughout itself.
    fn poison(s: *Suite, p: *Proj, seed: u64) !usize {
        const at: usize = 9;
        if (!try p.prepare(.{ .rows = rows_17, .n = 1, .k = dim }, .random, seed)) return error.Refused;
        try p.run();
        const whole = try s.a.dupe(u8, try p.rig.outputBytes());
        defer s.a.free(whole);
        const rb = p.copies() * dim * 2;
        p.reload();
        const bytes = p.rig.bytes(p.out, rows_17 * rb);
        var i: usize = 0;
        while (i < rb) : (i += 2) std.mem.writeInt(u16, bytes[at * rb + i ..][0..2], 0x7fc0, .little);
        const pass = p.rig.begin();
        p.encode(pass.e, 0, rows_17);
        try p.rig.finish(pass);
        const got = try p.rig.outputBytes();
        var bad: usize = 0;
        if (!std.mem.eql(u8, got[0 .. at * rb], whole[0 .. at * rb]) or !std.mem.eql(u8, got[(at + 1) * rb ..], whole[(at + 1) * rb ..]))
            bad += fail("row separation {s}: a poisoned row changed another row's bytes", .{p.name});
        for (0..rb / 2) |q| if (!std.math.isNan(c.read(.bf16, got, at * rb / 2 + q))) {
            bad += fail("row separation {s}: the poisoned row has a finite output at {d}", .{ p.name, q });
            break;
        };
        return bad;
    }

    /// On integer inputs every copy of a column moves by the same amount: the one correction reaches all four.
    fn sameCorrection(p: *Proj, seed: u64) !usize {
        if (!try p.prepare(.{ .rows = 4, .n = 1, .k = dim }, .stress, seed)) return error.Refused;
        try p.run();
        const cs = p.case.?;
        const out = p.rig.bytes(p.out, cs.count() * 2);
        var apart: usize = 0;
        var moved = false;
        for (0..cs.rows) |r| for (0..cs.dim) |i| {
            const first = (r * cs.copies) * cs.dim + i;
            const d0 = c.read(.bf16, out, first) - c.fromBf16(cs.x[first]);
            moved = moved or d0 != 0;
            for (1..cs.copies) |cc| {
                const at = first + cc * cs.dim;
                if (c.read(.bf16, out, at) - c.fromBf16(cs.x[at]) != d0) apart += 1;
            }
        };
        if (apart > 0) return fail("same correction {s}: {d} elements moved a copy by a different amount", .{ p.name, apart });
        if (!moved) return fail("same correction {s}: nothing moved", .{p.name});
        return 0;
    }

    /// One-hot project removes its mean component while preserving every unselected element byte for byte.
    fn rankOne(s: *Suite, p: *Proj, seed: u64) !usize {
        p.one_hot = true;
        defer p.one_hot = false;
        if (!try p.prepare(.{ .rows = 8, .n = 1, .k = dim }, .random, seed)) return error.Refused;
        try p.run();
        const cs = p.case.?;
        const out = p.rig.bytes(p.out, cs.count() * 2);
        const was = p.rig.bytes(p.xin, cs.count() * 2);
        var bad: usize = 0;
        var moved: usize = 0;
        for (0..cs.count()) |n| {
            if (cs.d[n % cs.dim] == 0 and !std.mem.eql(u8, out[2 * n ..][0..2], was[2 * n ..][0..2])) moved += 1;
        }
        if (moved > 0) bad += fail("rank one {s}: {d} elements off the direction changed", .{ p.name, moved });
        const rank = try ref.rankOne(cs, s.a, out, spec(p.v, p.name));
        if (!(rank.worst <= 1)) bad += fail("rank one {s}: the output mean keeps {d:.2} of its bound along d", .{ p.name, rank.worst });
        if (!(rank.least >= 8)) bad += fail("rank one {s}: the input's component is only {d:.1}x the bound: no proof", .{ p.name, rank.least });
        return bad;
    }

    /// Strength 0 edits nothing: every byte of a launch, NaN payloads and infinities included, leaves as it came.
    fn alphaZero(p: *Proj, seed: u64) !usize {
        if (!try p.prepare(.{ .rows = 8, .n = 1, .k = dim }, .nonfinite, seed)) return error.Refused;
        p.case.?.alpha = 0;
        const n = p.case.?.x.len * 2;
        const was = p.rig.bytes(p.xin, n);
        for ([_]u16{ 0x7fc1, 0xffc3, 0x7f81, 0x7f80, 0xff80, 0x0001, 0x8000 }, 0..) |v, i| {
            std.mem.writeInt(u16, was[(i * 97 + 5) * 2 ..][0..2], v, .little);
        }
        try p.run();
        if (!std.mem.eql(u8, p.rig.bytes(p.out, n), was)) return fail("alpha 0 {s}: the launch changed a byte", .{p.name});
        return 0;
    }

    /// BF16-max copies must keep a finite mean; the sum-first control must overflow on the same witness.
    fn maxMean(s: *Suite, seed: u64) !usize {
        const prod = s.find(.fold, true, 0) orelse return 0;
        const hi: u16 = 0x7f7f;
        const lo: u16 = 0xff7f;
        const m = c.fromBf16(hi);
        const k0 = dim / 3;
        var bad: usize = 0;
        for ([_]u8{ 0, 2 }) |ctl| {
            const q = if (ctl == 0) prod else s.find(.fold, true, 2) orelse continue;
            q.begin();
            var cs = try ref.Case.init(q.arena.allocator(), 4, true, 3, dim);
            cs.fill(.random, seed, false);
            @memset(cs.d, 0);
            cs.d[k0] = 1;
            for (0..cs.rows) |r| for (0..4) |cc| for (0..dim) |i| {
                const at = (r * 4 + cc) * dim + i;
                if (r == 0) cs.x[at] = hi else if (r == 1) cs.x[at] = lo else if (i == k0) cs.x[at] = hi;
            };
            try q.bind(cs);
            try q.run();
            const n = cs.x.len * 2;
            const out = q.rig.bytes(q.out, n);
            const was = q.rig.bytes(q.xin, n);
            var finite = true;
            for (0..cs.x.len) |e| finite = finite and std.math.isFinite(c.read(.bf16, out, e));
            if (ctl == 2) {
                if (finite) bad += fail("{s}: the max-copy mean stayed finite, so the witness proves nothing", .{q.name});
                continue;
            }
            if (!finite) bad += fail("max-copy mean {s}: a finite input gave a non-finite output", .{q.name});
            for (0..cs.rows) |r| for (0..4) |cc| for (0..dim) |i| {
                const at = (r * 4 + cc) * dim + i;
                if (i != k0) {
                    if (r < 2 and !std.mem.eql(u8, out[2 * at ..][0..2], was[2 * at ..][0..2])) {
                        bad += fail("max-copy mean {s}: a column with a zero direction changed at row {d}", .{ q.name, r });
                        break;
                    }
                    continue;
                }
                const want = (if (r == 1) -m else m) * (1 - @as(f64, cs.alpha));
                const got = c.read(.bf16, out, at);
                if (!(@abs(got - want) <= 0x1p-6 * @abs(want))) {
                    bad += fail("max-copy mean {s}: row {d} copy {d} is {e}, not {e}", .{ q.name, r, cc, got, want });
                    break;
                }
            };
        }
        return bad;
    }

    /// Equal copies reproduce the writer's edit; differing copies must use their mean, not copy zero.
    fn foldIsWriter(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        const shape = c.Shape{ .rows = 9, .n = 1, .k = dim };
        const row = dim * 2;
        for ([_]bool{ false, true }) |project| {
            const f = s.find(.fold, project, 0) orelse continue;
            const w = s.find(.writer, project, 0) orelse continue;
            if (!try w.prepare(shape, .random, seed)) return error.Refused;
            try w.run();
            try f.prepareFrom(w.case.?);
            try f.run();
            var apart: usize = 0;
            for (0..shape.rows) |r| for (0..4) |cc| {
                const got = (try f.rig.outputBytes())[(r * 4 + cc) * row ..][0..row];
                if (!std.mem.eql(u8, got, (try w.rig.outputBytes())[r * row ..][0..row])) apart += 1;
            };
            if (apart > 0) bad += fail("{s}: {d} of {d} copies of four equal copies differ from the writer's output", .{ f.name, apart, 4 * shape.rows });
            if (!project) continue;
            if (!try f.prepare(shape, .random, seed +% 1)) return error.Refused;
            try f.run();
            try w.prepareFrom(f.case.?);
            try w.run();
            var same: usize = 0;
            for (0..shape.rows) |r| {
                if (std.mem.eql(u8, (try f.rig.outputBytes())[r * 4 * row ..][0..row], (try w.rig.outputBytes())[r * row ..][0..row])) same += 1;
            }
            if (same > 0) bad += fail("{s}: {d} rows of distinct copies equal the writer's output on the first", .{ f.name, same });
        }
        return bad;
    }

    /// Each control's bytes differ from the kernel it spoils on the same inputs: mean for sum, add for project, copy 0.
    fn controlsDiffer(s: *Suite, seed: u64) !usize {
        var bad: usize = 0;
        for (s.probes.items) |cp| {
            if (cp.v.ctl != 2 and cp.v.ctl != 4 and cp.v.ctl != 5) continue;
            const pp = s.find(cp.v.kind, cp.v.project, 0) orelse continue;
            const shape = c.Shape{ .rows = 16, .n = 1, .k = dim };
            if (!try cp.prepare(shape, .random, seed) or !try pp.prepare(shape, .random, seed)) return error.Refused;
            try cp.run();
            try pp.run();
            if (std.mem.eql(u8, try cp.rig.outputBytes(), try pp.rig.outputBytes())) bad += fail("{s}: its bytes equal the kernel's", .{cp.name});
        }
        return bad;
    }
};
