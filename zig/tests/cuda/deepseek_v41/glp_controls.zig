//! Required nonproduction fault images; each control must fail its declared gate while production passes.

const std = @import("std");
const cuda = @import("cuda");
const check = @import("../check.zig");
const ref = @import("glp_ref.zig");
const rig = @import("glp_rig.zig");
const fold = @import("glp_fold.zig");

const glp = rig.glp;

pub const Control = enum { sum, half, drop_tail, drop_nonfinite, atomic, truncate };

/// The controls module and, per control, the stream and writer entry points (the production final beside them).
pub const Controls = struct {
    mod: cuda.Module,
    fns: [6]glp.Functions,

    pub fn load(c: rig.Ctx, dir: []const u8, prod: glp.Functions) !Controls {
        const a = c.gpu.gpa;
        const path = try std.fs.path.join(a, &.{ dir, "dsv41_glp_controls.fatbin" });
        defer a.free(path);
        const raw = std.Io.Dir.cwd().readFileAlloc(c.gpu.io, path, a, .limited(1 << 28)) catch |e| {
            std.debug.print("FAIL glp: the controls image {s} is required and cannot be read: {t}\n", .{ path, e });
            return e;
        };
        defer a.free(raw);
        const image = try a.alignedAlloc(u8, .@"16", raw.len);
        defer a.free(image);
        @memcpy(image, raw);
        var mod = try cuda.Module.load(c.gpu.d, image);
        errdefer mod.unload();
        var fns: [6]glp.Functions = undefined;
        inline for (std.enums.values(Control), 0..) |which, i| fns[i] = .{
            .stream = try mod.function(std.fmt.comptimePrint("dsv41_glp_stream_{s}", .{@tagName(which)})),
            .writer = try mod.function(std.fmt.comptimePrint("dsv41_glp_writer_{s}", .{@tagName(which)})),
            .final = prod.final,
        };
        return .{ .mod = mod, .fns = fns };
    }

    pub fn unload(s: *Controls) void {
        s.mod.unload();
    }

    pub fn ops(s: *const Controls, which: Control, c: rig.Ctx) glp.Ops {
        return .{ .f = &s.fns[@backingInt(which)], .s = c.stream };
    }
};

/// What a control must break: the FP64 bound, the FP64 non-finite classes, or the fixed fp32 order (the host model).
const Gate = enum { fp64_bound, nonfinite, order };

fn red(g: Gate, t: ref.Tally) bool {
    return switch (g) {
        .fp64_bound => t.over > 0,
        .nonfinite => t.class > 0,
        .order => t.model_diff > 0,
    };
}

const Plan = struct { control: Control, gate: Gate, stream: fold.FoldCase, writer: ?fold.WriterCase };

fn plans() [6]Plan {
    return .{
        // mean vs sum: x' = x - 4 s d, past the bound
        .{ .control = .sum, .gate = .fp64_bound, .stream = .{ .name = "mean vs sum", .width = 5120, .rows = 16, .mode = .project, .seed = 0x5001 }, .writer = null },
        // fp16 accumulation of the dot on integer terms: every order of fp32 is exact, fp16 is not
        .{ .control = .half, .gate = .fp64_bound, .stream = .{ .name = "fp16 dot", .width = 5120, .rows = 4, .mode = .project, .alpha = 1, .fill = .stress, .seed = 0x5002 }, .writer = .{ .name = "fp16 dot", .width = 5120, .rows = 4, .world = 2, .mode = .project, .alpha = 1, .fill = .stress, .seed = 0x5003 } },
        // the last column's term dropped from the dot: its copies are 64 and its direction 1
        .{ .control = .drop_tail, .gate = .fp64_bound, .stream = .{ .name = "dropped tail", .width = 1032, .rows = 8, .mode = .project, .witness = true, .seed = 0x5004 }, .writer = .{ .name = "dropped tail", .width = 1032, .rows = 8, .world = 3, .mode = .project, .seed = 0x5005 } },
        // a NaN or infinity product dropped from the dot: a finite row where FP64 says NaN
        .{ .control = .drop_nonfinite, .gate = .nonfinite, .stream = .{ .name = "dropped nonfinite", .width = 1032, .rows = 6, .mode = .project, .fill = .nonfinite, .seed = 0x5006 }, .writer = .{ .name = "dropped nonfinite", .width = 1032, .rows = 6, .world = 2, .mode = .project, .fill = .nonfinite, .seed = 0x5007 } },
        // Cancellation noise distinguishes an unordered shared-memory partial sum from the fixed FP32 order.
        .{ .control = .atomic, .gate = .order, .stream = .{ .name = "unordered add", .width = 5120, .rows = 64, .mode = .project, .alpha = 64, .fill = .cancel, .seed = 0x5008 }, .writer = .{ .name = "unordered add", .width = 5120, .rows = 64, .world = 2, .mode = .project, .alpha = 64, .fill = .cancel, .seed = 0x5009 } },
        // truncation instead of round to nearest even
        .{ .control = .truncate, .gate = .fp64_bound, .stream = .{ .name = "truncation", .width = 1032, .rows = 8, .mode = .add, .seed = 0x500a }, .writer = .{ .name = "truncation", .width = 1032, .rows = 8, .world = 2, .mode = .add, .seed = 0x500b } },
    };
}

fn controlLine(p: Plan, half: []const u8, n: usize, bad: ref.Tally, repeats: usize) void {
    check.pass("glp control {s} ({s}): production green on the same {d} elements; control red at the {t} gate: " ++
        "{d} off the model, {d} past the FP64 bound, {d} class errors; repeats differing {d} of 4", .{ @tagName(p.control), half, n, p.gate, bad.model_diff, bad.over, bad.class, repeats });
}

pub fn run(c: rig.Ctx, controls: *const Controls) !void {
    for (plans()) |p| {
        const ops = controls.ops(p.control, c);
        const prod = try fold.foldRun(c, c.ops, p.stream);
        if (!prod.ok()) return rig.fail("control {s}: the production fold is not green on its inputs ({d} off the model, {d} past the bound)", .{ @tagName(p.control), prod.t.model_diff, prod.t.over });
        const bad = try fold.foldRun(c, ops, p.stream);
        if (!red(p.gate, bad.t)) return rig.fail("control {s} stayed green at the {t} gate on the fold: {d} off the model, {d} past the bound, {d} class", .{ @tagName(p.control), p.gate, bad.t.model_diff, bad.t.over, bad.t.class });
        controlLine(p, "fold", prod.t.n, bad.t, bad.repeat_diff);
        const wc = p.writer orelse continue;
        const wprod = try fold.writerRun(c, c.ops, wc);
        if (!wprod.ok()) return rig.fail("control {s}: the production writer is not green on its inputs", .{@tagName(p.control)});
        const wbad = try fold.writerRun(c, ops, wc);
        // the writer's fault is seen at the fixed order whichever gate the fold's names: any deviation is red
        if (wbad.t.model_diff + wbad.t.over + wbad.t.class == 0) return rig.fail("control {s} stayed green on the writer", .{@tagName(p.control)});
        controlLine(p, "writer", wprod.t.n, wbad.t, wbad.repeat_diff);
    }
}
