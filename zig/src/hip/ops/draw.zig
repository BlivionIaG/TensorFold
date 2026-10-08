//! The draw inputs read from logits rows: argmax, the picked token's probability and the top ids.

const t = @import("types.zig");
const Ops = @import("ops.zig").Ops;
const Error = t.Error;
const Tensor = t.Tensor;
const p = t.p;
const f = t.f;
const i = t.i;
const int = t.int;

/// torch.argmax of each of `rows` logits rows (width `n`, fp16 or bf16) into device i32 `out`.
pub fn argmaxRows(o: Ops, logits: Tensor, rows: usize, n: usize, out: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.lib.call("tf_argmax_rows", .{ p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), i(out), o.stream });
}

/// softmax of each of `rows` logits rows at the token device i32 `ids` names (0: the row's largest), into f32 `out`.
pub fn tokenProb(o: Ops, logits: Tensor, rows: usize, n: usize, ids: u64, out: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.lib.call("tf_token_prob", .{ p(logits.ptr), @backingInt(logits.kind), int(rows), int(n), if (ids == 0) null else @ptrFromInt(ids), f(out), o.stream });
}

/// Each row's `ks[r]` largest by (value desc, id asc): ids (i32) and the values' 16-bit patterns, `stride` apart.
pub fn topkRows(o: Ops, logits: Tensor, rows: usize, n: usize, ks: u64, stride: usize, ids: u64, values: u64) Error!void {
    if (logits.kind == .f32) return error.BadShape;
    try o.lib.call("tf_topk_rows", .{ p(logits.ptr), int(rows), int(n), i(ks), int(stride), i(ids), p(values), o.stream });
}
