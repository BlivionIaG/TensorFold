//! Gemma's fused GDN stays on Metal. Other heaps are not that kernel, and leftovers stay dense.
const std = @import("std");

pub const Heap = enum { gemma_metal, qwen, glm, deepseek, dense };
pub const Kind = enum { gemma_metal_gdn, qwen_gdn, qwen_qsa, glm_kda, deepseek_csa2, leftover };

pub const Error = error{ MixerRetargetRefused, MetalOnly, NotPorted, DenseUnported };

pub fn heap(kind: Kind) Heap {
    return switch (kind) {
        .gemma_metal_gdn => .gemma_metal,
        .qwen_gdn, .qwen_qsa => .qwen,
        .glm_kda => .glm,
        .deepseek_csa2 => .deepseek,
        .leftover => .dense,
    };
}

pub fn retarget(from: Kind, onto: Kind) Error!void {
    if (heap(from) != heap(onto)) return error.MixerRetargetRefused;
}

pub fn launch(kind: Kind) Error!void {
    switch (kind) {
        .gemma_metal_gdn => return error.MetalOnly,
        .leftover => return error.DenseUnported,
        .qwen_gdn, .qwen_qsa, .glm_kda, .deepseek_csa2 => return error.NotPorted,
    }
}

test "Gemma Metal GDN is not retargeted onto Qwen, GLM, or DeepSeek" {
    try std.testing.expect(heap(.gemma_metal_gdn) == .gemma_metal);
    try std.testing.expect(heap(.qwen_gdn) == heap(.qwen_qsa));
    try std.testing.expect(heap(.qwen_gdn) != heap(.glm_kda));
    try std.testing.expect(heap(.glm_kda) != heap(.deepseek_csa2));
    try std.testing.expect(heap(.leftover) == .dense);
    try std.testing.expectError(error.MixerRetargetRefused, retarget(.gemma_metal_gdn, .qwen_gdn));
    try std.testing.expectError(error.MixerRetargetRefused, retarget(.gemma_metal_gdn, .qwen_qsa));
    try std.testing.expectError(error.MixerRetargetRefused, retarget(.gemma_metal_gdn, .glm_kda));
    try std.testing.expectError(error.MixerRetargetRefused, retarget(.gemma_metal_gdn, .deepseek_csa2));
    try std.testing.expectError(error.MixerRetargetRefused, retarget(.leftover, .gemma_metal_gdn));
    try retarget(.qwen_gdn, .qwen_qsa);
    try std.testing.expectError(error.MetalOnly, launch(.gemma_metal_gdn));
    try std.testing.expectError(error.NotPorted, launch(.qwen_gdn));
    try std.testing.expectError(error.NotPorted, launch(.glm_kda));
    try std.testing.expectError(error.NotPorted, launch(.deepseek_csa2));
    try std.testing.expectError(error.DenseUnported, launch(.leftover));
}
