//! The model-free kernels' code objects, built by hipcc for -Dhip-arch and embedded, one a source group.
const objects = @import("hip_kernels");

/// The source groups of zig/kernels/hip, in the order the build embeds them.
pub const Group = enum {
    ops,
    act,
    attention,
    gated_delta,
    prefill,
    decode,
    plan,
    affine_tiles,
    affine_dot2,
    flashnext,
    mamba,
    glm_ocr,
    joint_schema,
    lighton_ocr,
    qwen_vision,
    draft,
    /// affine_tiles again, built with the expert-offload code; empty on a chip without caps `expert_ptrs`.
    affine_tiles_ptr,
};
pub const group_count = @typeInfo(Group).@"enum".field_names.len;

/// The architecture the objects were built for; the loader takes them only on that device.
pub const arch: []const u8 = objects.arch;

/// The objects take expert pointers (caps `expert_ptrs`): launches pass the table, `affine_tiles_ptr` reads it.
pub const expert_ptrs: bool = objects.expert_ptrs;

/// The objects in `Group` order; empty in a build without hipcc.
pub const images: [group_count][]align(8) const u8 = objects.images;
