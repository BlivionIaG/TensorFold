//! Host-side EXL3 index for Qwen3.8-Flash-Next.
//! CUDA, Metal, and the engine stay out of this file.
//! `exl3_format.zig` in this directory is a symlink to `zig/src/core/exl3_format.zig`.
//! `zig test` of a file here uses this directory as the module root, so a
//! `../` import of the core decoder is rejected.

pub const shape = @import("exl3_shape.zig");
pub const ngram = @import("exl3_ngram.zig");
pub const ple = @import("exl3_ple.zig");
pub const scan = @import("exl3_scan.zig");
pub const tp = @import("exl3_tp.zig");
pub const offload = @import("exl3_offload.zig");
pub const graph = @import("exl3_graph.zig");
pub const cache = @import("exl3_cache.zig");
pub const strips = @import("exl3_strips.zig");
pub const cpu = @import("exl3_cpu.zig");
pub const place = @import("exl3_place.zig");
pub const load = @import("exl3_load.zig");
pub const admit = @import("exl3_admit.zig");
pub const route = @import("exl3_route.zig");
pub const embed = @import("exl3_embed.zig");

test {
    _ = shape;
    _ = ngram;
    _ = ple;
    _ = scan;
    _ = tp;
    _ = offload;
    _ = graph;
    _ = cache;
    _ = strips;
    _ = cpu;
    _ = place;
    _ = load;
    _ = admit;
    _ = route;
    _ = embed;
}
