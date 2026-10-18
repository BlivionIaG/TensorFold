//! Host-side EXL3 index for Qwen3.8-Flash-Next.
//! CUDA, Metal, and the engine stay out of this file.
//! `exl3_format.zig` in this directory is a symlink to `zig/src/core/exl3_format.zig`.
//! `zig test` of a file here uses this directory as the module root, so a
//! `../` import of the core decoder is rejected.

pub const shape = @import("exl3_shape.zig");
pub const ngram = @import("exl3_ngram.zig");
pub const ple = @import("exl3_ple.zig");
pub const ple_ids = @import("exl3_ple_ids.zig");
pub const ple_cpu = @import("exl3_ple_cpu.zig");
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
pub const ple_launch = @import("exl3_ple_launch.zig");
pub const route = @import("exl3_route.zig");
pub const kv = @import("exl3_kv.zig");
pub const attn = @import("exl3_attn.zig");
pub const attn_cpu = @import("exl3_attn_cpu.zig");
pub const reduce = @import("exl3_reduce.zig");
pub const step = @import("exl3_step.zig");
pub const router = @import("exl3_router.zig");
pub const norm = @import("exl3_norm.zig");
pub const gdn = @import("exl3_gdn.zig");
pub const gdn_cpu = @import("exl3_gdn_cpu.zig");
pub const hc = @import("exl3_hc.zig");
pub const moe = @import("exl3_moe.zig");
pub const head = @import("exl3_head.zig");
pub const sample = @import("exl3_sample.zig");
pub const lane = @import("exl3_lane.zig");
pub const embed = @import("exl3_embed.zig");
pub const window = @import("exl3_window.zig");
pub const fwd_cpu = @import("exl3_fwd_cpu.zig");
pub const fwd_ple = @import("exl3_fwd_ple.zig");
pub const prefix = @import("exl3_prefix.zig");

test {
    _ = shape;
    _ = ngram;
    _ = ple;
    _ = ple_ids;
    _ = ple_cpu;
    _ = scan;
    _ = tp;
    _ = offload;
    _ = graph;
    _ = cache;
    _ = prefix;
    _ = strips;
    _ = cpu;
    _ = place;
    _ = load;
    _ = admit;
    _ = ple_launch;
    _ = route;
    _ = kv;
    _ = attn;
    _ = attn_cpu;
    _ = reduce;
    _ = step;
    _ = router;
    _ = norm;
    _ = gdn;
    _ = gdn_cpu;
    _ = hc;
    _ = moe;
    _ = head;
    _ = sample;
    _ = lane;
    _ = embed;
    _ = window;
    _ = fwd_cpu;
    _ = fwd_ple;
}
