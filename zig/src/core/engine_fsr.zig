//! The flags of SSD residency (--ssd-experts, --ple-on-ssd, --fsr-route-trace) and the engines that take them.
const std = @import("std");

const budget_refusal = "--ssd-experts requires a positive finite GiB budget below 2^33";

fn budgetOk(g: f64) bool {
    return std.math.isFinite(g) and g > 0 and g < 0x1p33;
}

/// Why `backend` cannot take the flags in `o` (null: it can, or none is set); asked before a device opens.
pub fn refusal(o: anytype, backend: []const u8) ?[]const u8 {
    if (o.ssd_experts_gib == null and !o.ple_on_ssd and o.fsr_trace == null) return null;
    // the HIP expert offload (hip/expert_store.zig): --ssd-experts GIB sizes the device cache of a Qwen3.5 MoE
    if (std.mem.eql(u8, backend, "hip") and std.mem.eql(u8, o.model_type, "qwen3_5_moe") and !o.ple_on_ssd and o.fsr_trace == null) {
        if (o.ssd_experts_gib) |g| if (!budgetOk(g)) return budget_refusal;
        return null;
    }
    if (!std.mem.eql(u8, backend, "metal") or !std.mem.eql(u8, o.model_type, "qwen4_exp"))
        return "--ssd-experts, --ple-on-ssd and --fsr-route-trace require Flash Next (qwen4_exp) on Metal";
    if (o.speed_up != null) return "FSR SSD residency requires one Metal device; --speed-up is unsupported";
    if (o.ssd_experts_gib) |g| if (!budgetOk(g)) return budget_refusal;
    return null;
}
