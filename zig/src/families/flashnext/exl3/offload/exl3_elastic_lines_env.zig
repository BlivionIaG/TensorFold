//! TF_FLASHNEXT_EXL3_ELASTIC_LINES: the opt-in switch for expert lines in a fixed virtual range.
const std = @import("std");

/// `1` puts every pool's lines in a fixed virtual range (unset or `0`: one allocation as before); else refused.
pub fn parse(value: ?[]const u8) !bool {
    const text = value orelse return false;
    if (std.mem.eql(u8, text, "0")) return false;
    if (std.mem.eql(u8, text, "1")) return true;
    return error.BadFlashNextElasticLines;
}

pub fn fromEnv() !bool {
    return parse(if (std.c.getenv("TF_FLASHNEXT_EXL3_ELASTIC_LINES")) |v| std.mem.span(v) else null);
}

test "exl3: elastic lines are off unless the variable is exactly 1" {
    try std.testing.expect(!try parse(null));
    try std.testing.expect(!try parse("0"));
    try std.testing.expect(try parse("1"));
    for ([_][]const u8{ "", "true", "2", "-1", "on" }) |text| try std.testing.expectError(error.BadFlashNextElasticLines, parse(text));
}
