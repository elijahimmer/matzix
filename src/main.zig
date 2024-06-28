pub fn main() !void {
    // what should the main function be?
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    defer bw.flush() catch {};

    const stdout = bw.writer();

    const t = matzix.Matrix(2, 2, isize);
    const a = t{ .rows = .{ .{ 1, 2 }, .{ 3, 4 } } };
    const b = t{ .rows = .{ .{ 2, -1 }, .{ -5, 3 } } };
    const res = a.mul(2, b);

    try stdout.print("mul: {}", .{res});

    try bw.flush();
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const matzix = @import("root.zig");
