pub fn main() !void {
    // what should the main function be?
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();

    try stdout.print("Run `zig build test` to run the tests.\n", .{});

    try bw.flush();
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const matrix = @import("matrix.zig");
const matrix_simd = @import("matrix_simd.zig");
