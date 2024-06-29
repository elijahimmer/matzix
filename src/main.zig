pub fn main() !void {
    // what should the main function be?
    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    defer bw.flush() catch {};

    const stdout = bw.writer();

    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = false }){};
    defer _ = gpa.deinit();
    //var arena = std.heap.ArenaAllocator.init(gpa.allocator());
    //defer arena.deinit();
    const alloc = gpa.allocator();

    const t = Matrix(1_000, 1_000, f64);

    const large_matrix = try alloc.create(t);
    defer alloc.destroy(large_matrix);

    large_matrix.* = t.uniform(0);

    try stdout.print("{}", .{@sizeOf(t)});

    try bw.flush();
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const matzix = @import("root.zig");
const Matrix = matzix.Matrix;
