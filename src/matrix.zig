pub fn Matrix(comptime n: u32, comptime m: u32) type {
    return struct {
        rows: [n]@Vector(m, f64),

        pub fn add(lhs: @This(), rhs: @This()) @This() {
            var rows = .{@splat(0)} ** m;

            for (0.., &rows) |idx, row| {
                row = lhs.rows[idx] + rhs.rows[idx];
            }

            return .{
                .rows = rows,
            };
        }
    };
}

test {
    std.testing.TestAllDecls(@This());
}

const std = @import("std");
