/// A matrix
pub fn Matrix(comptime m: u32, comptime n: u32, comptime t: type) type {
    return struct {
        rows: [m][n]t,

        fn multiplicative_identity() @This() {
            @setEvalBranchQuota(m * n * 2);
            comptime assert(m == n); // it has to be a square matrix

            var new = uniform(0);

            for (0..m) |i| new.rows[i][i] = 1;

            return new;
        }

        pub const I = multiplicative_identity();
        pub const O = uniform(0);

        /// create a matrix with every value being `scalar`
        pub fn uniform(scalar: t) @This() {
            return .{
                .rows = .{.{scalar} ** n} ** m,
            };
        }

        /// add two same dimension matrices together
        pub fn add(lhs: *@This(), rhs: @This()) void {
            for (&lhs.rows, rhs.rows) |*l_row, r_row| {
                for (l_row, r_row) |*l, r|
                    l.* += r;
            }
        }

        /// take the rhs from the lhs matrices
        pub fn sub(lhs: *@This(), rhs: @This()) void {
            for (&lhs.rows, rhs.rows) |*l_row, r_row| {
                for (l_row, r_row) |*l, r|
                    l.* -= r;
            }
        }

        /// add a scalar to every element of the matrix
        pub fn add_scalar(lhs: *@This(), scalar: t) void {
            for (&lhs.rows) |*row| {
                for (row) |*val|
                    val.* += scalar;
            }
        }

        /// subtract a scalar from every element of the matrix
        pub fn sub_scalar(lhs: *@This(), scalar: t) void {
            for (&lhs.rows) |*row| {
                for (row) |*val|
                    val.* -= scalar;
            }
        }

        /// multiply every element of the matrix by a scalar
        pub fn mul_scalar(lhs: *@This(), scalar: t) void {
            for (&lhs.rows) |*row| {
                for (row) |*val|
                    val.* *= scalar;
            }
        }

        /// divide every element of the matrix by a scalar.
        pub fn div_scalar(lhs: *@This(), scalar: t) void {
            for (&lhs.rows) |*row| {
                for (row) |*val|
                    val.* = @divExact(val.*, scalar);
            }
        }

        /// turns every row of the matrix into the columns of the output matrix.
        /// takes a m by n matrix to a n by m matrix.
        pub fn transpose(self: @This()) Matrix(n, m, t) {
            var new: Matrix(n, m, t) = undefined;

            for (self.rows, 0..) |row, j| {
                for (row, 0..) |val, i|
                    new.rows[i][j] = val;
            }

            return new;
        }
    };
}

test "Uniform" {
    const a = Matrix(10, 10, isize).uniform(5);
    assert(meta.eql(a.rows, .{.{5} ** 10} ** 10));
}

test "Add" {
    const t = Matrix(10, 10, isize);

    var a = t.uniform(5);
    var b = t.uniform(9);

    const a_clone = a;
    a.add(b);
    b.add(a_clone);

    assert(meta.eql(a, b));
}

test "Sub" {
    const t = Matrix(10, 10, isize);

    var a = t.uniform(5);
    var b = t.uniform(9);
    const a_clone = a;

    a.sub(b);

    assert(meta.eql(a, t.uniform(-4)));

    b.sub(a_clone);
    assert(meta.eql(b, t.uniform(4)));
}

test "Scalar Add + Sub" {
    const t = Matrix(10, 10, isize);

    var a = t.uniform(5);

    a.add_scalar(5);
    assert(meta.eql(a, t.uniform(10)));

    a.sub_scalar(5);
    assert(meta.eql(a, t.uniform(5)));
}

test "Scalar Mul + Div" {
    const t = Matrix(10, 10, isize);

    var a = t.uniform(5);
    var b = t.uniform(5);

    a.mul_scalar(5);
    b.div_scalar(5);

    assert(meta.eql(a, t.uniform(25)));
    assert(meta.eql(b, t.uniform(1)));

    a.div_scalar(5);
    b.mul_scalar(5);

    assert(meta.eql(a, t.uniform(5)));
    assert(meta.eql(a, b));
}

test "Transpose" {
    const start = Matrix(4, 3, isize);
    const result = Matrix(3, 4, isize);

    const a = start{ .rows = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, 0 } } };
    const b = a.transpose();

    assert(meta.eql(b, result{ .rows = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } } }));
}

test "Multiplicative Identity" {
    const t = Matrix(3, 3, isize);
    assert(meta.eql(t.I.rows, .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } }));
}

test "Additive Identity" {
    const t = Matrix(3, 3, isize);
    assert(meta.eql(t.O.rows, .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } }));
}

test "Large Matrix" {
    const t = Matrix(1_000, 1_000, i8); // that should be big enough
    var a = t.uniform(1);
    const b = t.uniform(5);

    a.add(b);

    assert(meta.eql(a, t.uniform(6)));
}

const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const meta = std.meta;
