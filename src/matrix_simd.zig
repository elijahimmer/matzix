/// A matrix using SIMD vectors
pub fn MatrixSIMD(comptime m: u32, comptime n: u32, comptime t: type) type {
    return struct {
        rows: [m]@Vector(n, t),

        fn multiplicative_identity() @This() {
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
                .rows = .{@as(@Vector(n, t), @splat(scalar))} ** m,
            };
        }

        /// add two same dimension matrices together
        pub fn add(lhs: *@This(), rhs: @This()) void {
            for (&lhs.rows, rhs.rows) |*l, r|
                l.* += r;
        }

        /// take the rhs from the lhs matrices
        pub fn sub(lhs: *@This(), rhs: @This()) void {
            for (&lhs.rows, rhs.rows) |*l, r| l.* -= r;
        }

        /// add a scalar to every element of the matrix
        pub fn add_scalar(lhs: *@This(), scalar: t) void {
            const scalar_mul = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| l.* += scalar_mul;
        }

        /// subtract a scalar from every element of the matrix
        pub fn sub_scalar(lhs: *@This(), scalar: t) void {
            const scalar_vec = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| l.* -= scalar_vec;
        }

        /// multiply every element of the matrix by a scalar
        pub fn mul_scalar(lhs: *@This(), scalar: t) void {
            const scalar_mul = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| l.* *= scalar_mul;
        }

        /// divide every element of the matrix by a scalar.
        pub fn div_scalar(lhs: *@This(), scalar: t) void {
            const scalar_vec = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| l.* /= scalar_vec;
        }

        /// turns every row of the matrix into the columns of the output matrix.
        /// takes a m by n matrix to a n by m matrix.
        pub fn transpose(self: *const @This()) MatrixSIMD(n, m, t) {
            var new: MatrixSIMD(n, m, t) = undefined;

            for (self.rows, 0..) |row_vec, j| {
                const row = @as([n]t, row_vec);
                for (row, 0..) |val, i| new.rows[i][j] = val;
            }

            return new;
        }
    };
}

test "Uniform" {
    const a = MatrixSIMD(10, 10, isize).uniform(5);
    assert(meta.eql(a.rows, .{.{5} ** 10} ** 10));
}

test "Add" {
    const t = MatrixSIMD(10, 10, isize);

    var a = t.uniform(5);
    var b = t.uniform(9);

    const a_clone = a;
    a.add(b);
    b.add(a_clone);

    assert(meta.eql(a, b));
}

test "Sub" {
    const t = MatrixSIMD(10, 10, isize);

    var a = t.uniform(5);
    var b = t.uniform(9);
    const a_clone = a;

    a.sub(b);
    assert(meta.eql(a, t.uniform(-4)));

    b.sub(a_clone);
    assert(meta.eql(b, t.uniform(4)));
}

test "Scalar Add + Sub" {
    const t = MatrixSIMD(10, 10, isize);

    var a = t.uniform(5);

    a.add_scalar(5);
    assert(meta.eql(a, t.uniform(10)));

    a.sub_scalar(5);
    assert(meta.eql(a, t.uniform(5)));
}

test "Scalar Mul + Div" {
    const t = MatrixSIMD(10, 10, isize);

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
    const start = MatrixSIMD(4, 3, isize);
    const result = MatrixSIMD(3, 4, isize);

    const a = start{ .rows = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, 0 } } };
    const b = a.transpose();

    assert(meta.eql(b, result{ .rows = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } } }));
}

test "Multiplicative Identity" {
    const t = MatrixSIMD(3, 3, isize);
    assert(meta.eql(t.I.rows, .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } }));
}

test "Additive Identity" {
    const t = MatrixSIMD(3, 3, isize);
    assert(meta.eql(t.O.rows, .{ .{ 0, 0, 0 }, .{ 0, 0, 0 }, .{ 0, 0, 0 } }));
}

test "Large Matrix" {
    const t = MatrixSIMD(1_000, 1_000, i8); // that should be big enough
    var a = t.uniform(1);
    const b = t.uniform(5);

    a.add(b);

    assert(meta.eql(a, t.uniform(6)));
}

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const meta = std.meta;
