/// A matrix
pub fn Matrix(comptime m: usize, comptime n: usize, comptime t: type) type {
    return struct {
        rows: [m][n]t,

        pub fn multiplicative_identity(alloc: Allocator) Allocator.Error!@This() {
            @setEvalBranchQuota(m * n * 2);

            var new = @This().uniform(alloc, 0);

            for (0..m) |i| {
                for (0..n) |j| new.rows[i][j] = @as(t, @intFromBool(i == j));
            }

            return new;
        }

        fn clone(self: @This(), alloc: Allocator) Allocator.Error!@This() {
            const result = alloc.create(@This());

            result.* = self;

            return result;
        }

        //pub const I = multiplicative_identity();
        //pub const O = uniform(0);

        /// create a matrix with every value being `scalar`
        pub fn uniform(alloc: Allocator, scalar: t) Allocator.Error!@This() {
            var res = alloc.create(@This());

            @memset(@as(*[m * n]t, @ptrCast(&res.rows)), scalar);

            return res;
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
        pub fn transpose(self: @This(), alloc: Allocator) Allocator.Error!Matrix(n, m, t) {
            var new = alloc.create(Matrix(n, m, t));

            for (self.rows, 0..) |row, j| {
                for (row, 0..) |val, i|
                    new.rows[i][j] = val;
            }

            return new;
        }

        /// a simple matrix multiplication via transposing the rhs array.
        pub fn mul(lhs: @This(), r: comptime_int, rhs: Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            var result = lhs.alloc.create(Matrix(m, r, t));

            for (lhs.rows, 0..) |left, idx| {
                for (0..r) |jdx| {
                    var sum: t = 0;

                    for (0..n) |offset|
                        sum += left[offset] * rhs.rows[offset][jdx];

                    result.rows[idx][jdx] = sum;
                }
            }

            return result;
        }

        /// a simple matrix multiplication via transposing the rhs array.
        pub fn mul_transpose(alloc: Allocator, lhs: @This(), r: comptime_int, rhs: Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            const r_trans = rhs.transpose();
            var result = alloc.create(Matrix(m, r, t));

            for (lhs.rows, 0..) |left, idx| {
                for (r_trans.rows, 0..) |right, jdx| {
                    var sum: t = 0;

                    for (0..n) |offset|
                        sum += left[offset] * right[offset];

                    result.rows[idx][jdx] = sum;
                }
            }

            return result;
        }

        pub fn mul_simd(alloc: Allocator, lhs: @This(), r: comptime_int, rhs: Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            var result = alloc.create(Matrix(m, r, t));

            for (0..r) |jdx| {
                var right: @Vector(n, t) = undefined;

                for (0..n) |idx| {
                    right[idx] = rhs.rows[idx][jdx];
                }

                for (lhs.rows, 0..m) |l_normal, idx| {
                    const left: @Vector(n, t) = l_normal;

                    result.rows[idx][jdx] = @reduce(.Add, left * right);
                }
            }

            return result;
        }

        pub fn mul_simd_transpose(alloc: Allocator, lhs: @This(), r: comptime_int, rhs: Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            const rhs_trans = rhs.transpose();
            var result = alloc.create(Matrix(m, r, t));

            for (lhs.rows, 0..) |l_normal, idx| {
                const left: @Vector(n, t) = l_normal;
                for (rhs_trans.rows, 0..) |r_normal, jdx| {
                    const right: @Vector(n, t) = r_normal;
                    result.rows[idx][jdx] = @reduce(.Add, left * right);
                }
            }

            return result;
        }
    };
}

test "Uniform" {
    const a = try Matrix(10, 10, isize).uniform(testing.allocator, 5);
    try expect(meta.eql(a.rows, .{.{5} ** 10} ** 10));
}

test "Add" {
    const t = Matrix(10, 10, isize);

    var a = try t.uniform(testing.allocator, 5);
    var b = try t.uniform(testing.allocator, 9);

    const a_clone = try a.clone(testing.allocator);
    a.add(b);
    b.add(a_clone);

    try expect(meta.eql(a, b));
}

test "Sub" {
    const t = Matrix(10, 10, isize);

    var a = try t.uniform(testing.allocator, 5);
    var b = try t.uniform(testing.allocator, 9);
    const a_clone = try a.clone(testing.allocator);

    a.sub(b);

    try expect(meta.eql(a, try t.uniform(testing.allocator, -4)));

    b.sub(a_clone);
    try expect(meta.eql(b, try t.uniform(testing.allocator, 4)));
}

test "Scalar Add + Sub" {
    const t = Matrix(10, 10, isize);

    var a = try t.uniform(testing.allocator, 5);

    a.add_scalar(5);
    try expect(meta.eql(a, try t.uniform(testing.allocator, 10)));

    a.sub_scalar(5);
    try expect(meta.eql(a, try t.uniform(testing.allocator, 5)));
}

test "Scalar Mul + Div" {
    const t = Matrix(10, 10, isize);

    var a = try t.uniform(testing.allocator, 5);
    var b = try t.uniform(testing.allocator, 5);

    a.mul_scalar(5);
    b.div_scalar(5);

    try expect(meta.eql(a, try t.uniform(testing.allocator, 25)));
    try expect(meta.eql(b, try t.uniform(testing.allocator, 5)));

    a.div_scalar(5);
    b.mul_scalar(5);

    try expect(meta.eql(a, try t.uniform(testing.allocator, 5)));
    try expect(meta.eql(a, b));
}

test "Transpose" {
    const start = Matrix(4, 3, isize);
    const result = Matrix(3, 4, isize);

    const a = start{ .rows = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, 0 } } };
    const b = try a.transpose(testing.allocator);
    defer testing.allocator.free(b);

    try expect(meta.eql(b, result{ .rows = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } } }));
}

test "Multiplicative Identity" {
    const t = Matrix(3, 3, isize);
    const I = try t.multiplicative_identity(testing.allocator);
    defer testing.allocator.free(I);
    try expect(meta.eql(I.rows, .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } }));
}

test "Large Matrix" {
    const t = Matrix(1_000, 1_000, i8); // that should be big enough
    var a = try t.uniform(testing.allocator, 1);
    defer testing.allocator.free(a);
    const b = try t.uniform(testing.allocator, 5);
    defer testing.allocator.free(b);

    a.add(b);

    const c = t.uniform(testing.allocator, 6);
    defer testing.allocator.free(c);

    try expect(meta.eql(a, c));
}

fn test_multiply(m: usize, n: usize, r: usize, fun: anytype, fun2: anytype) !void {
    const left = Matrix(m, n, i8);
    const right = Matrix(n, r, i8);
    const result = Matrix(m, r, i8);

    const a = Matrix(m, m, i8).uniform(5);

    const should_be_a = fun(a, m, @TypeOf(a).I);

    try expect(meta.eql(a, should_be_a));

    const b = left.uniform(1);
    const c = right.uniform(1);

    const b_mul = fun2(b, r, c);

    try expect(std.meta.eql(b_mul, result.uniform(n)));
}

test "Multiply Basic" {
    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul, Matrix(5, 6, i8).mul);
}

test "Multiply Transpose" {
    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_transpose, Matrix(5, 6, i8).mul_transpose);
}

test "Multiply SIMD" {
    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_simd, Matrix(5, 6, i8).mul_simd);
}

test "Multiply SIMD Transpose" {
    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_simd_transpose, Matrix(5, 6, i8).mul_simd_transpose);
}

const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const expect = std.testing.expect;
const meta = std.meta;
const Allocator = std.mem.Allocator;
