/// A matrix
pub fn Matrix(comptime m: usize, comptime n: usize, comptime t: type) type {
    return struct {
        rows: [m][n]t,

        pub fn multiplicative_identity(alloc: Allocator) Allocator.Error!*@This() {
            var new = try @This().uniform(alloc, 0);

            for (0..m) |i| {
                for (0..n) |j| new.rows[i][j] = @as(t, @intFromBool(i == j));
            }

            return new;
        }

        fn clone(self: @This(), alloc: Allocator) Allocator.Error!*@This() {
            const result = try alloc.create(@This());

            result.* = self;

            return result;
        }

        //pub const I = multiplicative_identity();
        //pub const O = uniform(0);

        /// create a matrix with every value being `scalar`
        pub fn uniform(alloc: Allocator, scalar: t) Allocator.Error!*@This() {
            const res = try alloc.create(@This());

            @memset(@as(*[m * n]t, @ptrCast(&res.rows)), scalar);

            return res;
        }

        /// add two same dimension matrices together
        pub fn add(lhs: *@This(), rhs: *const @This()) void {
            for (&lhs.rows, rhs.rows) |*l_row, r_row| {
                for (l_row, r_row) |*l, r|
                    l.* += r;
            }
        }

        /// take the rhs from the lhs matrices
        pub fn sub(lhs: *@This(), rhs: *const @This()) void {
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
        pub fn transpose(self: *const @This(), alloc: Allocator) Allocator.Error!*Matrix(n, m, t) {
            var new = try alloc.create(Matrix(n, m, t));

            for (self.rows, 0..) |row, j| {
                for (row, 0..) |val, i|
                    new.rows[i][j] = val;
            }

            return new;
        }

        /// a simple matrix multiplication via transposing the rhs array.
        pub fn mul(alloc: Allocator, lhs: *@This(), r: comptime_int, rhs: *Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            var result = try alloc.create(Matrix(m, r, t));

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
        pub fn mul_transpose(alloc: Allocator, lhs: *const @This(), r: comptime_int, rhs: *const Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            const r_trans = try rhs.transpose(alloc);
            var result = try alloc.create(Matrix(m, r, t));

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

        pub fn mul_simd(alloc: Allocator, lhs: *const @This(), r: comptime_int, rhs: *const Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            var result = try alloc.create(Matrix(m, r, t));

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

        pub fn mul_simd_transpose(alloc: Allocator, lhs: *const @This(), r: comptime_int, rhs: *const Matrix(n, r, t)) Allocator.Error!*Matrix(m, r, t) {
            const rhs_trans = try rhs.transpose(alloc);
            var result = try alloc.create(Matrix(m, r, t));

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
    defer testing.allocator.destroy(a);
    try expect(meta.eql(a.rows, .{.{5} ** 10} ** 10));
}

test "Add" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = Matrix(10, 10, isize);

    const a = try t.uniform(alloc, 5);
    const b = try t.uniform(alloc, 9);
    const a_clone = try a.clone(alloc);

    a.add(b);
    b.add(a_clone);

    try expect(meta.eql(a.*, (try t.uniform(alloc, 14)).*));
    try expect(meta.eql(a.*, b.*));
}

test "Sub" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = Matrix(10, 10, isize);

    const a = try t.uniform(alloc, 5);
    const b = try t.uniform(alloc, 9);
    const a_clone = try a.clone(alloc);

    a.sub(b);

    try expect(meta.eql(a.*, (try t.uniform(alloc, -4)).*));

    b.sub(a_clone);
    try expect(meta.eql(b.*, (try t.uniform(alloc, 4)).*));
}

test "Scalar Add + Sub" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = Matrix(10, 10, isize);

    var a = try t.uniform(alloc, 5);

    a.add_scalar(5);
    try expect(meta.eql(a.*, (try t.uniform(alloc, 10)).*));

    a.sub_scalar(5);
    try expect(meta.eql(a.*, (try t.uniform(alloc, 5)).*));
}

test "Scalar Mul + Div" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = Matrix(10, 10, isize);

    var a = try t.uniform(alloc, 5);
    var b = try t.uniform(alloc, 5);

    a.mul_scalar(5);
    b.div_scalar(5);

    try expect(meta.eql(a.*, (try t.uniform(alloc, 25)).*));
    try expect(meta.eql(b.*, (try t.uniform(alloc, 1)).*));

    a.div_scalar(5);
    b.mul_scalar(5);

    try expect(meta.eql(a.*, (try t.uniform(alloc, 5)).*));
    try expect(meta.eql(a.*, b.*));
}

test "Transpose" {
    const start = Matrix(4, 3, isize);
    const result = Matrix(3, 4, isize);

    const a = start{ .rows = .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 }, .{ 0, 0, 0 } } };
    const b = try a.transpose(testing.allocator);
    defer testing.allocator.destroy(b);

    try expect(meta.eql(b.*, result{ .rows = .{ .{ 1, 0, 0, 0 }, .{ 0, 1, 0, 0 }, .{ 0, 0, 1, 0 } } }));
}

test "Multiplicative Identity" {
    const t = Matrix(3, 3, isize);
    const I = try t.multiplicative_identity(testing.allocator);
    defer testing.allocator.destroy(I);
    try expect(meta.eql(I.rows, .{ .{ 1, 0, 0 }, .{ 0, 1, 0 }, .{ 0, 0, 1 } }));
}

test "Large Matrix" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const alloc = arena.allocator();

    const t = Matrix(1_000, 1_000, i8); // that should be big enough

    const a = try t.uniform(alloc, 1);
    const b = try t.uniform(alloc, 5);

    a.add(b);

    const c = try t.uniform(alloc, 6);

    try expect(meta.eql(a.*, c.*));
}

fn test_multiply(m: comptime_int, n: comptime_int, r: comptime_int, fun: anytype, fun2: anytype) !void {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    const alloc = arena.allocator();
    defer arena.deinit();

    const a = try Matrix(m, m, i8).uniform(alloc, 5);
    const I = try @TypeOf(a.*).multiplicative_identity(alloc);

    const should_be_a = try fun(alloc, a, m, I);

    try expect(meta.eql(a, should_be_a));

    const b = try Matrix(m, n, i8).uniform(alloc, 1);
    const c = try Matrix(n, r, i8).uniform(alloc, 1);

    const b_mul = try fun2(alloc, b, r, c);
    const expected = try Matrix(m, r, i8).uniform(alloc, n);

    try expect(meta.eql(b_mul, expected));
}

//test "Multiply Basic" {
//    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul, Matrix(5, 5, i8).mul);
//}
//
//test "Multiply Transpose" {
//    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_transpose, Matrix(5, 5, i8).mul_transpose);
//}
//
//test "Multiply SIMD" {
//    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_simd, Matrix(5, 5, i8).mul_simd);
//}
//
//test "Multiply SIMD Transpose" {
//    try test_multiply(5, 5, 6, Matrix(5, 5, i8).mul_simd_transpose, Matrix(5, 5, i8).mul_simd_transpose);
//}

const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const expect = std.testing.expect;
const meta = std.meta;
const Allocator = std.mem.Allocator;
