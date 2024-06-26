pub fn Matrix(comptime n: u32, comptime m: u32, comptime t: type) type {
    return struct {
        rows: [n]@Vector(m, t),

        pub fn uniform(scalar: t) @This() {
            return .{
                .rows = .{@as(@Vector(n, t), @splat(scalar))} ** m,
            };
        }

        pub fn add(lhs: *@This(), rhs: *const @This()) void {
            for (&lhs.rows, rhs.rows) |*l, r| {
                l.* += r;
            }
        }

        pub fn sub(lhs: *@This(), rhs: *const @This()) void {
            for (&lhs.rows, rhs.rows) |*l, r| {
                l.* -= r;
            }
        }

        pub fn add_scalar(lhs: *@This(), scalar: t) void {
            const scalar_mul = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| {
                l.* += scalar_mul;
            }
        }

        pub fn sub_scalar(lhs: *@This(), scalar: t) void {
            const scalar_vec = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| {
                l.* -= scalar_vec;
            }
        }

        pub fn mul_scalar(lhs: *@This(), scalar: t) void {
            const scalar_mul = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| {
                l.* *= scalar_mul;
            }
        }

        pub fn div_scalar(lhs: *@This(), scalar: t) void {
            const scalar_vec = @as(@Vector(n, t), @splat(scalar));

            for (&lhs.rows) |*l| {
                l.* /= scalar_vec;
            }
        }
    };
}

test "Matrix Uniform" {
    const a = Matrix(10, 10, f64).uniform(5);
    assert(meta.eql(a.rows, .{.{5} ** 10} ** 10));
}

test "Matrix Add" {
    const t = Matrix(10, 10, f64);

    var a = t.uniform(5);
    var b = t.uniform(9);

    const a_clone = a;
    a.add(&b);
    b.add(&a_clone);

    assert(meta.eql(a, b));
}

test "Matrix Sub" {
    const t = Matrix(10, 10, f64);

    var a = t.uniform(5);
    var b = t.uniform(9);
    const a_clone = a;

    a.sub(&b);
    assert(meta.eql(a, t.uniform(-4)));

    b.sub(&a_clone);
    assert(meta.eql(b, t.uniform(4)));
}

test "Matrix Scalar Add + Sub" {
    const t = Matrix(10, 10, f64);

    var a = t.uniform(5);

    a.add_scalar(5);
    assert(meta.eql(a, t.uniform(10)));

    a.sub_scalar(5);
    assert(meta.eql(a, t.uniform(5)));
}

test "Matrix Scalar Mul + Div" {
    const t = Matrix(10, 10, f64);

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

test {
    std.testing.refAllDecls(@This());
}

const std = @import("std");
const testing = std.testing;
const assert = std.debug.assert;
const meta = std.meta;
