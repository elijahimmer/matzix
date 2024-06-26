pub const Matrix = @import("matrix.zig").Matrix;
pub const MatrixSIMD = @import("matrix_simd.zig").MatrixSIMD;

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}
