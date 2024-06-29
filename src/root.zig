pub const Matrix = @import("matrix.zig").Matrix;

test {
    @import("std").testing.refAllDeclsRecursive(@This());
}
