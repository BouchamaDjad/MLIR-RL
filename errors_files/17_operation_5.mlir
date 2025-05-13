
module attributes {torch.debug_module_name = "Net"} {
  func.func private @nanoTime() -> i64 attributes {llvm.emit_c_interface}
  func.func private @printFlops(f64)
  func.func private @printI64(i64)
  func.func private @printNewline()
  func.func private @printMemrefF32(tensor<*xf32>)
  func.func @matmul() -> tensor<7x228xf32> {
    %cst = arith.constant 2.000000e+00 : f32
    %0 = bufferization.alloc_tensor() : tensor<3072x2048xf32>
    %1 = linalg.fill {tag = "operation_0"} ins(%cst : f32) outs(%0 : tensor<3072x2048xf32>) -> tensor<3072x2048xf32>
    %2 = bufferization.alloc_tensor() : tensor<7x2048xf32>
    %3 = linalg.fill {tag = "operation_1"} ins(%cst : f32) outs(%2 : tensor<7x2048xf32>) -> tensor<7x2048xf32>
    %4 = bufferization.alloc_tensor() : tensor<3072x7xf32>
    %5 = linalg.fill {tag = "operation_2"} ins(%cst : f32) outs(%4 : tensor<3072x7xf32>) -> tensor<3072x7xf32>
    %6 = call @nanoTime() : () -> i64
    %7 = linalg.matmul_transpose_b {tag = "operation_3"} ins(%1, %3 : tensor<3072x2048xf32>, tensor<7x2048xf32>) outs(%5 : tensor<3072x7xf32>) -> tensor<3072x7xf32>
    %8 = bufferization.alloc_tensor() : tensor<3072x228xf32>
    %9 = bufferization.alloc_tensor() : tensor<7x228xf32>
    %10 = linalg.matmul_transpose_a {tag = "operation_4"} ins(%7, %8 : tensor<3072x7xf32>, tensor<3072x228xf32>) outs(%9 : tensor<7x228xf32>) -> tensor<7x228xf32>
    %11 = bufferization.alloc_tensor() : tensor<7x228xf32>
    %12 = bufferization.alloc_tensor() : tensor<7x228xf32>
    %13 = linalg.add {tag = "operation_5"} ins(%10, %11 : tensor<7x228xf32>, tensor<7x228xf32>) outs(%12 : tensor<7x228xf32>) -> tensor<7x228xf32>
    %14 = call @nanoTime() : () -> i64
    %15 = arith.subi %14, %6 : i64
    call @printI64(%15) : (i64) -> ()
    call @printNewline() : () -> ()
    return %13 : tensor<7x228xf32>
  }
  func.func @main() {
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    scf.for %arg0 = %c0 to %c2 step %c1 {
      %0 = func.call @matmul() : () -> tensor<7x228xf32>
    }
    return
  }
}
